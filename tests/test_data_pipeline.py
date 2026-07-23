from __future__ import annotations

import json
import random

from rdkit import Chem

from glyph.markush.data.pipeline import (
    DecorationConfig,
    ScaffoldFilterConfig,
    add_m_annotations,
    assemble_cxsmiles,
    build_scaffold_pool,
    build_training_record,
    decorate_scaffold,
    generate_decorated_samples,
    load_scaffold_pool,
    main,
    passes_scaffold_filter,
    prepare_training_jsonl_from_manifest,
    split_extension_sections,
    standard_to_opt_cxsmiles,
    validate_cxsmiles,
    write_seed_outputs,
)
from glyph.markush.eval.markush_graph import markush_graph_equal


def test_scaffold_filter_accepts_druglike_ring_and_rejects_tiny_molecule():
    cfg = ScaffoldFilterConfig(exclude_pains=False)
    assert passes_scaffold_filter(Chem.MolFromSmiles("c1ccccc1CCO"), cfg)
    assert not passes_scaffold_filter(Chem.MolFromSmiles("CC"), cfg)


def test_split_extension_sections_preserves_sg_index_commas():
    ext = "$R1;;;;R2;;$,Sg:n:3,4,5:n:ht,m:0:1.2.3"
    assert split_extension_sections(ext) == [
        "$R1;;;;R2;;$",
        "Sg:n:3,4,5:n:ht",
        "m:0:1.2.3",
    ]


def test_split_extension_sections_extracts_atomprop_after_coordinates():
    ext = "(0,0,;1,1,),atomProp:1.dummyLabel.E,Sg:n:0,2:n:ht"

    assert split_extension_sections(ext) == [
        "(0,0,;1,1,)",
        "atomProp:1.dummyLabel.E",
        "Sg:n:0,2:n:ht",
    ]


def test_split_extension_sections_drops_trailing_extension_comma():
    assert split_extension_sections("$R1$,Sg:n:0:m:ht:::,") == [
        "$R1$",
        "Sg:n:0:m:ht:::",
    ]


def test_split_extension_sections_drops_repeated_empty_sections():
    ext = "$R4;;;R5;;;R5;;$,,,Sg:n:6,7:x:ht:::,Sg:n:3,4:y:ht:::,"

    assert split_extension_sections(ext) == [
        "$R4;;;R5;;;R5;;$",
        "Sg:n:6,7:x:ht:::",
        "Sg:n:3,4:y:ht:::",
    ]


def test_split_extension_sections_splits_dollar_before_coordinates():
    assert split_extension_sections("$;E;$,(0,0,;1,1,)") == [
        "$;E;$",
        "(0,0,;1,1,)",
    ]


def test_decorate_scaffold_outputs_valid_standard_and_opt_cxsmiles():
    cfg = DecorationConfig(
        min_rgroups=2,
        max_rgroups=2,
        label_style="R_only",
        sg_probability=0.0,
        m_probability=0.0,
    )
    sample = decorate_scaffold("c1ccccc1CCO", config=cfg, rng=random.Random(7))

    assert sample is not None
    assert sample.validation.valid
    assert validate_cxsmiles(sample.cxsmiles).valid
    assert sample.cxsmiles.count("$") == 2
    assert sample.cxsmiles_opt.count("<r>") == 2
    assert "R1" in sample.cxsmiles_opt
    assert "R2" in sample.cxsmiles_opt


def test_m_annotation_from_rgroup_on_ring_validates():
    mol = Chem.MolFromSmiles("*c1ccccc1")
    cfg = DecorationConfig(sg_probability=0.0, m_probability=1.0)
    anns = add_m_annotations(mol, [0], cfg, random.Random(1))

    assert anns
    cxsmiles = assemble_cxsmiles(
        "*c1ccccc1",
        list(range(mol.GetNumAtoms())),
        {0: "R1"},
        [],
        anns,
    )
    report = validate_cxsmiles(cxsmiles)
    assert report.valid, report.errors
    assert "m:" in cxsmiles


def test_standard_to_opt_removes_dollar_section_and_keeps_sg_m_extensions():
    standard = "*c1cc(*)ccc1 |$R1;;;;R2;;$,Sg:n:2,3:n:ht,m:0:1.2.3|"
    opt = standard_to_opt_cxsmiles(standard)

    assert opt == "<r>R1</r>c1cc(<r>R2</r>)ccc1|Sg:n:2,3:n:ht,m:0:1.2.3"


def test_standard_to_opt_converts_atomprop_dummy_labels():
    standard = "C*OC |(0,0,;1,1,),atomProp:1.dummyLabel.E|"

    assert standard_to_opt_cxsmiles(standard) == "C<r>E</r>OC|(0,0,;1,1,)"


def test_standard_to_opt_preserves_non_label_atomprops():
    standard = "C*OC |atomProp:1.dummyLabel.E:1.molparserConnectionPoint.1|"

    assert standard_to_opt_cxsmiles(standard) == "C<r>E</r>OC|atomProp:1.molparserConnectionPoint.1"


def test_standard_to_opt_converts_numbered_wildcards_and_trailing_empty_labels():
    standard = "[1*]CC |$R1;;$|"

    assert standard_to_opt_cxsmiles(standard) == "<r>R1</r>CC"


def test_standard_to_opt_handles_m2s_empty_sections_and_sgroups():
    standard = "*[N+]([H])(*O[H])*O[H] |$R4;;;R5;;;R5;;$,,,Sg:n:6,7:x:ht:::,Sg:n:3,4:y:ht:::,|"

    opt = standard_to_opt_cxsmiles(standard)

    assert opt == (
        "<r>R4</r>[N+]([H])(<r>R5</r>O[H])<r>R5</r>O[H]|Sg:n:6,7:x:ht:::,Sg:n:3,4:y:ht:::"
    )
    assert "$" not in opt
    assert markush_graph_equal(standard, opt).equal is True


def test_standard_to_opt_keeps_multicenter_indices_when_atom_order_is_unchanged():
    standard = "*C.C1=CC2=C(C=C1)NN=N2 |$R;;;;;;;;;;$,m:0:6.5.4.7.2.3|"

    opt = standard_to_opt_cxsmiles(standard)

    assert opt == "<r>R</r>C.C1=CC2=C(C=C1)NN=N2|m:0:6.5.4.7.2.3"
    assert markush_graph_equal(standard, opt).equal is True
    assert (
        markush_graph_equal(
            standard,
            "<r>R</r>C.C1=CC2=C(C=C1)NN=N2|m:1:2.3.4.5.6.7",
        ).equal
        is False
    )


def test_standard_to_opt_preserves_m2s_nonwildcard_atom_label_as_atomprop():
    standard = (
        "*C.*C(C)(*C1=CC=CC=C1)C([H])([H])C.*O "
        "|$;R21;RA;;;YA;;;;;;;;;;;;$,,,,Sg:n:17:b:ht:::,Sg:n:1:c:ht:::,"
        "Sg:n:13,12,14,3,2,5,6,7,8,9,10,11: :ht:::,"
        "m:16:6.7.8.9.10.11,m:0:6.7.8.9.10.11|"
    )

    opt = standard_to_opt_cxsmiles(standard)

    assert opt == (
        "*C.<r>RA</r>C(C)(<r>YA</r>C1=CC=CC=C1)C([H])([H])C.*O"
        "|atomProp:1.dummyLabel.R21,Sg:n:17:b:ht:::,Sg:n:1:c:ht:::,"
        "Sg:n:13,12,14,3,2,5,6,7,8,9,10,11: :ht:::,"
        "m:16:6.7.8.9.10.11,m:0:6.7.8.9.10.11"
    )
    assert "$" not in opt
    assert markush_graph_equal(standard, opt).equal is True


def test_validate_accepts_optimized_rgroup_tags():
    report = validate_cxsmiles("<r>R1</r>c1ccccc1|m:0:1.2.3")

    assert report.valid, report.errors


def test_validate_cxsmiles_treats_bracket_ar_as_pseudo_wildcard():
    report = validate_cxsmiles("[Ar]Nc1ccccc1C(N*)=O |$;;;;;;;;;;Q;$|")

    assert report.valid, report.errors


def test_generate_decorated_samples_is_reproducible():
    scaffolds = ["c1ccccc1CCO", "COc1ccccc1CCN"]
    cfg = DecorationConfig(
        min_rgroups=1,
        max_rgroups=2,
        label_style="R_only",
        sg_probability=0.0,
        m_probability=0.0,
    )
    a, stats_a = generate_decorated_samples(scaffolds, num_samples=3, seed=11, config=cfg)
    b, stats_b = generate_decorated_samples(scaffolds, num_samples=3, seed=11, config=cfg)

    assert [sample.cxsmiles for sample in a] == [sample.cxsmiles for sample in b]
    assert stats_a["generated"] == 3
    assert stats_b["generated"] == 3


def test_build_scaffold_pool_from_small_cid_smiles_file(tmp_path):
    cid_smiles = tmp_path / "CID-SMILES"
    cid_smiles.write_text(
        "\n".join(
            [
                "1\tCC",
                "2\tc1ccccc1CCO",
                "3\tCOc1ccccc1CCN",
                "4\tinvalid",
            ]
        )
        + "\n"
    )
    out = tmp_path / "scaffold_pool.txt"
    stats = build_scaffold_pool(
        cid_smiles,
        out,
        target_count=10,
        sample_rate=1.0,
        seed=1,
        filter_config=ScaffoldFilterConfig(exclude_pains=False),
        progress_interval=0,
    )

    scaffolds = load_scaffold_pool(out)
    assert stats["written"] == 2
    assert len(scaffolds) == 2
    assert Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1CCO"), canonical=True) in scaffolds


def test_write_seed_outputs_and_training_record(tmp_path):
    cfg = DecorationConfig(
        min_rgroups=1,
        max_rgroups=1,
        label_style="R_only",
        sg_probability=0.0,
        m_probability=0.0,
    )
    sample = decorate_scaffold("c1ccccc1CCO", config=cfg, rng=random.Random(3))
    assert sample is not None

    jsonl_path = tmp_path / "seeds.jsonl"
    csv_path = tmp_path / "seeds.csv"
    write_seed_outputs([sample], output_jsonl=jsonl_path, output_csv=csv_path, id_prefix="t")

    record = json.loads(jsonl_path.read_text().strip())
    assert record["id"] == "t_000000"
    assert record["validation"]["valid"] is True
    assert "cxsmiles" in csv_path.read_text().splitlines()[0]

    train = build_training_record(
        "t_000000",
        "images/t_000000.png",
        record["cxsmiles_opt"],
        description="wherein R1 is alkyl",
    )
    assert train["images"] == ["images/t_000000.png"]
    assert "<cxsmi>" in train["messages"][1]["content"]


def test_decorate_cli_writes_json_csv_and_stats(tmp_path):
    pool = tmp_path / "pool.txt"
    pool.write_text("c1ccccc1CCO\nCOc1ccccc1CCN\n")
    output_jsonl = tmp_path / "seeds.jsonl"
    output_csv = tmp_path / "seeds.csv"
    stats_json = tmp_path / "stats.json"

    main(
        [
            "decorate",
            "--scaffold-pool",
            str(pool),
            "--num-samples",
            "2",
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--stats-json",
            str(stats_json),
            "--label-style",
            "R_only",
            "--sg-probability",
            "0",
            "--m-probability",
            "0",
        ]
    )

    assert len(output_jsonl.read_text().splitlines()) == 2
    assert output_csv.read_text().splitlines()[0].startswith("id,scaffold_smiles,cxsmiles")
    stats = json.loads(stats_json.read_text())
    assert stats["generated"] == 2


def test_prepare_training_jsonl_from_render_manifest(tmp_path):
    data_root = tmp_path / "data"
    pages_dir = data_root / "generated" / "pages"
    pages_dir.mkdir(parents=True)
    page = pages_dir / "sample_0001_page.png"
    page.write_bytes(b"not actually an image")

    manifest = data_root / "generated" / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "sample_0001",
                    "page_path": str(page),
                    "description": "wherein R1 is alkyl",
                    "annotation": "R1:alkyl",
                    "cxsmiles_opt": "<r>R1</r>c1ccccc1",
                }
            ]
        )
    )
    out = data_root / "generated" / "train.jsonl"

    stats = prepare_training_jsonl_from_manifest(
        manifest,
        out,
        data_root=data_root,
        prompt_text_dropout=0.0,
        seed=1,
    )

    assert stats == {"total": 1, "written": 1, "failed": 0, "dropped_prompt_text": 0}
    record = json.loads(out.read_text())
    assert record["images"] == ["generated/pages/sample_0001_page.png"]
    assert "Patent text: wherein R1 is alkyl" in record["messages"][0]["content"]
    assert "the <cxsmi> value must use cxsmiles_opt" in record["messages"][0]["content"]
    assert "<stable>R1:alkyl</stable>" in record["messages"][1]["content"]


def test_prepare_training_jsonl_uses_manifest_cxsmiles_opt_with_renderer_annotation(tmp_path):
    page = tmp_path / "pages" / "sample_0001_page.png"
    page.parent.mkdir()
    page.write_bytes(b"not actually an image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "sample_0001",
                    "page_path": str(page),
                    "description": "wherein R1 is alkyl",
                    "annotation": (
                        "<markush><cxsmi>WRONG_COORDINATE_CXSMILES</cxsmi>"
                        "<stable>R1:alkyl</stable></markush>"
                    ),
                    "cxsmiles_opt": "<r>R1</r>c1ccccc1",
                }
            ]
        )
    )
    out = tmp_path / "train.jsonl"

    stats = prepare_training_jsonl_from_manifest(manifest, out, data_root=None)

    assert stats["written"] == 1
    record = json.loads(out.read_text())
    assert "<cxsmi><r>R1</r>c1ccccc1</cxsmi>" in record["messages"][1]["content"]
    assert "WRONG_COORDINATE_CXSMILES" not in record["messages"][1]["content"]
    assert "<stable>R1:alkyl</stable>" in record["messages"][1]["content"]


def test_prepare_training_jsonl_does_not_nest_renderer_markush_without_stable(tmp_path):
    page = tmp_path / "pages" / "sample_0001_page.png"
    page.parent.mkdir()
    page.write_bytes(b"not actually an image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "sample_0001",
                    "page_path": str(page),
                    "description": "wherein R1 is alkyl",
                    "annotation": "<markush><cxsmi>WRONG_COORDINATE_CXSMILES</cxsmi></markush>",
                    "cxsmiles_opt": "<r>R1</r>c1ccccc1",
                }
            ]
        )
    )
    out = tmp_path / "train.jsonl"

    stats = prepare_training_jsonl_from_manifest(manifest, out, data_root=None)

    assert stats["written"] == 1
    record = json.loads(out.read_text())
    assert "<cxsmi><r>R1</r>c1ccccc1</cxsmi>" in record["messages"][1]["content"]
    assert "<stable></stable>" in record["messages"][1]["content"]
    assert "WRONG_COORDINATE_CXSMILES" not in record["messages"][1]["content"]


def test_prepare_training_jsonl_can_drop_prompt_text(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample_0001",
                "page_path": "pages/sample_0001_page.png",
                "description": "wherein R1 is alkyl",
                "cxsmiles_opt": "<r>R1</r>c1ccccc1",
            }
        )
        + "\n"
    )
    out = tmp_path / "train.jsonl"

    stats = prepare_training_jsonl_from_manifest(
        manifest,
        out,
        data_root=None,
        prompt_text_dropout=1.0,
        seed=1,
    )

    record = json.loads(out.read_text())
    assert stats["dropped_prompt_text"] == 1
    assert "Patent text:" not in record["messages"][0]["content"]
    assert "the <cxsmi> value must use cxsmiles_opt" in record["messages"][0]["content"]
    assert record["images"] == ["pages/sample_0001_page.png"]
