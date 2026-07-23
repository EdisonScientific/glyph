from __future__ import annotations

import random
import re

from rdkit import Chem

from glyph.markush.data.canonicalize import (
    canonicalize_cxsmiles,
    canonicalize_cxsmiles_checked,
    clean_smiles_brackets,
    reindex_extensions,
    verify_molecule_unchanged,
)
from glyph.markush.data.pipeline import (
    DecorationConfig,
    decorate_scaffold,
    opt_to_standard_cxsmiles,
    split_cxsmiles,
    split_extension_sections,
    standard_to_opt_cxsmiles,
)
from glyph.markush.eval.markush_graph import canonical_markush_equal, markush_graph_equal


def assert_graph_equal(left: str, right: str) -> None:
    result = markush_graph_equal(left, right)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.error == ""
    assert result.equal is True


def assert_graph_not_equal(left: str, right: str) -> None:
    result = markush_graph_equal(left, right)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.equal is False


def _strip_atom_maps(smiles: str) -> str:
    return re.sub(r":\d+(?=\])", "", smiles)


def _renumber_standard_cxsmiles(cxsmiles: str, new_order: list[int]) -> str:
    """Return a serialization-equivalent standard CXSMILES with remapped extensions."""

    core, extension = split_cxsmiles(cxsmiles)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.strictCXSMILES = False
    mol = Chem.MolFromSmiles(core, params)
    assert mol is not None
    assert len(new_order) == mol.GetNumAtoms()

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    renumbered = Chem.RenumberAtoms(mol, new_order)
    smiles_with_maps = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=True)
    mapped_mol = Chem.MolFromSmiles(smiles_with_maps, params)
    assert mapped_mol is not None
    old_to_output = {
        atom.GetAtomMapNum() - 1: atom.GetIdx()
        for atom in mapped_mol.GetAtoms()
        if atom.GetAtomMapNum()
    }

    core_out = clean_smiles_brackets(_strip_atom_maps(smiles_with_maps))
    sections = split_extension_sections(extension)
    dollar_section = next(
        (section for section in sections if section.startswith("$") and section.endswith("$")),
        None,
    )
    other_sections = [section for section in sections if section != dollar_section]

    extension_parts: list[str] = []
    if dollar_section is not None:
        labels = dollar_section[1:-1].split(";")
        new_labels = ["" for _ in range(len(old_to_output))]
        for old_idx, label in enumerate(labels[: len(old_to_output)]):
            if label:
                new_labels[old_to_output[old_idx]] = label
        extension_parts.append("$" + ";".join(new_labels) + "$")

    reindexed = reindex_extensions(",".join(other_sections), old_to_output)
    if reindexed:
        extension_parts.append(reindexed)

    return f"{core_out} |{','.join(extension_parts)}|" if extension_parts else core_out


def test_canonicalization_metamorphic_preserves_known_true_variants():
    cases = [
        "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6",
        "CC(<r>R1</r>)C<r>R2</r>|m:1:0.1.3",
        "<r>R1</r>c1ccccc1",
    ]

    for cxsmiles in cases:
        canonical = canonicalize_cxsmiles(cxsmiles)
        assert canonical is not None
        assert_graph_equal(cxsmiles, canonical)
        assert canonical_markush_equal(cxsmiles, canonical).equal is True


def test_canonicalization_metamorphic_does_not_mask_wrong_variable_attachment():
    gt = "CC.S1C=NC2=C1C=CC=C2|m:1:2.3.4.5.6"
    pred = "C1=CC2=C(C=C1)N=CS2.CC|m:9:0.1.2.3.4.5"

    canonical_gt = canonicalize_cxsmiles(gt)
    canonical_pred = canonicalize_cxsmiles(pred)

    assert canonical_gt is not None
    assert canonical_pred is not None
    assert_graph_not_equal(canonical_gt, canonical_pred)
    assert canonical_markush_equal(canonical_gt, canonical_pred).equal is False


def test_canonicalization_rejects_rdkit_lossy_dummy_aromatic_rewrite():
    cxsmiles = (
        "<r>R1</r>C.<r>R2</r>C.<r>R3</r>C(C)(C)C(CC(=O)O)"
        "NC1=C(F)C=<r>X</r>C(C2=CNC3=<r>Y</r>C=<r>Z</r>C=C23)=N1"
        "|m:1:23.24.25.26.27.28,m:3:14.15.17.18.19.29"
    )
    lossy_rdkit_rewrite = (
        "<r>R1</r>C.<r>R2</r>C.<r>R3</r>C(C)(C)C(CC(=O)O)"
        "Nc1c(F)c<r>X</r>c(-c2c[nH]c3<r>Y</r>c<r>Z</r>cc23)n1"
        "|m:1:23.24.25.26.27.28,m:3:14.15.17.18.19.29"
    )

    is_same, message = verify_molecule_unchanged(cxsmiles, lossy_rdkit_rewrite)

    assert is_same is False
    assert "changed Markush graph" in message
    assert canonicalize_cxsmiles(cxsmiles) == cxsmiles
    result = canonicalize_cxsmiles_checked(cxsmiles)
    assert result.value == cxsmiles
    assert result.accepted is False
    assert result.fallback is True
    assert result.status == "fallback_unverified"


def test_canonicalization_checked_reports_verified_rewrite():
    result = canonicalize_cxsmiles_checked("C(C)O")

    assert result.value == "CCO"
    assert result.accepted is True
    assert result.changed is True
    assert result.fallback is False
    assert result.status == "canonicalized"


def test_canonicalization_converges_to_verified_fixed_point():
    cxsmiles = (
        "COC1=CC=CC=C1.C<r>V3</r>.C<r>V1</r>.C<r>V2</r>"
        "|m:8:2.3.4.5.6.7,m:10:2.3.4.5.6.7,m:12:2.3.4.5.6.7"
    )
    expected = (
        "C<r>V3</r>.C<r>V1</r>.C<r>V2</r>.COc1ccccc1"
        "|m:0:8.9.10.11.12.13,m:2:8.9.10.11.12.13,m:4:8.9.10.11.12.13"
    )

    result = canonicalize_cxsmiles_checked(cxsmiles)

    assert result.value == expected
    assert result.accepted is True
    assert result.changed is True
    assert result.fallback is False
    assert canonicalize_cxsmiles_checked(result.value).status == "already_canonical"


def test_canonicalization_rejects_lossy_second_pass():
    cxsmiles = "Cc1ccccc1C.Cc1ccccc1C.CBC|Sg:n:17:m:ht,m:16:9.10.11.12.13.14,m:18:1.2.3.4.5.6"

    result = canonicalize_cxsmiles_checked(cxsmiles)

    assert result.value == cxsmiles
    assert result.accepted is False
    assert result.fallback is True
    assert result.status == "fallback_unverified"
    assert "changed Markush graph" in (result.error or "")


def test_standard_atom_renumbering_preserves_label_m_and_sgroup_bindings():
    standard = "*c1cc(*)ccc1 |$R1;;;;R2;;$,Sg:n:2,3:n:ht,m:0:1.2.3|"
    orders = [
        [4, 3, 2, 1, 0, 5, 6, 7],
        list(reversed(range(8))),
    ]

    for order in orders:
        variant = _renumber_standard_cxsmiles(standard, order)
        opt_variant = standard_to_opt_cxsmiles(variant)

        assert_graph_equal(standard, variant)
        assert_graph_equal(standard, opt_variant)


def test_single_semantic_edits_flip_to_not_equal():
    cases = [
        (
            "*CC(*)O |$R1;;;R2;$|",
            "*CC(*)O |$R2;;;R1;$|",
        ),
        (
            "*c1ncccc1 |$R1;;;;;;$,m:0:1.2.3|",
            "*c1ncccc1 |$R1;;;;;;$,m:0:3.4.5|",
        ),
        (
            "*c1ncccc1 |$R1;;;;;;$,Sg:n:1,2:n:ht|",
            "*c1ncccc1 |$R1;;;;;;$,Sg:n:3,4:n:ht|",
        ),
        (
            "*CCCO |$R1;;;;$,Sg:n:1,2:n:ht|",
            "*CCCO |$R1;;;;$,Sg:n:1,2:m:ht|",
        ),
    ]

    for gt, edited in cases:
        assert_graph_not_equal(gt, edited)
        assert canonical_markush_equal(gt, edited).equal is False


def test_generated_converter_round_trips_are_structure_preserving():
    config = DecorationConfig(
        min_rgroups=1,
        max_rgroups=3,
        sg_probability=0.75,
        m_probability=0.75,
        zero_rgroup_probability=0.0,
        fragment_probability=0.0,
    )
    scaffolds = ["c1ccccc1CCO", "COc1ccccc1CCN", "c1ncccc1CCO"]
    samples = []
    for seed in range(12):
        sample = decorate_scaffold(
            scaffolds[seed % len(scaffolds)],
            config=config,
            rng=random.Random(seed),
        )
        if sample is not None:
            samples.append(sample)

    assert len(samples) >= 6
    assert any(sample.has_sg for sample in samples)
    assert any(sample.has_m for sample in samples)

    for sample in samples:
        standard = sample.cxsmiles
        opt = sample.cxsmiles_opt
        from_opt = opt_to_standard_cxsmiles(opt)
        from_standard = standard_to_opt_cxsmiles(standard)

        assert_graph_equal(standard, opt)
        assert_graph_equal(opt, from_opt)
        assert_graph_equal(standard, from_standard)
