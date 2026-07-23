from __future__ import annotations

import json
import random
from pathlib import Path

from glyph.markush.data.example_modes import (
    PATENT_IMAGE_MARKUSH,
    PATENT_IMAGE_MARKUSH_PROMPT,
    STRUCTURE_ONLY_MARKUSH,
    STRUCTURE_ONLY_MARKUSH_PROMPT,
)
from glyph.markush.data.ocsr import OCSR_TARGET_FORMAT
from glyph.markush.data.prepare_sft import build_training_example, target_cxsmiles_for_record
from glyph.markush.data.prompt_policies import (
    MARKUSH_PROMPT_POLICY_VISIBLE_SG,
    VISIBLE_SG_INSTRUCTION,
)
from glyph.markush.eval.benchmarks import (
    get_benchmark_spec,
    normalize_benchmark_name,
    resolve_benchmark_path,
)
from glyph.markush.eval.checkpoint import load_eval_data
from glyph.markush.eval.checkpoint import load_eval_data as load_package_eval_data
from glyph.markush.eval.fast_eval import load_eval_samples as load_fast_eval_samples
from glyph.markush.eval.predict import load_eval_samples as load_predict_eval_samples


def test_benchmark_aliases_and_prompts():
    assert normalize_benchmark_name("IP5-M") == "ip5m"
    assert normalize_benchmark_name("uspto_markush") == "uspto-markush"
    assert get_benchmark_spec("ip5m").prompt == STRUCTURE_ONLY_MARKUSH_PROMPT
    assert get_benchmark_spec("m2s").prompt == PATENT_IMAGE_MARKUSH_PROMPT


def test_resolve_benchmark_path_uses_canonical_layout_under_root(tmp_path: Path):
    # No flat ``data/eval_ip5m.jsonl`` ships; the default resolves to the
    # canonical glyph.data.layout benchmark location under the data root.
    path = resolve_benchmark_path("ip5m", root=tmp_path)

    assert path == tmp_path / "benchmarks" / "markush" / "ip5m" / "eval_ip5m.jsonl"


def test_resolve_benchmark_path_uses_underscore_subdir(tmp_path: Path):
    # Hyphenated canonical name maps to the filesystem-safe underscore subdir.
    path = resolve_benchmark_path("uspto-markush", root=tmp_path)

    assert (
        path == tmp_path / "benchmarks" / "markush" / "uspto_markush" / "eval_uspto_markush.jsonl"
    )


def test_resolve_benchmark_path_explicit_wins(tmp_path: Path):
    explicit = tmp_path / "custom.jsonl"
    assert resolve_benchmark_path("ip5m", explicit_path=explicit, root=tmp_path) == explicit


def test_load_eval_data_uses_benchmark_default_prompt(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "sample.png"
    image.write_bytes(b"not-used-by-loader")
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(
        json.dumps(
            {
                "id": "sample",
                "image": "images/sample.png",
                "cxsmiles_opt": "CC",
                "annotation": "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
            }
        )
        + "\n"
    )

    records = load_eval_data(eval_jsonl, tmp_path, default_prompt=PATENT_IMAGE_MARKUSH_PROMPT)

    assert len(records) == 1
    assert records[0].prompt == (
        PATENT_IMAGE_MARKUSH_PROMPT
        + "\nTarget format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        "the <cxsmi> value must use cxsmiles_opt."
    )
    assert records[0].image_path == str(image)
    assert records[0].cxsmiles_opt == "CC"
    assert records[0].annotation == "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"


def test_load_eval_data_uses_training_target_priority_over_annotation(tmp_path: Path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-used-by-loader")
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(
        json.dumps(
            {
                "id": "sample",
                "image": "sample.png",
                "cxsmiles_opt": "<r>R1</r>CC",
                "annotation": (
                    "<markush><cxsmi>WRONG_RENDERED_CXSMILES</cxsmi>"
                    "<stable>R1:alkyl</stable></markush>"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_eval_data(eval_jsonl, tmp_path, default_prompt=STRUCTURE_ONLY_MARKUSH_PROMPT)

    assert records[0].cxsmiles_opt == "<r>R1</r>CC"
    assert records[0].annotation == (
        "<markush><cxsmi><r>R1</r>CC</cxsmi><stable>R1:alkyl</stable></markush>"
    )
    assert "the <cxsmi> value must use cxsmiles_opt" in records[0].prompt


def test_fast_and_predict_eval_loaders_share_training_target_contract(tmp_path: Path):
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "opt",
                        "image": "opt.png",
                        "cxsmiles_opt": "<r>R1</r>CC",
                        "annotation": "<markush><cxsmi>WRONG</cxsmi></markush>",
                    }
                ),
                json.dumps(
                    {
                        "id": "standard",
                        "image": "standard.png",
                        "cxsmiles": "*CC |$R1;;$|",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fast_samples = load_fast_eval_samples(
        eval_jsonl,
        tmp_path,
        STRUCTURE_ONLY_MARKUSH_PROMPT,
    )
    predict_samples = load_predict_eval_samples(
        eval_jsonl,
        tmp_path,
        STRUCTURE_ONLY_MARKUSH_PROMPT,
    )

    assert [sample.ground_truth for sample in fast_samples] == ["<r>R1</r>CC", "<r>R1</r>CC"]
    assert [sample.ground_truth for sample in predict_samples] == [
        "<r>R1</r>CC",
        "<r>R1</r>CC",
    ]
    assert "the <cxsmi> value must use cxsmiles_opt" in fast_samples[0].prompt
    assert "the <cxsmi> value must use cxsmiles_opt" in predict_samples[0].prompt
    assert "the <cxsmi> value must use cxsmiles_opt" in fast_samples[1].prompt
    assert "the <cxsmi> value must use cxsmiles_opt" in predict_samples[1].prompt


def test_eval_loaders_accept_sft_prepared_messages_and_images(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "sample.png"
    image.write_bytes(b"not-used-by-loader")
    assistant = "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"
    user = (
        "<image>\n"
        "Extract the Markush structure from this patent image.\n"
        "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        "the <cxsmi> value must use canonicalized cxsmiles_opt."
    )
    row = {
        "id": "sft_row",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "images": ["images/sample.png"],
        "target_format": "canonicalized cxsmiles_opt",
        "target_source": "cxsmiles_opt",
    }
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    expected_prompt = user.removeprefix("<image>\n")

    script_records = load_eval_data(
        eval_jsonl, tmp_path, default_prompt=STRUCTURE_ONLY_MARKUSH_PROMPT
    )
    package_records = load_package_eval_data(
        eval_jsonl,
        tmp_path,
        default_prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
    )
    fast_samples = load_fast_eval_samples(eval_jsonl, tmp_path, STRUCTURE_ONLY_MARKUSH_PROMPT)
    predict_samples = load_predict_eval_samples(eval_jsonl, tmp_path, STRUCTURE_ONLY_MARKUSH_PROMPT)

    assert target_cxsmiles_for_record(row) == "CC"
    assert script_records[0].prompt == expected_prompt
    assert script_records[0].annotation == assistant
    assert script_records[0].cxsmiles_opt == "CC"
    assert script_records[0].image_path == str(image)
    assert package_records[0].prompt == expected_prompt
    assert package_records[0].annotation == assistant
    assert package_records[0].cxsmiles_opt == "CC"
    assert package_records[0].image_path == str(image)
    assert fast_samples[0].prompt == expected_prompt
    assert fast_samples[0].ground_truth == "CC"
    assert fast_samples[0].image_path == str(image)
    assert predict_samples[0].prompt == expected_prompt
    assert predict_samples[0].ground_truth == "CC"
    assert predict_samples[0].image_path == str(image)


def test_fast_eval_can_ignore_prepared_prompts_for_diagnostic_ablation(tmp_path: Path):
    row = {
        "id": "prompted",
        "image": "sample.png",
        "prompt": (
            "Extract the Markush structure from this image.\n"
            "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
            "the <cxsmi> value must use canonicalized cxsmiles_opt."
        ),
        "cxsmiles_opt": "CC",
        "target_format": "canonicalized cxsmiles_opt",
        "target_source": "cxsmiles_opt->canonicalized_cxsmiles_opt",
    }
    (tmp_path / "sample.png").write_bytes(b"not-used-by-loader")
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    samples = load_fast_eval_samples(
        eval_jsonl,
        tmp_path,
        "Diagnostic prompt.",
        canonicalize_markush_targets=True,
        require_canonical_markush_targets=True,
        ignore_row_prompts=True,
    )

    assert samples[0].prompt.startswith("Diagnostic prompt.")
    assert "the <cxsmi> value must use canonicalized cxsmiles_opt." in samples[0].prompt


def test_fast_eval_can_apply_markush_prompt_policy_to_row_prompts(tmp_path: Path):
    row = {
        "id": "prompted",
        "image": "sample.png",
        "prompt": (
            "Extract the Markush structure from this image.\n"
            "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
            "the <cxsmi> value must use canonicalized cxsmiles_opt."
        ),
        "cxsmiles_opt": "CC",
        "target_format": "canonicalized cxsmiles_opt",
        "target_source": "cxsmiles_opt->canonicalized_cxsmiles_opt",
    }
    (tmp_path / "sample.png").write_bytes(b"not-used-by-loader")
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    samples = load_fast_eval_samples(
        eval_jsonl,
        tmp_path,
        STRUCTURE_ONLY_MARKUSH_PROMPT,
        canonicalize_markush_targets=True,
        require_canonical_markush_targets=True,
        markush_prompt_policy=MARKUSH_PROMPT_POLICY_VISIBLE_SG,
    )

    assert VISIBLE_SG_INSTRUCTION in samples[0].prompt
    assert samples[0].prompt.index(VISIBLE_SG_INSTRUCTION) < samples[0].prompt.index(
        "Target format:"
    )


def test_markush_eval_loaders_skip_molecule_ocsr_smiles_rows(tmp_path: Path):
    row = {
        "id": "ocsr_row",
        "mode": "molecule_ocsr",
        "effective_mode": "molecule_ocsr",
        "source": "molscribe",
        "source_id": "ocsr_row",
        "target_source": "smiles->rdkit_canonical_isomeric_smiles",
        "target_format": OCSR_TARGET_FORMAT,
        "target_canonicalization_status": "already_canonical",
        "target_canonicalization_error": "",
        "messages": [
            {
                "role": "user",
                "content": (
                    "<image>\nExtract the molecular structure from this image as SMILES.\n"
                    "Target format: return a SMILES string."
                ),
            },
            {"role": "assistant", "content": "CCO"},
        ],
        "images": ["sample.png"],
    }
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert load_eval_data(eval_jsonl, tmp_path, default_prompt=STRUCTURE_ONLY_MARKUSH_PROMPT) == []
    assert (
        load_package_eval_data(eval_jsonl, tmp_path, default_prompt=STRUCTURE_ONLY_MARKUSH_PROMPT)
        == []
    )
    assert load_fast_eval_samples(eval_jsonl, tmp_path, STRUCTURE_ONLY_MARKUSH_PROMPT) == []
    assert load_predict_eval_samples(eval_jsonl, tmp_path, STRUCTURE_ONLY_MARKUSH_PROMPT) == []


def test_ip5m_and_m2s_eval_inputs_match_sft_preparation(tmp_path: Path):
    cases = [
        (
            "ip5m",
            STRUCTURE_ONLY_MARKUSH,
            {
                "id": "ip5m_1",
                "source": "ip5m",
                "split": "test",
                "task": "markush",
                "image": "ip5m.png",
                "image_has_text": False,
                "cxsmiles_opt": "<r>R1</r>CC",
                "annotation": (
                    "<markush><cxsmi>WRONG_RENDERED_CXSMILES</cxsmi>"
                    "<stable>R1:alkyl</stable></markush>"
                ),
            },
        ),
        (
            "m2s",
            PATENT_IMAGE_MARKUSH,
            {
                "id": "m2s_1",
                "source": "m2s",
                "split": "test",
                "task": "markush",
                "image": "m2s.png",
                "image_has_text": True,
                "cxsmiles": "*CC |$R1;;$|",
            },
        ),
    ]

    for benchmark, mode, row in cases:
        eval_jsonl = tmp_path / f"{benchmark}.jsonl"
        eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
        expected = build_training_example(
            row,
            mode,
            text_dropout_rate=0.0,
            rng=random.Random(1),
        )
        expected_prompt = expected["messages"][0]["content"].removeprefix("<image>\n")
        expected_ground_truth = target_cxsmiles_for_record(row)
        prompt = get_benchmark_spec(benchmark).prompt

        script_records = load_eval_data(eval_jsonl, tmp_path, default_prompt=prompt)
        fast_samples = load_fast_eval_samples(eval_jsonl, tmp_path, prompt)
        predict_samples = load_predict_eval_samples(eval_jsonl, tmp_path, prompt)

        assert script_records[0].prompt == expected_prompt
        assert script_records[0].annotation == expected["messages"][1]["content"]
        assert script_records[0].cxsmiles_opt == expected_ground_truth
        assert fast_samples[0].prompt == expected_prompt
        assert fast_samples[0].ground_truth == expected_ground_truth
        assert predict_samples[0].prompt == expected_prompt
        assert predict_samples[0].ground_truth == expected_ground_truth


def test_eval_loaders_can_match_canonicalized_sft_targets(tmp_path: Path):
    row = {
        "id": "ip5m_canonical",
        "source": "ip5m",
        "split": "test",
        "task": "markush",
        "image": "canonical.png",
        "image_has_text": False,
        "cxsmiles_opt": "C(C)O",
    }
    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    expected = build_training_example(
        row,
        STRUCTURE_ONLY_MARKUSH,
        canonicalize_markush=True,
        rng=random.Random(1),
    )
    expected_prompt = expected["messages"][0]["content"].removeprefix("<image>\n")

    script_records = load_eval_data(
        eval_jsonl,
        tmp_path,
        default_prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
        canonicalize_markush_targets=True,
    )
    fast_samples = load_fast_eval_samples(
        eval_jsonl,
        tmp_path,
        STRUCTURE_ONLY_MARKUSH_PROMPT,
        canonicalize_markush_targets=True,
    )
    predict_samples = load_predict_eval_samples(
        eval_jsonl,
        tmp_path,
        STRUCTURE_ONLY_MARKUSH_PROMPT,
        canonicalize_markush_targets=True,
    )

    assert script_records[0].prompt == expected_prompt
    assert script_records[0].annotation == expected["messages"][1]["content"]
    assert script_records[0].cxsmiles_opt == "CCO"
    assert fast_samples[0].prompt == expected_prompt
    assert fast_samples[0].ground_truth == "CCO"
    assert predict_samples[0].prompt == expected_prompt
    assert predict_samples[0].ground_truth == "CCO"
