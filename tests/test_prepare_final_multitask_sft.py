from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from glyph.data import markush_multitask as prepare_final_multitask_sft


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _touch_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def _markush_train_row(image: Path) -> dict:
    target = "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
    prompt = (
        "<image>\nExtract the Markush structure from this image.\n"
        "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        "the <cxsmi> value must use canonicalized cxsmiles_opt."
    )
    return {
        "id": "markush_1_structure_only_markush",
        "task": "markush",
        "mode": "structure_only_markush",
        "effective_mode": "structure_only_markush",
        "source": "uspto_mol_m",
        "split": "train",
        "benchmark": None,
        "source_id": "markush_1",
        "target_source": "cxsmiles_opt->canonicalized_cxsmiles_opt",
        "target_format": "canonicalized cxsmiles_opt",
        "target_canonicalization_status": "already_canonical",
        "target_canonicalization_error": "",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "images": [str(image)],
    }


def _ocsr_source_row(image: Path) -> dict:
    return {
        "id": "ocsr_source_1",
        "task": "ocsr",
        "source": "molscribe",
        "split": "train",
        "smiles": "C(C)O",
        "image_path": str(image),
    }


def _ocsr_eval_row(image_ref: str) -> dict:
    prompt = (
        "Extract the molecular structure from this image as SMILES.\n"
        "Target format: return a SMILES string."
    )
    return {
        "id": "uspto_eval_1",
        "task": "ocsr",
        "mode": "molecule_ocsr",
        "source": "uspto",
        "split": "eval",
        "benchmark": "uspto",
        "image_path": image_ref,
        "images": [image_ref],
        "smiles": "CCO",
        "target_source": "smiles->rdkit_canonical_isomeric_smiles",
        "target_format": "SMILES",
        "target_canonicalization_status": "already_canonical",
        "target_canonicalization_error": "",
        "prompt": prompt,
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": "CCO"},
        ],
    }


def _run_final_prep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    extra_args: list[str] | None = None,
) -> Path:
    markush_image = _touch_image(tmp_path / "markush.png")
    ocsr_image = _touch_image(tmp_path / "ocsr.tif")
    eval_dir = tmp_path / "ocsr_eval"
    _touch_image(eval_dir / "images" / "uspto" / "eval.png")

    agent_a_train = tmp_path / "agent_a_train.jsonl"
    agent_a_manifest = tmp_path / "agent_a_manifest.json"
    agent_a_config = tmp_path / "agent_a_config.yaml"
    ocsr_source = tmp_path / "ocsr_source.jsonl"
    output_dir = tmp_path / "artifact"

    _write_jsonl(agent_a_train, [_markush_train_row(markush_image)])
    agent_a_manifest.write_text(json.dumps({"recipe": "agent-a-placeholder"}), encoding="utf-8")
    agent_a_config.write_text("vision_lora: true\n", encoding="utf-8")
    _write_jsonl(ocsr_source, [_ocsr_source_row(ocsr_image)])
    _write_jsonl(eval_dir / "eval_ocsr_uspto.jsonl", [_ocsr_eval_row("images/uspto/eval.png")])

    argv = [
        "prepare_final_multitask_sft.py",
        "--agent-a-markush-train-jsonl",
        str(agent_a_train),
        "--agent-a-manifest",
        str(agent_a_manifest),
        "--agent-a-config",
        str(agent_a_config),
        "--ocsr-source-jsonl",
        str(ocsr_source),
        "--ocsr-eval-dir",
        str(eval_dir),
        "--total-samples",
        "2",
        "--ocsr-fraction",
        "0.5",
        "--output-dir",
        str(output_dir),
        "--workers",
        "1",
        "--run-name",
        "test-final-multitask",
        *(extra_args or []),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert prepare_final_multitask_sft.main() == 0
    return output_dir


def test_final_multitask_prep_writes_valid_artifact_and_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = _run_final_prep(monkeypatch, tmp_path)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
    examples_report = (output_dir / "examples_report.md").read_text(encoding="utf-8")
    pilot_template = (output_dir / "pilot_command_template.sh").read_text(encoding="utf-8")
    train_rows = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["validation_ok"] is True
    assert manifest["missing_images"] == 0
    sampling = manifest["sampling"]
    assert sampling["seed"] == 42
    assert sampling["total_samples"] == 2
    assert sampling["markush_count"] == 1
    assert sampling["ocsr_count"] == 1
    assert sampling["selection_policy"] == "total_samples_fraction"
    assert sampling["requested_total_samples"] == 2
    assert sampling["requested_markush_count"] is None
    assert sampling["requested_ocsr_count"] is None
    assert sampling["requested_ocsr_fraction"] == 0.5
    assert sampling["ocsr_fraction_arg"] == 0.5
    assert sampling["actual_ocsr_fraction"] == 0.5
    assert manifest["ocsr"]["official_metric_mode"] == "native_smiles"
    assert manifest["ocsr"]["selected_metric_mode_for_template"] == "native_smiles"
    assert manifest["image_validation"] == {
        "checked": True,
        "require_absolute_images": False,
        "path_resolution": "absolute or relative to the owning JSONL",
    }
    assert validation["ok"] is True
    assert validation["missing_images"] == 0
    assert validation["files"][0]["task_counts"] == {"markush": 1, "ocsr": 1}
    assert validation["files"][1]["path"].endswith("eval_ocsr_uspto.jsonl")
    assert validation["files"][1]["missing_images"] == 0

    by_task = {row["task"]: row for row in train_rows}
    assert by_task["ocsr"]["mode"] == "molecule_ocsr"
    assert by_task["ocsr"]["target_format"] == "SMILES"
    assert by_task["ocsr"]["messages"][1]["content"] == "CCO"
    assert "<markush" not in by_task["ocsr"]["messages"][1]["content"].lower()
    assert by_task["markush"]["target_format"] == "canonicalized cxsmiles_opt"
    assert by_task["markush"]["messages"][1]["content"].startswith("<markush><cxsmi>")

    assert "Example: molecule_ocsr" in examples_report
    assert "Target format: return a SMILES string." in examples_report
    assert "Example: structure_only_markush" in examples_report
    assert "canonicalized cxsmiles_opt" in examples_report
    assert "Example: diagnostic_ocsr_as_no_extension_markush" in examples_report
    assert "diagnostic only" in examples_report.lower()
    assert "prepared/markush/eval" in pilot_template
    assert "export GLYPH_DATA_ROOT=${GLYPH_DATA_ROOT:-" in pilot_template
    assert "/data}" in pilot_template
    assert "export PY=${PY:-$SRC/.venv/bin/python}" in pilot_template
    assert "OCSR_METRICS_PREDICTION_MODE=native_smiles" in pilot_template
    assert '"$PY" -m glyph.markush.train.sft --config "$CONFIG"' in pilot_template


def test_final_multitask_prep_can_treat_ocsr_as_no_extension_markush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = _run_final_prep(
        monkeypatch,
        tmp_path,
        extra_args=["--ocsr-target-convention", "markush_no_extension"],
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
    examples_report = (output_dir / "examples_report.md").read_text(encoding="utf-8")
    pilot_template = (output_dir / "pilot_command_template.sh").read_text(encoding="utf-8")
    train_rows = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["validation_ok"] is True
    assert manifest["ocsr"]["target_convention"] == "markush_no_extension"
    assert manifest["ocsr"]["selected_metric_mode_for_template"] == "markush_no_extension"
    assert manifest["ocsr"]["converted_eval_dir"].endswith("ocsr_eval_markush_no_extension")
    assert validation["ok"] is True
    assert validation["files"][0]["task_counts"] == {"markush": 2}
    assert validation["files"][0]["mode_counts"] == {
        "molecule_ocsr_as_markush_no_extension": 1,
        "structure_only_markush": 1,
    }
    assert validation["files"][1]["target_format_counts"] == {"canonicalized cxsmiles_opt": 1}

    ocsr_as_markush = next(
        row for row in train_rows if row["mode"] == "molecule_ocsr_as_markush_no_extension"
    )
    assert ocsr_as_markush["task"] == "markush"
    assert ocsr_as_markush["target_format"] == "canonicalized cxsmiles_opt"
    assert ocsr_as_markush["cxsmiles_opt"] == "CCO"
    assert ocsr_as_markush["messages"][1]["content"] == (
        "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
    )
    assert (
        "ordinary molecule with no Markush extension annotations"
        in (ocsr_as_markush["messages"][0]["content"])
    )
    assert "Example: molecule_ocsr_as_markush_no_extension" in examples_report
    assert "Target format: return a SMILES string." not in examples_report
    assert "OCSR_METRICS_PREDICTION_MODE=markush_no_extension" in pilot_template
    assert "ocsr_eval_markush_no_extension" in pilot_template


def test_final_multitask_prep_manifest_marks_explicit_ocsr_count_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = _run_final_prep(
        monkeypatch,
        tmp_path,
        extra_args=["--ocsr-count", "1", "--ocsr-fraction", "0.05"],
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    sampling = manifest["sampling"]

    assert sampling["selection_policy"] == "total_samples_with_explicit_counts"
    assert sampling["requested_total_samples"] == 2
    assert sampling["requested_ocsr_count"] == 1
    assert sampling["requested_ocsr_fraction"] is None
    assert sampling["ocsr_fraction_arg"] == 0.05
    assert sampling["actual_ocsr_fraction"] == 0.5


def test_final_multitask_prep_rejects_non_markush_agent_a_train_jsonl(
    tmp_path: Path,
) -> None:
    agent_a_train = tmp_path / "agent_a_train.jsonl"
    _write_jsonl(agent_a_train, [{"id": "bad", "task": "ocsr", "target_format": "SMILES"}])

    with pytest.raises(ValueError, match="must contain only Markush prepared rows"):
        prepare_final_multitask_sft.require_markush_only(
            prepare_final_multitask_sft.load_jsonl(agent_a_train),
            path=agent_a_train,
        )


def test_final_multitask_prep_can_opt_into_absolute_image_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="failed validation"):
        _run_final_prep(monkeypatch, tmp_path, extra_args=["--require-absolute-images"])

    validation = json.loads(
        (tmp_path / "artifact" / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["ok"] is False
    assert any(
        failure["field"] == "images" and "not absolute" in failure["message"]
        for failure in validation["failures"]
    )
