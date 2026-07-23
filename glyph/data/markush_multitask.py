"""OCSR-as-Markush conversion + multitask SFT prep (``prepare_final_multitask_sft`` home).

Holds the reusable logic behind the ``scripts/prepare_final_multitask_sft.py``
CLI (which is a thin wrapper over :func:`main`). The pieces the release
orchestrator (:mod:`glyph.data.markush`) needs live here:

* the OCSR -> no-extension-Markush target conversion (the "5% ordinary" arm),
* deterministic Markush/OCSR count + sampling helpers,
* an 80-row decontamination against the eval canonical-SMILES set,
* image-reference relativization on emit: :func:`relativize_row_images` (re-used
  from :mod:`glyph.data.markush_mix`) rewrites image paths ``$GLYPH_DATA_ROOT``-
  relative so no absolute path leaks into the prepared mix,
* the ``main``/``parse_args`` CLI that builds the artifact and validates it with
  :func:`glyph.data.markush_check.check_artifacts`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from glyph.data.markush_check import check_artifacts
from glyph.data.markush_mix import relativize_row_images
from glyph.markush.data.canonicalize import canonicalize_cxsmiles_checked
from glyph.markush.data.example_modes import MOLECULE_OCSR
from glyph.markush.data.ocsr import canonicalize_ocsr_smiles_checked
from glyph.markush.data.prepare_sft import MixWeights, prepare_sft_jsonl
from glyph.markush.eval import common
from glyph.markush.eval.ocsr import (
    OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
    OCSR_PREDICTION_MODE_NATIVE_SMILES,
)
from glyph.paths import data_root, models_dir, prepared_dir

OCSR_TARGET_CONVENTION_NATIVE_SMILES = "native_smiles"
OCSR_TARGET_CONVENTION_MARKUSH_NO_EXTENSION = "markush_no_extension"
OCSR_AS_MARKUSH_MODE = "molecule_ocsr_as_markush_no_extension"
MARKUSH_TARGET_FORMAT = "canonicalized cxsmiles_opt"


def _target_instruction(expected_target_format: str) -> str:
    """Return the target-format instruction appended to the training prompt.

    Byte-identical to :func:`glyph.data.markush_check.target_instruction` so the
    prepared prompts and the validator agree.
    """

    return (
        "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        f"the <cxsmi> value must use {expected_target_format}."
    )


MARKUSH_NO_EXTENSION_PROMPT = (
    "Extract the Markush structure from this image.\n"
    f"{_target_instruction(MARKUSH_TARGET_FORMAT)}\n"
    "This image is an ordinary molecule with no Markush extension annotations; "
    "return only the base molecule in <cxsmi> and leave <stable> empty."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def message_content(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def first_image(row: dict[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    image = row.get("image") or row.get("image_path")
    return str(image or "")


def count_rows(
    *,
    total_samples: int | None,
    markush_available: int,
    ocsr_fraction: float,
    ocsr_count: int | None,
    markush_count: int | None,
) -> tuple[int, int]:
    if not 0.0 <= ocsr_fraction < 1.0:
        raise ValueError("ocsr_fraction must be >=0 and <1")
    if total_samples is not None and total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if ocsr_count is not None and ocsr_count < 0:
        raise ValueError("ocsr_count must be non-negative")
    if markush_count is not None and markush_count <= 0:
        raise ValueError("markush_count must be positive")
    if total_samples is not None:
        target_ocsr = ocsr_count if ocsr_count is not None else round(total_samples * ocsr_fraction)
        if target_ocsr > total_samples:
            raise ValueError("ocsr_count cannot exceed total_samples")
        target_markush = markush_count if markush_count is not None else total_samples - target_ocsr
        if target_markush + target_ocsr != total_samples:
            raise ValueError("markush_count + ocsr_count must equal total_samples")
        return target_markush, target_ocsr
    target_markush = markush_count or markush_available
    if ocsr_count is not None:
        target_ocsr = ocsr_count
    elif ocsr_fraction == 0.0:
        target_ocsr = 0
    else:
        target_ocsr = math.ceil(target_markush * ocsr_fraction / (1.0 - ocsr_fraction))
    return target_markush, target_ocsr


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    rng: random.Random,
    oversample: bool,
    label: str,
) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if not rows:
        raise ValueError(f"No rows available for {label}")
    if count <= len(rows):
        return rng.sample(rows, count)
    if not oversample:
        raise ValueError(f"Requested {count} {label} rows but only {len(rows)} are available")
    return [rng.choice(rows) for _ in range(count)]


def require_markush_only(rows: list[dict[str, Any]], *, path: Path) -> None:
    bad_rows = [
        row.get("id", index)
        for index, row in enumerate(rows, 1)
        if str(row.get("task") or "").lower() == "ocsr"
        or str(row.get("target_format") or "") == "SMILES"
        or str(row.get("mode") or "") == MOLECULE_OCSR
    ]
    if bad_rows:
        preview = ", ".join(str(value) for value in bad_rows[:5])
        raise ValueError(
            "Markush train JSONL must contain only Markush prepared rows; "
            f"found OCSR/SMILES rows in {path}: {preview}"
        )


def link_eval_jsonls(paths: list[Path], output_dir: Path) -> Path | None:
    if not paths:
        return None
    link_dir = output_dir / "ocsr_eval_jsonls"
    link_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if not source.exists():
            raise FileNotFoundError(source)
        if not source.name.startswith("eval_ocsr_"):
            raise ValueError(f"OCSR eval JSONL must be named eval_ocsr_*.jsonl: {source}")
        dest = link_dir / source.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(source)
        except OSError:
            shutil.copy2(source, dest)
    return link_dir


def ocsr_eval_jsonl_paths(eval_dir: Path | None, eval_jsonls: list[Path]) -> list[Path]:
    paths: list[Path] = []
    if eval_dir is not None:
        if not eval_dir.exists():
            raise FileNotFoundError(eval_dir)
        paths.extend(sorted(eval_dir.glob("eval_ocsr_*.jsonl")))
    paths.extend(eval_jsonls)
    return sorted(dict.fromkeys(paths))


def canonical_ocsr_target_from_row(row: dict[str, Any]) -> str:
    raw = message_content(row, "assistant") or str(row.get("smiles") or "")
    cxsmi, _stable = common.extract_cxsmiles_and_stable(raw, require_wrapper=False)
    candidate = cxsmi or raw
    result = canonicalize_ocsr_smiles_checked(candidate)
    if result.value is None:
        row_id = row.get("id", "<unknown>")
        raise ValueError(f"OCSR row {row_id!r} has no valid canonical SMILES target")
    return result.value


def canonical_markush_no_extension_target_from_row(row: dict[str, Any]) -> tuple[str, str]:
    canonical_smiles = canonical_ocsr_target_from_row(row)
    result = canonicalize_cxsmiles_checked(canonical_smiles)
    if result.value is None or result.fallback or not result.accepted:
        row_id = row.get("id", "<unknown>")
        raise ValueError(
            f"OCSR row {row_id!r} cannot be represented as fixed-point "
            f"canonicalized cxsmiles_opt: {result.error or result.status}"
        )
    if "|" in result.value or "<r>" in result.value.lower():
        row_id = row.get("id", "<unknown>")
        raise ValueError(
            f"OCSR row {row_id!r} produced a Markush/CXSMILES extension in no-extension mode"
        )
    return result.value, result.status


def ocsr_row_as_no_extension_markush(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a prepared OCSR row into the Markush-subset target convention."""

    canonical_smiles, canonical_status = canonical_markush_no_extension_target_from_row(row)
    target = f"<markush><cxsmi>{canonical_smiles}</cxsmi><stable></stable></markush>"
    converted = dict(row)
    converted["task"] = "markush"
    converted["mode"] = OCSR_AS_MARKUSH_MODE
    converted["effective_mode"] = OCSR_AS_MARKUSH_MODE
    converted["target_source"] = (
        f"{row.get('target_source') or 'smiles->rdkit_canonical_isomeric_smiles'}"
        "->no_extension_markush_xml->canonicalized_cxsmiles_opt"
    )
    converted["target_format"] = MARKUSH_TARGET_FORMAT
    converted["target_cxsmiles_format"] = MARKUSH_TARGET_FORMAT
    converted["target_canonicalization_status"] = canonical_status
    converted["target_canonicalization_error"] = ""
    converted["target_conversion_status"] = "ocsr_no_extension_markush_xml"
    converted["target_conversion_reason"] = ""
    converted["target_conversion_detail"] = ""
    converted["smiles"] = canonical_smiles
    converted["cxsmiles_opt"] = canonical_smiles
    converted["stable"] = ""
    converted["annotation"] = target
    converted["ground_truth"] = target
    converted["prompt"] = MARKUSH_NO_EXTENSION_PROMPT
    converted["messages"] = [
        {"role": "user", "content": f"<image>\n{MARKUSH_NO_EXTENSION_PROMPT}"},
        {"role": "assistant", "content": target},
    ]
    return converted


def _absolute_image_ref(value: str, *, root: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else root / path)


def absolutize_row_images(row: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Absolutize a row's image refs against ``root``.

    Used by :func:`materialize_no_extension_ocsr_eval_jsonls` to pin each
    converted OCSR eval row's image paths to absolute locations before the
    JSONL is written. (The training mix instead emits ``$GLYPH_DATA_ROOT``-
    relative refs via :func:`glyph.data.markush_mix.relativize_row_images`.)
    """

    converted = dict(row)
    images = converted.get("images")
    if isinstance(images, list):
        converted["images"] = [
            _absolute_image_ref(str(value), root=root) if isinstance(value, str) else value
            for value in images
        ]
    for key in ("image", "image_path", "path", "png_path", "rendered_path", "file_name"):
        value = converted.get(key)
        if isinstance(value, str) and value.strip():
            converted[key] = _absolute_image_ref(value, root=root)
    return converted


def convert_ocsr_rows_to_no_extension_markush(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        try:
            converted.append(ocsr_row_as_no_extension_markush(row))
        except ValueError as exc:
            skipped.append(
                {
                    "id": row.get("id"),
                    "source_id": row.get("source_id"),
                    "source": row.get("source"),
                    "smiles": row.get("smiles") or message_content(row, "assistant"),
                    "reason": str(exc),
                }
            )
    return converted, skipped


def eval_canonical_smiles_set(eval_jsonls: list[Path]) -> set[str]:
    """Return the set of canonical OCSR SMILES appearing in the eval JSONLs.

    Used for the decontamination gate: any candidate OCSR train row whose
    canonical SMILES appears here is excluded from the mix.
    """

    canon: set[str] = set()
    for path in eval_jsonls:
        for row in load_jsonl(path):
            try:
                canon.add(canonical_ocsr_target_from_row(row))
            except ValueError:
                continue
    return canon


def decontaminate_ocsr_rows(
    rows: list[dict[str, Any]],
    *,
    eval_canonical_smiles: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop OCSR rows whose canonical SMILES collides with the eval set.

    Returns ``(kept, removed)``; ``removed`` entries carry the offending
    canonical SMILES so the caller can record the exclusion list (matching the
    cached ``ocsr-fraction-decontam-summary-seed74.json`` shape).
    """

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for row in rows:
        try:
            canonical = canonical_ocsr_target_from_row(row)
        except ValueError:
            kept.append(row)
            continue
        if canonical in eval_canonical_smiles:
            removed.append({"id": row.get("id"), "canonical_smiles": canonical})
        else:
            kept.append(row)
    return kept, removed


def relativize_rows(rows: list[dict[str, Any]], *, root: Path) -> tuple[list[dict[str, Any]], int]:
    """Rewrite every row's image refs to ``root``-relative; count unresolved rows."""

    out: list[dict[str, Any]] = []
    unresolved = 0
    for row in rows:
        rewritten, was_unresolved = relativize_row_images(row, root=root)
        if was_unresolved:
            unresolved += 1
        out.append(rewritten)
    return out, unresolved


def row_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters = {
        "task": Counter(str(row.get("task")) for row in rows),
        "mode": Counter(str(row.get("mode")) for row in rows),
        "target_format": Counter(str(row.get("target_format")) for row in rows),
        "target_source": Counter(str(row.get("target_source")) for row in rows),
        "source": Counter(str(row.get("source")) for row in rows),
        "image_suffix": Counter(Path(first_image(row)).suffix.lower() for row in rows),
    }
    return {key: dict(counter) for key, counter in counters.items()}


def build_ocsr_as_markush_partition(
    *,
    ocsr_source_jsonl: Path,
    out_jsonl: Path,
    benchmark_eval_jsonls: list[Path] | None = None,
) -> dict[str, Any]:
    """Build the "5% ordinary" OCSR-as-no-extension-Markush partition.

    Reads the imported MolScribe OCSR ``unified.jsonl``, converts each row to the
    no-extension Markush target convention, decontaminates against the benchmark
    canonical-SMILES set (the 80-row exclusion in the cached mix), and writes the
    resulting partition JSONL. Returns a summary matching the cached
    ``ocsr-fraction-decontam-summary-seed74.json`` shape.
    """

    rows = load_jsonl(ocsr_source_jsonl)
    converted, skipped = convert_ocsr_rows_to_no_extension_markush(rows)
    eval_canon = eval_canonical_smiles_set(benchmark_eval_jsonls or [])
    kept, removed = decontaminate_ocsr_rows(converted, eval_canonical_smiles=eval_canon)
    write_jsonl(kept, out_jsonl)
    return {
        "source_jsonl": str(ocsr_source_jsonl),
        "output_jsonl": str(out_jsonl),
        "source_rows": len(rows),
        "converted_rows": len(converted),
        "conversion_skipped_rows": len(skipped),
        "conversion_skipped_preview": skipped[:20],
        "eval_canonical_smiles": len(eval_canon),
        "decontamination_removed_count": len(removed),
        "decontamination_removed_preview": removed[:20],
        "partition_rows": len(kept),
    }


def assemble_multitask_mix(
    *,
    base_jsonl: Path,
    ocsr_source_jsonl: Path,
    out_dir: Path,
    data_root: Path,
    benchmark_eval_jsonls: list[Path] | None = None,
    seed: int = 74,
    ocsr_fraction: float = 0.05,
    shuffle: bool = True,
) -> dict[str, Any]:
    """Assemble the 5% multitask mix from a Markush base + a MolScribe OCSR source.

    Deterministic (seeded) recipe mirroring the cached
    ``ocsr-fraction-5pct-seed74-decontam`` artifact:

    1. build the OCSR-as-no-extension-Markush partition (convert + decontaminate);
    2. size the OCSR count as ``round(base * f/(1-f))`` (== 13,158 for 250K @ 5%);
    3. combine ``[base, ordinary]`` with :func:`glyph.data.markush_mix.build_mix`
       at the given seed, rewriting every image ref ``data_root``-relative.

    The 250K base itself (200K broad-canonical + 50K frozen checkpoint-mined-hard)
    is an *input* here (the hard slice is a frozen cached artifact, not regenerable
    from scratch). Returns the build_mix manifest augmented with the OCSR summary.
    """

    from glyph.data.markush_mix import build_mix, parse_partition

    base_jsonl = base_jsonl.expanduser()
    ocsr_source_jsonl = ocsr_source_jsonl.expanduser()
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_count = sum(1 for line in base_jsonl.open(encoding="utf-8") if line.strip())
    if base_count <= 0:
        raise ValueError(f"base partition {base_jsonl} is empty")
    if not 0.0 <= ocsr_fraction < 1.0:
        raise ValueError("ocsr_fraction must be >=0 and <1")
    ocsr_count = round(base_count * ocsr_fraction / (1.0 - ocsr_fraction))

    ocsr_partition = out_dir / "ordinary_partition.jsonl"
    ocsr_summary = build_ocsr_as_markush_partition(
        ocsr_source_jsonl=ocsr_source_jsonl,
        out_jsonl=ocsr_partition,
        benchmark_eval_jsonls=benchmark_eval_jsonls,
    )

    train_jsonl = out_dir / "train.jsonl"
    manifest = build_mix(
        partitions=[
            parse_partition(f"base={base_count}:{base_jsonl}"),
            parse_partition(f"ordinary={ocsr_count}:{ocsr_partition}"),
        ],
        output_jsonl=train_jsonl,
        manifest_json=out_dir / "manifest.json",
        report_md=out_dir / "report.md",
        image_roots={},
        seed=seed,
        oversample=True,
        shuffle=shuffle,
        require_absolute=False,
        allow_eval_sources=False,
        argv=[],
        relativize_to_root=data_root,
    )
    manifest["ocsr_partition_summary"] = ocsr_summary
    manifest["requested_ocsr_fraction"] = ocsr_fraction
    manifest["base_count"] = base_count
    manifest["ocsr_count"] = ocsr_count
    manifest["ocsr_count_formula"] = "round(base * f / (1 - f))"
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


# --------------------------------------------------------------------------
# Multitask SFT prep CLI (backs scripts/prepare_final_multitask_sft.py).
# --------------------------------------------------------------------------

#: Prepared-eval subdirectory (under the data root's ``prepared/``) used when
#: ``--markush-eval-dir`` is omitted. This is where the Markush eval prep writes
#: its per-benchmark eval JSONLs; pass ``--markush-eval-dir`` to point the pilot
#: template at a different prepared eval set.
DEFAULT_MARKUSH_EVAL_REL = Path("markush/eval")


def sampling_policy(args: argparse.Namespace) -> str:
    if args.total_samples is not None:
        return (
            "total_samples_with_explicit_counts"
            if args.ocsr_count is not None
            else "total_samples_fraction"
        )
    if args.ocsr_count is not None:
        return "explicit_ocsr_count"
    if args.ocsr_fraction == 0.0:
        return "markush_only"
    return "final_mix_fraction"


def materialize_no_extension_ocsr_eval_jsonls(
    paths: list[Path],
    output_dir: Path,
) -> tuple[Path | None, dict[str, Any]]:
    if not paths:
        return None, {"total_input_rows": 0, "written_rows": 0, "skipped_rows": 0, "files": []}
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "total_input_rows": 0,
        "written_rows": 0,
        "skipped_rows": 0,
        "files": [],
    }
    for source in paths:
        rows = load_jsonl(source)
        converted_rows, skipped = convert_ocsr_rows_to_no_extension_markush(rows)
        converted = [absolutize_row_images(row, root=source.parent) for row in converted_rows]
        write_jsonl(converted, output_dir / source.name)
        report["total_input_rows"] += len(rows)
        report["written_rows"] += len(converted)
        report["skipped_rows"] += len(skipped)
        report["files"].append(
            {
                "source": str(source),
                "output": str(output_dir / source.name),
                "input_rows": len(rows),
                "written_rows": len(converted),
                "skipped_rows": len(skipped),
                "skipped_preview": skipped[:20],
            }
        )
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir, report


def representative_examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for mode in (MOLECULE_OCSR, OCSR_AS_MARKUSH_MODE, "structure_only_markush"):
        row = next((item for item in rows if item.get("mode") == mode), None)
        if row is None:
            continue
        examples[mode] = {
            "id": row.get("id"),
            "task": row.get("task"),
            "mode": row.get("mode"),
            "source": row.get("source"),
            "image": first_image(row),
            "target_format": row.get("target_format"),
            "target_source": row.get("target_source"),
            "target_canonicalization_status": row.get("target_canonicalization_status"),
            "prompt": message_content(row, "user"),
            "target": message_content(row, "assistant"),
        }
    if MOLECULE_OCSR in examples:
        target = examples[MOLECULE_OCSR]["target"]
        examples["diagnostic_ocsr_as_no_extension_markush"] = {
            "diagnostic_only": True,
            "evaluator_prediction_mode": OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
            "source_example_id": examples[MOLECULE_OCSR]["id"],
            "prompt": (
                "<image>\n"
                "Diagnostic only: extract this ordinary molecule as no-extension Markush XML.\n"
                "Target format: return <markush><cxsmi>...</cxsmi><stable></stable></markush> "
                "with no CXSMILES extension section and no Markush annotations."
            ),
            "target": f"<markush><cxsmi>{target}</cxsmi><stable></stable></markush>",
            "note": (
                "Not included in train.jsonl. Use only for format-control diagnostics, "
                "not official OCSR benchmark reporting."
            ),
        }
    return examples


def validate_examples(examples: dict[str, dict[str, Any]]) -> None:
    ocsr = examples.get(MOLECULE_OCSR)
    if ocsr:
        target = str(ocsr.get("target") or "")
        canonical = canonicalize_ocsr_smiles_checked(target)
        if canonical.value != target:
            raise ValueError("Representative OCSR target is not bare RDKit-canonical SMILES")
        if "<markush" in target.lower() or "<cxsmi" in target.lower():
            raise ValueError("Representative OCSR target is Markush XML, not bare SMILES")
    ocsr_as_markush = examples.get(OCSR_AS_MARKUSH_MODE)
    if ocsr_as_markush:
        target = str(ocsr_as_markush.get("target") or "")
        cxsmi, stable = common.extract_cxsmiles_and_stable(target, require_wrapper=True)
        if not cxsmi:
            raise ValueError("Representative OCSR-as-Markush target has no <cxsmi> payload")
        if stable:
            raise ValueError("Representative OCSR-as-Markush target has non-empty stable text")
        canonical = canonicalize_ocsr_smiles_checked(cxsmi)
        if canonical.value != cxsmi:
            raise ValueError("Representative OCSR-as-Markush target is not canonical SMILES")
    markush = examples.get("structure_only_markush")
    if markush:
        target = str(markush.get("target") or "")
        if not target.startswith("<markush><cxsmi>") or "</cxsmi><stable>" not in target:
            raise ValueError("Representative Markush target does not use Markush XML")


def fenced(value: str) -> str:
    return "```text\n" + value + "\n```"


def write_examples_report(
    *,
    path: Path,
    artifact_root: Path,
    train_jsonl: Path,
    validation_report: Path,
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
    examples: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Final Multitask Prep Artifact",
        "",
        f"Artifact: {artifact_root}",
        f"Train JSONL: {train_jsonl}",
        f"Validation report: {validation_report}",
        f"Rows: {len(rows)}",
        f"Validation ok: {validation.get('ok')}",
        f"Missing images: {validation.get('missing_images')}",
        "Training launched: no",
        "",
        "## Composition",
        "",
    ]
    for name, counter in row_counts(rows).items():
        lines.append(f"- {name}: {counter}")
    lines.extend(
        [
            "",
            "## OCSR Metric Modes",
            "",
            (
                f"- Official OCSR scoring uses `{OCSR_PREDICTION_MODE_NATIVE_SMILES}`. "
                "This is the default and benchmark-compatible mode."
            ),
            (
                f"- Diagnostic no-extension Markush scoring uses "
                f"`{OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION}` and must be opt-in."
            ),
            "- Diagnostic examples below are not training rows unless explicitly staged later.",
        ]
    )
    for name, example in examples.items():
        lines.extend(["", f"## Example: {name}", ""])
        for key, value in example.items():
            if key in {"prompt", "target"}:
                continue
            lines.append(f"- {key}: {value}")
        lines.extend(["", "Prompt:", fenced(str(example.get("prompt") or ""))])
        lines.extend(["Target:", fenced(str(example.get("target") or ""))])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pilot_template(
    *,
    path: Path,
    train_jsonl: Path,
    agent_a_config: Path,
    markush_eval_dir: Path,
    ocsr_eval_data: Path | None,
    ocsr_metrics_prediction_mode: str,
    run_name: str,
    root: Path,
) -> None:
    ocsr_eval_value = str(ocsr_eval_data or "<OCSR_EVAL_ROOT_OR_JSONL>")
    model = models_dir("Qwen3.5-2B-Base", root=root)
    text = f"""#!/usr/bin/env bash
# Template only. Review the final recipe/config before submitting.
set -euo pipefail

export GLYPH_DATA_ROOT=${{GLYPH_DATA_ROOT:-{root}}}
export SRC=${{SRC:-$(pwd)}}
export PY=${{PY:-$SRC/.venv/bin/python}}
export MODEL=${{MODEL:-{model}}}
export CONFIG={agent_a_config}
export RUN_NAME={run_name}
export DATA={train_jsonl}
export EVAL_DIR={markush_eval_dir}
export OCSR_METRICS_EVAL_DATA={ocsr_eval_value}
export OCSR_METRICS_EVAL_SIZE=${{OCSR_METRICS_EVAL_SIZE:-100}}
export OCSR_METRICS_PREDICTION_MODE={ocsr_metrics_prediction_mode}
export VISION_LORA=${{VISION_LORA:-1}}
export NUM_PROCESSES=${{NUM_PROCESSES:-8}}

# Set MAX_STEPS/SAVE_STEPS/EVAL_STEPS from the final recipe before launch.
# Do not run this template until the recipe has been reviewed.
"$PY" -m glyph.markush.train.sft --config "$CONFIG" --data-root "$GLYPH_DATA_ROOT"
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root for MarkushGlyph data (default: GLYPH_DATA_ROOT or ./data).",
    )
    parser.add_argument("--agent-a-markush-train-jsonl", type=Path, required=True)
    parser.add_argument("--agent-a-manifest", type=Path, required=True)
    parser.add_argument("--agent-a-config", type=Path, required=True)
    parser.add_argument("--ocsr-source-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-samples", type=int)
    parser.add_argument("--markush-count", type=int)
    parser.add_argument("--ocsr-count", type=int)
    parser.add_argument("--ocsr-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--markush-oversample", action="store_true")
    parser.add_argument("--ocsr-oversample", action="store_true")
    parser.add_argument("--markush-eval-dir", type=Path)
    parser.add_argument("--ocsr-eval-dir", type=Path)
    parser.add_argument("--ocsr-eval-jsonl", type=Path, action="append", default=[])
    parser.add_argument(
        "--ocsr-metrics-prediction-mode",
        choices=[
            OCSR_PREDICTION_MODE_NATIVE_SMILES,
            OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
        ],
        default=OCSR_PREDICTION_MODE_NATIVE_SMILES,
    )
    parser.add_argument(
        "--ocsr-target-convention",
        choices=[
            OCSR_TARGET_CONVENTION_NATIVE_SMILES,
            OCSR_TARGET_CONVENTION_MARKUSH_NO_EXTENSION,
        ],
        default=OCSR_TARGET_CONVENTION_NATIVE_SMILES,
        help=(
            "Training target convention for OCSR rows. native_smiles preserves "
            "official OCSR prompts/targets. markush_no_extension treats ordinary "
            "molecule images as no-extension Markush targets."
        ),
    )
    parser.add_argument("--run-name", default="sft-2b-final-ocsr-markush-pilot")
    parser.add_argument("--no-check-images", action="store_true")
    parser.add_argument(
        "--require-absolute-images",
        action="store_true",
        help=(
            "Require image paths to be absolute. By default, validation accepts "
            "absolute paths or paths relative to each JSONL, matching the eval loader."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = data_root(args.data_root)
    if args.markush_eval_dir is None:
        args.markush_eval_dir = prepared_dir(DEFAULT_MARKUSH_EVAL_REL, root=root)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (args.agent_a_markush_train_jsonl, args.agent_a_manifest, args.agent_a_config):
        if not required.exists():
            raise FileNotFoundError(required)

    markush_rows = load_jsonl(args.agent_a_markush_train_jsonl)
    require_markush_only(markush_rows, path=args.agent_a_markush_train_jsonl)
    markush_count, ocsr_count = count_rows(
        total_samples=args.total_samples,
        markush_available=len(markush_rows),
        ocsr_fraction=args.ocsr_fraction,
        ocsr_count=args.ocsr_count,
        markush_count=args.markush_count,
    )

    rng = random.Random(args.seed)
    selected_markush = sample_rows(
        markush_rows,
        count=markush_count,
        rng=rng,
        oversample=args.markush_oversample,
        label="Markush",
    )

    ocsr_prepared = output_dir / "ocsr_prepared.jsonl"
    ocsr_stats = output_dir / "ocsr_prepared_stats.json"
    if ocsr_count:
        prepare_sft_jsonl(
            datasets=[args.ocsr_source_jsonl],
            output=ocsr_prepared,
            stats_json=ocsr_stats,
            total_samples=ocsr_count,
            text_dropout=0.0,
            weights=MixWeights(
                structure_only_markush=0.0,
                patent_image_markush=0.0,
                text_conditioned_markush=0.0,
                molecule_ocsr=1.0,
                molparser_native_markush=0.0,
                molparser_converted_markush=0.0,
                structured_transcript_markush=0.0,
            ),
            seed=args.seed,
            include_eval=False,
            oversample=args.ocsr_oversample,
            canonicalize_markush_targets=False,
            require_canonical_markush_targets=False,
            workers=args.workers,
        )
        selected_ocsr = load_jsonl(ocsr_prepared)
    else:
        write_jsonl([], ocsr_prepared)
        ocsr_stats.write_text(json.dumps({"written": 0}, indent=2) + "\n", encoding="utf-8")
        selected_ocsr = []

    ocsr_train_conversion_skipped: list[dict[str, Any]] = []
    ocsr_eval_conversion_report: dict[str, Any] = {}
    if args.ocsr_target_convention == OCSR_TARGET_CONVENTION_MARKUSH_NO_EXTENSION:
        selected_ocsr, ocsr_train_conversion_skipped = convert_ocsr_rows_to_no_extension_markush(
            selected_ocsr
        )
        (output_dir / "ocsr_train_no_extension_conversion_report.json").write_text(
            json.dumps(
                {
                    "requested_rows": ocsr_count,
                    "written_rows": len(selected_ocsr),
                    "skipped_rows": len(ocsr_train_conversion_skipped),
                    "skipped_preview": ocsr_train_conversion_skipped[:100],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.ocsr_metrics_prediction_mode == OCSR_PREDICTION_MODE_NATIVE_SMILES:
            args.ocsr_metrics_prediction_mode = OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION

    mixed_rows = selected_markush + selected_ocsr
    rng.shuffle(mixed_rows)
    train_jsonl = output_dir / "train.jsonl"
    write_jsonl(mixed_rows, train_jsonl)

    eval_dirs: list[Path] = []
    linked_eval_dir: Path | None = None
    converted_eval_dir: Path | None = None
    if args.ocsr_target_convention == OCSR_TARGET_CONVENTION_MARKUSH_NO_EXTENSION:
        converted_eval_dir, ocsr_eval_conversion_report = materialize_no_extension_ocsr_eval_jsonls(
            ocsr_eval_jsonl_paths(args.ocsr_eval_dir, args.ocsr_eval_jsonl),
            output_dir / "ocsr_eval_markush_no_extension",
        )
        if converted_eval_dir:
            eval_dirs.append(converted_eval_dir)
    else:
        if args.ocsr_eval_dir:
            eval_dirs.append(args.ocsr_eval_dir)
        linked_eval_dir = link_eval_jsonls(args.ocsr_eval_jsonl, output_dir)
        if linked_eval_dir:
            eval_dirs.append(linked_eval_dir)

    validation_report = output_dir / "validation_report.json"
    validation = check_artifacts(
        train_jsonls=[train_jsonl],
        eval_dirs=eval_dirs,
        expected_target_format="canonicalized cxsmiles_opt",
        require_canonicalized=True,
        allow_structured_transcript_targets=False,
        allow_ocsr_smiles_targets=True,
        check_images=not args.no_check_images,
        require_absolute_images=args.require_absolute_images,
        workers=args.workers,
    )
    validation_report.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if not validation.get("ok"):
        raise SystemExit(f"Prepared multitask artifact failed validation: {validation_report}")

    examples = representative_examples(mixed_rows)
    validate_examples(examples)
    examples_report = output_dir / "examples_report.md"
    write_examples_report(
        path=examples_report,
        artifact_root=output_dir,
        train_jsonl=train_jsonl,
        validation_report=validation_report,
        rows=mixed_rows,
        validation=validation,
        examples=examples,
    )

    pilot_template = output_dir / "pilot_command_template.sh"
    write_pilot_template(
        path=pilot_template,
        train_jsonl=train_jsonl,
        agent_a_config=args.agent_a_config,
        markush_eval_dir=args.markush_eval_dir,
        ocsr_eval_data=converted_eval_dir or args.ocsr_eval_dir or linked_eval_dir,
        ocsr_metrics_prediction_mode=args.ocsr_metrics_prediction_mode,
        run_name=args.run_name,
        root=root,
    )

    manifest = {
        "artifact_root": str(output_dir),
        "train_jsonl": str(train_jsonl),
        "validation_report": str(validation_report),
        "examples_report": str(examples_report),
        "pilot_command_template": str(pilot_template),
        "agent_a": {
            "markush_train_jsonl": str(args.agent_a_markush_train_jsonl),
            "manifest": str(args.agent_a_manifest),
            "config": str(args.agent_a_config),
            "markush_eval_dir": str(args.markush_eval_dir),
        },
        "ocsr": {
            "source_jsonl": str(args.ocsr_source_jsonl),
            "prepared_jsonl": str(ocsr_prepared),
            "prepared_stats": str(ocsr_stats),
            "target_convention": args.ocsr_target_convention,
            "as_markush_mode": (
                OCSR_AS_MARKUSH_MODE
                if args.ocsr_target_convention == OCSR_TARGET_CONVENTION_MARKUSH_NO_EXTENSION
                else ""
            ),
            "eval_dir": str(args.ocsr_eval_dir) if args.ocsr_eval_dir else "",
            "converted_eval_dir": str(converted_eval_dir) if converted_eval_dir else "",
            "train_conversion": {
                "requested_rows": ocsr_count,
                "written_rows": len(selected_ocsr),
                "skipped_rows": len(ocsr_train_conversion_skipped),
                "skipped_preview": ocsr_train_conversion_skipped[:20],
            },
            "eval_conversion": ocsr_eval_conversion_report,
            "eval_jsonls": [str(path) for path in args.ocsr_eval_jsonl],
            "official_metric_mode": OCSR_PREDICTION_MODE_NATIVE_SMILES,
            "selected_metric_mode_for_template": args.ocsr_metrics_prediction_mode,
        },
        "sampling": {
            "seed": args.seed,
            "total_samples": len(mixed_rows),
            "markush_count": len(selected_markush),
            "ocsr_count": len(selected_ocsr),
            "selection_policy": sampling_policy(args),
            "requested_total_samples": args.total_samples,
            "requested_markush_count": args.markush_count,
            "requested_ocsr_count": args.ocsr_count,
            "requested_ocsr_fraction": args.ocsr_fraction if args.ocsr_count is None else None,
            "ocsr_fraction_arg": args.ocsr_fraction,
            "actual_ocsr_fraction": len(selected_ocsr) / len(mixed_rows) if mixed_rows else 0.0,
        },
        "row_counts": row_counts(mixed_rows),
        "validation_ok": validation.get("ok"),
        "missing_images": validation.get("missing_images"),
        "image_validation": {
            "checked": not args.no_check_images,
            "require_absolute_images": args.require_absolute_images,
            "path_resolution": "absolute or relative to the owning JSONL",
        },
        "ocsr_metric_mode_policy": {
            "official_ocsr": {
                "prediction_mode": OCSR_PREDICTION_MODE_NATIVE_SMILES,
                "status": "default benchmark-compatible mode",
            },
            "diagnostic_no_extension_markush": {
                "prediction_mode": OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
                "status": "opt-in diagnostic only",
            },
        },
        "examples": examples,
        "training_status": "prep-only artifact; no training job launched",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0
