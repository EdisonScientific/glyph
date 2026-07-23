#!/usr/bin/env python3
"""Mix unified records into SFT-ready chat JSONL.

Input can be one or more unified JSONL files, a directory containing
``*.jsonl``/``*/unified.jsonl`` files, or a local HuggingFace dataset saved via
``DatasetDict.save_to_disk``.  Eval-only benchmarks are excluded by default.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glyph.markush.data.canonicalize import canonicalize_cxsmiles_checked
from glyph.markush.data.example_modes import (
    DEFAULT_MODE_WEIGHTS,
    MOLECULE_OCSR,
    MOLPARSER_CONVERTED_MARKUSH,
    MOLPARSER_NATIVE_MARKUSH,
    PATENT_IMAGE_MARKUSH,
    STRUCTURE_ONLY_MARKUSH,
    STRUCTURED_TRANSCRIPT_MARKUSH,
    TEXT_CONDITIONED_MARKUSH,
    mode_prompt,
    modes_for_unified_record,
)
from glyph.markush.data.molparser_convert import convert_molparser_esmiles_to_cxsmiles
from glyph.markush.data.molparser_esmiles import parse_molparser_esmiles
from glyph.markush.data.ocsr import (
    RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX,
    canonicalize_ocsr_smiles_checked,
)
from glyph.markush.data.pipeline import standard_to_opt_cxsmiles
from glyph.markush.data.structured_transcripts import (
    SCHEMA_VERSION as STRUCTURED_TRANSCRIPT_SCHEMA_VERSION,
)
from glyph.markush.data.structured_transcripts import (
    structured_transcript_for_markush_target,
    validate_structured_transcript_content,
)
from glyph.markush.eval import common

EVAL_ONLY_SOURCES = {"ip5m", "m2s", "uspto_markush", "uspto-markush", "ip5-markush"}
ExclusionKey = tuple[str, str, str]
MIX_MODE_ORDER = (
    STRUCTURE_ONLY_MARKUSH,
    PATENT_IMAGE_MARKUSH,
    TEXT_CONDITIONED_MARKUSH,
    MOLECULE_OCSR,
    MOLPARSER_NATIVE_MARKUSH,
    MOLPARSER_CONVERTED_MARKUSH,
    STRUCTURED_TRANSCRIPT_MARKUSH,
)


@dataclass(frozen=True)
class TargetSpec:
    """Prepared assistant target plus the target-format contract in the prompt."""

    content: str
    source: str
    format_label: str
    prompt_instruction: str
    canonicalization_status: str = ""
    canonicalization_error: str = ""
    conversion_status: str = ""
    conversion_reason: str = ""
    conversion_detail: str = ""


@dataclass(frozen=True)
class MixWeights:
    """Requested mixed-data example-mode weights."""

    structure_only_markush: float = DEFAULT_MODE_WEIGHTS[STRUCTURE_ONLY_MARKUSH]
    patent_image_markush: float = DEFAULT_MODE_WEIGHTS[PATENT_IMAGE_MARKUSH]
    text_conditioned_markush: float = DEFAULT_MODE_WEIGHTS[TEXT_CONDITIONED_MARKUSH]
    molecule_ocsr: float = DEFAULT_MODE_WEIGHTS[MOLECULE_OCSR]
    molparser_native_markush: float = DEFAULT_MODE_WEIGHTS[MOLPARSER_NATIVE_MARKUSH]
    molparser_converted_markush: float = DEFAULT_MODE_WEIGHTS[MOLPARSER_CONVERTED_MARKUSH]
    structured_transcript_markush: float = DEFAULT_MODE_WEIGHTS[STRUCTURED_TRANSCRIPT_MARKUSH]

    def normalized(self) -> dict[str, float]:
        values = {
            STRUCTURE_ONLY_MARKUSH: self.structure_only_markush,
            PATENT_IMAGE_MARKUSH: self.patent_image_markush,
            TEXT_CONDITIONED_MARKUSH: self.text_conditioned_markush,
            MOLECULE_OCSR: self.molecule_ocsr,
            MOLPARSER_NATIVE_MARKUSH: self.molparser_native_markush,
            MOLPARSER_CONVERTED_MARKUSH: self.molparser_converted_markush,
            STRUCTURED_TRANSCRIPT_MARKUSH: self.structured_transcript_markush,
        }
        total = sum(max(0.0, value) for value in values.values())
        if total <= 0:
            raise ValueError("At least one example-mode weight must be positive")
        return {key: max(0.0, value) / total for key, value in values.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping blank lines."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def discover_jsonl_inputs(paths: list[Path]) -> list[Path]:
    """Expand input files/directories into deterministic JSONL file paths."""

    out: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            out.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        combined = path / "all_unified.jsonl"
        if combined.exists():
            out.append(combined)
            continue
        dataset_unified = path / "unified.jsonl"
        if dataset_unified.exists():
            out.append(dataset_unified)
            continue
        direct = sorted(path.glob("*.jsonl"))
        nested_unified = sorted(path.glob("*/unified.jsonl"))
        out.extend(direct)
        out.extend(p for p in nested_unified if p not in direct)
    return sorted(dict.fromkeys(out))


def load_unified_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Load unified records from JSONL files or saved HF dataset directories."""

    records: list[dict[str, Any]] = []
    jsonl_paths = discover_jsonl_inputs(paths)
    if jsonl_paths:
        for path in jsonl_paths:
            records.extend(load_jsonl(path))
        return records

    # Fall back to HuggingFace save_to_disk layout.
    from datasets import load_from_disk

    for path in paths:
        dataset = load_from_disk(str(path.expanduser()))
        if hasattr(dataset, "items"):
            for split, split_dataset in dataset.items():
                for row in split_dataset:
                    record = dict(row)
                    record.setdefault("split", split)
                    records.append(record)
        else:
            for row in dataset:
                records.append(dict(row))
    return records


def record_exclusion_key(record: dict[str, Any]) -> ExclusionKey:
    """Return the stable key used to exclude audited bad unified rows."""

    return (
        str(record.get("source") or ""),
        str(record.get("split") or ""),
        str(record.get("id") or ""),
    )


def _exclusion_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("exclusion_candidates"), list):
        return _exclusion_items(value["exclusion_candidates"])
    training = value.get("training")
    if isinstance(training, dict) and isinstance(training.get("exclusion_candidates"), list):
        return _exclusion_items(training["exclusion_candidates"])
    return []


def load_exclusion_keys(paths: list[Path]) -> set[ExclusionKey]:
    """Load row exclusion keys from audit report JSON or JSONL files."""

    keys: set[ExclusionKey] = set()
    for path in paths:
        path = path.expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                items = [json.loads(line) for line in handle if line.strip()]
        else:
            items = _exclusion_items(json.loads(path.read_text(encoding="utf-8")))
        for item in items:
            row_id = item.get("id") or item.get("source_id")
            if row_id is None:
                continue
            keys.add(
                (
                    str(item.get("source") or ""),
                    str(item.get("split") or ""),
                    str(row_id),
                )
            )
    return keys


def filter_excluded_records(
    records: list[dict[str, Any]],
    excluded_keys: set[ExclusionKey],
) -> tuple[list[dict[str, Any]], int]:
    if not excluded_keys:
        return records, 0
    filtered = [record for record in records if record_exclusion_key(record) not in excluded_keys]
    return filtered, len(records) - len(filtered)


def is_trainable_source(record: dict[str, Any], *, include_eval: bool = False) -> bool:
    """Return whether a unified row is eligible for training."""

    if include_eval:
        return True
    source = str(record.get("source") or "").lower()
    split = str(record.get("split") or "").lower()
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if source in EVAL_ONLY_SOURCES or metadata.get("eval_only") is True:
        return False
    return split in {"", "train", "training"}


def _format_label_for_cxsmiles(
    record: dict[str, Any],
    *,
    source: str,
    cxsmiles: str,
) -> str:
    override = record.get("target_cxsmiles_format") or record.get("target_format")
    if isinstance(override, str) and override.strip():
        return override.strip()
    if source == "cxsmiles_opt" or source.endswith("->cxsmiles_opt"):
        return "cxsmiles_opt"
    if source == "annotation.cxsmi" and "<r>" in cxsmiles.lower():
        return "cxsmiles_opt"
    if "<r>" in cxsmiles.lower():
        return "cxsmiles_opt"
    return "CXSMILES"


def _markush_prompt_instruction(format_label: str) -> str:
    return (
        "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        f"the <cxsmi> value must use {format_label}."
    )


def _structured_transcript_prompt_instruction(final_format_label: str) -> str:
    return (
        "Target format: first return one compact JSON object using schema "
        f"{STRUCTURED_TRANSCRIPT_SCHEMA_VERSION}, then return "
        "<markush><cxsmi>...</cxsmi><stable>...</stable></markush>; the final <cxsmi> "
        f"value must use {final_format_label}. Do not include explanations."
    )


def _molparser_native_prompt_instruction() -> str:
    return "Target format: return MolParser E-SMILES (`SMILES<sep>EXTENSION`)."


def _cxsmiles_as_opt(cxsmiles: str) -> str:
    if "<r>" in cxsmiles.lower():
        return cxsmiles
    return standard_to_opt_cxsmiles(cxsmiles)


def _normalize_markush_target_to_opt(cxsmiles: str, *, source: str) -> tuple[str, str] | None:
    """Return a graph-verified cxsmiles_opt target and source label."""

    try:
        opt_cxsmiles = _cxsmiles_as_opt(cxsmiles)
    except Exception:
        return None
    if not opt_cxsmiles:
        return None
    should_validate = source != "cxsmiles_opt" and "<r>" not in cxsmiles.lower()
    if should_validate or opt_cxsmiles != cxsmiles:
        try:
            from glyph.markush.eval.markush_graph import markush_graph_equal

            result = markush_graph_equal(cxsmiles, opt_cxsmiles)
        except Exception:
            return None
        if not result.equal:
            return None
    if source == "cxsmiles_opt" or "<r>" in cxsmiles.lower():
        return opt_cxsmiles, source
    return opt_cxsmiles, f"{source}->cxsmiles_opt"


def _canonicalize_markush_target(
    cxsmiles: str,
    *,
    source: str,
    require_canonical: bool,
) -> tuple[str, str, str, str, str] | None:
    normalized = _normalize_markush_target_to_opt(cxsmiles, source=source)
    if normalized is None:
        return None
    opt_cxsmiles, opt_source = normalized

    result = canonicalize_cxsmiles_checked(opt_cxsmiles)
    if result.value is None or result.fallback:
        if require_canonical:
            return None
        return (
            opt_cxsmiles,
            opt_source,
            "cxsmiles_opt",
            result.status,
            result.error or "",
        )
    return (
        result.value,
        f"{opt_source}->canonicalized_cxsmiles_opt",
        "canonicalized cxsmiles_opt",
        result.status,
        "",
    )


def _molparser_esmiles_value(record: dict[str, Any]) -> str:
    value = record.get("molparser_esmiles")
    if isinstance(value, str) and value.strip():
        return value
    value = record.get("smiles")
    return value if isinstance(value, str) else ""


def _native_molparser_target(record: dict[str, Any]) -> TargetSpec | None:
    value = _molparser_esmiles_value(record)
    if not value.strip():
        return None
    parsed = parse_molparser_esmiles(value)
    if not parsed.valid:
        return None
    return TargetSpec(
        content=value,
        source="molparser_esmiles",
        format_label="MolParser E-SMILES",
        prompt_instruction=_molparser_native_prompt_instruction(),
        conversion_status="native_molparser_esmiles",
    )


def _converted_molparser_target(
    record: dict[str, Any],
    *,
    canonicalize_markush: bool,
    require_canonical: bool,
) -> TargetSpec | None:
    value = _molparser_esmiles_value(record)
    if not value.strip():
        return None
    conversion = convert_molparser_esmiles_to_cxsmiles(value)
    if not conversion.converted:
        return None
    cxsmiles = conversion.cxsmiles_opt
    source = "molparser_esmiles->cxsmiles_opt"
    format_label = conversion.target_cxsmiles_format
    canonicalization_status = ""
    canonicalization_error = ""
    if canonicalize_markush:
        canonicalized = _canonicalize_markush_target(
            cxsmiles,
            source="cxsmiles_opt",
            require_canonical=require_canonical,
        )
        if canonicalized is None:
            return None
        (
            cxsmiles,
            canonical_source,
            format_label,
            canonicalization_status,
            canonicalization_error,
        ) = canonicalized
        source = f"molparser_esmiles->{canonical_source}"
    return TargetSpec(
        content=f"<markush><cxsmi>{cxsmiles}</cxsmi><stable></stable></markush>",
        source=source,
        format_label=format_label,
        prompt_instruction=_markush_prompt_instruction(format_label),
        canonicalization_status=canonicalization_status,
        canonicalization_error=canonicalization_error,
        conversion_status=conversion.status,
        conversion_reason=conversion.reason,
        conversion_detail=conversion.detail,
    )


def _structured_transcript_target(record: dict[str, Any]) -> TargetSpec | None:
    base = target_spec_for_record(
        record,
        example_mode=None,
        canonicalize_markush=True,
        require_canonical=True,
    )
    if base is None or base.format_label != "canonicalized cxsmiles_opt":
        return None

    result = structured_transcript_for_markush_target(
        base.content,
        target_format=base.format_label,
    )
    if not result.valid:
        return None
    ok, _error = validate_structured_transcript_content(result.content)
    if not ok:
        return None
    final_cxsmiles, _stable = common.extract_cxsmiles_and_stable(
        base.content,
        require_wrapper=True,
    )
    if common.prediction_to_cxsmiles_opt(result.content) != final_cxsmiles:
        return None

    return TargetSpec(
        content=result.content,
        source=f"{base.source}->structured_transcript",
        format_label="structured transcript + final canonicalized cxsmiles_opt",
        prompt_instruction=_structured_transcript_prompt_instruction(base.format_label),
        canonicalization_status=base.canonicalization_status,
        canonicalization_error=base.canonicalization_error,
        conversion_status=base.conversion_status,
        conversion_reason=base.conversion_reason,
        conversion_detail=base.conversion_detail,
    )


def target_spec_for_record(
    record: dict[str, Any],
    *,
    example_mode: str | None = None,
    canonicalize_markush: bool = False,
    require_canonical: bool = False,
) -> TargetSpec | None:
    """Return the prepared target and target-format contract for one row."""

    task = str(record.get("task") or "markush").lower()
    if example_mode == STRUCTURED_TRANSCRIPT_MARKUSH:
        if task != "markush":
            return None
        return _structured_transcript_target(record)

    if task == "ocsr":
        smiles = record.get("smiles")
        result = canonicalize_ocsr_smiles_checked(smiles if isinstance(smiles, str) else None)
        if result.value is None:
            return None
        return TargetSpec(
            content=result.value,
            source=f"smiles{RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX}",
            format_label="SMILES",
            prompt_instruction="Target format: return a SMILES string.",
            canonicalization_status=result.status,
            canonicalization_error=result.error,
        )

    assistant_target = common.sft_assistant_target(record)
    if assistant_target:
        cxsmiles, stable = common.extract_cxsmiles_and_stable(
            assistant_target,
            require_wrapper=False,
        )
        if not cxsmiles:
            return None
        source = str(record.get("target_source") or "messages.assistant")
        canonicalization_status = str(record.get("target_canonicalization_status") or "")
        canonicalization_error = str(record.get("target_canonicalization_error") or "")
        conversion_status = record.get("target_conversion_status")
        conversion_reason = record.get("target_conversion_reason")
        conversion_detail = record.get("target_conversion_detail")
        if canonicalize_markush:
            canonicalized = _canonicalize_markush_target(
                cxsmiles,
                source=source,
                require_canonical=require_canonical,
            )
            if canonicalized is None:
                return None
            (
                cxsmiles,
                source,
                format_label,
                canonicalization_status,
                canonicalization_error,
            ) = canonicalized
            content = f"<markush><cxsmi>{cxsmiles}</cxsmi><stable>{stable}</stable></markush>"
        else:
            content = assistant_target
            format_label = _format_label_for_cxsmiles(record, source=source, cxsmiles=cxsmiles)
        return TargetSpec(
            content=content,
            source=source,
            format_label=format_label,
            prompt_instruction=_markush_prompt_instruction(format_label),
            canonicalization_status=canonicalization_status,
            canonicalization_error=canonicalization_error,
            conversion_status=str(conversion_status) if conversion_status is not None else "",
            conversion_reason=str(conversion_reason) if conversion_reason is not None else "",
            conversion_detail=conversion_detail,
        )

    is_molparser = str(record.get("source") or "").lower() == "molparser"
    if is_molparser:
        if example_mode == MOLPARSER_NATIVE_MARKUSH:
            return _native_molparser_target(record)
        if example_mode not in {None, MOLPARSER_CONVERTED_MARKUSH}:
            return None
        return _converted_molparser_target(
            record,
            canonicalize_markush=canonicalize_markush,
            require_canonical=require_canonical,
        )

    annotation = record.get("annotation")
    annotation_cxsmi = ""
    stable = ""
    if isinstance(annotation, str) and annotation.strip():
        annotation_cxsmi, stable = common.extract_cxsmiles_and_stable(
            annotation,
            require_wrapper=False,
        )
    candidates = [
        ("cxsmiles_opt", record.get("cxsmiles_opt")),
        ("annotation.cxsmi", annotation_cxsmi),
        ("cxsmiles", record.get("cxsmiles")),
        ("cxsmiles_standard", record.get("cxsmiles_standard")),
    ]
    source = ""
    cxsmiles = ""
    for candidate_source, value in candidates:
        if isinstance(value, str) and value.strip():
            source = candidate_source
            cxsmiles = value
            break
    if not source or not cxsmiles:
        return None
    canonicalization_status = ""
    canonicalization_error = ""
    if canonicalize_markush:
        canonicalized = _canonicalize_markush_target(
            cxsmiles,
            source=source,
            require_canonical=require_canonical,
        )
        if canonicalized is None:
            return None
        cxsmiles, source, format_label, canonicalization_status, canonicalization_error = (
            canonicalized
        )
    else:
        normalized = _normalize_markush_target_to_opt(cxsmiles, source=source)
        if normalized is None:
            return None
        cxsmiles, source = normalized
        format_label = _format_label_for_cxsmiles(record, source=source, cxsmiles=cxsmiles)
    return TargetSpec(
        content=f"<markush><cxsmi>{cxsmiles}</cxsmi><stable>{stable}</stable></markush>",
        source=source,
        format_label=format_label,
        prompt_instruction=_markush_prompt_instruction(format_label),
        canonicalization_status=canonicalization_status,
        canonicalization_error=canonicalization_error,
    )


def prompt_with_target_instruction(prompt: str, target: TargetSpec) -> str:
    """Append the target-format instruction unless the prompt already has one."""

    prompt = str(prompt or "").strip()
    if "Target format:" in prompt:
        return prompt
    if not prompt:
        return target.prompt_instruction
    return f"{prompt}\n{target.prompt_instruction}"


def target_cxsmiles_for_record(
    record: dict[str, Any],
    *,
    canonicalize_markush: bool = False,
    require_canonical: bool = False,
) -> str | None:
    """Return the bare evaluator target from the shared target contract."""

    target = target_spec_for_record(
        record,
        canonicalize_markush=canonicalize_markush,
        require_canonical=require_canonical,
    )
    if target is None:
        return None
    if str(record.get("task") or "markush").lower() == "ocsr":
        return target.content
    cxsmiles, _stable = common.extract_cxsmiles_and_stable(
        target.content,
        require_wrapper=False,
    )
    return cxsmiles or None


def _image_for_record(record: dict[str, Any]) -> str | None:
    sft_image = common.sft_image_path(record)
    if sft_image:
        return sft_image
    for key in ("image", "image_path", "local_image_path", "page_path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict) and value.get("path"):
            return str(value["path"])
    return None


def eligible_modes(
    record: dict[str, Any],
    *,
    canonicalize_markush: bool = False,
    require_canonical: bool = False,
) -> set[str]:
    """Return mixed-data example modes a unified row can produce."""

    task = str(record.get("task") or "markush").lower()
    if task == "ocsr":
        return (
            {MOLECULE_OCSR}
            if _image_for_record(record)
            and target_spec_for_record(
                record,
                example_mode=MOLECULE_OCSR,
                canonicalize_markush=canonicalize_markush,
                require_canonical=require_canonical,
            )
            else set()
        )

    if not _image_for_record(record):
        return set()

    modes = modes_for_unified_record(record)
    return {
        mode
        for mode in modes
        if target_spec_for_record(
            record,
            example_mode=mode,
            canonicalize_markush=canonicalize_markush,
            require_canonical=require_canonical,
        )
    }


def _eligible_modes_for_mix(
    record: dict[str, Any],
    *,
    canonicalize_markush: bool,
    require_canonical: bool,
    include_eval: bool,
    wants_structured_transcript: bool,
) -> tuple[str, ...]:
    if not is_trainable_source(record, include_eval=include_eval):
        return ()
    modes = eligible_modes(
        record,
        canonicalize_markush=canonicalize_markush,
        require_canonical=require_canonical,
    )
    if wants_structured_transcript and target_spec_for_record(
        record,
        example_mode=STRUCTURED_TRANSCRIPT_MARKUSH,
    ):
        modes.add(STRUCTURED_TRANSCRIPT_MARKUSH)
    return tuple(mode for mode in MIX_MODE_ORDER if mode in modes)


def _eligible_modes_worker(args: tuple[int, dict[str, Any], bool, bool, bool, bool]):
    idx, record, canonicalize_markush, require_canonical, include_eval, wants_structured = args
    return idx, _eligible_modes_for_mix(
        record,
        canonicalize_markush=canonicalize_markush,
        require_canonical=require_canonical,
        include_eval=include_eval,
        wants_structured_transcript=wants_structured,
    )


def build_training_example(
    record: dict[str, Any],
    mode: str,
    *,
    text_dropout_rate: float = 0.5,
    canonicalize_markush: bool = False,
    require_canonical: bool = False,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Convert one unified row to the chat JSONL shape used by SFT."""

    rng = random.Random() if rng is None else rng
    target = target_spec_for_record(
        record,
        example_mode=mode,
        canonicalize_markush=canonicalize_markush,
        require_canonical=require_canonical,
    )
    image = _image_for_record(record)
    if not target or not image:
        raise ValueError(f"Record {record.get('id', '<unknown>')} lacks target or image")

    effective_mode = mode
    if mode == TEXT_CONDITIONED_MARKUSH and rng.random() < text_dropout_rate:
        effective_mode = STRUCTURE_ONLY_MARKUSH

    user = "<image>\n" + prompt_with_target_instruction(
        mode_prompt(effective_mode, description=record.get("description")),
        target,
    )

    source_id = str(record.get("id", ""))
    task = str(record.get("task") or ("ocsr" if mode == MOLECULE_OCSR else "markush")).lower()
    return {
        "id": f"{source_id}_{mode}",
        "task": task,
        "mode": mode,
        "effective_mode": effective_mode,
        "source": record.get("source"),
        "split": record.get("split") or "train",
        "benchmark": record.get("benchmark"),
        "source_id": source_id,
        "target_source": target.source,
        "target_format": target.format_label,
        "target_canonicalization_status": target.canonicalization_status,
        "target_canonicalization_error": target.canonicalization_error,
        "target_conversion_status": target.conversion_status,
        "target_conversion_reason": target.conversion_reason,
        "target_conversion_detail": target.conversion_detail,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": target.content},
        ],
        "images": [image],
    }


def _build_training_example_worker(
    args: tuple[dict[str, Any], str, float, bool, bool, int],
) -> dict[str, Any]:
    record, mode, text_dropout_rate, canonicalize_markush, require_canonical, rng_seed = args
    return build_training_example(
        record,
        mode,
        text_dropout_rate=text_dropout_rate,
        canonicalize_markush=canonicalize_markush,
        require_canonical=require_canonical,
        rng=random.Random(rng_seed),
    )


def _process_pool_chunksize(n_items: int, workers: int) -> int:
    if workers <= 1 or n_items <= 0:
        return 1
    return max(1, min(512, n_items // max(1, workers * 8)))


def _collect_eligible_modes(
    records: list[dict[str, Any]],
    *,
    canonicalize_markush_targets: bool,
    require_canonical_markush_targets: bool,
    include_eval: bool,
    wants_structured_transcript: bool,
    workers: int,
) -> list[tuple[int, tuple[str, ...]]]:
    if workers <= 1:
        return [
            (
                idx,
                _eligible_modes_for_mix(
                    record,
                    canonicalize_markush=canonicalize_markush_targets,
                    require_canonical=require_canonical_markush_targets,
                    include_eval=include_eval,
                    wants_structured_transcript=wants_structured_transcript,
                ),
            )
            for idx, record in enumerate(records)
        ]
    tasks = [
        (
            idx,
            record,
            canonicalize_markush_targets,
            require_canonical_markush_targets,
            include_eval,
            wants_structured_transcript,
        )
        for idx, record in enumerate(records)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                _eligible_modes_worker,
                tasks,
                chunksize=_process_pool_chunksize(len(tasks), workers),
            )
        )


def _build_training_examples(
    selected: list[tuple[dict[str, Any], str]],
    *,
    text_dropout_rate: float,
    canonicalize_markush_targets: bool,
    require_canonical_markush_targets: bool,
    rng: random.Random,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [
            build_training_example(
                record,
                mode,
                text_dropout_rate=text_dropout_rate,
                canonicalize_markush=canonicalize_markush_targets,
                require_canonical=require_canonical_markush_targets,
                rng=rng,
            )
            for record, mode in selected
        ]
    tasks = [
        (
            record,
            mode,
            text_dropout_rate,
            canonicalize_markush_targets,
            require_canonical_markush_targets,
            rng.randrange(2**63),
        )
        for record, mode in selected
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                _build_training_example_worker,
                tasks,
                chunksize=_process_pool_chunksize(len(tasks), workers),
            )
        )


def _target_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in fractions.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    by_fractional = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in by_fractional[:remainder]:
        counts[key] += 1
    return counts


def mix_examples(
    records: list[dict[str, Any]],
    *,
    weights: MixWeights = MixWeights(),
    total_samples: int | None = None,
    text_dropout_rate: float = 0.5,
    seed: int = 42,
    include_eval: bool = False,
    oversample: bool = True,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a mixed list of SFT examples and return stats."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {
        STRUCTURE_ONLY_MARKUSH: [],
        PATENT_IMAGE_MARKUSH: [],
        TEXT_CONDITIONED_MARKUSH: [],
        MOLECULE_OCSR: [],
        MOLPARSER_NATIVE_MARKUSH: [],
        MOLPARSER_CONVERTED_MARKUSH: [],
        STRUCTURED_TRANSCRIPT_MARKUSH: [],
    }
    normalized_weights = weights.normalized()
    skipped = 0
    wants_structured_transcript = normalized_weights.get(STRUCTURED_TRANSCRIPT_MARKUSH, 0.0) > 0
    eligibility = _collect_eligible_modes(
        records,
        canonicalize_markush_targets=canonicalize_markush_targets,
        require_canonical_markush_targets=require_canonical_markush_targets,
        include_eval=include_eval,
        wants_structured_transcript=wants_structured_transcript,
        workers=workers,
    )
    for idx, modes in eligibility:
        record = records[idx]
        if not modes:
            skipped += 1
            continue
        for mode in modes:
            pools[mode].append(record)

    opt_in_modes = {
        MOLPARSER_NATIVE_MARKUSH,
        MOLPARSER_CONVERTED_MARKUSH,
        STRUCTURED_TRANSCRIPT_MARKUSH,
    }

    def _mode_active(mode: str) -> bool:
        # Auxiliary modes are opt-in: their default weight is 0, so they
        # participate only when explicitly weighted, in both the capped and
        # uncapped paths. Standard modes always participate.
        return mode not in opt_in_modes or normalized_weights.get(mode, 0.0) > 0

    available_once = sum(len(pool) for mode, pool in pools.items() if _mode_active(mode))
    if available_once <= 0:
        raise ValueError("No eligible training examples found")
    if total_samples is not None and total_samples <= 0:
        raise ValueError("total_samples must be positive")

    examples: list[dict[str, Any]] = []
    shortages: dict[str, int] = {}
    selected_examples: list[tuple[dict[str, Any], str]] = []
    if total_samples is None:
        targets = {mode: len(pool) for mode, pool in pools.items() if _mode_active(mode)}
        for mode, pool in pools.items():
            if not _mode_active(mode):
                continue
            for record in pool:
                selected_examples.append((record, mode))
        examples = _build_training_examples(
            selected_examples,
            text_dropout_rate=text_dropout_rate,
            canonicalize_markush_targets=canonicalize_markush_targets,
            require_canonical_markush_targets=require_canonical_markush_targets,
            rng=rng,
            workers=workers,
        )
        rng.shuffle(examples)
        return examples, _mix_stats(
            records=records,
            skipped=skipped,
            pools=pools,
            targets=targets,
            examples=examples,
            shortages=shortages,
            include_eval=include_eval,
            oversample=oversample,
            canonicalize_markush_targets=canonicalize_markush_targets,
            require_canonical_markush_targets=require_canonical_markush_targets,
        )

    targets = _target_counts(total_samples, weights.normalized())
    for mode, target_count in targets.items():
        if target_count <= 0:
            continue
        pool = pools[mode]
        if not pool:
            shortages[mode] = target_count
            continue
        if target_count <= len(pool):
            selected = rng.sample(pool, target_count)
        elif oversample:
            selected = [rng.choice(pool) for _ in range(target_count)]
            shortages[mode] = target_count - len(pool)
        else:
            selected = list(pool)
            shortages[mode] = target_count - len(pool)
        for record in selected:
            selected_examples.append((record, mode))

    examples = _build_training_examples(
        selected_examples,
        text_dropout_rate=text_dropout_rate,
        canonicalize_markush_targets=canonicalize_markush_targets,
        require_canonical_markush_targets=require_canonical_markush_targets,
        rng=rng,
        workers=workers,
    )
    rng.shuffle(examples)
    return examples, _mix_stats(
        records=records,
        skipped=skipped,
        pools=pools,
        targets=targets,
        examples=examples,
        shortages=shortages,
        include_eval=include_eval,
        oversample=oversample,
        canonicalize_markush_targets=canonicalize_markush_targets,
        require_canonical_markush_targets=require_canonical_markush_targets,
    )


def _mix_stats(
    *,
    records: list[dict[str, Any]],
    skipped: int,
    pools: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    examples: list[dict[str, Any]],
    shortages: dict[str, int],
    include_eval: bool,
    oversample: bool,
    canonicalize_markush_targets: bool,
    require_canonical_markush_targets: bool,
) -> dict[str, Any]:
    """Return structured stats for a prepared SFT mix."""

    return {
        "input_records": len(records),
        "skipped_records": skipped,
        "pool_sizes": {key: len(value) for key, value in pools.items()},
        "requested_counts": targets,
        "written": len(examples),
        "shortages": shortages,
        "mode_counts": dict(Counter(example["mode"] for example in examples)),
        "effective_mode_counts": dict(Counter(example["effective_mode"] for example in examples)),
        "target_source_counts": dict(Counter(example["target_source"] for example in examples)),
        "target_format_counts": dict(Counter(example["target_format"] for example in examples)),
        "target_canonicalization_status_counts": dict(
            Counter(
                example["target_canonicalization_status"]
                for example in examples
                if example["target_canonicalization_status"]
            )
        ),
        "target_conversion_status_counts": dict(
            Counter(
                example["target_conversion_status"]
                for example in examples
                if example["target_conversion_status"]
            )
        ),
        "include_eval": include_eval,
        "oversample": oversample,
        "canonicalize_markush_targets": canonicalize_markush_targets,
        "require_canonical_markush_targets": require_canonical_markush_targets,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare mixed SFT JSONL from unified dataset records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--total-samples", type=int, default=None)
    parser.add_argument("--text-dropout", type=float, default=0.5)
    parser.add_argument(
        "--structure-only-markush-fraction",
        type=float,
        default=DEFAULT_MODE_WEIGHTS[STRUCTURE_ONLY_MARKUSH],
    )
    parser.add_argument(
        "--patent-image-markush-fraction",
        type=float,
        default=DEFAULT_MODE_WEIGHTS[PATENT_IMAGE_MARKUSH],
    )
    parser.add_argument(
        "--text-conditioned-markush-fraction",
        type=float,
        default=DEFAULT_MODE_WEIGHTS[TEXT_CONDITIONED_MARKUSH],
    )
    parser.add_argument(
        "--molecule-ocsr-fraction",
        type=float,
        default=DEFAULT_MODE_WEIGHTS[MOLECULE_OCSR],
    )
    parser.add_argument(
        "--molparser-converted-markush-fraction",
        type=float,
        default=DEFAULT_MODE_WEIGHTS[MOLPARSER_CONVERTED_MARKUSH],
        help="Opt-in fraction for MolParser E-SMILES rows validated as converted cxsmiles_opt.",
    )
    parser.add_argument(
        "--molparser-native-markush-fraction",
        type=float,
        default=DEFAULT_MODE_WEIGHTS[MOLPARSER_NATIVE_MARKUSH],
        help="Opt-in fraction for valid MolParser-Markush rows trained as native E-SMILES.",
    )
    parser.add_argument(
        "--structured-transcript-markush-fraction",
        type=float,
        default=DEFAULT_MODE_WEIGHTS[STRUCTURED_TRANSCRIPT_MARKUSH],
        help=(
            "Opt-in fraction for deterministic transcript-plus-final canonicalized "
            "cxsmiles_opt Markush rows."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument(
        "--exclude-file",
        type=Path,
        action="append",
        default=[],
        help="Audit report/exclusion JSON or JSONL with rows to omit from mixes",
    )
    parser.add_argument(
        "--canonicalize-markush-targets",
        action="store_true",
        help="Prefer verified canonicalized cxsmiles_opt targets for Markush examples",
    )
    parser.add_argument(
        "--require-canonical-markush-targets",
        action="store_true",
        help="Skip Markush rows whose canonicalized target cannot be verified",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process workers for target validation/transcript generation during mixing",
    )
    return parser


def prepare_sft_jsonl(
    *,
    datasets: list[Path],
    output: Path,
    stats_json: Path | None = None,
    total_samples: int | None = None,
    text_dropout: float = 0.5,
    weights: MixWeights = MixWeights(),
    seed: int = 42,
    include_eval: bool = False,
    oversample: bool = True,
    exclude_files: list[Path] | None = None,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Load unified records, write mixed SFT JSONL, and return stats."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    records = load_unified_records(datasets)
    raw_record_count = len(records)
    excluded_keys = load_exclusion_keys(exclude_files or [])
    records, excluded_count = filter_excluded_records(records, excluded_keys)
    examples, stats = mix_examples(
        records,
        weights=weights,
        total_samples=total_samples,
        text_dropout_rate=text_dropout,
        seed=seed,
        include_eval=include_eval,
        oversample=oversample,
        canonicalize_markush_targets=canonicalize_markush_targets,
        require_canonical_markush_targets=require_canonical_markush_targets,
        workers=workers,
    )
    stats["loaded_records"] = raw_record_count
    stats["excluded_records"] = excluded_count
    stats["exclude_keys"] = len(excluded_keys)
    stats["workers"] = workers
    write_jsonl(examples, output)
    if stats_json:
        stats_json.parent.mkdir(parents=True, exist_ok=True)
        stats_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    args = build_parser().parse_args()
    stats = prepare_sft_jsonl(
        datasets=args.dataset,
        output=args.output,
        stats_json=args.stats_json,
        total_samples=args.total_samples,
        text_dropout=args.text_dropout,
        weights=MixWeights(
            structure_only_markush=args.structure_only_markush_fraction,
            patent_image_markush=args.patent_image_markush_fraction,
            text_conditioned_markush=args.text_conditioned_markush_fraction,
            molecule_ocsr=args.molecule_ocsr_fraction,
            molparser_native_markush=args.molparser_native_markush_fraction,
            molparser_converted_markush=args.molparser_converted_markush_fraction,
            structured_transcript_markush=args.structured_transcript_markush_fraction,
        ),
        seed=args.seed,
        include_eval=args.include_eval,
        oversample=not args.no_oversample,
        exclude_files=args.exclude_file,
        canonicalize_markush_targets=args.canonicalize_markush_targets,
        require_canonical_markush_targets=args.require_canonical_markush_targets,
        workers=args.workers,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
