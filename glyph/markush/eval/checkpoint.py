#!/usr/bin/env python3
"""Offline evaluation of a LoRA checkpoint on the IP5-M benchmark.

Loads a base Qwen-VL model + LoRA adapter, runs greedy generation on each
eval sample, and scores predictions with the MCS-based Markush evaluator
(``glyph.markush.eval.metrics_mcs``).  Produces a JSON report with aggregate,
per-category, and per-sample metrics, plus a summary table on stdout.

Example
-------
python -m glyph.markush.eval.checkpoint \
    --checkpoint /path/to/checkpoint-4000 \
    --base-model /path/to/Qwen3.5-2B-Base \
    --eval-data /path/to/eval_ip5m.jsonl \
    --data-root /path/to/data \
    --n-samples 100 \
    --output results/eval_step_4000.json \
    --device cuda:0 \
    --protocol both
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from glyph.markush.data.example_modes import STRUCTURE_ONLY_MARKUSH_PROMPT
from glyph.markush.data.ocsr import row_declares_ocsr_smiles
from glyph.markush.data.prepare_sft import (
    prompt_with_target_instruction,
    target_cxsmiles_for_record,
    target_spec_for_record,
)
from glyph.markush.eval import common
from glyph.markush.eval.benchmarks import (
    benchmark_choices,
    get_benchmark_spec,
    normalize_benchmark_name,
    resolve_benchmark_path,
)
from glyph.markush.eval.metrics_mcs import (
    evaluate_markush_mcs,
)

LOGGER = logging.getLogger("glyph.markush.eval.checkpoint")

# Token ids shared with sft.py
ENDOFTEXT_TOKEN_ID = 248044
IM_END_TOKEN_ID = 248046

DEFAULT_PROMPT = STRUCTURE_ONLY_MARKUSH_PROMPT
IMAGE_FIELDS = ("image_path", "image", "path", "png_path", "rendered_path", "file_name")


def _processor_image_patch_size(processor: Any) -> int:
    """Return the processor image patch size used by qwen_vl_utils."""

    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is None:
        return 14
    try:
        return int(patch_size)
    except (TypeError, ValueError):
        LOGGER.warning("Unexpected processor patch_size=%r; falling back to 14", patch_size)
        return 14


def _size_value(size: Any, key: str) -> Any:
    if size is None:
        return None
    if isinstance(size, dict):
        return size.get(key)
    return getattr(size, key, None)


def _int_pixel_bound(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


def _processor_image_pixel_kwargs(processor: Any) -> dict[str, int]:
    """Return min/max pixel kwargs for qwen_vl_utils image loading."""

    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    min_pixels = _int_pixel_bound(getattr(image_processor, "min_pixels", None))
    max_pixels = _int_pixel_bound(getattr(image_processor, "max_pixels", None))
    if min_pixels is None:
        min_pixels = _int_pixel_bound(_size_value(size, "shortest_edge"))
    if max_pixels is None:
        max_pixels = _int_pixel_bound(_size_value(size, "longest_edge"))

    kwargs: dict[str, int] = {}
    if min_pixels is not None:
        kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        kwargs["max_pixels"] = max_pixels
    return kwargs


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalRecord:
    """One evaluation sample ready for inference."""

    sample_id: str
    image_path: str
    cxsmiles_opt: str
    annotation: str
    prompt: str
    # category flags (derived from cxsmiles_opt)
    has_r: bool = False
    has_sg: bool = False
    has_m: bool = False


def _resolve_image_path(row: dict[str, Any], data_root: Path | None) -> str | None:
    """Find the first image field and resolve it against data_root.

    Tries three resolutions in order:
      1. ``data_root / relative_path`` (preserves directory structure)
      2. ``data_root / basename`` (flat directory fallback)
      3. The path as-is (already absolute or CWD-relative)
    """
    candidates: list[Any] = []
    sft_image = common.sft_image_path(row)
    if sft_image:
        candidates.append(sft_image)
    for fld in IMAGE_FIELDS:
        value = row.get(fld)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    for value in candidates:
        candidate = Path(value).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        if data_root is not None:
            # Try full relative path first
            full = data_root / candidate
            if full.exists():
                return str(full)
            # Flat-directory fallback: just the basename
            flat = data_root / candidate.name
            if flat.exists():
                return str(flat)
            # Return full path even if it doesn't exist (will be caught later)
            return str(full)
        return str(candidate)
    return None


def _categorise(cxsmiles_opt: str) -> tuple[bool, bool, bool]:
    """Return (has_r, has_sg, has_m) flags for a cxsmiles_opt string."""
    has_r = "<r>" in cxsmiles_opt.lower()
    ext = cxsmiles_opt.split("|", 1)[1] if "|" in cxsmiles_opt else ""
    has_sg = "Sg:" in ext
    has_m = bool(common.RE_M_SECTION.search(ext))
    return has_r, has_sg, has_m


def _row_prompt(
    row: dict[str, Any],
    default_prompt: str,
    *,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
) -> str:
    sft_prompt = common.sft_user_prompt(row)
    if sft_prompt:
        return sft_prompt
    base_prompt = str(row.get("prompt") or default_prompt)
    target = target_spec_for_record(
        row,
        canonicalize_markush=canonicalize_markush_targets,
        require_canonical=require_canonical_markush_targets,
    )
    if target is None:
        return base_prompt
    return prompt_with_target_instruction(base_prompt, target)


def load_eval_data(
    path: str | Path,
    data_root: str | Path | None,
    *,
    limit: int | None = None,
    default_prompt: str = DEFAULT_PROMPT,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
) -> list[EvalRecord]:
    """Load evaluation JSONL, resolve image paths, and classify categories."""
    records: list[EvalRecord] = []
    data_root_path = Path(data_root).expanduser() if data_root else None

    with Path(path).expanduser().open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("%s:%d: invalid JSON — %s", path, line_no, exc)
                continue
            if row_declares_ocsr_smiles(row):
                LOGGER.debug("Skipping OCSR SMILES row %d in Markush eval loader", line_no)
                continue

            sample_id = str(row.get("id", row.get("sample_id", line_no)))
            target = target_spec_for_record(
                row,
                canonicalize_markush=canonicalize_markush_targets,
                require_canonical=require_canonical_markush_targets,
            )
            if (
                canonicalize_markush_targets
                and require_canonical_markush_targets
                and target is None
            ):
                LOGGER.debug("Skipping row %d without verified canonical target", line_no)
                continue
            cxsmiles_opt = (
                target_cxsmiles_for_record(
                    row,
                    canonicalize_markush=canonicalize_markush_targets,
                    require_canonical=require_canonical_markush_targets,
                )
                or ""
            )
            annotation = target.content if target is not None else str(row.get("annotation", ""))
            img_path = _resolve_image_path(row, data_root_path)

            if not cxsmiles_opt and not annotation:
                LOGGER.debug("Skipping row %d without ground truth", line_no)
                continue
            if img_path is None:
                LOGGER.warning("Row %s has no image field; skipping", sample_id)
                continue

            has_r, has_sg, has_m = _categorise(cxsmiles_opt or annotation)
            records.append(
                EvalRecord(
                    sample_id=sample_id,
                    image_path=img_path,
                    cxsmiles_opt=cxsmiles_opt,
                    annotation=annotation,
                    prompt=_row_prompt(
                        row,
                        default_prompt,
                        canonicalize_markush_targets=canonicalize_markush_targets,
                        require_canonical_markush_targets=require_canonical_markush_targets,
                    ),
                    has_r=has_r,
                    has_sg=has_sg,
                    has_m=has_m,
                )
            )
            if limit is not None and len(records) >= limit:
                break

    return records


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_processor(
    base_model: str,
    checkpoint: str | None,
    device: str,
    *,
    dtype: str = "bfloat16",
    merge_adapter: bool = True,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = 1_048_576,
) -> tuple[Any, Any]:
    """Load a full checkpoint or the Qwen-VL base model plus optional LoRA adapter.

    Security: ``base_model``/``checkpoint`` are loaded via
    ``transformers.from_pretrained(..., trust_remote_code=True)`` and may execute
    Python code shipped in the referenced repository (and unpickle torch weights).
    Only pass checkpoints/models from sources you trust.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = torch_dtype_map.get(dtype, "auto")

    # --- Processor ---
    processor_path = checkpoint if checkpoint else base_model
    processor_kwargs: dict[str, Any] = {}
    if image_min_pixels is not None:
        processor_kwargs["min_pixels"] = image_min_pixels
    if image_max_pixels is not None:
        processor_kwargs["max_pixels"] = image_max_pixels
    try:
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            **processor_kwargs,
        )
    except Exception:
        LOGGER.info(
            "Could not load processor from %s; falling back to %s", processor_path, base_model
        )
        processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            **processor_kwargs,
        )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if image_min_pixels is not None and hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = image_min_pixels
        if image_max_pixels is not None and hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = image_max_pixels
    if processor_kwargs:
        LOGGER.info("Configured processor image pixels: %s", processor_kwargs)

    # Chat-template fix: copy from tokenizer if processor lacks one
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(processor, "chat_template", None) is None and getattr(
        tokenizer, "chat_template", None
    ):
        processor.chat_template = tokenizer.chat_template

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    checkpoint_is_adapter = bool(checkpoint) and (Path(checkpoint) / "adapter_config.json").exists()
    model_path = base_model
    if checkpoint and not checkpoint_is_adapter:
        model_path = checkpoint

    # --- Model ---
    LOGGER.info("Loading model from %s", model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if checkpoint and checkpoint_is_adapter:
        from peft import PeftModel

        LOGGER.info("Loading LoRA adapter from %s", checkpoint)
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        if merge_adapter:
            LOGGER.info("Merging adapter weights")
            model = model.merge_and_unload()
    elif checkpoint:
        LOGGER.info(
            "Checkpoint %s: no local adapter_config.json found; loaded directly "
            "without a PEFT merge",
            checkpoint,
        )

    model.eval()
    model.to(device)
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _build_messages(
    image_path: str,
    prompt: str,
    *,
    image_pixel_kwargs: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Build the Qwen-VL chat message dict for one sample."""
    image_content: dict[str, Any] = {"type": "image", "image": image_path}
    if image_pixel_kwargs:
        image_content.update(image_pixel_kwargs)
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt},
            ],
        },
    ]


def generate_one(
    model: Any,
    processor: Any,
    record: EvalRecord,
    *,
    max_new_tokens: int = common.DEFAULT_MAX_NEW_TOKENS,
    device: str = "cuda:0",
) -> str:
    """Generate a single prediction for one eval sample."""
    from qwen_vl_utils import process_vision_info

    image_patch_size = _processor_image_patch_size(processor)
    image_pixel_kwargs = _processor_image_pixel_kwargs(processor)
    messages = _build_messages(
        record.image_path,
        record.prompt,
        image_pixel_kwargs=image_pixel_kwargs,
    )

    # Apply chat template
    try:
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older processors may not support enable_thinking
        try:
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            text_prompt = processor.apply_chat_template(messages, tokenize=False)

    imgs, _ = process_vision_info(
        messages,
        image_patch_size=image_patch_size,
    )

    inputs = processor(
        text=[text_prompt],
        images=imgs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(
        tokenizer, "eos_token_id", None
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID],
            use_cache=True,
        )

    generated = output_ids[0, prompt_len:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    return common.clean_prediction(raw_text)


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------


def run_eval(
    model: Any,
    processor: Any,
    records: list[EvalRecord],
    *,
    max_new_tokens: int = common.DEFAULT_MAX_NEW_TOKENS,
    device: str = "cuda:0",
    protocol: str = "both",
) -> dict[str, Any]:
    """Run inference + scoring on all records and return structured results."""
    # --- Phase 1: Generate predictions ---
    predictions_list: list[dict[str, Any]] = []
    ground_truths_list: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    n_skipped = 0

    start_gen = time.perf_counter()
    for record in tqdm(records, desc="Generating predictions"):
        # Check image exists
        if not Path(record.image_path).exists():
            LOGGER.warning(
                "Image not found: %s (sample %s); skipping", record.image_path, record.sample_id
            )
            n_skipped += 1
            continue

        try:
            prediction = generate_one(
                model,
                processor,
                record,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        except Exception as exc:
            LOGGER.warning("Generation failed for %s: %s", record.sample_id, exc)
            prediction = ""

        # Extract cxsmiles from prediction
        pred_cxsmiles = common.prediction_to_cxsmiles_opt(prediction)

        predictions_list.append(
            {
                "id": record.sample_id,
                "prediction": prediction,
                "cxsmiles_opt": pred_cxsmiles,
            }
        )
        ground_truths_list.append(
            {
                "id": record.sample_id,
                "cxsmiles_opt": record.cxsmiles_opt,
                "annotation": record.annotation,
            }
        )
        per_sample.append(
            {
                "id": record.sample_id,
                "target": record.cxsmiles_opt,
                "prediction": pred_cxsmiles,
                "raw_prediction": prediction,
                "has_r": record.has_r,
                "has_sg": record.has_sg,
                "has_m": record.has_m,
            }
        )

    gen_seconds = time.perf_counter() - start_gen
    LOGGER.info(
        "Generation complete: %d samples in %.1fs (%.2f s/sample), %d skipped",
        len(predictions_list),
        gen_seconds,
        gen_seconds / max(len(predictions_list), 1),
        n_skipped,
    )

    # --- Phase 2: Score with MCS evaluator ---
    start_eval = time.perf_counter()

    protocols_to_run = ["ours", "mg2"] if protocol == "both" else [protocol]

    all_metrics: dict[str, dict[str, Any]] = {}
    all_details: dict[str, list[dict[str, Any]]] = {}

    for proto in protocols_to_run:
        try:
            metrics, details = evaluate_markush_mcs(
                predictions_list,
                ground_truths_list,
                include_missing=True,
                return_details=True,
                protocol=proto,
            )
            all_metrics[proto] = metrics
            all_details[proto] = details
        except Exception as exc:
            LOGGER.error("MCS evaluation failed for protocol '%s': %s", proto, exc)
            all_metrics[proto] = {"error": str(exc)}
            all_details[proto] = []

    eval_seconds = time.perf_counter() - start_eval

    # --- Phase 3: Merge per-sample scores ---
    # Build a lookup from details for each protocol
    for proto, details in all_details.items():
        detail_by_id = {d["id"]: d for d in details}
        for sample in per_sample:
            sid = sample["id"]
            detail = detail_by_id.get(sid, {})
            prefix = f"{proto}_" if len(protocols_to_run) > 1 else ""
            for key in (
                "string_equality",
                "backbone_inchi_match",
                "cxsmi_equality",
                "cxsmiles_full_match",
                "r_group_accuracy",
                "m_section_match",
                "sg_section_match",
                "tanimoto_similarity",
                "stable_equality",
                "markush_full_match",
            ):
                sample[f"{prefix}{key}"] = detail.get(key)

    # --- Phase 4: Per-category breakdowns ---
    by_category = _compute_category_breakdowns(per_sample, all_details, protocols_to_run)

    return {
        "aggregate": all_metrics,
        "by_category": by_category,
        "per_sample": per_sample,
        "timing": {
            "generation_seconds": round(gen_seconds, 2),
            "evaluation_seconds": round(eval_seconds, 2),
            "total_seconds": round(gen_seconds + eval_seconds, 2),
            "samples_per_second": round(len(predictions_list) / gen_seconds, 2)
            if gen_seconds > 0
            else None,
        },
        "n_skipped": n_skipped,
    }


def _compute_category_breakdowns(
    per_sample: list[dict[str, Any]],
    all_details: dict[str, list[dict[str, Any]]],
    protocols: list[str],
) -> dict[str, Any]:
    """Compute per-category metric breakdowns."""
    # Build category masks
    categories = {
        "r_only": [s for s in per_sample if s["has_r"] and not s["has_sg"] and not s["has_m"]],
        "has_r": [s for s in per_sample if s["has_r"]],
        "has_sg": [s for s in per_sample if s["has_sg"]],
        "has_m": [s for s in per_sample if s["has_m"]],
        "combined": [s for s in per_sample if sum([s["has_r"], s["has_sg"], s["has_m"]]) >= 2],
    }

    result: dict[str, Any] = {}
    for cat_name, cat_samples in categories.items():
        if not cat_samples:
            result[cat_name] = {"n": 0}
            continue

        cat_ids = {s["id"] for s in cat_samples}
        cat_result: dict[str, Any] = {"n": len(cat_samples)}

        for proto in protocols:
            details = all_details.get(proto, [])
            cat_details = [d for d in details if d["id"] in cat_ids]
            if not cat_details:
                continue

            prefix = f"{proto}_" if len(protocols) > 1 else ""
            metric_keys = [
                "string_equality",
                "backbone_inchi_match",
                "cxsmi_equality",
                "cxsmiles_full_match",
                "tanimoto_similarity",
                "markush_full_match",
            ]
            for key in metric_keys:
                values = [d.get(key) for d in cat_details if d.get(key) is not None]
                if values:
                    cat_result[f"{prefix}{key}"] = round(
                        sum(float(v) for v in values) / len(values), 4
                    )

            # Component-specific metrics (only aggregate where applicable)
            for key in (
                "r_group_accuracy",
                "m_section_match",
                "sg_section_match",
                "stable_equality",
            ):
                values = [d.get(key) for d in cat_details if d.get(key) is not None]
                if values:
                    cat_result[f"{prefix}{key}"] = round(
                        sum(float(v) for v in values) / len(values), 4
                    )
                    cat_result[f"{prefix}{key}_n"] = len(values)

        result[cat_name] = cat_result

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_METRIC_DISPLAY = [
    ("String Equality", "string_equality"),
    ("CXSMI Equality", "cxsmi_equality"),
    ("Backbone InChI Match", "backbone_inchi_match"),
    ("CXSMILES Full Match", "cxsmiles_full_match"),
    ("R-group Accuracy", "r_group_accuracy"),
    ("Sg Section Match", "sg_section_match"),
    ("M Section Match", "m_section_match"),
    ("Stable Equality", "stable_equality_all"),
    ("Tanimoto Similarity", "tanimoto_similarity"),
    ("Markush Full Match", "markush_full_match"),
    ("Valid Rate", "valid_rate"),
]


def print_summary(results: dict[str, Any], *, checkpoint: str | None) -> None:
    """Print a compact summary table to stdout."""
    print()
    print("=" * 76)
    benchmark = results.get("benchmark")
    title = f"MARKUSH EVAL — checkpoint: {checkpoint or '(base model only)'}"
    if benchmark:
        title += f" — benchmark: {benchmark}"
    print(f"  {title}")
    print("=" * 76)

    aggregate = results["aggregate"]
    for proto, metrics in aggregate.items():
        if "error" in metrics:
            print(f"\n  [{proto}] ERROR: {metrics['error']}")
            continue
        print(f"\n  Protocol: {proto}  (n={metrics.get('n_samples', '?')})")
        print("  " + "-" * 50)
        for label, key in _METRIC_DISPLAY:
            val = metrics.get(key)
            if val is None:
                continue
            if isinstance(val, float):
                print(f"    {label:<28s} {val:>8.4f}")
            else:
                print(f"    {label:<28s} {val!s:>8s}")

    # Category breakdown
    by_category = results.get("by_category", {})
    if by_category:
        print("\n  Per-category breakdown:")
        print("  " + "-" * 50)
        for cat_name, cat_data in by_category.items():
            n = cat_data.get("n", 0)
            if n == 0:
                continue
            # Show the first protocol's cxsmi_equality as the representative metric
            cxsmi_keys = [k for k in cat_data if "cxsmi_equality" in k and not k.endswith("_n")]
            cxsmi_vals = " / ".join(f"{cat_data[k]:.4f}" for k in sorted(cxsmi_keys))
            backbone_keys = [
                k for k in cat_data if "backbone_inchi_match" in k and not k.endswith("_n")
            ]
            backbone_vals = " / ".join(f"{cat_data[k]:.4f}" for k in sorted(backbone_keys))
            print(
                f"    {cat_name:<12s}  n={n:<5d}  cxsmi={cxsmi_vals or 'N/A':<20s}  backbone={backbone_vals or 'N/A'}"
            )

    timing = results.get("timing", {})
    print(
        f"\n  Timing: gen={timing.get('generation_seconds', '?')}s  eval={timing.get('evaluation_seconds', '?')}s  total={timing.get('total_seconds', '?')}s"
    )
    if timing.get("samples_per_second"):
        print(f"  Throughput: {timing['samples_per_second']:.2f} samples/s")
    print("=" * 76)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline evaluation of a LoRA checkpoint on IP5-M.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to LoRA adapter checkpoint directory (omit for base-model-only eval). "
            "Security: loaded with trust_remote_code=True / torch unpickling and may "
            "execute code from the referenced repo — only use trusted sources."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help=(
            "Path to the base Qwen-VL model. Security: loaded with "
            "trust_remote_code=True and may execute code from the referenced repo "
            "— only use trusted sources."
        ),
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help=(
            "Path to evaluation JSONL file. Optional when --benchmark is supplied; "
            "then the benchmark default path is used."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=benchmark_choices(),
        help=(
            "Benchmark prompt preset: ip5m/uspto-markush use structure_only_markush, "
            "m2s uses patent_image_markush"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for resolving relative image paths in eval data",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples (default: all)",
    )
    parser.add_argument(
        "--canonicalize-markush-targets",
        action="store_true",
        help="Prepare eval Markush targets as verified canonicalized cxsmiles_opt.",
    )
    parser.add_argument(
        "--require-canonical-markush-targets",
        action="store_true",
        help="Skip eval Markush rows whose canonicalized target cannot be verified.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print summary only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for inference",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="both",
        choices=("ours", "mg2", "both"),
        help=(
            "MCS scoring protocol: ours (stricter), mg2 (APPROXIMATE in-package "
            "MG2 replication — not byte-exact; official numbers require the pinned "
            "DS4SD scorer in tools/mg2_scorer/), or both"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=common.DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=None,
        help="Minimum processor image pixels; omitted uses model default.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=1_048_576,
        help="Maximum processor image pixels.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32", "auto"),
        help="Model dtype",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge the LoRA adapter (keep as PeftModel)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (DEBUG level)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Determine device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA not available; falling back to CPU")
        device = "cpu"
        if args.dtype in ("bfloat16", "float16"):
            args.dtype = "float32"

    benchmark_name = normalize_benchmark_name(args.benchmark) if args.benchmark else None
    if args.eval_data is None and benchmark_name is None:
        parser.error("--eval-data is required unless --benchmark is supplied")

    benchmark_spec = get_benchmark_spec(benchmark_name) if benchmark_name else None
    eval_data = (
        resolve_benchmark_path(
            benchmark_name,
            explicit_path=args.eval_data,
            root=args.data_root,
        )
        if benchmark_name
        else Path(args.eval_data).expanduser()
    )
    default_prompt = benchmark_spec.prompt if benchmark_spec else DEFAULT_PROMPT

    # Load eval data
    LOGGER.info("Loading eval data from %s", eval_data)
    records = load_eval_data(
        eval_data,
        args.data_root,
        limit=args.n_samples,
        default_prompt=default_prompt,
        canonicalize_markush_targets=args.canonicalize_markush_targets,
        require_canonical_markush_targets=args.require_canonical_markush_targets,
    )
    if not records:
        LOGGER.error("No evaluation samples loaded")
        sys.exit(1)
    LOGGER.info("Loaded %d evaluation samples", len(records))

    # Category stats
    n_r = sum(1 for r in records if r.has_r)
    n_sg = sum(1 for r in records if r.has_sg)
    n_m = sum(1 for r in records if r.has_m)
    LOGGER.info("Categories — has_r: %d, has_sg: %d, has_m: %d", n_r, n_sg, n_m)

    # Load model
    model, processor = load_model_and_processor(
        args.base_model,
        args.checkpoint,
        device,
        dtype=args.dtype,
        merge_adapter=not args.no_merge,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )

    # Run evaluation
    results = run_eval(
        model,
        processor,
        records,
        max_new_tokens=args.max_new_tokens,
        device=device,
        protocol=args.protocol,
    )

    # Add metadata
    results["checkpoint"] = args.checkpoint
    results["base_model"] = args.base_model
    results["benchmark"] = benchmark_name
    results["n_samples"] = len(records)
    results["timestamp"] = datetime.now(UTC).isoformat()
    results["config"] = {
        "eval_data": args.eval_data,
        "resolved_eval_data": str(eval_data),
        "benchmark": benchmark_name,
        "prompt": default_prompt,
        "data_root": args.data_root,
        "n_samples_requested": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "device": device,
        "dtype": args.dtype,
        "protocol": args.protocol,
        "merge_adapter": not args.no_merge,
    }

    # Print summary
    print_summary(results, checkpoint=args.checkpoint)

    # Save JSON
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        LOGGER.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
