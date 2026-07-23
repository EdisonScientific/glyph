#!/usr/bin/env python3
"""Evaluation module for Markush structure recognition.

Uses position-aware Markush structure equality as the primary lightweight
evaluation metric.
Supports loading LoRA checkpoints, running inference, and computing
structured metrics for CXSMILES predictions.

Usage:
    python -m glyph.markush.eval.predict \\
        --checkpoint checkpoints/sft_2b/checkpoint-3000 \\
        --base-model Qwen/Qwen3.5-2B-Base \\
        --eval-data data/benchmarks/markush/ip5m/eval_ip5m.jsonl \\
        --image-root data/ \\
        --limit 50 \\
        --output eval_results.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from glyph.markush.data.ocsr import row_declares_ocsr_smiles
from glyph.markush.data.prepare_sft import (
    prompt_with_target_instruction,
    target_cxsmiles_for_record,
    target_spec_for_record,
)
from glyph.markush.eval import common
from glyph.markush.eval.markush_graph import markush_graph_equal

# Optional RDKit import
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchi

    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


LOGGER = logging.getLogger("glyph.markush.eval.predict")

DEFAULT_PROMPT = (
    "Convert this patent Markush structure image to CXSMILES. "
    "Return only the XML string in the form "
    "<markush><cxsmi>...</cxsmi><stable>...</stable></markush>."
)

IMAGE_FIELDS = ("image_path", "image", "path", "png_path", "rendered_path", "file_name")
TARGET_FIELDS = ("annotation", "target", "output", "text", "ground_truth", "label")

# Regex patterns: re-exported from glyph.markush.eval.common so this module
# can't drift from metrics_mcs / metrics_string / reward.  All XML-tag
# regexes use IGNORECASE | DOTALL via the common module.
_RE_CXSMI = common.RE_CXSMI
_RE_STABLE = common.RE_STABLE
_RE_R_TAG = common.RE_R_TAG
_RE_THINKING = common.RE_THINKING


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalSample:
    """A single evaluation sample."""

    sample_id: str
    image_path: str | None
    prompt: str
    ground_truth: str  # bare cxsmiles_opt (without XML wrapper)
    raw: dict[str, Any]


@dataclass
class SampleMetrics:
    """Per-sample evaluation metrics."""

    sample_id: str
    valid_xml: bool = False
    string_em: bool = False
    canonical_em: bool = False
    backbone_inchi_match: bool | None = None  # None if unparseable
    r_group_recall: float = 0.0
    r_group_precision: float = 0.0
    has_extensions: bool = False
    prediction: str = ""
    ground_truth: str = ""
    pred_canonical: str | None = None
    gt_canonical: str | None = None
    error: str = ""


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics."""

    n_samples: int = 0
    valid_rate: float = 0.0
    string_em: float = 0.0
    canonical_em: float = 0.0
    backbone_inchi_match: float = 0.0
    r_group_recall: float = 0.0
    r_group_precision: float = 0.0
    r_group_f1: float = 0.0
    has_extensions: float = 0.0


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str | Path | None,
    base_model_path: str | Path,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    merge_adapter: bool = True,
) -> tuple[Any, Any]:
    """Load base model + LoRA adapter, optionally merge, return model + processor.

    Security: ``base_model_path``/``checkpoint_path`` are loaded via
    ``transformers.from_pretrained(..., trust_remote_code=True)`` and may execute
    Python code from the referenced repository (and unpickle torch weights). Only
    load checkpoints/models from sources you trust.

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to LoRA adapter checkpoint. If None, loads only the base model.
    base_model_path : str | Path
        Path to the base model (e.g., Qwen/Qwen3.5-2B-Base).
    device : str
        Device to load model on (e.g., "cuda:0", "cpu").
    dtype : str
        Data type: "bfloat16", "float16", or "auto".
    merge_adapter : bool
        If True, merge the LoRA adapter into the base model for faster inference.

    Returns
    -------
    model : torch.nn.Module
        The loaded (and optionally merged) model.
    processor : AutoProcessor
        The processor/tokenizer for the model.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    common.silence_dependency_logging()
    LOGGER.info("Loading base model from %s", base_model_path)

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "auto": "auto",
    }.get(dtype, "auto")

    # Load processor
    processor_path = checkpoint_path if checkpoint_path else base_model_path
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
    )

    # Set up tokenizer
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    # Load base model
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load and optionally merge adapter
    if checkpoint_path:
        from peft import PeftModel

        LOGGER.info("Loading adapter from %s", checkpoint_path)
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        if merge_adapter:
            LOGGER.info("Merging adapter into base model")
            model = model.merge_and_unload()

    model.eval()
    model.to(device)
    LOGGER.info("Model loaded on %s", device)

    return model, processor


def generate(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = common.DEFAULT_MAX_NEW_TOKENS,
    device: str | None = None,
) -> str:
    """Generate from a single image + prompt. Returns cleaned prediction string.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
    processor : AutoProcessor
        The processor for the model.
    image : PIL.Image.Image
        Input image.
    prompt : str
        Text prompt for the model.
    max_new_tokens : int
        Maximum number of tokens to generate.
    device : str | None
        Device to use. If None, infers from model.

    Returns
    -------
    str
        The cleaned prediction string.
    """
    if device is None:
        device = next(model.parameters()).device

    # Build chat messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    try:
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        prompt_text = processor.apply_chat_template(messages, tokenize=False)

    # Encode
    encoded = processor(text=prompt_text, images=image, return_tensors="pt")
    encoded = {k: v.to(device) if torch.is_tensor(v) else v for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode
    generated_tokens = outputs[0, prompt_len:]
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return clean_prediction(raw_output)


# ---------------------------------------------------------------------------
# Post-processing and extraction
# ---------------------------------------------------------------------------


def clean_prediction(text: str) -> str:
    """Clean raw model output for evaluation.

    Thin wrapper over :func:`glyph.markush.eval.common.clean_prediction`.
    """
    return common.clean_prediction(text)


def extract_cxsmi(text: str) -> str | None:
    """Extract content from ``<cxsmi>...</cxsmi>`` tags (case-insensitive)."""
    if not text:
        return None
    m = _RE_CXSMI.search(text)
    return m.group(1).strip() if m else None


def extract_stable(text: str) -> str | None:
    """Extract content from ``<stable>...</stable>`` tags (case-insensitive)."""
    if not text:
        return None
    m = _RE_STABLE.search(text)
    return m.group(1).strip() if m else None


def has_valid_markush_xml(text: str) -> bool:
    """Check if text contains valid <markush><cxsmi>...</cxsmi></markush> structure."""
    if not text:
        return False
    cxsmi = extract_cxsmi(text)
    return cxsmi is not None and len(cxsmi) > 0


def extract_r_labels(cxsmiles: str) -> set[str]:
    """Extract R-group labels from ``<r>...</r>`` tags (case-insensitive).

    Returns a set; for the ordered list (with duplicates) see
    :func:`glyph.markush.eval.common.extract_r_labels`.
    """
    return set(common.extract_r_labels(cxsmiles))


def has_cxsmiles_extensions(cxsmiles: str) -> bool:
    """Check if CXSMILES has extension part (after |)."""
    if not cxsmiles:
        return False
    return "|" in cxsmiles


# ---------------------------------------------------------------------------
# Canonical SMILES comparison
# ---------------------------------------------------------------------------


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string.

    Process:
        1. Strip <r>...</r> tags and replace with * (wildcard atom)
        2. Take backbone part only (before |)
        3. Parse with RDKit and canonicalize

    Parameters
    ----------
    smiles : str
        Input SMILES string (possibly with <r>...</r> tags and | extensions).

    Returns
    -------
    str | None
        Canonical SMILES string, or None if parsing fails.
    """
    if not RDKIT_AVAILABLE:
        return None

    if not smiles or not isinstance(smiles, str):
        return None

    # Strip <r>...</r> tags, replace with *
    stripped = _RE_R_TAG.sub("*", smiles)

    # Take backbone only (before |)
    backbone = stripped.split("|")[0].strip()

    if not backbone:
        return None

    try:
        mol = Chem.MolFromSmiles(backbone)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonical_match(pred: str, gt: str) -> bool:
    """Compare two CXSMILES using position-aware Markush structure equality.

    The function name is retained for compatibility with older evaluation
    outputs. Unlike the previous backbone-only canonical-SMILES comparison,
    this predicate keeps R-label identity and CXSMILES feature positions bound
    to atoms.

    Parameters
    ----------
    pred : str
        Predicted CXSMILES string.
    gt : str
        Ground truth CXSMILES string.

    Returns
    -------
    bool
        True if the position-aware Markush structures match, False otherwise.
    """
    result = markush_graph_equal(gt, pred)
    return bool(result.equal and result.valid_gt and result.valid_pred)


def backbone_inchi_match(pred: str, gt: str) -> bool | None:
    """Compare backbones via InChI after wildcard→C and stereo removal.

    This is more permissive than canonical_match as it:
        1. Replaces * with C (carbon) for proper InChI generation
        2. Removes stereochemistry
        3. Compares InChI strings

    Parameters
    ----------
    pred : str
        Predicted CXSMILES string.
    gt : str
        Ground truth CXSMILES string.

    Returns
    -------
    bool | None
        True if InChIs match, False if they don't, None if unparseable.
    """
    if not RDKIT_AVAILABLE:
        return None

    def to_inchi(cxsmiles: str) -> str | None:
        if not cxsmiles:
            return None

        # Strip R-tags and replace with C
        stripped = _RE_R_TAG.sub("C", cxsmiles)
        backbone = stripped.split("|")[0].strip()
        # Also replace any remaining wildcards
        backbone = backbone.replace("*", "C")

        try:
            params = Chem.SmilesParserParams()
            params.sanitize = False
            mol = Chem.MolFromSmiles(backbone, params)
            if mol is None:
                return None
            Chem.RemoveStereochemistry(mol)
            with contextlib.suppress(Exception):
                Chem.SanitizeMol(mol)
            return MolToInchi(mol)
        except Exception:
            return None

    pred_inchi = to_inchi(pred)
    gt_inchi = to_inchi(gt)

    if pred_inchi is None or gt_inchi is None:
        return None

    return pred_inchi == gt_inchi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _extract_ground_truth(
    row: dict[str, Any],
    *,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
) -> str:
    """Extract ground truth CXSMILES from a data row.

    Returns bare cxsmiles_opt (without <markush> wrapper).
    """
    shared_target = target_cxsmiles_for_record(
        row,
        canonicalize_markush=canonicalize_markush_targets,
        require_canonical=require_canonical_markush_targets,
    )
    if shared_target:
        return shared_target.strip()
    if canonicalize_markush_targets and require_canonical_markush_targets:
        return ""

    # Compatibility fallback for legacy eval rows with only cxsmi/target fields.
    cxsmiles = row.get("cxsmi")
    if isinstance(cxsmiles, str) and cxsmiles.strip():
        return cxsmiles.strip()

    # Fall back to annotation fields
    for fld in TARGET_FIELDS:
        value = row.get(fld)
        if isinstance(value, str) and value.strip():
            # If it's XML wrapped, extract the cxsmi part
            extracted = extract_cxsmi(value)
            if extracted:
                return extracted
            # Otherwise return as-is
            return value.strip()

    return ""


def _row_prompt(
    row: dict[str, Any],
    prompt_template: str,
    *,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
) -> str:
    sft_prompt = common.sft_user_prompt(row)
    if sft_prompt:
        return sft_prompt
    base_prompt = str(row.get("prompt") or prompt_template)
    target = target_spec_for_record(
        row,
        canonicalize_markush=canonicalize_markush_targets,
        require_canonical=require_canonical_markush_targets,
    )
    if target is None:
        return base_prompt
    return prompt_with_target_instruction(base_prompt, target)


def _resolve_image_path(row: dict[str, Any], image_root: Path | None) -> str | None:
    """Resolve image path from a data row."""
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
        if not candidate.is_absolute() and image_root is not None:
            candidate = image_root / candidate
        return str(candidate)
    return None


def load_eval_samples(
    path: str | Path,
    image_root: str | Path | None,
    prompt_template: str,
    limit: int | None = None,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
) -> list[EvalSample]:
    """Load evaluation samples from a JSONL file.

    Parameters
    ----------
    path : str | Path
        Path to JSONL file.
    image_root : str | Path | None
        Root directory for relative image paths.
    prompt_template : str
        Default prompt to use if not specified in data.
    limit : int | None
        Maximum number of samples to load.

    Returns
    -------
    list[EvalSample]
        List of evaluation samples.
    """
    samples = []
    image_root_p = Path(image_root).expanduser() if image_root else None

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if row_declares_ocsr_smiles(row):
                LOGGER.debug("Skipping OCSR SMILES row %d in Markush predict loader", line_no)
                continue
            gt = _extract_ground_truth(
                row,
                canonicalize_markush_targets=canonicalize_markush_targets,
                require_canonical_markush_targets=require_canonical_markush_targets,
            )
            if not gt:
                continue

            img = _resolve_image_path(row, image_root_p)
            sample_id = str(row.get("id") or row.get("sample_id") or row.get("uid") or line_no)
            prompt = _row_prompt(
                row,
                prompt_template,
                canonicalize_markush_targets=canonicalize_markush_targets,
                require_canonical_markush_targets=require_canonical_markush_targets,
            )

            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    image_path=img,
                    prompt=prompt,
                    ground_truth=gt,
                    raw=row,
                )
            )

            if limit is not None and len(samples) >= limit:
                break

    return samples


def load_image(path: str | None) -> Image.Image | None:
    """Load and convert an image to RGB."""
    if not path:
        return None
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_sample(
    prediction: str,
    ground_truth: str,
    sample_id: str = "",
) -> SampleMetrics:
    """Evaluate a single prediction against ground truth.

    Parameters
    ----------
    prediction : str
        Model prediction (may be full XML or just cxsmiles).
    ground_truth : str
        Ground truth cxsmiles_opt (bare, without XML wrapper).
    sample_id : str
        Identifier for the sample.

    Returns
    -------
    SampleMetrics
        Computed metrics for this sample.
    """
    metrics = SampleMetrics(
        sample_id=sample_id,
        ground_truth=ground_truth,
    )

    # Check for valid XML structure
    metrics.valid_xml = has_valid_markush_xml(prediction)

    # Extract cxsmi from prediction
    pred_cxsmi = extract_cxsmi(prediction)
    if pred_cxsmi is None:
        # If no XML wrapper, use the whole prediction
        pred_cxsmi = prediction if prediction else ""

    metrics.prediction = pred_cxsmi

    # String exact match
    metrics.string_em = pred_cxsmi == ground_truth

    # Canonical SMILES match
    pred_canonical = canonicalize_smiles(pred_cxsmi)
    gt_canonical = canonicalize_smiles(ground_truth)
    metrics.pred_canonical = pred_canonical
    metrics.gt_canonical = gt_canonical

    metrics.canonical_em = canonical_match(pred_cxsmi, ground_truth)

    # Backbone InChI match
    metrics.backbone_inchi_match = backbone_inchi_match(pred_cxsmi, ground_truth)

    # R-group metrics
    pred_r_labels = extract_r_labels(pred_cxsmi)
    gt_r_labels = extract_r_labels(ground_truth)

    if gt_r_labels:
        matched = pred_r_labels & gt_r_labels
        metrics.r_group_recall = len(matched) / len(gt_r_labels)
    else:
        metrics.r_group_recall = 1.0 if not pred_r_labels else 0.0

    if pred_r_labels:
        matched = pred_r_labels & gt_r_labels
        metrics.r_group_precision = len(matched) / len(pred_r_labels)
    else:
        metrics.r_group_precision = 1.0 if not gt_r_labels else 0.0

    # Has extensions
    metrics.has_extensions = has_cxsmiles_extensions(pred_cxsmi)

    return metrics


def compute_aggregate_metrics(sample_metrics: list[SampleMetrics]) -> AggregateMetrics:
    """Compute aggregate metrics from per-sample results.

    Parameters
    ----------
    sample_metrics : list[SampleMetrics]
        List of per-sample metrics.

    Returns
    -------
    AggregateMetrics
        Aggregated metrics.
    """
    n = len(sample_metrics)
    if n == 0:
        return AggregateMetrics()

    valid_count = sum(1 for m in sample_metrics if m.valid_xml)
    string_em_count = sum(1 for m in sample_metrics if m.string_em)
    canonical_em_count = sum(1 for m in sample_metrics if m.canonical_em)

    # Backbone InChI match (exclude None values)
    backbone_values = [
        m.backbone_inchi_match for m in sample_metrics if m.backbone_inchi_match is not None
    ]
    backbone_match_count = sum(1 for v in backbone_values if v)
    backbone_total = len(backbone_values) if backbone_values else n

    r_group_recall = sum(m.r_group_recall for m in sample_metrics) / n
    r_group_precision = sum(m.r_group_precision for m in sample_metrics) / n

    if r_group_precision + r_group_recall > 0:
        r_group_f1 = 2 * r_group_precision * r_group_recall / (r_group_precision + r_group_recall)
    else:
        r_group_f1 = 0.0

    has_extensions_count = sum(1 for m in sample_metrics if m.has_extensions)

    return AggregateMetrics(
        n_samples=n,
        valid_rate=valid_count / n,
        string_em=string_em_count / n,
        canonical_em=canonical_em_count / n,
        backbone_inchi_match=backbone_match_count / backbone_total if backbone_total > 0 else 0.0,
        r_group_recall=r_group_recall,
        r_group_precision=r_group_precision,
        r_group_f1=r_group_f1,
        has_extensions=has_extensions_count / n,
    )


def run_evaluation(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    max_new_tokens: int = common.DEFAULT_MAX_NEW_TOKENS,
    device: str | None = None,
    gc_every: int = 10,
    show_progress: bool = True,
) -> tuple[AggregateMetrics, list[SampleMetrics]]:
    """Run evaluation on a list of samples.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded model.
    processor : AutoProcessor
        The processor for the model.
    samples : list[EvalSample]
        List of evaluation samples.
    max_new_tokens : int
        Maximum number of tokens to generate.
    device : str | None
        Device to use. If None, infers from model.
    gc_every : int
        Run garbage collection every N samples (0 to disable).
    show_progress : bool
        Whether to show a progress bar.

    Returns
    -------
    aggregate : AggregateMetrics
        Aggregated metrics.
    sample_metrics : list[SampleMetrics]
        Per-sample metrics.
    """
    sample_metrics = []
    iterator = tqdm(samples, desc="Evaluating") if show_progress else samples

    for i, sample in enumerate(iterator):
        try:
            # Load image
            image = load_image(sample.image_path)
            if image is None:
                result = SampleMetrics(
                    sample_id=sample.sample_id,
                    ground_truth=sample.ground_truth,
                    error=f"Could not load image: {sample.image_path}",
                )
                sample_metrics.append(result)
                continue

            # Generate prediction
            prediction = generate(model, processor, image, sample.prompt, max_new_tokens, device)

            # Evaluate
            result = evaluate_sample(prediction, sample.ground_truth, sample.sample_id)
            sample_metrics.append(result)

            # Update progress bar
            if show_progress:
                current_agg = compute_aggregate_metrics(sample_metrics)
                iterator.set_postfix(
                    {
                        "canonical_em": f"{current_agg.canonical_em:.2%}",
                        "valid": f"{current_agg.valid_rate:.2%}",
                    }
                )

        except Exception as e:
            result = SampleMetrics(
                sample_id=sample.sample_id,
                ground_truth=sample.ground_truth,
                error=str(e),
            )
            sample_metrics.append(result)
            LOGGER.warning("Error evaluating sample %s: %s", sample.sample_id, e)

        # Garbage collection
        if gc_every > 0 and (i + 1) % gc_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = compute_aggregate_metrics(sample_metrics)
    return aggregate, sample_metrics


# ---------------------------------------------------------------------------
# W&B integration
# ---------------------------------------------------------------------------


def log_to_wandb(
    aggregate: AggregateMetrics,
    sample_metrics: list[SampleMetrics],
    run_name: str | None = None,
    project: str = "markush-eval",
    max_samples_in_table: int = 100,
) -> None:
    """Log evaluation results to Weights & Biases.

    Parameters
    ----------
    aggregate : AggregateMetrics
        Aggregated metrics.
    sample_metrics : list[SampleMetrics]
        Per-sample metrics.
    run_name : str | None
        Name for the W&B run.
    project : str
        W&B project name.
    max_samples_in_table : int
        Maximum number of samples to include in the table.
    """
    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb not installed. Skipping W&B logging.")
        return

    # Initialize run
    wandb.init(project=project, name=run_name, reinit=True)

    # Log aggregate metrics
    wandb.log(asdict(aggregate))

    # Create a sample table
    columns = [
        "sample_id",
        "valid_xml",
        "string_em",
        "canonical_em",
        "backbone_inchi_match",
        "r_group_recall",
        "r_group_precision",
        "prediction",
        "ground_truth",
    ]
    table = wandb.Table(columns=columns)

    for m in sample_metrics[:max_samples_in_table]:
        table.add_data(
            m.sample_id,
            m.valid_xml,
            m.string_em,
            m.canonical_em,
            m.backbone_inchi_match,
            m.r_group_recall,
            m.r_group_precision,
            m.prediction[:500] if m.prediction else "",
            m.ground_truth[:500] if m.ground_truth else "",
        )

    wandb.log({"samples": table})
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Markush structure recognition with canonical SMILES comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to LoRA checkpoint directory. Security: loaded with "
            "trust_remote_code=True / torch unpickling and may execute code from the "
            "referenced repo — only use trusted sources."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help=(
            "Path to base model (e.g., Qwen/Qwen3.5-2B-Base). Security: loaded with "
            "trust_remote_code=True and may execute code from the referenced repo "
            "— only use trusted sources."
        ),
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge adapter (slower but uses less memory)",
    )

    # Data arguments
    parser.add_argument(
        "--eval-data",
        type=str,
        required=True,
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory for relative image paths",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt template for generation",
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

    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=common.DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (e.g., cuda:0, cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="Model dtype",
    )

    # MCS-protocol eval (record both protocols on the full set, or just one)
    parser.add_argument(
        "--protocol",
        choices=("ours", "mg2", "both", "none"),
        default="none",
        help=(
            "Also run the MCS-based evaluator on generated predictions: "
            "'ours' = our stricter eval; 'mg2' = APPROXIMATE in-package "
            "MarkushGrapher-2 replication (not byte-exact — official numbers "
            "require the pinned DS4SD scorer in tools/mg2_scorer/); "
            "'both' = compute and record both side-by-side; "
            "'none' (default) = skip MCS eval and only report the lightweight "
            "canonical/backbone metrics."
        ),
    )

    # W&B arguments
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="markush-eval",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    # Misc arguments
    parser.add_argument(
        "--gc-every",
        type=int,
        default=10,
        help="Run garbage collection every N samples (0 to disable)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    common.silence_dependency_logging()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Check RDKit availability
    if not RDKIT_AVAILABLE:
        LOGGER.warning(
            "RDKit not available. canonical_em and backbone_inchi_match metrics will not work."
        )

    # Load evaluation samples
    LOGGER.info("Loading evaluation samples from %s", args.eval_data)
    samples = load_eval_samples(
        args.eval_data,
        args.image_root,
        args.prompt,
        limit=args.limit,
        canonicalize_markush_targets=args.canonicalize_markush_targets,
        require_canonical_markush_targets=args.require_canonical_markush_targets,
    )
    LOGGER.info("Loaded %d evaluation samples", len(samples))

    if not samples:
        LOGGER.error("No evaluation samples found")
        sys.exit(1)

    # Load model
    model, processor = load_model(
        checkpoint_path=args.checkpoint,
        base_model_path=args.base_model,
        device=args.device,
        dtype=args.dtype,
        merge_adapter=not args.no_merge,
    )

    # Run evaluation
    aggregate, sample_metrics = run_evaluation(
        model,
        processor,
        samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        gc_every=args.gc_every,
        show_progress=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (lightweight metrics)")
    print("=" * 60)
    print(f"  Samples evaluated:     {aggregate.n_samples:>8d}")
    print(f"  Valid XML rate:        {aggregate.valid_rate:>8.2%}")
    print(f"  String exact match:    {aggregate.string_em:>8.2%}")
    print(f"  Position-correct CX:   {aggregate.canonical_em:>8.2%}  <-- PRIMARY METRIC")
    print(f"  Backbone InChI match:  {aggregate.backbone_inchi_match:>8.2%}")
    print(f"  R-group recall:        {aggregate.r_group_recall:>8.2%}")
    print(f"  R-group precision:     {aggregate.r_group_precision:>8.2%}")
    print(f"  R-group F1:            {aggregate.r_group_f1:>8.2%}")
    print(f"  Has extensions:        {aggregate.has_extensions:>8.2%}")
    print("=" * 60)

    # Optionally run the full MCS evaluator under the requested protocol(s).
    mcs_payload: dict[str, Any] | None = None
    if args.protocol != "none":
        from glyph.markush.eval.metrics_mcs import (
            evaluate_markush_both_protocols,
            evaluate_markush_mcs,
            format_metrics_summary,
        )

        pred_rows = [
            {"id": m.sample_id, "prediction": m.prediction} for m in sample_metrics if m.prediction
        ]
        gt_rows = [
            {"id": m.sample_id, "cxsmiles_opt": m.ground_truth}
            for m in sample_metrics
            if m.ground_truth
        ]
        if not pred_rows or not gt_rows:
            LOGGER.warning("No predictions/ground-truth rows for MCS eval; skipping.")
        elif args.protocol == "both":
            both_metrics, both_details = evaluate_markush_both_protocols(
                pred_rows,
                gt_rows,
                include_missing=True,
                return_details=True,
            )
            print()
            for proto, m in both_metrics.items():
                print(format_metrics_summary(m, label=f"MCS [{proto}]"))
                print()
            mcs_payload = {"protocol": "both", "metrics": both_metrics, "per_sample": both_details}
        else:
            mcs_metrics, mcs_details = evaluate_markush_mcs(
                pred_rows,
                gt_rows,
                include_missing=True,
                return_details=True,
                protocol=args.protocol,
            )
            print()
            print(format_metrics_summary(mcs_metrics, label=f"MCS [{args.protocol}]"))
            mcs_payload = {
                "protocol": args.protocol,
                "metrics": mcs_metrics,
                "per_sample": mcs_details,
            }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data: dict[str, Any] = {
            "aggregate": asdict(aggregate),
            "samples": [
                {
                    "sample_id": m.sample_id,
                    "valid_xml": m.valid_xml,
                    "string_em": m.string_em,
                    "canonical_em": m.canonical_em,
                    "backbone_inchi_match": m.backbone_inchi_match,
                    "r_group_recall": m.r_group_recall,
                    "r_group_precision": m.r_group_precision,
                    "has_extensions": m.has_extensions,
                    "prediction": m.prediction[:1000] if m.prediction else "",
                    "ground_truth": m.ground_truth[:1000] if m.ground_truth else "",
                    "pred_canonical": m.pred_canonical,
                    "gt_canonical": m.gt_canonical,
                    "error": m.error,
                }
                for m in sample_metrics
            ],
        }
        if mcs_payload is not None:
            output_data["mcs"] = mcs_payload

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str)

        LOGGER.info("Results saved to %s", output_path)

    # Log to W&B
    if args.wandb:
        LOGGER.info("Logging to Weights & Biases")
        log_to_wandb(
            aggregate,
            sample_metrics,
            run_name=args.wandb_run_name,
            project=args.wandb_project,
        )


if __name__ == "__main__":
    main()
