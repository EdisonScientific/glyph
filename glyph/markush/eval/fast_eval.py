#!/usr/bin/env python3
"""Fast batched inference and MG2 MCS evaluation for Markush recognition.

This module is a throughput-oriented alternative to ``glyph.markush.eval.predict``.
It batches Qwen-VL image/text prompts on each GPU and, when requested, shards
samples across multiple GPUs with one worker process per GPU.

Example
-------
python -m glyph.markush.eval.fast_eval \
    --checkpoint "${GLYPH_DATA_ROOT:-./data}/checkpoints/run/checkpoint-12000" \
    --base-model "${GLYPH_DATA_ROOT:-./data}/models/Qwen3.5-2B-Base" \
    --eval-data "${GLYPH_DATA_ROOT:-./data}/evals/eval_ip5m.jsonl" \
    --image-root "${GLYPH_DATA_ROOT:-./data}" \
    --batch-size 8 \
    --max-new-tokens 1024 \
    --output results.json \
    --num-gpus 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue as queue_mod
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
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
from glyph.markush.data.prompt_policies import (
    MARKUSH_PROMPT_POLICIES,
    MARKUSH_PROMPT_POLICY_DEFAULT,
    apply_markush_prompt_policy,
)
from glyph.markush.eval import common
from glyph.markush.eval.markush_graph import canonical_markush_signature

LOGGER = logging.getLogger("glyph.markush.eval.fast_eval")

DEFAULT_PROMPT = (
    "Convert this patent Markush structure image to CXSMILES. "
    "Return only the XML string in the form "
    "<markush><cxsmi>...</cxsmi><stable>...</stable></markush>."
)

IMAGE_FIELDS = ("image_path", "image", "path", "png_path", "rendered_path", "file_name")
TARGET_FIELDS = ("annotation", "target", "output", "text", "ground_truth", "label")
DEFAULT_EOS_TOKEN_IDS = (248044, 248046)  # <|endoftext|> and <|im_end|> for this Qwen setup.

# Canonical self-consistency (majority-vote) sampling recipe.  These knobs
# reproduce the IP5-M MV@8 headline (532/878 = 60.59%); see docs/reproducing.md.  The
# single-image predictor reuses them so that MV@8 on one image matches the
# batched eval decode exactly.
MV_DEFAULT_VOTE_K = 8
MV_DEFAULT_TEMPERATURE = 0.7
MV_DEFAULT_TOP_P = 0.95
MV_DEFAULT_SEED = 20260625


@contextmanager
def temporary_tokenizer_padding_side(processor: Any, padding_side: str):
    """Temporarily set decoder generation padding without mutating training setup."""

    tokenizer = getattr(processor, "tokenizer", processor)
    missing = object()
    original = getattr(tokenizer, "padding_side", missing)
    if original is missing:
        yield
        return

    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original


@dataclass(slots=True)
class EvalSample:
    """One JSONL row prepared for evaluation."""

    index: int
    sample_id: str
    image_path: str | None
    prompt: str
    ground_truth: str


@dataclass(slots=True)
class PredictionResult:
    """Generation result for one sample."""

    index: int
    sample_id: str
    image_path: str | None
    ground_truth: str
    prediction: str = ""
    prediction_cxsmiles: str = ""
    candidates: list[dict[str, Any]] = dataclass_field(default_factory=list)
    majority_vote: dict[str, Any] = dataclass_field(default_factory=dict)
    error: str = ""
    gpu: int | None = None


@dataclass(slots=True)
class WorkerConfig:
    """Pickleable worker configuration for multiprocessing."""

    checkpoint: str | None
    base_model: str
    batch_size: int
    max_new_tokens: int
    dtype: str
    merge_adapter: bool
    eos_token_ids: tuple[int, ...]
    num_beams: int
    length_penalty: float
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int | None
    num_return_sequences: int
    seed: int | None
    force_output_prefix: str | None
    multi_image_mode: str
    crop_overlap: float
    image_min_pixels: int | None
    image_max_pixels: int | None
    show_progress: bool
    majority_vote: bool = False
    # Compute per-token transition log-probs for vote tiebreaking. Default OFF:
    # output_scores=True materializes a (num_seq x steps x vocab) tensor that
    # OOMs at MV@8 with long sequences (~60GB). Logprobs are only a vote
    # tiebreaker; without them ties fall back to the first valid candidate.
    vote_logprobs: bool = False


@dataclass(slots=True)
class GeneratedCandidate:
    """One decoded candidate plus non-oracle vote metadata."""

    prediction: str
    prediction_cxsmiles: str
    logprob_sum: float | None = None
    token_count: int | None = None
    mean_logprob: float | None = None
    canonical_signature: str = ""
    valid_for_vote: bool = False
    vote_error: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _extract_cxsmi_from_xml(text: str) -> str | None:
    match = common.RE_CXSMI.search(text)
    return match.group(1).strip() if match else None


def _extract_ground_truth(
    row: dict[str, Any],
    *,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
) -> str:
    """Extract a bare ``cxsmiles_opt`` value from a JSONL row."""
    shared_target = target_cxsmiles_for_record(
        row,
        canonicalize_markush=canonicalize_markush_targets,
        require_canonical=require_canonical_markush_targets,
    )
    if shared_target:
        return shared_target.strip()
    if canonicalize_markush_targets and require_canonical_markush_targets:
        return ""

    cxsmiles = row.get("cxsmi")
    if isinstance(cxsmiles, str) and cxsmiles.strip():
        return cxsmiles.strip()

    for field in TARGET_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            extracted = _extract_cxsmi_from_xml(value)
            return extracted if extracted else value.strip()
    return ""


def _row_prompt(
    row: dict[str, Any],
    default_prompt: str,
    *,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
    ignore_row_prompts: bool = False,
    markush_prompt_policy: str = MARKUSH_PROMPT_POLICY_DEFAULT,
) -> str:
    sft_prompt = "" if ignore_row_prompts else common.sft_user_prompt(row)
    if sft_prompt:
        return apply_markush_prompt_policy(sft_prompt, markush_prompt_policy)
    base_prompt = str(default_prompt if ignore_row_prompts else row.get("prompt") or default_prompt)
    target = target_spec_for_record(
        row,
        canonicalize_markush=canonicalize_markush_targets,
        require_canonical=require_canonical_markush_targets,
    )
    if target is None:
        return apply_markush_prompt_policy(base_prompt, markush_prompt_policy)
    return apply_markush_prompt_policy(
        prompt_with_target_instruction(base_prompt, target),
        markush_prompt_policy,
    )


def _row_id(row: dict[str, Any], line_no: int) -> str:
    """Return a stable sample id without treating valid falsy ids as missing."""
    for key in ("id", "sample_id", "uid"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(line_no)


def _resolve_image_path(row: dict[str, Any], image_root: Path | None) -> str | None:
    candidates: list[Any] = []
    sft_image = common.sft_image_path(row)
    if sft_image:
        candidates.append(sft_image)
    for field in IMAGE_FIELDS:
        value = row.get(field)
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
    prompt: str,
    *,
    limit: int | None = None,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
    ignore_row_prompts: bool = False,
    markush_prompt_policy: str = MARKUSH_PROMPT_POLICY_DEFAULT,
) -> list[EvalSample]:
    """Load JSONL samples with image paths and ground-truth CXSMILES."""
    samples: list[EvalSample] = []
    image_root_path = Path(image_root).expanduser() if image_root else None

    if limit is not None and limit <= 0:
        return []

    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row_declares_ocsr_smiles(row):
                LOGGER.debug("Skipping OCSR SMILES row %d in Markush fast-eval loader", line_no)
                continue
            ground_truth = _extract_ground_truth(
                row,
                canonicalize_markush_targets=canonicalize_markush_targets,
                require_canonical_markush_targets=require_canonical_markush_targets,
            )
            if not ground_truth:
                LOGGER.debug("Skipping row %d without ground truth", line_no)
                continue

            sample_id = _row_id(row, line_no)
            samples.append(
                EvalSample(
                    index=len(samples),
                    sample_id=sample_id,
                    image_path=_resolve_image_path(row, image_root_path),
                    prompt=_row_prompt(
                        row,
                        prompt,
                        canonicalize_markush_targets=canonicalize_markush_targets,
                        require_canonical_markush_targets=require_canonical_markush_targets,
                        ignore_row_prompts=ignore_row_prompts,
                        markush_prompt_policy=markush_prompt_policy,
                    ),
                    ground_truth=ground_truth,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_image(path: str | None) -> Image.Image:
    """Load a PIL image as RGB and detach it from the file handle."""
    if not path:
        raise ValueError("missing image path")
    with Image.open(path) as image:
        image = image.convert("RGB") if image.mode != "RGB" else image.copy()
        image.load()
        return image


# ---------------------------------------------------------------------------
# Model and generation
# ---------------------------------------------------------------------------


def _torch_dtype(dtype: str) -> torch.dtype | str:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }.get(dtype, "auto")


def _load_processor(checkpoint: str | None, base_model: str) -> Any:
    from transformers import AutoProcessor

    paths = [checkpoint, base_model] if checkpoint else [base_model]
    last_error: Exception | None = None
    for processor_path in paths:
        if not processor_path:
            continue
        try:
            processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
            tokenizer = getattr(processor, "tokenizer", processor)
            # Chat-template fix: base VLMs (and slimmed adapter checkpoints) expose
            # the chat template on the tokenizer, not the processor; copy it across
            # so apply_chat_template works. Mirrors eval.checkpoint.load_model_and_processor.
            if getattr(processor, "chat_template", None) is None and getattr(
                tokenizer, "chat_template", None
            ):
                processor.chat_template = tokenizer.chat_template
            if getattr(tokenizer, "pad_token_id", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            return processor
        except Exception as exc:  # pragma: no cover - depends on local checkpoint layout.
            last_error = exc
            LOGGER.warning("Could not load processor from %s: %s", processor_path, exc)
    raise RuntimeError(f"Could not load processor from checkpoint or base model: {last_error}")


def _configure_processor_pixels(
    processor: Any,
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if image_min_pixels is not None and hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = image_min_pixels
    if image_max_pixels is not None and hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = image_max_pixels


def _checkpoint_is_peft_adapter(checkpoint: str | None) -> bool:
    return bool(checkpoint) and (Path(checkpoint) / "adapter_config.json").exists()


def load_model_and_processor(config: WorkerConfig, device: str) -> tuple[Any, Any]:
    """Load a full checkpoint or the base VLM plus optional LoRA adapter."""
    from transformers import AutoModelForImageTextToText

    processor = _load_processor(config.checkpoint, config.base_model)
    _configure_processor_pixels(
        processor,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
    )
    model_kwargs = {
        "dtype": _torch_dtype(config.dtype),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    checkpoint_is_adapter = _checkpoint_is_peft_adapter(config.checkpoint)
    model_path = config.base_model
    if config.checkpoint and not checkpoint_is_adapter:
        model_path = config.checkpoint

    LOGGER.info("Loading model from %s on %s", model_path, device)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    except Exception as exc:
        LOGGER.warning("ImageTextToText load failed (%s); retrying with device_map", exc)
        model_kwargs["device_map"] = device
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)

    if config.checkpoint and checkpoint_is_adapter:
        from peft import PeftModel

        LOGGER.info("Loading LoRA adapter from %s", config.checkpoint)
        model = PeftModel.from_pretrained(model, config.checkpoint, is_trainable=False)
        if config.merge_adapter:
            LOGGER.info("Merging LoRA adapter into base model")
            model = model.merge_and_unload()
    elif config.checkpoint:
        LOGGER.info(
            "Checkpoint %s: no local adapter_config.json found; loaded directly "
            "without a PEFT merge",
            config.checkpoint,
        )

    model.eval()
    model.to(device)
    return model, processor


def build_prompt_text(processor: Any, prompt: str, *, num_images: int = 1) -> str:
    """Apply the Qwen chat template for one or more image views + instruction."""
    image_items = [{"type": "image"} for _ in range(max(1, num_images))]
    if num_images > 1:
        prompt = (
            "You are given multiple views of the same Markush structure. "
            "The first image is the full original image. The remaining images "
            "are zoomed crops in reading order. Use all views to recover small "
            "labels, attachment points, and Sg/m annotations. "
            f"{prompt}"
        )
    messages = [
        {
            "role": "user",
            "content": [*image_items, {"type": "text", "text": prompt}],
        }
    ]
    if not hasattr(processor, "apply_chat_template"):
        return prompt

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return processor.apply_chat_template(messages, tokenize=False)


def _move_tensors_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _tile_box(
    width: int,
    height: int,
    *,
    grid: int,
    row: int,
    col: int,
    overlap: float,
) -> tuple[int, int, int, int]:
    tile_w = width / grid
    tile_h = height / grid
    x0 = int(max(0, round(col * tile_w - overlap * tile_w)))
    y0 = int(max(0, round(row * tile_h - overlap * tile_h)))
    x1 = int(min(width, round((col + 1) * tile_w + overlap * tile_w)))
    y1 = int(min(height, round((row + 1) * tile_h + overlap * tile_h)))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def build_image_views(
    image: Image.Image,
    *,
    mode: str,
    crop_overlap: float,
) -> list[Image.Image]:
    """Return the full image plus deterministic zoom crops for eval-time probing."""
    if mode == "single":
        return [image]
    if mode == "global_2x2":
        grid = 2
    elif mode == "global_3x3":
        grid = 3
    else:
        raise ValueError(f"Unknown multi-image mode: {mode}")

    views = [image]
    for row in range(grid):
        for col in range(grid):
            box = _tile_box(
                image.width,
                image.height,
                grid=grid,
                row=row,
                col=col,
                overlap=crop_overlap,
            )
            views.append(image.crop(box))
    return views


def _normalize_image_groups(
    images: list[Image.Image] | list[list[Image.Image]],
) -> list[list[Image.Image]]:
    if not images:
        return []
    first = images[0]
    if isinstance(first, Image.Image):
        return [[image] for image in images]  # type: ignore[list-item]
    return images  # type: ignore[return-value]


class ForceOutputPrefixLogitsProcessor:
    """Force the first generated tokens to match a literal text prefix."""

    def __init__(self, *, prompt_len: int, prefix_token_ids: list[int]) -> None:
        self.prompt_len = prompt_len
        self.prefix_token_ids = prefix_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_len = int(input_ids.shape[1]) - self.prompt_len
        if generated_len < 0 or generated_len >= len(self.prefix_token_ids):
            return scores
        forced_token_id = self.prefix_token_ids[generated_len]
        forced_scores = scores.new_full(scores.shape, -float("inf"))
        forced_scores[:, forced_token_id] = 0
        return forced_scores


def _tokenize_output_prefix(tokenizer: Any, prefix: str | None) -> list[int]:
    if not prefix:
        return []
    encoded = tokenizer(prefix, add_special_tokens=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _worker_seed(seed: int | None, gpu: int | None) -> int | None:
    if seed is None:
        return None
    # Keep per-GPU streams deterministic but disjoint for multi-process eval.
    return int(seed) + (0 if gpu is None else int(gpu) * 1_000_003)


def _seed_torch(seed: int | None, gpu: int | None = None) -> None:
    effective_seed = _worker_seed(seed, gpu)
    if effective_seed is None:
        return
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def _candidate_vote_signature(prediction_cxsmiles: str) -> tuple[str, str]:
    """Return the strict-graph canonical signature used for self-consistency voting."""

    if not prediction_cxsmiles:
        return "", "missing <cxsmi> payload"
    try:
        result = canonical_markush_signature(prediction_cxsmiles)
    except Exception as exc:  # pragma: no cover - defensive around RDKit corner cases.
        return "", str(exc)
    if not result.valid:
        return "", result.error or "invalid Markush graph"
    return result.signature, ""


def _transition_logprobs(
    model: Any, outputs: Any, *, prompt_len: int
) -> tuple[list[float | None], list[int | None]]:
    """Compute model log-likelihoods for generated sequences when HF exposes scores."""

    sequences = getattr(outputs, "sequences", outputs)
    scores = getattr(outputs, "scores", None)
    if scores is None or not hasattr(model, "compute_transition_scores"):
        n = int(sequences.shape[0]) if torch.is_tensor(sequences) else 0
        return [None] * n, [None] * n
    try:
        transition = model.compute_transition_scores(
            sequences,
            scores,
            getattr(outputs, "beam_indices", None),
            normalize_logits=True,
        )
    except TypeError:
        transition = model.compute_transition_scores(sequences, scores, normalize_logits=True)
    except Exception as exc:  # pragma: no cover - depends on HF model internals.
        LOGGER.warning("Could not compute generation transition scores: %s", exc)
        n = int(sequences.shape[0])
        return [None] * n, [None] * n

    # HF pads positions after EOS with exactly 0 transition scores.  Real token
    # log-probabilities are negative, so this mask is a practical generated-token
    # length estimate for tie-breaking only.
    active = transition != 0
    sums = transition.masked_fill(~active, 0).sum(dim=1).detach().cpu().tolist()
    counts = active.sum(dim=1).detach().cpu().tolist()
    return [float(value) for value in sums], [int(value) for value in counts]


def generate_batch_candidate_details(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    images: list[Image.Image] | list[list[Image.Image]],
    *,
    config: WorkerConfig | None = None,
    max_new_tokens: int | None = None,
    eos_token_ids: tuple[int, ...] = DEFAULT_EOS_TOKEN_IDS,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    num_return_sequences: int = 1,
    force_output_prefix: str | None = None,
    device: str,
    include_scores: bool | None = None,
) -> list[list[GeneratedCandidate]]:
    """Generate cleaned candidates and vote metadata for a same-sized batch."""

    if config is not None:
        max_new_tokens = config.max_new_tokens
        eos_token_ids = config.eos_token_ids
        num_beams = config.num_beams
        length_penalty = config.length_penalty
        do_sample = config.do_sample
        temperature = config.temperature
        top_p = config.top_p
        top_k = config.top_k
        num_return_sequences = config.num_return_sequences
        force_output_prefix = config.force_output_prefix
        if include_scores is None:
            include_scores = config.vote_logprobs and (
                config.majority_vote or config.num_return_sequences > 1
            )
    if max_new_tokens is None:
        raise ValueError("max_new_tokens is required when config is not provided")
    if num_return_sequences <= 0:
        raise ValueError("num_return_sequences must be > 0")

    with temporary_tokenizer_padding_side(processor, "left"):
        image_groups = _normalize_image_groups(images)
        if len(image_groups) != len(samples):
            raise ValueError(f"Expected {len(samples)} image group(s), got {len(image_groups)}")

        prompts = [
            build_prompt_text(processor, sample.prompt, num_images=len(image_group))
            for sample, image_group in zip(samples, image_groups, strict=True)
        ]
        flat_images = [image for image_group in image_groups for image in image_group]
        encoded = processor(
            text=prompts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )
        encoded = _move_tensors_to_device(dict(encoded), device)
        prompt_len = int(encoded["input_ids"].shape[1])

        tokenizer = getattr(processor, "tokenizer", processor)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        effective_eos = list(eos_token_ids) if eos_token_ids else []
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None and tokenizer_eos not in effective_eos:
            effective_eos.append(int(tokenizer_eos))

        generation_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "eos_token_id": effective_eos or tokenizer_eos,
            "use_cache": True,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "num_return_sequences": num_return_sequences,
        }
        if include_scores:
            generation_kwargs["return_dict_in_generate"] = True
            generation_kwargs["output_scores"] = True
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            if top_k is not None:
                generation_kwargs["top_k"] = top_k
        if num_beams > 1:
            generation_kwargs["early_stopping"] = True

        prefix_token_ids = _tokenize_output_prefix(tokenizer, force_output_prefix)
        if prefix_token_ids:
            from transformers import LogitsProcessorList

            generation_kwargs["logits_processor"] = LogitsProcessorList(
                [
                    ForceOutputPrefixLogitsProcessor(
                        prompt_len=prompt_len,
                        prefix_token_ids=prefix_token_ids,
                    )
                ]
            )

        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                **generation_kwargs,
            )

        output_ids = getattr(outputs, "sequences", outputs)
        generated = output_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        logprob_sums, token_counts = _transition_logprobs(model, outputs, prompt_len=prompt_len)

    cleaned = [common.clean_prediction(text) for text in decoded]
    expected = len(samples) * num_return_sequences
    if len(cleaned) != expected:
        raise RuntimeError(
            f"Expected {expected} generated sequence(s), got {len(cleaned)} "
            f"for batch size {len(samples)} and num_return_sequences={num_return_sequences}"
        )

    candidate_details: list[GeneratedCandidate] = []
    for idx, prediction in enumerate(cleaned):
        prediction_cxsmiles = common.prediction_to_cxsmiles_opt(prediction)
        signature, error = _candidate_vote_signature(prediction_cxsmiles)
        logprob_sum = logprob_sums[idx] if idx < len(logprob_sums) else None
        token_count = token_counts[idx] if idx < len(token_counts) else None
        mean_logprob = None
        if logprob_sum is not None and token_count:
            mean_logprob = float(logprob_sum) / max(1, int(token_count))
        candidate_details.append(
            GeneratedCandidate(
                prediction=prediction,
                prediction_cxsmiles=prediction_cxsmiles,
                logprob_sum=logprob_sum,
                token_count=token_count,
                mean_logprob=mean_logprob,
                canonical_signature=signature,
                valid_for_vote=bool(signature),
                vote_error=error,
            )
        )

    return [
        candidate_details[start : start + num_return_sequences]
        for start in range(0, len(candidate_details), num_return_sequences)
    ]


def generate_batch_candidates(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    images: list[Image.Image] | list[list[Image.Image]],
    *,
    config: WorkerConfig | None = None,
    max_new_tokens: int | None = None,
    eos_token_ids: tuple[int, ...] = DEFAULT_EOS_TOKEN_IDS,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    num_return_sequences: int = 1,
    force_output_prefix: str | None = None,
    device: str,
) -> list[list[str]]:
    """Generate cleaned candidate predictions for a same-sized batch.

    The returned outer list has one entry per input sample.  Each inner list
    contains ``num_return_sequences`` decoded candidates in the order returned
    by ``model.generate``.  ``generate_batch`` below preserves the historical
    top-prediction API for callers that do not need oracle/reranking data.
    """

    details = generate_batch_candidate_details(
        model,
        processor,
        samples,
        images,
        config=config,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        force_output_prefix=force_output_prefix,
        device=device,
    )
    return [[candidate.prediction for candidate in group] for group in details]


def generate_batch(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    images: list[Image.Image] | list[list[Image.Image]],
    *,
    config: WorkerConfig | None = None,
    max_new_tokens: int | None = None,
    eos_token_ids: tuple[int, ...] = DEFAULT_EOS_TOKEN_IDS,
    num_beams: int = 1,
    length_penalty: float = 1.0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    num_return_sequences: int = 1,
    force_output_prefix: str | None = None,
    device: str,
) -> list[str]:
    """Generate cleaned top predictions for a same-sized batch of samples/images."""
    candidate_groups = generate_batch_candidates(
        model,
        processor,
        samples,
        images,
        config=config,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        force_output_prefix=force_output_prefix,
        device=device,
    )
    return [candidates[0] if candidates else "" for candidates in candidate_groups]


def _empty_result(sample: EvalSample, *, gpu: int | None, error: str) -> PredictionResult:
    return PredictionResult(
        index=sample.index,
        sample_id=sample.sample_id,
        image_path=sample.image_path,
        ground_truth=sample.ground_truth,
        error=error,
        gpu=gpu,
    )


def _candidate_payload(candidate: GeneratedCandidate) -> dict[str, Any]:
    return asdict(candidate)


def select_majority_vote_candidate(
    candidates: list[GeneratedCandidate],
) -> tuple[GeneratedCandidate | None, dict[str, Any]]:
    """Select a modal strict-graph canonical form without using ground truth."""

    if not candidates:
        return None, {
            "invalid_count": 0,
            "invalid_rate": 0.0,
            "vote_count": 0,
            "tied": False,
            "all_invalid": True,
            "selection_source": "empty",
        }
    valid = [
        candidate
        for candidate in candidates
        if candidate.valid_for_vote and candidate.canonical_signature
    ]
    invalid_count = len(candidates) - len(valid)
    if not valid:
        # Non-oracle fallback: preserve the highest-likelihood sampled output if
        # scores are available, otherwise the first sampled output.
        fallback = max(
            candidates,
            key=lambda candidate: (
                candidate.mean_logprob if candidate.mean_logprob is not None else -float("inf"),
                candidate.logprob_sum if candidate.logprob_sum is not None else -float("inf"),
            ),
        )
        return fallback, {
            "invalid_count": invalid_count,
            "invalid_rate": invalid_count / len(candidates),
            "vote_count": 0,
            "tied": False,
            "all_invalid": True,
            "selection_source": "likelihood_fallback",
        }

    counts = Counter(candidate.canonical_signature for candidate in valid)
    max_votes = max(counts.values())
    tied_signatures = sorted(signature for signature, count in counts.items() if count == max_votes)
    by_signature: dict[str, list[GeneratedCandidate]] = defaultdict(list)
    for candidate in valid:
        by_signature[candidate.canonical_signature].append(candidate)

    def score_signature(signature: str) -> tuple[float, float, str]:
        group = by_signature[signature]
        mean_scores = [
            candidate.mean_logprob for candidate in group if candidate.mean_logprob is not None
        ]
        sum_scores = [
            candidate.logprob_sum for candidate in group if candidate.logprob_sum is not None
        ]
        mean_score = sum(mean_scores) / len(mean_scores) if mean_scores else -float("inf")
        best_sum = max(sum_scores) if sum_scores else -float("inf")
        return mean_score, best_sum, signature

    selected_signature = max(tied_signatures, key=score_signature)
    selected_group = by_signature[selected_signature]
    selected = max(
        selected_group,
        key=lambda candidate: (
            candidate.mean_logprob if candidate.mean_logprob is not None else -float("inf"),
            candidate.logprob_sum if candidate.logprob_sum is not None else -float("inf"),
        ),
    )
    return selected, {
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(candidates),
        "valid_count": len(valid),
        "vote_count": max_votes,
        "tied": len(tied_signatures) > 1,
        "tie_size": len(tied_signatures),
        "all_invalid": False,
        "selection_source": "majority_vote",
        "selected_canonical_signature": selected_signature,
        "selected_mean_logprob": selected.mean_logprob,
    }


def generate_majority_vote(
    model: Any,
    processor: Any,
    image_path: str,
    prompt: str,
    *,
    vote_k: int = MV_DEFAULT_VOTE_K,
    temperature: float = MV_DEFAULT_TEMPERATURE,
    top_p: float = MV_DEFAULT_TOP_P,
    max_new_tokens: int = common.DEFAULT_MAX_NEW_TOKENS,
    device: str = "cuda:0",
    seed: int | None = MV_DEFAULT_SEED,
    eos_token_ids: tuple[int, ...] = DEFAULT_EOS_TOKEN_IDS,
    multi_image_mode: str = "single",
    crop_overlap: float = 0.0,
) -> tuple[GeneratedCandidate | None, dict[str, Any], list[GeneratedCandidate]]:
    """Self-consistency (majority-vote) decode for a single image.

    Samples ``vote_k`` candidates with the canonical MV recipe, then returns the
    modal strict-graph consensus winner via :func:`select_majority_vote_candidate`
    — the same voting key the batched IP5-M MV@8 eval uses.  Reusing this path
    means MV@8 on one image reproduces the eval headline (532/878 = 60.59%)
    rather than re-implementing the voting logic.

    Returns the selected candidate (``None`` only for an empty batch), the vote
    metadata dict, and the full candidate list.
    """

    if vote_k <= 0:
        raise ValueError("vote_k must be > 0")

    _seed_torch(seed)
    image = load_image(image_path)
    views = build_image_views(image, mode=multi_image_mode, crop_overlap=crop_overlap)
    sample = EvalSample(
        index=0,
        sample_id=str(image_path),
        image_path=str(image_path),
        prompt=prompt,
        ground_truth="",
    )
    candidate_groups = generate_batch_candidate_details(
        model,
        processor,
        [sample],
        [views],
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=vote_k,
        device=device,
    )
    candidates = candidate_groups[0] if candidate_groups else []
    selected, vote_stats = select_majority_vote_candidate(candidates)
    vote_stats = {"n_candidates": len(candidates), **vote_stats}
    return selected, vote_stats, candidates


def _prediction_result(
    sample: EvalSample,
    candidates: list[GeneratedCandidate],
    *,
    gpu: int | None,
    majority_vote: bool = False,
) -> PredictionResult:
    if majority_vote:
        selected, vote_stats = select_majority_vote_candidate(candidates)
    else:
        selected = candidates[0] if candidates else None
        vote_stats = {"selection_source": "first_sequence"}
    prediction = selected.prediction if selected is not None else ""
    top_cxsmiles = selected.prediction_cxsmiles if selected is not None else ""
    candidate_payloads = [_candidate_payload(candidate) for candidate in candidates]
    if not candidate_payloads:
        candidate_payloads = [
            {
                "prediction": prediction,
                "prediction_cxsmiles": top_cxsmiles,
                "valid_for_vote": False,
                "vote_error": "empty candidate list",
            }
        ]
    return PredictionResult(
        index=sample.index,
        sample_id=sample.sample_id,
        image_path=sample.image_path,
        ground_truth=sample.ground_truth,
        prediction=prediction,
        prediction_cxsmiles=top_cxsmiles,
        candidates=candidate_payloads,
        majority_vote=vote_stats,
        gpu=gpu,
    )


def _chunks(items: list[EvalSample], size: int) -> list[list[EvalSample]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def run_worker_inference(
    samples: list[EvalSample],
    config: WorkerConfig,
    *,
    device: str,
    gpu: int | None,
) -> tuple[list[PredictionResult], dict[str, Any]]:
    """Run batched inference for one shard on one device."""
    _seed_torch(config.seed, gpu)
    start_time = time.perf_counter()
    model, processor = load_model_and_processor(config, device)
    load_seconds = time.perf_counter() - start_time

    results: list[PredictionResult] = []
    batches = _chunks(samples, config.batch_size)
    iterator = tqdm(
        batches,
        desc=f"GPU {gpu}" if gpu is not None else "Evaluating",
        disable=not config.show_progress,
    )

    generation_start = time.perf_counter()
    for batch_samples in iterator:
        valid_samples: list[EvalSample] = []
        image_groups: list[list[Image.Image]] = []

        for sample in batch_samples:
            try:
                image = load_image(sample.image_path)
                image_groups.append(
                    build_image_views(
                        image,
                        mode=config.multi_image_mode,
                        crop_overlap=config.crop_overlap,
                    )
                )
                valid_samples.append(sample)
            except Exception as exc:
                results.append(_empty_result(sample, gpu=gpu, error=f"image load failed: {exc}"))

        if not valid_samples:
            continue

        retry_prefix: str | None = None
        try:
            candidate_groups = generate_batch_candidate_details(
                model,
                processor,
                valid_samples,
                image_groups,
                config=config,
                device=device,
            )
            batch_results = [
                _prediction_result(
                    sample,
                    candidates,
                    gpu=gpu,
                    majority_vote=config.majority_vote,
                )
                for sample, candidates in zip(valid_samples, candidate_groups, strict=True)
            ]
            results.extend(batch_results)
        except torch.cuda.OutOfMemoryError as exc:
            retry_prefix = f"CUDA OOM at batch_size={len(valid_samples)}: {exc}"
            LOGGER.warning("%s; retrying batch one sample at a time", retry_prefix)
        except Exception as exc:
            retry_prefix = f"batch generation failed: {exc}"
            LOGGER.warning("%s; retrying batch one sample at a time", retry_prefix)

        if retry_prefix is not None:
            # Drop batch-local tensor/image references before retrying.  In particular,
            # OOM tracebacks can otherwise keep CUDA tensors alive through exception locals.
            image_groups.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(valid_samples) == 1:
                results.extend(_record_failed_batch(valid_samples, gpu=gpu, prefix=retry_prefix))
                continue
            results.extend(
                _retry_individual_generation(
                    model,
                    processor,
                    valid_samples,
                    config,
                    device=device,
                    gpu=gpu,
                    prefix=retry_prefix,
                )
            )

    generation_seconds = time.perf_counter() - generation_start
    total_seconds = time.perf_counter() - start_time
    results.sort(key=lambda result: result.index)
    stats = {
        "n_samples": len(samples),
        "n_results": len(results),
        "n_errors": sum(1 for result in results if result.error),
        "n_batches": len(batches),
        "batch_size": config.batch_size,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "samples_per_second": len(results) / generation_seconds if generation_seconds else None,
        "device": device,
        "gpu": gpu,
    }
    return results, stats


def _retry_individual_generation(
    model: Any,
    processor: Any,
    samples: list[EvalSample],
    config: WorkerConfig,
    *,
    device: str,
    gpu: int | None,
    prefix: str,
) -> list[PredictionResult]:
    """Recover per-sample errors after a failed batch."""
    results: list[PredictionResult] = []
    for sample in samples:
        try:
            image = load_image(sample.image_path)
            image_group = build_image_views(
                image,
                mode=config.multi_image_mode,
                crop_overlap=config.crop_overlap,
            )
            candidates = generate_batch_candidate_details(
                model,
                processor,
                [sample],
                [image_group],
                config=config,
                device=device,
            )[0]
            results.append(
                _prediction_result(
                    sample,
                    candidates,
                    gpu=gpu,
                    majority_vote=config.majority_vote,
                )
            )
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(
                _empty_result(sample, gpu=gpu, error=f"{prefix}; single retry failed: {exc}")
            )
    return results


def _record_failed_batch(
    samples: list[EvalSample],
    *,
    gpu: int | None,
    prefix: str,
) -> list[PredictionResult]:
    """Return error rows for a batch that could not be retried safely."""
    return [_empty_result(sample, gpu=gpu, error=prefix) for sample in samples]


# ---------------------------------------------------------------------------
# Multi-GPU orchestration
# ---------------------------------------------------------------------------


def split_contiguous(samples: list[EvalSample], num_shards: int) -> list[list[EvalSample]]:
    """Split samples into contiguous, near-equal shards."""
    if num_shards <= 1:
        return [samples]
    base, remainder = divmod(len(samples), num_shards)
    shards: list[list[EvalSample]] = []
    start = 0
    for shard_idx in range(num_shards):
        shard_size = base + (1 if shard_idx < remainder else 0)
        shards.append(samples[start : start + shard_size])
        start += shard_size
    return [shard for shard in shards if shard]


def _write_worker_payload(
    path: str | Path,
    *,
    rank: int,
    error: str | None,
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    """Atomically write a worker result payload.

    Multiprocessing queues can be fragile for large nested payloads after a
    worker exits.  The queue is used only as a notification channel; the actual
    rows live in a per-worker file.
    """
    payload_path = Path(path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = payload_path.with_suffix(f"{payload_path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "error": error,
                "rows": rows,
                "stats": stats,
            },
            default=str,
        ),
        encoding="utf-8",
    )
    tmp_path.replace(payload_path)


def _read_worker_payload(
    path: str | Path,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("error"), payload.get("rows", []), payload.get("stats", {})


def _gpu_worker_entry(
    rank: int,
    gpu_id: int,
    shard: list[EvalSample],
    config: WorkerConfig,
    result_queue: Any,
    result_path: str,
) -> None:
    """Multiprocessing entry point for one GPU shard."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [gpu {gpu_id}] %(message)s",
    )
    common.silence_dependency_logging()
    try:
        torch.cuda.set_device(gpu_id)
        results, stats = run_worker_inference(
            shard,
            config,
            device=f"cuda:{gpu_id}",
            gpu=gpu_id,
        )
        _write_worker_payload(
            result_path,
            rank=rank,
            error=None,
            rows=[asdict(result) for result in results],
            stats=stats,
        )
        result_queue.put(rank)
    except BaseException:  # pragma: no cover - exercised only in subprocess failures.
        _write_worker_payload(
            result_path,
            rank=rank,
            error=traceback.format_exc(),
            rows=[],
            stats={},
        )
        result_queue.put(rank)
        raise


def run_multi_gpu_inference(
    samples: list[EvalSample],
    config: WorkerConfig,
    *,
    num_gpus: int,
) -> tuple[list[PredictionResult], dict[str, Any]]:
    """Shard samples across GPUs and collect ordered predictions."""
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    shards = split_contiguous(samples, num_gpus)
    processes = []
    with tempfile.TemporaryDirectory(prefix="markush-fast-eval-") as result_dir:
        result_paths: dict[int, Path] = {}
        for rank, shard in enumerate(shards):
            gpu_id = rank
            result_path = Path(result_dir) / f"worker_{rank}.json"
            result_paths[rank] = result_path
            process = ctx.Process(
                target=_gpu_worker_entry,
                args=(rank, gpu_id, shard, config, result_queue, str(result_path)),
            )
            process.start()
            processes.append(process)
            LOGGER.info(
                "Started GPU %d worker with %d samples (%d-%d)",
                gpu_id,
                len(shard),
                shard[0].index,
                shard[-1].index,
            )

        received: dict[int, tuple[str | None, list[dict[str, Any]], dict[str, Any]]] = {}

        def collect_payload(rank: int) -> None:
            if rank in received:
                return
            result_path = result_paths[rank]
            if result_path.exists():
                received[rank] = _read_worker_payload(result_path)

        while len(received) < len(processes):
            try:
                rank = int(result_queue.get(timeout=5.0))
                collect_payload(rank)
            except queue_mod.Empty:
                for rank in range(len(processes)):
                    collect_payload(rank)

            for rank, process in enumerate(processes):
                if rank in received:
                    continue
                if not process.is_alive() and process.exitcode is not None:
                    collect_payload(rank)
                if not process.is_alive() and process.exitcode is not None and rank not in received:
                    raise RuntimeError(
                        f"GPU worker {rank} exited with code {process.exitcode} "
                        "before writing results"
                    ) from None

        for process in processes:
            process.join()

        errors = {rank: payload[0] for rank, payload in received.items() if payload[0]}
        if errors:
            formatted = "\n\n".join(f"Worker {rank}:\n{error}" for rank, error in errors.items())
            raise RuntimeError(f"One or more GPU workers failed:\n{formatted}")

        all_results: list[PredictionResult] = []
        worker_stats: dict[str, Any] = {}
        for rank in sorted(received):
            _, rows, stats = received[rank]
            all_results.extend(PredictionResult(**row) for row in rows)
            worker_stats[str(rank)] = stats
    all_results.sort(key=lambda result: result.index)
    stats = {
        "workers": worker_stats,
        "n_samples": len(samples),
        "n_results": len(all_results),
        "n_errors": sum(1 for result in all_results if result.error),
    }
    return all_results, stats


# ---------------------------------------------------------------------------
# MCS evaluation and output
# ---------------------------------------------------------------------------


def run_mcs_eval(
    results: list[PredictionResult],
    *,
    protocol: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any] | None]]:
    """Run the same high-level Markush evaluator used by inline SFT metrics."""
    from glyph.markush.eval.metrics_mcs import evaluate_markush_mcs

    predictions = [
        {
            "id": result.sample_id,
            "prediction": result.prediction,
            "cxsmiles_opt": result.prediction_cxsmiles,
        }
        for result in results
    ]
    ground_truths = [
        {
            "id": result.sample_id,
            "cxsmiles_opt": result.ground_truth,
        }
        for result in results
    ]
    aggregate, detail_rows = evaluate_markush_mcs(
        predictions,
        ground_truths,
        include_missing=True,
        return_details=True,
        protocol=protocol,
    )
    detail_by_id = {str(row.get("id", "")): row for row in detail_rows}
    details = {index: detail_by_id.get(result.sample_id) for index, result in enumerate(results)}
    return aggregate, details


def print_summary(aggregate: dict[str, Any], *, protocol: str, timing: dict[str, Any]) -> None:
    """Print compact metrics and throughput information to stdout."""
    print("\n" + "=" * 72)
    print(f"FAST MARKUSH EVAL — MCS protocol: {protocol}")
    print("=" * 72)
    preferred_order = [
        "size",
        "valid",
        "tanimoto",
        "tanimoto_similarity",
        "inchi_equality",
        "backbone_inchi_match",
        "string_equality",
        "cxsmi_equality",
        "cxsmiles_full_match",
        "r",
        "m",
        "sg",
        "invalid_gt",
    ]
    for key in preferred_order:
        if key in aggregate:
            print(f"  {key:24s} {aggregate[key]}")
    for key in sorted(key for key in aggregate if key not in preferred_order):
        print(f"  {key:24s} {aggregate[key]}")
    print("-" * 72)
    print(f"  generation_seconds       {timing.get('generation_seconds'):.2f}")
    print(f"  mcs_seconds              {timing.get('mcs_seconds'):.2f}")
    print(f"  total_seconds            {timing.get('total_seconds'):.2f}")
    if timing.get("samples_per_second") is not None:
        print(f"  samples_per_second       {timing['samples_per_second']:.2f}")
    print("=" * 72)


def save_results(
    output_path: str | Path,
    *,
    args: argparse.Namespace,
    results: list[PredictionResult],
    aggregate: dict[str, Any],
    details: dict[int, dict[str, Any] | None],
    timing: dict[str, Any],
    effective_dtype: str,
    inference_stats: dict[str, Any],
) -> None:
    """Write aggregate metrics and per-sample predictions/scores to JSON."""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    samples_payload = []
    for eval_idx, result in enumerate(results):
        row = asdict(result)
        row["mcs_score"] = details.get(eval_idx)
        samples_payload.append(row)

    majority_vote_stats: dict[str, Any] = {}
    if getattr(args, "majority_vote", False):
        vote_rows = [result.majority_vote for result in results]
        total_candidates = sum(len(result.candidates) for result in results)
        invalid_candidates = sum(int(row.get("invalid_count", 0)) for row in vote_rows)
        majority_vote_stats = {
            "sample_count": len(results),
            "candidate_count": total_candidates,
            "invalid_candidate_count": invalid_candidates,
            "invalid_candidate_rate": invalid_candidates / total_candidates
            if total_candidates
            else 0.0,
            "tie_count": sum(1 for row in vote_rows if row.get("tied")),
            "tie_rate": sum(1 for row in vote_rows if row.get("tied")) / len(vote_rows)
            if vote_rows
            else 0.0,
            "all_invalid_count": sum(1 for row in vote_rows if row.get("all_invalid")),
        }

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "base_model": args.base_model,
            "eval_data": args.eval_data,
            "image_root": args.image_root,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_return_sequences": args.num_return_sequences,
            "majority_vote": args.majority_vote,
            "seed": args.seed,
            "force_output_prefix": args.force_output_prefix,
            "num_gpus": args.num_gpus,
            "dtype": effective_dtype,
            "requested_dtype": args.dtype,
            "protocol": args.protocol,
            "skip_mcs": args.skip_mcs,
            "eos_token_ids": list(parse_eos_token_ids(args.eos_token_ids)),
            "merge_adapter": not args.no_merge,
            "multi_image_mode": args.multi_image_mode,
            "crop_overlap": args.crop_overlap,
            "image_min_pixels": args.image_min_pixels,
            "image_max_pixels": args.image_max_pixels,
        },
        "aggregate": aggregate,
        "majority_vote_stats": majority_vote_stats,
        "timing": timing,
        "inference_stats": inference_stats,
        "samples": samples_payload,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    LOGGER.info("Saved results to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_eos_token_ids(value: str | None) -> tuple[int, ...]:
    """Parse comma/space separated EOS token ids from CLI."""
    if value is None:
        return DEFAULT_EOS_TOKEN_IDS
    tokens = [part.strip() for part in value.replace(" ", ",").split(",")]
    return tuple(int(part) for part in tokens if part)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast batched Qwen-VL inference + MG2 MCS evaluation for Markush recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "LoRA adapter checkpoint. Security: loaded with trust_remote_code=True / "
            "torch unpickling and may execute code from the referenced repo — only "
            "use trusted sources."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help=(
            "Base Qwen-VL model path. Security: loaded with trust_remote_code=True and "
            "may execute code from the referenced repo — only use trusted sources."
        ),
    )
    parser.add_argument("--eval-data", type=str, required=True, help="Evaluation JSONL")
    parser.add_argument(
        "--image-root", type=str, default=None, help="Root for relative image paths"
    )
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU generation batch size")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=common.DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam count for deterministic generation.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty passed to generate(); mainly useful with --num-beams > 1.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use stochastic decoding instead of greedy/beam decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature; used only with --do-sample.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold; used only with --do-sample.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff; used only with --do-sample. Omit to use model default.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help=(
            "Number of decoded candidates to return per input. For oracle@N, "
            "use this with --do-sample and a reduced batch size if needed."
        ),
    )
    parser.add_argument(
        "--majority-vote",
        action="store_true",
        help=(
            "Select the modal strict-graph canonical form among returned candidates. "
            "Requires --do-sample and --num-return-sequences > 1 for self-consistency eval."
        ),
    )
    parser.add_argument(
        "--vote-logprobs",
        action="store_true",
        help=(
            "Compute per-token transition log-probs for majority-vote tiebreaking. "
            "OFF by default because output_scores=True OOMs at MV@8 with long "
            "sequences (~60GB alloc); ties otherwise fall back to the first valid candidate."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic sampling; multi-GPU workers get disjoint offsets.",
    )
    parser.add_argument(
        "--force-output-prefix",
        type=str,
        default=None,
        help="Literal prefix to force at the start of each generated answer.",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs to use; 0 means all"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for single-worker mode"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32", "auto"),
        help="Model dtype",
    )
    parser.add_argument("--no-merge", action="store_true", help="Do not merge the LoRA adapter")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Instruction prompt")
    parser.add_argument(
        "--ignore-row-prompts",
        action="store_true",
        help=(
            "Use --prompt instead of prompt/messages fields in eval rows. Intended for "
            "diagnostic prompt ablations; official evals should preserve prepared prompts."
        ),
    )
    parser.add_argument(
        "--markush-prompt-policy",
        choices=MARKUSH_PROMPT_POLICIES,
        default=MARKUSH_PROMPT_POLICY_DEFAULT,
        help=(
            "Optional Markush prompt wording policy applied after target-format "
            "construction. Default preserves prepared/offline prompts."
        ),
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
        "--protocol",
        type=str,
        default="mg2",
        choices=("mg2", "ours", "strict"),
        help=(
            "MCS scoring protocol. 'mg2' is an APPROXIMATE in-package MG2 "
            "replication (not byte-exact; official numbers require the pinned "
            "DS4SD scorer in tools/mg2_scorer/); 'ours'/'strict' is our stricter eval."
        ),
    )
    parser.add_argument(
        "--eos-token-ids",
        type=str,
        default=",".join(str(token_id) for token_id in DEFAULT_EOS_TOKEN_IDS),
        help="Comma-separated EOS token ids passed to generate()",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--multi-image-mode",
        choices=("single", "global_2x2", "global_3x3"),
        default="single",
        help=(
            "Eval-time multi-view mode. global_2x2 sends the full image plus "
            "four overlapping quadrant crops; global_3x3 sends the full image "
            "plus nine overlapping crops."
        ),
    )
    parser.add_argument(
        "--crop-overlap",
        type=float,
        default=0.12,
        help="Fractional overlap added to each tile crop in multi-image modes.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=None,
        help="Override processor minimum image pixels for eval.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=None,
        help="Override processor maximum image pixels for eval.",
    )
    parser.add_argument(
        "--skip-mcs",
        action="store_true",
        help="Skip MCS scoring and save generated predictions only.",
    )
    return parser


def _effective_num_gpus(requested: int) -> int:
    if requested < 0:
        raise ValueError("--num-gpus must be >= 0")
    if not torch.cuda.is_available():
        if requested not in (0, 1):
            LOGGER.warning("CUDA is not available; falling back to single-process CPU evaluation")
        return 0
    available = torch.cuda.device_count()
    if requested == 0:
        return available
    if requested > available:
        LOGGER.warning(
            "Requested %d GPUs but only %d are available; using %d", requested, available, available
        )
        return available
    return requested


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.num_beams <= 0:
        parser.error("--num-beams must be > 0")
    if args.length_penalty <= 0:
        parser.error("--length-penalty must be > 0")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be > 0 when provided")
    if args.num_return_sequences <= 0:
        parser.error("--num-return-sequences must be > 0")
    if args.majority_vote and (not args.do_sample or args.num_return_sequences <= 1):
        parser.error("--majority-vote requires --do-sample and --num-return-sequences > 1")
    if (
        args.num_return_sequences > 1
        and not args.do_sample
        and args.num_beams < args.num_return_sequences
    ):
        parser.error(
            "--num-return-sequences > 1 requires --do-sample or "
            "--num-beams >= --num-return-sequences"
        )
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")
    if not 0 <= args.crop_overlap < 0.5:
        parser.error("--crop-overlap must be in [0, 0.5)")
    if args.multi_image_mode != "single" and args.batch_size != 1:
        LOGGER.warning(
            "Multi-image mode is active with batch size %d; consider --batch-size 1 "
            "to reduce memory pressure.",
            args.batch_size,
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    common.silence_dependency_logging()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

    samples = load_eval_samples(
        args.eval_data,
        args.image_root,
        args.prompt,
        limit=args.limit,
        canonicalize_markush_targets=args.canonicalize_markush_targets,
        require_canonical_markush_targets=args.require_canonical_markush_targets,
        ignore_row_prompts=args.ignore_row_prompts,
        markush_prompt_policy=args.markush_prompt_policy,
    )
    if not samples:
        LOGGER.error("No evaluation samples loaded from %s", args.eval_data)
        sys.exit(1)
    LOGGER.info("Loaded %d evaluation samples", len(samples))

    eos_token_ids = parse_eos_token_ids(args.eos_token_ids)
    num_gpus = _effective_num_gpus(args.num_gpus)
    if num_gpus <= 1:
        device = args.device if torch.cuda.is_available() else "cpu"
        gpu = None
        if device.startswith("cuda:"):
            gpu = int(device.split(":", 1)[1])
    else:
        device = None
        gpu = None

    effective_dtype = args.dtype
    using_cpu = num_gpus <= 1 and device == "cpu"
    if using_cpu and effective_dtype in {"bfloat16", "float16"}:
        LOGGER.warning(
            "Using CPU inference; overriding --dtype %s to float32",
            effective_dtype,
        )
        effective_dtype = "float32"
    config = WorkerConfig(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dtype=effective_dtype,
        merge_adapter=not args.no_merge,
        eos_token_ids=eos_token_ids,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_return_sequences=args.num_return_sequences,
        majority_vote=args.majority_vote,
        vote_logprobs=args.vote_logprobs,
        seed=args.seed,
        force_output_prefix=args.force_output_prefix,
        multi_image_mode=args.multi_image_mode,
        crop_overlap=args.crop_overlap,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        show_progress=not args.quiet,
    )

    total_start = time.perf_counter()
    generation_start = time.perf_counter()
    if num_gpus <= 1:
        results, inference_stats = run_worker_inference(samples, config, device=device, gpu=gpu)
    else:
        results, inference_stats = run_multi_gpu_inference(samples, config, num_gpus=num_gpus)
    generation_seconds = time.perf_counter() - generation_start

    if len(results) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} results, got {len(results)}")
    results.sort(key=lambda result: result.index)

    mcs_start = time.perf_counter()
    if args.skip_mcs:
        aggregate = {"size": len(results), "skipped_mcs": True}
        details = {}
        mcs_seconds = 0.0
    else:
        aggregate, details = run_mcs_eval(results, protocol=args.protocol)
        mcs_seconds = time.perf_counter() - mcs_start
    total_seconds = time.perf_counter() - total_start
    timing = {
        "generation_seconds": generation_seconds,
        "mcs_seconds": mcs_seconds,
        "total_seconds": total_seconds,
        "samples_per_second": len(results) / generation_seconds if generation_seconds else None,
    }

    print_summary(aggregate, protocol=args.protocol, timing=timing)
    save_results(
        args.output,
        args=args,
        results=results,
        aggregate=aggregate,
        details=details,
        timing=timing,
        effective_dtype=effective_dtype,
        inference_stats=inference_stats,
    )


if __name__ == "__main__":
    main()
