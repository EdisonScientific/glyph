#!/usr/bin/env python3
"""Supervised fine-tuning for Markush structure recognition.

This script trains a Qwen3.5-VL model to extract Markush structures from patent
images using LoRA adapters on the language model and (optionally) the vision
encoder.

Vision encoder LoRA (``--vision-lora``) closes the domain gap between synthetic
training images and real patent images by letting the ViT adapt its features
with a low-rank update.  It is off by default so existing configs work
unchanged; enable it with ``--vision-lora`` or ``vision_lora: true`` in YAML.

Format (chat template applied by the collator)::

    <|im_start|>user
    <|vision_start|><|image_pad|>...<|vision_end|>{user_text}<|im_end|>
    <|im_start|>assistant
    {target}<|im_end|>

The processor handles vision token expansion.  Labels mask the user turn so
loss is computed only on the assistant response.

Features:
    - Config-driven: Load from YAML or command-line args
    - Text dropout: Randomly drop OCR/patent text to force image learning
    - LoRA with Qwen3.5 hybrid architecture targets (language + optional vision)
    - Proper EOS tokens: both <|endoftext|> and <|im_end|> for generation
    - W&B integration with detailed metrics logging

Usage::

    # From command line with args
    python -m glyph.markush.train.sft \\
        --model-path Qwen/Qwen3.5-2B-Base \\
        --data-path hub:markush/train.jsonl \\
        --output-dir checkpoints/sft_2b \\
        --text-dropout 0.5

    # From config file
    python -m glyph.markush.train.sft --config configs/ablations/ocsr-as-markush-fraction/sft_2b_markush_ocsr5pct.yaml

    # Multi-GPU with accelerate
    accelerate launch -m glyph.markush.train.sft --config configs/ablations/ocsr-as-markush-fraction/sft_2b_markush_ocsr5pct.yaml

    # Enable vision encoder LoRA
    accelerate launch -m glyph.markush.train.sft \\
        --config configs/ablations/ocsr-as-markush-fraction/sft_2b_markush_ocsr5pct.yaml \\
        --vision-lora --vision-lora-r 32

    # Disable vision encoder LoRA explicitly
    accelerate launch -m glyph.markush.train.sft --no-vision-lora --config configs/ablations/ocsr-as-markush-fraction/sft_2b_markush_ocsr5pct.yaml

Example config (configs/ablations/ocsr-as-markush-fraction/sft_2b_markush_ocsr5pct.yaml)::

    model_path: Qwen/Qwen3.5-2B-Base
    data_path: hub:markush/train.jsonl
    output_dir: checkpoints/sft_2b
    text_dropout: 0.5
    image_augmentation: true
    image_augmentation_prob: 0.6
    lora_r: 128
    lora_alpha: 128
    learning_rate: 2e-4
    warmup_steps: 200
    max_steps: 30000
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 2
    save_steps: 1000
    eval_steps: 2000
    logging_steps: 5
    bf16: true
    # Optional: vision encoder LoRA
    vision_lora: true
    vision_lora_r: 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from glyph.markush.data.example_modes import MOLECULE_OCSR_PROMPT, PATENT_IMAGE_MARKUSH_PROMPT
from glyph.markush.data.ocsr import canonicalize_ocsr_smiles_checked, row_declares_ocsr_smiles
from glyph.markush.data.prepare_sft import prompt_with_target_instruction, target_spec_for_record
from glyph.markush.data.prompt_policies import (
    MARKUSH_PROMPT_POLICIES,
    MARKUSH_PROMPT_POLICY_DEFAULT,
    apply_markush_prompt_policy,
)
from glyph.markush.eval import common
from glyph.markush.eval.benchmarks import benchmark_eval_jsonl, get_benchmark_spec
from glyph.markush.eval.ocsr import OCSR_PREDICTION_MODE_NATIVE_SMILES
from glyph.paths import data_root, expand_config_vars

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    _QWEN_VL_UTILS_IMPORT_ERROR = exc

    def process_vision_info(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        raise ImportError(
            "qwen-vl-utils and its vision dependencies are required to process SFT image inputs."
        ) from _QWEN_VL_UTILS_IMPORT_ERROR


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

__all__ = [
    "RESOLVED_CONFIG_FILENAME",
    "DetailedLoggingCallback",
    "MarkushCollator",
    "MarkushDataset",
    "MetricsEvalCallback",
    "MultiMetricsEvalCallback",
    "OCSRMetricsEvalCallback",
    "OCSRMetricsEvalSpec",
    "ResolvedConfigCallback",
    "SFTConfig",
    "SaveProcessorCallback",
    "_load_ocsr_metrics_samples",
    "_require_ocsr_metric_samples",
    "_resolve_metrics_benchmark_path",
    "_resolve_model_identifier",
    "_resolve_ocsr_metrics_eval_specs",
    "compute_inline_eval_aggregate_metrics",
    "detect_silent_config_overrides",
    "explicit_cli_keys",
    "install_processor_save_callback_first",
    "main",
    "persist_resolved_config",
    "resolve_config",
    "train",
    "validate_best_metric_config",
]

LOGGER = logging.getLogger(__name__)

# The delimiter that separates context from target
PROMPT_DELIMITER = "Extract the Markush structure:\n"

# Fallback instruction when text is dropped
FALLBACK_INSTRUCTION = PATENT_IMAGE_MARKUSH_PROMPT

# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------

# Language model targets for Qwen3.5 hybrid architecture (DeltaNet + standard attention).
# Qwen3.5 uses a hybrid architecture where some layers are standard Transformer attention
# and others are DeltaNet linear attention. The DeltaNet layers have different projections
# (in_proj_*, out_proj) than standard attention (q/k/v/o_proj). PEFT automatically skips
# modules that don't exist in a given layer, so having both sets is safe.
LM_LORA_TARGET_MODULES = [
    # Standard attention projections
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # MLP projections (dense and MoE layers)
    "gate_proj",
    "up_proj",
    "down_proj",
    # DeltaNet layers (Qwen3.5 hybrid-specific linear attention)
    "out_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
]

# Vision encoder targets for Qwen3.5 ViT (~331M params).
# These are the linear layers inside each ViT block's attention and MLP.
# Module paths: model.visual.blocks.{i}.attn.{qkv,proj}
#               model.visual.blocks.{i}.mlp.{linear_fc1,linear_fc2}
VISION_LORA_TARGET_MODULES = [
    "qkv",  # Combined Q/K/V projection in ViT attention
    "proj",  # Attention output projection
    "linear_fc1",  # MLP up-projection (first layer)
    "linear_fc2",  # MLP down-projection (second layer)
]

# Token IDs for generation termination
# Qwen models use both <|endoftext|> (248044) and <|im_end|> (248046)
ENDOFTEXT_TOKEN_ID = 248044
IM_END_TOKEN_ID = 248046


def _is_hf_model_id(value: str) -> bool:
    """Return whether a model reference should pass through to HF caching."""

    if value.startswith(("/", "./", "../", "~")):
        return False
    if value.startswith(("models/", "checkpoints/")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts)


def _resolve_model_identifier(model_path: str | Path, root: Path) -> str | Path:
    """Resolve local model paths while preserving Hugging Face model ids.

    Existing configs historically used repo-relative paths such as
    ``models/Qwen3.5-2B-Base``.  Cluster runs may instead pass a normal HF id
    such as ``org/model-name`` and rely on ``HF_HOME`` for the large cached
    blobs.  This helper keeps both behaviours explicit.
    """

    raw = str(model_path)
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded
    repo_path = root / expanded
    if repo_path.exists():
        return repo_path
    if _is_hf_model_id(raw):
        return raw
    return repo_path


@dataclass(frozen=True)
class OCSRMetricsEvalSpec:
    """Resolved OCSR metrics eval artifact and metric prefix name."""

    name: str
    path: Path

    @property
    def prefix(self) -> str:
        return f"eval_ocsr_{self.name.replace('-', '_')}"


@dataclass
class SFTConfig:
    """Configuration for SFT training with optional vision encoder LoRA."""

    # Model and data paths
    model_path: str = "Qwen/Qwen3.5-2B-Base"
    data_path: str = "hub:markush/train.jsonl"
    output_dir: str = "checkpoints/sft"
    logging_dir: str | None = None
    init_adapter_path: str | None = None

    # Data processing
    text_dropout: float = 0.5
    eval_size: int = 1000
    image_augmentation: bool = False
    image_augmentation_prob: float = 0.6

    # LoRA configuration (language model)
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Vision encoder LoRA
    vision_lora: bool = False
    vision_lora_r: int = 32
    vision_lora_alpha: int | None = None  # defaults to vision_lora_r

    # Adaptation mode. When true, train all model weights instead of creating a
    # PEFT LoRA adapter. If init_adapter_path is also set, that adapter is
    # loaded and merged into the base model before full fine-tuning begins.
    full_finetune: bool = False

    # Training hyperparameters
    learning_rate: float = 2e-4
    warmup_steps: int = 200
    max_steps: int = 30000
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 2000
    save_total_limit: int = 5

    # Logging
    logging_steps: int = 5
    report_to: list[str] = field(default_factory=lambda: ["wandb"])
    wandb_project: str = "glyph-markush"
    wandb_entity: str | None = None
    wandb_mode: str | None = None
    run_name: str | None = None

    # Precision and optimization
    bf16: bool = True
    optim: str = "adamw_torch_fused"

    # Data loading
    dataloader_num_workers: int = 4
    # Batches prefetched per worker (None keeps the PyTorch default of 2).
    # The 104-core training nodes can sustain more to smooth step-time spikes.
    dataloader_prefetch_factor: int | None = None

    # Text-length outlier filtering. Sequence peak memory scales with the
    # longest example in a batch (the fp32/bf16 CE logits are
    # [batch, seq, 248k vocab]); a handful of pathological rows (0.006% of the
    # public train set exceeds 4000 assistant chars, max 21k) otherwise
    # dictate the whole run's memory ceiling. Dropped rows are counted and
    # logged. None or 0 disables a bound.
    max_user_chars: int | None = None
    max_assistant_chars: int | None = 4000

    # Compute the LM loss with FLA's fused Triton cross-entropy, which never
    # materializes the fp32 copy of the [batch, seq, vocab] logits tensor
    # (the allocation that OOM'd bs>=4 runs). Falls back to the standard
    # transformers loss with a warning when fla is not importable.
    fused_cross_entropy: bool = True

    # Random seed
    seed: int = 42

    # Disable W&B
    no_wandb: bool = False

    # Chemistry-aware metrics during training (generation-based MCS eval on a
    # subset). When metrics_eval_size > 0, a MetricsEvalCallback runs on every
    # eval_steps and logs both `eval/ours/...` and `eval/mg2/...` to W&B.
    metrics_eval_size: int = 0  # 0 disables the callback
    metrics_eval_data: str | None = (
        None  # external eval JSONL (e.g. IP5-M); overrides train holdout
    )
    metrics_eval_benchmark: str | None = None  # prompt preset for metrics_eval_data
    metrics_eval_benchmarks: list[str] = field(default_factory=list)
    metrics_eval_data_dir: str | None = None  # directory containing eval_ip5m.jsonl, etc.
    metrics_eval_protocol: str = "both"  # "ours" | "mg2" | "both"
    metrics_eval_max_new_tokens: int = 512
    # Per-rank generation batch size for the inline eval. Defaults to 1 for
    # memory safety during training; raise after a small cluster validation run.
    metrics_eval_batch_size: int = 1
    # Optional coarser cadence than eval_steps. When unset, the inline eval runs
    # on every Trainer eval (i.e. every eval_steps).
    metrics_eval_steps: int | None = None
    metrics_eval_canonicalize_markush_targets: bool = False
    metrics_eval_require_canonical_markush_targets: bool = False
    metrics_eval_markush_prompt_policy: str = MARKUSH_PROMPT_POLICY_DEFAULT

    # OCSR bare-SMILES metrics during training. These callbacks use the same
    # generation path as the Markush metrics callback but score outputs with the
    # native OCSR RDKit-canonical SMILES contract.
    ocsr_metrics_eval_size: int = 0
    ocsr_metrics_eval_data: str | None = None
    ocsr_metrics_eval_name: str = "ocsr"
    ocsr_metrics_eval_max_new_tokens: int = 256
    ocsr_metrics_eval_batch_size: int | None = None
    ocsr_metrics_eval_steps: int | None = None
    ocsr_metrics_prediction_mode: str = OCSR_PREDICTION_MODE_NATIVE_SMILES

    # Best-checkpoint selection. When `load_best_model_at_end` is true and
    # `metric_for_best_model` is set (e.g. "eval_ours_cxsmi_equality"), the
    # trainer auto-loads the checkpoint with the best score on that metric.
    # MetricsEvalCallback writes underscore-flat keys into the metrics dict
    # specifically so this works alongside W&B's slashed keys.
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        """Load configuration from a YAML file."""
        from glyph.markush.train import coerce_yaml_values

        with open(path) as f:
            data = expand_config_vars(yaml.safe_load(f) or {})
        normalized = {k.replace("-", "_"): v for k, v in data.items()}
        coerced = coerce_yaml_values(normalized, cls)
        return cls(
            **{k: v for k, v in coerced.items() if hasattr(cls, k) or k in cls.__dataclass_fields__}
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> SFTConfig:
        """Create config from parsed command-line arguments."""
        config = cls()
        for k, v in vars(args).items():
            if v is not None and hasattr(config, k):
                setattr(config, k, v)
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}


BEST_MODEL_METRIC_BASE_KEYS = {
    "cxsmi_equality",
    "cxsmiles_full_match",
    "backbone_inchi_match",
    "r_group_accuracy",
    "m_section_match",
    "sg_section_match",
    "stable_match",
    "stable_equality_all",
    "num_fragments_equal",
    "tanimoto_similarity",
    "valid_rate",
}

OCSR_BEST_MODEL_METRIC_BASE_KEYS = {
    "canon_exact_match",
    "graph_exact_match",
    "valid_smiles_rate",
}

INLINE_EVAL_MARKUSH_DATASETS = ("ip5m", "m2s", "uspto_markush")


def _numeric_metric_value(metrics: dict[str, Any], key: str) -> float | None:
    """Return a numeric metric for a slashed key, accepting HF's flat alias."""

    for candidate in (key, key.replace("/", "_")):
        value = metrics.get(candidate)
        if isinstance(value, int | float):
            return float(value)
    return None


def _mean_metric_values(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def compute_inline_eval_aggregate_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Compute family-level inline eval aggregates from per-benchmark metrics."""

    aggregate: dict[str, float] = {}
    markush_protocol_scores: dict[str, list[float]] = {"ours": [], "mg2": []}
    for dataset in INLINE_EVAL_MARKUSH_DATASETS:
        dataset_scores: list[float] = []
        for protocol in ("ours", "mg2"):
            value = _numeric_metric_value(
                metrics,
                f"eval_{dataset}/{protocol}/cxsmi_equality",
            )
            if value is None:
                continue
            dataset_scores.append(value)
            markush_protocol_scores[protocol].append(value)
        dataset_avg = _mean_metric_values(dataset_scores)
        if dataset_avg is not None:
            aggregate[f"eval_{dataset}/avg"] = dataset_avg

    markush_overall_scores: list[float] = []
    for protocol in ("ours", "mg2"):
        protocol_avg = _mean_metric_values(markush_protocol_scores[protocol])
        if protocol_avg is None:
            continue
        aggregate[f"markush/{protocol}"] = protocol_avg
        markush_overall_scores.append(protocol_avg)
    markush_overall = _mean_metric_values(markush_overall_scores)
    if markush_overall is not None:
        aggregate["markush/overall"] = markush_overall

    ocsr_scores_by_prefix: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, int | float):
            continue
        if key.endswith("/canon_exact_match") and key.startswith("eval_ocsr_"):
            prefix = key.removesuffix("/canon_exact_match")
            ocsr_scores_by_prefix[prefix] = float(value)
            continue
        if key.startswith("eval_ocsr_") and key.endswith("_canon_exact_match"):
            prefix = key.removesuffix("_canon_exact_match")
            ocsr_scores_by_prefix[prefix] = float(value)

    ocsr_overall = _mean_metric_values(list(ocsr_scores_by_prefix.values()))
    if ocsr_overall is not None:
        aggregate["ocsr/overall"] = ocsr_overall

    combined_scores = [
        value
        for value in (
            aggregate.get("markush/overall"),
            aggregate.get("ocsr/overall"),
        )
        if value is not None
    ]
    combined = _mean_metric_values(combined_scores)
    if combined is not None:
        aggregate["overall/combined"] = combined
    return aggregate


def _metrics_eval_protocols(protocol: str) -> list[str]:
    if protocol == "both":
        return ["ours", "mg2"]
    if protocol in {"ours", "mg2"}:
        return [protocol]
    raise ValueError("metrics_eval_protocol must be one of: ours, mg2, both")


def _ocsr_eval_name_from_path(path: Path) -> str:
    name = path.stem
    if name.startswith("eval_ocsr_"):
        name = name.removeprefix("eval_ocsr_")
    return (name or "ocsr").replace("-", "_")


def _resolve_ocsr_metrics_eval_specs(
    config: SFTConfig,
    *,
    root: Path,
    require_exists: bool = False,
) -> list[OCSRMetricsEvalSpec]:
    """Resolve one or more OCSR metrics JSONLs from config.

    ``ocsr_metrics_eval_data`` accepts either the historical single JSONL or a
    directory containing unified ``eval_ocsr_*.jsonl`` artifacts. Directory mode
    registers one metrics callback per benchmark so per-source OCSR metrics stay
    visible in W&B and best-checkpoint keys.
    """

    if not config.ocsr_metrics_eval_data:
        return []
    data_path = Path(config.ocsr_metrics_eval_data).expanduser()
    if not data_path.is_absolute():
        data_path = root / data_path
    if data_path.is_dir():
        paths = sorted(data_path.glob("eval_ocsr_*.jsonl"))
        if require_exists and not paths:
            raise ValueError(
                f"OCSR metrics eval directory has no eval_ocsr_*.jsonl files: {data_path}"
            )
        return [
            OCSRMetricsEvalSpec(name=_ocsr_eval_name_from_path(path), path=path) for path in paths
        ]
    if require_exists and not data_path.exists():
        raise FileNotFoundError(data_path)
    name = config.ocsr_metrics_eval_name or _ocsr_eval_name_from_path(data_path)
    return [OCSRMetricsEvalSpec(name=name, path=data_path)]


def expected_best_metric_keys(config: SFTConfig, *, root: Path | None = None) -> set[str]:
    """Return callback metric keys available for best-checkpoint selection."""

    keys: set[str] = set()
    has_markush_aggregate = False
    has_ocsr_aggregate = False
    if config.metrics_eval_size > 0:
        prefixes: list[str] = []
        if config.metrics_eval_benchmarks:
            for name in config.metrics_eval_benchmarks:
                spec = get_benchmark_spec(name)
                prefixes.append(f"eval_{spec.name.replace('-', '_')}")
        elif config.metrics_eval_data:
            if config.metrics_eval_benchmark:
                spec = get_benchmark_spec(config.metrics_eval_benchmark)
                prefixes.append(f"eval_{spec.name.replace('-', '_')}")
            else:
                prefixes.append("eval_external")
        else:
            prefixes.append("eval_train")

        protocols = _metrics_eval_protocols(config.metrics_eval_protocol)
        keys.update(
            f"{prefix}_{protocol}_{metric}"
            for prefix in prefixes
            for protocol in protocols
            for metric in BEST_MODEL_METRIC_BASE_KEYS
        )
        evaluated_markush_datasets = [
            dataset for dataset in INLINE_EVAL_MARKUSH_DATASETS if f"eval_{dataset}" in prefixes
        ]
        keys.update(f"eval_{dataset}_avg" for dataset in evaluated_markush_datasets)
        has_markush_aggregate = bool(evaluated_markush_datasets)
        if has_markush_aggregate:
            keys.add("markush_overall")
            if "ours" in protocols:
                keys.add("markush_ours")
            if "mg2" in protocols:
                keys.add("markush_mg2")
    if config.ocsr_metrics_eval_size > 0 and config.ocsr_metrics_eval_data:
        specs = _resolve_ocsr_metrics_eval_specs(
            config,
            root=root or Path.cwd(),
            require_exists=False,
        )
        keys.update(
            f"{spec.prefix}_{metric}"
            for spec in specs
            for metric in OCSR_BEST_MODEL_METRIC_BASE_KEYS
        )
        has_ocsr_aggregate = bool(specs)
        if has_ocsr_aggregate:
            keys.add("ocsr_overall")
    if has_markush_aggregate or has_ocsr_aggregate:
        keys.add("overall_combined")
    return keys


def validate_best_metric_config(config: SFTConfig, *, root: Path | None = None) -> None:
    """Fail early when best-checkpoint metric selection cannot be satisfied."""

    if not config.load_best_model_at_end:
        return
    if config.eval_steps <= 0 or config.save_steps <= 0 or config.save_steps != config.eval_steps:
        raise ValueError(
            "load_best_model_at_end requires save_steps == eval_steps so every best-metric "
            "evaluation has a matching checkpoint. "
            f"Found save_steps={config.save_steps}, eval_steps={config.eval_steps}."
        )
    if not config.metric_for_best_model:
        return
    metric = config.metric_for_best_model
    if metric in {"eval_loss", "loss"}:
        return
    expected = expected_best_metric_keys(config, root=root)
    if metric not in expected:
        sample = ", ".join(sorted(expected)[:12]) or "eval_loss"
        raise ValueError(
            "metric_for_best_model is not produced by the configured metrics callbacks: "
            f"{metric!r}. Disable load_best_model_at_end or choose one of: {sample}"
        )
    has_markush_aggregate = any(
        key in expected for key in ("markush_ours", "markush_mg2", "markush_overall")
    )
    has_ocsr_aggregate = "ocsr_overall" in expected
    if metric == "overall_combined":
        uses_markush_metrics = has_markush_aggregate
        uses_ocsr_metrics = has_ocsr_aggregate
    else:
        uses_ocsr_metrics = metric == "ocsr_overall" or metric.startswith("eval_ocsr_")
        uses_markush_metrics = not uses_ocsr_metrics

    effective_ocsr_metrics_steps = (
        config.ocsr_metrics_eval_steps
        if config.ocsr_metrics_eval_steps is not None
        else config.metrics_eval_steps
    )
    if (
        uses_ocsr_metrics
        and effective_ocsr_metrics_steps
        and config.eval_steps % effective_ocsr_metrics_steps != 0
    ):
        raise ValueError(
            "metric_for_best_model uses OCSR metrics, but "
            "ocsr_metrics_eval_steps/metrics_eval_steps="
            f"{effective_ocsr_metrics_steps} skips some evals (eval_steps={config.eval_steps}). "
            "Every eval needs the metric; set ocsr_metrics_eval_steps to a divisor of "
            "eval_steps or unset it."
        )
    if (
        uses_markush_metrics
        and config.metrics_eval_steps
        and config.eval_steps % config.metrics_eval_steps != 0
    ):
        raise ValueError(
            "metric_for_best_model uses Markush metrics, but metrics_eval_steps="
            f"{config.metrics_eval_steps} skips some evals (eval_steps={config.eval_steps}). "
            "Every eval needs the metric; set metrics_eval_steps to a divisor of "
            "eval_steps or unset it."
        )


def _distributed_rank_world() -> tuple[int, int]:
    """Return the active distributed rank/world size, or single-process defaults."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _distributed_all_gather_object(payload: Any) -> list[Any]:
    """Gather a small Python payload from every DDP rank.

    Falls back to ``[payload]`` when distributed is not initialized so the
    single-process path behaves identically.
    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[Any] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, payload)
        return gathered
    return [payload]


def _distributed_broadcast_object(payload: Any, *, src: int = 0) -> Any:
    """Broadcast a small Python payload from ``src`` to every DDP rank."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        objects = [payload if torch.distributed.get_rank() == src else None]
        torch.distributed.broadcast_object_list(objects, src=src)
        return objects[0]
    return payload


def filter_records_by_text_length(
    records: list[dict[str, Any]],
    *,
    max_user_chars: int | None = None,
    max_assistant_chars: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop text-length outliers before batching.

    Truncating assistant XML would create invalid labels, so long examples are
    skipped and counted instead.
    """

    if not max_user_chars and not max_assistant_chars:
        return records, {"kept": len(records), "dropped_user": 0, "dropped_assistant": 0}

    kept: list[dict[str, Any]] = []
    stats = {"kept": 0, "dropped_user": 0, "dropped_assistant": 0}
    for record in records:
        messages = record.get("messages")
        user_text = ""
        assistant_text = ""
        if isinstance(messages, list):
            if len(messages) > 0 and isinstance(messages[0], dict):
                user_text = str(messages[0].get("content") or "")
            if len(messages) > 1 and isinstance(messages[1], dict):
                assistant_text = str(messages[1].get("content") or "")

        if max_user_chars and len(user_text) > max_user_chars:
            stats["dropped_user"] += 1
            continue
        if max_assistant_chars and len(assistant_text) > max_assistant_chars:
            stats["dropped_assistant"] += 1
            continue
        kept.append(record)
        stats["kept"] += 1
    return kept, stats


def _fused_cross_entropy_loss_function() -> Any | None:
    """Build a ``ForCausalLMLoss``-compatible callable backed by FLA's fused CE.

    Mirrors :func:`transformers.loss.loss_utils.ForCausalLMLoss` exactly
    (label shifting, flattening, ``num_items_in_batch`` normalization for
    gradient accumulation) but feeds the bf16 logits straight into FLA's
    Triton kernel, which upcasts per block in SRAM instead of materializing a
    full fp32 ``[batch, seq, vocab]`` copy.
    """

    try:
        from fla.modules import FusedCrossEntropyLoss
    except Exception as exc:
        LOGGER.warning("FLA fused cross-entropy unavailable (%s); keeping default loss.", exc)
        return None

    ce_mean = FusedCrossEntropyLoss(ignore_index=-100, reduction="mean", inplace_backward=True)
    ce_sum = FusedCrossEntropyLoss(ignore_index=-100, reduction="sum", inplace_backward=True)

    def loss_function(
        logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        vocab_size: int | None = None,
        num_items_in_batch: torch.Tensor | None = None,
        ignore_index: int = -100,
        shift_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if ignore_index != -100 or kwargs.get("label_smoothing"):
            # The kernel instances are fixed to ignore_index=-100 and no
            # smoothing; defer anything else to the reference implementation.
            from transformers.loss.loss_utils import ForCausalLMLoss

            return ForCausalLMLoss(
                logits,
                labels,
                cast("int", vocab_size),
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                **kwargs,
            )
        if shift_labels is None:
            labels = torch.nn.functional.pad(
                cast("torch.Tensor", labels), (0, 1), value=ignore_index
            )
            shift_labels = labels[..., 1:].contiguous()
        logits = logits.view(-1, cast("int", vocab_size))
        shift_labels = shift_labels.view(-1).to(logits.device)
        if num_items_in_batch is None:
            return ce_mean(logits, shift_labels)
        loss = ce_sum(logits, shift_labels)
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        return loss / num_items_in_batch

    return loss_function


def _model_compute_placement(base: Any) -> tuple[torch.device, torch.dtype]:
    """Resolve a (device, dtype) to probe the fused-CE kernel on.

    Prefer a real (non-meta) floating-point parameter of the base model so the
    probe runs on the exact arch/dtype training will use. Fall back to the
    active CUDA device (bf16) when params are absent or still on ``meta`` (e.g.
    lazily materialized), and finally to CPU (fp32) so the probe stays runnable
    off-GPU (where the Triton kernel simply raises and we degrade gracefully).
    """
    try:
        params = base.parameters()
    except (AttributeError, TypeError):
        params = iter(())
    for param in params:
        if param.is_floating_point() and param.device.type != "meta":
            return param.device, param.dtype
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def _probe_fused_cross_entropy(
    loss_function: Any, *, device: torch.device, dtype: torch.dtype
) -> None:
    """Run a tiny fused-CE forward+backward to surface arch/kernel failures.

    FLA's fused cross-entropy is a Triton kernel that JIT-compiles on first use,
    so a GPU whose arch the installed Triton/kernel cannot target fails there,
    not at import. Exercising it once on a microscopic input (1x4x16) turns a
    would-be mid-training crash into a fast, catchable probe. Re-raises whatever
    the kernel raises so the caller can decide to fall back.
    """
    logits = torch.randn(1, 4, 16, device=device, dtype=dtype, requires_grad=True)
    labels = torch.tensor([[0, 5, -100, 9]], device=device)
    loss = loss_function(logits, labels, vocab_size=16)
    loss.backward()


def install_fused_cross_entropy(model: Any) -> bool:
    """Swap the base model's LM loss for FLA's fused cross-entropy.

    Returns whether the swap happened. Works through PEFT wrappers: the loss
    is invoked as ``self.loss_function`` inside the base transformers model's
    ``forward``, so the override is set there.

    Degrades gracefully in two stages: if fla is not importable the loss
    function is ``None`` and we keep the default loss; if fla *is* importable
    but its Triton kernel cannot run on the local GPU, a probe catches the
    failure and we fall back to standard cross-entropy with a warning rather
    than crashing partway through training.
    """

    loss_function = _fused_cross_entropy_loss_function()
    if loss_function is None:
        return False
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Probe the kernel on the model's own arch/dtype before committing the swap.
    # fla can import cleanly yet fail to run on the local GPU (unsupported arch,
    # Triton compile error); probing here degrades to standard CE with a warning
    # instead of crashing on the first training step.
    device, dtype = _model_compute_placement(base)
    try:
        _probe_fused_cross_entropy(loss_function, device=device, dtype=dtype)
    except Exception as exc:
        LOGGER.warning(
            "FLA fused cross-entropy is importable but its Triton kernel failed on "
            "%s (%s); falling back to standard cross-entropy (identical loss, higher "
            "memory - you may need a smaller batch size).",
            device,
            exc,
        )
        return False
    try:
        base.loss_function = loss_function
    except AttributeError:
        LOGGER.warning(
            "FLA fused cross-entropy is available but could not be installed on %s.",
            type(base).__name__,
        )
        return False
    LOGGER.info("Installed FLA fused cross-entropy (no fp32 [batch, seq, vocab] materialization).")
    return True


def _persist_metrics_jsonl(
    output_dir: str | Path | None, filename: str, record: dict[str, Any]
) -> None:
    """Append one metrics record to ``<output_dir>/<filename>`` as JSON.

    Local persistence is the source of truth when W&B ingestion is degraded
    (a run can end up with summaries but no queryable history). Failures are
    logged and swallowed so disk issues can never take down training.
    """

    if not output_dir:
        return
    try:
        path = Path(output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        LOGGER.warning("Failed to persist metrics to %s/%s: %s", output_dir, filename, exc)


# ---------------------------------------------------------------------------
# Dataset and collator
# ---------------------------------------------------------------------------


class MarkushDataset(Dataset):
    """Dataset for Markush structure extraction.

    Loads JSONL files with the format::

        {
            "id": "sample_0",
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\\nPatent text: ...\\nExtract the Markush structure from this patent image.",
                },
                {
                    "role": "assistant",
                    "content": "<markush><cxsmi>...</cxsmi><stable>...</stable></markush>",
                },
            ],
            "images": ["images/sample_0.png"],
        }

    Args:
        records: List of JSONL records
        root: Root directory for resolving relative ``images/...`` paths. Defaults
            to the shared MarkushGlyph data root (``GLYPH_DATA_ROOT``) via
            :func:`glyph.paths.data_root` when omitted.
        text_dropout: Probability of dropping OCR/patent text (0.0 = keep all, 1.0 = drop all)
    """

    # Class-level flag so the "data/data" fallback warning only fires once
    # per process, no matter how many records or workers hit it.
    _fallback_warned: bool = False

    def __init__(
        self,
        records: list[dict[str, Any]],
        root: Path | None = None,
        text_dropout: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.records = records
        # Relative ``images/...`` paths resolve under the shared MarkushGlyph data
        # root (``GLYPH_DATA_ROOT``); absolute paths are used as-is (back-compat).
        self.root = data_root() if root is None else Path(root)
        self.text_dropout = text_dropout
        self._base_seed = seed
        # Epoch seed updated by set_epoch() for DDP reproducibility
        self._epoch_seed = seed

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic dropout across DDP workers."""
        self._epoch_seed = self._base_seed + epoch * 100000

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        r = self.records[idx]

        # Extract user content, stripping <image> placeholder
        user_content = r["messages"][0]["content"]
        user_text = user_content.replace("<image>\n", "").replace("<image>", "")

        # Extract target (assistant response)
        target = r["messages"][1]["content"]

        # Resolve image path. New artifacts store paths relative to the data root;
        # older artifacts may include a leading "data/" component.
        image_rel = r["images"][0] if r.get("images") else None
        image_path: str | None = None
        if image_rel:
            candidate = Path(image_rel)
            if candidate.is_absolute():
                image_path = str(candidate)
            else:
                primary = self.root / candidate
                legacy = self.root / "data" / candidate
                if not primary.exists() and legacy.exists():
                    if not MarkushDataset._fallback_warned:
                        LOGGER.warning(
                            "Image path %r resolved only through the legacy 'data/' layout. "
                            "Falling back to %s. This warning is logged once per process.",
                            image_rel,
                            legacy,
                        )
                        MarkushDataset._fallback_warned = True
                    image_path = str(legacy)
                else:
                    image_path = str(primary)

        # Apply text dropout: deterministic based on sample index for DDP reproducibility
        if self.text_dropout > 0.0:
            rng = random.Random(idx + self._epoch_seed)
            if rng.random() < self.text_dropout:
                user_text = self._apply_text_dropout(user_text)

        return {
            "id": r.get("id", str(idx)),
            "image": image_path,
            "user_text": user_text,
            "assistant_text": target,
        }

    def _apply_text_dropout(self, text: str) -> str:
        """Drop OCR/patent text, keeping only the extraction instruction.

        Input format examples::

            Patent text: wherein R1 is selected from alkyl...
            OCR text: R1, N, OH
            Extract the Markush structure from this patent image.

        or::

            OCR text: R1=alkyl, R2=aryl
            Extract the Markush structure from this patent image.

        Output: "Extract the Markush structure from this patent image."
        """
        # Find the extraction instruction and return only that
        # Pattern: "Extract the Markush structure" at the end
        match = re.search(r"(Extract the Markush structure[^\n]*)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback: return the default instruction
        return FALLBACK_INSTRUCTION


class MarkushCollator:
    """Collator using message-dict format with process_vision_info.

    Builds proper Qwen VL chat messages so the processor expands vision
    placeholders to the correct ``<|vision_start|><|image_pad|>...<|vision_end|>``
    token sequence.  Labels mask the user turn + generation prompt so loss is
    computed only on the assistant response.

    Args:
        processor: The Qwen VL processor
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        *,
        image_augmentation: bool = False,
        image_augmentation_prob: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.processor = processor
        self.tokenizer = cast("Any", processor).tokenizer
        pad = self.tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else self.tokenizer.eos_token_id
        self.image_augmentation = image_augmentation
        self.image_augmentation_prob = image_augmentation_prob
        self._augment_rng = random.Random(seed)
        self._albumentations_checked = False
        self._albumentations_transform = None

    def _augment_image(self, image):
        """Apply MG2-style patent-scan augmentation with optional albumentations."""

        if not self.image_augmentation:
            return image
        if self._augment_rng.random() >= self.image_augmentation_prob:
            return image

        from PIL import Image, ImageEnhance, ImageFilter

        image = image.convert("RGB")
        width, height = image.size

        augmented = self._augment_with_albumentations(image)
        if augmented is not None:
            image = augmented
        else:
            # Fallback matching the MG2-style augmentation intent when albumentations is not
            # installed: scale down by up to 20%, shift by +/-1%, then blur or
            # downscale with high probability.
            scale = self._augment_rng.uniform(0.8, 1.0)
            if abs(scale - 1.0) > 0.01:
                scaled_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                scaled = image.resize(scaled_size, Image.Resampling.BICUBIC)
                canvas = Image.new("RGB", (width, height), "white")
                x = (width - scaled_size[0]) // 2
                y = (height - scaled_size[1]) // 2
                canvas.paste(scaled, (x, y))
                image = canvas

            max_dx = max(1, int(width * 0.01))
            max_dy = max(1, int(height * 0.01))
            dx = self._augment_rng.randint(-max_dx, max_dx)
            dy = self._augment_rng.randint(-max_dy, max_dy)
            if dx or dy:
                translated = Image.new("RGB", (width, height), "white")
                translated.paste(image, (dx, dy))
                image = translated

            if self._augment_rng.random() < 0.8:
                if self._augment_rng.random() < 0.5:
                    image = image.filter(
                        ImageFilter.GaussianBlur(radius=self._augment_rng.uniform(0.1, 1.0))
                    )
                else:
                    factor = self._augment_rng.uniform(0.5, 0.85)
                    small = image.resize(
                        (max(1, int(width * factor)), max(1, int(height * factor))),
                        Image.Resampling.BILINEAR,
                    )
                    image = small.resize((width, height), Image.Resampling.BILINEAR)

        brightness = self._augment_rng.uniform(0.8, 1.2)
        contrast = self._augment_rng.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)

        image = self._add_random_scan_artifacts(image)

        return image

    def _augment_with_albumentations(self, image):
        """Return an albumentations-augmented image, or None when unavailable."""

        if not self._albumentations_checked:
            self._albumentations_checked = True
            try:
                import albumentations as A

                self._albumentations_transform = A.Compose(
                    [
                        A.Affine(
                            scale=(0.8, 1.0),
                            translate_percent=(-0.01, 0.01),
                            rotate=0,
                            fit_output=False,
                            fill=255,
                            p=1.0,
                        ),
                        A.OneOf(
                            [
                                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                                A.Downscale(scale_range=(0.5, 0.85), p=1.0),
                            ],
                            p=0.8,
                        ),
                    ]
                )
            except Exception:
                self._albumentations_transform = None

        if self._albumentations_transform is None:
            return None

        try:
            import numpy as np
            from PIL import Image

            array = np.asarray(image)
            augmented = self._albumentations_transform(image=array)["image"]
            return Image.fromarray(augmented).convert("RGB")
        except Exception:
            return None

    def _add_random_scan_artifacts(self, image):
        """Add low-probability line and pepper-patch artifacts."""

        from PIL import ImageDraw

        width, height = image.size
        draw = ImageDraw.Draw(image)
        if self._augment_rng.random() < 0.05:
            for _ in range(self._augment_rng.randint(1, 3)):
                y = self._augment_rng.randint(0, max(0, height - 1))
                shade = self._augment_rng.randint(0, 80)
                draw.line((0, y, width, y), fill=(shade, shade, shade), width=1)
        if self._augment_rng.random() < 0.05:
            for _ in range(self._augment_rng.randint(4, 12)):
                x = self._augment_rng.randint(0, max(0, width - 2))
                y = self._augment_rng.randint(0, max(0, height - 2))
                size = self._augment_rng.randint(1, 3)
                draw.rectangle((x, y, x + size, y + size), fill=(0, 0, 0))
        return image

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages_batch = []
        for s in samples:
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": s["image"]},  # path as string
                            {"type": "text", "text": s["user_text"]},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": s["assistant_text"]},
                        ],
                    },
                ]
            )

        texts = [
            self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for msgs in messages_batch
        ]

        all_images = []
        for msgs in messages_batch:
            imgs, _ = cast("tuple[Any, Any]", process_vision_info(msgs))
            imgs = [self._augment_image(img) for img in imgs]
            all_images.extend(imgs)

        inputs = self.processor(
            text=texts,
            images=all_images,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100

        for i, msgs in enumerate(messages_batch):
            prefix_msgs = [msgs[0]]
            prefix_text = self.processor.apply_chat_template(
                prefix_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prefix_imgs, _ = cast("tuple[Any, Any]", process_vision_info(prefix_msgs))
            prefix_inputs = self.processor(
                text=[prefix_text],
                images=prefix_imgs,
                padding=False,
                return_tensors="pt",
            )
            prefix_len = prefix_inputs["input_ids"].shape[1]
            labels[i, :prefix_len] = -100

        inputs["labels"] = labels
        return inputs


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class DetailedLoggingCallback(TrainerCallback):
    """Callback for detailed W&B logging.

    Logs additional metrics beyond the defaults:
    - GPU memory usage
    - Training throughput
    - Gradient statistics
    """

    def __init__(self) -> None:
        self.last_log_time = time.time()
        self.last_step = 0

    def on_log(
        self,
        args: TrainingArguments,
        state,
        control,
        logs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero or not logs:
            return

        current_time = time.time()
        elapsed = current_time - self.last_log_time
        steps_done = state.global_step - self.last_step

        extra_metrics: dict[str, Any] = {}

        # GPU memory metrics (rank-0 device only; per-rank peaks differ).
        if torch.cuda.is_available():
            extra_metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            extra_metrics["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            extra_metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        # Throughput metrics
        if elapsed > 0 and steps_done > 0:
            extra_metrics["throughput/steps_per_second"] = steps_done / elapsed
            if "train/samples_per_second" in logs:
                extra_metrics["throughput/samples_per_second"] = logs.get(
                    "train/samples_per_second", 0
                )

        self.last_log_time = current_time
        self.last_step = state.global_step

        # Persist raw trainer logs plus GPU/throughput extras locally
        # regardless of W&B availability; checkpointed trainer_state.json only
        # snapshots at saves, so this keeps a complete step-by-step record on
        # disk (the W&B history backend has been unreliable).
        _persist_metrics_jsonl(
            getattr(args, "output_dir", None),
            "trainer_logs.jsonl",
            {"ts": current_time, "step": state.global_step, **logs, **extra_metrics},
        )

        try:
            import wandb

            if wandb.run is None or not extra_metrics:
                return

            extra_metrics["trainer/global_step"] = state.global_step
            with suppress(Exception):
                wandb.define_metric("gpu/*", step_metric="trainer/global_step")
                wandb.define_metric("throughput/*", step_metric="trainer/global_step")
            current_wandb_step = getattr(wandb.run, "step", None)
            log_step = state.global_step
            if isinstance(current_wandb_step, int) and log_step <= current_wandb_step:
                log_step = current_wandb_step + 1
            wandb.log(extra_metrics, step=log_step)

        except Exception as e:
            LOGGER.debug("W&B logging error: %s", e)


class ResolvedConfigCallback(TrainerCallback):
    """Record the fully-resolved ``SFTConfig`` in the W&B run config.

    HF's ``WandbCallback`` logs only ``TrainingArguments``, so custom
    ``SFTConfig`` fields (e.g. ``fused_cross_entropy`` and the Markush target
    canonicalization flags) never reach the tracker — a real provenance gap,
    since the published checkpoint's recipe was otherwise only visible in the
    run's stdout. This runs after ``WandbCallback`` initialises the run, so
    ``wandb.run`` is available. Defensive: a no-op when W&B is
    disabled/offline and never raises if the tracker is unavailable.
    """

    def __init__(self, config_dict: dict[str, Any]) -> None:
        self._config_dict = config_dict

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs) -> None:
        if not state.is_world_process_zero:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            wandb.config.update(self._config_dict, allow_val_change=True)
            LOGGER.info("Logged resolved SFTConfig to W&B run config.")
        except Exception as exc:
            LOGGER.warning("Failed to log resolved SFTConfig to W&B: %s (continuing).", exc)


class SaveProcessorCallback(TrainerCallback):
    """Callback to save the processor with each checkpoint."""

    def __init__(self, processor: ProcessorMixin) -> None:
        self.processor = processor

    def on_save(
        self,
        args: TrainingArguments,
        state,
        control,
        **kwargs,
    ) -> None:
        if not state.is_world_process_zero:
            return

        # Save processor to the current checkpoint directory
        checkpoint_dir = Path(cast("str", args.output_dir)) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            try:
                self.processor.save_pretrained(str(checkpoint_dir))
                LOGGER.info("Saved processor to %s", checkpoint_dir)
            except Exception as e:
                LOGGER.warning("Failed to save processor: %s", e)


def install_processor_save_callback_first(trainer: Trainer, processor: ProcessorMixin) -> None:
    """Install processor saving before reporting callbacks observe checkpoints."""

    trainer.callback_handler.callbacks.insert(0, SaveProcessorCallback(processor))


class MetricsEvalCallback(TrainerCallback):
    """Generate predictions on a fixed subset at each eval and score with MCS.

    Records both ``"ours"`` (our stricter eval) and ``"mg2"`` (approximate,
    in-package MG2 replication — not byte-exact; official numbers require the
    pinned DS4SD scorer) metrics so dashboards can track both side-by-side.
    Logs are namespaced as ``eval/ours/<key>`` and ``eval/mg2/<key>``.

    The subset is fixed by ``samples`` and re-used every eval step, so the
    metric values are directly comparable across steps.

    Under DDP, every rank generates a deterministic ``samples[rank::world]``
    shard in parallel (batched), then rank 0 gathers all rank payloads, scores
    once, and logs.  This keeps all GPUs busy during eval instead of running
    the whole subset serially on rank 0.

    Args:
        samples: List of dataset rows to score on. Each row must have the same
            shape as :class:`MarkushDataset` items (``image``, ``user_text``,
            ``assistant_text``, ``id``).
        processor: Processor for tokenization + chat templating.
        protocol: ``"ours"``, ``"mg2"``, or ``"both"`` (default).
        max_new_tokens: Generation budget per sample.
        batch_size: Per-rank generation batch size. ``1`` keeps the original
            one-sample-at-a-time path; larger values batch through
            :func:`glyph.markush.eval.fast_eval.generate_batch`.
        prefix: Log-key prefix; defaults to ``"eval"`` so keys come out as
            ``eval/ours/cxsmi_equality`` etc., next to HF's ``eval/loss``.
        interval_steps: When set, only run on steps divisible by this value
            (a coarser cadence than ``eval_steps``). ``None`` runs every eval.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        processor: ProcessorMixin,
        protocol: str = "both",
        max_new_tokens: int = 512,
        batch_size: int = 1,
        prefix: str = "eval",
        interval_steps: int | None = None,
    ) -> None:
        if protocol not in ("ours", "mg2", "both"):
            raise ValueError(f"protocol must be ours|mg2|both, got {protocol!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        self.samples = samples
        self.processor = processor
        self.protocol = protocol
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.prefix = prefix
        self.interval_steps = interval_steps

    @staticmethod
    def _build_user_text(user_text: str) -> str:
        """Strip trailing 'Extract the Markush structure...' so we can append our own."""
        return re.sub(
            r"\n?Extract the Markush structure[^\n]*\.?\n?$",
            "",
            user_text,
            flags=re.IGNORECASE,
        ).rstrip()

    def _build_generation_prompt(self, sample: dict[str, Any]) -> str:
        """Return the user text (instruction + delimiter) for one sample."""
        cleaned = self._build_user_text(sample.get("user_text", ""))
        # Prepared eval rows already include the exact target-format
        # instruction used by SFT and offline eval. Do not append the legacy
        # delimiter, or inline SFT eval will use a different prompt.
        if "Target format:" in cleaned:
            return cleaned
        return f"{cleaned}\n{PROMPT_DELIMITER}" if cleaned else PROMPT_DELIMITER

    def _generate_one(self, model: Any, sample: dict[str, Any]) -> str:
        """Greedy-generate the model's completion for one sample."""
        user_text = self._build_generation_prompt(sample)

        # Build message-dict so the processor correctly expands vision
        # placeholders (same pattern as MarkushCollator).
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},  # path string
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        imgs, _ = cast("tuple[Any, Any]", process_vision_info(messages))

        tokenizer = cast("Any", self.processor).tokenizer
        from glyph.markush.eval.fast_eval import temporary_tokenizer_padding_side

        with temporary_tokenizer_padding_side(self.processor, "left"):
            inputs = self.processor(
                text=[text_prompt],
                images=imgs,
                return_tensors="pt",
                padding=True,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=[ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID],
                    # setup_model disables use_cache for gradient checkpointing;
                    # without the KV cache greedy decode degrades to O(n^2)
                    # prefill per token. The batched path already forces this.
                    use_cache=True,
                )
        generated = output_ids[0, prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        # Strip any chat-template control tokens and <think> blocks before
        # handing the prediction to the MCS evaluator. Avoids over-counting
        # generation length against the metric.
        from glyph.markush.eval.common import clean_prediction

        return clean_prediction(text)

    def _generate_many(self, model: Any, samples: list[dict[str, Any]]) -> tuple[list[str], int]:
        """Greedy-generate a local batch, reusing fast_eval's batched primitive.

        Returns ``(predictions, n_failed)`` aligned to ``samples``. Images that
        fail to load are skipped (counted as failures); a batched-generation
        failure falls back to per-sample generation so one bad row does not lose
        the whole batch.
        """

        from glyph.markush.eval.fast_eval import EvalSample as FastEvalSample
        from glyph.markush.eval.fast_eval import generate_batch, load_image

        predictions = [""] * len(samples)
        failed = 0
        valid_samples: list[FastEvalSample] = []
        valid_positions: list[int] = []
        valid_raw: list[dict[str, Any]] = []
        images: list[Any] = []

        for position, sample in enumerate(samples):
            sid = str(sample.get("id", ""))
            image_path = str(sample.get("image", ""))
            try:
                image = load_image(image_path)
            except Exception as exc:
                LOGGER.warning(
                    "MetricsEvalCallback: could not load eval image for %s: %s", sid, exc
                )
                failed += 1
                continue
            valid_samples.append(
                FastEvalSample(
                    index=position,
                    sample_id=sid,
                    image_path=image_path,
                    prompt=self._build_generation_prompt(sample),
                    ground_truth=sample.get("assistant_text", ""),
                )
            )
            valid_positions.append(position)
            valid_raw.append(sample)
            images.append(image)

        if not valid_samples:
            return predictions, failed

        device = next(model.parameters()).device
        try:
            generated = generate_batch(
                model,
                self.processor,
                valid_samples,
                images,
                max_new_tokens=self.max_new_tokens,
                eos_token_ids=(ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID),
                device=str(device),
            )
        except Exception as exc:
            LOGGER.warning(
                "MetricsEvalCallback: batched generation failed for %d samples; "
                "retrying individually: %s",
                len(valid_samples),
                exc,
            )
            for position, sample in zip(valid_positions, valid_raw, strict=True):
                try:
                    predictions[position] = self._generate_one(model, sample)
                except Exception as retry_exc:
                    LOGGER.warning(
                        "MetricsEvalCallback: individual retry failed for %s: %s",
                        sample.get("id", ""),
                        retry_exc,
                    )
                    failed += 1
            return predictions, failed

        for position, prediction in zip(valid_positions, generated, strict=True):
            predictions[position] = prediction
        return predictions, failed

    def _local_ground_truth_rows(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return evaluator ground-truth rows for the local generation shard."""

        return [
            {"id": str(sample.get("id", "")), "annotation": sample.get("assistant_text", "")}
            for sample in samples
        ]

    def _score_predictions(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Score gathered predictions and return slashed log keys."""

        from glyph.markush.eval.metrics_mcs import (
            evaluate_markush_both_protocols,
            evaluate_markush_mcs,
            flatten_both_protocol_metrics,
        )

        if self.protocol == "both":
            both_metrics = evaluate_markush_both_protocols(
                predictions, ground_truths, include_missing=True, return_details=False
            )
            return flatten_both_protocol_metrics(
                cast("dict[str, dict[str, Any]]", both_metrics), prefix=self.prefix
            )
        metrics = evaluate_markush_mcs(
            predictions,
            ground_truths,
            include_missing=True,
            return_details=False,
            protocol=self.protocol,
        )
        return {
            f"{self.prefix}/{self.protocol}/{key}": value
            for key, value in cast("dict[str, Any]", metrics).items()
        }

    def _compact_log_payload(self, log_payload: dict[str, Any]) -> dict[str, Any]:
        """Return the concise subset surfaced in the trainer log stream."""

        return {
            key: value
            for key, value in log_payload.items()
            if "/cxsmi_equality" in key or "/loss" in key
        }

    def _should_run(self, state) -> bool:
        """Gate on the optional coarser ``interval_steps`` cadence."""
        if self.interval_steps is None or self.interval_steps <= 0:
            return True
        return state.global_step > 0 and state.global_step % self.interval_steps == 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state,
        control,
        model: Any | None = None,
        **kwargs,
    ) -> None:
        if not self._should_run(state):
            return
        if model is None:
            LOGGER.warning(
                "MetricsEvalCallback: no `model` in kwargs at on_evaluate; skipping step %d",
                state.global_step,
            )
            return
        if not self.samples:
            return

        # Shard the fixed subset across ranks; every rank generates its slice in
        # parallel, then rank 0 gathers all rank payloads and scores once. All
        # ranks MUST reach the gather, so generation is wrapped so per-rank
        # failures still produce a payload rather than deadlocking the collective.
        rank, world_size = _distributed_rank_world()

        was_training = bool(getattr(model, "training", False))
        local_samples: list[dict[str, Any]] = []
        local_preds: list[dict[str, Any]] = []
        local_gts: list[dict[str, Any]] = []
        n_failed = 0
        error: str | None = None
        start = time.time()
        if rank == 0:
            LOGGER.info(
                "MetricsEvalCallback step=%d prefix=%s starting: %d samples across %d rank(s), "
                "batch_size=%d, max_new_tokens=%d",
                state.global_step,
                self.prefix,
                len(self.samples),
                world_size,
                self.batch_size,
                self.max_new_tokens,
            )
        try:
            local_samples = self.samples[rank::world_size]
            model.eval()
            local_gts = self._local_ground_truth_rows(local_samples)
            n_batches = (len(local_samples) + self.batch_size - 1) // self.batch_size
            for batch_index, batch_start in enumerate(
                range(0, len(local_samples), self.batch_size)
            ):
                batch = local_samples[batch_start : batch_start + self.batch_size]
                batch_preds, batch_failed = self._generate_many(model, batch)
                n_failed += batch_failed
                for sample, pred in zip(batch, batch_preds, strict=True):
                    local_preds.append({"id": str(sample.get("id", "")), "prediction": pred})
                # Heartbeat so long evals (minutes of silent generation) are
                # distinguishable from a hang in the trainer log stream.
                if rank == 0 and (batch_index % 5 == 4 or batch_index == n_batches - 1):
                    LOGGER.info(
                        "MetricsEvalCallback step=%d prefix=%s rank0 progress: %d/%d local "
                        "samples in %.0fs",
                        state.global_step,
                        self.prefix,
                        len(local_preds),
                        len(local_samples),
                        time.time() - start,
                    )
        except Exception as exc:
            error = str(exc)
            LOGGER.warning(
                "MetricsEvalCallback rank=%d local eval shard failed at step %d: %s",
                rank,
                state.global_step,
                exc,
            )
        finally:
            if was_training:
                with suppress(Exception):
                    model.train()
            # Return generation-time allocator blocks (KV caches, vision
            # activations) to CUDA before training resumes; the post-eval
            # cross-entropy spike on long-sequence batches needs the headroom.
            if torch.cuda.is_available():
                with suppress(Exception):
                    torch.cuda.empty_cache()
        local_elapsed = time.time() - start

        local_payload = {
            "rank": rank,
            "preds": local_preds,
            "gts": local_gts,
            "n_failed": n_failed,
            "elapsed_s": local_elapsed,
            "error": error,
        }
        gathered = _distributed_all_gather_object(local_payload)

        metrics_dict = kwargs.get("metrics")
        if rank != 0:
            flat_payload = _distributed_broadcast_object(None, src=0)
            if isinstance(metrics_dict, dict) and isinstance(flat_payload, dict):
                metrics_dict.update(flat_payload)
            return

        flat_payload: dict[str, Any] = {}
        log_payload: dict[str, Any] = {}
        try:
            preds: list[dict[str, Any]] = []
            gts: list[dict[str, Any]] = []
            n_failed = 0
            elapsed_by_rank: list[float] = []
            rank_errors: list[str] = []
            for payload in gathered:
                if not isinstance(payload, dict):
                    continue
                preds.extend(payload.get("preds", []))
                gts.extend(payload.get("gts", []))
                n_failed += int(payload.get("n_failed", 0) or 0)
                elapsed_by_rank.append(float(payload.get("elapsed_s", 0.0) or 0.0))
                if payload.get("error"):
                    rank_errors.append(f"rank {payload.get('rank', '?')}: {payload['error']}")
            # Ranks that died mid-shard contribute GTs without matching preds.
            n_failed += max(0, len(gts) - len(preds))
            elapsed = max(elapsed_by_rank) if elapsed_by_rank else (time.time() - start)
            if not gts:
                LOGGER.warning(
                    "MetricsEvalCallback step=%d prefix=%s gathered no samples; skipping",
                    state.global_step,
                    self.prefix,
                )
            else:
                if rank_errors:
                    LOGGER.warning(
                        "MetricsEvalCallback step=%d prefix=%s saw %d rank error(s): %s",
                        state.global_step,
                        self.prefix,
                        len(rank_errors),
                        "; ".join(rank_errors[:3]),
                    )

                score_start = time.time()
                log_payload = self._score_predictions(preds, gts)

                log_payload[f"{self.prefix}/trainer_step"] = state.global_step
                log_payload[f"{self.prefix}/gen/n_samples"] = len(gts)
                log_payload[f"{self.prefix}/gen/n_failed"] = n_failed
                log_payload[f"{self.prefix}/gen/elapsed_s"] = round(elapsed, 2)
                log_payload[f"{self.prefix}/gen/score_elapsed_s"] = round(
                    time.time() - score_start,
                    2,
                )
                log_payload[f"{self.prefix}/gen/world_size"] = world_size
                log_payload[f"{self.prefix}/gen/batch_size"] = self.batch_size
                log_payload[f"{self.prefix}/gen/rank_errors"] = len(rank_errors)
                flat_payload = {
                    slashed_key.replace("/", "_"): value
                    for slashed_key, value in log_payload.items()
                }

                # Persist locally first; W&B is best-effort on top of this.
                _persist_metrics_jsonl(
                    getattr(args, "output_dir", None),
                    "metrics_eval.jsonl",
                    {
                        "ts": time.time(),
                        "step": state.global_step,
                        "prefix": self.prefix,
                        **log_payload,
                    },
                )

                # Forward to W&B if it's the active reporter. Trainer's own logs go
                # through state.log_history. Here we use a monotonic W&B row step and
                # expose the actual trainer step as a metric-specific step axis.
                try:
                    import wandb

                    if wandb.run is not None:
                        with suppress(Exception):
                            wandb.define_metric(
                                f"{self.prefix}/*",
                                step_metric=f"{self.prefix}/trainer_step",
                            )
                        current_wandb_step = getattr(wandb.run, "step", None)
                        log_step = state.global_step
                        if isinstance(current_wandb_step, int) and log_step <= current_wandb_step:
                            log_step = current_wandb_step + 1
                        wandb.log(log_payload, step=log_step)
                except Exception:
                    pass
        except Exception:
            LOGGER.exception(
                "MetricsEvalCallback step=%d prefix=%s rank0 scoring/logging failed; "
                "broadcasting empty metrics",
                state.global_step,
                self.prefix,
            )
            flat_payload = {}
        finally:
            # Trainer's best-model check runs on every DDP rank, so rank 0 must
            # always release the matching receive even if scoring/logging failed.
            with suppress(Exception):
                _distributed_broadcast_object(flat_payload, src=0)

        # Also write underscore-flattened keys into the `metrics` dict HF
        # passes through `on_evaluate`.
        if isinstance(metrics_dict, dict):
            metrics_dict.update(flat_payload)

        # Surface a short summary in the trainer log stream.
        compact = self._compact_log_payload(log_payload)
        if compact:
            LOGGER.info("MetricsEvalCallback step=%d: %s", state.global_step, compact)


class OCSRMetricsEvalCallback(MetricsEvalCallback):
    """Generate and score OCSR rows with configured SMILES metrics."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        processor: ProcessorMixin,
        max_new_tokens: int = 256,
        batch_size: int = 1,
        prefix: str = "eval_ocsr",
        interval_steps: int | None = None,
        prediction_mode: str = OCSR_PREDICTION_MODE_NATIVE_SMILES,
    ) -> None:
        super().__init__(
            samples=samples,
            processor=processor,
            protocol="ours",
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            prefix=prefix,
            interval_steps=interval_steps,
        )
        from glyph.markush.eval.ocsr import OCSREvaluationConfig

        self.evaluation_config = OCSREvaluationConfig(prediction_mode=prediction_mode)

    def _local_ground_truth_rows(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"id": str(sample.get("id", "")), "smiles": sample.get("assistant_text", "")}
            for sample in samples
        ]

    def _score_predictions(
        self,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from glyph.markush.eval.ocsr import evaluate_ocsr_smiles, flatten_ocsr_metrics

        metrics = evaluate_ocsr_smiles(
            predictions,
            ground_truths,
            include_missing=True,
            return_details=False,
            config=self.evaluation_config,
        )
        return flatten_ocsr_metrics(cast("dict[str, Any]", metrics), prefix=self.prefix)

    def _compact_log_payload(self, log_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in log_payload.items()
            if key.endswith("/canon_exact_match") or key.endswith("/valid_smiles_rate")
        }


class MultiMetricsEvalCallback(TrainerCallback):
    """Run multiple metrics-eval jobs through one Trainer callback instance."""

    def __init__(self, callbacks: list[MetricsEvalCallback]) -> None:
        self.callbacks = callbacks

    def on_evaluate(
        self,
        args: TrainingArguments,
        state,
        control,
        **kwargs,
    ) -> None:
        for callback in self.callbacks:
            callback.on_evaluate(args, state, control, **kwargs)

        metrics_dict = kwargs.get("metrics")
        rank, _ = _distributed_rank_world()
        if rank != 0:
            flat_payload = _distributed_broadcast_object(None, src=0)
            if isinstance(metrics_dict, dict) and isinstance(flat_payload, dict):
                metrics_dict.update(flat_payload)
            return

        aggregate_payload: dict[str, float] = {}
        flat_payload: dict[str, Any] = {}
        try:
            if isinstance(metrics_dict, dict):
                aggregate_payload = compute_inline_eval_aggregate_metrics(metrics_dict)
            if aggregate_payload:
                flat_payload = {
                    slashed_key.replace("/", "_"): value
                    for slashed_key, value in aggregate_payload.items()
                }

                _persist_metrics_jsonl(
                    getattr(args, "output_dir", None),
                    "metrics_eval.jsonl",
                    {
                        "ts": time.time(),
                        "step": state.global_step,
                        "prefix": "aggregate",
                        **aggregate_payload,
                    },
                )

                try:
                    import wandb

                    if wandb.run is not None:
                        current_wandb_step = getattr(wandb.run, "step", None)
                        log_step = state.global_step
                        if isinstance(current_wandb_step, int) and log_step <= current_wandb_step:
                            log_step = current_wandb_step + 1
                        wandb.log(aggregate_payload, step=log_step)
                except Exception:
                    pass
        except Exception:
            LOGGER.exception(
                "MultiMetricsEvalCallback step=%d aggregate scoring/logging failed; "
                "broadcasting empty metrics",
                state.global_step,
            )
            aggregate_payload = {}
            flat_payload = {}
        finally:
            with suppress(Exception):
                _distributed_broadcast_object(flat_payload, src=0)

        if isinstance(metrics_dict, dict):
            metrics_dict.update(flat_payload)
        if aggregate_payload:
            LOGGER.info(
                "MultiMetricsEvalCallback step=%d aggregates: %s",
                state.global_step,
                aggregate_payload,
            )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_records(data_path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a file."""
    records = []
    with open(data_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                LOGGER.warning("Invalid JSON at %s:%d: %s", data_path, line_no, e)
    return records


def _metric_image_path(row: dict[str, Any], *, data_dir: Path) -> str:
    """Resolve image fields used by eval/unified JSONL rows for metric callbacks."""

    for key in ("image", "image_path", "path", "png_path", "rendered_path", "file_name"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("path"):
            value = value["path"]
        if isinstance(value, str) and value.strip():
            path = Path(value)
            return str(path if path.is_absolute() else data_dir / path)
    images = row.get("images")
    if isinstance(images, list) and images:
        path = Path(str(images[0]))
        return str(path if path.is_absolute() else data_dir / path)
    return ""


def _load_metrics_samples(
    eval_jsonl: Path,
    *,
    data_dir: Path,
    limit: int,
    prompt: str,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
    markush_prompt_policy: str = MARKUSH_PROMPT_POLICY_DEFAULT,
) -> list[dict[str, Any]]:
    """Load external eval JSONL rows into ``MetricsEvalCallback`` samples."""

    with eval_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    samples = []
    for row in rows:
        if row_declares_ocsr_smiles(row):
            LOGGER.debug(
                "Skipping OCSR SMILES row %s in Markush metrics eval loader", row.get("id")
            )
            continue
        target = target_spec_for_record(
            row,
            canonicalize_markush=canonicalize_markush_targets,
            require_canonical=require_canonical_markush_targets,
        )
        if target is not None:
            gt = target.content
            user_text = prompt_with_target_instruction(str(row.get("prompt") or prompt), target)
        else:
            if canonicalize_markush_targets and require_canonical_markush_targets:
                continue
            gt = row.get("annotation") or row.get("cxsmiles_opt") or row.get("target") or ""
            user_text = row.get("prompt") or prompt
        user_text = apply_markush_prompt_policy(user_text, markush_prompt_policy)
        if not gt:
            continue
        samples.append(
            {
                "id": row.get("id", ""),
                "image": _metric_image_path(row, data_dir=data_dir),
                "user_text": user_text,
                "assistant_text": gt,
            }
        )
        if len(samples) >= limit:
            break
    return samples


def _target_ocsr_smiles_for_row(row: dict[str, Any]) -> str | None:
    """Return a canonical bare SMILES target for an OCSR eval/prepared row.

    Training data should use explicit unified rows (``task="ocsr"`` or
    prepared ``molecule_ocsr`` metadata), but metric-eval staging also accepts
    raw benchmark rows that only contain a parseable ``smiles`` field.
    """
    task = str(row.get("task") or "").lower()
    if task == "ocsr":
        target = target_spec_for_record(row)
        if target is not None:
            return target.content
    if row_declares_ocsr_smiles(row):
        fields = (
            common.sft_assistant_target(row),
            row.get("smiles"),
            row.get("gt_smiles"),
            row.get("target"),
            row.get("label"),
        )
    else:
        fields = (row.get("smiles"),)
    for value in fields:
        result = canonicalize_ocsr_smiles_checked(value if isinstance(value, str) else None)
        if result.value is not None:
            return result.value
    return None


def _load_ocsr_metrics_samples(
    eval_jsonl: Path,
    *,
    data_dir: Path,
    limit: int,
    prompt: str = MOLECULE_OCSR_PROMPT,
) -> list[dict[str, Any]]:
    """Load OCSR eval JSONL rows into ``OCSRMetricsEvalCallback`` samples."""

    with eval_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    samples: list[dict[str, Any]] = []
    for row in rows:
        target = _target_ocsr_smiles_for_row(row)
        if target is None:
            continue
        user_text = common.sft_user_prompt(row)
        if not user_text:
            target_spec = target_spec_for_record({"task": "ocsr", "smiles": target})
            user_text = prompt
            if target_spec is not None:
                user_text = prompt_with_target_instruction(user_text, target_spec)
        samples.append(
            {
                "id": row.get("id", ""),
                "image": _metric_image_path(row, data_dir=data_dir),
                "user_text": user_text,
                "assistant_text": target,
            }
        )
        if len(samples) >= limit:
            break
    return samples


def _require_ocsr_metric_samples(samples: list[dict[str, Any]], eval_jsonl: Path) -> None:
    """Fail loudly when an enabled OCSR metrics job has no scorable rows."""

    if samples:
        return
    raise ValueError(
        "OCSR metrics eval loaded zero samples from "
        f"{eval_jsonl}. Rows must either be prepared/unified OCSR rows "
        "or raw benchmark rows with a parseable smiles field."
    )


def _resolve_metrics_benchmark_path(
    benchmark: str,
    *,
    root: Path,
    metrics_eval_data_dir: str | None = None,
) -> Path:
    """Resolve benchmark JSONL for inline metrics eval.

    By default benchmark paths resolve to the canonical ``glyph.data.layout``
    location under ``$GLYPH_DATA_ROOT`` (``benchmarks/markush/<subdir>/
    eval_<subdir>.jsonl``). Cluster runs can instead set
    ``metrics_eval_data_dir`` to a prepared artifact directory (typically
    ``$GLYPH_DATA_ROOT/benchmarks/markush``).

    In that prepared tree the ``glyph data prepare --markush ... benchmarks``
    importer writes each benchmark into its own subdirectory named after the
    benchmark's filesystem-safe (underscore) name, i.e.
    ``<dir>/<name>/eval_<name>.jsonl`` (see
    :func:`glyph.data.layout.markush_benchmark_dir` and
    :data:`glyph.data.layout.MARKUSH_BENCHMARK_NAMES`). The canonical spec name
    may use a hyphen (``uspto-markush``) while the on-disk subdir uses an
    underscore (``uspto_markush``); normalize so both consumers agree.
    """

    spec = get_benchmark_spec(benchmark)
    if metrics_eval_data_dir:
        base = Path(metrics_eval_data_dir).expanduser()
        if not base.is_absolute():
            base = root / base
        return base / spec.subdir / f"eval_{spec.subdir}.jsonl"

    return benchmark_eval_jsonl(benchmark)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def setup_model(
    model_path: str | Path,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    vision_lora: bool = False,
    vision_lora_r: int = 32,
    vision_lora_alpha: int | None = None,
    init_adapter_path: str | Path | None = None,
    full_finetune: bool = False,
) -> PreTrainedModel:
    """Load the base model and apply LoRA to language model + (optionally) vision encoder.

    When ``vision_lora`` is True, LoRA adapters are added to both the language
    model and the vision encoder.  The vision encoder uses a lower rank
    (``vision_lora_r``, default 32) to preserve pretrained features while still
    allowing domain adaptation.

    When ``full_finetune`` is True, all loaded model parameters are trainable
    and no new LoRA adapter is created. If an ``init_adapter_path`` is provided,
    that PEFT adapter is merged into the base model before full fine-tuning.

    Different ranks are achieved via PEFT's ``rank_pattern`` / ``alpha_pattern``,
    which accept ``{regex: rank}`` dicts matched against full module paths.
    Modules under ``model.visual.*`` get the vision rank; everything else gets
    the default (language) rank.

    Args:
        model_path: Path to the pretrained model
        lora_r: LoRA rank for language model modules
        lora_alpha: LoRA alpha for language model modules
        lora_dropout: LoRA dropout rate (shared)
        vision_lora: Whether to add LoRA to vision encoder
        vision_lora_r: LoRA rank for vision encoder modules
        vision_lora_alpha: LoRA alpha for vision encoder (defaults to vision_lora_r)
        full_finetune: Train all base-model weights instead of adding LoRA

    Returns:
        PEFT model with LoRA adapters, or a full-finetune base model
    """
    if vision_lora_alpha is None:
        vision_lora_alpha = vision_lora_r

    LOGGER.info("Loading model from %s", model_path)

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Disable caching for training
    model.config.use_cache = False

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Required for gradient checkpointing with LoRA
    model.enable_input_require_grads()

    if full_finetune:
        if init_adapter_path is not None:
            LOGGER.info(
                "Full fine-tune requested; loading PEFT adapter from %s and merging into base",
                init_adapter_path,
            )
            model = PeftModel.from_pretrained(
                model,
                init_adapter_path,
                is_trainable=False,
            )
            model = model.merge_and_unload()
        for param in model.parameters():
            param.requires_grad_(True)
        LOGGER.info("Full fine-tune ENABLED: all model parameters are trainable")
        _log_trainable_parameter_count(model)
        _configure_generation_eos(model)
        return model

    if init_adapter_path is not None:
        LOGGER.info("Loading trainable PEFT adapter from %s", init_adapter_path)
        model = PeftModel.from_pretrained(
            model,
            init_adapter_path,
            is_trainable=True,
        )
        model.print_trainable_parameters()
        _configure_generation_eos(model)
        return model

    # Build target modules list
    if vision_lora:
        target_modules = LM_LORA_TARGET_MODULES + VISION_LORA_TARGET_MODULES
        LOGGER.info(
            "Vision LoRA ENABLED: LM rank=%d, vision rank=%d (alpha: LM=%d, vision=%d)",
            lora_r,
            vision_lora_r,
            lora_alpha,
            vision_lora_alpha,
        )
    else:
        target_modules = list(LM_LORA_TARGET_MODULES)
        LOGGER.info("Vision LoRA DISABLED: LM-only rank=%d, alpha=%d", lora_r, lora_alpha)

    # Build rank_pattern / alpha_pattern for per-module rank control.
    # PEFT's get_pattern_key() matches via re.match(rf"(.*\.)?({key})$", path),
    # so the pattern key must be a regex matching the SUFFIX of the full module
    # path. Using r"visual\..*" matches all modules under model.visual.* (e.g.
    # "model.visual.blocks.0.attn.qkv") without touching LM modules.
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    vision_pattern = r"visual\..*"
    if vision_lora and vision_lora_r != lora_r:
        rank_pattern[vision_pattern] = vision_lora_r
        alpha_pattern[vision_pattern] = vision_lora_alpha
        LOGGER.info(
            "Using rank_pattern={%r: %d}, alpha_pattern={%r: %d}",
            vision_pattern,
            vision_lora_r,
            vision_pattern,
            vision_lora_alpha,
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Log vision vs LM parameter breakdown
    if vision_lora:
        _log_parameter_breakdown(model)

    _configure_generation_eos(model)

    return model


def _log_trainable_parameter_count(model: PreTrainedModel) -> None:
    """Log trainable and total parameters for non-PEFT adaptation modes."""

    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    LOGGER.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )


def _configure_generation_eos(model: Any) -> None:
    """Set generation EOS ids on a base or PEFT-wrapped model."""

    # Include both <|endoftext|> (248044) and <|im_end|> (248046) to prevent
    # runaway generation.
    eos_ids_to_add: set[int] = {ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID}
    eos_token_id = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids_to_add.add(eos_token_id)
    elif isinstance(eos_token_id, list):
        eos_ids_to_add.update(eos_token_id)
    model.config.eos_token_id = sorted(eos_ids_to_add)

    LOGGER.info("Configured eos_token_id: %s", model.config.eos_token_id)


def _log_parameter_breakdown(model: PreTrainedModel) -> None:
    """Log trainable parameter counts split by vision vs language modules."""
    vision_params = 0
    lm_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if "visual" in name:
            vision_params += n
        elif "lora_" in name:
            lm_params += n
        else:
            other_params += n

    total = vision_params + lm_params + other_params
    LOGGER.info(
        "Trainable parameter breakdown: "
        "vision=%s (%.1f%%), LM=%s (%.1f%%), other=%s (%.1f%%), total=%s",
        f"{vision_params:,}",
        100.0 * vision_params / max(total, 1),
        f"{lm_params:,}",
        100.0 * lm_params / max(total, 1),
        f"{other_params:,}",
        100.0 * other_params / max(total, 1),
        f"{total:,}",
    )


def setup_processor(model_path: str | Path) -> ProcessorMixin:
    """Load and configure the processor.

    Sets padding_side to "right" to ensure consistent label masking.
    """
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    # Some Qwen3.5 checkpoints ship the chat template on the tokenizer but not
    # the processor.  Copy it over so processor.apply_chat_template() works.
    if processor.chat_template is None and getattr(processor.tokenizer, "chat_template", None):
        processor.chat_template = processor.tokenizer.chat_template
    # Ensure right-padding for consistent label masking during training
    processor.tokenizer.padding_side = "right"
    return processor


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


# Filename of the resolved-config snapshot written next to each run's
# checkpoints. Shipped alongside adapter weights so a checkpoint always carries
# the exact recipe it trained with.
RESOLVED_CONFIG_FILENAME = "sft_config.resolved.yaml"


def persist_resolved_config(config: SFTConfig, output_dir: str | Path) -> Path | None:
    """Write the fully-resolved ``SFTConfig`` next to the run's checkpoints.

    Produces ``<output_dir>/sft_config.resolved.yaml`` at training start so any
    checkpoint carries the exact recipe it trained with. The published HF repo
    ships only adapter weights + ``adapter_config.json``, and the W&B run
    records only HF ``TrainingArguments``, so this file is the authoritative
    on-disk provenance record. Defensive: logs a warning and returns ``None``
    instead of crashing training if the path is not writable.
    """
    path = Path(output_dir) / RESOLVED_CONFIG_FILENAME
    try:
        with open(path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=True)
    except OSError as exc:
        LOGGER.warning("Failed to persist resolved config to %s: %s (continuing).", path, exc)
        return None
    LOGGER.info("Saved resolved training config to %s", path)
    return path


def train(config: SFTConfig) -> None:
    """Run SFT training with the given configuration.

    Args:
        config: Training configuration
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Resolve paths relative to the project root.
    root = Path(__file__).resolve().parents[3]
    LOGGER.info("Starting SFT training with config: %s", config.to_dict())
    validate_best_metric_config(config, root=root)
    set_seed(config.seed)

    model_path = _resolve_model_identifier(config.model_path, root)
    # A ``hub:`` data path auto-downloads the frozen flat training index from
    # EdisonScientific/glyph-datasets (``hub:`` or ``hub:markush/train.jsonl``),
    # so no manual huggingface-cli step is needed.
    if isinstance(config.data_path, str) and config.data_path.startswith("hub:"):
        from glyph.markush.train.data import resolve_markush_train_jsonl

        data_path = resolve_markush_train_jsonl(config.data_path[len("hub:") :] or None)
    else:
        data_path = Path(config.data_path)
        if not data_path.is_absolute():
            data_path = root / data_path
    init_adapter_path = Path(config.init_adapter_path) if config.init_adapter_path else None
    if init_adapter_path is not None and not init_adapter_path.is_absolute():
        init_adapter_path = root / init_adapter_path
    # Image root is the configured root for prepared JSONL image paths.
    image_root = data_root()
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    logging_dir = Path(config.logging_dir) if config.logging_dir else output_dir / "logs"
    if not logging_dir.is_absolute():
        logging_dir = root / logging_dir

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Save the fully-resolved config next to the checkpoints for provenance.
    persist_resolved_config(config, output_dir)

    # Resolve vision_lora_alpha default
    vision_lora_alpha = config.vision_lora_alpha
    if vision_lora_alpha is None:
        vision_lora_alpha = config.vision_lora_r

    # Load model and processor
    processor = setup_processor(model_path)
    model = setup_model(
        model_path,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        vision_lora=config.vision_lora,
        vision_lora_r=config.vision_lora_r,
        vision_lora_alpha=vision_lora_alpha,
        init_adapter_path=init_adapter_path,
        full_finetune=config.full_finetune,
    )
    if config.fused_cross_entropy:
        install_fused_cross_entropy(model)

    # Load data
    LOGGER.info("Loading data from %s", data_path)
    records = load_records(data_path)
    LOGGER.info("Loaded %d records", len(records))
    records, text_filter_stats = filter_records_by_text_length(
        records,
        max_user_chars=config.max_user_chars,
        max_assistant_chars=config.max_assistant_chars,
    )
    if text_filter_stats["dropped_user"] or text_filter_stats["dropped_assistant"]:
        LOGGER.info(
            "Text-length filter (max_user_chars=%s, max_assistant_chars=%s): kept %d, "
            "dropped %d over-long user / %d over-long assistant",
            config.max_user_chars,
            config.max_assistant_chars,
            text_filter_stats["kept"],
            text_filter_stats["dropped_user"],
            text_filter_stats["dropped_assistant"],
        )

    # Split into train/eval. Fail loudly if the dataset is too small to leave
    # any training records after carving off the eval slice — auto-shrinking
    # would just hide a misconfigured eval_size.
    if len(records) <= config.eval_size:
        raise ValueError(
            f"Dataset has {len(records)} records but eval_size={config.eval_size}; "
            "no training records would remain. Reduce eval_size or use a larger dataset."
        )
    gen = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(records), generator=gen).tolist()
    eval_indices = set(indices[: config.eval_size])
    train_records = [r for i, r in enumerate(records) if i not in eval_indices]
    eval_records = [r for i, r in enumerate(records) if i in eval_indices]
    LOGGER.info("Split: %d train / %d eval", len(train_records), len(eval_records))

    # Create datasets
    train_dataset = MarkushDataset(
        train_records, image_root, text_dropout=config.text_dropout, seed=config.seed
    )
    eval_dataset = MarkushDataset(
        eval_records,
        image_root,
        text_dropout=0.0,
        seed=config.seed,  # No dropout for eval
    )

    # Configure W&B
    if config.no_wandb:
        report_to = []
        os.environ["WANDB_MODE"] = "disabled"
    else:
        report_to = config.report_to
        os.environ["WANDB_PROJECT"] = config.wandb_project
        if config.wandb_entity:
            os.environ["WANDB_ENTITY"] = config.wandb_entity
        if config.wandb_mode:
            os.environ["WANDB_MODE"] = config.wandb_mode

    # Generate run name — include vision LoRA info when enabled
    if config.run_name:
        run_name = config.run_name
    elif config.vision_lora:
        run_name = (
            f"markush-sft-vlora-r{config.lora_r}-vr{config.vision_lora_r}-td{config.text_dropout}"
        )
    else:
        run_name = f"markush-sft-r{config.lora_r}-td{config.text_dropout}"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        report_to=report_to,
        run_name=run_name,
        # Batch sizes
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Learning rate and schedule
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        # Training length
        max_steps=config.max_steps,
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        # Logging
        logging_steps=config.logging_steps,
        logging_first_step=True,
        # Precision
        bf16=config.bf16,
        # Optimization
        optim=config.optim,
        gradient_checkpointing=True,
        # DDP settings
        ddp_find_unused_parameters=False,
        # Data loading
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_prefetch_factor=(
            config.dataloader_prefetch_factor if config.dataloader_num_workers > 0 else None
        ),
        remove_unused_columns=False,
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
    )

    # Create collator
    collator = MarkushCollator(
        processor,
        image_augmentation=config.image_augmentation,
        image_augmentation_prob=config.image_augmentation_prob,
        seed=config.seed,
    )

    # Setup callbacks
    callbacks: list[TrainerCallback] = [
        DetailedLoggingCallback(),
    ]
    # Capture the full resolved SFTConfig in the W&B run config (HF only logs
    # TrainingArguments). Skipped entirely when W&B is disabled.
    if not config.no_wandb:
        callbacks.append(ResolvedConfigCallback(config.to_dict()))
    metrics_callbacks: list[MetricsEvalCallback] = []

    # Chemistry-aware MCS eval on a small subset, scored under both protocols.
    if config.metrics_eval_size > 0:
        # Guard the silent-gap footgun: if the chemistry eval cadence never
        # divides into max_steps, the callback never fires and the run looks
        # "evaluated" while producing no metrics (this has bitten a short 4B
        # run before). Warn loudly rather than fail, since short runs
        # legitimately may not reach the first eval.
        effective_metrics_steps = config.metrics_eval_steps or config.eval_steps
        if (
            effective_metrics_steps
            and config.max_steps
            and (config.max_steps < effective_metrics_steps)
        ):
            LOGGER.warning(
                "metrics_eval_size>0 but max_steps=%d < metrics-eval cadence=%d: the "
                "chemistry eval will NEVER fire. Lower metrics_eval_steps/eval_steps to "
                "exercise it (short runs should set metrics_eval_steps<=max_steps).",
                config.max_steps,
                effective_metrics_steps,
            )
        metric_jobs: list[tuple[str, Path, str]] = []
        if config.metrics_eval_benchmarks:
            for name in config.metrics_eval_benchmarks:
                spec = get_benchmark_spec(name)
                eval_jsonl = _resolve_metrics_benchmark_path(
                    name,
                    root=root,
                    metrics_eval_data_dir=config.metrics_eval_data_dir,
                )
                metric_jobs.append((spec.name, eval_jsonl, spec.prompt))
        elif config.metrics_eval_data:
            # Load one external eval dataset (e.g. IP5-M real patent images).
            # Expected fields: id, image, annotation or cxsmiles_opt, optionally prompt.
            eval_jsonl = Path(config.metrics_eval_data)
            if not eval_jsonl.is_absolute():
                eval_jsonl = root / config.metrics_eval_data
            if config.metrics_eval_benchmark:
                spec = get_benchmark_spec(config.metrics_eval_benchmark)
                metric_jobs.append((spec.name, eval_jsonl, spec.prompt))
            else:
                metric_jobs.append(("external", eval_jsonl, PATENT_IMAGE_MARKUSH_PROMPT))

        if metric_jobs:
            data_dir = root / "data"
            for job_name, eval_jsonl, prompt in metric_jobs:
                LOGGER.info("Loading external %s eval data from %s", job_name, eval_jsonl)
                metric_samples = _load_metrics_samples(
                    eval_jsonl,
                    data_dir=data_dir,
                    limit=config.metrics_eval_size,
                    prompt=prompt,
                    canonicalize_markush_targets=config.metrics_eval_canonicalize_markush_targets,
                    require_canonical_markush_targets=(
                        config.metrics_eval_require_canonical_markush_targets
                    ),
                    markush_prompt_policy=config.metrics_eval_markush_prompt_policy,
                )
                prefix = f"eval_{job_name.replace('-', '_')}"
                metrics_callbacks.append(
                    MetricsEvalCallback(
                        samples=metric_samples,
                        processor=processor,
                        protocol=config.metrics_eval_protocol,
                        max_new_tokens=config.metrics_eval_max_new_tokens,
                        batch_size=config.metrics_eval_batch_size,
                        prefix=prefix,
                        interval_steps=config.metrics_eval_steps,
                    )
                )
                LOGGER.info(
                    "MetricsEvalCallback enabled: %d samples from %s, protocol=%s, prefix=%s",
                    len(metric_samples),
                    eval_jsonl,
                    config.metrics_eval_protocol,
                    prefix,
                )
        else:
            # Carve the subset from the eval split so we don't pull from train.
            # Cap at the eval split size; fall back to whole eval set if smaller.
            n_metric = min(config.metrics_eval_size, len(eval_dataset))
            if n_metric < config.metrics_eval_size:
                LOGGER.warning(
                    "metrics_eval_size=%d exceeds eval split (%d); using %d.",
                    config.metrics_eval_size,
                    len(eval_dataset),
                    n_metric,
                )
            metric_samples = [eval_dataset[i] for i in range(n_metric)]
            # eval_dataset.__getitem__ returns dicts with id/image/user_text/assistant_text;
            # MetricsEvalCallback consumes those directly.
            metrics_callbacks.append(
                MetricsEvalCallback(
                    samples=metric_samples,
                    processor=processor,
                    protocol=config.metrics_eval_protocol,
                    max_new_tokens=config.metrics_eval_max_new_tokens,
                    batch_size=config.metrics_eval_batch_size,
                    prefix="eval_train",
                    interval_steps=config.metrics_eval_steps,
                )
            )
            LOGGER.info(
                "MetricsEvalCallback enabled: %d samples from training holdout, "
                "protocol=%s, prefix=eval_train",
                n_metric,
                config.metrics_eval_protocol,
            )

    if config.ocsr_metrics_eval_size > 0:
        if not config.ocsr_metrics_eval_data:
            raise ValueError(
                "ocsr_metrics_eval_size > 0 requires --ocsr-metrics-eval-data "
                "pointing to an OCSR eval JSONL or directory of eval_ocsr_*.jsonl files."
            )
        effective_ocsr_metrics_steps = (
            config.ocsr_metrics_eval_steps or config.metrics_eval_steps or config.eval_steps
        )
        if (
            effective_ocsr_metrics_steps
            and config.max_steps
            and (config.max_steps < effective_ocsr_metrics_steps)
        ):
            LOGGER.warning(
                "ocsr_metrics_eval_size>0 but max_steps=%d < OCSR metrics-eval cadence=%d: "
                "the OCSR eval will NEVER fire. Lower ocsr_metrics_eval_steps/eval_steps "
                "to exercise it.",
                config.max_steps,
                effective_ocsr_metrics_steps,
            )
        ocsr_eval_specs = _resolve_ocsr_metrics_eval_specs(
            config,
            root=root,
            require_exists=True,
        )
        for spec in ocsr_eval_specs:
            LOGGER.info("Loading OCSR eval data from %s", spec.path)
            ocsr_metric_samples = _load_ocsr_metrics_samples(
                spec.path,
                data_dir=spec.path.parent,
                limit=config.ocsr_metrics_eval_size,
            )
            _require_ocsr_metric_samples(ocsr_metric_samples, spec.path)
            metrics_callbacks.append(
                OCSRMetricsEvalCallback(
                    samples=ocsr_metric_samples,
                    processor=processor,
                    max_new_tokens=config.ocsr_metrics_eval_max_new_tokens,
                    batch_size=config.ocsr_metrics_eval_batch_size
                    or config.metrics_eval_batch_size,
                    prefix=spec.prefix,
                    interval_steps=config.ocsr_metrics_eval_steps
                    if config.ocsr_metrics_eval_steps is not None
                    else config.metrics_eval_steps,
                    prediction_mode=config.ocsr_metrics_prediction_mode,
                )
            )
            LOGGER.info(
                "OCSRMetricsEvalCallback enabled: %d samples from %s, prefix=%s, "
                "prediction_mode=%s",
                len(ocsr_metric_samples),
                spec.path,
                spec.prefix,
                config.ocsr_metrics_prediction_mode,
            )

    if metrics_callbacks:
        callbacks.append(MultiMetricsEvalCallback(metrics_callbacks))

    # Detect existing checkpoints in output_dir. HF Trainer will silently
    # auto-resume from these; warn the user so they're not surprised when
    # they think they're starting fresh.
    existing_checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if existing_checkpoints:
        latest = existing_checkpoints[-1].name
        LOGGER.warning(
            "Detected existing checkpoint in %s: %s. Trainer will auto-resume. "
            "Pass a different --output-dir to start fresh.",
            output_dir,
            latest,
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        callbacks=callbacks,
    )
    # Trainer installs default reporting callbacks before user callbacks. Put the
    # processor saver first so W&B checkpoint artifact packaging sees processor
    # files that are written during the same on_save event.
    install_processor_save_callback_first(trainer, processor)

    # Train
    LOGGER.info("Starting training...")
    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    LOGGER.info("Saving final model to %s", final_dir)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    LOGGER.info("Training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file. Command-line args override config values.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Root for MarkushGlyph data (default: GLYPH_DATA_ROOT or ./data).",
    )

    # Model and data
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to pretrained model (default: Qwen/Qwen3.5-2B-Base)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help=(
            "Path to training data JSONL (default: hub:markush/train.jsonl). "
            "Use 'hub:markush/train.jsonl' (or bare 'hub:') to auto-download the "
            "frozen flat training index from EdisonScientific/glyph-datasets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for checkpoints (default: checkpoints/sft)",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        help="Directory for logs (default: <output_dir>/logs)",
    )
    parser.add_argument(
        "--init-adapter-path",
        type=str,
        help=(
            "Existing PEFT adapter checkpoint to load as trainable before continuing SFT. "
            "When set, fresh LoRA initialization is skipped."
        ),
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        default=None,
        dest="full_finetune",
        help=(
            "Train all model weights instead of creating a LoRA adapter. If "
            "--init-adapter-path is also set, merge that adapter into the base first."
        ),
    )
    parser.add_argument(
        "--no-full-finetune",
        action="store_false",
        dest="full_finetune",
        help="Disable full fine-tuning and use the LoRA path.",
    )

    # Data processing
    parser.add_argument(
        "--text-dropout",
        type=float,
        help="Probability of dropping OCR/patent text (default: 0.5)",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        help="Number of samples for evaluation set (default: 1000)",
    )
    parser.add_argument(
        "--image-augmentation",
        action="store_true",
        default=None,
        dest="image_augmentation",
        help="Enable lightweight on-the-fly PIL image augmentation (default: off)",
    )
    parser.add_argument(
        "--no-image-augmentation",
        action="store_false",
        dest="image_augmentation",
        help="Disable on-the-fly PIL image augmentation",
    )
    parser.add_argument(
        "--image-augmentation-prob",
        type=float,
        help="Probability of applying image augmentation to each training image (default: 0.6)",
    )

    # LoRA — language model
    parser.add_argument(
        "--lora-r",
        type=int,
        help="LoRA rank for language model (default: 128)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        help="LoRA alpha/scaling for language model (default: 128)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        help="LoRA dropout (default: 0.05)",
    )

    # LoRA — vision encoder
    vision_group = parser.add_argument_group("Vision encoder LoRA")
    vision_group.add_argument(
        "--vision-lora",
        action="store_true",
        default=None,
        dest="vision_lora",
        help="Enable LoRA on vision encoder (default: off)",
    )
    vision_group.add_argument(
        "--no-vision-lora",
        action="store_false",
        dest="vision_lora",
        help="Disable LoRA on vision encoder",
    )
    vision_group.add_argument(
        "--vision-lora-r",
        type=int,
        help="LoRA rank for vision encoder (default: 32). Lower than LM rank to "
        "preserve pretrained features while allowing domain adaptation.",
    )
    vision_group.add_argument(
        "--vision-lora-alpha",
        type=int,
        help="LoRA alpha for vision encoder (default: same as --vision-lora-r).",
    )

    # Training
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Warmup steps (default: 200)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps (default: 30000)",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        help="Per-device training batch size (default: 2)",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        help="Per-device batch size for the HF loss eval loop (default: 1).",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        help="Dataloader workers per rank (default: 4).",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        help="Batches prefetched per dataloader worker (default: PyTorch's 2).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Gradient accumulation steps (default: 2)",
    )

    # Checkpointing
    parser.add_argument(
        "--save-steps",
        type=int,
        help="Save checkpoint every N steps (default: 1000)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        help="Evaluate every N steps (default: 2000)",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        help="Maximum number of checkpoints to keep (default: 5)",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        help="Log every N steps (default: 5)",
    )

    # W&B
    parser.add_argument(
        "--run-name",
        type=str,
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project (overrides YAML and WANDB_PROJECT).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity (overrides YAML and WANDB_ENTITY).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        help="W&B operating mode; offline avoids network access.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=None,
        help="Disable W&B logging and network access",
    )

    # MCS metric eval during training
    parser.add_argument(
        "--metrics-eval-size",
        type=int,
        help=(
            "Number of held-out samples to run generation + MCS scoring on at "
            "every eval step. 0 disables (default). Both 'ours' and 'mg2' "
            "protocols are recorded under eval/ours/* and eval/mg2/*."
        ),
    )
    parser.add_argument(
        "--metrics-eval-data",
        type=str,
        help=(
            "External eval JSONL (e.g. IP5-M) for inline MCS scoring. "
            "When set, the callback evaluates on these samples instead of "
            "a training holdout. Expected fields: id, image, annotation or "
            "cxsmiles_opt. Paths resolved relative to project root."
        ),
    )
    parser.add_argument(
        "--metrics-eval-benchmark",
        choices=("ip5m", "m2s", "uspto-markush"),
        help="Prompt preset for --metrics-eval-data.",
    )
    parser.add_argument(
        "--metrics-eval-benchmarks",
        nargs="+",
        choices=("ip5m", "m2s", "uspto-markush"),
        help=(
            "Named benchmark callbacks to run from their default local JSONL paths. "
            "Use with metrics_eval_size to track IP5-M/M2S/USPTO-Markush together."
        ),
    )
    parser.add_argument(
        "--metrics-eval-data-dir",
        type=str,
        help=(
            "Directory containing benchmark JSONLs named eval_ip5m.jsonl, "
            "eval_m2s.jsonl, etc. Used with --metrics-eval-benchmarks."
        ),
    )
    parser.add_argument(
        "--metrics-eval-protocol",
        choices=("ours", "mg2", "both"),
        help="Protocol(s) to record in MetricsEvalCallback (default: both).",
    )
    parser.add_argument(
        "--metrics-eval-max-new-tokens",
        type=int,
        help="Max new tokens for MetricsEvalCallback generation (default: 512).",
    )
    parser.add_argument(
        "--max-user-chars",
        type=int,
        help="Drop training rows whose user text exceeds this many chars (0 disables).",
    )
    parser.add_argument(
        "--max-assistant-chars",
        type=int,
        help=(
            "Drop training rows whose assistant text exceeds this many chars "
            "(default 4000, ~0.006%% of the train set; 0 disables). Long rows set "
            "the per-batch CE memory peak."
        ),
    )
    parser.add_argument(
        "--fused-cross-entropy",
        dest="fused_cross_entropy",
        action="store_const",
        const=True,
        default=None,
        help="Use FLA's fused Triton cross-entropy for the LM loss (default: on).",
    )
    parser.add_argument(
        "--no-fused-cross-entropy",
        dest="fused_cross_entropy",
        action="store_const",
        const=False,
        help="Disable the fused cross-entropy loss override.",
    )
    parser.add_argument(
        "--metrics-eval-batch-size",
        type=int,
        help=(
            "Per-rank generation batch size for MetricsEvalCallback (default: 1). "
            "Raise after a small cluster validation run to speed up large inline evals."
        ),
    )
    parser.add_argument(
        "--metrics-eval-steps",
        type=int,
        help=(
            "Run MetricsEvalCallback only on steps divisible by this value "
            "(default: every eval_steps)."
        ),
    )
    parser.add_argument(
        "--metrics-eval-canonicalize-markush-targets",
        action="store_true",
        # default=None (not False) so an unset flag stays None and never
        # overrides a `true` from the YAML config. See resolve_config().
        default=None,
        help="Prepare MetricsEvalCallback Markush targets as verified canonicalized cxsmiles_opt.",
    )
    parser.add_argument(
        "--metrics-eval-require-canonical-markush-targets",
        action="store_true",
        # default=None (not False) so an unset flag stays None and never
        # overrides a `true` from the YAML config. See resolve_config().
        default=None,
        help="Skip MetricsEvalCallback Markush rows whose canonicalized target cannot be verified.",
    )
    parser.add_argument(
        "--metrics-eval-markush-prompt-policy",
        choices=MARKUSH_PROMPT_POLICIES,
        help=(
            "Optional Markush prompt wording policy for inline Markush metrics eval. "
            "Default preserves benchmark/prepared prompts."
        ),
    )
    parser.add_argument(
        "--ocsr-metrics-eval-size",
        type=int,
        help=(
            "Number of OCSR eval samples to run generation + strict RDKit-canonical "
            "SMILES scoring on at eval time. 0 disables (default)."
        ),
    )
    parser.add_argument(
        "--ocsr-metrics-eval-data",
        type=str,
        help=(
            "OCSR eval JSONL, or a directory containing eval_ocsr_*.jsonl files, "
            "for inline SMILES scoring. Rows may be unified OCSR records or "
            "prepared molecule_ocsr SFT rows."
        ),
    )
    parser.add_argument(
        "--ocsr-metrics-eval-name",
        type=str,
        help="Name used in the OCSR metric prefix eval_ocsr_<name> (default: ocsr).",
    )
    parser.add_argument(
        "--ocsr-metrics-eval-max-new-tokens",
        type=int,
        help="Max new tokens for OCSRMetricsEvalCallback generation (default: 256).",
    )
    parser.add_argument(
        "--ocsr-metrics-eval-batch-size",
        type=int,
        help=(
            "Per-rank generation batch size for OCSRMetricsEvalCallback. "
            "Defaults to metrics_eval_batch_size."
        ),
    )
    parser.add_argument(
        "--ocsr-metrics-eval-steps",
        type=int,
        help=(
            "Run OCSRMetricsEvalCallback only on steps divisible by this value "
            "(default: metrics_eval_steps/every eval_steps)."
        ),
    )
    parser.add_argument(
        "--ocsr-metrics-prediction-mode",
        type=str,
        help=(
            "OCSR metrics prediction parser. Default native_smiles requires bare SMILES. "
            "markush_no_extension accepts no-extension Markush/XML outputs but rejects "
            "Markush annotations."
        ),
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (default: 42)",
    )

    return parser


# Environment variables that override YAML config values (CLI still wins).
_ENV_CONFIG_OVERRIDES: dict[str, str] = {
    "WANDB_PROJECT": "wandb_project",
    "WANDB_ENTITY": "wandb_entity",
    "WANDB_MODE": "wandb_mode",
}


def explicit_cli_keys(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> set[str]:
    """Return the set of dest names the user *actually* passed on the CLI.

    argparse cannot otherwise distinguish "user passed ``--flag``" from "flag
    left off, so argparse supplied its default". A ``store_true`` flag left off
    still lands in the namespace as ``False``; treating that ``False`` as an
    explicit value is exactly the silent-override bug this guards against. We
    pre-seed a namespace with a unique sentinel per dest and re-parse: only
    options the user supplied get overwritten, so anything still holding the
    sentinel was not passed.
    """
    sentinel = object()
    namespace = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            setattr(namespace, action.dest, sentinel)
    parser.parse_args(argv, namespace=namespace)
    return {dest for dest, value in vars(namespace).items() if value is not sentinel}


def detect_silent_config_overrides(
    yaml_config: dict[str, Any],
    resolved_config: dict[str, Any],
    allowed_keys: set[str],
) -> dict[str, tuple[Any, Any]]:
    """Find config keys whose resolved value diverged from the YAML silently.

    Returns ``{key: (yaml_value, resolved_value)}`` for every key that changed
    between the YAML-loaded config and the fully-resolved config *without* an
    explicit CLI/env override in ``allowed_keys``. An empty dict means every
    change is accounted for. A non-empty dict signals the ``store_true``
    ``default`` bug class (or any other logic silently clobbering the YAML).
    """
    drift: dict[str, tuple[Any, Any]] = {}
    for key, yaml_value in yaml_config.items():
        if key in allowed_keys:
            continue
        resolved_value = resolved_config.get(key)
        if resolved_value != yaml_value:
            drift[key] = (yaml_value, resolved_value)
    return drift


def resolve_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> SFTConfig:
    """Merge YAML config, environment, and *explicit* CLI overrides.

    Precedence (lowest to highest): dataclass defaults < YAML file < environment
    (see ``_ENV_CONFIG_OVERRIDES``) < CLI flags the user actually passed. Only
    keys the user explicitly typed override the YAML, so an argparse-supplied
    default (e.g. a ``store_true`` flag left off) can never clobber a YAML
    value. Logs the fully-resolved config and raises if any key drifted from the
    YAML without an explicit override, keeping this class of silent override
    visible and non-regressable.
    """
    environ: Mapping[str, str] = os.environ if env is None else env

    config = SFTConfig.from_yaml(args.config) if getattr(args, "config", None) else SFTConfig()
    yaml_snapshot = config.to_dict()

    allowed_keys: set[str] = set()

    # Environment overrides take precedence over YAML; explicit CLI values win below.
    for env_key, attr in _ENV_CONFIG_OVERRIDES.items():
        if environ.get(env_key):
            setattr(config, attr, environ[env_key])
            allowed_keys.add(attr)

    # Apply only the options the user explicitly passed on the command line.
    for key in explicit_cli_keys(parser, argv):
        if key == "config":
            continue
        value = getattr(args, key, None)
        if hasattr(config, key):
            setattr(config, key, value)
            allowed_keys.add(key)

    resolved = config.to_dict()
    LOGGER.info(
        "Resolved training config:\n%s",
        json.dumps(resolved, indent=2, sort_keys=True, default=str),
    )

    drift = detect_silent_config_overrides(yaml_snapshot, resolved, allowed_keys)
    if drift:
        details = ", ".join(
            f"{key}: yaml={yaml_value!r} -> resolved={resolved_value!r}"
            for key, (yaml_value, resolved_value) in sorted(drift.items())
        )
        raise RuntimeError(
            "Config resolution silently changed values that were neither set in "
            f"the YAML nor passed on the CLI: {details}. This usually means an "
            "argparse store_true/store_false flag is missing `default=None` and "
            "its argparse default is overriding the YAML. Add `default=None` to "
            "that flag in build_argument_parser()."
        )

    return config


def main() -> None:
    """Main entry point for SFT training."""
    parser = build_argument_parser()
    args = parser.parse_args()

    # GLYPH_DATA_ROOT must be set before from_yaml() so ${GLYPH_DATA_ROOT}
    # config-var expansion resolves against the requested root.
    if args.data_root is not None:
        os.environ["GLYPH_DATA_ROOT"] = str(data_root(args.data_root))

    config = resolve_config(args, parser)

    # Run training
    train(config)


if __name__ == "__main__":
    main()
