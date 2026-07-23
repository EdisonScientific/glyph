#!/usr/bin/env python3
"""Tests for glyph.markush.train.sft.

These tests cover everything that can be exercised without loading the actual
Qwen-VL weights or running on a GPU:

- ``SFTConfig.from_yaml`` parsing and key normalization
- ``MarkushDataset`` text-dropout determinism (used for DDP reproducibility)
- ``MarkushCollator.__call__`` label-masking and padding handling
- ``load_records`` JSONL parsing

We stub out the VL processor with a deterministic mock that emulates
chat-template formatting and vision-token expansion. Real Qwen-VL weight
behaviour (gradient, generation, dtype) is out of scope here.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
import torch
import yaml
from PIL import Image

from glyph.markush.data.example_modes import (
    MOLECULE_OCSR,
    STRUCTURE_ONLY_MARKUSH,
    STRUCTURE_ONLY_MARKUSH_PROMPT,
)
from glyph.markush.data.prepare_sft import build_training_example
from glyph.markush.data.prompt_policies import (
    MARKUSH_PROMPT_POLICY_VISIBLE_SG,
    VISIBLE_SG_INSTRUCTION,
)
from glyph.markush.eval.fast_eval import load_eval_samples as load_fast_eval_samples
from glyph.markush.eval.ocsr import (
    OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
    OCSREvaluationConfig,
    evaluate_ocsr_smiles,
    flatten_ocsr_metrics,
)
from glyph.markush.train.sft import (
    PROMPT_DELIMITER,
    RESOLVED_CONFIG_FILENAME,
    DetailedLoggingCallback,
    MarkushCollator,
    MarkushDataset,
    MetricsEvalCallback,
    MultiMetricsEvalCallback,
    OCSRMetricsEvalCallback,
    ResolvedConfigCallback,
    SaveProcessorCallback,
    SFTConfig,
    _load_metrics_samples,
    _load_ocsr_metrics_samples,
    _require_ocsr_metric_samples,
    _resolve_metrics_benchmark_path,
    _resolve_model_identifier,
    _resolve_ocsr_metrics_eval_specs,
    build_argument_parser,
    compute_inline_eval_aggregate_metrics,
    detect_silent_config_overrides,
    expected_best_metric_keys,
    explicit_cli_keys,
    filter_records_by_text_length,
    install_fused_cross_entropy,
    install_processor_save_callback_first,
    persist_resolved_config,
    resolve_config,
    setup_model,
    validate_best_metric_config,
)

# ---------------------------------------------------------------------------
# Mock processor / tokenizer
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Stand-in for ``processor.tokenizer`` used in the collator."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 99
        self.eos_token = "<eos>"
        self.padding_side = "right"


class _MockProcessor:
    """Mock VL processor.

    Tokenizes by whitespace + character with two rules:
      * ``<image>`` expands to ``N_VIS`` consecutive vision-token ids (1).
      * Every other word becomes one token id seeded by hashing the word.
    Always appends an EOS id (99) IF the text ends with the literal ``<eos>``
    token (matching how the collator builds ``full_text``).

    Also provides ``apply_chat_template`` which formats messages into a flat
    string with ``<image>`` placeholders, matching the Qwen VL processor API.
    """

    N_VIS = 4  # Number of vision-tokens emitted per <image>
    VISION_TOKEN_ID = 1

    def __init__(self) -> None:
        self.tokenizer = _MockTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        """Format messages into a flat text string with <image> placeholders."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        parts.append("<image>")
                    elif item.get("type") == "text":
                        parts.append(item["text"])
            else:
                parts.append(content)
            parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _tokenize(self, text: str) -> list[int]:
        tokens: list[int] = []
        # Split on <image> first, expand each occurrence to N_VIS vision tokens.
        chunks = text.split("<image>")
        for i, chunk in enumerate(chunks):
            if i > 0:
                tokens.extend([self.VISION_TOKEN_ID] * self.N_VIS)
            # Naive word/punct tokenizer
            for word in chunk.replace("\n", " <nl> ").split():
                if word == "<eos>":
                    tokens.append(self.tokenizer.eos_token_id)
                else:
                    # Hash word to a stable non-pad, non-eos, non-vision id
                    tid = (abs(hash(word)) % 2000) + 100  # ids 100..2099
                    tokens.append(tid)
        return tokens

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image] | None = None,
        padding: bool | str = True,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        sequences = [self._tokenize(t) for t in text]
        if truncation and max_length is not None:
            sequences = [s[:max_length] for s in sequences]
        if padding:
            max_len = max((len(s) for s in sequences), default=0)
            if max_length is not None:
                max_len = min(max_len, max_length)
            # Right padding (matches the configured padding_side)
            sequences = [s + [self.tokenizer.pad_token_id] * (max_len - len(s)) for s in sequences]
        input_ids = torch.tensor(sequences, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if images is not None:
            # Pretend vision features (B, 3, 224, 224); shape isn't validated by collator.
            out["pixel_values"] = torch.zeros(len(images), 3, 4, 4)
        return out


def _mock_process_vision_info(messages):
    """Mock replacement for qwen_vl_utils.process_vision_info.

    Extracts image paths from message dicts and opens them as PIL Images.
    Returns (images, videos) matching the real API.
    """
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image":
                    path = item.get("image")
                    if path:
                        img = Image.open(path)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        images.append(img)
    return images, []


# ---------------------------------------------------------------------------
# SFTConfig
# ---------------------------------------------------------------------------


class TestSFTConfig:
    def test_from_yaml_normalizes_hyphens(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "model-path: foo\ndata-path: bar.jsonl\ntext-dropout: 0.25\nlora-r: 64\n"
        )
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.model_path == "foo"
        assert cfg.data_path == "bar.jsonl"
        assert cfg.text_dropout == 0.25
        assert cfg.lora_r == 64
        # Defaults preserved
        assert cfg.lora_alpha == 128

    def test_from_yaml_expands_data_root_in_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GLYPH_DATA_ROOT", "/tmp/mkroot")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "model_path: ${GLYPH_DATA_ROOT}/models/base\n"
            "data_path: ${GLYPH_DATA_ROOT}/prepared/train.jsonl\n"
        )

        cfg = SFTConfig.from_yaml(cfg_path)

        assert cfg.model_path == "/tmp/mkroot/models/base"
        assert cfg.data_path == "/tmp/mkroot/prepared/train.jsonl"

    def test_wandb_cli_overrides_are_available(self) -> None:
        args = build_argument_parser().parse_args(
            [
                "--wandb-project",
                "public-project",
                "--wandb-entity",
                "public-entity",
                "--wandb-mode",
                "offline",
            ]
        )
        cfg = SFTConfig.from_args(args)

        assert cfg.wandb_project == "public-project"
        assert cfg.wandb_entity == "public-entity"
        assert cfg.wandb_mode == "offline"

    def test_from_yaml_silently_drops_unknown_keys(self, tmp_path: Path) -> None:
        # Documenting behaviour: unknown keys are dropped, not raised. This is
        # a known footgun (typos silently ignored) flagged in the audit.
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model_path: foo\ntotaly_not_a_field: 42\n")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.model_path == "foo"
        assert not hasattr(cfg, "totaly_not_a_field")

    def test_from_yaml_parses_full_finetune(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("model_path: foo\nfull_finetune: true\n")

        cfg = SFTConfig.from_yaml(cfg_path)

        assert cfg.model_path == "foo"
        assert cfg.full_finetune is True

    def test_from_yaml_empty(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.model_path == "Qwen/Qwen3.5-2B-Base"

    def test_to_dict_serializes_paths(self) -> None:
        cfg = SFTConfig()
        d = cfg.to_dict()
        assert d["model_path"] == "Qwen/Qwen3.5-2B-Base"
        assert "learning_rate" in d

    def test_scientific_notation_coerced_to_float(self, tmp_path: Path) -> None:
        # PyYAML parses `2e-4` (no decimal point) as a string; SFTConfig must
        # coerce based on the dataclass annotation.
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("learning_rate: 2e-4\nmax_steps: 5000\n")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert isinstance(cfg.learning_rate, float)
        assert cfg.learning_rate == 2e-4
        assert isinstance(cfg.max_steps, int)
        assert cfg.max_steps == 5000

    def test_real_sft_yaml_learning_rate_is_float(self) -> None:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "sft_2b.yaml"
        if not cfg_path.exists():
            pytest.skip(f"missing real config at {cfg_path}")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert isinstance(cfg.learning_rate, float), (
            f"sft_2b.yaml learning_rate must coerce to float, got {type(cfg.learning_rate)}"
        )

    def test_vision_lora_defaults(self) -> None:
        cfg = SFTConfig()
        assert cfg.vision_lora is False
        assert cfg.vision_lora_r == 32
        assert cfg.vision_lora_alpha is None
        assert cfg.image_augmentation is False
        assert cfg.image_augmentation_prob == 0.6

    def test_vision_lora_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("vision_lora: true\nvision_lora_r: 64\nvision_lora_alpha: 32\n")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.vision_lora is True
        assert cfg.vision_lora_r == 64
        assert cfg.vision_lora_alpha == 32

    def test_image_augmentation_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("image_augmentation: true\nimage_augmentation_prob: 0.25\n")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.image_augmentation is True
        assert cfg.image_augmentation_prob == 0.25

    def test_metrics_eval_benchmarks_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("metrics_eval_benchmarks: [ip5m, m2s]\n")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.metrics_eval_benchmarks == ["ip5m", "m2s"]

    def test_metrics_eval_data_dir_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("metrics_eval_data_dir: /data/prepared/eval\n")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.metrics_eval_data_dir == "/data/prepared/eval"

    def test_ocsr_metrics_eval_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "ocsr_metrics_eval_size: 100\n"
            "ocsr_metrics_eval_data: /data/prepared/eval/eval_ocsr_uspto.jsonl\n"
            "ocsr_metrics_eval_name: uspto\n"
            "ocsr_metrics_eval_batch_size: 4\n"
            "ocsr_metrics_prediction_mode: markush_no_extension\n"
        )
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.ocsr_metrics_eval_size == 100
        assert cfg.ocsr_metrics_eval_data == "/data/prepared/eval/eval_ocsr_uspto.jsonl"
        assert cfg.ocsr_metrics_eval_name == "uspto"
        assert cfg.ocsr_metrics_eval_batch_size == 4
        assert cfg.ocsr_metrics_prediction_mode == "markush_no_extension"

    def test_init_adapter_path_from_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("init_adapter_path: /checkpoints/parent/checkpoint-100\n")
        cfg = SFTConfig.from_yaml(cfg_path)
        assert cfg.init_adapter_path == "/checkpoints/parent/checkpoint-100"

    def test_save_total_limit_cli_override(self) -> None:
        args = build_argument_parser().parse_args(["--save-total-limit", "20"])
        cfg = SFTConfig.from_args(args)

        assert cfg.save_total_limit == 20

    def test_resolve_metrics_benchmark_path_uses_prepared_eval_dir(self, tmp_path: Path) -> None:
        # The prepared tree is nested per-benchmark: <dir>/<name>/eval_<name>.jsonl.
        path = _resolve_metrics_benchmark_path(
            "m2s",
            root=tmp_path / "repo",
            metrics_eval_data_dir=str(tmp_path / "prepared"),
        )

        assert path == tmp_path / "prepared" / "m2s" / "eval_m2s.jsonl"

    def test_resolve_metrics_benchmark_path_uses_canonical_layout_default(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # No flat ``data/eval_ip5m.jsonl`` ships; the default resolves to the
        # canonical glyph.data.layout location under ``$GLYPH_DATA_ROOT``.
        monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path / "data_root"))
        path = _resolve_metrics_benchmark_path("ip5m", root=tmp_path / "repo")

        assert (
            path == tmp_path / "data_root" / "benchmarks" / "markush" / "ip5m" / "eval_ip5m.jsonl"
        )

    def test_resolve_metrics_benchmark_path_matches_importer_output(self, tmp_path: Path) -> None:
        """Regression guard: the inline-eval resolver must return the exact path the
        ``glyph data prepare --markush ... benchmarks`` importer writes on disk.

        The importer stores each benchmark under
        ``markush_benchmark_dir(<name>)/eval_<name>.jsonl`` where ``<name>`` is a
        member of ``MARKUSH_BENCHMARK_NAMES`` (underscore form). Training configs
        pass ``metrics_eval_data_dir=$GLYPH_DATA_ROOT/benchmarks/markush`` and list
        the benchmarks by their canonical spec name (``uspto-markush`` with a
        hyphen). Both must agree for all three benchmarks.
        """
        from glyph.data import layout
        from glyph.data.layout import MARKUSH_BENCHMARK_NAMES

        data_root = tmp_path / "data_root"
        base = layout.markush_benchmarks_dir(root=data_root)

        # Canonical spec names the training config lists, keyed to the on-disk
        # (underscore) name the importer materializes.
        config_to_importer = {
            "ip5m": "ip5m",
            "m2s": "m2s",
            "uspto-markush": "uspto_markush",
        }
        assert set(config_to_importer.values()) == set(MARKUSH_BENCHMARK_NAMES)

        for config_name, importer_name in config_to_importer.items():
            resolved = _resolve_metrics_benchmark_path(
                config_name,
                root=data_root,
                metrics_eval_data_dir=str(base),
            )
            importer_output = (
                layout.markush_benchmark_dir(importer_name, root=data_root)
                / f"eval_{importer_name}.jsonl"
            )
            assert resolved == importer_output

    def test_resolve_ocsr_metrics_eval_specs_discovers_directory(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "eval_ocsr_uspto.jsonl").write_text("", encoding="utf-8")
        (eval_dir / "eval_ocsr_clef.jsonl").write_text("", encoding="utf-8")
        (eval_dir / "eval_ip5m.jsonl").write_text("", encoding="utf-8")
        cfg = SFTConfig(
            ocsr_metrics_eval_size=10,
            ocsr_metrics_eval_data=str(eval_dir),
            ocsr_metrics_eval_name="ignored",
        )

        specs = _resolve_ocsr_metrics_eval_specs(cfg, root=tmp_path, require_exists=True)

        assert [(spec.name, spec.path.name, spec.prefix) for spec in specs] == [
            ("clef", "eval_ocsr_clef.jsonl", "eval_ocsr_clef"),
            ("uspto", "eval_ocsr_uspto.jsonl", "eval_ocsr_uspto"),
        ]

    def test_resolve_ocsr_metrics_eval_specs_keeps_single_file_name(self, tmp_path: Path) -> None:
        eval_jsonl = tmp_path / "eval_ocsr_uspto.jsonl"
        eval_jsonl.write_text("", encoding="utf-8")
        cfg = SFTConfig(
            ocsr_metrics_eval_size=10,
            ocsr_metrics_eval_data=str(eval_jsonl),
            ocsr_metrics_eval_name="uspto-public",
        )

        specs = _resolve_ocsr_metrics_eval_specs(cfg, root=tmp_path, require_exists=True)

        assert [(spec.name, spec.path, spec.prefix) for spec in specs] == [
            ("uspto-public", eval_jsonl, "eval_ocsr_uspto_public")
        ]

    def test_resolve_ocsr_metrics_eval_specs_rejects_empty_directory(self, tmp_path: Path) -> None:
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        cfg = SFTConfig(ocsr_metrics_eval_size=10, ocsr_metrics_eval_data=str(eval_dir))

        with pytest.raises(ValueError, match="no eval_ocsr_\\*\\.jsonl files"):
            _resolve_ocsr_metrics_eval_specs(cfg, root=tmp_path, require_exists=True)

    def test_resolve_model_identifier_keeps_hf_model_id(self, tmp_path: Path) -> None:
        model = _resolve_model_identifier("org/model-name", tmp_path / "repo")

        assert model == "org/model-name"

    def test_resolve_model_identifier_keeps_repo_relative_model_path(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        model = _resolve_model_identifier("models/Qwen3.5-2B-Base", root)

        assert model == root / "models/Qwen3.5-2B-Base"

    def test_resolve_model_identifier_prefers_existing_repo_path(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        model_dir = root / "custom/model"
        model_dir.mkdir(parents=True)

        model = _resolve_model_identifier("custom/model", root)

        assert model == model_dir

    def test_expected_best_metric_keys_include_benchmark_prefixes(self) -> None:
        cfg = SFTConfig(
            metrics_eval_size=10,
            metrics_eval_benchmarks=["ip5m", "m2s"],
            metrics_eval_protocol="both",
        )

        keys = expected_best_metric_keys(cfg)

        assert "eval_ip5m_ours_cxsmi_equality" in keys
        assert "eval_m2s_mg2_cxsmi_equality" in keys
        assert "eval_ip5m_ours_valid_rate" in keys
        assert "eval_ip5m_ours_tanimoto_similarity" in keys
        assert "eval_ip5m_ours_m_section_match" in keys
        assert "eval_ip5m_avg" in keys
        assert "eval_m2s_avg" in keys
        assert "markush_ours" in keys
        assert "markush_mg2" in keys
        assert "markush_overall" in keys
        assert "overall_combined" in keys
        assert "eval_ip5m_ours_m" not in keys

    def test_expected_best_metric_keys_include_ocsr_prefix(self) -> None:
        cfg = SFTConfig(
            ocsr_metrics_eval_size=10,
            ocsr_metrics_eval_data="/data/eval_ocsr_uspto.jsonl",
            ocsr_metrics_eval_name="uspto",
        )

        keys = expected_best_metric_keys(cfg)

        assert "eval_ocsr_uspto_canon_exact_match" in keys
        assert "eval_ocsr_uspto_graph_exact_match" in keys
        assert "eval_ocsr_uspto_valid_smiles_rate" in keys
        assert "ocsr_overall" in keys
        assert "overall_combined" in keys
        assert "eval_ocsr_uspto_ours_cxsmi_equality" not in keys

    def test_expected_best_metric_keys_include_directory_ocsr_prefixes(
        self, tmp_path: Path
    ) -> None:
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "eval_ocsr_uspto.jsonl").write_text("", encoding="utf-8")
        (eval_dir / "eval_ocsr_clef.jsonl").write_text("", encoding="utf-8")
        cfg = SFTConfig(
            ocsr_metrics_eval_size=10,
            ocsr_metrics_eval_data=str(eval_dir),
            ocsr_metrics_eval_name="ignored",
        )

        keys = expected_best_metric_keys(cfg, root=tmp_path)

        assert "eval_ocsr_clef_canon_exact_match" in keys
        assert "eval_ocsr_uspto_graph_exact_match" in keys
        assert "eval_ocsr_ignored_canon_exact_match" not in keys

    def test_validate_best_metric_config_fails_early_for_missing_callback_key(self) -> None:
        cfg = SFTConfig(
            load_best_model_at_end=True,
            metric_for_best_model="eval_ip5m_ours_cxsmi_equality",
            metrics_eval_size=10,
            metrics_eval_benchmarks=["m2s"],
            metrics_eval_protocol="both",
            save_steps=200,
            eval_steps=200,
        )

        with pytest.raises(ValueError, match="metric_for_best_model"):
            validate_best_metric_config(cfg)

    def test_validate_best_metric_config_requires_matching_save_eval_cadence(self) -> None:
        cfg = SFTConfig(
            load_best_model_at_end=True,
            metric_for_best_model="eval_ip5m_ours_cxsmi_equality",
            metrics_eval_size=10,
            metrics_eval_benchmarks=["ip5m"],
            metrics_eval_protocol="both",
            save_steps=1000,
            eval_steps=200,
        )

        with pytest.raises(ValueError, match="save_steps == eval_steps"):
            validate_best_metric_config(cfg)

    def test_validate_best_metric_config_accepts_matching_callback_key(self) -> None:
        cfg = SFTConfig(
            load_best_model_at_end=True,
            metric_for_best_model="eval_ip5m_ours_cxsmi_equality",
            metrics_eval_size=10,
            metrics_eval_benchmarks=["ip5m"],
            metrics_eval_protocol="both",
            save_steps=200,
            eval_steps=200,
        )

        validate_best_metric_config(cfg)

    def test_validate_best_metric_config_accepts_aggregate_callback_key(self) -> None:
        cfg = SFTConfig(
            load_best_model_at_end=True,
            metric_for_best_model="overall_combined",
            metrics_eval_size=10,
            metrics_eval_benchmarks=["ip5m"],
            metrics_eval_protocol="both",
            ocsr_metrics_eval_size=10,
            ocsr_metrics_eval_data="/data/eval_ocsr_uspto.jsonl",
            ocsr_metrics_eval_name="uspto",
            save_steps=200,
            eval_steps=200,
        )

        validate_best_metric_config(cfg)

    def test_validate_best_metric_config_rejects_skipping_metrics_eval_steps(self) -> None:
        cfg = SFTConfig(
            load_best_model_at_end=True,
            metric_for_best_model="eval_ip5m_ours_cxsmi_equality",
            metrics_eval_size=10,
            metrics_eval_benchmarks=["ip5m"],
            metrics_eval_protocol="both",
            save_steps=200,
            eval_steps=200,
            metrics_eval_steps=600,
        )

        with pytest.raises(ValueError, match="metrics_eval_steps"):
            validate_best_metric_config(cfg)

        cfg.metrics_eval_steps = 200
        validate_best_metric_config(cfg)

    def test_validate_best_metric_config_rejects_skipping_aggregate_metrics_eval_steps(
        self,
    ) -> None:
        cfg = SFTConfig(
            load_best_model_at_end=True,
            metric_for_best_model="overall_combined",
            metrics_eval_size=10,
            metrics_eval_benchmarks=["ip5m"],
            metrics_eval_protocol="both",
            ocsr_metrics_eval_size=10,
            ocsr_metrics_eval_data="/data/eval_ocsr_uspto.jsonl",
            ocsr_metrics_eval_name="uspto",
            save_steps=200,
            eval_steps=200,
            metrics_eval_steps=600,
        )

        with pytest.raises(ValueError, match="metrics_eval_steps"):
            validate_best_metric_config(cfg)

        cfg.metrics_eval_steps = 200
        cfg.ocsr_metrics_eval_steps = 600
        with pytest.raises(ValueError, match="ocsr_metrics_eval_steps"):
            validate_best_metric_config(cfg)

        cfg.ocsr_metrics_eval_steps = 200
        validate_best_metric_config(cfg)

    def test_validate_best_metric_config_rejects_skipping_ocsr_metrics_eval_steps(
        self,
    ) -> None:
        cfg = SFTConfig(
            load_best_model_at_end=True,
            metric_for_best_model="eval_ocsr_uspto_canon_exact_match",
            ocsr_metrics_eval_size=10,
            ocsr_metrics_eval_data="/data/eval_ocsr_uspto.jsonl",
            ocsr_metrics_eval_name="uspto",
            save_steps=200,
            eval_steps=200,
            ocsr_metrics_eval_steps=600,
        )

        with pytest.raises(ValueError, match="ocsr_metrics_eval_steps"):
            validate_best_metric_config(cfg)

        cfg.ocsr_metrics_eval_steps = 200
        validate_best_metric_config(cfg)

    def test_install_processor_save_callback_runs_before_existing_callbacks(self) -> None:
        reporter = object()
        trainer = SimpleNamespace(callback_handler=SimpleNamespace(callbacks=[reporter]))

        install_processor_save_callback_first(trainer, _MockProcessor())

        assert isinstance(trainer.callback_handler.callbacks[0], SaveProcessorCallback)
        assert trainer.callback_handler.callbacks[1] is reporter

    def test_setup_model_loads_init_adapter_as_trainable(self, monkeypatch) -> None:
        from glyph.markush.train import sft as sft_module

        calls: dict[str, Any] = {}

        class _FakeConfig:
            eos_token_id = 7
            use_cache = True

        class _FakeBaseModel:
            def __init__(self) -> None:
                self.config = _FakeConfig()
                self.gradient_checkpointing_enabled = False
                self.input_grads_enabled = False

            def gradient_checkpointing_enable(self, **kwargs) -> None:
                self.gradient_checkpointing_enabled = True
                calls["gradient_checkpointing_kwargs"] = kwargs

            def enable_input_require_grads(self) -> None:
                self.input_grads_enabled = True

        class _FakeAdapterModel:
            def __init__(self, base: _FakeBaseModel) -> None:
                self.base = base
                self.config = base.config
                self.printed = False

            def print_trainable_parameters(self) -> None:
                self.printed = True
                calls["printed"] = True

        base = _FakeBaseModel()
        adapter = _FakeAdapterModel(base)

        monkeypatch.setattr(
            sft_module.AutoModelForImageTextToText,
            "from_pretrained",
            lambda *args, **kwargs: base,
        )
        monkeypatch.setattr(
            sft_module.PeftModel,
            "from_pretrained",
            lambda model, path, is_trainable: (
                calls.setdefault("adapter_args", (model, path, is_trainable)) and adapter
            ),
        )

        def _fresh_lora_should_not_run(*args, **kwargs):
            raise AssertionError("fresh LoRA initialization should be skipped")

        monkeypatch.setattr(sft_module, "get_peft_model", _fresh_lora_should_not_run)

        model = setup_model(
            "base-model",
            lora_r=128,
            lora_alpha=128,
            lora_dropout=0.05,
            vision_lora=True,
            init_adapter_path="/checkpoints/parent/checkpoint-100",
        )

        assert model is adapter
        assert calls["adapter_args"] == (
            base,
            "/checkpoints/parent/checkpoint-100",
            True,
        )
        assert base.config.use_cache is False
        assert base.gradient_checkpointing_enabled is True
        assert base.input_grads_enabled is True
        assert adapter.config.eos_token_id == [7, 248044, 248046]
        assert calls["printed"] is True

    def test_setup_model_full_finetune_skips_lora(self, monkeypatch) -> None:
        from glyph.markush.train import sft as sft_module

        calls: dict[str, Any] = {}

        class _FakeConfig:
            eos_token_id = 7
            use_cache = True

        class _FakeParam:
            def __init__(self) -> None:
                self.requires_grad = False

            def numel(self) -> int:
                return 10

            def requires_grad_(self, value: bool) -> None:
                self.requires_grad = value

        class _FakeBaseModel:
            def __init__(self) -> None:
                self.config = _FakeConfig()
                self.gradient_checkpointing_enabled = False
                self.input_grads_enabled = False
                self.params = [_FakeParam(), _FakeParam()]

            def gradient_checkpointing_enable(self, **kwargs) -> None:
                self.gradient_checkpointing_enabled = True
                calls["gradient_checkpointing_kwargs"] = kwargs

            def enable_input_require_grads(self) -> None:
                self.input_grads_enabled = True

            def parameters(self):
                return iter(self.params)

        base = _FakeBaseModel()

        monkeypatch.setattr(
            sft_module.AutoModelForImageTextToText,
            "from_pretrained",
            lambda *args, **kwargs: base,
        )

        def _fresh_lora_should_not_run(*args, **kwargs):
            raise AssertionError("fresh LoRA initialization should be skipped")

        def _adapter_should_not_run(*args, **kwargs):
            raise AssertionError("adapter loading should be skipped")

        monkeypatch.setattr(sft_module, "get_peft_model", _fresh_lora_should_not_run)
        monkeypatch.setattr(sft_module.PeftModel, "from_pretrained", _adapter_should_not_run)

        model = setup_model(
            "base-model",
            lora_r=128,
            lora_alpha=128,
            lora_dropout=0.05,
            vision_lora=True,
            full_finetune=True,
        )

        assert model is base
        assert base.config.use_cache is False
        assert base.gradient_checkpointing_enabled is True
        assert base.input_grads_enabled is True
        assert [param.requires_grad for param in base.params] == [True, True]
        assert base.config.eos_token_id == [7, 248044, 248046]

    def test_setup_model_full_finetune_merges_init_adapter(self, monkeypatch) -> None:
        from glyph.markush.train import sft as sft_module

        calls: dict[str, Any] = {}

        class _FakeConfig:
            eos_token_id = 7
            use_cache = True

        class _FakeParam:
            def __init__(self) -> None:
                self.requires_grad = False

            def numel(self) -> int:
                return 10

            def requires_grad_(self, value: bool) -> None:
                self.requires_grad = value

        class _FakeBaseModel:
            def __init__(self) -> None:
                self.config = _FakeConfig()
                self.params = [_FakeParam()]

            def gradient_checkpointing_enable(self, **kwargs) -> None:
                calls["gradient_checkpointing_kwargs"] = kwargs

            def enable_input_require_grads(self) -> None:
                calls["input_grads_enabled"] = True

            def parameters(self):
                return iter(self.params)

        class _FakeAdapterModel:
            def __init__(self, base: _FakeBaseModel) -> None:
                self.base = base
                self.config = base.config

            def merge_and_unload(self):
                calls["merged"] = True
                return self.base

        base = _FakeBaseModel()
        adapter = _FakeAdapterModel(base)

        monkeypatch.setattr(
            sft_module.AutoModelForImageTextToText,
            "from_pretrained",
            lambda *args, **kwargs: base,
        )
        monkeypatch.setattr(
            sft_module.PeftModel,
            "from_pretrained",
            lambda model, path, is_trainable: (
                calls.setdefault("adapter_args", (model, path, is_trainable)) and adapter
            ),
        )
        monkeypatch.setattr(
            sft_module,
            "get_peft_model",
            lambda *args, **kwargs: pytest.fail("fresh LoRA should not run"),
        )

        model = setup_model(
            "base-model",
            lora_r=128,
            lora_alpha=128,
            lora_dropout=0.05,
            init_adapter_path="/adapter",
            full_finetune=True,
        )

        assert model is base
        assert calls["adapter_args"] == (base, "/adapter", False)
        assert calls["merged"] is True
        assert base.params[0].requires_grad is True
        assert base.config.eos_token_id == [7, 248044, 248046]

    def test_metrics_eval_callback_uses_position_correct_cxsmi_equality(self) -> None:
        gt = "<r>Ra</r>S(=O)(=O)[N-]S(<r>CF3</r>)(=O)=O"
        pred = "<r>CF3</r>S(=O)(=O)[N-]S(<r>Ra</r>)(=O)=O|m:1:2.3.4"
        callback = MetricsEvalCallback(
            samples=[
                {
                    "id": "ip5m_13",
                    "assistant_text": f"<markush><cxsmi>{gt}</cxsmi></markush>",
                    "user_text": "",
                    "image": "",
                }
            ],
            processor=None,  # type: ignore[arg-type]
            protocol="ours",
            prefix="eval_ip5m",
        )
        callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
            [f"<markush><cxsmi>{pred}</cxsmi></markush>" for _ in batch],
            0,
        )

        class DummyModel:
            training = False

            def eval(self) -> None:
                self.training = False

            def train(self) -> None:
                self.training = True

        metrics: dict[str, Any] = {}
        callback.on_evaluate(
            SimpleNamespace(),
            SimpleNamespace(is_world_process_zero=True, global_step=1),
            SimpleNamespace(),
            model=DummyModel(),
            metrics=metrics,
        )

        assert metrics["eval_ip5m_ours_cxsmi_equality"] == 0.0
        assert metrics["eval_ip5m_trainer_step"] == 1

    def test_metrics_eval_prompt_preserves_prepared_target_format_instruction(self) -> None:
        callback = MetricsEvalCallback(samples=[], processor=None)  # type: ignore[arg-type]
        user_text = (
            "Extract the Markush structure from this image.\n"
            "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
            "the <cxsmi> value must use canonicalized cxsmiles_opt."
        )

        assert callback._build_generation_prompt({"user_text": user_text}) == user_text

    def test_metrics_eval_prompt_keeps_legacy_delimiter_for_unstructured_prompt(self) -> None:
        callback = MetricsEvalCallback(samples=[], processor=None)  # type: ignore[arg-type]

        assert callback._build_generation_prompt({"user_text": "Patent context."}) == (
            f"Patent context.\n{PROMPT_DELIMITER}"
        )

    def test_multi_metrics_eval_callback_dispatches_distinct_metric_jobs(self) -> None:
        callbacks = []
        for prefix in ("eval_ip5m", "eval_m2s"):
            callback = MetricsEvalCallback(
                samples=[
                    {
                        "id": prefix,
                        "assistant_text": "<markush><cxsmi>CCO</cxsmi></markush>",
                        "user_text": "",
                        "image": "",
                    }
                ],
                processor=None,  # type: ignore[arg-type]
                protocol="ours",
                prefix=prefix,
            )
            callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
                ["<markush><cxsmi>CCO</cxsmi></markush>" for _ in batch],
                0,
            )
            callbacks.append(callback)

        class DummyModel:
            training = False

            def eval(self) -> None:
                self.training = False

            def train(self) -> None:
                self.training = True

        metrics: dict[str, Any] = {}
        MultiMetricsEvalCallback(callbacks).on_evaluate(
            SimpleNamespace(),
            SimpleNamespace(is_world_process_zero=True, global_step=3),
            SimpleNamespace(),
            model=DummyModel(),
            metrics=metrics,
        )

        assert metrics["eval_ip5m_ours_cxsmi_equality"] == 1.0
        assert metrics["eval_m2s_ours_cxsmi_equality"] == 1.0
        assert metrics["eval_ip5m_trainer_step"] == 3
        assert metrics["eval_m2s_trainer_step"] == 3
        assert metrics["eval_ip5m_avg"] == 1.0
        assert metrics["eval_m2s_avg"] == 1.0
        assert metrics["markush_ours"] == 1.0
        assert metrics["markush_overall"] == 1.0
        assert metrics["overall_combined"] == 1.0

    def test_inline_eval_aggregate_metrics_from_known_inputs(self) -> None:
        aggregate = compute_inline_eval_aggregate_metrics(
            {
                "eval_ip5m_ours_cxsmi_equality": 0.8,
                "eval_ip5m_mg2_cxsmi_equality": 0.6,
                "eval_m2s_ours_cxsmi_equality": 0.4,
                "eval_m2s_mg2_cxsmi_equality": 0.2,
                "eval_uspto_markush/ours/cxsmi_equality": 0.9,
                "eval_ocsr_clef_canon_exact_match": 0.5,
                "eval_ocsr_uspto/canon_exact_match": 1.0,
            }
        )

        assert aggregate["eval_ip5m/avg"] == pytest.approx(0.7)
        assert aggregate["eval_m2s/avg"] == pytest.approx(0.3)
        assert aggregate["eval_uspto_markush/avg"] == pytest.approx(0.9)
        assert aggregate["markush/ours"] == pytest.approx(0.7)
        assert aggregate["markush/mg2"] == pytest.approx(0.4)
        assert aggregate["markush/overall"] == pytest.approx(0.55)
        assert aggregate["ocsr/overall"] == pytest.approx(0.75)
        assert aggregate["overall/combined"] == pytest.approx(0.65)

    def test_inline_eval_aggregate_metrics_only_one_family(self) -> None:
        aggregate = compute_inline_eval_aggregate_metrics(
            {
                "eval_ocsr_clef_canon_exact_match": 0.25,
                "eval_ocsr_jpo_canon_exact_match": 0.75,
            }
        )

        assert aggregate == {
            "ocsr/overall": 0.5,
            "overall/combined": 0.5,
        }

    def test_metrics_eval_callback_collective_sequence_matches_success(self, monkeypatch) -> None:
        from glyph.markush.train import sft as sft_module

        class _Model:
            training = False

            def eval(self):
                pass

        samples = [
            {"id": "s0", "image": "/tmp/0.png", "user_text": "", "assistant_text": "target-0"},
            {"id": "s1", "image": "/tmp/1.png", "user_text": "", "assistant_text": "target-1"},
        ]

        def run_rank(rank: int) -> list[str]:
            sequence: list[str] = []
            monkeypatch.setattr(sft_module, "_distributed_rank_world", lambda: (rank, 2))

            def fake_gather(payload):
                sequence.append("gather")
                return [payload]

            def fake_broadcast(payload, *, src=0):
                assert src == 0
                sequence.append("broadcast_recv" if payload is None else "broadcast_send")
                return {} if payload is None else payload

            monkeypatch.setattr(sft_module, "_distributed_all_gather_object", fake_gather)
            monkeypatch.setattr(sft_module, "_distributed_broadcast_object", fake_broadcast)
            callback = MetricsEvalCallback(
                samples,
                _MockProcessor(),
                protocol="ours",
                prefix="eval_ip5m",
            )
            callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
                [f"pred-{sample['id']}" for sample in batch],
                0,
            )
            callback._score_predictions = lambda _preds, _gts: {  # type: ignore[method-assign]
                "eval_ip5m/ours/cxsmi_equality": 1.0
            }

            callback.on_evaluate(
                SimpleNamespace(),
                SimpleNamespace(is_world_process_zero=(rank == 0), global_step=11),
                SimpleNamespace(),
                model=_Model(),
                metrics={},
            )
            return sequence

        assert run_rank(0) == ["gather", "broadcast_send"]
        assert run_rank(1) == ["gather", "broadcast_recv"]

    def test_metrics_eval_callback_rank_local_failure_still_reaches_collectives(
        self, monkeypatch
    ) -> None:
        from glyph.markush.train import sft as sft_module

        class _Model:
            training = False

            def eval(self):
                pass

        samples = [
            {"id": "s0", "image": "/tmp/0.png", "user_text": "", "assistant_text": "target-0"},
            {"id": "s1", "image": "/tmp/1.png", "user_text": "", "assistant_text": "target-1"},
        ]
        sequence: list[str] = []
        gathered: list[dict[str, Any]] = []
        monkeypatch.setattr(sft_module, "_distributed_rank_world", lambda: (1, 2))

        def fake_gather(payload):
            sequence.append("gather")
            gathered.append(payload)
            return [payload]

        def fake_broadcast(payload, *, src=0):
            assert src == 0
            sequence.append("broadcast_recv" if payload is None else "broadcast_send")
            return {}

        monkeypatch.setattr(sft_module, "_distributed_all_gather_object", fake_gather)
        monkeypatch.setattr(sft_module, "_distributed_broadcast_object", fake_broadcast)
        callback = MetricsEvalCallback(
            samples, _MockProcessor(), protocol="ours", prefix="eval_ip5m"
        )
        callback._local_ground_truth_rows = (  # type: ignore[method-assign]
            lambda _samples: (_ for _ in ()).throw(RuntimeError("bad setup"))
        )

        callback.on_evaluate(
            SimpleNamespace(),
            SimpleNamespace(is_world_process_zero=False, global_step=12),
            SimpleNamespace(),
            model=_Model(),
            metrics={},
        )

        assert sequence == ["gather", "broadcast_recv"]
        assert gathered[0]["rank"] == 1
        assert gathered[0]["preds"] == []
        assert gathered[0]["gts"] == []
        assert gathered[0]["error"] == "bad setup"

    def test_multi_metrics_eval_callback_collective_sequence_has_aggregate_broadcast(
        self, monkeypatch
    ) -> None:
        from glyph.markush.train import sft as sft_module

        class _Model:
            training = False

            def eval(self):
                pass

        def run_rank(rank: int) -> tuple[list[str], dict[str, Any]]:
            sequence: list[str] = []
            monkeypatch.setattr(sft_module, "_distributed_rank_world", lambda: (rank, 2))

            def fake_gather(payload):
                sequence.append("gather")
                return [payload]

            def fake_broadcast(payload, *, src=0):
                assert src == 0
                sequence.append("broadcast_recv" if payload is None else "broadcast_send")
                return {} if payload is None else payload

            monkeypatch.setattr(sft_module, "_distributed_all_gather_object", fake_gather)
            monkeypatch.setattr(sft_module, "_distributed_broadcast_object", fake_broadcast)
            callbacks = []
            for prefix, score in (("eval_ip5m", 0.5), ("eval_m2s", 0.25)):
                callback = MetricsEvalCallback(
                    samples=[
                        {
                            "id": f"{prefix}_{rank}",
                            "image": "/tmp/0.png",
                            "user_text": "",
                            "assistant_text": "target",
                        }
                    ],
                    processor=_MockProcessor(),
                    protocol="ours",
                    prefix=prefix,
                )
                callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
                    ["pred" for _ in batch],
                    0,
                )
                callback._score_predictions = (  # type: ignore[method-assign]
                    lambda _preds, _gts, prefix=prefix, score=score: {
                        f"{prefix}/ours/cxsmi_equality": score
                    }
                )
                callbacks.append(callback)
            metrics: dict[str, Any] = {}

            MultiMetricsEvalCallback(callbacks).on_evaluate(
                SimpleNamespace(),
                SimpleNamespace(is_world_process_zero=(rank == 0), global_step=13),
                SimpleNamespace(),
                model=_Model(),
                metrics=metrics,
            )
            return sequence, metrics

        rank0_sequence, rank0_metrics = run_rank(0)
        rank1_sequence, _ = run_rank(1)

        assert rank0_sequence == [
            "gather",
            "broadcast_send",
            "gather",
            "broadcast_send",
            "broadcast_send",
        ]
        assert rank1_sequence == [
            "gather",
            "broadcast_recv",
            "gather",
            "broadcast_recv",
            "broadcast_recv",
        ]
        assert rank0_metrics["overall_combined"] == pytest.approx(0.375)

    def test_metrics_eval_callback_nonzero_rank_generates_its_shard(self, monkeypatch) -> None:
        from glyph.markush.train import sft as sft_module

        class _MockModel:
            training = True
            train_called = False

            def eval(self):
                self.training = False

            def train(self):
                self.train_called = True
                self.training = True

        samples = [
            {"id": "s0", "image": "/tmp/0.png", "user_text": "", "assistant_text": "target-0"},
            {"id": "s1", "image": "/tmp/1.png", "user_text": "", "assistant_text": "target-1"},
            {"id": "s2", "image": "/tmp/2.png", "user_text": "", "assistant_text": "target-2"},
        ]
        generated: list[str] = []
        gathered: list[dict[str, Any]] = []
        monkeypatch.setattr(sft_module, "_distributed_rank_world", lambda: (1, 2))

        def fake_gather(payload):
            gathered.append(payload)
            return [{"rank": 0, "preds": [], "gts": [], "n_failed": 0}, payload]

        def fake_broadcast(payload, *, src=0):
            assert src == 0
            assert payload is None
            return {"eval_ours_cxsmi_equality": 0.25, "eval_trainer_step": 5}

        monkeypatch.setattr(sft_module, "_distributed_all_gather_object", fake_gather)
        monkeypatch.setattr(sft_module, "_distributed_broadcast_object", fake_broadcast)

        class _FakeWandb:
            run = object()
            logs: ClassVar[list[tuple[dict[str, Any], int | None]]] = []

            @staticmethod
            def log(payload, step=None):
                _FakeWandb.logs.append((payload, step))

        monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)

        callback = MetricsEvalCallback(samples, _MockProcessor())

        def fake_generate(_model, batch):
            generated.extend(sample["id"] for sample in batch)
            return [f"pred-{sample['id']}" for sample in batch], 0

        callback._generate_many = fake_generate  # type: ignore[method-assign]
        model = _MockModel()
        state = SimpleNamespace(is_world_process_zero=False, global_step=5)
        metrics: dict[str, Any] = {}

        callback.on_evaluate(
            SimpleNamespace(), state, SimpleNamespace(), model=model, metrics=metrics
        )

        assert generated == ["s1"]
        assert gathered[0]["rank"] == 1
        assert gathered[0]["preds"] == [{"id": "s1", "prediction": "pred-s1"}]
        assert gathered[0]["gts"] == [{"id": "s1", "annotation": "target-1"}]
        assert model.train_called is True
        # Non-zero ranks gather and receive flat metrics for HF best-model
        # selection, but never score or log themselves.
        assert metrics["eval_ours_cxsmi_equality"] == 0.25
        assert metrics["eval_trainer_step"] == 5
        assert _FakeWandb.logs == []

    def test_metrics_eval_callback_rank_zero_scores_gathered_rank_payloads(
        self, monkeypatch
    ) -> None:
        from glyph.markush.eval import metrics_mcs
        from glyph.markush.train import sft as sft_module

        class _MockModel:
            training = True
            train_called = False

            def eval(self):
                self.training = False

            def train(self):
                self.train_called = True
                self.training = True

        samples = [
            {"id": "s0", "image": "/tmp/0.png", "user_text": "", "assistant_text": "target-0"},
            {"id": "s1", "image": "/tmp/1.png", "user_text": "", "assistant_text": "target-1"},
            {"id": "s2", "image": "/tmp/2.png", "user_text": "", "assistant_text": "target-2"},
        ]
        generated: list[str] = []
        captured: dict[str, Any] = {}
        monkeypatch.setattr(sft_module, "_distributed_rank_world", lambda: (0, 2))

        def fake_gather(payload):
            return [
                payload,
                {
                    "rank": 1,
                    "preds": [{"id": "s1", "prediction": "pred-s1"}],
                    "gts": [{"id": "s1", "annotation": "target-1"}],
                    "n_failed": 0,
                    "elapsed_s": 1.25,
                    "error": None,
                },
            ]

        def fake_evaluate(preds, gts, **kwargs):
            captured["preds"] = preds
            captured["gts"] = gts
            captured["kwargs"] = kwargs
            return {"cxsmi_equality": 0.5}

        class _FakeWandb:
            run = object()
            logs: ClassVar[list[tuple[dict[str, Any], int | None]]] = []

            @staticmethod
            def log(payload, step=None):
                _FakeWandb.logs.append((payload, step))

        monkeypatch.setattr(sft_module, "_distributed_all_gather_object", fake_gather)
        monkeypatch.setattr(metrics_mcs, "evaluate_markush_mcs", fake_evaluate)
        monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)

        callback = MetricsEvalCallback(
            samples,
            _MockProcessor(),
            protocol="ours",
            prefix="eval_ip5m",
        )

        def fake_generate(_model, batch):
            generated.extend(sample["id"] for sample in batch)
            return [f"pred-{sample['id']}" for sample in batch], 0

        callback._generate_many = fake_generate  # type: ignore[method-assign]
        model = _MockModel()
        state = SimpleNamespace(is_world_process_zero=True, global_step=5)
        metrics: dict[str, Any] = {}

        callback.on_evaluate(
            SimpleNamespace(), state, SimpleNamespace(), model=model, metrics=metrics
        )

        assert generated == ["s0", "s2"]
        assert {row["id"] for row in captured["preds"]} == {"s0", "s1", "s2"}
        assert {row["id"] for row in captured["gts"]} == {"s0", "s1", "s2"}
        assert captured["kwargs"]["include_missing"] is True
        assert captured["kwargs"]["protocol"] == "ours"
        payload, step = _FakeWandb.logs[-1]
        assert step == 5
        assert payload["eval_ip5m/ours/cxsmi_equality"] == 0.5
        assert payload["eval_ip5m/gen/n_samples"] == 3
        assert payload["eval_ip5m/gen/world_size"] == 2
        assert payload["eval_ip5m/gen/batch_size"] == 1
        assert payload["eval_ip5m/gen/rank_errors"] == 0
        # Flat keys still flow into the HF metrics dict for best-model logic.
        assert metrics["eval_ip5m_ours_cxsmi_equality"] == 0.5
        assert model.train_called is True

    def test_metrics_eval_callback_rank_zero_scoring_failure_broadcasts_empty(
        self, monkeypatch
    ) -> None:
        from glyph.markush.train import sft as sft_module

        class _MockModel:
            training = False

            def eval(self):
                pass

        samples = [
            {"id": "s0", "image": "/tmp/0.png", "user_text": "", "assistant_text": "target-0"}
        ]
        monkeypatch.setattr(sft_module, "_distributed_rank_world", lambda: (0, 2))
        monkeypatch.setattr(
            sft_module,
            "_distributed_all_gather_object",
            lambda payload: [payload],
        )
        broadcasts: list[dict[str, Any]] = []

        def fake_broadcast(payload, *, src=0):
            assert src == 0
            broadcasts.append(payload)
            return payload

        monkeypatch.setattr(sft_module, "_distributed_broadcast_object", fake_broadcast)

        callback = MetricsEvalCallback(samples, _MockProcessor(), prefix="eval_ip5m")
        callback._generate_many = lambda _model, batch: (["pred" for _ in batch], 0)  # type: ignore[method-assign]

        def fail_score(_preds, _gts):
            raise RuntimeError("pathological row")

        callback._score_predictions = fail_score  # type: ignore[method-assign]
        metrics: dict[str, Any] = {}

        callback.on_evaluate(
            SimpleNamespace(),
            SimpleNamespace(is_world_process_zero=True, global_step=6),
            SimpleNamespace(),
            model=_MockModel(),
            metrics=metrics,
        )

        assert broadcasts == [{}]
        assert metrics == {}

    def test_metrics_eval_callback_uses_configured_generation_batches(self, monkeypatch) -> None:
        from glyph.markush.eval import metrics_mcs

        class _MockModel:
            training = False

            def eval(self):
                pass

        samples = [
            {
                "id": f"s{idx}",
                "image": f"/tmp/{idx}.png",
                "user_text": "",
                "assistant_text": f"target-{idx}",
            }
            for idx in range(5)
        ]
        batch_sizes: list[int] = []
        captured: dict[str, Any] = {}

        def fake_generate_many(_model, batch):
            batch_sizes.append(len(batch))
            return [f"pred-{sample['id']}" for sample in batch], 0

        def fake_evaluate(preds, gts, **kwargs):
            captured["preds"] = preds
            captured["gts"] = gts
            return {"cxsmi_equality": 1.0}

        class _FakeWandb:
            run = object()
            logs: ClassVar[list[tuple[dict[str, Any], int | None]]] = []

            @staticmethod
            def log(payload, step=None):
                _FakeWandb.logs.append((payload, step))

        monkeypatch.setattr(metrics_mcs, "evaluate_markush_mcs", fake_evaluate)
        monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)

        callback = MetricsEvalCallback(
            samples,
            _MockProcessor(),
            protocol="ours",
            prefix="eval_ip5m",
            batch_size=2,
        )
        callback._generate_many = fake_generate_many  # type: ignore[method-assign]
        state = SimpleNamespace(is_world_process_zero=True, global_step=5)

        callback.on_evaluate(SimpleNamespace(), state, SimpleNamespace(), model=_MockModel())

        assert batch_sizes == [2, 2, 1]
        assert {row["id"] for row in captured["preds"]} == {f"s{idx}" for idx in range(5)}
        assert {row["id"] for row in captured["gts"]} == {f"s{idx}" for idx in range(5)}
        payload, step = _FakeWandb.logs[-1]
        assert step == 5
        assert payload["eval_ip5m/gen/batch_size"] == 2
        assert payload["eval_ip5m/gen/n_failed"] == 0

    def test_metrics_eval_callback_interval_steps_gates_runs(self) -> None:
        callback = MetricsEvalCallback(
            samples=[{"id": "s0", "image": "", "user_text": "", "assistant_text": "t"}],
            processor=_MockProcessor(),
            interval_steps=5,
        )
        generated: list[int] = []
        callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
            generated.append(1) or ["pred" for _ in batch],
            0,
        )

        class _Model:
            training = False

            def eval(self):
                pass

        for step in (3, 4):
            state = SimpleNamespace(is_world_process_zero=True, global_step=step)
            callback.on_evaluate(SimpleNamespace(), state, SimpleNamespace(), model=_Model())
        assert generated == []

        state = SimpleNamespace(is_world_process_zero=True, global_step=5)
        callback.on_evaluate(SimpleNamespace(), state, SimpleNamespace(), model=_Model())
        assert generated == [1]

    def test_metrics_eval_callback_rejects_nonpositive_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            MetricsEvalCallback(
                samples=[],
                processor=_MockProcessor(),
                batch_size=0,
            )

    def test_metrics_eval_callback_persists_metrics_to_disk(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from glyph.markush.eval import metrics_mcs

        monkeypatch.setattr(
            metrics_mcs,
            "evaluate_markush_mcs",
            lambda preds, gts, **kwargs: {"cxsmi_equality": 0.75},
        )

        callback = MetricsEvalCallback(
            samples=[{"id": "s0", "image": "", "user_text": "", "assistant_text": "t"}],
            processor=_MockProcessor(),
            protocol="ours",
            prefix="eval_ip5m",
        )
        callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
            ["pred" for _ in batch],
            0,
        )

        class _Model:
            training = False

            def eval(self):
                pass

        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(is_world_process_zero=True, global_step=7)
        callback.on_evaluate(args, state, SimpleNamespace(), model=_Model())

        records = [
            json.loads(line) for line in (tmp_path / "metrics_eval.jsonl").read_text().splitlines()
        ]
        assert len(records) == 1
        assert records[0]["step"] == 7
        assert records[0]["prefix"] == "eval_ip5m"
        assert records[0]["eval_ip5m/ours/cxsmi_equality"] == 0.75
        assert records[0]["eval_ip5m/gen/n_samples"] == 1

    def test_ocsr_metrics_eval_callback_scores_bare_smiles(self, tmp_path: Path) -> None:
        callback = OCSRMetricsEvalCallback(
            samples=[
                {
                    "id": "uspto_1",
                    "image": "",
                    "user_text": "Extract the molecular structure from this image as SMILES.",
                    "assistant_text": "CCO",
                }
            ],
            processor=_MockProcessor(),
            prefix="eval_ocsr_uspto",
        )
        callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
            ["C(C)O" for _ in batch],
            0,
        )

        class _Model:
            training = False

            def eval(self):
                pass

        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(is_world_process_zero=True, global_step=9)
        metrics: dict[str, Any] = {}

        callback.on_evaluate(args, state, SimpleNamespace(), model=_Model(), metrics=metrics)

        assert metrics["eval_ocsr_uspto_canon_exact_match"] == 1.0
        assert metrics["eval_ocsr_uspto_valid_smiles_rate"] == 1.0
        assert metrics["eval_ocsr_uspto_trainer_step"] == 9
        records = [
            json.loads(line) for line in (tmp_path / "metrics_eval.jsonl").read_text().splitlines()
        ]
        assert records[0]["eval_ocsr_uspto/canon_exact_match"] == 1.0

    def test_ocsr_metrics_eval_callback_counts_markush_xml_as_wrong(self) -> None:
        callback = OCSRMetricsEvalCallback(
            samples=[
                {
                    "id": "uspto_1",
                    "image": "",
                    "user_text": "Extract the molecular structure from this image as SMILES.",
                    "assistant_text": "CCO",
                }
            ],
            processor=_MockProcessor(),
            prefix="eval_ocsr_uspto",
        )
        callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
            ["<markush><cxsmi>CCO</cxsmi><stable></stable></markush>" for _ in batch],
            0,
        )

        class _Model:
            training = False

            def eval(self):
                pass

        metrics: dict[str, Any] = {}
        callback.on_evaluate(
            SimpleNamespace(),
            SimpleNamespace(is_world_process_zero=True, global_step=10),
            SimpleNamespace(),
            model=_Model(),
            metrics=metrics,
        )

        assert metrics["eval_ocsr_uspto_canon_exact_match"] == 0.0
        assert metrics["eval_ocsr_uspto_n_markush_xml_pred"] == 1
        assert metrics["eval_ocsr_uspto_markush_xml_prediction_rate"] == 1.0

    def test_ocsr_metrics_eval_callback_accepts_no_extension_markush_opt_in(self) -> None:
        callback = OCSRMetricsEvalCallback(
            samples=[
                {
                    "id": "uspto_1",
                    "image": "",
                    "user_text": "Extract the molecular structure from this image as SMILES.",
                    "assistant_text": "CCO",
                }
            ],
            processor=_MockProcessor(),
            prefix="eval_ocsr_uspto",
            prediction_mode=OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
        )
        callback._generate_many = lambda _model, batch: (  # type: ignore[method-assign]
            ["<markush><cxsmi>C(C)O</cxsmi><stable></stable></markush>" for _ in batch],
            0,
        )

        class _Model:
            training = False

            def eval(self):
                pass

        metrics: dict[str, Any] = {}
        callback.on_evaluate(
            SimpleNamespace(),
            SimpleNamespace(is_world_process_zero=True, global_step=10),
            SimpleNamespace(),
            model=_Model(),
            metrics=metrics,
        )

        assert metrics["eval_ocsr_uspto_canon_exact_match"] == 1.0
        assert metrics["eval_ocsr_uspto_valid_smiles_rate"] == 1.0
        assert metrics["eval_ocsr_uspto_n_markush_extension_pred"] == 0
        assert (
            metrics["eval_ocsr_uspto_prediction_mode"] == OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION
        )

    def test_ocsr_inline_callback_and_offline_eval_share_config_path(self) -> None:
        predictions = [
            {
                "id": "uspto_1",
                "prediction": "<markush><cxsmi>C(C)O</cxsmi><stable></stable></markush>",
            }
        ]
        ground_truths = [{"id": "uspto_1", "smiles": "CCO"}]
        config = OCSREvaluationConfig(prediction_mode=OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION)
        callback = OCSRMetricsEvalCallback(
            samples=[],
            processor=_MockProcessor(),
            prefix="eval_ocsr_uspto",
            prediction_mode=OCSR_PREDICTION_MODE_MARKUSH_NO_EXTENSION,
        )

        offline = flatten_ocsr_metrics(
            evaluate_ocsr_smiles(
                predictions,
                ground_truths,
                include_missing=True,
                return_details=False,
                config=config,
            ),
            prefix="eval_ocsr_uspto",
        )
        inline = callback._score_predictions(predictions, ground_truths)

        assert callback.evaluation_config.prediction_mode == config.prediction_mode
        assert inline == offline

    def test_detailed_logging_callback_persists_trainer_logs_to_disk(self, tmp_path: Path) -> None:
        callback = DetailedLoggingCallback()
        args = SimpleNamespace(output_dir=str(tmp_path))
        state = SimpleNamespace(is_world_process_zero=True, global_step=12)

        callback.on_log(args, state, SimpleNamespace(), logs={"loss": 0.5, "epoch": 1.0})
        callback.on_log(args, state, SimpleNamespace(), logs={"eval_loss": 0.4})

        records = [
            json.loads(line) for line in (tmp_path / "trainer_logs.jsonl").read_text().splitlines()
        ]
        assert len(records) == 2
        assert records[0]["loss"] == 0.5
        assert records[0]["step"] == 12
        assert records[1]["eval_loss"] == 0.4

    def test_metrics_eval_loader_matches_sft_target_contract(self, tmp_path: Path) -> None:
        row = {
            "id": "ip5m_1",
            "source": "ip5m",
            "split": "test",
            "task": "markush",
            "image": "ip5m.png",
            "image_has_text": False,
            "cxsmiles_opt": "<r>R1</r>CC",
            "annotation": (
                "<markush><cxsmi>WRONG_RENDERED_CXSMILES</cxsmi><stable>R1:alkyl</stable></markush>"
            ),
        }
        eval_jsonl = tmp_path / "eval_ip5m.jsonl"
        eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
        expected = build_training_example(
            row,
            STRUCTURE_ONLY_MARKUSH,
            rng=random.Random(1),
        )

        samples = _load_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=1,
            prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
        )

        assert samples == [
            {
                "id": "ip5m_1",
                "image": str(tmp_path / "ip5m.png"),
                "user_text": expected["messages"][0]["content"].removeprefix("<image>\n"),
                "assistant_text": expected["messages"][1]["content"],
            }
        ]

    def test_metrics_eval_loader_can_match_canonicalized_targets(self, tmp_path: Path) -> None:
        rows = [
            {
                "id": "good",
                "source": "ip5m",
                "split": "test",
                "task": "markush",
                "image": "good.png",
                "image_has_text": False,
                "cxsmiles_opt": "C(C)O",
            },
            {
                "id": "bad",
                "source": "ip5m",
                "split": "test",
                "task": "markush",
                "image": "bad.png",
                "image_has_text": False,
                "cxsmiles_opt": "not_a_smiles_XYZ123",
            },
        ]
        eval_jsonl = tmp_path / "eval_ip5m.jsonl"
        eval_jsonl.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

        samples = _load_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=2,
            prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
            canonicalize_markush_targets=True,
            require_canonical_markush_targets=True,
        )

        assert [sample["id"] for sample in samples] == ["good"]
        assert samples[0]["assistant_text"] == (
            "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
        )
        assert "canonicalized cxsmiles_opt" in samples[0]["user_text"]

    def test_metrics_eval_loader_skips_molecule_ocsr_rows(self, tmp_path: Path) -> None:
        row = {
            "id": "ocsr_row",
            "mode": "molecule_ocsr",
            "target_format": "SMILES",
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
            "images": ["ocsr.png"],
        }
        eval_jsonl = tmp_path / "mixed_eval.jsonl"
        eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

        samples = _load_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=1,
            prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
        )

        assert samples == []

    def test_metrics_eval_loader_matches_fast_eval_loader_for_canonical_rows(
        self, tmp_path: Path
    ) -> None:
        row = {
            "id": "canonical",
            "source": "ip5m",
            "split": "test",
            "task": "markush",
            "image": "canonical.png",
            "image_has_text": False,
            "cxsmiles_opt": "C(C)O",
        }
        eval_jsonl = tmp_path / "eval_ip5m.jsonl"
        eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

        metrics_samples = _load_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=1,
            prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
            canonicalize_markush_targets=True,
            require_canonical_markush_targets=True,
        )
        fast_samples = load_fast_eval_samples(
            eval_jsonl,
            tmp_path,
            STRUCTURE_ONLY_MARKUSH_PROMPT,
            canonicalize_markush_targets=True,
            require_canonical_markush_targets=True,
        )

        assert metrics_samples == [
            {
                "id": fast_samples[0].sample_id,
                "image": fast_samples[0].image_path,
                "user_text": fast_samples[0].prompt,
                "assistant_text": (
                    f"<markush><cxsmi>{fast_samples[0].ground_truth}</cxsmi>"
                    "<stable></stable></markush>"
                ),
            }
        ]
        assert "canonicalized cxsmiles_opt" in fast_samples[0].prompt

    def test_metrics_eval_loader_applies_markush_prompt_policy(self, tmp_path: Path) -> None:
        row = {
            "id": "canonical",
            "source": "ip5m",
            "split": "test",
            "task": "markush",
            "image": "canonical.png",
            "image_has_text": False,
            "cxsmiles_opt": "CC",
        }
        eval_jsonl = tmp_path / "eval_ip5m.jsonl"
        eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

        samples = _load_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=1,
            prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
            canonicalize_markush_targets=True,
            require_canonical_markush_targets=True,
            markush_prompt_policy=MARKUSH_PROMPT_POLICY_VISIBLE_SG,
        )

        assert VISIBLE_SG_INSTRUCTION in samples[0]["user_text"]
        assert samples[0]["user_text"].index(VISIBLE_SG_INSTRUCTION) < samples[0][
            "user_text"
        ].index("Target format:")

    def test_ocsr_metrics_eval_loader_loads_unified_ocsr_rows(self, tmp_path: Path) -> None:
        row = {
            "id": "uspto_1",
            "task": "ocsr",
            "source": "uspto",
            "split": "test",
            "image": "uspto_1.png",
            "smiles": "C(C)O",
        }
        eval_jsonl = tmp_path / "eval_ocsr_uspto.jsonl"
        eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

        samples = _load_ocsr_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=1,
        )

        assert samples == [
            {
                "id": "uspto_1",
                "image": str(tmp_path / "uspto_1.png"),
                "user_text": (
                    "Extract the molecular structure from this image as SMILES.\n"
                    "Target format: return a SMILES string."
                ),
                "assistant_text": "CCO",
            }
        ]

    def test_ocsr_metrics_eval_loader_accepts_raw_smiles_rows(self, tmp_path: Path) -> None:
        row = {
            "id": "raw_ocsr_1",
            "image": "raw.png",
            "smiles": "C(C)O",
        }
        eval_jsonl = tmp_path / "raw_ocsr.jsonl"
        eval_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

        samples = _load_ocsr_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=1,
        )

        assert samples == [
            {
                "id": "raw_ocsr_1",
                "image": str(tmp_path / "raw.png"),
                "user_text": (
                    "Extract the molecular structure from this image as SMILES.\n"
                    "Target format: return a SMILES string."
                ),
                "assistant_text": "CCO",
            }
        ]

    def test_ocsr_metrics_eval_guard_rejects_zero_loaded_samples(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="loaded zero samples"):
            _require_ocsr_metric_samples([], tmp_path / "empty_ocsr.jsonl")

    def test_ocsr_metrics_eval_loader_preserves_prepared_prompt(self, tmp_path: Path) -> None:
        unified = {
            "id": "prepared_ocsr",
            "task": "ocsr",
            "source": "uspto",
            "split": "test",
            "image": "prepared.png",
            "smiles": "C(C)O",
        }
        prepared = build_training_example(
            unified,
            MOLECULE_OCSR,
            rng=random.Random(1),
        )
        eval_jsonl = tmp_path / "prepared_ocsr.jsonl"
        eval_jsonl.write_text(json.dumps(prepared) + "\n", encoding="utf-8")

        samples = _load_ocsr_metrics_samples(
            eval_jsonl,
            data_dir=tmp_path,
            limit=1,
        )

        assert samples == [
            {
                "id": prepared["id"],
                "image": str(tmp_path / "prepared.png"),
                "user_text": prepared["messages"][0]["content"].removeprefix("<image>\n"),
                "assistant_text": "CCO",
            }
        ]


# ---------------------------------------------------------------------------
# Config resolution (YAML + env + CLI merge)
# ---------------------------------------------------------------------------


class TestConfigResolution:
    """Regression coverage for the YAML -> CLI merge silent-override bug.

    A ``store_true``/``store_false`` flag left off the command line must never
    clobber a value the YAML set. Before the fix, the two Markush target
    canonicalization flags defaulted to ``False`` in argparse, and the merge
    treated that ``False`` as an explicit user value, silently flipping a
    ``true`` from the YAML to ``false``.
    """

    def _resolve(self, argv: list[str]) -> Any:
        parser = build_argument_parser()
        args = parser.parse_args(argv)
        # env={} isolates the merge from any ambient WANDB_* environment.
        return resolve_config(args, parser, argv=argv, env={})

    # -- the two canonicalization flags: YAML true survives when flag is absent

    def test_canonicalize_markush_targets_yaml_true_survives_no_cli(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("metrics_eval_canonicalize_markush_targets: true\n")
        cfg = self._resolve(["--config", str(cfg_path)])
        # The bug flipped this to False; the fix keeps the YAML value.
        assert cfg.metrics_eval_canonicalize_markush_targets is True

    def test_require_canonical_markush_targets_yaml_true_survives_no_cli(
        self, tmp_path: Path
    ) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("metrics_eval_require_canonical_markush_targets: true\n")
        cfg = self._resolve(["--config", str(cfg_path)])
        assert cfg.metrics_eval_require_canonical_markush_targets is True

    def test_both_canonicalization_flags_survive_via_real_config(self) -> None:
        """A shipped config's values must survive CLI resolution unchanged.

        Value-agnostic on purpose: this guards the shipped YAML against silent
        clobbering whatever the value is. The bug-detecting cases (YAML ``true``
        vs argparse's ``False`` default) are covered by the synthetic
        ``*_yaml_true_survives_no_cli`` tests above.
        """
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "sft_2b_markush_joint.yaml"
        if not cfg_path.exists():
            pytest.skip(f"missing real config at {cfg_path}")
        raw = yaml.safe_load(cfg_path.read_text())
        cfg = self._resolve(["--config", str(cfg_path)])
        for key in (
            "metrics_eval_canonicalize_markush_targets",
            "metrics_eval_require_canonical_markush_targets",
        ):
            assert getattr(cfg, key) is raw[key], (
                f"{key}: shipped YAML has {raw[key]!r} but resolution produced "
                f"{getattr(cfg, key)!r}"
            )

    # -- and they still flip when the user actually passes the flag

    def test_canonicalize_markush_targets_flag_flips_value(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("metrics_eval_canonicalize_markush_targets: false\n")
        cfg = self._resolve(
            ["--config", str(cfg_path), "--metrics-eval-canonicalize-markush-targets"]
        )
        assert cfg.metrics_eval_canonicalize_markush_targets is True

    def test_require_canonical_markush_targets_flag_flips_value(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("metrics_eval_require_canonical_markush_targets: false\n")
        cfg = self._resolve(
            ["--config", str(cfg_path), "--metrics-eval-require-canonical-markush-targets"]
        )
        assert cfg.metrics_eval_require_canonical_markush_targets is True

    # -- at least one other flag: paired store_true/store_false (vision_lora)

    def test_vision_lora_yaml_true_survives_no_cli(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("vision_lora: true\n")
        cfg = self._resolve(["--config", str(cfg_path)])
        assert cfg.vision_lora is True

    def test_vision_lora_flag_flips_value(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("vision_lora: true\n")
        cfg = self._resolve(["--config", str(cfg_path), "--no-vision-lora"])
        assert cfg.vision_lora is False

    def test_full_finetune_yaml_true_survives_no_cli(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("full_finetune: true\n")
        cfg = self._resolve(["--config", str(cfg_path)])
        assert cfg.full_finetune is True

    # -- explicit-CLI-key detection (the mechanism the merge relies on)

    def test_explicit_cli_keys_ignores_unpassed_store_true(self) -> None:
        parser = build_argument_parser()
        keys = explicit_cli_keys(parser, [])
        assert "metrics_eval_canonicalize_markush_targets" not in keys
        assert "vision_lora" not in keys

    def test_explicit_cli_keys_reports_passed_flags(self) -> None:
        parser = build_argument_parser()
        keys = explicit_cli_keys(
            parser,
            ["--metrics-eval-canonicalize-markush-targets", "--learning-rate", "1e-4"],
        )
        assert "metrics_eval_canonicalize_markush_targets" in keys
        assert "learning_rate" in keys
        # store_false paired option resolves to the shared dest.
        assert explicit_cli_keys(parser, ["--no-vision-lora"]) == {"vision_lora"}

    # -- resolved-config logging + silent-override guard (the tool-fix)

    def test_resolve_config_logs_resolved_config(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("metrics_eval_canonicalize_markush_targets: true\n")
        parser = build_argument_parser()
        argv = ["--config", str(cfg_path)]
        args = parser.parse_args(argv)
        with caplog.at_level("INFO", logger="glyph.markush.train.sft"):
            resolve_config(args, parser, argv=argv, env={})
        assert "Resolved training config:" in caplog.text
        assert "metrics_eval_canonicalize_markush_targets" in caplog.text

    def test_detect_silent_config_overrides_flags_unexplained_drift(self) -> None:
        drift = detect_silent_config_overrides(
            {"metrics_eval_canonicalize_markush_targets": True, "lora_r": 128},
            {"metrics_eval_canonicalize_markush_targets": False, "lora_r": 128},
            allowed_keys=set(),
        )
        assert drift == {"metrics_eval_canonicalize_markush_targets": (True, False)}

    def test_detect_silent_config_overrides_allows_explicit_change(self) -> None:
        drift = detect_silent_config_overrides(
            {"lora_r": 128},
            {"lora_r": 256},
            allowed_keys={"lora_r"},
        )
        assert drift == {}

    def test_resolve_config_raises_on_silent_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the guard to observe drift and confirm it fails loudly instead
        # of letting a silent override slip through into training.
        import glyph.markush.train.sft as sft_module

        monkeypatch.setattr(
            sft_module,
            "detect_silent_config_overrides",
            lambda *_a, **_k: {"vision_lora": (True, False)},
        )
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("vision_lora: true\n")
        parser = build_argument_parser()
        argv = ["--config", str(cfg_path)]
        args = parser.parse_args(argv)
        with pytest.raises(RuntimeError, match="silently changed"):
            resolve_config(args, parser, argv=argv, env={})

    def test_resolve_config_applies_env_override(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("wandb_project: from_yaml\n")
        parser = build_argument_parser()
        argv = ["--config", str(cfg_path)]
        args = parser.parse_args(argv)
        cfg = resolve_config(args, parser, argv=argv, env={"WANDB_PROJECT": "from_env"})
        # Env overrides YAML and must not trip the silent-override guard.
        assert cfg.wandb_project == "from_env"


# ---------------------------------------------------------------------------
# Resolved-config provenance (persist to disk + log to W&B)
# ---------------------------------------------------------------------------


class TestResolvedConfigProvenance:
    """The published HF repo ships no training config and the W&B run records
    only HF TrainingArguments, so the trainer must persist the resolved recipe
    next to the checkpoints and mirror it into the W&B run config.
    """

    def test_persist_resolved_config_writes_recipe(self, tmp_path: Path) -> None:
        cfg = SFTConfig()
        cfg.metrics_eval_canonicalize_markush_targets = True
        cfg.metrics_eval_require_canonical_markush_targets = True
        cfg.fused_cross_entropy = False

        path = persist_resolved_config(cfg, tmp_path)

        assert path == tmp_path / RESOLVED_CONFIG_FILENAME
        assert path.exists()
        with open(path) as f:
            dumped = yaml.safe_load(f)
        # The full resolved config is captured, including custom SFTConfig
        # fields absent from HF TrainingArguments.
        assert dumped["metrics_eval_canonicalize_markush_targets"] is True
        assert dumped["metrics_eval_require_canonical_markush_targets"] is True
        assert dumped["fused_cross_entropy"] is False
        assert dumped == cfg.to_dict()

    def test_persist_resolved_config_roundtrips_via_from_yaml(self, tmp_path: Path) -> None:
        cfg = SFTConfig()
        cfg.metrics_eval_canonicalize_markush_targets = True
        cfg.vision_lora = True

        path = persist_resolved_config(cfg, tmp_path)
        assert path is not None
        reloaded = SFTConfig.from_yaml(path)

        assert reloaded.metrics_eval_canonicalize_markush_targets is True
        assert reloaded.vision_lora is True

    def test_persist_resolved_config_is_defensive(self, tmp_path: Path) -> None:
        # output_dir is actually a file, so writing inside it fails. The helper
        # must warn and return None rather than crash training.
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("x")

        result = persist_resolved_config(SFTConfig(), blocker)

        assert result is None

    def test_resolved_config_callback_noop_without_active_run(self) -> None:
        # With no active W&B run (wandb.run is None or wandb missing), the
        # callback must be a silent no-op and never raise.
        callback = ResolvedConfigCallback(SFTConfig().to_dict())
        state = SimpleNamespace(is_world_process_zero=True)
        callback.on_train_begin(SimpleNamespace(), state, SimpleNamespace())

    def test_resolved_config_callback_skips_non_main_process(self) -> None:
        callback = ResolvedConfigCallback(SFTConfig().to_dict())
        state = SimpleNamespace(is_world_process_zero=False)
        callback.on_train_begin(SimpleNamespace(), state, SimpleNamespace())


# ---------------------------------------------------------------------------
# MarkushDataset
# ---------------------------------------------------------------------------


class TestMarkushDataset:
    def _record(self, idx: int = 0) -> dict[str, Any]:
        return {
            "id": f"sample_{idx}",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "<image>\nPatent text: wherein R1 is selected from alkyl...\n"
                        "Extract the Markush structure from this patent image."
                    ),
                },
                {"role": "assistant", "content": "<markush><cxsmi>c1ccccc1</cxsmi></markush>"},
            ],
            "images": [f"images/img_{idx}.png"],
        }

    def test_strips_image_placeholder(self, tmp_path: Path) -> None:
        ds = MarkushDataset([self._record()], root=tmp_path, text_dropout=0.0)
        item = ds[0]
        assert "<image>" not in item["user_text"]
        assert item["assistant_text"].startswith("<markush>")

    def test_text_dropout_deterministic_per_index(self, tmp_path: Path) -> None:
        # With dropout=1.0 every call drops the OCR/patent text.
        ds = MarkushDataset(
            [self._record(i) for i in range(5)], root=tmp_path, text_dropout=1.0, seed=42
        )
        first_pass = [ds[i]["user_text"] for i in range(5)]
        second_pass = [ds[i]["user_text"] for i in range(5)]
        assert first_pass == second_pass
        # All should be the short extraction instruction (no patent text).
        for ut in first_pass:
            assert "Patent text" not in ut
            assert "Extract the Markush structure" in ut

    def test_text_dropout_zero_keeps_all(self, tmp_path: Path) -> None:
        ds = MarkushDataset([self._record()], root=tmp_path, text_dropout=0.0)
        assert "Patent text" in ds[0]["user_text"]

    def test_set_epoch_changes_dropout_seed(self, tmp_path: Path) -> None:
        # With partial dropout, set_epoch must alter the sequence of decisions.
        records = [self._record(i) for i in range(50)]
        ds = MarkushDataset(records, root=tmp_path, text_dropout=0.5, seed=42)
        before = [ds[i]["user_text"] for i in range(50)]
        ds.set_epoch(1)
        after = [ds[i]["user_text"] for i in range(50)]
        assert before != after, "set_epoch should change dropout pattern"

    def test_image_path_resolution(self, tmp_path: Path) -> None:
        ds = MarkushDataset([self._record()], root=tmp_path, text_dropout=0.0)
        item = ds[0]
        # Path layout: root / images/img_0.png
        assert item["image"] is not None
        assert item["image"].endswith("images/img_0.png")
        assert str(tmp_path) in item["image"]

    def test_image_path_legacy_data_fallback(self, tmp_path: Path, caplog) -> None:
        import logging

        MarkushDataset._fallback_warned = False
        legacy_dir = tmp_path / "data" / "images"
        legacy_dir.mkdir(parents=True)
        legacy_image = legacy_dir / "x.png"
        Image.new("RGB", (8, 8)).save(legacy_image)
        record = self._record()
        record["images"] = ["images/x.png"]

        ds = MarkushDataset([record], root=tmp_path, text_dropout=0.0)
        with caplog.at_level(logging.WARNING, logger="glyph.markush.train.sft"):
            item = ds[0]

        assert Path(item["image"]) == legacy_image
        assert "legacy 'data/' layout" in " ".join(rec.message for rec in caplog.records)

    def test_image_path_prefers_portable_root(self, tmp_path: Path) -> None:
        portable_dir = tmp_path / "images"
        portable_dir.mkdir(parents=True)
        portable_image = portable_dir / "x.png"
        Image.new("RGB", (8, 8)).save(portable_image)
        legacy_dir = tmp_path / "data" / "images"
        legacy_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(legacy_dir / "x.png")
        record = self._record()
        record["images"] = ["images/x.png"]

        ds = MarkushDataset([record], root=tmp_path, text_dropout=0.0)

        assert Path(ds[0]["image"]) == portable_image

    def test_returns_assistant_text_key(self, tmp_path: Path) -> None:
        """Dataset items must use ``assistant_text`` (not ``target``)."""
        ds = MarkushDataset([self._record()], root=tmp_path, text_dropout=0.0)
        item = ds[0]
        assert "assistant_text" in item
        assert "target" not in item

    def test_preserves_molecule_ocsr_bare_smiles_target(self, tmp_path: Path) -> None:
        record = {
            "id": "ocsr",
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
            "images": ["images/ocsr.png"],
            "mode": "molecule_ocsr",
            "target_format": "SMILES",
        }
        ds = MarkushDataset([record], root=tmp_path, text_dropout=0.0)
        item = ds[0]

        assert "<image>" not in item["user_text"]
        assert "Target format: return a SMILES string." in item["user_text"]
        assert item["assistant_text"] == "CCO"


# ---------------------------------------------------------------------------
# MarkushCollator
# ---------------------------------------------------------------------------


def _make_image(tmp_path: Path, name: str = "img.png") -> Path:
    path = tmp_path / name
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(path)
    return path


def _sample(
    idx: int,
    image_path: str | None,
    *,
    user_text: str = "Extract the Markush structure from this patent image.",
    assistant_text: str = "<markush><cxsmi>c1ccccc1</cxsmi></markush>",
) -> dict[str, Any]:
    return {
        "id": f"sample_{idx}",
        "image": image_path,
        "user_text": user_text,
        "assistant_text": assistant_text,
    }


class TestMarkushCollator:
    @patch("glyph.markush.train.sft.process_vision_info", side_effect=_mock_process_vision_info)
    def test_basic_masking(self, mock_pvi, tmp_path: Path) -> None:
        proc = _MockProcessor()
        col = MarkushCollator(proc)
        img = _make_image(tmp_path)
        batch = [_sample(0, str(img))]
        out = col(batch)

        assert "input_ids" in out and "labels" in out
        assert out["input_ids"].shape == out["labels"].shape

        labels = out["labels"][0]
        masked = (labels == -100).sum().item()
        unmasked = (labels != -100).sum().item()
        assert masked > 0, "prompt tokens must be masked"
        assert unmasked > 0, "target tokens must NOT all be masked"

    @patch("glyph.markush.train.sft.process_vision_info", side_effect=_mock_process_vision_info)
    def test_padding_tokens_masked_in_labels(self, mock_pvi, tmp_path: Path) -> None:
        proc = _MockProcessor()
        col = MarkushCollator(proc)
        img = _make_image(tmp_path)
        # Two samples: one with short target, one with much longer target,
        # forces padding on the shorter sample after right-padding.
        batch = [
            _sample(0, str(img), assistant_text="A"),
            _sample(1, str(img), assistant_text="A B C D E F G H I J"),
        ]
        out = col(batch)
        labels = out["labels"]
        input_ids = out["input_ids"]
        # Every position where input_ids == pad_token_id must be -100 in labels.
        pad_id = proc.tokenizer.pad_token_id
        pad_mask = input_ids == pad_id
        if pad_mask.any():
            assert (labels[pad_mask] == -100).all(), "all PAD positions must be -100"

    @patch("glyph.markush.train.sft.process_vision_info", side_effect=_mock_process_vision_info)
    def test_prompt_target_boundary_no_off_by_one(self, mock_pvi, tmp_path: Path) -> None:
        """Verify the first non-masked label IS a target token, not a prompt token."""
        proc = _MockProcessor()
        col = MarkushCollator(proc)
        img = _make_image(tmp_path)

        target = "UNIQUETARGETWORD"
        batch = [_sample(0, str(img), assistant_text=target)]
        out = col(batch)

        labels = out["labels"][0]
        input_ids = out["input_ids"][0]

        # Find first non-masked label position
        non_masked = (labels != -100).nonzero(as_tuple=True)[0]
        assert len(non_masked) > 0
        first_unmasked_pos = non_masked[0].item()

        # The token at that position should be a target token (not pad).
        first_token_id = input_ids[first_unmasked_pos].item()
        assert first_token_id != proc.tokenizer.pad_token_id

        # And the prompt token IMMEDIATELY before should be masked.
        if first_unmasked_pos > 0:
            assert labels[first_unmasked_pos - 1].item() == -100

    @patch("glyph.markush.train.sft.process_vision_info", side_effect=_mock_process_vision_info)
    def test_molecule_ocsr_smiles_target_has_unmasked_labels(
        self, mock_pvi, tmp_path: Path
    ) -> None:
        proc = _MockProcessor()
        col = MarkushCollator(proc)
        img = _make_image(tmp_path)
        batch = [
            _sample(
                0,
                str(img),
                user_text=(
                    "Extract the molecular structure from this image as SMILES.\n"
                    "Target format: return a SMILES string."
                ),
                assistant_text="CCO",
            )
        ]

        out = col(batch)
        labels = out["labels"][0]

        assert (labels != -100).any(), "bare SMILES response tokens must be supervised"

    @patch("glyph.markush.train.sft.process_vision_info", side_effect=_mock_process_vision_info)
    def test_batch_label_shape_matches_inputs(self, mock_pvi, tmp_path: Path) -> None:
        proc = _MockProcessor()
        col = MarkushCollator(proc)
        img = _make_image(tmp_path)
        batch = [_sample(i, str(img), assistant_text=f"T{i}") for i in range(3)]
        out = col(batch)
        assert out["labels"].shape == out["input_ids"].shape
        assert out["input_ids"].shape[0] == 3

    def test_image_augmentation_preserves_size_and_mode(self, tmp_path: Path) -> None:
        proc = _MockProcessor()
        col = MarkushCollator(
            proc,
            image_augmentation=True,
            image_augmentation_prob=1.0,
            seed=123,
        )
        img = Image.open(_make_image(tmp_path)).convert("RGB")
        aug = col._augment_image(img)

        assert aug.size == img.size
        assert aug.mode == "RGB"


# ---------------------------------------------------------------------------
# load_records (JSONL loading)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# train() guardrails: eval-size sanity + auto-resume detection
# ---------------------------------------------------------------------------


class TestTrainGuardrails:
    def test_eval_size_too_large_raises(self, tmp_path: Path, monkeypatch) -> None:
        """If eval_size >= len(records), train() must raise ValueError with a
        descriptive message naming both numbers. We stop execution before any
        model load by raising in the split step, so no real weights are
        needed."""
        from glyph.markush.train import sft as sft_module

        # Build a tiny JSONL with 3 records and an eval_size of 1000 (default).
        data = tmp_path / "tiny.jsonl"
        records = [
            {
                "id": f"r{i}",
                "messages": [
                    {"role": "user", "content": "<image>\nExtract."},
                    {"role": "assistant", "content": "T"},
                ],
                "images": [f"images/r{i}.png"],
            }
            for i in range(3)
        ]
        data.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        # Stub out model + processor loading; the function should raise BEFORE
        # those would be needed in practice, but the call order in train()
        # is: load processor -> load model -> load records -> SPLIT (raises).
        # So we still need to make those calls cheap.
        monkeypatch.setattr(sft_module, "setup_processor", lambda _p: _MockProcessor())
        monkeypatch.setattr(sft_module, "setup_model", lambda *a, **kw: object())

        cfg = SFTConfig(
            model_path=str(tmp_path),  # not actually loaded
            data_path=str(data),
            output_dir=str(tmp_path / "out"),
            eval_size=1000,
        )
        with pytest.raises(ValueError, match=r"3 records.*eval_size=1000"):
            sft_module.train(cfg)

    def test_auto_resume_warning_when_checkpoint_exists(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """When output_dir already contains a checkpoint-* directory, train()
        must log a clear warning before constructing Trainer."""
        import logging

        from glyph.markush.train import sft as sft_module

        # Pre-populate output_dir with a fake checkpoint.
        output_dir = tmp_path / "out"
        (output_dir / "checkpoint-500").mkdir(parents=True)

        # Build a JSONL big enough to pass the eval-split sanity check.
        data = tmp_path / "data.jsonl"
        records = [
            {
                "id": f"r{i}",
                "messages": [
                    {"role": "user", "content": "<image>\nExtract."},
                    {"role": "assistant", "content": "T"},
                ],
                "images": [f"images/r{i}.png"],
            }
            for i in range(20)
        ]
        data.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        # Stub everything that would touch real weights / GPUs / W&B.
        monkeypatch.setattr(sft_module, "setup_processor", lambda _p: _MockProcessor())
        monkeypatch.setattr(sft_module, "setup_model", lambda *a, **kw: object())

        # Stub Trainer so we don't actually train. We want to confirm the
        # warning fired BEFORE Trainer() is constructed, so we raise inside
        # the constructor to short-circuit the rest of train().
        class _StopHereError(RuntimeError):
            pass

        def _fake_trainer(*args, **kwargs) -> None:
            raise _StopHereError("stop after warning")

        monkeypatch.setattr(sft_module, "Trainer", _fake_trainer)
        # Skip TrainingArguments construction too, since it can be expensive.
        monkeypatch.setattr(
            sft_module,
            "TrainingArguments",
            lambda **kwargs: type("FakeArgs", (), kwargs)(),
        )

        monkeypatch.delenv("WANDB_MODE", raising=False)
        cfg = SFTConfig(
            model_path=str(tmp_path),
            data_path=str(data),
            output_dir=str(output_dir),
            eval_size=2,
            no_wandb=True,
        )

        with (
            caplog.at_level(logging.WARNING, logger="glyph.markush.train.sft"),
            pytest.raises(_StopHereError),
        ):
            sft_module.train(cfg)

        assert os.environ["WANDB_MODE"] == "disabled"
        warn_text = " ".join(rec.message for rec in caplog.records)
        assert "checkpoint-500" in warn_text, (
            f"expected resume warning naming the checkpoint, got: {warn_text!r}"
        )
        assert "auto-resume" in warn_text.lower() or "different --output-dir" in warn_text


def _record(user: str, assistant: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


class TestTextLengthFilter:
    def test_disabled_filter_keeps_everything(self) -> None:
        records = [_record("u" * 10, "a" * 10_000)]
        kept, stats = filter_records_by_text_length(records)
        assert kept == records
        assert stats == {"kept": 1, "dropped_user": 0, "dropped_assistant": 0}

    def test_zero_bound_means_disabled(self) -> None:
        records = [_record("u" * 10, "a" * 10_000)]
        kept, stats = filter_records_by_text_length(records, max_assistant_chars=0)
        assert kept == records
        assert stats["kept"] == 1

    def test_drops_long_assistant_and_counts(self) -> None:
        records = [
            _record("u", "a" * 100),
            _record("u", "a" * 5000),
            _record("u" * 5000, "a"),
        ]
        kept, stats = filter_records_by_text_length(
            records, max_user_chars=4000, max_assistant_chars=4000
        )
        assert len(kept) == 1
        assert stats == {"kept": 1, "dropped_user": 1, "dropped_assistant": 1}


class TestFusedCrossEntropy:
    def _reference_loss(self, logits, labels, vocab_size, num_items_in_batch=None):
        from transformers.loss.loss_utils import ForCausalLMLoss

        return ForCausalLMLoss(
            logits.clone(), labels, vocab_size, num_items_in_batch=num_items_in_batch
        )

    def test_wrapper_matches_reference_semantics(self, monkeypatch) -> None:
        """The wrapper's shift/flatten/normalize logic must mirror ForCausalLMLoss.

        FLA's Triton kernel needs a GPU, so substitute a torch-backed stand-in
        with the same interface; this isolates the wrapper logic under test.
        """
        import types

        class _TorchCE:
            def __init__(self, ignore_index=-100, reduction="mean", **_kwargs) -> None:
                self.ignore_index = ignore_index
                self.reduction = reduction

            def __call__(self, logits, target):
                return torch.nn.functional.cross_entropy(
                    logits.float(),
                    target,
                    ignore_index=self.ignore_index,
                    reduction=self.reduction,
                )

        fake_modules = types.ModuleType("fla.modules")
        fake_modules.FusedCrossEntropyLoss = _TorchCE
        fake_fla = types.ModuleType("fla")
        fake_fla.modules = fake_modules
        monkeypatch.setitem(sys.modules, "fla", fake_fla)
        monkeypatch.setitem(sys.modules, "fla.modules", fake_modules)

        from glyph.markush.train.sft import _fused_cross_entropy_loss_function

        loss_fn = _fused_cross_entropy_loss_function()
        assert loss_fn is not None

        torch.manual_seed(0)
        batch, seq, vocab = 2, 9, 31
        logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        labels[0, :3] = -100  # masked prompt tokens

        got = loss_fn(logits, labels, vocab)
        want = self._reference_loss(logits, labels, vocab)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

        # Gradient-accumulation path: sum reduction normalized by token count.
        num_items = torch.tensor(11.0)
        got_sum = loss_fn(logits, labels, vocab, num_items_in_batch=num_items)
        want_sum = self._reference_loss(logits, labels, vocab, num_items_in_batch=num_items)
        torch.testing.assert_close(got_sum, want_sum, rtol=1e-5, atol=1e-6)

    def test_install_returns_false_without_fla(self, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def block_fla(name, *args, **kwargs):
            if name.startswith("fla"):
                raise ImportError("fla not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "fla", raising=False)
        monkeypatch.delitem(sys.modules, "fla.modules", raising=False)
        monkeypatch.setattr(builtins, "__import__", block_fla)

        class _Model:
            pass

        assert install_fused_cross_entropy(_Model()) is False

    @staticmethod
    def _fake_fla(monkeypatch, ce_cls) -> None:
        """Register a fake ``fla.modules`` exposing ``ce_cls`` as the kernel."""
        import types

        fake_modules = types.ModuleType("fla.modules")
        fake_modules.FusedCrossEntropyLoss = ce_cls
        fake_fla = types.ModuleType("fla")
        fake_fla.modules = fake_modules
        monkeypatch.setitem(sys.modules, "fla", fake_fla)
        monkeypatch.setitem(sys.modules, "fla.modules", fake_modules)

    def test_install_sets_loss_on_peft_base_model(self, monkeypatch) -> None:
        # A torch-backed stand-in so the install-time kernel probe succeeds
        # (fla's real Triton kernel needs a GPU); install must reach through the
        # PEFT wrapper and set the loss on the base model.
        class _TorchCE:
            def __init__(self, ignore_index=-100, reduction="mean", **_kwargs) -> None:
                self.ignore_index = ignore_index
                self.reduction = reduction

            def __call__(self, logits, target):
                return torch.nn.functional.cross_entropy(
                    logits.float(),
                    target,
                    ignore_index=self.ignore_index,
                    reduction=self.reduction,
                )

        self._fake_fla(monkeypatch, _TorchCE)

        class _Base:
            pass

        class _Peft:
            def __init__(self) -> None:
                self._base = _Base()

            def get_base_model(self):
                return self._base

        peft_model = _Peft()
        assert install_fused_cross_entropy(peft_model) is True
        assert callable(peft_model._base.loss_function)

    # -- fallback hardening: fla present but its kernel fails on this arch ----

    def test_install_falls_back_when_kernel_probe_fails(self, monkeypatch, caplog) -> None:
        """fla imports fine but its kernel errors on the local GPU.

        The trainer must degrade to standard CE (warn + leave the base loss
        untouched) rather than crash on the first training step. Simulated CPU-
        side by a kernel stand-in that raises the classic unsupported-arch error.
        """

        class _FailingCE:
            def __init__(self, **_kwargs) -> None:
                pass

            def __call__(self, logits, target):
                raise RuntimeError("no kernel image is available for execution on the device")

        self._fake_fla(monkeypatch, _FailingCE)

        class _Base:
            loss_function = "ORIGINAL"

        class _Peft:
            def __init__(self) -> None:
                self._base = _Base()

            def get_base_model(self):
                return self._base

        peft_model = _Peft()
        with caplog.at_level(logging.WARNING, logger="glyph.markush.train.sft"):
            installed = install_fused_cross_entropy(peft_model)

        assert installed is False
        # Fell back cleanly: the base model's loss stays as it was.
        assert peft_model._base.loss_function == "ORIGINAL"
        assert "falling back to standard cross-entropy" in caplog.text

    def test_probe_runs_forward_and_backward(self) -> None:
        """The probe invokes the loss once and backpropagates (CPU-safe)."""
        from glyph.markush.train.sft import _probe_fused_cross_entropy

        calls: list[tuple] = []

        def fake_loss(logits, labels, vocab_size=None, **kwargs):
            calls.append((tuple(logits.shape), vocab_size, tuple(labels.shape)))
            return (logits.float() ** 2).sum()

        _probe_fused_cross_entropy(fake_loss, device=torch.device("cpu"), dtype=torch.float32)

        assert calls, "probe did not invoke the loss function"
        shape, vocab, _ = calls[0]
        assert shape == (1, 4, 16)
        assert vocab == 16

    def test_probe_propagates_kernel_error(self) -> None:
        """A failing kernel surfaces out of the probe for the caller to catch."""
        from glyph.markush.train.sft import _probe_fused_cross_entropy

        def boom(*_a, **_k):
            raise RuntimeError("kernel exploded")

        with pytest.raises(RuntimeError, match="kernel exploded"):
            _probe_fused_cross_entropy(boom, device=torch.device("cpu"), dtype=torch.float32)

    def test_model_compute_placement_prefers_floating_param(self) -> None:
        from glyph.markush.train.sft import _model_compute_placement

        model = torch.nn.Linear(4, 3)  # CPU, float32
        device, dtype = _model_compute_placement(model)
        assert device.type == "cpu"
        assert dtype == torch.float32

    def test_model_compute_placement_without_parameters(self, monkeypatch) -> None:
        from glyph.markush.train.sft import _model_compute_placement

        # No parameters and no CUDA -> the safe CPU/fp32 fallback.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        class _NoParams:
            pass

        device, dtype = _model_compute_placement(_NoParams())
        assert device.type == "cpu"
        assert dtype == torch.float32


class TestLoadRecords:
    def test_load_valid_jsonl(self, tmp_path: Path) -> None:
        from glyph.markush.train.sft import load_records

        f = tmp_path / "data.jsonl"
        records = [
            {"id": "a", "messages": [], "images": []},
            {"id": "b", "messages": [], "images": []},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        loaded = load_records(f)
        assert len(loaded) == 2
        assert loaded[0]["id"] == "a"

    def test_load_skips_blank_and_invalid_lines(self, tmp_path: Path) -> None:
        from glyph.markush.train.sft import load_records

        f = tmp_path / "data.jsonl"
        f.write_text('{"id": "a"}\n\nnot valid json\n{"id": "b"}\n')
        loaded = load_records(f)
        assert [r["id"] for r in loaded] == ["a", "b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
