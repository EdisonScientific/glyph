"""Baseline-recipe config aligned to MolNexTR's training knobs.

This is the *production* recipe target. The compact variant ``TinyRecipe``
derives from this and shrinks dimensions so end-to-end infrastructure tests
run cheaply. We never reduce/widen production knobs implicitly — every
difference between the baseline and the compact variant must be visible in
the diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .smiles_tokenizer import EOS_ID, PAD_ID, SOS_ID


@dataclass
class BaselineRecipe:
    """MolNexTR-style baseline knobs for random-init reproduction.

    These values mirror MolNexTR's `exps/train.sh` defaults except where
    explicitly noted. We intentionally do not fix bond/coord-bin defaults that
    only matter for the chartok_coords format — that wiring is a later step.
    """

    # Encoder
    encoder_name: str = "swin_base_patch4_window12_384"
    input_size: int = 384
    use_checkpoint: bool = True
    # Generic ImageNet pretraining for the vision backbone IS allowed under
    # the current constraint. OCSR-specific pretrained checkpoints (MolScribe,
    # MolNexTR, MolSight, MolParser, MolGrapher, DECIMER, MarkushGrapher etc.)
    # remain forbidden unless explicitly approved later. See
    # notes/constraint_update_imagenet_pretraining.md.
    backbone_pretrained: bool = True

    # Decoder (transformer arm; we will add coords/edges heads later)
    decoder_kind: str = "transformer"
    embed_dim: int = 256
    dec_num_layers: int = 6
    dec_attn_heads: int = 8
    dec_ff_dim: int = 1024
    max_target_len: int = 480  # FORMAT_INFO['chartok_coords']['max_len']
    label_smoothing: float = 0.1

    # Optimization
    encoder_lr: float = 4e-4
    decoder_lr: float = 4e-4
    weight_decay: float = 1e-6
    warmup_ratio: float = 0.02
    cosine_schedule: bool = True
    grad_clip_max_norm: float = 5.0
    epochs: int = 40

    # Dataloading
    batch_size: int = 256  # global; scale via gradient accumulation per node
    num_workers: int = 8

    # Token/vocab contract
    vocab_size: int = 101  # matches MolNexTR/MolScribe vocab_chars.json
    sos_id: int = SOS_ID
    eos_id: int = EOS_ID
    pad_id: int = PAD_ID

    # Reproducibility
    seed: int = 42

    # W&B project
    wandb_project: str = "glyph-ocsr"
    wandb_entity: str | None = None

    # Notes for posterity. Not used by code.
    # Documentation-only metadata; not used by code. Aligned to the
    # constraint update at notes/constraint_update_imagenet_pretraining.md.
    notes: dict = field(
        default_factory=lambda: {
            "primary_recipe": "MolNexTR",
            "secondary_recipe": "MolScribe",
            "imagenet_pretrained_backbone_allowed": True,
            "ocsr_specific_pretrained_checkpoints_forbidden": True,
            "single_inference_model": True,
        }
    )


@dataclass
class SwinLargeRecipe(BaselineRecipe):
    """Swin-Large 384 ImageNet-pretrained (196M backbone params)."""

    encoder_name: str = "swin_large_patch4_window12_384"


@dataclass
class TinyRecipe(BaselineRecipe):
    """Compact variant of ``BaselineRecipe`` used for infrastructure tests."""

    encoder_name: str = "swin_tiny_patch4_window7_224"
    input_size: int = 224
    use_checkpoint: bool = False
    backbone_pretrained: bool = False  # stays weight-download-free
    embed_dim: int = 128
    dec_num_layers: int = 2
    dec_attn_heads: int = 4
    dec_ff_dim: int = 256
    max_target_len: int = 64
    epochs: int = 1
    batch_size: int = 4
    num_workers: int = 0
