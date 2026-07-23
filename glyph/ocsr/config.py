"""Central configuration constants and dataclasses for the OCSR model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OCSRModelConfig:
    """OCSR model architecture config.

    Holds the encoder/decoder dimensions and token contract consumed by
    :class:`~glyph.ocsr.model.OCSRModel`. The defaults here are compact so
    the model can be instantiated cheaply in tests; production values are
    supplied by :class:`~glyph.ocsr.baseline_config.BaselineRecipe`.
    """

    # Model
    encoder_name: str = "swin_tiny_patch4_window7_224"
    input_size: int = 224
    embed_dim: int = 256
    dec_num_layers: int = 2
    dec_attn_heads: int = 4
    dec_ff_dim: int = 512
    max_len: int = 32
    vocab_size: int = 64  # tiny placeholder; real tokenizer vocab is set later

    # Optimization (compact defaults; production values come from BaselineRecipe)
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-6
    label_smoothing: float = 0.0

    # Reproducibility
    seed: int = 42

    # W&B project naming
    wandb_project: str = "glyph-ocsr"
    wandb_entity: str | None = None

    # Special token ids in the tokenizer
    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2

    # ImageNet-pretrained-backbone toggle.
    # Defaults False so lightweight instantiations do not trigger a timm
    # weight download. Production recipes (BaselineRecipe) flip this to True
    # to match the public MolNexTR/MolScribe training path.
    backbone_pretrained: bool = False

    extras: dict = field(default_factory=dict)
