"""OCSR model: a timm image encoder with a transformer SMILES decoder.

The architecture is configured by :class:`OCSRModelConfig`; production recipes
supply the checkpoint's full dimensions while tests can still use its compact
defaults. ImageNet backbone weights are optional, and OCSR checkpoints load
only through the explicit inference and evaluation surfaces.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from .config import OCSRModelConfig


class OCSREncoder(nn.Module):
    """Vision backbone and projection used by the OCSR model."""

    def __init__(self, cfg: OCSRModelConfig):
        super().__init__()
        # Constraint: generic ImageNet vision-backbone pretraining is allowed
        # (see notes/constraint_update_imagenet_pretraining.md). OCSR-
        # specific pretrained checkpoints (MolScribe, MolNexTR, MolSight, etc.)
        # are NOT loaded here; the only weights that ``timm.create_model`` can
        # pull are the upstream ImageNet weights for ``cfg.encoder_name``.
        # ``cfg.backbone_pretrained`` defaults to False for OCSRModelConfig so
        # lightweight instantiations stay weight-download-free; BaselineRecipe
        # sets True.
        # Only pass img_size to models that support it (Swin, EfficientViT, etc.);
        # ConvNeXt-style backbones determine input shape at forward time.
        create_kwargs = {
            "pretrained": cfg.backbone_pretrained,
            "features_only": True,
        }
        # Heuristic: Swin and EfficientViT families need explicit img_size
        if any(name in cfg.encoder_name.lower() for name in ("swin", "efficientvit", "swinv2")):
            create_kwargs["img_size"] = cfg.input_size
        self.backbone = timm.create_model(
            cfg.encoder_name,
            **create_kwargs,
        )
        feat_dim = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Linear(feat_dim, cfg.embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)[-1]  # [B, C, H, W] or [B, H, W, C]
        # timm Swin returns NHWC by default; detect by matching the configured
        # channel dimension to the actual axis size.
        target_c = self.backbone.feature_info.channels()[-1]
        if feats.dim() == 4 and feats.shape[-1] == target_c and feats.shape[1] != target_c:
            feats = feats.permute(0, 3, 1, 2).contiguous()
        _b, _c, _h, _w = feats.shape
        seq = feats.flatten(2).transpose(1, 2)  # [B, HW, C]
        return self.proj(seq)  # [B, HW, embed_dim]


class OCSRDecoder(nn.Module):
    """Autoregressive transformer decoder over image memory."""

    def __init__(self, cfg: OCSRModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_id)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.dec_attn_heads,
            dim_feedforward=cfg.dec_ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.dec_num_layers)
        self.head = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        x = self.tok_embed(tokens) + self.pos_embed(pos)
        causal = torch.triu(torch.full((t, t), float("-inf"), device=tokens.device), diagonal=1)
        out = self.decoder(tgt=x, memory=memory, tgt_mask=causal)
        return self.head(out)


class OCSRModel(nn.Module):
    """End-to-end image encoder and SMILES decoder."""

    def __init__(self, cfg: OCSRModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = OCSREncoder(cfg)
        self.decoder = OCSRDecoder(cfg)

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(image)
        return self.decoder(tokens, memory)


class GraphPredictor(nn.Module):
    """MLP that predicts pairwise bond classes from atom hidden states.

    Given decoder hidden states ``H`` of shape ``[B, T, D]`` and per-atom
    indices ``I`` of shape ``[B, A]`` (with PAD entries pointing at a
    sentinel index 0), gather ``H[b, I[b]]`` -> ``[B, A, D]``, build a
    pairwise concat ``[B, A, A, 2D]``, run a 2-layer MLP, return logits
    ``[B, 7, A, A]``.
    """

    def __init__(self, decoder_dim: int, n_classes: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, n_classes),
        )

    def forward(self, hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, _t, d = hidden.shape
        a = indices.shape[1]
        batch_id = torch.arange(b, device=hidden.device).unsqueeze(1).expand(b, a)
        gathered = hidden[batch_id, indices]  # [B, A, D]
        h_i = gathered.unsqueeze(2).expand(b, a, a, d)
        h_j = gathered.unsqueeze(1).expand(b, a, a, d)
        edge_in = torch.cat([h_i, h_j], dim=-1)  # [B, A, A, 2D]
        return self.mlp(edge_in).permute(0, 3, 1, 2)  # [B, n_classes, A, A]


class MultitaskModel(nn.Module):
    """OCSRModel + GraphPredictor head.

    The decoder must expose its pre-head hidden states. We modify the model so
    that ``forward_with_hidden`` returns both logits and hidden states.
    """

    def __init__(self, cfg: OCSRModelConfig, n_bond_classes: int = 7):
        super().__init__()
        self.cfg = cfg
        self.encoder = OCSREncoder(cfg)
        self.decoder = OCSRDecoder(cfg)
        self.graph_predictor = GraphPredictor(cfg.embed_dim, n_classes=n_bond_classes)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def decode_hidden(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        x = self.decoder.tok_embed(tokens) + self.decoder.pos_embed(pos)
        causal = torch.triu(torch.full((t, t), float("-inf"), device=tokens.device), diagonal=1)
        return self.decoder.decoder(tgt=x, memory=memory, tgt_mask=causal)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder.head(hidden)

    def forward(
        self, image: torch.Tensor, tokens: torch.Tensor, atom_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode(image)
        hidden = self.decode_hidden(tokens, memory)
        logits = self.head(hidden)
        # Clamp atom_indices to valid range in case any per-sample position
        # equals or exceeds hidden.shape[1] (can happen when teacher forcing
        # drops the final token but a coord token landed at the very end).
        max_idx = hidden.shape[1] - 1
        ai = atom_indices.clamp(min=0, max=max_idx)
        edges_logits = self.graph_predictor(hidden, ai)
        return logits, edges_logits
