"""Tiny in-memory dataset wrapper for lightweight testing only.

Wraps a list of TrainRow records, renders images on demand using the toy
RDKit renderer, and returns ``(image_tensor, target_smiles_string)``.

Production datasets (MolNexTR-style with dynamic Indigo, augmentation,
char/coord/edge multitask labels) are provided separately. This file exists
so that we can write a deterministic dataset-level test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import TrainRow
from .renderer import render_smiles_to_pil


class TinySmilesDataset(Dataset):
    def __init__(
        self,
        rows: list[TrainRow],
        image_size: int = 224,
        transform: Callable[..., Any] | None = None,
    ):
        if not rows:
            raise ValueError("TinySmilesDataset requires at least one row")
        self.rows = list(rows)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:  # ty: ignore[invalid-method-override]
        row = self.rows[idx]
        img = render_smiles_to_pil(row.smiles, size=self.image_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        # Normalize to [-1, 1] without ImageNet stats so we don't *implicitly*
        # encode any pretrained-model assumption.
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, row.smiles, row.image_id
