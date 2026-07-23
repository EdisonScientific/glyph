"""Real-image loader for the cached HF USPTO OCSR benchmark.

The HF dataset stores each row's image as a dict with a ``bytes`` key holding
the raw PNG bytes. ``glyph.ocsr.eval.load_uspto_benchmark`` already flattens
that into ``image_bytes``. This module turns those bytes into model-ready
tensors, paired with the row's ground-truth SMILES and image_id.

The benchmark images are 1-bit / grayscale PNGs at variable resolutions
(observed: (826, 594) for the first row). We coerce them to 3-channel RGB,
resize to a square ``image_size`` with bilinear interpolation, and normalize
to the same [-1, 1] range used by ``TinySmilesDataset`` so that a single
random-init model can consume both pipelines interchangeably.

Constraints honored: no pretrained-stat normalization (no ImageNet mean/std).
No augmentation. Deterministic order.
"""

from __future__ import annotations

import io
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .eval import load_uspto_benchmark


@dataclass
class BenchmarkSample:
    image: torch.Tensor  # [3, H, W] in [-1, 1]
    smiles: str | None
    image_id: str
    raw_size: tuple[int, int]  # (W, H) of the original PNG


def _decode_png_bytes_to_rgb(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Empty image bytes")
    img = Image.open(io.BytesIO(image_bytes))
    img.load()  # force decode now to catch corrupt PNGs at the boundary
    return img.convert("RGB")


def _pil_to_chw_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """Resize ``img`` to ``size`` x ``size`` and return a [-1,1] CHW tensor."""

    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_benchmark_samples(
    parquet_path: Path | str | None = None,
    indices: Iterable[int] | None = None,
    n: int | None = None,
    shuffle_seed: int | None = None,
    image_size: int = 224,
    require_parseable_smiles: bool = True,
) -> list[BenchmarkSample]:
    """Load a small slice of real benchmark samples decoded into tensors.

    If ``indices`` is given, those exact rows are loaded (in order). Otherwise
    the first ``n`` rows (default: all rows; cap with ``n``) are scanned and
    returned, optionally filtered to rows that have a non-NaN SMILES.
    """

    rows = load_uspto_benchmark(parquet_path)
    if indices is not None:
        chosen = [rows[i] for i in indices]
    else:
        chosen = rows
        if require_parseable_smiles:
            chosen = [r for r in chosen if isinstance(r["smiles"], str) and r["smiles"]]
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            chosen = list(chosen)
            rng.shuffle(chosen)
        if n is not None:
            chosen = chosen[:n]

    out: list[BenchmarkSample] = []
    for r in chosen:
        image_bytes = r.get("image_bytes")
        if image_bytes is None:
            continue
        img = _decode_png_bytes_to_rgb(image_bytes)
        raw_size = img.size  # (W, H) before resize
        tensor = _pil_to_chw_tensor(img, size=image_size)
        out.append(
            BenchmarkSample(
                image=tensor,
                smiles=r.get("smiles") if isinstance(r.get("smiles"), str) else None,
                image_id=str(r["id"]),
                raw_size=raw_size,
            )
        )
    return out


def stack_batch(samples: list[BenchmarkSample]) -> dict[str, Any]:
    """Stack samples into a tensor batch + metadata lists."""

    if not samples:
        raise ValueError("Cannot stack an empty list of BenchmarkSamples")
    images = torch.stack([s.image for s in samples], dim=0)
    return {
        "images": images,
        "smiles": [s.smiles for s in samples],
        "image_ids": [s.image_id for s in samples],
        "raw_sizes": [s.raw_size for s in samples],
    }
