"""Chemistry-safe image augmentation pipeline.

The training-side augmentation recipe:

- JPEG-quality 18-70 (random subset)
- Gaussian blur sigma 0.3-1.8 (random subset)
- Resolution downscale + upscale 0.35-0.8x (random subset)
- +/-3 deg rotation
- Light salt-and-pepper noise

We deliberately skip:
- Large rotations / flips (would change chemistry)
- Heavy crops (would lose structure)
- Color shifts that confuse atom labels

Parameter ``p_clean`` controls the share of unaugmented images; the default is
~30% clean / 70% augmented.
"""

from __future__ import annotations

import io
import random
from typing import Any

from PIL import Image, ImageFilter


def jpeg_compress(img: Image.Image, quality_min: int = 18, quality_max: int = 70) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    q = random.randint(quality_min, quality_max)
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma_min: float = 0.3, sigma_max: float = 1.8) -> Image.Image:
    sigma = random.uniform(sigma_min, sigma_max)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def res_downscale_upscale(
    img: Image.Image, factor_min: float = 0.35, factor_max: float = 0.80
) -> Image.Image:
    w, h = img.size
    f = random.uniform(factor_min, factor_max)
    nw, nh = max(int(w * f), 16), max(int(h * f), 16)
    small = img.resize((nw, nh), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def small_rotate(img: Image.Image, deg: float = 3.0) -> Image.Image:
    angle = random.uniform(-deg, deg)
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))


def salt_and_pepper(img: Image.Image, prob: float = 0.005) -> Image.Image:
    """Cheap S&P noise without numpy dependency at this layer."""

    if random.random() > 0.5:
        return img
    px: Any = img.load()
    w, h = img.size
    n = int(w * h * prob)
    for _ in range(n):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        v = 0 if random.random() < 0.5 else 255
        px[x, y] = (v, v, v) if img.mode == "RGB" else v
    return img


_AUG_CHOICES = [
    ("jpeg", jpeg_compress),
    ("blur", gaussian_blur),
    ("res", res_downscale_upscale),
    ("rotate", small_rotate),
    ("snp", salt_and_pepper),
]


def chemistry_safe_augment(
    img: Image.Image,
    p_clean: float = 0.3,
    p_each: float = 0.4,
) -> Image.Image:
    """Apply a random subset of chemistry-safe augmentations.

    With probability ``p_clean`` returns the image unchanged. Otherwise each
    augmentation in the cocktail is applied independently with probability
    ``p_each``.
    """

    if random.random() < p_clean:
        return img
    cur = img
    for _name, fn in _AUG_CHOICES:
        if random.random() < p_each:
            cur = fn(cur)
    return cur
