"""Tests for chemistry-safe augmentation pipeline."""

from PIL import Image

from glyph.ocsr.augment import (
    chemistry_safe_augment,
    gaussian_blur,
    jpeg_compress,
    res_downscale_upscale,
    salt_and_pepper,
    small_rotate,
)


def _white_image(size=(64, 64)):
    return Image.new("RGB", size, (255, 255, 255))


def test_each_aug_returns_pil():
    img = _white_image()
    for fn in [jpeg_compress, gaussian_blur, res_downscale_upscale, small_rotate, salt_and_pepper]:
        out = fn(img)
        assert isinstance(out, Image.Image)
        # Either the size is preserved or close to it (jpeg/blur preserve size,
        # rotate may add small fillcolor margins inside the same canvas).
        assert out.size == img.size


def test_chemistry_safe_handles_p_clean():
    img = _white_image()
    out = chemistry_safe_augment(img, p_clean=1.0)
    assert isinstance(out, Image.Image)


def test_chemistry_safe_handles_p_each_one():
    img = _white_image()
    out = chemistry_safe_augment(img, p_clean=0.0, p_each=1.0)
    assert isinstance(out, Image.Image)
