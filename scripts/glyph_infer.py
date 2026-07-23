#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "glyph @ git+https://github.com/EdisonScientific/glyph.git",
#   "accelerate>=1.13",
#   "datasets>=4.8",
#   "huggingface-hub>=0.24",
#   "numpy>=1.26,<3.0",
#   "peft>=0.19",
#   "Pillow>=10.3",
#   "pydantic>=2.0",
#   "pyarrow>=15.0",
#   "PyYAML>=6.0",
#   "qwen-vl-utils",
#   "rdkit>=2024.3.1",
#   "sentencepiece>=0.2.0",
#   "SmilesPE>=0.0.3",
#   "tiktoken>=0.7",
#   "timm>=1.0",
#   "torch>=2.7",
#   "torchvision",
#   "tqdm>=4.66",
#   "transformers==5.6.2",
#   "typer>=0.20",
#   "wandb>=0.26",
# ]
# [tool.uv]
# index-strategy = "unsafe-best-match"
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
# ]
# torchvision = [
#   { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
# ]
# ///
"""Run a released Glyph model on one local or remote chemical-structure image.

Quick one-liners:
    uv run https://raw.githubusercontent.com/EdisonScientific/glyph/main/scripts/glyph_infer.py --model ocsr IMAGE
    uv run https://raw.githubusercontent.com/EdisonScientific/glyph/main/scripts/glyph_infer.py --model markush IMAGE
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CHECKPOINTS = {
    "ocsr": "EdisonScientific/OCSRGlyph",
    "markush": "EdisonScientific/MarkushGlyph",
}
MARKUSH_BASE_MODEL = "Qwen/Qwen3.5-2B-Base"
SUPPORTED_FORMATS = frozenset({"BMP", "GIF", "JPEG", "PNG", "TIFF", "WEBP"})
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 60_000_000
LOW_DPI = 72.0
LOW_SHORT_EDGE = 200
LOW_PIXEL_COUNT = 90_000
EXTREME_ASPECT_RATIO = 5.0
# A decode dominated by one repeated character for this many characters is a
# runaway/degenerate output (typically triggered by non-chemistry images).
DEGENERATE_MIN_LENGTH = 64
DEGENERATE_RUN_FRACTION = 0.9


class InputError(RuntimeError):
    """A clear, user-actionable input error."""


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _looks_degenerate(structure: str) -> bool:
    """Detect a decode that collapsed into a near-single-character run.

    An image with no chemical structure can push a model into a runaway decode
    that repeats one atom symbol up to the length limit. The check is
    deliberately conservative — it requires a long string dominated by a single
    repeated character — so genuine molecules never trip it.
    """

    core = structure.split(maxsplit=1)[0] if structure else ""
    if len(core) < DEGENERATE_MIN_LENGTH:
        return False
    longest = run = 1
    for previous, current in itertools.pairwise(core):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    return longest >= DEGENERATE_RUN_FRACTION * len(core)


def _download_image(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "glyph-infer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final_scheme = urllib.parse.urlparse(response.geturl()).scheme.lower()
            if final_scheme not in {"http", "https"}:
                raise InputError("the image URL redirected to an unsupported protocol")
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_DOWNLOAD_BYTES:
                        raise InputError("the remote image is larger than 50 MiB")
                except ValueError:
                    pass

            size = 0
            with destination.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise InputError("the remote image is larger than 50 MiB")
                    output.write(chunk)
    except InputError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise InputError(f"could not download image: {exc}") from exc

    if destination.stat().st_size == 0:
        raise InputError("the image URL returned an empty response")


def _materialize_input(source: str, directory: Path) -> Path:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme.lower() in {"http", "https"}:
        downloaded = directory / "downloaded-image"
        _download_image(source, downloaded)
        return downloaded
    if parsed.scheme:
        raise InputError("image must be a local path or an http(s) URL")

    path = Path(source).expanduser()
    if not path.exists():
        raise InputError(f"image does not exist: {path}")
    if not path.is_file():
        raise InputError(f"image is not a file: {path}")
    try:
        with path.open("rb"):
            pass
    except OSError as exc:
        raise InputError(f"image is not readable: {exc}") from exc
    return path


def _dpi_pair(value: Any) -> tuple[float, float] | None:
    try:
        if isinstance(value, int | float):
            pair = (float(value), float(value))
        else:
            pair = (float(value[0]), float(value[1]))
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(item) and item > 0 for item in pair):
        return None
    return pair


def _rgb_on_white(image: Image.Image) -> Image.Image:
    if "A" not in image.getbands():
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    return Image.alpha_composite(background, rgba).convert("RGB")


def _validate_and_normalize(source: Path, destination: Path) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                image_format = (image.format or "").upper()
                if image_format not in SUPPORTED_FORMATS:
                    supported = ", ".join(sorted(SUPPORTED_FORMATS))
                    found = image_format or "unknown"
                    raise InputError(
                        f"unsupported image format {found!r}; supported formats: {supported}"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise InputError("animated or multi-page images are not supported")

                dpi_value = image.info.get("dpi")
                dpi = _dpi_pair(dpi_value)
                if dpi_value is not None and dpi is None:
                    _warn("invalid DPI metadata was ignored")
                elif dpi is not None and min(dpi) < LOW_DPI:
                    _warn(
                        f"low DPI metadata ({dpi[0]:g} x {dpi[1]:g}); "
                        "fine labels and bonds may be missed"
                    )

                image.load()
                normalized = _rgb_on_white(ImageOps.exif_transpose(image))
    except InputError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InputError("image dimensions are too large to process safely") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InputError(f"unreadable image: {exc}") from exc

    width, height = normalized.size
    if width <= 0 or height <= 0:
        raise InputError("image has invalid dimensions")
    if width * height > MAX_IMAGE_PIXELS:
        raise InputError(
            f"image is too large ({width} x {height}); use at most {MAX_IMAGE_PIXELS:,} pixels"
        )

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > EXTREME_ASPECT_RATIO:
        _warn(
            f"extreme aspect ratio ({width}:{height}); crop excess whitespace around the structure"
        )
    if min(width, height) < LOW_SHORT_EDGE or width * height < LOW_PIXEL_COUNT:
        _warn(
            f"low resolution ({width} x {height}); use a sharper image for reliable bond and label OCR"
        )

    try:
        normalized.save(destination, format="PNG")
    except OSError as exc:
        raise InputError(f"could not normalize image: {exc}") from exc


def _configure_inference_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for logger_name in ("huggingface_hub", "peft", "transformers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _predict_ocsr(image: Path, checkpoint: str) -> str:
    from glyph.ocsr.predict import OCSRPredictor

    prediction = _one_line(OCSRPredictor(checkpoint=checkpoint).predict(image))
    if not prediction:
        raise RuntimeError("OCSRGlyph returned an empty prediction")
    if _looks_degenerate(prediction):
        _warn(
            "prediction looks degenerate; the image may not contain a readable chemical structure"
        )
    return prediction


def _cuda_is_usable() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        torch.empty(1, device="cuda")
    except RuntimeError:
        return False
    return True


def _format_substituents(stable: str) -> str:
    if not stable.strip():
        return "(none)"
    return _one_line(stable.replace("<ns>", "; ").replace("<n>", ", "))


def _predict_markush(image: Path, checkpoint: str) -> str:
    from glyph.markush.inference import MarkushPredictor

    use_cuda = _cuda_is_usable()
    predictor = MarkushPredictor(
        checkpoint=checkpoint,
        base_model=MARKUSH_BASE_MODEL,
        device="cuda:0" if use_cuda else "cpu",
        dtype="bfloat16" if use_cuda else "float32",
    )
    prediction = predictor.predict(image)
    cxsmiles = _one_line(prediction.cxsmiles)
    if not cxsmiles:
        raise RuntimeError("MarkushGlyph returned no valid CXSMILES")
    if _looks_degenerate(cxsmiles):
        _warn("prediction looks degenerate; the image may not contain a readable Markush structure")
    table = _format_substituents(prediction.stable)
    return f"CXSMILES={cxsmiles}; substituents={table}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OCSRGlyph (image to SMILES) or MarkushGlyph (image to CXSMILES).",
        epilog=(
            "Checkpoints download from Hugging Face on first use. Set HF_TOKEN if access "
            "requires authentication."
        ),
    )
    parser.add_argument("image", help="Local image path or http(s) image URL.")
    parser.add_argument(
        "--model",
        choices=tuple(DEFAULT_CHECKPOINTS),
        required=True,
        help="Released predictor to run: ocsr or markush.",
    )
    parser.add_argument(
        "--checkpoint",
        help=(
            "Local checkpoint path or Hugging Face repo id. Defaults to "
            "EdisonScientific/OCSRGlyph or EdisonScientific/MarkushGlyph for the chosen model."
        ),
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.model]
    _configure_inference_logging()

    with tempfile.TemporaryDirectory(prefix="glyph-infer-") as temporary:
        directory = Path(temporary)
        source = _materialize_input(args.image, directory)
        normalized = directory / "normalized.png"
        _validate_and_normalize(source, normalized)

        if args.model == "ocsr":
            result = _predict_ocsr(normalized, checkpoint)
        else:
            result = _predict_markush(normalized, checkpoint)

    print(f"result: {result}")
    return 0


def main() -> None:
    try:
        raise SystemExit(run())
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"error: inference failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
