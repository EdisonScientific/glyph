"""Stereo-200K white-crop preprocessing (the ``stereo_200k_white`` step).

The ``stereo_200k_white`` dataset is *our preprocessing* of the EXTERNAL public
dataset ``Robert-zwr/Stereo-200k`` (cite MolSight, ``zhang2026molsight``). We do
not render or re-host the base images; the only contribution is this
white-background + tight-crop pass, after which the SMILES targets are
byte-identical to the source and only the image set differs.

The two image ops -- :func:`background_to_white` and :func:`smart_tight_crop`
-- reproduce the ``preprocess_canonical`` pipeline that produced the images
behind the published headline numbers.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Cap on the auto-selected worker count. The per-image cost is I/O + a couple of
# NumPy passes, so a handful of processes saturates a typical box without
# oversubscribing memory (each worker only ever holds one image at a time).
_WORKER_CAP = 8


def default_workers() -> int:
    """Sane default pool size: ``min(cpu_count, 8)`` (at least 1)."""

    return max(1, min(os.cpu_count() or 1, _WORKER_CAP))


def background_to_white(img: Image.Image, white_thresh: int = 220) -> Image.Image:
    """Detect a uniform-ish background and convert it to pure white.

    Strategy: take the median pixel of the four 8x8 corner patches as the
    presumed background colour. If that colour is "near-white" (all channels
    ``>= white_thresh``) the image is already white-bg and returned as-is.
    Otherwise any pixel within +/-25 of the background colour is set to white.
    """

    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img)
    corners = [arr[:8, :8], arr[:8, -8:], arr[-8:, :8], arr[-8:, -8:]]
    bg = np.stack([c.reshape(-1, 3) for c in corners], axis=0).reshape(-1, 3)
    bg_med = np.median(bg, axis=0).astype(np.int32)
    if (bg_med >= white_thresh).all():
        return img
    diff = np.abs(arr.astype(np.int32) - bg_med)
    mask = (diff <= 25).all(axis=-1)
    out = arr.copy()
    out[mask] = 255
    return Image.fromarray(out.astype(np.uint8))


def smart_tight_crop(img: Image.Image, pad_frac: float = 0.03) -> Image.Image:
    """Crop tightly around the largest non-white content cluster.

    The "largest cluster" approximation is conservative: threshold to a binary
    mask of any non-white pixel, then take the bounding box of all non-isolated
    pixels (drop pixels with fewer than two non-white neighbours in a 3x3
    window). This filters out tiny isolated text labels in corners. ``pad_frac``
    of the content bounding box (min 2px) is added back as a white margin.
    """

    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img)
    mask = (arr < 240).any(axis=-1).astype(np.uint8)
    pad = np.zeros_like(mask)
    pad[1:-1, 1:-1] = (
        mask[:-2, :-2]
        + mask[:-2, 1:-1]
        + mask[:-2, 2:]
        + mask[1:-1, :-2]
        + mask[1:-1, 2:]
        + mask[2:, :-2]
        + mask[2:, 1:-1]
        + mask[2:, 2:]
    )
    keep = mask & (pad >= 2)
    rows = np.where(keep.any(axis=1))[0]
    cols = np.where(keep.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]
    h = y1 - y0
    w = x1 - x0
    pad_h = max(int(h * pad_frac), 2)
    pad_w = max(int(w * pad_frac), 2)
    y0 = max(0, y0 - pad_h)
    y1 = min(arr.shape[0] - 1, y1 + pad_h)
    x0 = max(0, x0 - pad_w)
    x1 = min(arr.shape[1] - 1, x1 + pad_w)
    return img.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))


def white_crop_image(img: Image.Image) -> Image.Image:
    """Background-to-white then smart tight crop. The full ``_white`` pass."""

    if img.mode != "RGB":
        img = img.convert("RGB")
    return smart_tight_crop(background_to_white(img))


def _white_crop_one(task: tuple[str, str]) -> bool:
    """White-crop ``source`` and save to ``dest``; return success.

    Runs in a worker process, so it owns its own error handling: a single
    corrupt/unreadable image returns ``False`` instead of raising, keeping one
    bad frame from aborting the whole pool. Output bytes are independent of
    which worker runs the task, so results are identical to the serial path.
    """

    source, dest = task
    try:
        with Image.open(source) as image:
            cropped = white_crop_image(image)
        cropped.save(dest)
        return True
    except Exception:
        return False


def _iter_white_crop(tasks: list[tuple[str, str]], workers: int) -> Iterator[bool]:
    """Yield per-task success flags, in input order, serial or via a pool.

    ``workers <= 1`` runs the exact serial path (no pool). Otherwise a
    :class:`~concurrent.futures.ProcessPoolExecutor` maps the work in order;
    ``chunksize`` batches dispatch so IPC stays cheap while each worker still
    only holds one image at a time (memory stays bounded).
    """

    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield _white_crop_one(task)
        return
    chunksize = max(1, min(16, len(tasks) // (workers * 4) or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_white_crop_one, tasks, chunksize=chunksize)


@dataclass
class WhiteCropResult:
    """Summary of a :func:`white_crop_stereo_split` run."""

    total_rows: int
    processed: int
    written: int
    skipped_existing: int
    missing_images: int
    failed: int
    out_csv: Path
    out_image_root: Path


def _resolve_source_image(file_path: str, image_root: Path) -> Path:
    """Resolve a CSV ``file_path`` against the extracted source image root.

    Source CSV rows use paths like ``stereo/train/0.png``. The extracted root
    may point at either the archive root (so ``stereo/train/0.png`` resolves) or
    directly at ``.../stereo`` (so ``train/0.png`` resolves). Try both.
    """

    candidate = image_root / file_path
    if candidate.exists():
        return candidate
    parts = Path(file_path).parts
    if len(parts) > 1:
        return image_root / Path(*parts[1:])
    return candidate


def white_crop_stereo_split(
    src_csv: Path | str,
    src_image_root: Path | str,
    out_dir: Path | str,
    *,
    split: str = "train",
    limit: int | None = None,
    force: bool = False,
    smiles_col: str = "SMILES",
    file_path_col: str = "file_path",
    workers: int | None = None,
) -> WhiteCropResult:
    """White-crop every image in one Stereo-200K split into ``out_dir``.

    Reads ``src_csv`` (schema ``SMILES,file_path``), white-crops each referenced
    image into ``out_dir/<split>/<name>.png``, and writes a rewritten CSV whose
    ``file_path`` points at ``<split>/<name>.png`` (matching the frozen
    ``stereo_200k_white`` layout). SMILES targets are copied unchanged.

    Idempotent: images already present are skipped unless ``force`` is set.

    ``workers`` sets the process-pool size for the image ops. ``None`` picks
    :func:`default_workers` (``min(cpu_count, 8)``); ``1`` runs a plain serial
    loop. The image bytes and the rewritten CSV (row order and content) are
    identical regardless of the worker count -- the row bookkeeping runs serially
    in input order and only the independent per-image crop/save is parallelized.
    """

    src_csv = Path(src_csv).expanduser()
    src_image_root = Path(src_image_root).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_image_dir = out_dir / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{split}.csv"
    workers = default_workers() if workers is None else max(1, workers)

    with src_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    result = WhiteCropResult(
        total_rows=len(rows),
        processed=0,
        written=0,
        skipped_existing=0,
        missing_images=0,
        failed=0,
        out_csv=out_csv,
        out_image_root=out_dir,
    )

    # Serial, cheap bookkeeping pass (input order): resolve the rewritten CSV
    # rows and the skip/missing counters exactly as the serial path did, and
    # collect the (source, dest) pairs that still need the expensive crop.
    out_rows: list[dict[str, str]] = []
    tasks: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        if limit is not None and result.processed >= limit:
            break
        result.processed += 1
        smiles = (row.get(smiles_col) or "").strip()
        file_path = (row.get(file_path_col) or "").strip()
        if not file_path:
            result.missing_images += 1
            continue
        name = Path(file_path).name or f"{index}.png"
        rel_path = f"{split}/{name}"
        dest = out_image_dir / name
        out_rows.append({smiles_col: smiles, file_path_col: rel_path})

        if dest.exists() and not force:
            result.skipped_existing += 1
            continue
        source = _resolve_source_image(file_path, src_image_root)
        if not source.exists():
            result.missing_images += 1
            continue
        tasks.append((str(source), str(dest)))

    # Expensive crop/save pass, parallelized when workers > 1. Order does not
    # affect the counts, and a corrupt image yields False (counted as failed)
    # without aborting the pool.
    for ok in _iter_white_crop(tasks, workers):
        if ok:
            result.written += 1
        else:
            result.failed += 1

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[smiles_col, file_path_col])
        writer.writeheader()
        writer.writerows(out_rows)

    return result
