#!/usr/bin/env python3
"""Download public Markush/OCSR datasets and convert them to the unified schema.

The script is intentionally split into small pure helpers so the schema mapping
can be tested without network access.  Running the CLI requires HuggingFace
``datasets`` and network/auth access for the selected repos.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import zipfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from PIL import Image

from glyph.paths import data_root

LOGGER = logging.getLogger(__name__)

# Cap on the auto-selected worker count for the image-materialization pass and how
# many input rows the parallel path buffers per pool dispatch. Mirrors the OCSR
# ``white_crop``/``adjacent_ring`` parallel pattern (``glyph/data/ocsr_*.py``): a
# handful of processes saturates the PIL decode+re-encode without oversubscribing
# memory, and bounded chunks keep peak memory ~one chunk of decoded images.
_WORKER_CAP = 8
_IMAGE_CHUNK_ROWS = 512


def default_workers() -> int:
    """Sane default pool size: ``min(cpu_count, 8)`` (at least 1)."""

    return max(1, min(os.cpu_count() or 1, _WORKER_CAP))


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata needed to load and normalize one upstream dataset config."""

    name: str
    repo: str
    config: str | None
    source: str
    task: str
    default_image_has_text: bool
    eval_only: bool = False
    allowed_splits: tuple[str, ...] | None = None
    kind: str = "dataset"
    archive_filename: str | None = None
    repo_type: str = "dataset"
    revision: str | None = None
    fallback_id_prefix: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mg1-synthetic": DatasetSpec(
        name="mg1-synthetic",
        repo="docling-project/MarkushGrapher-Datasets",
        config="markushgrapher-synthetic-training",
        source="mg1_synthetic",
        task="markush",
        default_image_has_text=True,
    ),
    "uspto-mol-m": DatasetSpec(
        name="uspto-mol-m",
        repo="docling-project/MarkushGrapher-2-Datasets",
        config="uspto-mol-m-54k",
        source="uspto_mol_m",
        task="markush",
        default_image_has_text=False,
    ),
    "ip5m": DatasetSpec(
        name="ip5m",
        repo="docling-project/MarkushGrapher-2-Datasets",
        config="ip5-markush",
        source="ip5m",
        task="markush",
        default_image_has_text=False,
        eval_only=True,
        allowed_splits=("test",),
    ),
    "m2s": DatasetSpec(
        name="m2s",
        repo="docling-project/MarkushGrapher-2-Datasets",
        config="m2s",
        source="m2s",
        task="markush",
        default_image_has_text=True,
        eval_only=True,
        allowed_splits=("test",),
    ),
    "uspto-markush": DatasetSpec(
        name="uspto-markush",
        repo="docling-project/MarkushGrapher-2-Datasets",
        config="uspto-markush",
        source="uspto_markush",
        task="markush",
        default_image_has_text=False,
        eval_only=True,
        allowed_splits=("test",),
    ),
    "molparser-markush": DatasetSpec(
        name="molparser-markush",
        repo="UniParser/MolParser-7M",
        config="sft_real",
        source="molparser",
        task="markush",
        default_image_has_text=False,
        fallback_id_prefix="molparser-markush",
    ),
    "molscribe-ocsr": DatasetSpec(
        name="molscribe-ocsr",
        repo="yujieq/MolScribe",
        config=None,
        source="molscribe",
        task="ocsr",
        default_image_has_text=False,
        kind="archive",
        archive_filename="uspto_mol.zip",
        repo_type="model",
    ),
}


IMAGE_FIELDS = ("image", "page_image", "structure_image", "png", "img")
ID_FIELDS = ("id", "sample_id", "source_id", "uuid", "file_name", "filename")
SMILES_FIELDS = ("smiles", "SMILES", "canonical_smiles", "label")
MOLSCRIBE_IMAGE_FIELDS = ("file_path", "image", "image_path", "path", "filename")
MOLSCRIBE_SMILES_FIELDS = ("raw_SMILES", *SMILES_FIELDS)


def safe_sample_id(value: Any, fallback: str) -> str:
    """Return a stable filesystem-safe sample id."""

    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        text = fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def parse_patent_id(sample_id: str) -> tuple[str | None, str | None]:
    """Best-effort patent id and patent-office extraction from dataset ids."""

    text = sample_id.strip()
    patterns = [
        (r"(US[0-9A-Z-]+)", "USPTO"),
        (r"(EP[0-9A-Z]+)", "EPO"),
        (r"(KR[0-9A-Z]+|10[0-9]{8,})", "KIPO"),
        (r"(CN[0-9A-Z]+)", "CNIPA"),
        (r"(JP[0-9A-Z]+)", "JPO"),
    ]
    for pattern, office in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper(), office
    return None, None


def first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    """Return the first non-empty field value from ``row``."""

    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return None


def first_present_with_field(
    row: dict[str, Any], fields: tuple[str, ...]
) -> tuple[str | None, Any]:
    """Return the first non-empty ``(field, value)`` pair from ``row``."""

    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return field, value
    return None, None


def _jsonable(value: Any) -> Any:
    """Convert common dataset values to JSON-serializable objects."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return str(value)


def save_image(value: Any, path: Path) -> str | None:
    """Save a HuggingFace/PIL image value to ``path`` and return the path.

    A value that cannot be decoded/materialized (corrupt bytes, an unreadable
    source file, a truncated PNG) returns ``None`` rather than raising, so a
    single bad frame never aborts a whole import. This matches the OCSR
    white-crop contract (``glyph/data/ocsr_white_crop.py``) where a corrupt image
    is counted as skipped, and keeps the serial and parallel materialization
    paths byte-for-byte identical on the same input. Bytes for a *valid* image
    are unchanged (deterministic ``Image.open(...).convert("RGB").save(png)``).
    """

    if value is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(value, Image.Image):
            value.convert("RGB").save(path)
            return str(path)

        if isinstance(value, dict):
            if value.get("path"):
                src = Path(value["path"]).expanduser()
                if src.exists():
                    with Image.open(src) as image:
                        image.convert("RGB").save(path)
                    return str(path)
            if value.get("bytes"):
                from io import BytesIO

                with Image.open(BytesIO(value["bytes"])) as image:
                    image.convert("RGB").save(path)
                return str(path)

        if isinstance(value, str):
            src = Path(value).expanduser()
            if src.exists():
                with Image.open(src) as image:
                    image.convert("RGB").save(path)
                return str(path)
            return value
    except (OSError, ValueError):
        # Undecodable/corrupt image value: keep going with no materialized image
        # (identical outcome in the serial and parallel loops).
        return None

    return None


def unified_sample_id(
    row: dict[str, Any], *, spec: DatasetSpec, split: str, index: int
) -> tuple[str | None, str]:
    """Return the ``(id_field, sample_id)`` that :func:`normalize_row` assigns.

    The ``sample_id`` is the source-id-keyed identity the unified row (and the
    frozen flat training index, whose ``source_id`` mirrors this row's ``id``)
    uses to name the materialized image (``images/<source>/<split>/<sample_id>.png``).
    Exposed so a caller can decide whether a row is index-referenced *before*
    paying to decode/re-encode its image, without duplicating the id logic.
    """

    id_field, id_value = first_present_with_field(row, ID_FIELDS)
    fallback_prefix = spec.fallback_id_prefix or spec.name
    sample_id = safe_sample_id(id_value, f"{fallback_prefix}_{split}_{index:06d}")
    return id_field, sample_id


def normalize_row(
    row: dict[str, Any],
    *,
    spec: DatasetSpec,
    split: str,
    index: int,
    images_dir: Path | None = None,
) -> dict[str, Any]:
    """Map one upstream row into ``UNIFIED_DATASET_FORMAT.md`` fields."""

    id_field, sample_id = unified_sample_id(row, spec=spec, split=split, index=index)
    patent_id, patent_office = parse_patent_id(sample_id)

    image_field, image_value = first_present_with_field(row, IMAGE_FIELDS)
    image_path = None
    if images_dir is not None:
        # A source id names one source image, independent of row order. Keep all
        # Markush training images under the source key used by the flat index.
        filename = f"{sample_id}.png"
        target = images_dir / spec.source / split / filename
        saved = save_image(image_value, target)
        if saved == str(target):
            image_path = (Path("images") / spec.source / split / filename).as_posix()
        elif isinstance(saved, str):
            # External source that could not be re-materialized: keep as-is.
            image_path = saved
    elif isinstance(image_value, str):
        image_path = image_value

    cxsmiles_opt = row.get("cxsmiles_opt") or row.get("cxsmiles_optimized")
    cxsmiles_standard = row.get("cxsmiles") or row.get("cxsmiles_standard")
    cxsmiles_dataset = row.get("cxsmiles_dataset")
    annotation = row.get("annotation")
    smiles_field, smiles = first_present_with_field(row, SMILES_FIELDS)
    molparser_esmiles = smiles if spec.name == "molparser-markush" else row.get("molparser_esmiles")
    if spec.task == "markush" and smiles == cxsmiles_standard:
        smiles = None

    return {
        "id": sample_id,
        "source": spec.source,
        "split": split,
        "image": image_path,
        "image_has_text": bool(row.get("image_has_text", spec.default_image_has_text)),
        "task": spec.task,
        "cxsmiles_opt": cxsmiles_opt,
        "smiles": smiles,
        "molparser_esmiles": molparser_esmiles,
        "annotation": annotation,
        "cxsmiles_dataset": cxsmiles_dataset,
        "cxsmiles_standard": cxsmiles_standard,
        "description": row.get("description"),
        "cells": _jsonable(row.get("cells")),
        "patent_id": row.get("patent_id") or patent_id,
        "patent_office": row.get("patent_office") or patent_office,
        "metadata": {
            "dataset": spec.name,
            "repo": spec.repo,
            "config": spec.config,
            "eval_only": spec.eval_only,
            "upstream_id": sample_id,
            "row_index": index,
            "source_split": split,
            "source_row_index": index,
            "source_id_field": id_field,
            "source_image_field": image_field,
            "source_smiles_field": smiles_field,
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_splits(dataset: Any) -> list[tuple[str, Any]]:
    """Return ``(split_name, split_dataset)`` pairs from HF Dataset/Dict values."""

    if hasattr(dataset, "items"):
        return list(dataset.items())
    split = getattr(dataset, "split", None)
    return [(str(split) if split else "train", dataset)]


def split_name_from_expression(value: str | None) -> str | None:
    """Return the base split name from an HF split expression."""

    if not value:
        return None
    return value.split("[", 1)[0].strip() or None


def convert_loaded_dataset(
    dataset: Any,
    *,
    spec: DatasetSpec,
    output_dir: Path,
    limit: int | None = None,
    forced_split_name: str | None = None,
    save_images: bool = True,
    workers: int | None = None,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Convert an already-loaded HF dataset object and write local artifacts.

    ``workers`` sets the process-pool size for the per-row image
    decode/re-encode when materializing images. ``None`` picks
    :func:`default_workers` (``min(cpu_count, 8)``); ``1`` runs the serial loop.
    The materialized PNG bytes and the emitted JSONL (row order and content) are
    byte-for-byte identical regardless of the worker count.

    ``keep_ids`` (when not ``None``) restricts materialization to rows whose
    :func:`unified_sample_id` is in the set -- the ``source_id`` values the frozen
    flat training index references for this source. Rows outside the set are
    skipped before their image is decoded, so a full source pull materializes only
    the index-referenced images instead of the entire upstream tree. ``None`` keeps
    the full-source behavior. The bytes/JSONL of kept rows are unchanged.
    """

    converted_by_split: dict[str, list[dict[str, Any]]] = {}
    for detected_split, split_dataset in _iter_splits(dataset):
        split = forced_split_name or split_name_from_expression(detected_split) or detected_split
        if spec.allowed_splits and split not in spec.allowed_splits:
            continue
        if not save_images and hasattr(split_dataset, "column_names"):
            image_columns = [field for field in IMAGE_FIELDS if field in split_dataset.column_names]
            if image_columns:
                split_dataset = split_dataset.remove_columns(image_columns)
        if limit is not None and hasattr(split_dataset, "select"):
            split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))
            split_limit = None
        else:
            split_limit = limit
        rows = convert_split_dataset_rows(
            split_dataset,
            spec=spec,
            split=split,
            # Materialize images under GLYPH_DATA_ROOT/images so the exported
            # relative paths resolve at train time; JSONL/manifest stay in output_dir.
            images_dir=(data_root() / "images") if save_images else None,
            limit=split_limit,
            workers=workers,
            keep_ids=keep_ids,
        )
        converted_by_split[split] = rows
        write_jsonl(rows, output_dir / spec.name / f"{split}.jsonl")

    all_rows = [row for rows in converted_by_split.values() for row in rows]
    write_jsonl(all_rows, output_dir / spec.name / "unified.jsonl")
    manifest = {
        "spec": asdict(spec),
        "splits": {split: len(rows) for split, rows in converted_by_split.items()},
        "total": len(all_rows),
        "unified_jsonl": str(output_dir / spec.name / "unified.jsonl"),
    }
    _record_keep_coverage(manifest, spec=spec, rows=all_rows, keep_ids=keep_ids, limit=limit)
    (output_dir / spec.name / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _record_keep_coverage(
    manifest: dict[str, Any],
    *,
    spec: DatasetSpec,
    rows: list[dict[str, Any]],
    keep_ids: set[str] | None,
    limit: int | None,
) -> None:
    """Annotate ``manifest`` with keep-set coverage and warn on missing ids.

    Only meaningful on a full pull (``limit is None``); a limited/sliced run scans
    a prefix of the source, so absent ids are expected and not reported. A missing
    id is an index-referenced ``source_id`` with no matching upstream row: surfaced
    (warning + count) rather than silently dropped.
    """

    if keep_ids is None or limit is not None:
        return
    matched = {str(row.get("id")) for row in rows} & keep_ids
    missing = sorted(keep_ids - matched)
    manifest["keep_requested"] = len(keep_ids)
    manifest["keep_matched"] = len(matched)
    manifest["keep_missing"] = len(missing)
    if missing:
        preview = ", ".join(missing[:5])
        LOGGER.warning(
            "%s: %d of %d index-referenced source_ids not found in the upstream "
            "source (first missing: %s)",
            spec.name,
            len(missing),
            len(keep_ids),
            preview,
        )


def _normalize_public_row_for_map(
    row: dict[str, Any],
    index: int,
    *,
    spec_payload: dict[str, Any],
    split: str,
    images_dir: str | None,
) -> dict[str, Any]:
    """Pickleable map worker for public dataset normalization."""

    return normalize_row(
        dict(row),
        spec=DatasetSpec(**spec_payload),
        split=split,
        index=index,
        images_dir=Path(images_dir) if images_dir else None,
    )


def _convert_public_row(
    task: tuple[int, dict[str, Any]],
    *,
    spec_payload: dict[str, Any],
    split: str,
    images_dir: str | None,
) -> tuple[int, dict[str, Any]]:
    """Pickleable pool worker: materialize one row's image and normalize it.

    Runs in a worker process. It decodes+re-encodes the row's image value via
    :func:`save_image` (a corrupt frame yields ``image=None`` instead of raising,
    keeping one bad image from aborting the pool) and returns the row's original
    ``index`` alongside the normalized record so the caller can reassemble the
    split in input order. The output is independent of which worker runs the
    task, so results are byte-identical to the serial loop.
    """

    index, row = task
    normalized = normalize_row(
        dict(row),
        spec=DatasetSpec(**spec_payload),
        split=split,
        index=index,
        images_dir=Path(images_dir) if images_dir else None,
    )
    return index, normalized


def _indexed_rows(split_dataset: Any, limit: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(index, row_dict)`` pairs in input order, honoring ``limit``.

    Matches the serial loop's iteration/limit contract exactly so the parallel
    path scans the same rows in the same order.
    """

    for index, row in enumerate(split_dataset):
        if limit is not None and index >= limit:
            break
        yield index, dict(row)


def _kept_indexed_rows(
    split_dataset: Any,
    *,
    spec: DatasetSpec,
    split: str,
    limit: int | None,
    keep_ids: set[str] | None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield the ``(index, row)`` pairs to materialize, filtered by ``keep_ids``.

    The original enumeration ``index`` is preserved for every kept row, so the
    ``sample_id`` (and hence the image path and JSONL) a kept row receives is
    identical to the unfiltered pass -- only never-referenced rows are dropped.
    ``keep_ids is None`` yields every row (full-source behavior).
    """

    for index, row in _indexed_rows(split_dataset, limit):
        if keep_ids is not None:
            _, sample_id = unified_sample_id(row, spec=spec, split=split, index=index)
            if sample_id not in keep_ids:
                continue
        yield index, row


def _chunked(
    items: Iterator[tuple[int, dict[str, Any]]], size: int
) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    """Batch an iterator into lists of at most ``size`` items."""

    chunk: list[tuple[int, dict[str, Any]]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _convert_split_parallel(
    split_dataset: Any,
    *,
    spec: DatasetSpec,
    split: str,
    images_dir: Path,
    limit: int | None,
    workers: int,
    keep_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Materialize + normalize one split across a process pool, in input order.

    This is the direct :class:`~concurrent.futures.ProcessPoolExecutor` path that
    parallelizes the per-row image decode/re-encode, *bypassing* HF ``.map`` (whose
    ``num_proc`` re-encodes a returned image-path column back into an Image feature,
    leaking a PIL object into JSON). Rows are read serially in input order,
    dispatched in bounded chunks (peak memory ~one chunk of decoded images), and
    reassembled by their original index, so the PNG bytes and the JSONL row
    order/content are byte-identical to the serial loop.

    ``keep_ids`` (when not ``None``) drops never-referenced rows in the reader
    process before they are dispatched to the pool, so a skipped row's image is
    never decoded -- the kept rows are byte-identical to the serial keep-set loop.
    """

    worker = partial(
        _convert_public_row,
        spec_payload=asdict(spec),
        split=split,
        images_dir=str(images_dir),
    )
    by_index: dict[int, dict[str, Any]] = {}
    indexed = _kept_indexed_rows(
        split_dataset, spec=spec, split=split, limit=limit, keep_ids=keep_ids
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk in _chunked(indexed, _IMAGE_CHUNK_ROWS):
            chunksize = max(1, min(16, len(chunk) // (workers * 4) or 1))
            for index, normalized in pool.map(worker, chunk, chunksize=chunksize):
                by_index[index] = normalized
    return [by_index[index] for index in sorted(by_index)]


def convert_split_dataset_rows(
    split_dataset: Any,
    *,
    spec: DatasetSpec,
    split: str,
    images_dir: Path | None = None,
    limit: int | None = None,
    workers: int | None = None,
    keep_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert one HF split into unified rows, optionally in parallel.

    ``workers`` sets the process-pool size for the image decode/re-encode.
    ``None`` picks :func:`default_workers` (``min(cpu_count, 8)``); ``1`` runs
    the serial loop (today's behavior). Output is byte-identical regardless of
    the worker count.

    ``keep_ids`` (when not ``None``) restricts materialization to rows whose
    :func:`unified_sample_id` is in the set; never-referenced rows are skipped
    before their image is decoded. The serial and parallel paths apply the same
    filter, so kept rows stay byte-identical.
    """

    workers = default_workers() if workers is None else max(1, workers)

    # Direct process-pool path for the per-row decode->save when materializing
    # images. It bypasses HF .map() entirely and reassembles rows in input order,
    # so it is byte-identical to serial -- and, unlike the .map() branch below,
    # it is safe with image materialization, so it is NOT forced back to serial.
    if workers > 1 and images_dir is not None and _split_row_count(split_dataset, limit) != 1:
        return _convert_split_parallel(
            split_dataset,
            spec=spec,
            split=split,
            images_dir=images_dir,
            limit=limit,
            workers=workers,
            keep_ids=keep_ids,
        )

    # The HF .map() multiprocessing path re-encodes a returned "image" path back
    # into an Image feature, leaking a PIL object into JSON serialization
    # (observed on MolParser-7M sft_real). It is only safe WITHOUT image
    # materialization; we reach it solely when images_dir is None, so the leak
    # cannot occur. (The image-materialization case is handled above.) The
    # keep-set filter is an image-materialization concern, so this metadata-only
    # path never receives one.
    if (
        workers > 1
        and images_dir is None
        and keep_ids is None
        and hasattr(split_dataset, "map")
        and hasattr(split_dataset, "column_names")
    ):
        mapped = split_dataset.map(
            _normalize_public_row_for_map,
            with_indices=True,
            fn_kwargs={
                "spec_payload": asdict(spec),
                "split": split,
                "images_dir": None,
            },
            remove_columns=list(split_dataset.column_names),
            num_proc=workers,
            desc=f"Converting {spec.name}/{split}",
            load_from_cache_file=False,
        )
        return [dict(row) for row in mapped]

    rows: list[dict[str, Any]] = []
    for index, row in _kept_indexed_rows(
        split_dataset, spec=spec, split=split, limit=limit, keep_ids=keep_ids
    ):
        rows.append(
            normalize_row(
                row,
                spec=spec,
                split=split,
                index=index,
                images_dir=images_dir,
            )
        )
    return rows


def _split_row_count(split_dataset: Any, limit: int | None) -> int | None:
    """Best-effort effective row count (capped by ``limit``); ``None`` if unknown.

    Used only to skip spinning up a pool for a trivial (<=1 row) split; a
    streaming dataset with no ``__len__`` returns ``None`` and still parallelizes.
    """

    try:
        total = len(split_dataset)
    except TypeError:
        return None
    return min(total, limit) if limit is not None else total


def save_unified_dataset(rows: list[dict[str, Any]], path: Path) -> None:
    """Save converted rows as a local HuggingFace DatasetDict."""

    from datasets import Dataset, DatasetDict

    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row.get("split") or "train"), []).append(row)
    dataset_dict = DatasetDict(
        {split: Dataset.from_list(items) for split, items in by_split.items()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(path))


def split_from_csv_path(path: Path) -> str:
    """Infer a unified split name from a MolScribe CSV filename."""

    stem = path.stem.lower()
    if stem.startswith("train"):
        return "train"
    if stem.startswith("valid") or stem.startswith("val"):
        return "validation"
    if stem.startswith("test"):
        return "test"
    return "train"


def _molscribe_train_csv_size(path: Path) -> int | None:
    match = re.fullmatch(r"train_(\d+)k", path.stem.lower())
    if not match:
        return None
    return int(match.group(1))


def selected_molscribe_csv_paths(csv_paths: list[Path]) -> list[Path]:
    """Select non-overlapping MolScribe CSVs, preferring the largest train split."""

    by_split: dict[str, list[Path]] = {}
    for path in sorted(csv_paths):
        by_split.setdefault(split_from_csv_path(path), []).append(path)

    selected: list[Path] = []
    for split, paths in sorted(by_split.items()):
        if split == "train":
            ranked = [
                (size, path)
                for path in paths
                if (size := _molscribe_train_csv_size(path)) is not None
            ]
            if ranked:
                selected.append(max(ranked)[1])
                continue
        selected.extend(paths)
    return sorted(selected)


def _safe_zip_member_path(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or any(part in {"", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path


def _extract_zip_member_if_needed(
    archive: zipfile.ZipFile,
    member: str,
    extract_root: Path,
) -> bool:
    info = archive.getinfo(member)
    target = extract_root / _safe_zip_member_path(member)
    if target.exists() and target.stat().st_size == info.file_size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return True


def extract_molscribe_archive_inputs(
    archive_path: Path,
    extract_root: Path,
    *,
    limit: int | None = None,
    spec: DatasetSpec | None = None,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Extract only MolScribe CSVs and referenced images needed for conversion.

    ``keep_ids`` (when not ``None``) restricts image extraction to rows whose
    eventual ``sample_id`` is in the set (the ``source_id`` values the frozen flat
    training index references), so a full pull unzips only the index-referenced
    images instead of the whole ~680K image tree. All selected CSVs are still
    extracted (the conversion pass reads them). Requires ``spec`` for the id logic;
    ``None`` keeps the full-source behavior.
    """

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = selected_molscribe_csv_paths(
            [Path(name) for name in archive.namelist() if name.lower().endswith(".csv")]
        )
        selected_csv_names = [path.as_posix() for path in csv_names]
        image_names: set[str] = set()
        for csv_name in selected_csv_names:
            split = split_from_csv_path(Path(csv_name))
            with archive.open(csv_name) as binary_handle:
                reader = csv.DictReader(TextIOWrapper(binary_handle, encoding="utf-8", newline=""))
                for index, row in enumerate(reader):
                    if limit is not None and index >= limit:
                        break
                    image_value = first_present(row, MOLSCRIBE_IMAGE_FIELDS)
                    if not image_value:
                        continue
                    if keep_ids is not None and spec is not None:
                        sample_id = _molscribe_eventual_sample_id(
                            dict(row), spec=spec, split=split, index=index
                        )
                        if sample_id not in keep_ids:
                            continue
                    image_names.add(str(image_value).strip())

        missing_images = sorted(name for name in image_names if name not in archive.NameToInfo)
        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"{len(missing_images)} MolScribe image files referenced by CSV are "
                f"missing from {archive_path}; first missing: {preview}"
            )

        extracted = 0
        reused = 0
        for member in [*selected_csv_names, *sorted(image_names)]:
            if _extract_zip_member_if_needed(archive, member, extract_root):
                extracted += 1
            else:
                reused += 1

    return {
        "archive_path": str(archive_path),
        "selected_csvs": selected_csv_names,
        "referenced_images": len(image_names),
        "extracted_members": extracted,
        "reused_members": reused,
    }


def resolve_molscribe_image_path(value: Any, *, csv_path: Path, extract_root: Path) -> Path | None:
    """Resolve a MolScribe CSV image path against common archive layouts."""

    text = "" if value is None else str(value).strip()
    if not text:
        return None
    raw_path = Path(text).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([extract_root / raw_path, csv_path.parent / raw_path])
        parts = raw_path.parts
        if parts and parts[0] == csv_path.parent.name:
            candidates.append(csv_path.parent.parent / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _molscribe_sample_id(
    row: dict[str, Any],
    *,
    spec: DatasetSpec,
    split: str,
    index: int,
    source_image: Path | None,
    image_value: Any,
) -> str:
    """Return the ``sample_id`` :func:`normalize_molscribe_csv_row` assigns.

    Shared by the row normalizer and the keep-set filters so the id logic lives in
    one place. Prefers an explicit ``image_id``/``id`` column, then the resolved
    image stem, then the raw image value, falling back to a split+index token.
    """

    fallback_id = f"{spec.name}_{split}_{index:06d}"
    return safe_sample_id(
        row.get("image_id")
        or row.get("id")
        or (source_image.stem if source_image is not None else image_value),
        fallback_id,
    )


def _molscribe_eventual_sample_id(
    row: dict[str, Any], *, spec: DatasetSpec, split: str, index: int
) -> str:
    """Return the ``sample_id`` a row will get once its image is extracted.

    Used to filter extraction, before any archive member exists on disk. The
    conversion pass resolves the image against ``extract_root``; every resolution
    candidate shares the CSV image value's filename, so the eventual ``stem`` is
    ``Path(image_value).stem``. Passing that as ``source_image`` makes this id
    equal the one :func:`normalize_molscribe_csv_row` computes post-extraction, so
    a kept image is always extracted and never falsely skipped.
    """

    image_value = first_present(row, MOLSCRIBE_IMAGE_FIELDS)
    resolved = Path(str(image_value).strip()) if image_value else None
    return _molscribe_sample_id(
        row,
        spec=spec,
        split=split,
        index=index,
        source_image=resolved,
        image_value=image_value,
    )


def normalize_molscribe_csv_row(
    row: dict[str, Any],
    *,
    spec: DatasetSpec,
    split: str,
    index: int,
    csv_path: Path,
    extract_root: Path,
    images_dir: Path | None = None,
) -> dict[str, Any]:
    """Map one MolScribe OCSR CSV row into the unified schema."""

    image_field, image_value = first_present_with_field(row, MOLSCRIBE_IMAGE_FIELDS)
    source_image = resolve_molscribe_image_path(
        image_value,
        csv_path=csv_path,
        extract_root=extract_root,
    )
    sample_id = _molscribe_sample_id(
        row,
        spec=spec,
        split=split,
        index=index,
        source_image=source_image,
        image_value=image_value,
    )
    image_path = None
    if images_dir is not None and source_image is not None:
        filename = f"{sample_id}.png"
        target = images_dir / spec.source / split / filename
        saved = save_image(str(source_image), target)
        if saved == str(target):
            image_path = (Path("images") / spec.source / split / filename).as_posix()
        elif isinstance(saved, str):
            image_path = saved
    elif source_image is not None:
        image_path = str(source_image)
    elif isinstance(image_value, str):
        image_path = image_value

    patent_id, patent_office = parse_patent_id(sample_id)
    smiles_field, smiles = first_present_with_field(row, MOLSCRIBE_SMILES_FIELDS)
    return {
        "id": sample_id,
        "source": spec.source,
        "split": split,
        "image": image_path,
        "image_has_text": False,
        "task": "ocsr",
        "cxsmiles_opt": None,
        "smiles": str(smiles) if smiles else None,
        "annotation": None,
        "cxsmiles_dataset": None,
        "cxsmiles_standard": None,
        "description": None,
        "cells": None,
        "patent_id": row.get("patent_id") or patent_id,
        "patent_office": row.get("patent_office") or patent_office,
        "metadata": {
            "dataset": spec.name,
            "repo": spec.repo,
            "config": spec.config,
            "archive_filename": spec.archive_filename,
            "eval_only": spec.eval_only,
            "upstream_id": sample_id,
            "row_index": index,
            "source_split": split,
            "source_row_index": index,
            "source_image_field": image_field,
            "source_smiles_field": smiles_field,
        },
    }


def convert_molscribe_archive_dir(
    extract_root: Path,
    *,
    spec: DatasetSpec,
    output_dir: Path,
    limit: int | None = None,
    save_images: bool = True,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Convert extracted MolScribe CSV/image files into local unified artifacts.

    ``keep_ids`` (when not ``None``) restricts materialization to rows whose
    ``sample_id`` is in the set (the index-referenced ``source_id`` values);
    never-referenced rows are skipped before their image is decoded. Kept rows are
    byte-identical to the full pull. ``None`` keeps the full-source behavior.
    """

    discovered_csv_paths = sorted(extract_root.rglob("*.csv"))
    csv_paths = selected_molscribe_csv_paths(discovered_csv_paths)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {extract_root}")

    converted_by_split: dict[str, list[dict[str, Any]]] = {}
    for csv_path in csv_paths:
        split = split_from_csv_path(csv_path)
        rows = converted_by_split.setdefault(split, [])
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, raw in enumerate(reader):
                if limit is not None and index >= limit:
                    break
                row = dict(raw)
                if keep_ids is not None:
                    _, image_value = first_present_with_field(row, MOLSCRIBE_IMAGE_FIELDS)
                    source_image = resolve_molscribe_image_path(
                        image_value, csv_path=csv_path, extract_root=extract_root
                    )
                    sample_id = _molscribe_sample_id(
                        row,
                        spec=spec,
                        split=split,
                        index=index,
                        source_image=source_image,
                        image_value=image_value,
                    )
                    if sample_id not in keep_ids:
                        continue
                rows.append(
                    normalize_molscribe_csv_row(
                        row,
                        spec=spec,
                        split=split,
                        index=index,
                        csv_path=csv_path,
                        extract_root=extract_root,
                        images_dir=(data_root() / "images") if save_images else None,
                    )
                )

    for split, rows in converted_by_split.items():
        write_jsonl(rows, output_dir / spec.name / f"{split}.jsonl")

    all_rows = [row for rows in converted_by_split.values() for row in rows]
    write_jsonl(all_rows, output_dir / spec.name / "unified.jsonl")
    manifest = {
        "spec": asdict(spec),
        "splits": {split: len(rows) for split, rows in converted_by_split.items()},
        "total": len(all_rows),
        "unified_jsonl": str(output_dir / spec.name / "unified.jsonl"),
        "source_dir": str(extract_root),
        "save_images": save_images,
        "discovered_csvs": [str(path) for path in discovered_csv_paths],
        "selected_csvs": [str(path) for path in csv_paths],
    }
    _record_keep_coverage(manifest, spec=spec, rows=all_rows, keep_ids=keep_ids, limit=limit)
    (output_dir / spec.name / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def download_and_convert_archive(
    spec: DatasetSpec,
    *,
    output_dir: Path,
    cache_dir: str | None = None,
    limit: int | None = None,
    save_images: bool = True,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Download an HF-hosted archive dataset and convert it.

    ``keep_ids`` (when not ``None``) is threaded to both the MolScribe extraction
    and conversion passes so only the index-referenced images are unzipped and
    materialized. ``None`` keeps the full-source behavior.
    """

    if not spec.archive_filename:
        raise ValueError(f"Archive spec {spec.name} is missing archive_filename")

    from huggingface_hub import hf_hub_download

    archive_path = Path(
        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.archive_filename,
            repo_type=spec.repo_type,
            revision=spec.revision,
            cache_dir=cache_dir,
        )
    )
    extract_root = output_dir / spec.name / "hf"
    if spec.source == "molscribe":
        extraction = extract_molscribe_archive_inputs(
            archive_path,
            extract_root,
            limit=limit,
            spec=spec,
            keep_ids=keep_ids,
        )
    else:
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)
        extraction = {
            "archive_path": str(archive_path),
            "selected_csvs": [],
            "referenced_images": None,
            "extracted_members": None,
            "reused_members": None,
        }
    manifest = convert_molscribe_archive_dir(
        extract_root,
        spec=spec,
        output_dir=output_dir,
        limit=limit,
        save_images=save_images,
        keep_ids=keep_ids,
    )
    manifest["archive_extraction"] = extraction
    (output_dir / spec.name / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def selected_dataset_names(values: list[str]) -> list[str]:
    """Resolve CLI dataset selections, treating any ``all`` as all specs."""

    if "all" in values:
        return list(DATASET_SPECS)
    return values


def config_validation_rows(
    specs: list[DatasetSpec],
    *,
    configs_by_repo: dict[str, list[str]],
    splits_by_name: dict[str, list[str]] | None = None,
    files_by_repo: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build a JSON-serializable config validation report."""

    splits_by_name = splits_by_name or {}
    files_by_repo = files_by_repo or {}
    rows = []
    for spec in specs:
        configs = configs_by_repo.get(spec.repo, [])
        files = files_by_repo.get(spec.repo, [])
        if spec.kind == "archive":
            config_present = bool(spec.archive_filename and spec.archive_filename in files)
        else:
            config_present = spec.config is None or spec.config in configs
        splits = splits_by_name.get(spec.name, [])
        required_splits = list(spec.allowed_splits or [])
        required_splits_present = all(split in splits for split in required_splits)
        rows.append(
            {
                "name": spec.name,
                "repo": spec.repo,
                "config": spec.config,
                "kind": spec.kind,
                "archive_filename": spec.archive_filename,
                "config_present": config_present,
                "splits": splits,
                "required_splits": required_splits,
                "required_splits_present": required_splits_present,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and convert public Markush/OCSR datasets into the unified schema",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=Path("data/unified"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *DATASET_SPECS],
        help="Dataset specs to download/convert",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Per-split row limit for quick trial runs"
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--hf-split",
        type=str,
        default=None,
        help="Optional HuggingFace split expression, e.g. train[:1], for quick validation",
    )
    parser.add_argument(
        "--validate-configs-only",
        action="store_true",
        help="Validate selected HuggingFace repo/config/split names without downloading rows",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-original", action="store_true", help="Save raw HF datasets to disk")
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Do not export images while writing unified JSONL metadata",
    )
    parser.add_argument(
        "--save-unified",
        action="store_true",
        help="Also save a combined local HuggingFace DatasetDict",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Per-split HuggingFace map workers for dataset normalization/image export",
    )
    return parser


def download_public_datasets(
    *,
    output: Path = Path("data/unified"),
    datasets: list[str] | None = None,
    limit: int | None = None,
    cache_dir: str | None = None,
    hf_split: str | None = None,
    validate_configs_only: bool = False,
    trust_remote_code: bool = False,
    save_original: bool = False,
    save_images: bool = True,
    save_unified: bool = False,
    workers: int = 1,
    revisions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Download/validate selected public datasets and return a manifest report.

    ``revisions`` optionally pins each selected spec to an upstream revision
    (keyed by spec name); a missing key keeps the spec's own ``revision``.
    """

    selected = selected_dataset_names(datasets or ["all"])
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    revisions = revisions or {}
    specs = [
        replace(DATASET_SPECS[name], revision=revisions.get(name, DATASET_SPECS[name].revision))
        for name in selected
    ]
    if validate_configs_only:
        from huggingface_hub import list_repo_files

        configs_by_repo: dict[str, list[str]] = {}
        splits_by_name: dict[str, list[str]] = {}
        files_by_repo: dict[str, list[str]] = {}
        for spec in specs:
            if spec.kind == "archive":
                if spec.repo not in files_by_repo:
                    files_by_repo[spec.repo] = list_repo_files(
                        repo_id=spec.repo,
                        repo_type=spec.repo_type,
                    )
                continue
            if spec.repo not in configs_by_repo:
                configs_by_repo[spec.repo] = get_dataset_config_names(
                    spec.repo,
                    cache_dir=cache_dir,
                )
            if spec.config is None or spec.config in configs_by_repo[spec.repo]:
                try:
                    splits_by_name[spec.name] = get_dataset_split_names(
                        spec.repo,
                        spec.config,
                        cache_dir=cache_dir,
                    )
                except Exception as exc:
                    splits_by_name[spec.name] = [f"ERROR: {exc}"]
        report = config_validation_rows(
            specs,
            configs_by_repo=configs_by_repo,
            splits_by_name=splits_by_name,
            files_by_repo=files_by_repo,
        )
        if not all(row["config_present"] and row["required_splits_present"] for row in report):
            raise SystemExit(1)
        return {"datasets": report}

    manifests = []
    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.kind == "archive":
            manifest = download_and_convert_archive(
                spec,
                output_dir=output,
                cache_dir=cache_dir,
                limit=limit,
                save_images=save_images,
            )
            manifests.append(manifest)
            unified_path = output / spec.name / "unified.jsonl"
            with unified_path.open(encoding="utf-8") as handle:
                all_rows.extend(json.loads(line) for line in handle if line.strip())
            continue

        dataset = load_dataset(
            spec.repo,
            spec.config,
            split=hf_split,
            revision=spec.revision,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if save_original:
            original_dir = output / spec.name / "hf"
            original_dir.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(original_dir))

        manifest = convert_loaded_dataset(
            dataset,
            spec=spec,
            output_dir=output,
            limit=limit,
            forced_split_name=split_name_from_expression(hf_split),
            save_images=save_images,
            workers=workers,
        )
        manifests.append(manifest)
        unified_path = output / spec.name / "unified.jsonl"
        with unified_path.open(encoding="utf-8") as handle:
            all_rows.extend(json.loads(line) for line in handle if line.strip())

    write_jsonl(all_rows, output / "all_unified.jsonl")
    if save_unified:
        save_unified_dataset(all_rows, output / "hf_unified")

    top_manifest = {"datasets": manifests, "total": len(all_rows)}
    (output / "manifest.json").write_text(
        json.dumps(top_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return top_manifest


def main() -> None:
    args = build_parser().parse_args()
    report = download_public_datasets(
        output=args.output,
        datasets=args.datasets,
        limit=args.limit,
        cache_dir=args.cache_dir,
        hf_split=args.hf_split,
        validate_configs_only=args.validate_configs_only,
        trust_remote_code=args.trust_remote_code,
        save_original=args.save_original,
        save_images=not args.no_save_images,
        save_unified=args.save_unified,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
