"""Materialize OCSR benchmark rows into task-conditioned eval JSONL.

Library backing the ``scripts/materialize_ocsr_eval_jsonl.py`` CLI (a thin
wrapper over this module). The output rows target the task-conditioned OCSR
convention used by the multitask eval path::

    <image>
    Extract the molecular structure from this image as SMILES.
    Target format: return a SMILES string.

Targets are RDKit-canonical isomeric SMILES. Rows with unparseable ground-truth
SMILES are skipped and counted in the returned stats.

This module depends on ``glyph.markush.data`` for the shared OCSR prompt/target
contract; it is intentionally *not* imported by the core ``glyph data prepare
--ocsr`` path (which materializes the four datasets + the 5719 benchmark), so
that path stays free of the Markush multitask stack.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from glyph.markush.data.example_modes import MOLECULE_OCSR_PROMPT
from glyph.markush.data.ocsr import (
    OCSR_TARGET_FORMAT,
    RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX,
    canonicalize_ocsr_smiles_checked,
)
from glyph.markush.data.prepare_sft import (
    prompt_with_target_instruction,
    target_spec_for_record,
)

ID_COLUMNS = ("id", "image_id", "sample_id", "uid")
SMILES_COLUMNS = ("smiles", "SMILES", "gt_smiles", "target", "label")
IMAGE_COLUMNS = (
    "image_path",
    "file_path",
    "image",
    "path",
    "png_path",
    "rendered_path",
    "file_name",
)


@dataclass(slots=True)
class MaterializeStats:
    dataset_name: str
    input_path: str
    output_jsonl: str
    image_dir: str
    source_format: str
    total_rows: int = 0
    written_rows: int = 0
    skipped_invalid_smiles: int = 0
    skipped_missing_image: int = 0
    image_decode_failures: int = 0
    copied_images: int = 0
    embedded_images: int = 0
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None
    ids_sorted_sha256: str = ""
    canonical_smiles_sorted_sha256: str = ""


def _hash_sorted(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _first_existing_column(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    lower_lookup = {key.lower(): key for key in row}
    for candidate in candidates:
        if candidate in row:
            return candidate
        found = lower_lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _row_id(row: dict[str, Any], fallback: int) -> str:
    column = _first_existing_column(row, ID_COLUMNS)
    if column is not None and row.get(column) not in {None, ""}:
        return str(row[column])
    return str(fallback)


def _row_smiles(row: dict[str, Any]) -> str | None:
    column = _first_existing_column(row, SMILES_COLUMNS)
    value = row.get(column) if column is not None else None
    return value if isinstance(value, str) else None


def _image_value(row: dict[str, Any]) -> Any:
    column = _first_existing_column(row, IMAGE_COLUMNS)
    return row.get(column) if column is not None else None


def _safe_image_name(sample_id: str, suffix: str = ".png") -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in sample_id)
    if not safe:
        safe = "sample"
    return safe if safe.lower().endswith(suffix.lower()) else f"{safe}{suffix}"


def _update_image_stats(stats: MaterializeStats, image_path: Path) -> None:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        stats.image_decode_failures += 1
        return
    stats.min_width = width if stats.min_width is None else min(stats.min_width, width)
    stats.max_width = width if stats.max_width is None else max(stats.max_width, width)
    stats.min_height = height if stats.min_height is None else min(stats.min_height, height)
    stats.max_height = height if stats.max_height is None else max(stats.max_height, height)


def _write_embedded_image(value: Any, output_path: Path) -> bool:
    image_bytes: bytes | None = None
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            image_bytes = raw
    elif isinstance(value, bytes):
        image_bytes = value
    if not image_bytes:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return True


def _resolve_csv_image(
    row: dict[str, Any], *, image_root: Path | None, input_dir: Path
) -> Path | None:
    value = _image_value(row)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if image_root is not None:
        return image_root / path
    return input_dir / path


def _row_payload(
    *,
    sample_id: str,
    dataset_name: str,
    image_path: Path,
    canonical_smiles: str,
    raw_smiles: str,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        image_ref = str(image_path.relative_to(output_dir))
    except ValueError:
        image_ref = str(image_path)
    target = target_spec_for_record({"task": "ocsr", "smiles": canonical_smiles})
    if target is None:
        raise ValueError(
            f"Canonical OCSR SMILES unexpectedly failed target spec: {canonical_smiles}"
        )
    user_prompt = prompt_with_target_instruction(MOLECULE_OCSR_PROMPT, target)
    return {
        "id": sample_id,
        "task": "ocsr",
        "mode": "molecule_ocsr",
        "source": dataset_name,
        "split": "eval",
        "benchmark": dataset_name,
        "image_path": image_ref,
        "smiles": canonical_smiles,
        "raw_smiles": raw_smiles,
        "target_source": f"smiles{RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX}",
        "target_format": OCSR_TARGET_FORMAT,
        "target_canonicalization_status": (
            "already_canonical" if canonical_smiles == raw_smiles.strip() else "canonicalized"
        ),
        "prompt": user_prompt,
        "messages": [
            {"role": "user", "content": f"<image>\n{user_prompt}"},
            {"role": "assistant", "content": canonical_smiles},
        ],
        "images": [image_ref],
    }


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def materialize_ocsr_eval(
    *,
    input_path: Path,
    output_dir: Path,
    dataset_name: str,
    source_format: str,
    image_root: Path | None = None,
    limit: int | None = None,
) -> MaterializeStats:
    """Materialize an OCSR benchmark (CSV or Parquet) into eval JSONL + images."""

    input_path = input_path.expanduser()
    output_dir = output_dir.expanduser()
    image_root = image_root.expanduser() if image_root is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images" / dataset_name
    image_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / f"eval_ocsr_{dataset_name}.jsonl"

    if source_format == "csv":
        rows = _load_csv_rows(input_path)
    elif source_format == "parquet":
        rows = _load_parquet_rows(input_path)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")

    stats = MaterializeStats(
        dataset_name=dataset_name,
        input_path=str(input_path),
        output_jsonl=str(output_jsonl),
        image_dir=str(image_dir),
        source_format=source_format,
    )
    written_ids: list[str] = []
    written_smiles: list[str] = []

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if limit is not None and stats.written_rows >= limit:
                break
            stats.total_rows += 1
            raw_smiles = _row_smiles(row)
            result = canonicalize_ocsr_smiles_checked(raw_smiles)
            if result.value is None:
                stats.skipped_invalid_smiles += 1
                continue
            sample_id = _row_id(row, index)
            out_image = image_dir / _safe_image_name(sample_id)
            if source_format == "parquet":
                if not _write_embedded_image(_image_value(row), out_image):
                    stats.skipped_missing_image += 1
                    continue
                stats.embedded_images += 1
            else:
                src_image = _resolve_csv_image(
                    row, image_root=image_root, input_dir=input_path.parent
                )
                if src_image is None or not src_image.exists():
                    stats.skipped_missing_image += 1
                    continue
                out_image = image_dir / _safe_image_name(
                    sample_id, suffix=src_image.suffix or ".png"
                )
                shutil.copy2(src_image, out_image)
                stats.copied_images += 1

            _update_image_stats(stats, out_image)
            payload = _row_payload(
                sample_id=sample_id,
                dataset_name=dataset_name,
                image_path=out_image,
                canonical_smiles=result.value,
                raw_smiles=raw_smiles or "",
                output_dir=output_dir,
            )
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            stats.written_rows += 1
            written_ids.append(sample_id)
            written_smiles.append(result.value)

    stats.ids_sorted_sha256 = _hash_sorted(written_ids)
    stats.canonical_smiles_sorted_sha256 = _hash_sorted(written_smiles)
    return stats
