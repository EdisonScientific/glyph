"""Data-format adapters for MolNexTR/MolScribe-style row schemas.

The two reference repos share an effectively-identical dataset layout. We
adapt their CSV schemas into a small set of typed records so that downstream
training code can stay independent of which specific CSV is on disk. This
module deliberately does not depend on cv2 or albumentations — it is purely
about parsing rows. Image rendering and augmentation belong elsewhere.

Reference (read-only inspection of upstream repos):
- MolNexTR/MolNexTR/dataset.py:332 (TrainDataset)
- MolScribe/molscribe/dataset.py:327 (TrainDataset)
- MolNexTR/main.py:466-469 (image_id derivation)
"""

from __future__ import annotations

import ast
import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainRow:
    """A single training-row record.

    Either ``image_path`` is a real file (when ``dynamic_indigo`` is False) or
    the row is intended for on-the-fly Indigo rendering from ``smiles`` (when
    ``dynamic_indigo`` is True for the training split).
    """

    smiles: str
    image_id: str
    image_path: str | None = None
    node_coords: list[list[float]] | None = None
    edges: list[list[int]] | None = None


@dataclass
class EvalRow:
    """A single test/eval-row record. ``smiles`` may be None for blind sets."""

    image_id: str
    image_path: str
    smiles: str | None = None


def _derive_image_id(row: dict[str, str], file_path: str | None) -> str:
    if row.get("image_id"):
        return str(row["image_id"])
    if row.get("pubchem_cid"):
        return str(row["pubchem_cid"])
    if file_path:
        # Match upstream behavior: stem of last path component.
        return Path(file_path).stem
    raise ValueError("row has no image_id, pubchem_cid, or file_path")


def _safe_literal(text: str | None) -> object | None:
    if text is None or text == "":
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def parse_pubchem_train_row(row: dict[str, str]) -> TrainRow:
    """Parse a PubChem-style train row.

    Required column: ``SMILES``. ``pubchem_cid`` is used as ``image_id`` if
    present. Used with dynamic Indigo rendering, so no ``file_path`` is
    required and none is read.
    """

    smiles = (row.get("SMILES") or "").strip()
    if not smiles:
        raise ValueError(f"PubChem row missing SMILES: {row!r}")
    image_id = _derive_image_id(row, file_path=None)
    return TrainRow(smiles=smiles, image_id=image_id)


def parse_uspto_train_row(row: dict[str, str]) -> TrainRow:
    """Parse a MolNexTR/MolScribe USPTO-style aux train row.

    Required: ``file_path``, ``SMILES``. Optional but commonly present:
    ``node_coords`` (literal list of [x,y]), ``edges`` (literal list of
    [u,v,bond_type] triples).
    """

    smiles = (row.get("SMILES") or row.get("raw_SMILES") or "").strip()
    file_path = (row.get("file_path") or "").strip()
    if not smiles:
        raise ValueError(f"USPTO row missing SMILES: {row!r}")
    if not file_path:
        raise ValueError(f"USPTO row missing file_path: {row!r}")
    image_id = _derive_image_id(row, file_path=file_path)

    coords_raw: Any = _safe_literal(row.get("node_coords"))
    edges_raw: Any = _safe_literal(row.get("edges"))
    coords: list[list[float]] | None = None
    edges: list[list[int]] | None = None
    if isinstance(coords_raw, list):
        coords = [[float(x), float(y)] for x, y in coords_raw]
    if isinstance(edges_raw, list):
        edges = [[int(u), int(v), int(bond)] for u, v, bond in edges_raw]
    return TrainRow(
        smiles=smiles,
        image_id=image_id,
        image_path=file_path,
        node_coords=coords,
        edges=edges,
    )


def parse_real_eval_row(row: dict[str, str]) -> EvalRow:
    """Parse a real-benchmark CSV row (e.g. data/real/USPTO.csv)."""

    file_path = (row.get("file_path") or "").strip()
    if not file_path:
        raise ValueError(f"Eval row missing file_path: {row!r}")
    image_id = _derive_image_id(row, file_path=file_path)
    smiles_raw = row.get("SMILES")
    smiles = smiles_raw.strip() if isinstance(smiles_raw, str) else None
    if smiles == "":
        smiles = None
    return EvalRow(image_id=image_id, image_path=file_path, smiles=smiles)


def iter_csv_rows(path: Path | str) -> Iterator[dict[str, str]]:
    """Yield rows from a CSV file as plain dicts (no pandas dependency)."""

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        yield from reader


def parse_pubchem_csv(path: Path | str) -> list[TrainRow]:
    return [parse_pubchem_train_row(r) for r in iter_csv_rows(path)]


def parse_uspto_csv(path: Path | str) -> list[TrainRow]:
    return [parse_uspto_train_row(r) for r in iter_csv_rows(path)]


def parse_real_eval_csv(path: Path | str) -> list[EvalRow]:
    return [parse_real_eval_row(r) for r in iter_csv_rows(path)]


def to_train_rows_from_hf_benchmark(
    rows: Iterable[dict],
) -> list[TrainRow]:
    """Adapter: convert HF-USPTO-benchmark rows into TrainRow records.

    Useful only for tiny sanity runs; the real benchmark is *eval*, not train.
    """

    out: list[TrainRow] = []
    for r in rows:
        smi = r.get("smiles")
        if not isinstance(smi, str) or not smi:
            continue
        out.append(TrainRow(smiles=smi.strip(), image_id=str(r.get("id"))))
    return out
