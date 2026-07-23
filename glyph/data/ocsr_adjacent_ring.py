"""Adjacent-ring PubChem mining (the ``adjacent_ring`` step).

This is the ONLY "This work" OCSR training data. It is a pure-logic mined SUBSET
of the PubChem-1M source (``pubchem_1m``), not a standalone dataset -- it copies
whole rows out of the base PubChem CSV and therefore overlaps ``pubchem_1m`` 100%
by construction.

The row filter keeps a PubChem row when all of:

1. ``has_adjacent_ring_stereo`` -- >=2 chiral centers with at least two of them
   adjacent atoms in the same ring;
2. SMILES length in ``[min_len, max_len]`` (defaults 50..180);
3. RDKit-parseable.

Motivated by canary patents US07320972 (pyrrolidinone lactam, adjacent
stereocenters, fluorobenzene) and US07320974 (oxetane/THP thiazolidinedione
stereo). Scanning the full 1,000,000-row PubChem-1M keeps 64,752 rows (6.49%).
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

DEFAULT_MIN_LEN = 50
DEFAULT_MAX_LEN = 180
_SMILES_COLUMNS = ("SMILES", "smiles", "Smiles", "canonical_smiles")

# Cap on the auto-selected worker count and how many input rows the parallel
# path buffers per pool dispatch (keeps peak memory bounded for the 1M-row scan).
_WORKER_CAP = 8
_CHUNK_ROWS = 10_000


def default_workers() -> int:
    """Sane default pool size: ``min(cpu_count, 8)`` (at least 1)."""

    return max(1, min(os.cpu_count() or 1, _WORKER_CAP))


def has_adjacent_ring_stereo(mol: Any) -> bool:
    """Return True iff ``mol`` has two adjacent ring atoms both stereo-tagged."""

    from rdkit import Chem

    if mol is None:
        return False
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    centers_idx = {c[0] for c in centers}
    if len(centers_idx) < 2:
        return False
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        in_ring = set(ring) & centers_idx
        if len(in_ring) < 2:
            continue
        for atom_idx in in_ring:
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                if neighbor.GetIdx() in in_ring and neighbor.GetIdx() != atom_idx:
                    return True
    return False


def _detect_smiles_column(fieldnames: Sequence[str] | None) -> str:
    for candidate in _SMILES_COLUMNS:
        if fieldnames and candidate in fieldnames:
            return candidate
    raise ValueError(f"No SMILES column in {fieldnames}")


def _adjacent_ring_keep(smiles: str, *, min_len: int, max_len: int) -> bool:
    """The full per-row filter: length window, RDKit-parseable, adjacent-ring.

    This single predicate drives both the serial and the parallel paths, so the
    kept set (and therefore the output row order/content) is identical. Length is
    checked first to short-circuit the RDKit parse, matching the original loop.
    """

    if not (min_len <= len(smiles) <= max_len):
        return False
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return has_adjacent_ring_stereo(mol)


def _init_worker() -> None:
    """Silence RDKit's C++ logger in each pool worker (cosmetic only)."""

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]


def _chunked(rows: Iterator[dict[str, str]], size: int) -> Iterator[list[dict[str, str]]]:
    chunk: list[dict[str, str]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


@dataclass
class AdjacentRingMineResult:
    """Summary of a :func:`mine_adjacent_ring` run."""

    scanned: int
    kept: int
    out_csv: Path

    @property
    def keep_rate(self) -> float:
        return (self.kept / self.scanned) if self.scanned else 0.0


def mine_adjacent_ring(
    in_csv: Path | str,
    out_csv: Path | str,
    *,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    max_rows: int | None = None,
    workers: int | None = None,
) -> AdjacentRingMineResult:
    """Filter a PubChem-1M CSV to the adjacent-ring-stereo subset.

    Whole rows are copied (preserving all columns, e.g. ``pubchem_cid``,
    ``InChI``, ``SMILES``, ``num_atoms``) so the output remains a strict subset
    of the input. ``max_rows`` caps the number of *scanned* input rows (used for
    quick sampling), matching the original ``--max_rows`` semantics.

    ``workers`` sets the process-pool size for the RDKit filter. ``None`` picks
    :func:`default_workers` (``min(cpu_count, 8)``); ``1`` runs the serial loop.
    The kept rows (order and content) are identical regardless of worker count:
    rows are read and written serially in input order and only the per-row RDKit
    predicate is parallelized, in-order, in bounded chunks.
    """

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]

    in_csv = Path(in_csv).expanduser()
    out_csv = Path(out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    workers = default_workers() if workers is None else max(1, workers)

    keep = partial(_adjacent_ring_keep, min_len=min_len, max_len=max_len)
    scanned = kept = 0
    with (
        in_csv.open(newline="", encoding="utf-8") as fin,
        out_csv.open("w", newline="", encoding="utf-8") as fout,
    ):
        reader = csv.DictReader(fin)
        smiles_col = _detect_smiles_column(reader.fieldnames)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames or [])
        writer.writeheader()

        def capped_rows() -> Iterator[dict[str, str]]:
            nonlocal scanned
            for row in reader:
                if max_rows is not None and scanned >= max_rows:
                    break
                scanned += 1
                yield row

        if workers <= 1:
            for row in capped_rows():
                if keep(row.get(smiles_col) or ""):
                    writer.writerow(row)
                    kept += 1
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                for chunk in _chunked(capped_rows(), _CHUNK_ROWS):
                    smiles_list = [row.get(smiles_col) or "" for row in chunk]
                    chunksize = max(1, min(512, len(chunk) // (workers * 4) or 1))
                    keeps = pool.map(keep, smiles_list, chunksize=chunksize)
                    for row, is_kept in zip(chunk, keeps, strict=True):
                        if is_kept:
                            writer.writerow(row)
                            kept += 1

    return AdjacentRingMineResult(scanned=scanned, kept=kept, out_csv=out_csv)
