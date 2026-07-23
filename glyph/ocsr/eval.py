"""Evaluation utilities for the USPTO OCSR benchmark.

This module is the *only* place that defines the canonicalization and
exact-match rule, so all training and evaluation code share the same
definition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

# Silence RDKit's parser noise; we surface failures via return values.
RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]

EVAL_CONTRACT_VERSION = "v1"

USPTO_BENCHMARK_TOTAL_ROWS = 5719
USPTO_BENCHMARK_PARSEABLE_ROWS = 5704  # 5719 minus 15 NaN-SMILES rows
USPTO_BENCHMARK_TARGET_CORRECT_FOR_95 = 5419  # ceil(0.95 * 5704)


def canonical_smiles(s: str | None) -> str | None:
    """Return RDKit canonical SMILES, or None if unparseable / empty."""

    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def graph_canonical_smiles(s: str | None) -> str | None:
    """Canonical SMILES with stereochemistry removed, used for the graph-only diagnostic."""

    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def chirality_canonical_smiles(s: str | None) -> str | None:
    """Chirality-kept canonical SMILES (ignore double-bond cis/trans geometry).

    Removes the ``/`` and ``\\`` bond-direction markers (E/Z / cis-trans
    stereochemistry) before parsing, then RDKit-canonicalizes with tetrahedral
    (``@``/``@@``) chirality preserved. This mirrors MolScribe's ``canon_smiles``
    helper, which strips bond directions before canonicalization, and defines
    the "B" grading convention: stricter than the graph-only check (which drops
    *all* stereo) but more lenient than full isomeric matching (which also
    requires cis/trans to agree).
    """

    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s.replace("/", "").replace("\\", ""))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@dataclass
class EvalResult:
    n: int
    n_correct: int
    n_correct_graph: int
    n_valid_pred: int
    n_invalid_gt: int
    canon_exact_match: float
    graph_exact_match: float
    valid_smiles_rate: float
    # Chirality-kept ("B") convention. Defaults keep positional/legacy callers
    # working; ``evaluate_pairs`` always populates them.
    n_correct_chirality: int = 0
    chirality_exact_match: float = 0.0


def evaluate_pairs(
    pred_smiles: Iterable[str | None],
    gt_smiles: Iterable[str | None],
) -> EvalResult:
    """Compute the eval-contract metrics from two parallel iterables.

    A prediction is "correct" iff its RDKit canonical SMILES exactly matches
    the canonical SMILES of the ground truth. Rows where the ground truth is
    unparseable are excluded from the primary metric (matching the contract).
    """

    pred_list = list(pred_smiles)
    gt_list = list(gt_smiles)
    if len(pred_list) != len(gt_list):
        raise ValueError("Length mismatch between predictions and ground truth")

    n = 0
    n_correct = 0
    n_correct_graph = 0
    n_correct_chirality = 0
    n_valid_pred = 0
    n_invalid_gt = 0
    for p, g in zip(pred_list, gt_list, strict=True):
        cg = canonical_smiles(g)
        if cg is None:
            n_invalid_gt += 1
            continue
        n += 1
        cp = canonical_smiles(p)
        if cp is not None:
            n_valid_pred += 1
        if cp is not None and cp == cg:
            n_correct += 1
        # chirality-kept ("B") check: cis/trans ignored, chirality kept
        bp = chirality_canonical_smiles(p)
        bg = chirality_canonical_smiles(g)
        if bp is not None and bg is not None and bp == bg:
            n_correct_chirality += 1
        # graph-only check uses canonical form with stereochemistry removed
        gp = graph_canonical_smiles(p)
        gg = graph_canonical_smiles(g)
        if gp is not None and gg is not None and gp == gg:
            n_correct_graph += 1

    if n == 0:
        return EvalResult(0, 0, 0, 0, n_invalid_gt, 0.0, 0.0, 0.0)
    return EvalResult(
        n=n,
        n_correct=n_correct,
        n_correct_graph=n_correct_graph,
        n_valid_pred=n_valid_pred,
        n_invalid_gt=n_invalid_gt,
        canon_exact_match=n_correct / n,
        graph_exact_match=n_correct_graph / n,
        valid_smiles_rate=n_valid_pred / n,
        n_correct_chirality=n_correct_chirality,
        chirality_exact_match=n_correct_chirality / n,
    )


def hash_strings_sorted(strs: Iterable[str]) -> str:
    """Match the contract's hash convention: sha256 of '\n'-joined sorted strs."""

    return hashlib.sha256("\n".join(sorted(strs)).encode()).hexdigest()


# ---------- Hugging Face benchmark loader --------------------------------

# The frozen, self-contained benchmark is hosted on our dataset repo
# ``EdisonScientific/glyph-datasets`` at ``ocsr/uspto_ocsr_benchmark.parquet``: the
# public benchmark images joined to the complete SMILES labels. Of the 15 upstream
# null labels, 13 are recovered from the MolScribe ``real/USPTO.csv`` release and 2
# are retained unlabelled/unscoreable — so the eval scores directly from it with no
# local reconstruction. ``USPTO_PARQUET`` (and ``--benchmark-parquet``) remain explicit
# overrides; without them we prefer a snapshot under GLYPH_DATA_ROOT, then download
# the hosted parquet. The underlying images originate from the public
# ``hheiden/USPTO_OCSR_benchmark`` dataset (credited in the dataset README).

OCSR_BENCHMARK_REPO = "EdisonScientific/glyph-datasets"
OCSR_BENCHMARK_PARQUET_PATH = "ocsr/uspto_ocsr_benchmark.parquet"
# Pin for the hosted self-contained benchmark. ``main`` resolves the published
# parquet; repin to the exact upload commit SHA for byte-exact fetches.
OCSR_BENCHMARK_REVISION = "efc5f409de068fce1b1f3802b352380e8fddc12d"

_LOCAL_BENCHMARK_NAMES = (
    "uspto_ocsr_benchmark.parquet",
    "uspto_ocsr_benchmark.arrow",
    "USPTO_OCSR_benchmark.parquet",
    "USPTO_OCSR_benchmark.arrow",
)


def _resolve_default_parquet() -> Path:
    """Resolve the USPTO benchmark parquet path.

    Order: the ``USPTO_PARQUET`` env override, then a local Parquet/Arrow snapshot
    under ``GLYPH_DATA_ROOT``, then the hosted self-contained parquet downloaded
    from ``EdisonScientific/glyph-datasets``.
    """

    import os

    from glyph.paths import datasets_dir

    env = os.environ.get("USPTO_PARQUET")
    if env:
        return Path(env).expanduser()

    for directory in (datasets_dir("ocsr"), datasets_dir()):
        for name in _LOCAL_BENCHMARK_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=OCSR_BENCHMARK_REPO,
            filename=OCSR_BENCHMARK_PARQUET_PATH,
            repo_type="dataset",
            revision=OCSR_BENCHMARK_REVISION,
        )
    )


def _read_benchmark_table(path: Path):
    """Read either a Parquet snapshot or a Hugging Face Arrow IPC cache."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if path.suffix.lower() == ".arrow":
        with pa.memory_map(str(path), "r") as source:
            try:
                return pa.ipc.open_stream(source).read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                return pa.ipc.open_file(source).read_all()
    return pq.read_table(path)


def load_uspto_benchmark(
    parquet_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load the benchmark from a Parquet or Arrow IPC snapshot.

    Each dict has at least: ``id``, ``smiles`` (may be None), ``source_of_label``
    (label provenance, may be None), ``image_bytes`` (bytes for the PNG, may be
    None), ``mol`` (V2000 string). The image column is flattened so downstream
    code does not need to know the HF representation.
    """

    path = Path(parquet_path) if parquet_path is not None else _resolve_default_parquet()
    table = _read_benchmark_table(path)
    cols = {c: table.column(c).to_pylist() for c in table.column_names}

    rows: list[dict[str, Any]] = []
    n = table.num_rows
    for i in range(n):
        image = cols["image"][i] if "image" in cols else None
        image_bytes = image.get("bytes") if isinstance(image, dict) and "bytes" in image else image
        rows.append(
            {
                "id": str(cols["id"][i]) if "id" in cols else None,
                "smiles": cols["smiles"][i] if "smiles" in cols else None,
                "selfies": cols["selfies"][i] if "selfies" in cols else None,
                "inchi": cols["inchi"][i] if "inchi" in cols else None,
                "mol": cols["mol"][i] if "mol" in cols else None,
                "source_of_label": (
                    cols["source_of_label"][i] if "source_of_label" in cols else None
                ),
                "image_bytes": image_bytes,
            }
        )
    return rows
