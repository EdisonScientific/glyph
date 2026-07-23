"""Orchestration for ``glyph data prepare --ocsr``.

Materializes the four OCSR datasets and the 5,719-row benchmark under
``$GLYPH_DATA_ROOT``, each with a provenance sidecar. The heavy library logic lives in the sibling modules
(:mod:`glyph.data.ocsr_sources`, :mod:`glyph.data.ocsr_white_crop`,
:mod:`glyph.data.ocsr_adjacent_ring`) and :mod:`glyph.ocsr.benchmark`; this module is
the thin orchestrator the CLI calls.

Pipeline (idempotent; each already-materialized artifact is skipped unless
``force``):

1. ``pubchem_1m``            -- MolNexTR ``train_pubchem.csv``               (external, ~210 MB)
2. ``uspto_680k``            -- MolNexTR ``train_uspto.zip`` (images + CSV)  (external, ~2.1 GB)
3. ``stereo_200k_white``     -- Robert-zwr/Stereo-200k white-cropped         (external, ~4.3 GB)
4. ``adjacent_ring``        -- adjacent-ring-stereo subset of ``pubchem_1m`` (this work, derived)
5. benchmark                 -- ``uspto_ocsr_5719_complete.csv``             (~23 MB)

Full-run invocation and expected counts/sizes are documented in
``glyph/data/README.md``. Multi-GB fetches require ``allow_large=True``; a
``limit`` streams/samples a small slice for validation.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from glyph.data import layout
from glyph.data.layout import (
    ADJACENT_RING,
    PUBCHEM_1M,
    STATUS_ERROR,
    STATUS_MATERIALIZED,
    STATUS_PLANNED,
    STATUS_REFUSED,
    STATUS_SKIPPED_EXISTS,
    STEREO_200K_WHITE,
    USPTO_680K,
    PrepareReport,
    PrepareStep,
)
from glyph.paths import data_root as data_root_path

PUBCHEM_CSV_NAME = "pubchem_1m.csv"
ADJACENT_RING_CSV_NAME = "adjacent_ring.csv"
LOGGER = logging.getLogger(__name__)


def _record_provenance(name: str, directory: Path) -> None:
    record = layout.OCSR_PROVENANCE.get(name)
    if record is not None:
        layout.write_provenance(directory, record)


def _prepare_pubchem_1m(
    report: PrepareReport,
    *,
    limit: int | None,
    dry_run: bool,
    allow_large: bool,
    force: bool,
) -> Path | None:
    from glyph.data.ocsr_sources import fetch_pubchem_1m

    dataset_dir = layout.ocsr_dataset_dir(PUBCHEM_1M, root=report.data_root)
    out_csv = dataset_dir / PUBCHEM_CSV_NAME

    if out_csv.exists() and not force and not dry_run:
        report.add(PrepareStep(PUBCHEM_1M, STATUS_SKIPPED_EXISTS, "already present", out_csv))
        return out_csv
    try:
        outcome = fetch_pubchem_1m(out_csv, limit=limit, allow_large=allow_large, dry_run=dry_run)
    except Exception as exc:
        report.add(PrepareStep(PUBCHEM_1M, STATUS_ERROR, repr(exc), out_csv))
        return None
    step = report.add(
        PrepareStep(PUBCHEM_1M, outcome.status, outcome.detail, outcome.path, outcome.counts)
    )
    if step.status == STATUS_MATERIALIZED:
        _record_provenance(PUBCHEM_1M, dataset_dir)
        return out_csv
    return None


def _files_identical(left: Path, right: Path) -> bool:
    """Return whether two regular files have the same size and SHA-256 digest."""

    if not left.is_file() or not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False

    def _sha256(path: Path) -> bytes:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()

    return _sha256(left) == _sha256(right)


def _normalize_uspto_680k_layout(dataset_dir: Path, *, conflicts: list[Path] | None = None) -> int:
    """Normalize the extracted USPTO-680K tree to the documented layout.

    ``train_uspto.zip`` extracts into ``<dataset_dir>/train_uspto/`` with the CSV
    at ``train_uspto/train_uspto.csv`` and the images under
    ``train_uspto/<year>/...``. The trainer defaults, the config's ``uspto_680k``
    component, and the data-layout doc all expect ``<dataset_dir>/train.csv`` plus
    an ``<dataset_dir>/images/`` root (CSV ``file_path`` values keep their
    ``uspto_mol/`` prefix, stripped at load time so ``uspto_mol/<year>/...`` -->
    ``images/<year>/...``). Move the CSV to ``train.csv`` and the image subtree
    under ``images/`` so the natural defaults resolve with no env overrides.

    Self-healing / merge semantics: this walks ``train_uspto/`` **recursively** and
    moves each file to its mirror path under ``images/<relative>`` (or ``train.csv``
    for the top-level ``train_uspto.csv``), *merging* into any partial ``images/``
    left by an interrupted run rather than skipping a whole year-dir. On a
    destination collision, an identical nested duplicate is discarded. Differing
    files are both preserved: the destination remains untouched, the nested source
    remains in place, and ``train_uspto/`` is retained so the conflict is visible
    and can be resolved manually. Callers may pass ``conflicts`` to receive the
    conflicting source paths.

    Idempotent: a no-op once the nested ``train_uspto/`` tree is absent; unresolved
    conflicts remain unchanged on subsequent calls. Returns the number of files
    moved (identical duplicates discarded during the merge are not counted).
    """

    nested = dataset_dir / "train_uspto"
    if not nested.is_dir():
        return 0
    train_csv = dataset_dir / "train.csv"
    images_dir = dataset_dir / "images"
    moved = 0
    conflict_paths: list[Path] = []
    for src in sorted(p for p in nested.rglob("*") if p.is_file()):
        rel = src.relative_to(nested)
        dest = train_csv if rel == Path("train_uspto.csv") else images_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _files_identical(src, dest):
                src.unlink()
            else:
                conflict_paths.append(src)
            continue
        shutil.move(str(src), str(dest))
        moved += 1

    if conflicts is not None:
        conflicts.extend(conflict_paths)
    if conflict_paths:
        LOGGER.warning(
            "Preserved %d conflicting USPTO-680K file(s) under %s; destination files were not overwritten",
            len(conflict_paths),
            nested,
        )
    else:
        # The nested tree now holds only empty dirs; drop it (best effort).
        shutil.rmtree(nested, ignore_errors=True)
    return moved


def _prepare_uspto_680k(
    report: PrepareReport,
    *,
    limit: int | None,
    dry_run: bool,
    allow_large: bool,
    force: bool,
) -> None:
    from glyph.data.ocsr_sources import fetch_uspto_680k

    dataset_dir = layout.ocsr_dataset_dir(USPTO_680K, root=report.data_root)
    marker = dataset_dir / "train.csv"
    nested = dataset_dir / "train_uspto"

    # "Already present" only when fully normalized: train.csv exists AND no
    # stranded train_uspto/ remains. A leftover nested tree is healed below.
    if marker.exists() and not nested.is_dir() and not force and not dry_run:
        report.add(PrepareStep(USPTO_680K, STATUS_SKIPPED_EXISTS, "already present", marker))
        return

    # Heal a pre-existing extract in place rather than re-downloading ~2.1 GB:
    # merge a stranded train_uspto/ (nested tree, with or without a partial
    # top-level train.csv) into train.csv + images/.
    if not force and not dry_run and nested.is_dir():
        conflicts: list[Path] = []
        moved = _normalize_uspto_680k_layout(dataset_dir, conflicts=conflicts)
        if conflicts:
            report.add(
                PrepareStep(
                    USPTO_680K,
                    STATUS_ERROR,
                    f"preserved {len(conflicts)} conflicting file(s) under {nested}; "
                    "resolve them before retrying",
                    nested,
                    {"moved": moved, "conflicts": len(conflicts)},
                )
            )
            return
        if marker.exists():
            report.add(
                PrepareStep(
                    USPTO_680K,
                    STATUS_MATERIALIZED,
                    f"normalized existing extract to train.csv + images/ ({moved} moved)",
                    marker,
                )
            )
            _record_provenance(USPTO_680K, dataset_dir)
            return

    try:
        outcome = fetch_uspto_680k(
            dataset_dir, limit=limit, allow_large=allow_large, dry_run=dry_run
        )
    except Exception as exc:
        report.add(PrepareStep(USPTO_680K, STATUS_ERROR, repr(exc), dataset_dir))
        return
    if outcome.status == STATUS_MATERIALIZED and not dry_run:
        # Fold the archive's nested train_uspto/ tree into train.csv + images/.
        conflicts = []
        moved = _normalize_uspto_680k_layout(dataset_dir, conflicts=conflicts)
        if conflicts:
            report.add(
                PrepareStep(
                    USPTO_680K,
                    STATUS_ERROR,
                    f"preserved {len(conflicts)} conflicting file(s) under {nested}; "
                    "resolve them before retrying",
                    nested,
                    {"moved": moved, "conflicts": len(conflicts)},
                )
            )
            return
    reported_path = marker if marker.exists() else outcome.path
    step = report.add(
        PrepareStep(USPTO_680K, outcome.status, outcome.detail, reported_path, outcome.counts)
    )
    if step.status == STATUS_MATERIALIZED:
        _record_provenance(USPTO_680K, dataset_dir)


def _find_stereo_split_csv(root: Path, split: str) -> Path | None:
    """Locate the ``<split>.csv`` inside an extracted Stereo-200K tree."""

    direct = root / f"{split}.csv"
    if direct.exists():
        return direct
    matches = sorted(root.rglob(f"{split}.csv"))
    return matches[0] if matches else None


def _prepare_stereo_200k_white(
    report: PrepareReport,
    *,
    limit: int | None,
    dry_run: bool,
    allow_large: bool,
    force: bool,
    workers: int | None = None,
) -> None:
    from glyph.data.ocsr_sources import fetch_stereo_200k
    from glyph.data.ocsr_white_crop import white_crop_stereo_split

    dataset_dir = layout.ocsr_dataset_dir(STEREO_200K_WHITE, root=report.data_root)
    out_csv = dataset_dir / "train.csv"

    if out_csv.exists() and not force and not dry_run:
        report.add(
            PrepareStep(STEREO_200K_WHITE, STATUS_SKIPPED_EXISTS, "already present", out_csv)
        )
        return

    source_dir = dataset_dir / "_source"
    try:
        outcome = fetch_stereo_200k(source_dir, allow_large=allow_large, dry_run=dry_run)
    except Exception as exc:
        report.add(PrepareStep(STEREO_200K_WHITE, STATUS_ERROR, repr(exc), dataset_dir))
        return
    if outcome.status != STATUS_MATERIALIZED:
        report.add(PrepareStep(STEREO_200K_WHITE, outcome.status, outcome.detail, outcome.path))
        return

    try:
        src_csv = _find_stereo_split_csv(source_dir, "train")
        if src_csv is None:
            raise FileNotFoundError(f"no train.csv found under extracted {source_dir}")
        crop = white_crop_stereo_split(
            src_csv, src_csv.parent, dataset_dir, split="train", limit=limit, workers=workers
        )
    except Exception as exc:
        report.add(PrepareStep(STEREO_200K_WHITE, STATUS_ERROR, repr(exc), dataset_dir))
        return

    report.add(
        PrepareStep(
            STEREO_200K_WHITE,
            STATUS_MATERIALIZED,
            f"white-cropped {crop.written} images "
            f"(skipped {crop.skipped_existing}, missing {crop.missing_images})",
            out_csv,
            {
                "written": crop.written,
                "skipped_existing": crop.skipped_existing,
                "missing": crop.missing_images,
            },
        )
    )
    _record_provenance(STEREO_200K_WHITE, dataset_dir)


def _prepare_adjacent_ring(
    report: PrepareReport,
    *,
    pubchem_csv: Path | None,
    min_len: int,
    max_len: int,
    dry_run: bool,
    force: bool,
    workers: int | None = None,
) -> None:
    from glyph.data.ocsr_adjacent_ring import mine_adjacent_ring

    dataset_dir = layout.ocsr_dataset_dir(ADJACENT_RING, root=report.data_root)
    out_csv = dataset_dir / ADJACENT_RING_CSV_NAME

    if out_csv.exists() and not force and not dry_run:
        report.add(PrepareStep(ADJACENT_RING, STATUS_SKIPPED_EXISTS, "already present", out_csv))
        return

    if pubchem_csv is None:
        pubchem_csv = layout.ocsr_dataset_dir(PUBCHEM_1M, root=report.data_root) / PUBCHEM_CSV_NAME

    if dry_run:
        report.add(
            PrepareStep(
                ADJACENT_RING,
                STATUS_PLANNED,
                f"would mine adjacent-ring-stereo subset of {pubchem_csv} "
                f"(len in [{min_len},{max_len}]) -> {out_csv}",
                out_csv,
            )
        )
        return

    if not pubchem_csv.exists():
        report.add(
            PrepareStep(
                ADJACENT_RING,
                STATUS_REFUSED,
                f"requires pubchem_1m first (missing {pubchem_csv})",
                out_csv,
            )
        )
        return

    try:
        result = mine_adjacent_ring(
            pubchem_csv, out_csv, min_len=min_len, max_len=max_len, workers=workers
        )
    except Exception as exc:
        report.add(PrepareStep(ADJACENT_RING, STATUS_ERROR, repr(exc), out_csv))
        return

    report.add(
        PrepareStep(
            ADJACENT_RING,
            STATUS_MATERIALIZED,
            f"mined {result.kept}/{result.scanned} rows ({result.keep_rate * 100:.2f}%)",
            out_csv,
            {"kept": result.kept, "scanned": result.scanned},
        )
    )
    _record_provenance(ADJACENT_RING, dataset_dir)


def _prepare_benchmark_5719(
    report: PrepareReport,
    *,
    benchmark_parquet: Path | None,
    dry_run: bool,
    allow_large: bool,
    force: bool,
) -> None:
    name = "uspto_ocsr_5719"
    out_csv = layout.uspto_5719_benchmark_path(root=report.data_root)

    if out_csv.exists() and not force and not dry_run:
        report.add(PrepareStep(name, STATUS_SKIPPED_EXISTS, "already present", out_csv))
        return

    if dry_run:
        report.add(
            PrepareStep(
                name,
                STATUS_PLANNED,
                "would build hheiden/USPTO_OCSR_benchmark (5704) + MolScribe recovery (13) "
                f"+ 2 none = 5719 -> {out_csv}",
                out_csv,
            )
        )
        return

    resolved_molscribe = None
    if allow_large:
        from glyph.data.ocsr_sources import fetch_molscribe_real_uspto_csv

        cache_csv = out_csv.parent / "molscribe_real_USPTO.csv"
        outcome = fetch_molscribe_real_uspto_csv(cache_csv, allow_large=True, dry_run=False)
        if outcome.status == STATUS_MATERIALIZED:
            resolved_molscribe = cache_csv

    try:
        from glyph.ocsr.benchmark import build_uspto5719_benchmark

        result = build_uspto5719_benchmark(
            benchmark_parquet=benchmark_parquet,
            molscribe_csv=resolved_molscribe,
            out=out_csv,
        )
    except Exception as exc:
        report.add(PrepareStep(name, STATUS_ERROR, repr(exc), out_csv))
        return

    counts = result.provenance["source_counts"]
    report.add(
        PrepareStep(
            name,
            STATUS_MATERIALIZED,
            f"built {result.provenance['total_rows']} rows "
            f"(hf={counts['hf_benchmark']}, molscribe={counts['molscribe_real']}, "
            f"none={counts['none_unrecovered']}; "
            f"scoreable={result.provenance['rdkit_scoreable_rows']})",
            out_csv,
            {k: int(v) for k, v in counts.items()},
        )
    )
    layout.write_provenance(out_csv.parent, layout.BENCHMARK_5719_PROVENANCE)


def prepare_ocsr(
    *,
    data_root: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    allow_large: bool = False,
    force: bool = False,
    benchmark_parquet: str | Path | None = None,
    adjacent_ring_min_len: int = 50,
    adjacent_ring_max_len: int = 180,
    workers: int | None = None,
) -> PrepareReport:
    """Prepare the OCSR datasets + benchmark; return a per-step report.

    Args:
        data_root: Override for ``$GLYPH_DATA_ROOT`` (defaults to the env/``./data``).
        limit: Sample only this many rows/images per source where supported
            (PubChem streaming head, USPTO/Stereo image caps). ``None`` = full.
        dry_run: Plan every step without touching the network or disk.
        allow_large: Permit the multi-GB source downloads (USPTO/Stereo, and the
            full PubChem CSV / MolScribe recovery archive). The 5719 benchmark
            auto-fetches MolScribe ``real/USPTO.csv`` for null recovery under this flag.
        force: Re-materialize even if the target artifact already exists.
        benchmark_parquet: Local HF benchmark snapshot (else auto-resolved).
        adjacent_ring_min_len / adjacent_ring_max_len: SMILES length bounds for adjacent-ring mining.
        workers: Process-pool size for the Stereo white-crop and the adjacent-ring
            mine. ``None`` uses ``min(cpu_count, 8)``; ``1`` runs serially. Output
            is byte-identical regardless of the worker count.
    """

    report = PrepareReport(family="ocsr", data_root=data_root_path(data_root))

    parquet_path = Path(benchmark_parquet).expanduser() if benchmark_parquet else None

    pubchem_csv = _prepare_pubchem_1m(
        report, limit=limit, dry_run=dry_run, allow_large=allow_large, force=force
    )
    _prepare_uspto_680k(report, limit=limit, dry_run=dry_run, allow_large=allow_large, force=force)
    _prepare_stereo_200k_white(
        report, limit=limit, dry_run=dry_run, allow_large=allow_large, force=force, workers=workers
    )
    _prepare_adjacent_ring(
        report,
        pubchem_csv=pubchem_csv,
        min_len=adjacent_ring_min_len,
        max_len=adjacent_ring_max_len,
        dry_run=dry_run,
        force=force,
        workers=workers,
    )
    _prepare_benchmark_5719(
        report,
        benchmark_parquet=parquet_path,
        dry_run=dry_run,
        allow_large=allow_large,
        force=force,
    )
    return report
