"""Orchestration for ``glyph data prepare --markush``.

Materializes the Markush training sources and the three DS4SD eval benchmarks
under ``$GLYPH_DATA_ROOT``, each with a provenance sidecar. The heavy library
logic lives in the sibling modules (:mod:`glyph.data.markush_sources`,
:mod:`glyph.data.markush_eval_jsonl`); this module is the thin orchestrator the
CLI calls.

Pipeline (idempotent; each already-materialized artifact is skipped unless
``force``):

1. sources     -- mg1_synthetic / mg2_structure_only / molparser / ordinary_5pct
                  imported from the pinned public HF datasets (external, multi-GB)
2. benchmarks  -- ip5m (n=878) / m2s (103) / uspto_markush (74) eval JSONLs

Full-run invocation and expected counts/sizes are documented in
``glyph/data/README.md``. Multi-GB source pulls require ``allow_large=True``; a
``limit`` slices each source for a small validation run. The three eval
benchmarks are tiny and build WITHOUT ``allow_large``.

By default a full ``sources`` run materializes only the ~263K images the frozen
flat training index (``markush/train.jsonl``) references -- not the ~1M images in
the full upstream trees. The index is resolved the same way the trainer resolves
``--data-path`` (see :func:`glyph.data.markush_sources.load_markush_keep_ids`);
if it cannot be resolved the run falls back to full-source materialization with a
warning. ``limit`` runs keep their prefix-slice behavior (no index subsetting).

The ``only`` selector (``sources``/``benchmarks``, see :data:`MARKUSH_STAGES`)
runs just one stage: ``only="sources"`` (the default) materializes the training
image sources, while ``only="benchmarks"`` (optionally narrowed to a subset via
``benchmarks=("ip5m",)``) builds only the small DS4SD eval JSONLs -- no multi-GB
training sources and no ``allow_large`` -- so a fresh user can cheaply build just
IP5-M (n=878) for eval.

The released 263,158-row flat training index (``markush/train.jsonl``) is hosted
on ``EdisonScientific/glyph-datasets``; source preparation does not reconstruct
it (see ``glyph/data/README.md``).

The emitted eval JSONLs (``benchmarks/markush/{ip5m,m2s,uspto_markush}/eval_*.jsonl``)
carry ``$GLYPH_DATA_ROOT``-relative image refs (``images/<benchmark>/<split>/…``);
pass ``--image-root "$GLYPH_DATA_ROOT"`` to ``glyph.markush.eval.fast_eval`` so
they resolve.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from glyph.data import layout
from glyph.data.layout import (
    MARKUSH_BENCHMARK_NAMES,
    MARKUSH_DATASET_NAMES,
    STATUS_ERROR,
    STATUS_MATERIALIZED,
    STATUS_PLANNED,
    STATUS_SKIPPED_EXISTS,
    PrepareReport,
    PrepareStep,
)
from glyph.paths import data_root as data_root_path

LOGGER = logging.getLogger(__name__)

#: Valid ``--markush-only`` stage selectors for :func:`prepare_markush`.
#:   * ``sources``    -- the four multi-GB training sources (default)
#:   * ``benchmarks`` -- only the small DS4SD eval benchmarks (no ``--allow-large``)
MARKUSH_STAGE_SOURCES = "sources"
MARKUSH_STAGE_BENCHMARKS = "benchmarks"
MARKUSH_STAGES = (
    MARKUSH_STAGE_SOURCES,
    MARKUSH_STAGE_BENCHMARKS,
)


def _record_provenance(name: str, directory: Path) -> None:
    record = layout.MARKUSH_PROVENANCE.get(name)
    if record is not None:
        layout.write_provenance(directory, record)


def _prepare_source(
    report: PrepareReport,
    dataset_name: str,
    *,
    limit: int | None,
    dry_run: bool,
    allow_large: bool,
    force: bool,
    workers: int | None = None,
    keep_ids: set[str] | None = None,
) -> None:
    from glyph.data.markush_sources import import_markush_source, unified_jsonl_path

    datasets_root = layout.markush_datasets_dir(root=report.data_root)
    dataset_dir = layout.markush_dataset_dir(dataset_name, root=report.data_root)
    unified = unified_jsonl_path(datasets_root, dataset_name)

    if unified.exists() and not force and not dry_run:
        report.add(PrepareStep(dataset_name, STATUS_SKIPPED_EXISTS, "already present", unified))
        return
    try:
        outcome = import_markush_source(
            dataset_name,
            datasets_root,
            limit=limit,
            allow_large=allow_large,
            dry_run=dry_run,
            workers=workers,
            keep_ids=keep_ids,
        )
    except Exception as exc:
        report.add(PrepareStep(dataset_name, STATUS_ERROR, repr(exc), dataset_dir))
        return
    step = report.add(
        PrepareStep(dataset_name, outcome.status, outcome.detail, outcome.path, outcome.counts)
    )
    if step.status == STATUS_MATERIALIZED:
        _record_provenance(dataset_name, dataset_dir)


def _prepare_benchmark(
    report: PrepareReport,
    benchmark_name: str,
    *,
    limit: int | None,
    dry_run: bool,
    allow_large: bool,
    force: bool,
    workers: int | None = None,
) -> None:
    from glyph.data.markush_eval_jsonl import prepare_benchmark
    from glyph.data.markush_sources import import_markush_benchmark, unified_jsonl_path

    name = f"benchmark_{benchmark_name}"
    bench_dir = layout.markush_benchmark_dir(benchmark_name, root=report.data_root)
    out_jsonl = bench_dir / f"eval_{benchmark_name}.jsonl"
    source_root = layout.markush_benchmarks_dir("_source", root=report.data_root)
    unified = unified_jsonl_path(source_root, benchmark_name)

    if out_jsonl.exists() and not force and not dry_run:
        report.add(PrepareStep(name, STATUS_SKIPPED_EXISTS, "already present", out_jsonl))
        return

    if dry_run:
        report.add(
            PrepareStep(
                name,
                STATUS_PLANNED,
                f"would import {benchmark_name} @ pinned rev and prepare eval JSONL -> {out_jsonl}",
                out_jsonl,
            )
        )
        return

    if not unified.exists():
        try:
            outcome = import_markush_benchmark(
                benchmark_name,
                source_root,
                limit=limit,
                allow_large=allow_large,
                dry_run=False,
                workers=workers,
            )
        except Exception as exc:
            report.add(PrepareStep(name, STATUS_ERROR, repr(exc), bench_dir))
            return
        if outcome.status != STATUS_MATERIALIZED:
            report.add(PrepareStep(name, outcome.status, outcome.detail, outcome.path))
            return

    try:
        stats = prepare_benchmark(
            source_jsonl=unified,
            output_jsonl=out_jsonl,
            benchmark=benchmark_name,
            data_root=report.data_root,
        )
    except Exception as exc:
        report.add(PrepareStep(name, STATUS_ERROR, repr(exc), bench_dir))
        return

    detail = f"wrote {stats.written_rows} eval rows (skipped {stats.skipped_rows})"
    if stats.absolute_image_rows:
        detail += f"; WARNING {stats.absolute_image_rows} rows kept an absolute image path"
    report.add(
        PrepareStep(
            name,
            STATUS_MATERIALIZED,
            detail,
            out_jsonl,
            {"written": stats.written_rows, "skipped": stats.skipped_rows},
        )
    )
    record = layout.MARKUSH_BENCHMARK_PROVENANCE.get(benchmark_name)
    if record is not None:
        layout.write_provenance(bench_dir, record)


def prepare_markush(
    *,
    data_root: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    allow_large: bool = False,
    force: bool = False,
    only: str = MARKUSH_STAGE_SOURCES,
    benchmarks: Sequence[str] | None = None,
    workers: int | None = None,
    index_path: str | Path | None = None,
) -> PrepareReport:
    """Prepare the Markush training sources and the three DS4SD eval benchmarks.

    Args:
        data_root: Override for ``$GLYPH_DATA_ROOT`` (defaults to the env/``./data``).
        limit: Slice each source to this many rows (small validation runs). ``None`` = full.
        dry_run: Plan every step without touching the network or disk.
        allow_large: Permit the multi-GB source downloads. The eval benchmarks are
            small and never require this.
        force: Re-materialize even if the target artifact already exists.
        only: Which stage to run -- one of :data:`MARKUSH_STAGES`
            (``sources``/``benchmarks``). ``sources`` (the default) materializes the
            training image sources; ``benchmarks`` builds just the small DS4SD eval
            JSONLs cheaply (no training sources, no ``--allow-large``).
        benchmarks: Restrict the benchmark stage to this subset of
            :data:`~glyph.data.layout.MARKUSH_BENCHMARK_NAMES` (e.g. ``("ip5m",)``
            to build only IP5-M, n=878). ``None`` builds all three.
        workers: Process-pool size for the per-row source-image decode/re-encode.
            ``None`` uses ``min(cpu_count, 8)``; ``1`` runs serially. The
            materialized PNG bytes and unified JSONL are byte-identical regardless
            of the worker count.
        index_path: Frozen flat training index to derive the per-source keep-set
            from (a filesystem path or a ``hub:`` value, mirroring the trainer's
            ``--data-path``). ``None`` resolves ``hub:markush/train.jsonl`` the same
            way the trainer does. On a full (``limit is None``) ``sources`` run the
            index subsets materialization to only the referenced ``source_id`` values;
            if the index cannot be resolved, the run falls back to full-source
            materialization with a warning.
    """

    if only not in MARKUSH_STAGES:
        raise ValueError(f"only must be one of {MARKUSH_STAGES}, got {only!r}")

    selected_benchmarks: tuple[str, ...] = (
        tuple(benchmarks) if benchmarks else MARKUSH_BENCHMARK_NAMES
    )
    unknown = [name for name in selected_benchmarks if name not in MARKUSH_BENCHMARK_NAMES]
    if unknown:
        raise ValueError(f"unknown benchmark(s) {unknown}; valid: {MARKUSH_BENCHMARK_NAMES}")

    do_sources = only == MARKUSH_STAGE_SOURCES
    do_benchmarks = only == MARKUSH_STAGE_BENCHMARKS

    report = PrepareReport(family="markush", data_root=data_root_path(data_root))

    if do_sources:
        keep_map = _resolve_source_keep_map(dry_run=dry_run, limit=limit, index_path=index_path)
        for dataset_name in MARKUSH_DATASET_NAMES:
            keep_ids = _keep_ids_for(keep_map, dataset_name)
            _prepare_source(
                report,
                dataset_name,
                limit=limit,
                dry_run=dry_run,
                allow_large=allow_large,
                force=force,
                workers=workers,
                keep_ids=keep_ids,
            )

    if do_benchmarks:
        for benchmark_name in selected_benchmarks:
            _prepare_benchmark(
                report,
                benchmark_name,
                limit=limit,
                dry_run=dry_run,
                allow_large=allow_large,
                force=force,
                workers=workers,
            )
    return report


def _resolve_source_keep_map(
    *, dry_run: bool, limit: int | None, index_path: str | Path | None
) -> dict[str, set[str]] | None:
    """Resolve the per-source keep-set from the frozen flat training index.

    Only a full (``limit is None``) materialization run is subset by the index; a
    ``dry_run`` (no writes, no network) and a sliced ``limit`` run (an intentional
    prefix validation) keep the full-source scan. Returns ``None`` when subsetting
    does not apply or the index cannot be resolved, in which case the sources are
    materialized in full (with a warning when an index was expected but missing).
    """

    if dry_run or limit is not None:
        return None

    from glyph.data.markush_sources import load_markush_keep_ids

    keep_map = load_markush_keep_ids(index_path)
    if keep_map is None:
        LOGGER.warning(
            "Markush training index not resolvable; materializing the full upstream "
            "sources. Provide the index (hub:markush/train.jsonl or a --markush-index "
            "path) to materialize only the ~263K referenced images."
        )
    return keep_map


def _keep_ids_for(keep_map: dict[str, set[str]] | None, dataset_name: str) -> set[str] | None:
    """Return the keep-set for ``dataset_name``'s source, or ``None`` for full pull."""

    if keep_map is None:
        return None
    from glyph.data.markush_sources import markush_source_for_dataset

    return keep_map.get(markush_source_for_dataset(dataset_name))
