"""Fetchers for the public Markush sources behind ``glyph data prepare --markush``.

Every fetch here touches the network and, for the training corpora, downloads a
multi-gigabyte image tree (MG1 ~25 GB, USPTO-MOL-M ~2.9 GB, MolParser ~2 GB,
MolScribe ~2.2 GB). To keep the pipeline safe by default, large downloads are
*refused* unless the caller passes ``allow_large=True`` -- the CLI surfaces this
as ``--allow-large``. The three DS4SD eval benchmarks are tiny (878/103/74 rows,
~100 MB), so they are *not* size-guarded and build without ``--allow-large``.
``--dry-run`` plans without touching the network; a ``limit`` slices the upstream
split so a small validation run stays bounded.

The heavy import + unified-schema conversion logic already lives in
:mod:`glyph.markush.data.importers.public_datasets`; this module is the thin,
guarded, §3-aware adapter the release orchestrator calls. Each source is
materialized under ``$GLYPH_DATA_ROOT/datasets/markush/<dataset_name>/`` with a
``unified.jsonl`` (image refs are ``$GLYPH_DATA_ROOT``-relative), and the
importer name / pinned revision are threaded from :mod:`glyph.data.layout`.

None of these datasets are re-hosted by this work; they are referenced by
manifest at the pinned revisions recorded in :mod:`glyph.data.layout`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from glyph.data.layout import (
    MARKUSH_DATASET_TO_SPEC,
    MARKUSH_SPEC_REVISIONS,
    STATUS_MATERIALIZED,
    STATUS_PLANNED,
    STATUS_REFUSED,
)
from glyph.data.ocsr_sources import ALLOW_LARGE_HINT, FetchOutcome

LOGGER = logging.getLogger(__name__)

#: Rough on-disk download sizes per §3 dataset name (from the layout provenance),
#: surfaced in the refusal/dry-run messages so a caller knows the cost.
MARKUSH_SOURCE_SIZES: dict[str, str] = {
    "mg1_synthetic": "~25 GB",
    "mg2_structure_only": "~2.9 GB",
    "molparser": "~2.0 GB",
    "ordinary_5pct": "~2.2 GB",
}
MARKUSH_BENCHMARK_SIZE = "~100 MB"


def markush_source_for_dataset(dataset_name: str) -> str:
    """Return the flat-index ``source`` key for a §3 ``dataset_name``.

    The training index and materialized image tree are keyed by ``source``
    (``mg1_synthetic``/``uspto_mol_m``/``molparser``/``molscribe``), which diverges
    from two §3 directory names (``mg2_structure_only`` -> ``uspto_mol_m``,
    ``ordinary_5pct`` -> ``molscribe``). This resolves the divergence via the
    importer spec so the keep-set is looked up under the right key.
    """

    from glyph.markush.data.importers.public_datasets import DATASET_SPECS

    return DATASET_SPECS[MARKUSH_DATASET_TO_SPEC[dataset_name]].source


def load_markush_keep_ids(index_path: str | Path | None = None) -> dict[str, set[str]] | None:
    """Return ``{source: set(source_id)}`` from the frozen flat training index.

    The index (``markush/train.jsonl``) is resolved with the same resolver the
    trainer uses (:func:`glyph.markush.train.data.resolve_markush_train_jsonl`):

    * ``index_path`` starting with ``hub:`` -> the hub repo-relative path (a bare
      ``hub:`` means the default ``markush/train.jsonl``);
    * any other ``index_path`` -> an explicit filesystem path;
    * ``None`` -> the default resolution order (local mirror, ``$GLYPH_DATA_ROOT``
      snapshot, then ``hf_hub_download``).

    Each row carries ``source`` + ``source_id``; the returned map is the set of
    ``source_id`` values to materialize per source. Returns ``None`` when the index
    cannot be resolved or read (e.g. offline with no snapshot), so the caller can
    fall back to full-source materialization.
    """

    from glyph.markush.train.data import resolve_markush_train_jsonl

    try:
        if index_path is not None and str(index_path).startswith("hub:"):
            resolved = resolve_markush_train_jsonl(str(index_path)[len("hub:") :] or None)
        elif index_path is not None:
            resolved = resolve_markush_train_jsonl(explicit=index_path)
        else:
            resolved = resolve_markush_train_jsonl()
    except Exception as exc:  # any resolution failure -> full-source fallback
        LOGGER.warning("Could not resolve the Markush training index (%s)", exc)
        return None

    if not resolved.is_file():
        LOGGER.warning("Resolved Markush training index does not exist: %s", resolved)
        return None

    keep: dict[str, set[str]] = {}
    try:
        with resolved.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = row.get("source")
                source_id = row.get("source_id")
                if source and source_id is not None:
                    keep.setdefault(str(source), set()).add(str(source_id))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read the Markush training index %s (%s)", resolved, exc)
        return None

    return keep or None


def _pinned_spec(importer_name: str, dataset_name: str):
    """Return a ``DatasetSpec`` renamed to ``dataset_name`` at its pinned revision.

    The rename makes the importer materialize under the standard directory
    (``datasets/markush/<dataset_name>/`` + ``images/<dataset_name>/``) rather than
    the raw importer name, and keeps image refs ``$GLYPH_DATA_ROOT``-relative.
    """

    from glyph.markush.data.importers.public_datasets import DATASET_SPECS

    spec = DATASET_SPECS[importer_name]
    return replace(
        spec,
        name=dataset_name,
        revision=MARKUSH_SPEC_REVISIONS.get(importer_name, spec.revision),
    )


def unified_jsonl_path(out_root: Path, dataset_name: str) -> Path:
    """Return the ``unified.jsonl`` the importer writes for ``dataset_name``."""

    return out_root / dataset_name / "unified.jsonl"


def _should_refuse_large(*, limit: int | None, allow_large: bool, guard_large: bool) -> bool:
    """Return whether a full (``limit is None``) pull must be refused.

    Only the multi-GB training sources are size-guarded (``guard_large=True``): a
    full pull is refused unless the caller passes ``allow_large``. The three DS4SD
    eval benchmarks are tiny (878/103/74 rows, ~100 MB) so they set
    ``guard_large=False`` and build without ``--allow-large``.
    """

    return guard_large and limit is None and not allow_large


def _import_spec(
    importer_name: str,
    dataset_name: str,
    out_root: Path,
    *,
    size_hint: str,
    limit: int | None,
    allow_large: bool,
    dry_run: bool,
    save_images: bool,
    guard_large: bool = True,
    workers: int | None = None,
    keep_ids: set[str] | None = None,
) -> FetchOutcome:
    """Import one HF dataset/archive spec into ``out_root/<dataset_name>/``.

    ``limit`` slices the upstream split (``train[:limit]`` for datasets, a row cap
    for the MolScribe archive) so a small validation run stays bounded. A full
    (``limit is None``) pull of a size-guarded source downloads the entire multi-GB
    image tree and is refused unless ``allow_large`` is set. Small sources (the
    eval benchmarks) pass ``guard_large=False`` and are never size-refused.

    ``workers`` sets the process-pool size for the per-row source-image
    decode/re-encode. ``None`` uses ``min(cpu_count, 8)``; ``1`` runs serially.
    The materialized PNG bytes and unified JSONL are byte-identical either way.

    ``keep_ids`` (when not ``None``) restricts materialization to the ``source_id``
    values the frozen flat training index references, so a full pull materializes
    only the index-referenced images instead of the entire upstream tree. ``None``
    keeps the full-source behavior.
    """

    out_dir = out_root / dataset_name
    unified = unified_jsonl_path(out_root, dataset_name)

    if dry_run:
        plan = (
            f"slice first {limit} rows"
            if limit is not None
            else f"download the full {size_hint} image tree"
        )
        return FetchOutcome(
            STATUS_PLANNED,
            f"would import {importer_name} @ pinned rev ({plan}) -> {unified}",
            path=unified,
        )

    if _should_refuse_large(limit=limit, allow_large=allow_large, guard_large=guard_large):
        return FetchOutcome(
            STATUS_REFUSED,
            f"{importer_name} is a {size_hint} image tree; {ALLOW_LARGE_HINT}",
            path=out_dir,
        )

    from glyph.markush.data.importers.public_datasets import (
        convert_loaded_dataset,
        download_and_convert_archive,
        split_name_from_expression,
    )

    spec = _pinned_spec(importer_name, dataset_name)
    if spec.kind == "archive":
        manifest = download_and_convert_archive(
            spec,
            output_dir=out_root,
            limit=limit,
            save_images=save_images,
            keep_ids=keep_ids,
        )
    else:
        from datasets import load_dataset

        hf_split = None
        if limit is not None:
            base_split = (spec.allowed_splits or ("train",))[0]
            hf_split = f"{base_split}[:{limit}]"
        dataset = load_dataset(
            spec.repo,
            spec.config,
            split=hf_split,
            revision=spec.revision,
        )
        manifest = convert_loaded_dataset(
            dataset,
            spec=spec,
            output_dir=out_root,
            limit=limit,
            forced_split_name=split_name_from_expression(hf_split),
            save_images=save_images,
            workers=workers,
            keep_ids=keep_ids,
        )

    total = int(manifest.get("total", 0))
    detail = f"imported {total} rows from {importer_name} @ pinned rev -> {unified}"
    counts = {"rows": total}
    keep_requested = manifest.get("keep_requested")
    if keep_requested is not None:
        keep_missing = int(manifest.get("keep_missing", 0))
        counts["keep_requested"] = int(keep_requested)
        counts["keep_matched"] = int(manifest.get("keep_matched", 0))
        if keep_missing:
            counts["keep_missing"] = keep_missing
            detail += f"; WARNING {keep_missing} index source_ids not found upstream"
    return FetchOutcome(
        STATUS_MATERIALIZED,
        detail,
        path=unified,
        counts=counts,
    )


def import_markush_source(
    dataset_name: str,
    out_root: Path,
    *,
    limit: int | None = None,
    allow_large: bool = False,
    dry_run: bool = False,
    save_images: bool = True,
    workers: int | None = None,
    keep_ids: set[str] | None = None,
) -> FetchOutcome:
    """Import one training source (``mg1_synthetic``/``mg2_structure_only``/...).

    ``workers`` sets the process-pool size for the source-image materialization
    (``None`` = ``min(cpu_count, 8)``; ``1`` = serial). Output is byte-identical.

    ``keep_ids`` (when not ``None``) restricts materialization to the ``source_id``
    values the frozen flat training index references for this source. ``None``
    materializes the full upstream source.
    """

    importer_name = MARKUSH_DATASET_TO_SPEC[dataset_name]
    return _import_spec(
        importer_name,
        dataset_name,
        out_root,
        size_hint=MARKUSH_SOURCE_SIZES.get(dataset_name, "large"),
        limit=limit,
        allow_large=allow_large,
        dry_run=dry_run,
        save_images=save_images,
        workers=workers,
        keep_ids=keep_ids,
    )


def import_markush_benchmark(
    benchmark_name: str,
    out_root: Path,
    *,
    limit: int | None = None,
    allow_large: bool = False,
    dry_run: bool = False,
    save_images: bool = True,
    workers: int | None = None,
) -> FetchOutcome:
    """Import one eval benchmark (``ip5m``/``m2s``/``uspto_markush``) unified JSONL."""

    from glyph.data.layout import MARKUSH_BENCHMARK_TO_SPEC

    importer_name = MARKUSH_BENCHMARK_TO_SPEC[benchmark_name]
    return _import_spec(
        importer_name,
        benchmark_name,
        out_root,
        size_hint=MARKUSH_BENCHMARK_SIZE,
        limit=limit,
        allow_large=allow_large,
        dry_run=dry_run,
        save_images=save_images,
        guard_large=False,
        workers=workers,
    )
