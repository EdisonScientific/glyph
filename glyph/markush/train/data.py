"""Resolve the frozen MarkushGlyph flat training index.

The 263,158-row flat training index is shipped on the dataset repo
``EdisonScientific/glyph-datasets`` as ``markush/train.jsonl``: the exact row
order and repeats used for training, each record carrying ``$GLYPH_DATA_ROOT``-
relative image refs (``images/<source>/train/<id>.png``). This resolver removes
the manual ``huggingface-cli`` step so ``glyph markush train`` can pull the index
itself when ``--data-path hub:markush/train.jsonl`` (or a bare ``hub:``) is passed.

The resolver shape mirrors the eval-side resolvers
(:func:`glyph.ocsr.eval._resolve_default_parquet` and
:func:`glyph.markush.eval.benchmark_data._resolve_repo_file`): an explicit path
override, then a local mirror env, then a ``GLYPH_DATA_ROOT`` snapshot, then
``hf_hub_download``. The source *images* the index references are a separate,
multi-GB prerequisite materialized by ``glyph data prepare --markush
--markush-only sources --allow-large``.
"""

from __future__ import annotations

import os
from pathlib import Path

from glyph.paths import datasets_dir

#: Dataset repo that hosts the frozen flat training index.
GLYPH_DATASETS_REPO = "EdisonScientific/glyph-datasets"

#: Pin for the hosted index. ``main`` resolves the published file; repin to the
#: exact upload commit SHA for byte-exact, reproducible fetches (same follow-up
#: as :data:`glyph.markush.eval.benchmark_data.MARKUSH_BENCHMARK_REVISION`).
MARKUSH_TRAIN_REVISION = "86f8e17d75699d8ed0799eb6d100ceba3faac56e"

#: Repo-relative path to the flat training index.
MARKUSH_TRAIN_JSONL_PATH = "markush/train.jsonl"

#: Env var pointing at a local mirror of the dataset-repo root (contains
#: ``markush/…``). Checked before the ``GLYPH_DATA_ROOT`` snapshot and the download.
MARKUSH_TRAIN_ROOT_ENV = "GLYPH_MARKUSH_TRAIN_ROOT"


def resolve_markush_train_jsonl(
    repo_path: str | None = None,
    *,
    explicit: str | Path | None = None,
) -> Path:
    """Resolve the frozen MarkushGlyph flat training index.

    Resolution order:

    1. ``explicit`` path override (used as-is).
    2. A local mirror under ``$GLYPH_MARKUSH_TRAIN_ROOT``.
    3. A ``$GLYPH_DATA_ROOT`` snapshot under ``datasets/``.
    4. ``hf_hub_download`` from :data:`GLYPH_DATASETS_REPO` at
       :data:`MARKUSH_TRAIN_REVISION`.

    ``repo_path`` defaults to :data:`MARKUSH_TRAIN_JSONL_PATH`; a bare ``hub:``
    argument (an empty string) falls back to that default too.
    """

    if explicit is not None:
        return Path(explicit).expanduser()

    path = repo_path or MARKUSH_TRAIN_JSONL_PATH

    mirror = os.environ.get(MARKUSH_TRAIN_ROOT_ENV)
    if mirror:
        candidate = Path(mirror).expanduser() / path
        if candidate.is_file():
            return candidate

    snapshot = datasets_dir() / path
    if snapshot.is_file():
        return snapshot

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=GLYPH_DATASETS_REPO,
            filename=path,
            repo_type="dataset",
            revision=MARKUSH_TRAIN_REVISION,
        )
    )
