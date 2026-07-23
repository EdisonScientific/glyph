"""Resolve and materialize the self-contained MarkushGlyph eval benchmarks.

The frozen benchmark artifacts live on the dataset repo
``EdisonScientific/glyph-datasets`` under ``markush-eval/<bench>/``. Each
benchmark ships **one parquet** carrying the pinned DS4SD MarkushGrapher-2
images (PNG bytes) joined to the prepared prompt/messages and the CXSMILES
target, so the predict step scores directly from it with no local ``glyph data
prepare`` reconstruction. This mirrors the OCSR resolver shape in
:mod:`glyph.ocsr.eval`: an explicit path override, then a local snapshot under
``GLYPH_DATA_ROOT``/``GLYPH_MARKUSH_EVAL_ROOT``, then ``hf_hub_download``.

Because :mod:`glyph.markush.eval.fast_eval` consumes a JSONL + ``--image-root``,
:func:`materialize_benchmark` unpacks one parquet into the same
``benchmarks/markush/<bench>/eval_<bench>.jsonl`` + ``images/<bench>/<split>/…``
layout ``glyph data prepare --markush`` would have written, byte-for-byte, so
the predict path is unchanged downstream.

The per-benchmark ground truth for the official DS4SD scorer
(``tools/mg2_scorer/rescore_with_public_mg2.py``) is resolved by
:func:`resolve_benchmark_gt`. ip5m uses DS4SD's own published eval output
(``markush-eval/ip5m/ds4sd_mg2_predictions_878.jsonl``: the authoritative
878-row contract with 8 unparseable ``None`` rows plus the MG2 baseline
column). m2s and uspto_markush use the derived DS4SD-form ground truth staged
alongside their parquet (no DS4SD model run is published for those two
benchmarks).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from glyph.paths import data_path, datasets_dir

#: Dataset repo that hosts the frozen self-contained benchmark artifacts.
MARKUSH_BENCHMARK_REPO = "EdisonScientific/glyph-datasets"

#: Pin for the hosted artifacts. ``main`` resolves the published revision;
#: repin to the exact upload commit SHA for byte-exact fetches (same follow-up
#: as :data:`glyph.ocsr.eval.OCSR_BENCHMARK_REVISION`).
MARKUSH_BENCHMARK_REVISION = "86f8e17d75699d8ed0799eb6d100ceba3faac56e"

#: Canonical benchmark names (matching ``glyph.data.layout.MARKUSH_BENCHMARK_NAMES``).
MARKUSH_BENCHMARK_NAMES = ("ip5m", "m2s", "uspto_markush")

_ALIASES: dict[str, str] = {
    "ip5": "ip5m",
    "ip5-m": "ip5m",
    "ip5_m": "ip5m",
    "ip5-markush": "ip5m",
    "ip5_markush": "ip5m",
    "uspto": "uspto_markush",
    "uspto-m": "uspto_markush",
    "uspto_m": "uspto_markush",
    "uspto-markush": "uspto_markush",
}

#: Repo-relative path to each benchmark's self-contained predict parquet.
_PARQUET_REPO_PATH: dict[str, str] = {
    "ip5m": "markush-eval/ip5m/ip5m_benchmark.parquet",
    "m2s": "markush-eval/m2s/m2s_benchmark.parquet",
    "uspto_markush": "markush-eval/uspto_markush/uspto_markush_benchmark.parquet",
}

#: Repo-relative path to each benchmark's official-scorer ground truth. ip5m is
#: DS4SD's own published eval output; m2s/uspto_markush are the derived DS4SD-form
#: ground truth staged alongside their parquet.
_GT_REPO_PATH: dict[str, str] = {
    "ip5m": "markush-eval/ip5m/ds4sd_mg2_predictions_878.jsonl",
    "m2s": "markush-eval/m2s/ds4sd_mg2_gt_103.jsonl",
    "uspto_markush": "markush-eval/uspto_markush/ds4sd_mg2_gt_74.jsonl",
}

#: Env var pointing at a local mirror of the dataset-repo root (contains
#: ``markush-eval/…``). Checked before the ``GLYPH_DATA_ROOT``
#: snapshot and the Hugging Face download.
MARKUSH_EVAL_ROOT_ENV = "GLYPH_MARKUSH_EVAL_ROOT"


def normalize_benchmark_name(value: str) -> str:
    """Normalize a user-facing benchmark name or alias to its canonical form."""

    key = str(value).strip().lower().replace(" ", "").replace("_", "-")
    key = _ALIASES.get(key, key.replace("-", "_"))
    key = _ALIASES.get(key, key)
    if key not in MARKUSH_BENCHMARK_NAMES:
        valid = ", ".join(MARKUSH_BENCHMARK_NAMES)
        raise ValueError(f"Unknown benchmark {value!r}; expected one of: {valid}")
    return key


def _resolve_repo_file(repo_path: str, explicit: str | Path | None = None) -> Path:
    """Resolve one hosted repo file.

    Order: an explicit path override, then a local mirror under
    ``$GLYPH_MARKUSH_EVAL_ROOT``, then a ``$GLYPH_DATA_ROOT`` snapshot under
    ``datasets/``, then ``hf_hub_download`` from ``EdisonScientific/glyph-datasets``.
    """

    if explicit is not None:
        return Path(explicit).expanduser()

    mirror = os.environ.get(MARKUSH_EVAL_ROOT_ENV)
    if mirror:
        candidate = Path(mirror).expanduser() / repo_path
        if candidate.is_file():
            return candidate

    snapshot = datasets_dir() / repo_path
    if snapshot.is_file():
        return snapshot

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=MARKUSH_BENCHMARK_REPO,
            filename=repo_path,
            repo_type="dataset",
            revision=MARKUSH_BENCHMARK_REVISION,
        )
    )


def resolve_benchmark_parquet(
    benchmark: str,
    *,
    explicit: str | Path | None = None,
) -> Path:
    """Resolve the self-contained predict parquet for ``benchmark``."""

    return _resolve_repo_file(_PARQUET_REPO_PATH[normalize_benchmark_name(benchmark)], explicit)


def resolve_benchmark_gt(
    benchmark: str,
    *,
    explicit: str | Path | None = None,
) -> Path:
    """Resolve the official-scorer ground-truth JSONL for ``benchmark``."""

    return _resolve_repo_file(_GT_REPO_PATH[normalize_benchmark_name(benchmark)], explicit)


def load_benchmark_records(parquet_path: str | Path) -> list[dict[str, Any]]:
    """Load one benchmark parquet into ``[{id, image_bytes, image_path, record}]``.

    ``record`` is the full prepared eval row (with a ``GLYPH_DATA_ROOT``-relative
    image ref); ``image_bytes`` is the PNG payload flattened from the ``image``
    struct so callers do not need to know the Arrow representation.
    """

    import pyarrow.parquet as pq

    table = pq.read_table(Path(parquet_path))
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    rows: list[dict[str, Any]] = []
    for i in range(table.num_rows):
        image = cols["image"][i] if "image" in cols else None
        image_bytes = image.get("bytes") if isinstance(image, dict) and "bytes" in image else None
        record = json.loads(cols["record"][i]) if "record" in cols else {}
        rows.append(
            {
                "id": str(cols["id"][i]) if "id" in cols else record.get("id"),
                "image_bytes": image_bytes,
                "image_path": record.get("image"),
                "record": record,
            }
        )
    return rows


def default_materialize_root(benchmark: str) -> Path:
    """Return the default cache root for :func:`materialize_benchmark`."""

    return data_path("eval_out", "markush", "benchmark_data", normalize_benchmark_name(benchmark))


def materialize_benchmark(
    benchmark: str,
    dest_root: str | Path | None = None,
    *,
    explicit_parquet: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Unpack the hosted parquet into a fast_eval-ready JSONL + image tree.

    Returns ``(eval_jsonl, image_root)``. Image refs in the emitted rows are
    ``images/<bench>/<split>/…`` relative to ``image_root`` (== ``dest_root``),
    exactly as ``glyph data prepare --markush`` writes them. Idempotent: an
    already-materialized JSONL with the expected row count and present images is
    reused unless ``force`` is set.
    """

    name = normalize_benchmark_name(benchmark)
    root = Path(dest_root).expanduser() if dest_root is not None else default_materialize_root(name)
    eval_jsonl = root / "benchmarks" / "markush" / name / f"eval_{name}.jsonl"

    rows = load_benchmark_records(resolve_benchmark_parquet(name, explicit=explicit_parquet))

    if not force and eval_jsonl.is_file():
        existing = sum(1 for line in eval_jsonl.read_text(encoding="utf-8").splitlines() if line)
        if existing == len(rows) and all(
            (root / row["image_path"]).is_file() for row in rows if row["image_path"]
        ):
            return eval_jsonl, root

    eval_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with eval_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            image_path = row["image_path"]
            image_bytes = row["image_bytes"]
            if image_path and image_bytes is not None:
                target = root / image_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(image_bytes)
            handle.write(json.dumps(row["record"], ensure_ascii=False) + "\n")
    return eval_jsonl, root


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Resolve/materialize the hosted MarkushGlyph eval benchmarks."
    )
    parser.add_argument("benchmark", help="Benchmark name (ip5m, m2s, uspto_markush).")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--parquet", action="store_true", help="Print the resolved predict parquet path."
    )
    action.add_argument(
        "--gt", action="store_true", help="Print the resolved ground-truth JSONL path."
    )
    action.add_argument(
        "--materialize",
        action="store_true",
        help="Materialize the eval JSONL + images; print '<eval_jsonl>\\t<image_root>'.",
    )
    parser.add_argument(
        "--dest-root", default=None, help="Materialize destination root (default under eval_out)."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Thin CLI mirror of the resolver/materializer library above."""

    args = _build_parser().parse_args(argv)
    if args.parquet:
        print(resolve_benchmark_parquet(args.benchmark))
    elif args.gt:
        print(resolve_benchmark_gt(args.benchmark))
    else:
        eval_jsonl, image_root = materialize_benchmark(args.benchmark, args.dest_root)
        print(f"{eval_jsonl}\t{image_root}")


if __name__ == "__main__":
    main()
