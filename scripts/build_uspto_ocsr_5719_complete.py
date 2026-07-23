#!/usr/bin/env python3
"""Assemble the "most complete" USPTO OCSR ground-truth label set.

The Hugging Face benchmark ``hheiden/USPTO_OCSR_benchmark`` ships 5719 image
rows, of which 15 carry a null/empty SMILES label. This builder joins those
5719 rows against the original MolScribe ``real.zip`` release
(``real/USPTO.csv``, columns ``image_id,file_path,SMILES``) to recover as many
of the 15 missing labels as possible, and emits a single versioned CSV that
downstream evaluation can join by ``image_id``.

Provenance of every label is recorded in the ``source_of_label`` column:

* ``hf_benchmark``      -- SMILES present in the HF benchmark (5704 rows).
* ``molscribe_real``    -- SMILES was null on HF and recovered from MolScribe
                           ``real/USPTO.csv`` (13 rows).
* ``none_unrecovered``  -- SMILES null on HF and absent/empty in MolScribe too
                           (2 rows).

Images are *not* copied into the CSV; they remain in the HF benchmark and are
loaded by ``glyph.ocsr.bench_image`` at evaluation time. This artifact is a
label sidecar keyed on ``image_id``.

The MolScribe lineage is intentional: ``hheiden/USPTO_OCSR_benchmark`` is a
re-hosting of the USPTO split used by MolScribe (Qian et al., 2023), so the
row ids line up 1:1 with ``real/USPTO.csv``.

Usage
-----
    python scripts/build_uspto_ocsr_5719_complete.py \
        --benchmark-parquet /path/to/train-00000-of-00001.parquet \
        --out               data/uspto_ocsr_5719_complete.csv

If ``--benchmark-parquet`` is omitted, the standard ``glyph.ocsr.eval``
resolver is used (env ``USPTO_PARQUET`` / ``GLYPH_DATA_ROOT`` / datasets
cache). The script fetches MolScribe ``real/USPTO.csv`` from its pinned source
archive into the output directory when needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyph.data.layout import STATUS_MATERIALIZED  # noqa: E402
from glyph.data.ocsr_sources import fetch_molscribe_real_uspto_csv  # noqa: E402
from glyph.ocsr.benchmark import build_uspto5719_benchmark  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-parquet",
        type=str,
        default=None,
        help="Path to the HF benchmark Parquet/Arrow snapshot (default: standard resolver).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "data" / "uspto_ocsr_5719_complete.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Optional JSON provenance summary path (default: <out>.provenance.json).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.out).expanduser()
    recovery_csv = out.parent / "molscribe_real_USPTO.csv"
    if not recovery_csv.is_file():
        outcome = fetch_molscribe_real_uspto_csv(recovery_csv, allow_large=True)
        if outcome.status != STATUS_MATERIALIZED:
            raise RuntimeError(outcome.detail)

    result = build_uspto5719_benchmark(
        benchmark_parquet=args.benchmark_parquet,
        molscribe_csv=recovery_csv,
        out=out,
        summary_out=args.summary_out,
    )

    print(f"[out] wrote {len(result.rows)} label rows -> {result.output_path}")
    print(f"[out] provenance -> {result.provenance_path}")
    print(json.dumps(result.provenance["source_counts"], indent=2))
    print(
        f"rdkit-scoreable rows: {result.provenance['rdkit_scoreable_rows']} "
        f"(unscoreable: {result.provenance['unscoreable_rows']})"
    )


if __name__ == "__main__":
    main()
