"""Build the complete 5,719-row USPTO OCSR benchmark label sidecar.

The Hugging Face benchmark ``hheiden/USPTO_OCSR_benchmark`` contains 5,719
images, but only 5,704 rows have labels. The original MolScribe
``real/USPTO.csv`` release can recover 13 of the 15 missing labels. This module
assembles those sources and records the provenance of every label.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from glyph.paths import data_path

from .eval import USPTO_BENCHMARK_TOTAL_ROWS, canonical_smiles, load_uspto_benchmark

SOURCE_HF = "hf_benchmark"
SOURCE_MOLSCRIBE = "molscribe_real"
SOURCE_NONE = "none_unrecovered"
COMPLETE_BENCHMARK_FILENAME = "uspto_ocsr_5719_complete.csv"


def default_uspto5719_benchmark_path() -> Path:
    """Return the benchmark sidecar path below the configured Glyph data root."""

    return data_path(COMPLETE_BENCHMARK_FILENAME)


@dataclass
class BenchmarkBuildResult:
    """Rows, provenance, and any files produced by a benchmark build."""

    rows: list[dict[str, str]]
    provenance: dict[str, Any]
    output_path: Path | None = None
    provenance_path: Path | None = None


def _has_smiles(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def load_molscribe_labels(csv_path: Path | str) -> dict[str, str]:
    """Map ``image_id -> SMILES`` from MolScribe ``real/USPTO.csv``."""

    path = Path(csv_path).expanduser()
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        id_key = next((key for key in reader.fieldnames or [] if key.lower() == "image_id"), None)
        smi_key = next((key for key in reader.fieldnames or [] if key.lower() == "smiles"), None)
        if id_key is None or smi_key is None:
            raise ValueError(
                f"{path} must have image_id and SMILES columns; got {reader.fieldnames}"
            )
        for row in reader:
            image_id = str(row[id_key]).strip()
            smiles = row.get(smi_key)
            if image_id and _has_smiles(smiles):
                labels[image_id] = smiles.strip()
    return labels


def build_uspto5719_rows(
    benchmark_parquet: Path | str | None = None,
    molscribe_csv: Path | str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return merged benchmark label rows and their provenance summary."""

    parquet_path = Path(benchmark_parquet).expanduser() if benchmark_parquet else None
    bench = load_uspto_benchmark(parquet_path)
    if len(bench) != USPTO_BENCHMARK_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {USPTO_BENCHMARK_TOTAL_ROWS} benchmark rows, got {len(bench)}."
        )

    molscribe = load_molscribe_labels(molscribe_csv) if molscribe_csv else {}

    rows: list[dict[str, str]] = []
    recovered: list[str] = []
    unrecovered: list[str] = []
    unsanitizable_recovered: list[str] = []
    n_hf = 0

    for row in bench:
        image_id = str(row["id"])
        smiles = row["smiles"]
        if _has_smiles(smiles):
            rows.append(
                {
                    "image_id": image_id,
                    "smiles": smiles.strip(),
                    "source_of_label": SOURCE_HF,
                }
            )
            n_hf += 1
            continue

        recovered_smiles = molscribe.get(image_id)
        if _has_smiles(recovered_smiles):
            rows.append(
                {
                    "image_id": image_id,
                    "smiles": recovered_smiles,
                    "source_of_label": SOURCE_MOLSCRIBE,
                }
            )
            recovered.append(image_id)
            if canonical_smiles(recovered_smiles) is None:
                unsanitizable_recovered.append(image_id)
        else:
            rows.append(
                {
                    "image_id": image_id,
                    "smiles": "",
                    "source_of_label": SOURCE_NONE,
                }
            )
            unrecovered.append(image_id)

    scoreable = sum(1 for row in rows if canonical_smiles(row["smiles"]) is not None)
    provenance: dict[str, Any] = {
        "total_rows": len(rows),
        "source_counts": {
            SOURCE_HF: n_hf,
            SOURCE_MOLSCRIBE: len(recovered),
            SOURCE_NONE: len(unrecovered),
        },
        "recovered_from_molscribe": recovered,
        "unrecovered_null": unrecovered,
        "recovered_but_unsanitizable": unsanitizable_recovered,
        "rdkit_scoreable_rows": scoreable,
        "unscoreable_rows": len(rows) - scoreable,
    }
    return rows, provenance


def write_uspto5719_benchmark(
    rows: list[dict[str, str]],
    provenance: dict[str, Any],
    out: Path | str,
    summary_out: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write benchmark rows and their JSON provenance sidecar."""

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "smiles", "source_of_label"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = (
        Path(summary_out).expanduser()
        if summary_out
        else out_path.with_suffix(out_path.suffix + ".provenance.json")
    )
    summary_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return out_path, summary_path


def build_uspto5719_benchmark(
    benchmark_parquet: Path | str | None = None,
    molscribe_csv: Path | str | None = None,
    out: Path | str | None = None,
    summary_out: Path | str | None = None,
) -> BenchmarkBuildResult:
    """Build the complete USPTO benchmark, optionally writing it to disk.

    Args:
        benchmark_parquet: HF benchmark Parquet/Arrow snapshot. When omitted,
            :func:`glyph.ocsr.eval.load_uspto_benchmark` uses its standard resolver.
        molscribe_csv: Optional MolScribe ``real/USPTO.csv`` used to recover null labels.
        out: Optional output CSV path. If omitted, rows are returned in memory only.
        summary_out: Optional provenance JSON path. Defaults to
            ``<out>.provenance.json`` when ``out`` is provided.
    """

    rows, provenance = build_uspto5719_rows(benchmark_parquet, molscribe_csv)
    output_path: Path | None = None
    provenance_path: Path | None = None
    if out is not None:
        output_path, provenance_path = write_uspto5719_benchmark(
            rows,
            provenance,
            out,
            summary_out,
        )
    elif summary_out is not None:
        provenance_path = Path(summary_out).expanduser()
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    return BenchmarkBuildResult(
        rows=rows,
        provenance=provenance,
        output_path=output_path,
        provenance_path=provenance_path,
    )
