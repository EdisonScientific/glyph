"""Release data-preparation pipeline (``glyph data prepare``).

Pulls the public OCSR and Markush sources, preprocesses them, and lays them out
under ``$GLYPH_DATA_ROOT``. The library holds all logic
(:mod:`glyph.data.ocsr` orchestrates the OCSR path; :mod:`glyph.data.markush`
orchestrates the Markush path); :mod:`glyph.data.cli` is a thin Typer wrapper.

This package top level intentionally avoids importing the Markush multitask
stack, so ``import glyph.data`` and the core OCSR prepare path stay lightweight;
:func:`glyph.data.markush.prepare_markush` is imported lazily by the CLI.
"""

from __future__ import annotations

from glyph.data.layout import (
    MARKUSH_BENCHMARK_NAMES,
    MARKUSH_BENCHMARK_PROVENANCE,
    MARKUSH_DATASET_NAMES,
    MARKUSH_MULTITASK_MIX_PROVENANCE,
    MARKUSH_PROVENANCE,
    OCSR_DATASET_NAMES,
    OCSR_PROVENANCE,
    PrepareReport,
    PrepareStep,
    markush_benchmark_dir,
    markush_benchmarks_dir,
    markush_dataset_dir,
    markush_datasets_dir,
    markush_multitask_mix_path,
    markush_prepared_dir,
    ocsr_benchmarks_dir,
    ocsr_dataset_dir,
    ocsr_datasets_dir,
    uspto_5719_benchmark_path,
)
from glyph.data.ocsr import prepare_ocsr

__all__ = [
    "MARKUSH_BENCHMARK_NAMES",
    "MARKUSH_BENCHMARK_PROVENANCE",
    "MARKUSH_DATASET_NAMES",
    "MARKUSH_MULTITASK_MIX_PROVENANCE",
    "MARKUSH_PROVENANCE",
    "OCSR_DATASET_NAMES",
    "OCSR_PROVENANCE",
    "PrepareReport",
    "PrepareStep",
    "markush_benchmark_dir",
    "markush_benchmarks_dir",
    "markush_dataset_dir",
    "markush_datasets_dir",
    "markush_multitask_mix_path",
    "markush_prepared_dir",
    "ocsr_benchmarks_dir",
    "ocsr_dataset_dir",
    "ocsr_datasets_dir",
    "prepare_ocsr",
    "uspto_5719_benchmark_path",
]
