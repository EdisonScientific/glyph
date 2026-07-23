#!/usr/bin/env python3
"""Materialize OCSR-paper-inspired sources into unified OCSR JSONL.

Thin CLI wrapper. The logic lives in :mod:`glyph.data.markush_render`; this
script only re-exports its public surface and delegates to ``main``, so the two
mirror each other 1:1.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyph.data.markush_render import (  # noqa: E402
    RenderJob,
    RenderResult,
    _ensure_indigo_pythonpath,
    load_csv_smiles,
    load_jsonl,
    main,
    normalize_patent_row,
    parse_args,
    parse_count_path,
    pseudo_markush_smiles_reason,
    render_result_to_row,
    render_smiles_job,
    render_smiles_jobs,
    row_image,
    sample_items,
    smiles_from_csv_row,
    stable_render_name,
    valid_patent_rows,
    write_jsonl,
)

__all__ = [
    "RenderJob",
    "RenderResult",
    "_ensure_indigo_pythonpath",
    "load_csv_smiles",
    "load_jsonl",
    "main",
    "normalize_patent_row",
    "parse_args",
    "parse_count_path",
    "pseudo_markush_smiles_reason",
    "render_result_to_row",
    "render_smiles_job",
    "render_smiles_jobs",
    "row_image",
    "sample_items",
    "smiles_from_csv_row",
    "stable_render_name",
    "valid_patent_rows",
    "write_jsonl",
]


if __name__ == "__main__":
    raise SystemExit(main())
