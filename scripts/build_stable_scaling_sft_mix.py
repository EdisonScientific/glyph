#!/usr/bin/env python3
"""Build a deterministic stable-scaling SFT mix from prepared partitions.

Thin CLI wrapper. The logic lives in :mod:`glyph.data.markush_mix`
(``build_mix`` and friends); this script only wires ``sys.argv`` to it, so the
two mirror each other 1:1. It is downstream of ``glyph markush data prepare-sft``: it
combines already-prepared SFT JSONL partitions, records per-row partition
provenance, and writes a composition manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyph.data.markush_mix import (  # noqa: E402
    EVAL_SOURCES,
    Candidate,
    PartitionSpec,
    add_partition_metadata,
    build_mix,
    build_parser,
    candidate_pool,
    counter_by,
    is_eval_row,
    load_jsonl,
    main,
    parse_name_path,
    parse_partition,
    relativize_image_ref,
    relativize_row_images,
    require_absolute_images,
    resolve_images,
    row_key,
    select_candidates,
    write_jsonl,
    write_manifest,
    write_report,
)

__all__ = [
    "EVAL_SOURCES",
    "Candidate",
    "PartitionSpec",
    "add_partition_metadata",
    "build_mix",
    "build_parser",
    "candidate_pool",
    "counter_by",
    "is_eval_row",
    "load_jsonl",
    "main",
    "parse_name_path",
    "parse_partition",
    "relativize_image_ref",
    "relativize_row_images",
    "require_absolute_images",
    "resolve_images",
    "row_key",
    "select_candidates",
    "write_jsonl",
    "write_manifest",
    "write_report",
]


if __name__ == "__main__":
    main()
