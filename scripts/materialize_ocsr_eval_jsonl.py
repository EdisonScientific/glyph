#!/usr/bin/env python3
"""Materialize OCSR benchmark rows into Markush-compatible eval JSONL.

Thin CLI wrapper. The logic lives in
:mod:`glyph.data.ocsr_eval_jsonl` (``materialize_ocsr_eval``); this script only
parses args and calls it, so the two mirror each other 1:1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyph.data.ocsr_eval_jsonl import (  # noqa: E402
    MaterializeStats,
    materialize_ocsr_eval,
)

__all__ = ["MaterializeStats", "build_parser", "main", "materialize_ocsr_eval"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-format", choices=("csv", "parquet"), required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest-json", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = materialize_ocsr_eval(
        input_path=args.input,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        source_format=args.source_format,
        image_root=args.image_root,
        limit=args.limit,
    )
    payload = asdict(stats)
    manifest_path = args.manifest_json or (
        args.output_dir / f"manifest_ocsr_{args.dataset_name}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
