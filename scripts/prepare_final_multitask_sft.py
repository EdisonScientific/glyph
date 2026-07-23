#!/usr/bin/env python3
"""Prepare a final mixed OCSR + Markush SFT artifact.

Thin CLI wrapper. The logic lives in :mod:`glyph.data.markush_multitask`
(``main`` and friends); this script only wires ``sys.argv`` to it, so the two
mirror each other 1:1. It prepares a MolScribe OCSR component, mixes it with a
prepared Markush SFT JSONL by a requested OCSR policy, validates the result, and
writes an examples report plus a launch-command template. It does not launch
training.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyph.data.markush_multitask import (  # noqa: E402
    DEFAULT_MARKUSH_EVAL_REL,
    load_jsonl,
    main,
    materialize_no_extension_ocsr_eval_jsonls,
    parse_args,
    representative_examples,
    require_markush_only,
    sampling_policy,
    validate_examples,
    write_examples_report,
    write_pilot_template,
)

__all__ = [
    "DEFAULT_MARKUSH_EVAL_REL",
    "load_jsonl",
    "main",
    "materialize_no_extension_ocsr_eval_jsonls",
    "parse_args",
    "representative_examples",
    "require_markush_only",
    "sampling_policy",
    "validate_examples",
    "write_examples_report",
    "write_pilot_template",
]


if __name__ == "__main__":
    raise SystemExit(main())
