#!/usr/bin/env python3
"""Convert existing local Markush JSONL/manifests to the unified schema."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyph.markush.eval import common


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load JSON list or JSONL records."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        return [dict(row) for row in data]
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _first_text_message(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _image_path(record: dict[str, Any]) -> str | None:
    for key in ("image", "image_path", "page_path", "png_path", "rendered_path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict) and value.get("path"):
            return str(value["path"])
    images = record.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    return None


def _assistant_target(record: dict[str, Any]) -> str:
    return (
        _first_text_message(record, "assistant")
        or str(record.get("annotation") or "")
        or str(record.get("target") or "")
    )


def _description_from_user(record: dict[str, Any]) -> str | None:
    user = _first_text_message(record, "user")
    marker = "Patent text:"
    if marker not in user:
        return record.get("description")
    after = user.split(marker, 1)[1]
    before_instruction = after.split("Extract the", 1)[0]
    return before_instruction.strip() or record.get("description")


def normalize_existing_record(
    record: dict[str, Any],
    *,
    source: str,
    split: str,
    task: str,
    image_has_text: bool,
    index: int,
) -> dict[str, Any]:
    """Convert one local row to the unified schema."""

    sample_id = str(record.get("id") or record.get("sample_id") or f"{source}_{index:06d}")
    target = _assistant_target(record)
    cxsmi, stable = common.extract_cxsmiles_and_stable(target, require_wrapper=False)

    if task == "ocsr":
        smiles = record.get("smiles") or target or record.get("cxsmiles_opt")
        cxsmiles_opt = None
        annotation = None
    else:
        smiles = record.get("smiles")
        cxsmiles_opt = record.get("cxsmiles_opt") or cxsmi or record.get("cxsmiles_input")
        annotation = target if "<markush" in target.lower() else record.get("annotation")
        if not annotation and cxsmiles_opt:
            stable_section = f"<stable>{stable}</stable>" if stable else "<stable></stable>"
            annotation = f"<markush><cxsmi>{cxsmiles_opt}</cxsmi>{stable_section}</markush>"

    return {
        "id": sample_id,
        "source": source,
        "split": split,
        "image": _image_path(record),
        "image_has_text": bool(record.get("image_has_text", image_has_text)),
        "task": task,
        "cxsmiles_opt": cxsmiles_opt,
        "smiles": smiles,
        "annotation": annotation,
        "cxsmiles_standard": record.get("cxsmiles") or record.get("cxsmiles_standard"),
        "description": _description_from_user(record),
        "cells": record.get("cells") or record.get("page_cells"),
        "patent_id": record.get("patent_id"),
        "patent_office": record.get("patent_office"),
        "metadata": {
            "converted_from": "existing_local",
            "stable_raw": stable,
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_hf_dataset(rows: list[dict[str, Any]], path: Path) -> None:
    from datasets import Dataset, DatasetDict

    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row.get("split") or "train"), []).append(row)
    DatasetDict({split: Dataset.from_list(items) for split, items in by_split.items()}).save_to_disk(
        str(path)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert existing JSONL/render manifests to the unified schema",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=str, default="pipeline_generated")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--task", choices=("markush", "ocsr"), default="markush")
    parser.add_argument(
        "--image-has-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether images contain patent prose text when rows do not specify image_has_text",
    )
    parser.add_argument("--save-hf", type=Path, default=None)
    return parser


def convert_existing_files(
    *,
    inputs: list[Path],
    output: Path,
    source: str = "pipeline_generated",
    split: str = "train",
    task: str = "markush",
    image_has_text: bool = True,
    save_hf: Path | None = None,
) -> dict[str, Any]:
    """Convert existing JSON/JSONL files and return a small summary."""

    rows: list[dict[str, Any]] = []
    for path in inputs:
        for record in load_json_records(path.expanduser()):
            rows.append(
                normalize_existing_record(
                    record,
                    source=source,
                    split=split,
                    task=task,
                    image_has_text=image_has_text,
                    index=len(rows),
                )
            )
    write_jsonl(rows, output)
    if save_hf:
        save_hf_dataset(rows, save_hf)
    return {"input_files": len(inputs), "written": len(rows)}


def main() -> None:
    args = build_parser().parse_args()
    report = convert_existing_files(
        inputs=args.input,
        output=args.output,
        source=args.source,
        split=args.split,
        task=args.task,
        image_has_text=args.image_has_text,
        save_hf=args.save_hf,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
