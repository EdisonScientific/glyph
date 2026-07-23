#!/usr/bin/env python3
"""Push a local unified dataset artifact to HuggingFace Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def dataset_from_jsonl(paths: list[Path]):
    """Build a DatasetDict from one or more unified JSONL files."""

    from datasets import Dataset, DatasetDict

    by_split: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        for row in load_jsonl(path):
            by_split.setdefault(str(row.get("split") or "train"), []).append(row)
    return DatasetDict({split: Dataset.from_list(rows) for split, rows in by_split.items()})


def load_dataset_artifact(path: Path):
    """Load either a saved HF dataset directory or JSONL input."""

    from datasets import load_from_disk

    path = path.expanduser()
    if path.is_dir() and (path / "dataset_dict.json").exists():
        return load_from_disk(str(path))
    if path.is_dir():
        jsonl_paths = sorted(path.glob("*.jsonl")) + sorted(path.glob("*/unified.jsonl"))
        if jsonl_paths:
            return dataset_from_jsonl(jsonl_paths)
    if path.is_file() and path.suffix == ".jsonl":
        return dataset_from_jsonl([path])
    raise FileNotFoundError(f"Could not find a saved dataset or unified JSONL under {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push a unified dataset artifact to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", type=str, default="Upload unified Markush dataset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize the dataset without pushing",
    )
    return parser


def push_dataset_artifact(
    *,
    dataset_path: Path,
    repo: str,
    config_name: str | None = None,
    private: bool = False,
    commit_message: str = "Upload unified Markush dataset",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load a local dataset artifact, optionally push it, and return a summary."""

    dataset = load_dataset_artifact(dataset_path)
    summary = {split: len(dataset[split]) for split in dataset}
    if dry_run:
        return {"repo": repo, "splits": summary, "dry_run": True}
    dataset.push_to_hub(
        repo,
        config_name=config_name,
        private=private,
        commit_message=commit_message,
    )
    return {"repo": repo, "splits": summary, "pushed": True}


def main() -> None:
    args = build_parser().parse_args()
    report = push_dataset_artifact(
        dataset_path=args.dataset,
        repo=args.repo,
        config_name=args.config_name,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
