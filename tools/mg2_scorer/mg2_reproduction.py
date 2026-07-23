#!/usr/bin/env python3
"""Compatibility and reporting helpers for the published MG2 reproduction.

This module does not perform OCR, inference, or chemical grading. Those steps
remain delegated to DS4SD's published ChemicalOCR, ``markushgrapher.eval``, and
the release's pinned graders. The helpers only bridge on-disk schemas, adapt
MG2's JSONL to the scorer input envelope, compare artifacts, and print a rollup.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PREDICTION_FIELDS = ("id", "cxsmiles", "cxsmiles_opt", "gt_cxsmiles", "gt_cxsmiles_opt")


def _replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise ValueError(f"unexpected pinned upstream source in {path}: {old!r}")
    return text.replace(old, new, 1)


def patch_upstream(checkout: str | Path) -> list[Path]:
    """Apply the two narrow compatibility patches to a pinned MG2 checkout."""

    root = Path(checkout)
    utils_path = root / "markushgrapher/core/common/utils.py"
    utils_text = utils_path.read_text(encoding="utf-8")
    utils_text = _replace_once(
        utils_text,
        "from torch._utils import _accumulate\n",
        "try:\n"
        "    from torch._utils import _accumulate\n"
        "except ImportError:\n"
        "    from itertools import accumulate as _accumulate\n",
        path=utils_path,
    )
    utils_path.write_text(utils_text, encoding="utf-8")

    prepare_path = root / "scripts/dataset/prepare_hf_dataset.py"
    prepare_text = prepare_path.read_text(encoding="utf-8")
    replacements = (
        (
            "def load_source_dataset(dataset_name_or_path, config_name=None, split=None):",
            "def load_source_dataset("
            "dataset_name_or_path, config_name=None, split=None, revision=None):",
        ),
        (
            "return load_dataset(dataset_name_or_path, config_name, split=split)",
            "return load_dataset("
            "dataset_name_or_path, config_name, split=split, revision=revision)",
        ),
        (
            "dataset = load_dataset(dataset_name_or_path, config_name)",
            "dataset = load_dataset(dataset_name_or_path, config_name, revision=revision)",
        ),
        (
            'parser.add_argument("--split", default="test", help="Source split to load.")',
            'parser.add_argument("--split", default="test", help="Source split to load.")\n'
            '    parser.add_argument("--revision", default=None, help="Optional HF revision pin.")',
        ),
        (
            "dataset = load_source_dataset(args.dataset, args.config, args.split)",
            "dataset = load_source_dataset(args.dataset, args.config, args.split, args.revision)",
        ),
    )
    for old, new in replacements:
        prepare_text = _replace_once(prepare_text, old, new, path=prepare_path)
    prepare_path.write_text(prepare_text, encoding="utf-8")
    return [utils_path, prepare_path]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL objects from *path*."""

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON objects as UTF-8 JSONL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def legacy_feature_metadata(value: Any) -> Any:
    """Translate datasets-5 ``List`` metadata to datasets-2 ``Sequence`` metadata.

    Hugging Face datasets 5 serializes a list of structs as a ``List`` feature.
    datasets 2.14, which the pinned MG2 environment requires, represents the
    same Arrow storage as a one-element feature list. Lists of scalar values use
    the older explicit ``Sequence`` record. This changes metadata only.
    """

    if isinstance(value, list):
        return [legacy_feature_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("_type") == "List":
        feature = legacy_feature_metadata(value["feature"])
        if isinstance(feature, dict) and "_type" not in feature:
            return [feature]
        return {"feature": feature, "_type": "Sequence"}
    return {key: legacy_feature_metadata(item) for key, item in value.items()}


def convert_dataset(source: str | Path, output: str | Path, *, split: str = "test") -> int:
    """Make a datasets-5 OCR DatasetDict readable by pinned datasets 2.14.

    This command must run with the ChemicalOCR environment because that is the
    environment which can deserialize the source ``List`` feature. It copies the
    dataset through the datasets API, then rewrites both ``dataset_info.json``
    and the matching Arrow schema metadata to the equivalent legacy form.
    """

    try:
        import pyarrow as pa
        from datasets import Dataset, DatasetDict, load_from_disk
    except ImportError as exc:  # pragma: no cover - environment error path
        raise RuntimeError("convert-dataset requires datasets and pyarrow") from exc

    source_path = Path(source)
    output_path = Path(output)
    loaded = load_from_disk(str(source_path))
    dataset = loaded[split] if hasattr(loaded, "keys") else loaded
    if output_path.exists():
        shutil.rmtree(output_path)

    # from_generator avoids preserving source cache paths and gives the output
    # its own Arrow file before metadata is made legacy-compatible.
    copied = Dataset.from_generator(lambda: iter(dataset), features=dataset.features)
    DatasetDict({split: copied}).save_to_disk(str(output_path))

    info_path = output_path / split / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"] = legacy_feature_metadata(info["features"])
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    arrow_files = list((output_path / split).glob("*.arrow"))
    if not arrow_files:
        raise ValueError(f"no Arrow shards found under {output_path / split}")
    for arrow_path in arrow_files:
        with pa.memory_map(str(arrow_path), "r") as source_file:
            table = pa.ipc.open_stream(source_file).read_all()
        metadata = dict(table.schema.metadata or {})
        huggingface = json.loads(metadata[b"huggingface"])
        huggingface["info"]["features"] = info["features"]
        metadata[b"huggingface"] = json.dumps(huggingface, separators=(",", ":")).encode()
        table = table.replace_schema_metadata(metadata)
        temporary = arrow_path.with_suffix(".arrow.tmp")
        with (
            pa.OSFile(str(temporary), "wb") as sink,
            pa.ipc.new_stream(sink, table.schema) as writer,
        ):
            writer.write_table(table)
        temporary.replace(arrow_path)
    return len(dataset)


def adapt_for_scoring(source: str | Path, output: str | Path) -> int:
    """Wrap MG2 optimized predictions in the envelope expected by its scorer.

    ``markushgrapher.eval`` writes both standard and optimized CXSMILES. The
    official public scorer consumes the model's ``<markush><cxsmi>...`` text and
    applies MG2's own ``convert_opt_to_out`` routine. Supplying that envelope
    ensures scoring follows the same path instead of adding a local conversion.
    """

    rows = read_jsonl(source)
    adapted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        optimized = row.get("cxsmiles_opt")
        if sample_id is None:
            raise ValueError(f"prediction row {index} has no id")
        if not isinstance(optimized, str) or not optimized:
            prediction = None
        else:
            prediction = f"<markush><cxsmi>{optimized}</cxsmi><stable></stable></markush>"
        adapted.append(
            {"id": str(sample_id), "sample_id": str(sample_id), "prediction": prediction}
        )
    write_jsonl(output, adapted)
    return len(rows)


def compare_predictions(
    fresh_path: str | Path,
    cached_path: str | Path,
    *,
    fields: Sequence[str] = PREDICTION_FIELDS,
) -> dict[str, Any]:
    """Compare aligned prediction JSONLs row-for-row and field-for-field."""

    fresh = read_jsonl(fresh_path)
    cached = read_jsonl(cached_path)
    field_matches = dict.fromkeys(fields, 0)
    differing_rows: list[dict[str, Any]] = []
    aligned = min(len(fresh), len(cached))
    order_match = [row.get("id") for row in fresh] == [row.get("id") for row in cached]

    for index, (fresh_row, cached_row) in enumerate(zip(fresh, cached, strict=False)):
        differing_fields: list[str] = []
        for field in fields:
            if fresh_row.get(field) == cached_row.get(field):
                field_matches[field] += 1
            else:
                differing_fields.append(field)
        if differing_fields:
            differing_rows.append(
                {
                    "index": index,
                    "fresh_id": fresh_row.get("id"),
                    "cached_id": cached_row.get("id"),
                    "fields": differing_fields,
                }
            )

    return {
        "fresh_path": str(fresh_path),
        "cached_path": str(cached_path),
        "fresh_rows": len(fresh),
        "cached_rows": len(cached),
        "aligned_rows": aligned,
        "id_order_match": order_match,
        "exact_row_matches": aligned - len(differing_rows),
        "field_matches": field_matches,
        "differing_row_count": len(differing_rows),
        "differing_rows": differing_rows,
    }


def write_comparison(fresh: str | Path, cached: str | Path, output: str | Path) -> dict[str, Any]:
    report = compare_predictions(fresh, cached)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _score_counts(official_path: Path, strict_path: Path) -> dict[str, Any]:
    official = json.loads(official_path.read_text(encoding="utf-8"))["public_mg2"]["exact_counts"]
    strict = json.loads(strict_path.read_text(encoding="utf-8"))["strict"]
    return {
        "official_count": official["cxsmi_equality_count"],
        "official_total": official["size"],
        "strict_count": strict["count"],
        "strict_total": strict["total"],
    }


def summarize(run_root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    """Load three completed runs, print Table 3 counts, and optionally write JSON."""

    root = Path(run_root)
    rows: dict[str, Any] = {}
    for benchmark in ("ip5m", "m2s", "uspto_markush"):
        bench = root / benchmark
        score = _score_counts(bench / "scores.json", bench / "scores_strict.json")
        comparison = json.loads((bench / "comparison.json").read_text(encoding="utf-8"))
        rows[benchmark] = {**score, "comparison": comparison}

    print("MarkushGrapher-2 Table 3 reproduction (full denominators)")
    print("benchmark       official DS4SD        strict parsed graph       fresh vs cached")
    for benchmark, row in rows.items():
        official_pct = 100 * row["official_count"] / row["official_total"]
        strict_pct = 100 * row["strict_count"] / row["strict_total"]
        comparison = row["comparison"]
        print(
            f"{benchmark:<16}"
            f"{row['official_count']:>3} / {row['official_total']:<3} = {official_pct:6.2f}%    "
            f"{row['strict_count']:>3} / {row['strict_total']:<3} = {strict_pct:6.2f}%    "
            f"{comparison['exact_row_matches']} / {comparison['aligned_rows']} exact rows"
        )
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"benchmarks": rows}, indent=2) + "\n", encoding="utf-8")
    return {"benchmarks": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    patch = commands.add_parser("patch-upstream", help="apply pinned-runtime compatibility patches")
    patch.add_argument("--checkout", type=Path, required=True)

    convert = commands.add_parser(
        "convert-dataset", help="rewrite datasets-5 metadata for datasets 2"
    )
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--split", default="test")

    adapt = commands.add_parser(
        "adapt-for-scoring", help="adapt MG2 JSONL to official scorer input"
    )
    adapt.add_argument("--input", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare", help="compare fresh and cached MG2 JSONL rows")
    compare.add_argument("--fresh", type=Path, required=True)
    compare.add_argument("--cached", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    summary = commands.add_parser("summarize", help="print all three dual scores and comparisons")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "patch-upstream":
        paths = patch_upstream(args.checkout)
        print("patched " + ", ".join(str(path) for path in paths))
    elif args.command == "convert-dataset":
        print(f"converted {convert_dataset(args.input, args.output, split=args.split)} rows")
    elif args.command == "adapt-for-scoring":
        print(f"adapted {adapt_for_scoring(args.input, args.output)} rows")
    elif args.command == "compare":
        report = write_comparison(args.fresh, args.cached, args.output)
        print(
            f"fresh/cache exact rows: {report['exact_row_matches']} / {report['aligned_rows']}; "
            f"id order match: {report['id_order_match']}"
        )
    elif args.command == "summarize":
        summarize(args.run_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
