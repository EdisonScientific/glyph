from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "mg2_reproduction.py"
_SPEC = importlib.util.spec_from_file_location("mg2_reproduction", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mg2_reproduction = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mg2_reproduction)

adapt_for_scoring = mg2_reproduction.adapt_for_scoring
compare_predictions = mg2_reproduction.compare_predictions
legacy_feature_metadata = mg2_reproduction.legacy_feature_metadata
patch_upstream = mg2_reproduction.patch_upstream
summarize = mg2_reproduction.summarize


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_patch_upstream_is_idempotent(tmp_path: Path) -> None:
    checkout = tmp_path / "MarkushGrapher"
    utils = checkout / "markushgrapher/core/common/utils.py"
    prepare = checkout / "scripts/dataset/prepare_hf_dataset.py"
    utils.parent.mkdir(parents=True)
    prepare.parent.mkdir(parents=True)
    utils.write_text("from torch._utils import _accumulate\n", encoding="utf-8")
    prepare.write_text(
        "def load_source_dataset(dataset_name_or_path, config_name=None, split=None):\n"
        "    if split:\n"
        "        return load_dataset(dataset_name_or_path, config_name, split=split)\n"
        "    dataset = load_dataset(dataset_name_or_path, config_name)\n"
        '    parser.add_argument("--split", default="test", help="Source split to load.")\n'
        "    dataset = load_source_dataset(args.dataset, args.config, args.split)\n",
        encoding="utf-8",
    )

    assert patch_upstream(checkout) == [utils, prepare]
    first_utils = utils.read_text(encoding="utf-8")
    first_prepare = prepare.read_text(encoding="utf-8")
    assert patch_upstream(checkout) == [utils, prepare]

    assert utils.read_text(encoding="utf-8") == first_utils
    assert prepare.read_text(encoding="utf-8") == first_prepare
    assert "from itertools import accumulate as _accumulate" in first_utils
    assert "revision=revision" in first_prepare
    assert 'parser.add_argument("--revision"' in first_prepare


def test_legacy_feature_metadata_converts_nested_lists() -> None:
    modern = {
        "id": {"dtype": "string", "_type": "Value"},
        "cells": {
            "feature": {
                "bbox": {
                    "feature": {"dtype": "float64", "_type": "Value"},
                    "_type": "List",
                },
                "text": {"dtype": "string", "_type": "Value"},
            },
            "_type": "List",
        },
    }

    legacy = legacy_feature_metadata(modern)

    assert legacy["cells"] == [
        {
            "bbox": {
                "feature": {"dtype": "float64", "_type": "Value"},
                "_type": "Sequence",
            },
            "text": {"dtype": "string", "_type": "Value"},
        }
    ]
    assert legacy["id"] == modern["id"]


def test_adapt_for_scoring_wraps_optimized_prediction(tmp_path: Path) -> None:
    source = tmp_path / "predictions.jsonl"
    output = tmp_path / "scorer.jsonl"
    _write_jsonl(
        source,
        [
            {"id": "a", "cxsmiles_opt": "<r>R1</r>C"},
            {"id": "b", "cxsmiles_opt": None},
        ],
    )

    assert adapt_for_scoring(source, output) == 2
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "id": "a",
            "sample_id": "a",
            "prediction": ("<markush><cxsmi><r>R1</r>C</cxsmi><stable></stable></markush>"),
        },
        {"id": "b", "sample_id": "b", "prediction": None},
    ]


def test_compare_predictions_reports_order_rows_and_fields(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.jsonl"
    cached = tmp_path / "cached.jsonl"
    _write_jsonl(
        fresh,
        [
            {"id": "a", "cxsmiles": "C", "cxsmiles_opt": "C", "gt_cxsmiles": "C"},
            {"id": "b", "cxsmiles": "N", "cxsmiles_opt": "N", "gt_cxsmiles": "N"},
        ],
    )
    _write_jsonl(
        cached,
        [
            {"id": "a", "cxsmiles": "C", "cxsmiles_opt": "C", "gt_cxsmiles": "C"},
            {"id": "b", "cxsmiles": "O", "cxsmiles_opt": "O", "gt_cxsmiles": "N"},
        ],
    )

    report = compare_predictions(fresh, cached)

    assert report["id_order_match"] is True
    assert report["exact_row_matches"] == 1
    assert report["differing_row_count"] == 1
    assert report["field_matches"]["id"] == 2
    assert report["field_matches"]["cxsmiles"] == 1
    assert report["differing_rows"] == [
        {"index": 1, "fresh_id": "b", "cached_id": "b", "fields": ["cxsmiles", "cxsmiles_opt"]}
    ]


def test_compare_predictions_detects_id_order_mismatch(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.jsonl"
    cached = tmp_path / "cached.jsonl"
    _write_jsonl(fresh, [{"id": "a"}, {"id": "b"}])
    _write_jsonl(cached, [{"id": "b"}, {"id": "a"}])

    report = compare_predictions(fresh, cached)

    assert report["id_order_match"] is False
    assert report["exact_row_matches"] == 0


def test_summarize_prints_three_dual_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for index, benchmark in enumerate(("ip5m", "m2s", "uspto_markush"), start=1):
        run = tmp_path / benchmark
        run.mkdir()
        (run / "scores.json").write_text(
            json.dumps(
                {"public_mg2": {"exact_counts": {"cxsmi_equality_count": index, "size": 10}}}
            ),
            encoding="utf-8",
        )
        (run / "scores_strict.json").write_text(
            json.dumps({"strict": {"count": index + 1, "total": 10}}),
            encoding="utf-8",
        )
        (run / "comparison.json").write_text(
            json.dumps({"exact_row_matches": 8, "aligned_rows": 10}), encoding="utf-8"
        )

    output = tmp_path / "summary.json"
    result = summarize(tmp_path, output)

    printed = capsys.readouterr().out
    assert "ip5m" in printed
    assert "m2s" in printed
    assert "uspto_markush" in printed
    assert "1 / 10" in printed
    assert result["benchmarks"]["m2s"]["strict_count"] == 3
    assert json.loads(output.read_text(encoding="utf-8")) == result
