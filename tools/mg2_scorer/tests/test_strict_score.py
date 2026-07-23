"""Tests for the strict parsed-graph (ours) benchmark scorer.

The strict grader shares the DS4SD scorer's turnkey entry point
(``score_benchmark.sh``) but lives in :mod:`glyph.markush.eval.strict_score`.
Row adaptation and file loading are exercised without RDKit; the end-to-end
grading launch check is skipped when RDKit is not importable.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from glyph.markush.eval.strict_score import (
    build_grading_rows,
    load_ground_truth_rows,
    load_prediction_samples,
    strict_score,
)

_SAMPLES = [
    {"sample_id": "a", "prediction": "<markush><cxsmi><r>R1</r>C</cxsmi></markush>"},
    {"sample_id": "b", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"},
    {"sample_id": "c", "prediction": "<markush><cxsmi>CCO</cxsmi></markush>"},
]
_GT_ROWS = [
    {"id": "a", "gt_cxsmiles": "…", "gt_cxsmiles_opt": "<r>R1</r>C"},
    {"id": "b", "gt_cxsmiles": "…", "gt_cxsmiles_opt": "<r>R1</r>C"},
    {"id": "c", "gt_cxsmiles": None, "gt_cxsmiles_opt": None},
]


class BuildGradingRowsTests(unittest.TestCase):
    def test_join_preserves_order_and_none_gt(self) -> None:
        pred_rows, gt_rows = build_grading_rows(_SAMPLES, _GT_ROWS)
        self.assertEqual([r["id"] for r in pred_rows], ["a", "b", "c"])
        self.assertEqual([r["id"] for r in gt_rows], ["a", "b", "c"])
        # The DS4SD model-format GT lands under cxsmiles_opt, None retained.
        self.assertEqual(gt_rows[0]["cxsmiles_opt"], "<r>R1</r>C")
        self.assertIsNone(gt_rows[2]["cxsmiles_opt"])

    def test_gt_cxsmiles_opt_wins_over_baseline_cxsmiles_opt(self) -> None:
        # In DS4SD files, cxsmiles_opt is MG2's prediction, not the GT: it must
        # never be picked as the ground truth.
        rows = [{"id": "a", "cxsmiles_opt": "WRONG", "gt_cxsmiles_opt": "<r>R1</r>C"}]
        _, gt_rows = build_grading_rows([{"sample_id": "a", "prediction": "C"}], rows)
        self.assertEqual(gt_rows[0]["cxsmiles_opt"], "<r>R1</r>C")

    def test_missing_ground_truth_row_raises(self) -> None:
        with self.assertRaises(KeyError):
            build_grading_rows([{"sample_id": "z", "prediction": "C"}], _GT_ROWS)


class LoadPredictionSamplesTests(unittest.TestCase):
    def test_loads_samples_dict_list_and_jsonl(self) -> None:
        rows = [{"sample_id": "a", "prediction": "C"}]
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wrapped = tmp_path / "preds.json"
            wrapped.write_text(json.dumps({"config": {}, "samples": rows}), encoding="utf-8")
            bare = tmp_path / "bare.json"
            bare.write_text(json.dumps(rows), encoding="utf-8")
            lines = tmp_path / "preds.jsonl"
            lines.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

            self.assertEqual(load_prediction_samples(wrapped), rows)
            self.assertEqual(load_prediction_samples(bare), rows)
            self.assertEqual(load_prediction_samples(lines), rows)

    def test_loads_ground_truth_jsonl(self) -> None:
        rows = [{"id": "a", "gt_cxsmiles_opt": "<r>R1</r>C"}]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertEqual(load_ground_truth_rows(path), rows)


class StrictScoreLaunchCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import rdkit  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment guard
            raise unittest.SkipTest(f"RDKit is not importable: {exc}") from exc

    def test_exact_and_mismatch_and_none_gt(self) -> None:
        samples = [
            {"sample_id": "hit", "prediction": "<markush><cxsmi><r>R1</r>C</cxsmi></markush>"},
            {"sample_id": "miss", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"},
            {"sample_id": "none-gt", "prediction": "<markush><cxsmi>CCO</cxsmi></markush>"},
        ]
        gt_rows = [
            {"id": "hit", "gt_cxsmiles_opt": "<r>R1</r>C"},
            {"id": "miss", "gt_cxsmiles_opt": "<r>R1</r>C"},
            {"id": "none-gt", "gt_cxsmiles_opt": None},
        ]
        score, payload = strict_score(samples, gt_rows, benchmark="unit")
        self.assertEqual(score.total, 3)
        self.assertEqual(score.count, 1)
        self.assertEqual(score.invalid_gt_count, 1)
        self.assertAlmostEqual(score.percent, 100.0 / 3.0)
        self.assertIn("unit strict parsed-graph (ours) score: 1 / 3", score.headline())
        self.assertEqual(payload["protocol"], "ours")


if __name__ == "__main__":
    unittest.main()
