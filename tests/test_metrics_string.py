#!/usr/bin/env python3
"""Tests for glyph.markush.eval.metrics_string.

Audits:
- parsing of Markush XML / lenient SMILES
- r-tag stripping and label extraction
- exact-match, R-group set, extension, stable-table metrics
- backbone InChIKey comparison (RDKit)
- string similarity (Levenshtein + long-string fallback)
- aggregate compute_metrics / evaluate_predictions API
"""

from __future__ import annotations

import pytest

from glyph.markush.eval.metrics_string import (
    _RDKIT_AVAILABLE,
    MarkushPrediction,
    SampleResult,
    _extract_cxsmiles_and_stable,
    _extract_r_labels,
    _parse_stable,
    _split_cxsmiles,
    _string_similarity,
    _strip_r_tags,
    compute_metrics,
    evaluate_predictions,
    evaluate_sample,
    metric_backbone_match,
    metric_extension_match,
    metric_markush_em,
    metric_r_accuracy,
    metric_r_f1,
    metric_r_precision,
    metric_stable_accuracy,
    metric_stable_exact_match,
    metric_string_em,
    parse_ground_truth_entry,
    parse_markush_output,
)

# ----------------------------------------------------------------------------
# Helpers used in this file only (no shared fixtures with other test files)
# ----------------------------------------------------------------------------


def _mk(cxsmiles: str = "", stable_raw: str = "") -> MarkushPrediction:
    """Build a MarkushPrediction from raw cxsmiles_opt + stable for metric tests."""
    raw = f"<markush><cxsmi>{cxsmiles}</cxsmi><stable>{stable_raw}</stable></markush>"
    return parse_markush_output(raw, strict=True)


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------


class TestExtractCxsmilesAndStable:
    def test_full_markush_with_stable(self):
        raw = "<markush><cxsmi>CC</cxsmi><stable>R0:a</stable></markush>"
        cx, st = _extract_cxsmiles_and_stable(raw)
        assert cx == "CC"
        assert st == "R0:a"

    def test_markush_without_stable(self):
        raw = "<markush><cxsmi>CC</cxsmi></markush>"
        cx, st = _extract_cxsmiles_and_stable(raw)
        assert cx == "CC"
        assert st == ""

    def test_no_markush_returns_empty(self):
        # Without <markush> wrapper the function returns "", "" — required
        # for downstream lenient-mode parsing to kick in.
        cx, st = _extract_cxsmiles_and_stable("<cxsmi>CC</cxsmi>")
        assert cx == ""
        assert st == ""

    def test_empty_input(self):
        assert _extract_cxsmiles_and_stable("") == ("", "")

    def test_whitespace_inside_cxsmi_is_stripped(self):
        raw = "<markush><cxsmi>   CC   </cxsmi></markush>"
        cx, _ = _extract_cxsmiles_and_stable(raw)
        assert cx == "CC"


class TestStripRTags:
    def test_single_r(self):
        assert _strip_r_tags("<r>R1</r>C=O") == "*C=O"

    def test_multiple_r(self):
        assert _strip_r_tags("<r>R1</r>C<r>R2</r>O") == "*C*O"

    def test_no_r(self):
        assert _strip_r_tags("C=O") == "C=O"

    def test_r_with_arbitrary_label(self):
        # Multi-character labels still produce a single * each.
        assert _strip_r_tags("<r>R1a</r>C") == "*C"


class TestExtractRLabels:
    def test_ordered_extraction(self):
        # Returns ordered list (used by parse_markush_output for occurrence-aware metrics).
        assert _extract_r_labels("<r>R1</r>C<r>R2</r>") == ["R1", "R2"]

    def test_duplicate_labels_preserved_in_list(self):
        assert _extract_r_labels("<r>R1</r>C<r>R1</r>") == ["R1", "R1"]

    def test_empty_string(self):
        assert _extract_r_labels("") == []


class TestSplitCxsmiles:
    def test_with_extension(self):
        core, ext = _split_cxsmiles("CC |Sg:n:1,2:ht|")
        assert core == "CC"
        assert ext == "Sg:n:1,2:ht|"

    def test_no_extension(self):
        core, ext = _split_cxsmiles("CC")
        assert core == "CC"
        assert ext == ""

    def test_only_first_pipe_splits(self):
        # Trailing | is part of ext, not a second split point.
        core, ext = _split_cxsmiles("CC|a|b")
        assert core == "CC"
        assert ext == "a|b"


class TestParseStable:
    def test_basic_two_groups(self):
        parsed = _parse_stable("R0:hydroxyl<n>carboxyl<ns>R1:methyl")
        assert parsed == {"R0": ["hydroxyl", "carboxyl"], "R1": ["methyl"]}

    def test_empty_string(self):
        assert _parse_stable("") == {}

    def test_label_with_no_colon(self):
        # Documented behavior: keep label with no alternatives.
        assert _parse_stable("R0") == {"R0": []}

    def test_trailing_ns_ignored(self):
        parsed = _parse_stable("R0:a<ns>")
        assert parsed == {"R0": ["a"]}

    def test_whitespace_stripped(self):
        assert _parse_stable(" R0 : a <n> b ") == {"R0": ["a", "b"]}

    def test_empty_alternatives_filtered(self):
        # An entry like "R0:a<n><n>b" should ignore empty alternatives.
        assert _parse_stable("R0:a<n><n>b") == {"R0": ["a", "b"]}


# ----------------------------------------------------------------------------
# parse_markush_output / parse_ground_truth_entry
# ----------------------------------------------------------------------------


class TestParseMarkushOutput:
    def test_valid_strict(self):
        raw = "<markush><cxsmi><r>R1</r>CC</cxsmi></markush>"
        pred = parse_markush_output(raw, strict=True)
        assert pred.valid
        assert pred.cxsmiles_opt == "<r>R1</r>CC"
        assert pred.r_labels_set == {"R1"}
        assert pred.backbone_smiles == "*CC"

    def test_strict_missing_xml(self):
        pred = parse_markush_output("CC", strict=True)
        assert not pred.valid
        assert "missing" in pred.parse_error

    def test_lenient_bare_smiles(self):
        pred = parse_markush_output("CC", strict=False)
        assert pred.valid
        assert pred.cxsmiles_opt == "CC"

    def test_lenient_rejects_malformed_xml(self):
        # Has <markush> but no closing — lenient mode must still reject.
        pred = parse_markush_output("<markush><cxsmi>CC", strict=False)
        assert not pred.valid

    def test_empty_input_returns_invalid(self):
        pred = parse_markush_output("", strict=True)
        assert not pred.valid
        assert pred.parse_error

    def test_non_string_returns_invalid(self):
        pred = parse_markush_output(None, strict=True)  # type: ignore[arg-type]
        assert not pred.valid

    def test_extension_extracted(self):
        raw = "<markush><cxsmi>CC |Sg:n:1,2:ht|</cxsmi></markush>"
        pred = parse_markush_output(raw, strict=True)
        assert pred.backbone_smiles == "CC"
        assert "Sg" in pred.extensions

    def test_stable_parsed_when_present(self):
        raw = "<markush><cxsmi>CC</cxsmi><stable>R0:methyl</stable></markush>"
        pred = parse_markush_output(raw, strict=True)
        assert pred.stable == {"R0": ["methyl"]}


class TestParseGroundTruthEntry:
    def test_cxsmiles_opt_only(self):
        gt = parse_ground_truth_entry({"cxsmiles_opt": "<r>R1</r>CC"})
        assert gt.valid
        assert gt.r_labels_set == {"R1"}
        assert gt.backbone_smiles == "*CC"

    def test_annotation_with_cxsmiles_opt_override(self):
        gt = parse_ground_truth_entry(
            {
                "cxsmiles_opt": "<r>R2</r>CC",
                "annotation": "<markush><cxsmi><r>R1</r>CC</cxsmi></markush>",
            }
        )
        # cxsmiles_opt overrides what's parsed from annotation.
        assert gt.cxsmiles_opt == "<r>R2</r>CC"
        assert gt.r_labels_set == {"R2"}

    def test_annotation_only(self):
        gt = parse_ground_truth_entry(
            {
                "annotation": "<markush><cxsmi><r>R1</r>CC</cxsmi><stable>R1:methyl</stable></markush>"
            }
        )
        assert gt.valid
        assert gt.r_labels_set == {"R1"}
        assert gt.stable == {"R1": ["methyl"]}

    def test_neither_field_present(self):
        gt = parse_ground_truth_entry({})
        assert not gt.valid


# ----------------------------------------------------------------------------
# R-group metrics
# ----------------------------------------------------------------------------


class TestRGroupMetrics:
    def test_recall_perfect(self):
        pred = _mk("<r>R1</r><r>R2</r>CC")
        gt = _mk("<r>R1</r><r>R2</r>CC")
        assert metric_r_accuracy(pred, gt) == 1.0

    def test_recall_partial(self):
        pred = _mk("<r>R1</r>CC")
        gt = _mk("<r>R1</r><r>R2</r>CC")
        assert metric_r_accuracy(pred, gt) == 0.5

    def test_recall_empty_gt_is_vacuously_true(self):
        pred = _mk("CC")
        gt = _mk("CC")
        assert metric_r_accuracy(pred, gt) == 1.0

    def test_recall_zero_when_no_pred_labels_but_gt_has(self):
        pred = _mk("CC")
        gt = _mk("<r>R1</r>CC")
        assert metric_r_accuracy(pred, gt) == 0.0

    def test_precision_no_pred_labels_returns_one_when_gt_also_empty(self):
        pred = _mk("CC")
        gt = _mk("CC")
        assert metric_r_precision(pred, gt) == 1.0

    def test_precision_no_pred_labels_but_gt_has_returns_zero(self):
        pred = _mk("CC")
        gt = _mk("<r>R1</r>CC")
        assert metric_r_precision(pred, gt) == 0.0

    def test_precision_extra_pred_labels(self):
        pred = _mk("<r>R1</r><r>R2</r>CC")
        gt = _mk("<r>R1</r>CC")
        assert metric_r_precision(pred, gt) == 0.5

    def test_f1_perfect(self):
        pred = _mk("<r>R1</r>CC")
        gt = _mk("<r>R1</r>CC")
        assert metric_r_f1(pred, gt) == 1.0

    def test_f1_zero_on_disjoint_sets(self):
        pred = _mk("<r>RA</r>CC")
        gt = _mk("<r>RB</r>CC")
        assert metric_r_f1(pred, gt) == 0.0

    def test_f1_no_division_by_zero(self):
        # Edge case: both pred and gt empty -> both prec/recall are 1.0
        pred = _mk("CC")
        gt = _mk("CC")
        assert metric_r_f1(pred, gt) == 1.0


# ----------------------------------------------------------------------------
# string_em, extension_match
# ----------------------------------------------------------------------------


class TestStringEm:
    def test_identical(self):
        a, b = _mk("CC"), _mk("CC")
        assert metric_string_em(a, b) is True

    def test_different_strings(self):
        a, b = _mk("CC"), _mk("CCO")
        assert metric_string_em(a, b) is False

    def test_canonical_difference_still_string_mismatch(self):
        # string_em is intentionally strict — different canonical forms differ.
        a, b = _mk("C1=CC=CC=C1"), _mk("c1ccccc1")
        assert metric_string_em(a, b) is False


class TestExtensionMatch:
    def test_same_extension(self):
        a = _mk("CC |Sg:n:1,2:ht|")
        b = _mk("CC |Sg:n:1,2:ht|")
        assert metric_extension_match(a, b) is True

    def test_different_extension(self):
        a = _mk("CC |Sg:n:1,2:ht|")
        b = _mk("CC |Sg:n:3,4:ht|")
        assert metric_extension_match(a, b) is False

    def test_no_extensions_either_side(self):
        assert metric_extension_match(_mk("CC"), _mk("CC")) is True


# ----------------------------------------------------------------------------
# stable metrics
# ----------------------------------------------------------------------------


class TestStableMetrics:
    def test_perfect_match(self):
        pred = _mk("CC", "R0:a<n>b<ns>R1:c")
        gt = _mk("CC", "R0:a<n>b<ns>R1:c")
        assert metric_stable_accuracy(pred, gt) == 1.0
        assert metric_stable_exact_match(pred, gt) is True

    def test_alternatives_match_regardless_of_order(self):
        pred = _mk("CC", "R0:b<n>a")
        gt = _mk("CC", "R0:a<n>b")
        # Sorting before compare means order-insensitive.
        assert metric_stable_accuracy(pred, gt) == 1.0

    def test_partial_match_returns_fraction(self):
        pred = _mk("CC", "R0:a<ns>R1:wrong")
        gt = _mk("CC", "R0:a<ns>R1:c")
        assert metric_stable_accuracy(pred, gt) == 0.5

    def test_missing_label_in_pred_counts_as_miss(self):
        pred = _mk("CC", "R0:a")
        gt = _mk("CC", "R0:a<ns>R1:b")
        assert metric_stable_accuracy(pred, gt) == 0.5

    def test_empty_gt_and_empty_pred_is_vacuously_correct(self):
        pred = _mk("CC")
        gt = _mk("CC")
        assert metric_stable_accuracy(pred, gt) == 1.0
        assert metric_stable_exact_match(pred, gt) is True

    def test_empty_gt_but_pred_has_extras_penalized(self):
        # After the dead-code fix this should now consistently penalize extras.
        pred = _mk("CC", "R0:a")
        gt = _mk("CC")
        assert metric_stable_accuracy(pred, gt) == 0.0
        assert metric_stable_exact_match(pred, gt) is False

    def test_exact_match_only_when_complete(self):
        pred = _mk("CC", "R0:a<ns>R1:wrong")
        gt = _mk("CC", "R0:a<ns>R1:c")
        assert metric_stable_exact_match(pred, gt) is False


# ----------------------------------------------------------------------------
# RDKit-backed metrics
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
class TestBackboneMatch:
    def test_identical_backbone_matches(self):
        pred = _mk("CCO")
        gt = _mk("CCO")
        assert metric_backbone_match(pred, gt) is True

    def test_different_canonical_smiles_same_molecule(self):
        # Benzene written two ways must produce the same InChIKey.
        pred = _mk("c1ccccc1")
        gt = _mk("C1=CC=CC=C1")
        assert metric_backbone_match(pred, gt) is True

    def test_different_molecules_dont_match(self):
        pred = _mk("CCO")
        gt = _mk("CCN")
        assert metric_backbone_match(pred, gt) is False

    def test_wildcards_replaced_with_carbon(self):
        # <r>R1</r>C → *C → CC (after wildcard->C). Compare to CC.
        pred = _mk("<r>R1</r>C")
        gt = _mk("CC")
        assert metric_backbone_match(pred, gt) is True

    def test_returns_none_for_unparseable(self):
        # Garbage SMILES → both unparseable → None
        pred = _mk("!!!@@@")
        gt = _mk("CC")
        result = metric_backbone_match(pred, gt)
        # If RDKit decides "!!!@@@" is unparseable, returns None.
        assert result is None or result is False


@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="RDKit required")
class TestMarkushEm:
    def test_perfect_match(self):
        pred = _mk("<r>R1</r>CC", "R1:methyl")
        gt = _mk("<r>R1</r>CC", "R1:methyl")
        assert metric_markush_em(pred, gt) is True

    def test_backbone_diff_returns_false(self):
        pred = _mk("<r>R1</r>CCO", "R1:methyl")
        gt = _mk("<r>R1</r>CCN", "R1:methyl")
        assert metric_markush_em(pred, gt) is False

    def test_stable_diff_short_circuits_false(self):
        pred = _mk("<r>R1</r>CC", "R1:methyl")
        gt = _mk("<r>R1</r>CC", "R1:ethyl")
        # Stable mismatch → False, regardless of backbone.
        assert metric_markush_em(pred, gt) is False

    def test_falls_back_to_string_em_without_backbone(self):
        pred = _mk("CC")
        gt = _mk("CC")
        assert metric_markush_em(pred, gt, use_backbone=False) is True


# ----------------------------------------------------------------------------
# string similarity
# ----------------------------------------------------------------------------


class TestStringSimilarity:
    def test_identical(self):
        assert _string_similarity("CC", "CC") == 1.0

    def test_completely_different(self):
        assert _string_similarity("abc", "xyz") == pytest.approx(0.0)

    def test_one_edit_in_short_string(self):
        # Levenshtein distance 1, max_len 3 -> 1 - 1/3 = 0.666...
        assert _string_similarity("abc", "abd") == pytest.approx(2 / 3)

    def test_empty_strings_both(self):
        # Documented: identical strings return 1.0
        assert _string_similarity("", "") == 1.0

    def test_one_empty_one_not(self):
        assert _string_similarity("", "abc") == 0.0


# ----------------------------------------------------------------------------
# compute_metrics aggregate
# ----------------------------------------------------------------------------


class TestComputeMetrics:
    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics["n_samples"] == 0
        assert "error" in metrics

    def test_all_invalid_predictions(self):
        results = [
            SampleResult(sample_id="a", valid_pred=False),
            SampleResult(sample_id="b", valid_pred=False),
        ]
        m = compute_metrics(results)
        assert m["valid_rate"] == 0.0
        assert m["string_em"] == 0.0
        # No valid preds means no backbone comparisons either.
        assert m["backbone_match"] is None

    def test_mixed_validity(self):
        results = [
            SampleResult(sample_id="a", valid_pred=True, string_em=True, r_accuracy=1.0),
            SampleResult(sample_id="b", valid_pred=False),
        ]
        m = compute_metrics(results)
        assert m["valid_rate"] == 0.5
        assert m["string_em"] == 0.5
        # string_em_valid normalizes by number of valid preds (1).
        assert m["string_em_valid"] == 1.0


# ----------------------------------------------------------------------------
# evaluate_predictions / evaluate_sample
# ----------------------------------------------------------------------------


class TestEvaluatePredictions:
    def test_end_to_end_perfect(self):
        preds = [
            {"id": "x1", "prediction": "<markush><cxsmi><r>R1</r>CC</cxsmi></markush>"},
        ]
        gts = [
            {"id": "x1", "cxsmiles_opt": "<r>R1</r>CC"},
        ]
        metrics, results = evaluate_predictions(preds, gts)
        assert metrics["valid_rate"] == 1.0
        assert metrics["string_em"] == 1.0
        assert results[0].r_accuracy == 1.0

    def test_invalid_prediction_kept_in_results(self):
        preds = [{"id": "x1", "prediction": "garbage with no xml"}]
        gts = [{"id": "x1", "cxsmiles_opt": "CC"}]
        metrics, results = evaluate_predictions(preds, gts)
        assert metrics["valid_rate"] == 0.0
        assert results[0].valid_pred is False

    def test_missing_gt_adds_warning_but_runs(self):
        preds = [{"id": "x1", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"}]
        gts = []
        with pytest.warns(UserWarning):
            metrics, results = evaluate_predictions(preds, gts)
        # x1 had no GT → not counted at all (results empty)
        assert metrics["n_samples"] == 0

    def test_missing_pred_counted_as_invalid(self):
        preds = []
        gts = [{"id": "x1", "cxsmiles_opt": "CC"}]
        with pytest.warns(UserWarning):
            metrics, results = evaluate_predictions(preds, gts)
        assert metrics["n_samples"] == 1
        assert metrics["valid_rate"] == 0.0
        assert results[0].parse_error == "no prediction provided"

    def test_evaluate_sample_propagates_invalid(self):
        pred = parse_markush_output("garbage")
        gt = parse_ground_truth_entry({"cxsmiles_opt": "CC"})
        result = evaluate_sample("z", pred, gt)
        assert result.valid_pred is False
        # When invalid, early-return before metrics; all stay default.
        assert result.string_em is False
        assert result.r_accuracy == 0.0
