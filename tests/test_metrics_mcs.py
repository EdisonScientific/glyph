#!/usr/bin/env python3
"""Tests for glyph.markush.eval.metrics_mcs.

Audits:
- parsing helpers (XML extraction, r-tag stripping, stable parsing)
- atom-label tagging for inserted wildcards
- single-molecule scoring (compute_markush_prediction_quality)
- stable scoring (get_stable_score) — equality / recall / precision
- batch scoring (get_scores) reduced output
- evaluate_markush_mcs end-to-end with JSONL-style input
"""

from __future__ import annotations

import pytest

from glyph.markush.eval.metrics_mcs import (
    _RDKIT_AVAILABLE,
    _extract_cxsmiles_and_stable,
    _labels_for_inserted_wildcards,
    _normalize_protocol,
    _parse_m_sections,
    _parse_stable,
    _split_cxsmiles,
    _stable_normalized_text,
    compute_markush_prediction_quality,
    evaluate_markush_both_protocols,
    evaluate_markush_mcs,
    flatten_both_protocol_metrics,
    get_scores,
    get_stable_score,
    parse_markush_text,
    strip_r_tags,
)

pytestmark = pytest.mark.skipif(
    not _RDKIT_AVAILABLE,
    reason="metrics_mcs requires RDKit",
)


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------


class TestExtractCxsmilesAndStable:
    def test_basic(self):
        raw = "<markush><cxsmi>CC</cxsmi><stable>R0:a</stable></markush>"
        cx, st = _extract_cxsmiles_and_stable(raw)
        assert cx == "CC"
        assert st == "R0:a"

    def test_case_insensitive_tags(self):
        raw = "<MARKUSH><CXSMI>CC</CXSMI><STABLE>R0:a</STABLE></MARKUSH>"
        cx, st = _extract_cxsmiles_and_stable(raw)
        assert cx == "CC"
        assert st == "R0:a"

    def test_works_without_markush_wrapper(self):
        # Unlike metrics_string, this helper does NOT require <markush> wrapper.
        cx, st = _extract_cxsmiles_and_stable("<cxsmi>CC</cxsmi>")
        assert cx == "CC"
        assert st == ""

    def test_non_string_input(self):
        assert _extract_cxsmiles_and_stable(None) == ("", "")  # type: ignore[arg-type]


class TestStripRTags:
    def test_replaces_with_wildcard(self):
        assert strip_r_tags("<r>R1</r>C") == "*C"

    def test_case_insensitive(self):
        # metrics_mcs uses IGNORECASE in _RE_R_TAG, unlike metrics_string.
        assert strip_r_tags("<R>R1</R>C") == "*C"

    def test_empty(self):
        assert strip_r_tags("") == ""

    def test_none_safe(self):
        assert strip_r_tags(None) == ""  # type: ignore[arg-type]


class TestSplitCxsmiles:
    def test_with_extension(self):
        core, ext = _split_cxsmiles("CC |Sg:n:1,2:ht|")
        assert core == "CC"
        # rstrip("|") at the end means the trailing pipe is stripped.
        assert ext == "Sg:n:1,2:ht"

    def test_no_extension(self):
        core, ext = _split_cxsmiles("CC")
        assert core == "CC"
        assert ext == ""


class TestParseStable:
    def test_basic(self):
        assert _parse_stable("R0:a<n>b<ns>R1:c") == {"R0": ["a", "b"], "R1": ["c"]}

    def test_accepts_equals_separator(self):
        # Documented permissive behavior for mildly malformed model outputs.
        assert _parse_stable("R0=a<n>b") == {"R0": ["a", "b"]}

    def test_colon_wins_when_both_present(self):
        # "R0:a=b" — colon comes first, so label is "R0", alt is "a=b".
        assert _parse_stable("R0:a=b") == {"R0": ["a=b"]}

    def test_empty(self):
        assert _parse_stable("") == {}

    def test_label_without_separator(self):
        # Keep label with no alts so false positives don't disappear.
        assert _parse_stable("R0") == {"R0": []}


class TestLabelsForInsertedWildcards:
    def test_single_wildcard_label(self):
        labels = _labels_for_inserted_wildcards("<r>R1</r>C")
        assert labels == {1: "R1"}

    def test_multiple_wildcards_label_ordinal(self):
        # First * comes from <r>R1</r>, second from <r>R2</r>.
        labels = _labels_for_inserted_wildcards("<r>R1</r>C<r>R2</r>")
        assert labels == {1: "R1", 2: "R2"}

    def test_pre_existing_wildcard_shifts_ordinal(self):
        # The "*" in the SMILES is wildcard ordinal 1; <r>R1</r> becomes ordinal 2.
        labels = _labels_for_inserted_wildcards("*C<r>R1</r>")
        assert labels == {2: "R1"}

    def test_no_r_tags(self):
        assert _labels_for_inserted_wildcards("CC") == {}


class TestParseMSections:
    def test_single_m_section(self):
        sections = _parse_m_sections("CC |m:1:2.3|")
        assert sections == [{"atom_connector": 1, "ring_atoms": [2, 3]}]

    def test_no_m_section(self):
        assert _parse_m_sections("CC") == []

    def test_m_section_with_multiple_ring_atoms(self):
        sections = _parse_m_sections("CC |Sg:n:1,2:ht,m:5:6.7.8|")
        assert sections == [{"atom_connector": 5, "ring_atoms": [6, 7, 8]}]


class TestParseMarkushText:
    def test_strict_requires_cxsmi(self):
        result = parse_markush_text("CC", strict=True)
        assert not result.valid
        assert "cxsmi" in result.parse_error.lower()

    def test_lenient_bare_smiles(self):
        result = parse_markush_text("CC", strict=False)
        assert result.valid
        assert result.cxsmiles == "CC"
        assert result.rdkit_cxsmiles == "CC"

    def test_lenient_rejects_malformed_xml(self):
        # Contains "<markush" but no proper closure → reject in lenient mode.
        result = parse_markush_text("<markush><cxsmi>CC", strict=False)
        assert not result.valid

    def test_empty_input(self):
        result = parse_markush_text("", strict=False)
        assert not result.valid

    def test_none_input(self):
        result = parse_markush_text(None, strict=False)
        assert not result.valid

    def test_atom_labels_populated_after_parse(self):
        # <r>R1</r>CC → wildcard ordinal 1 has label R1.
        result = parse_markush_text("<r>R1</r>CC", strict=False)
        assert result.valid
        assert "R1" in result.atom_labels.values()


# ----------------------------------------------------------------------------
# Stable score (substituent tables)
# ----------------------------------------------------------------------------


class TestStableNormalizedText:
    def test_strips_articles_and_spaces(self):
        assert _stable_normalized_text("A hydroxyl group") == "hydroxyl"

    def test_lowercase(self):
        assert _stable_normalized_text("METHYL") == "methyl"


class TestGetStableScore:
    def test_perfect_match(self):
        gt = {"R0": ["methyl", "ethyl"]}
        pred = {"R0": ["methyl", "ethyl"]}
        s = get_stable_score(gt, pred)
        assert s["stable_equality"] is True
        assert s["stable_recall"] == 1.0
        assert s["stable_precision"] == 1.0

    def test_alternatives_order_independent(self):
        gt = {"R0": ["methyl", "ethyl"]}
        pred = {"R0": ["ethyl", "methyl"]}
        s = get_stable_score(gt, pred)
        assert s["stable_equality"] is True

    def test_both_empty_is_perfect(self):
        s = get_stable_score({}, {})
        assert s["stable_equality"] is True
        assert s["stable_recall"] == 1.0
        assert s["stable_precision"] == 1.0

    def test_empty_gt_with_pred_extras_not_equal(self):
        s = get_stable_score({}, {"R0": ["a"]})
        # gt empty + pred non-empty → equality False, recall/precision 0 (defaults).
        assert s["stable_equality"] is False

    def test_missing_label_in_pred(self):
        gt = {"R0": ["a"], "R1": ["b"]}
        pred = {"R0": ["a"]}
        s = get_stable_score(gt, pred)
        assert s["stable_equality"] is False
        assert s["stable_recall"] == pytest.approx(0.5)

    def test_extra_label_in_pred_breaks_equality(self):
        gt = {"R0": ["a"]}
        pred = {"R0": ["a"], "R1": ["b"]}
        s = get_stable_score(gt, pred)
        # GT recall is satisfied, but extra predicted label means not "equal".
        assert s["stable_equality"] is False
        assert s["stable_recall"] == 1.0
        # Precision: R0 row 1.0, R1 row 0.0 → mean 0.5
        assert s["stable_precision"] == pytest.approx(0.5)

    def test_normalize_article_handling(self):
        # GT "a methyl group", pred "methyl" — normalized match.
        gt = {"R0": ["a methyl group"]}
        pred = {"R0": ["methyl"]}
        s = get_stable_score(gt, pred)
        # With permissive+normalize, these match.
        assert s["stable_equality"] is True

    def test_case_insensitive_label_match(self):
        gt = {"R0": ["methyl"]}
        pred = {"r0": ["methyl"]}
        s = get_stable_score(gt, pred)
        # permissive=True lowercases labels.
        assert s["stable_equality"] is True


# ----------------------------------------------------------------------------
# compute_markush_prediction_quality (single-molecule MCS scoring)
# ----------------------------------------------------------------------------


class TestComputeMarkushPredictionQuality:
    def test_identical_strings(self):
        scores = compute_markush_prediction_quality("CCO", "CCO")
        assert scores["valid"] is True
        assert scores["string_equality"] is True
        assert scores["inchi_equality"] is True or scores["backbone_inchi_match"] is True

    def test_same_molecule_different_canonical_forms(self):
        # Benzene aromatic vs Kekulé — should yield same InChI.
        scores = compute_markush_prediction_quality("c1ccccc1", "C1=CC=CC=C1")
        assert scores["backbone_inchi_match"] is True

    def test_different_molecules(self):
        scores = compute_markush_prediction_quality("CCO", "CCN")
        assert scores["backbone_inchi_match"] is False
        assert scores["string_equality"] is False

    def test_unparseable_pred_returns_default_invalid(self):
        # Garbage SMILES doesn't crash, just returns the default-invalid scores.
        scores = compute_markush_prediction_quality("!!!@@@", "CC")
        assert scores["valid"] is False

    def test_r_group_label_match(self):
        # Two molecules with same R label at equivalent position.
        scores = compute_markush_prediction_quality("<r>R1</r>CC", "<r>R1</r>CC")
        # r_group_accuracy populated when GT has R labels.
        assert scores.get("r_group_accuracy") is not None
        assert scores["r_group_accuracy"] == 1.0

    def test_r_group_label_mismatch_lowers_accuracy(self):
        scores = compute_markush_prediction_quality("<r>RA</r>CC", "<r>RB</r>CC")
        # Same molecular backbone, different label — r accuracy is 0.
        assert scores["r_group_accuracy"] == 0.0
        assert scores["position_graph_equal"] is False
        assert scores["cxsmi_equality"] is False

    def test_no_r_groups_in_gt_results_in_none(self):
        scores = compute_markush_prediction_quality("CC", "CC")
        # When GT has no R-groups, r metrics are None.
        assert scores["r_group_accuracy"] is None

    def test_position_graph_equality_rejects_missing_cx_features(self):
        scores = compute_markush_prediction_quality("CC=CC |t:1|", "CC=CC |c:1|")

        assert scores["position_graph_equal"] is False
        assert scores["cxsmi_equality"] is False

        scores = compute_markush_prediction_quality(
            "[Na+].[OH-].c1ccccc1",
            "[Na+].[OH-].c1ccccc1 |f:0.1|",
        )

        assert scores["position_graph_equal"] is False
        assert scores["cxsmi_equality"] is False

        scores = compute_markush_prediction_quality("NCC=O", "NCC=O |LN:1:1.3|")

        assert scores["position_graph_equal"] is False
        assert scores["cxsmi_equality"] is False


# ----------------------------------------------------------------------------
# get_scores batch evaluator
# ----------------------------------------------------------------------------


class TestGetScores:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            get_scores(["CC"], ["CC", "CCO"])

    def test_perfect_batch(self):
        gts = ["CCO", "CCC"]
        preds = ["CCO", "CCC"]
        reduced, per_sample = get_scores(gts, preds)
        assert reduced["size"] == 2
        assert reduced["string_equality"] >= 0.9  # both perfect
        assert per_sample[0] is not None

    def test_invalid_gt_marked_none(self):
        # "" GT → parse_markush_text returns invalid → score is None for that idx.
        reduced, per_sample = get_scores(["", "CC"], ["CC", "CC"])
        assert per_sample[0] is None
        assert per_sample[1] is not None

    def test_invalid_pred_returns_default_incorrect(self):
        reduced, per_sample = get_scores(["CC"], [""])
        assert per_sample[0] is not None
        assert per_sample[0]["valid"] is False
        assert per_sample[0]["cxsmi_equality"] is False

    def test_markush_mode_uses_stable_lists(self):
        gt_stable = [{"R0": ["a"]}]
        pred_stable = [{"R0": ["a"]}]
        reduced, per_sample = get_scores(
            ["<r>R0</r>CC"],
            ["<r>R0</r>CC"],
            gt_stable_list=gt_stable,
            predicted_stable_list=pred_stable,
            markush=True,
        )
        # Stable equality is one of the keys produced by get_stable_score.
        assert per_sample[0] is not None
        assert "stable_equality" in per_sample[0]

    def test_default_equality_rejects_known_label_and_attachment_false_positive(self):
        gt = "<r>Ra</r>S(=O)(=O)[N-]S(<r>CF3</r>)(=O)=O"
        pred = "<r>CF3</r>S(=O)(=O)[N-]S(<r>Ra</r>)(=O)=O|m:1:2.3.4"

        _reduced, per_sample = get_scores([gt], [pred])

        assert per_sample[0] is not None
        assert per_sample[0]["position_graph_equal"] is False
        assert per_sample[0]["cxsmi_equality"] is False

    def test_default_equality_accepts_known_serialization_variant(self):
        gt = "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6"
        pred = "<r>Ym</r>C.NC1=CC=CC=C1|m:1:3.4.5.6.7.8"

        _reduced, per_sample = get_scores([gt], [pred])

        assert per_sample[0] is not None
        assert per_sample[0]["position_graph_equal"] is True
        assert per_sample[0]["cxsmi_equality"] is True


# ----------------------------------------------------------------------------
# evaluate_markush_mcs end-to-end
# ----------------------------------------------------------------------------


class TestEvaluateMarkushMcs:
    def test_simple_perfect_prediction(self):
        preds = [{"id": "a", "prediction": "<markush><cxsmi>CC</cxsmi><stable></stable></markush>"}]
        gts = [{"id": "a", "cxsmiles_opt": "CC"}]
        metrics = evaluate_markush_mcs(preds, gts)
        assert metrics["n_samples"] == 1
        assert metrics["valid_rate"] == 1.0
        assert metrics["string_exact_match"] == 1.0

    def test_missing_prediction_counts_as_invalid(self):
        preds = []
        gts = [{"id": "a", "cxsmiles_opt": "CC"}]
        metrics = evaluate_markush_mcs(preds, gts, include_missing=True)
        assert metrics["n_samples"] == 1
        assert metrics["valid_rate"] == 0.0

    def test_extra_prediction_ignored_with_warning(self):
        preds = [
            {"id": "a", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"},
            {"id": "b", "prediction": "<markush><cxsmi>CCO</cxsmi></markush>"},  # extra
        ]
        gts = [{"id": "a", "cxsmiles_opt": "CC"}]
        with pytest.warns(UserWarning):
            metrics = evaluate_markush_mcs(preds, gts)
        assert metrics["n_samples"] == 1

    def test_invalid_prediction_text(self):
        preds = [{"id": "a", "prediction": "no xml here"}]
        gts = [{"id": "a", "cxsmiles_opt": "CC"}]
        # lenient mode accepts bare cxsmiles, so "no xml here" will be parsed as
        # cxsmiles. That'd be unparseable by RDKit but parser marks it valid.
        # Verify we don't crash and we get a result.
        metrics, details = evaluate_markush_mcs(preds, gts, return_details=True)
        assert metrics["n_samples"] == 1
        assert details[0]["id"] == "a"

    def test_return_details_shape(self):
        preds = [{"id": "a", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"}]
        gts = [{"id": "a", "cxsmiles_opt": "CC"}]
        metrics, details = evaluate_markush_mcs(preds, gts, return_details=True)
        assert isinstance(metrics, dict)
        assert isinstance(details, list)
        assert details[0]["id"] == "a"


# ----------------------------------------------------------------------------
# Protocol toggle: ours vs mg2
# ----------------------------------------------------------------------------


class TestProtocolNormalization:
    def test_ours_passthrough(self):
        assert _normalize_protocol("ours") == "ours"

    def test_mg2_passthrough(self):
        assert _normalize_protocol("mg2") == "mg2"

    def test_strict_is_alias_for_ours(self):
        assert _normalize_protocol("strict") == "ours"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            _normalize_protocol("nope")


class TestMg2ProtocolHonesty:
    """Scientific-integrity guard: the in-package ``mg2`` protocol must be
    described as an *approximate* replication, never an *exact* one; official
    numbers require the pinned DS4SD scorer."""

    def _source(self) -> str:
        import inspect

        from glyph.markush.eval import metrics_mcs

        return inspect.getsource(metrics_mcs)

    def test_no_exact_replication_claim(self):
        assert "exact replication" not in self._source().lower()

    def test_describes_mg2_as_approximate(self):
        source = self._source().lower()
        assert "approximate" in source
        # The caveat must point users at the authoritative scorer.
        assert "tools/mg2_scorer" in self._source()


class TestStableProtocolDivergence:
    """Stable section: ours penalizes hallucinated pred labels; mg2 ignores them."""

    def test_extra_pred_label_ours_fails_equality(self):
        gt = {"R0": ["methyl"]}
        pred = {"R0": ["methyl"], "R1": ["hallucinated"]}
        s = get_stable_score(gt, pred, protocol="ours")
        assert s["stable_equality"] is False

    def test_extra_pred_label_mg2_passes_equality(self):
        gt = {"R0": ["methyl"]}
        pred = {"R0": ["methyl"], "R1": ["hallucinated"]}
        s = get_stable_score(gt, pred, protocol="mg2")
        # MG2 only checks GT recall; extra pred labels are ignored.
        assert s["stable_equality"] is True
        assert s["stable_recall"] == 1.0

    def test_recall_matches_across_protocols(self):
        gt = {"R0": ["methyl", "ethyl"]}
        pred = {"R0": ["methyl"]}
        s_ours = get_stable_score(gt, pred, protocol="ours")
        s_mg2 = get_stable_score(gt, pred, protocol="mg2")
        # Recall should be 0.5 under both protocols (1 of 2 GT alts found).
        assert s_ours["stable_recall"] == pytest.approx(0.5)
        assert s_mg2["stable_recall"] == pytest.approx(0.5)

    def test_normalize_diff_lowercase_only_in_permissive_pass_under_mg2(self):
        # MG2's normalizer doesn't lowercase, but the permissive pass does.
        # Effective behavior: case-insensitive equality holds under both.
        gt = {"R0": ["Methyl"]}
        pred = {"R0": ["methyl"]}
        s_ours = get_stable_score(gt, pred, protocol="ours")
        s_mg2 = get_stable_score(gt, pred, protocol="mg2")
        assert s_ours["stable_equality"] is True
        assert s_mg2["stable_equality"] is True


class TestComputeMarkushPredictionQualityProtocolDivergence:
    """cxsmi_equality: ours requires bijection, mg2 only requires GT recall."""

    def test_perfect_prediction_passes_both(self):
        s_ours = compute_markush_prediction_quality("<r>R0</r>CC", "<r>R0</r>CC", protocol="ours")
        s_mg2 = compute_markush_prediction_quality("<r>R0</r>CC", "<r>R0</r>CC", protocol="mg2")
        assert s_ours["cxsmi_equality"] is True
        assert s_mg2["cxsmi_equality"] is True

    def test_unparseable_returns_invalid_under_both(self):
        s_ours = compute_markush_prediction_quality("!!!", "CC", protocol="ours")
        s_mg2 = compute_markush_prediction_quality("!!!", "CC", protocol="mg2")
        assert s_ours["valid"] is False
        assert s_mg2["valid"] is False

    def test_protocol_accepts_strict_alias(self):
        # Backward-compat: callers passing "strict" should still work.
        s = compute_markush_prediction_quality("CC", "CC", protocol="strict")
        assert s["valid"] is True


class TestEvaluateBothProtocols:
    def test_returns_nested_dict(self):
        preds = [{"id": "a", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"}]
        gts = [{"id": "a", "cxsmiles_opt": "CC"}]
        both = evaluate_markush_both_protocols(preds, gts, return_details=False)
        assert set(both.keys()) == {"ours", "mg2"}
        assert "n_samples" in both["ours"]
        assert "n_samples" in both["mg2"]
        assert both["ours"]["n_samples"] == 1
        assert both["mg2"]["n_samples"] == 1

    def test_returns_details_when_requested(self):
        preds = [{"id": "a", "prediction": "<markush><cxsmi>CC</cxsmi></markush>"}]
        gts = [{"id": "a", "cxsmiles_opt": "CC"}]
        both_metrics, both_details = evaluate_markush_both_protocols(
            preds, gts, return_details=True
        )
        assert set(both_metrics.keys()) == {"ours", "mg2"}
        assert set(both_details.keys()) == {"ours", "mg2"}
        assert both_details["ours"][0]["id"] == "a"
        assert both_details["mg2"][0]["id"] == "a"


class TestFlattenBothProtocolMetrics:
    def test_namespaces_keys(self):
        both = {
            "ours": {"cxsmi_equality": 0.5, "valid_rate": 1.0},
            "mg2": {"cxsmi_equality": 0.7, "valid_rate": 1.0},
        }
        flat = flatten_both_protocol_metrics(both, prefix="eval")
        assert flat["eval/ours/cxsmi_equality"] == 0.5
        assert flat["eval/mg2/cxsmi_equality"] == 0.7
        assert flat["eval/ours/valid_rate"] == 1.0

    def test_custom_prefix(self):
        both = {"ours": {"x": 1}, "mg2": {"x": 2}}
        flat = flatten_both_protocol_metrics(both, prefix="step123")
        assert flat == {"step123/ours/x": 1, "step123/mg2/x": 2}
