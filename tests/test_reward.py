#!/usr/bin/env python3
"""Tests for the component-wise reward in glyph.markush.eval.reward.

Audits:
- parse_markush: XML detection, field extraction, R-label set, Sg/m flags
- component rewards: valid_parse, backbone, r_groups, sg, m, stable
- compute_reward total: bounds, weight overrides, garbage handling
- robustness: malformed predictions return finite values, never crash
"""

from __future__ import annotations

import math

import pytest

from glyph.markush.eval.reward import (
    RDKIT_AVAILABLE,
    _extract_m_section,
    _parse_sg_sections,
    compute_reward,
    compute_reward_breakdown,
    parse_markush,
    reward_backbone_match,
    reward_backbone_tanimoto,
    reward_extra_sg,
    reward_m_section,
    reward_mg2_exact_scores,
    reward_r_groups,
    reward_sg_section,
    reward_stable,
    reward_structure_match,
    reward_valid_parse,
)

# ----------------------------------------------------------------------------
# parse_markush
# ----------------------------------------------------------------------------


class TestParseMarkush:
    def test_full_valid(self):
        text = "<markush><cxsmi><r>R1</r>CC|Sg:n:1,2:ht</cxsmi><stable>R1:methyl</stable></markush>"
        p = parse_markush(text)
        assert p["valid_xml"] is True
        assert "<r>R1</r>CC" in p["cxsmiles"]
        assert p["r_labels"] == {"R1"}
        assert p["has_sg"] is True
        assert p["stable"] == "R1:methyl"

    def test_invalid_xml(self):
        p = parse_markush("no xml here")
        assert p["valid_xml"] is False
        assert p["cxsmiles"] == ""
        assert p["r_labels"] == set()

    def test_empty_input(self):
        p = parse_markush("")
        assert p["valid_xml"] is False

    def test_none_input(self):
        p = parse_markush(None)  # type: ignore[arg-type]
        assert p["valid_xml"] is False

    def test_non_string_input(self):
        p = parse_markush(12345)  # type: ignore[arg-type]
        assert p["valid_xml"] is False

    def test_no_stable_section(self):
        p = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert p["valid_xml"] is True
        assert p["stable"] == ""

    def test_no_extension(self):
        p = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert p["backbone"] == "CC"
        assert p["extensions"] == ""

    def test_has_m_flag(self):
        p = parse_markush("<markush><cxsmi>CC|m:1:2.3</cxsmi></markush>")
        assert p["has_m"] is True
        assert p["has_sg"] is False

    def test_has_sg_flag(self):
        p = parse_markush("<markush><cxsmi>CC|Sg:n:1,2:ht</cxsmi></markush>")
        assert p["has_sg"] is True
        assert p["has_m"] is False

    def test_case_insensitive_tags(self):
        # Models sometimes emit uppercase tags (`<CXSMI>`, `<R>`).  metrics_mcs
        # and predict already accept these; reward must do the same so that
        # reward scoring does not silently treat uppercase-tag outputs as garbage.
        p = parse_markush("<MARKUSH><CXSMI>c1ccccc1</CXSMI></MARKUSH>")
        assert p["valid_xml"] is True
        assert p["cxsmiles"] == "c1ccccc1"
        assert p["backbone"] == "c1ccccc1"

    def test_case_insensitive_r_tag(self):
        # `<R>...</R>` should be picked up the same as `<r>...</r>`.
        p = parse_markush("<markush><cxsmi><R>R1</R>c1ccccc1</cxsmi></markush>")
        assert p["valid_xml"] is True
        assert "R1" in p["r_labels"]


# ----------------------------------------------------------------------------
# _parse_sg_sections / _extract_m_section
# ----------------------------------------------------------------------------


class TestParseSgSections:
    def test_single_section(self):
        # Single Sg section, content after "Sg:" returned.
        assert _parse_sg_sections("Sg:n:1,2:ht") == ["n:1,2:ht"]

    def test_multiple_sg_sections(self):
        out = _parse_sg_sections("Sg:n:1,2:ht,Sg:n:3,4:eu")
        assert out == ["n:1,2:ht", "n:3,4:eu"]

    def test_empty(self):
        assert _parse_sg_sections("") == []

    def test_no_sg_section(self):
        # If extensions contain no Sg, the result is empty.
        assert _parse_sg_sections("m:1:2.3") == []

    def test_sg_does_not_swallow_following_m_section(self):
        # Regression: greedy `.split(",Sg:")` previously slurped the trailing
        # `,m:1:2.3` into the Sg payload.  The section-aware walker must stop
        # the Sg content at the next section-start marker.
        assert _parse_sg_sections("Sg:n:1,2:ht,m:1:2.3") == ["n:1,2:ht"]

    def test_sg_followed_by_m_then_another_sg(self):
        # Multiple Sg sections interleaved with other section types must all
        # be captured, none swallowing each other.
        assert _parse_sg_sections("Sg:n:1,2:ht,m:1:2.3,Sg:n:3,4:eu") == [
            "n:1,2:ht",
            "n:3,4:eu",
        ]

    def test_sg_at_end_of_extensions(self):
        # An Sg section preceded by other section types is still found.
        assert _parse_sg_sections("m:1:2.3,Sg:n:1,2:ht") == ["n:1,2:ht"]


class TestExtractMSection:
    def test_basic(self):
        assert _extract_m_section("m:1:2.3") == "1:2.3"

    def test_with_other_content_before(self):
        # Regex finds the first m: anywhere.
        assert _extract_m_section("Sg:n:1,2:ht,m:5:6.7") == "5:6.7"

    def test_empty(self):
        assert _extract_m_section("") is None

    def test_no_m(self):
        assert _extract_m_section("Sg:n:1,2:ht") is None


# ----------------------------------------------------------------------------
# Component rewards
# ----------------------------------------------------------------------------


class TestRewardValidParse:
    def test_invalid_xml(self):
        assert reward_valid_parse(parse_markush("garbage")) == 0.0

    def test_empty_cxsmiles(self):
        # <cxsmi></cxsmi> has empty content → not valid.
        assert reward_valid_parse(parse_markush("<markush><cxsmi></cxsmi></markush>")) == 0.0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_parseable_backbone(self):
        p = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        # With RDKit, a parseable CC backbone gets full credit.
        assert reward_valid_parse(p) == 1.0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_unparseable_backbone_partial_credit(self):
        # Valid XML, but the SMILES is garbage that RDKit can't parse.
        p = parse_markush("<markush><cxsmi>!!!@@@</cxsmi></markush>")
        score = reward_valid_parse(p)
        # Either 0.5 (unparseable but XML valid) or maybe 1.0 if RDKit is lenient;
        # crucially it must not crash and must return a finite score in [0, 1].
        assert 0.0 <= score <= 1.0


def test_compute_reward_breakdown_logs_components():
    gt = "<markush><cxsmi><r>R1</r>CC|Sg:n:1,2:ht|</cxsmi><stable>R1:methyl</stable></markush>"
    breakdown = compute_reward_breakdown(gt, gt)

    assert breakdown["total"] == compute_reward(gt, gt)
    assert {
        "valid",
        "backbone",
        "backbone_tanimoto",
        "r_groups",
        "sg",
        "sg_extra",
        "m",
        "stable",
        "structure",
        "position_graph_equal",
        "position_graph_valid_gt",
        "position_graph_valid_pred",
        "weights",
    }.issubset(breakdown)
    assert breakdown["valid"] > 0
    assert breakdown["r_groups"] == 1.0
    assert breakdown["structure"] == 1.0
    assert breakdown["position_graph_equal"] == 1.0
    assert breakdown["sg_extra"] == 0.0


class TestRewardBackboneMatch:
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_identical_backbones(self):
        gt = parse_markush("<markush><cxsmi>CCO</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CCO</cxsmi></markush>")
        assert reward_backbone_match(pred, gt) == 1.0

    def test_identical_unusual_backbone_text_short_circuits_rdkit(self):
        text = "<markush><cxsmi>O=C(<r>R''</r>c1c<r>Rx</r>ccc1)NC(Cc1ccc(Br)<r>E3</r>(Cl)c1)=NO</cxsmi></markush>"
        parsed = parse_markush(text)
        assert reward_backbone_match(parsed, parsed) == 1.0
        assert reward_backbone_tanimoto(parsed, parsed) == 1.0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_different_canonical_form_same_molecule(self):
        # Benzene aromatic vs Kekulé.
        gt = parse_markush("<markush><cxsmi>c1ccccc1</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>C1=CC=CC=C1</cxsmi></markush>")
        assert reward_backbone_match(pred, gt) == 1.0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_different_molecules(self):
        gt = parse_markush("<markush><cxsmi>CCO</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CCN</cxsmi></markush>")
        assert reward_backbone_match(pred, gt) == 0.0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_wildcards_treated_as_carbon(self):
        # <r>R1</r>C → *C → CC (the function uses re.sub(r"<r>.*?</r>", "C", ...)).
        gt = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi><r>R1</r>C</cxsmi></markush>")
        assert reward_backbone_match(pred, gt) == 1.0

    def test_empty_backbone_returns_zero(self):
        gt = parse_markush("")
        pred = parse_markush("")
        assert reward_backbone_match(pred, gt) == 0.0

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_backbone_tanimoto_gives_continuous_partial_credit(self):
        gt = parse_markush("<markush><cxsmi>CCCO</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CCCN</cxsmi></markush>")
        score = reward_backbone_tanimoto(pred, gt)
        assert 0.0 < score < 1.0


class TestRewardRGroups:
    def test_perfect_match(self):
        pred = {"r_labels": {"R1", "R2"}}
        gt = {"r_labels": {"R1", "R2"}}
        assert reward_r_groups(pred, gt) == 1.0

    def test_no_gt_no_pred_perfect(self):
        assert reward_r_groups({"r_labels": set()}, {"r_labels": set()}) == 1.0

    def test_no_gt_extras_in_pred_penalized(self):
        # Documented harsh behavior: any pred labels when GT has none → 0.
        assert reward_r_groups({"r_labels": {"R1"}}, {"r_labels": set()}) == 0.0

    def test_gt_has_pred_empty(self):
        assert reward_r_groups({"r_labels": set()}, {"r_labels": {"R1"}}) == 0.0

    def test_disjoint_sets_zero(self):
        pred = {"r_labels": {"RA"}}
        gt = {"r_labels": {"RB"}}
        assert reward_r_groups(pred, gt) == 0.0

    def test_partial_overlap_f1(self):
        # pred {R1,R2}, gt {R1,R3} → tp=1, prec=0.5, recall=0.5, F1=0.5
        pred = {"r_labels": {"R1", "R2"}}
        gt = {"r_labels": {"R1", "R3"}}
        assert reward_r_groups(pred, gt) == pytest.approx(0.5)


class TestRewardSgSection:
    def test_neither_has_sg(self):
        pred = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        gt = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert reward_sg_section(pred, gt) == 1.0

    def test_gt_has_no_sg_pred_has_one_zero_precision(self):
        pred = parse_markush("<markush><cxsmi>CC|Sg:n:1,2:ht</cxsmi></markush>")
        gt = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert reward_sg_section(pred, gt) == 0.0
        assert reward_extra_sg(pred, gt) == 1.0

    def test_gt_has_no_sg_pred_has_none_no_extra_penalty(self):
        pred = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        gt = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert reward_extra_sg(pred, gt) == 0.0

    def test_gt_has_sg_pred_missing_zero(self):
        gt = parse_markush("<markush><cxsmi>CC|Sg:n:1,2:ht</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert reward_sg_section(pred, gt) == 0.0

    def test_both_match_exactly(self):
        gt = parse_markush("<markush><cxsmi>CC|Sg:n:1,2:ht</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CC|Sg:n:1,2:ht</cxsmi></markush>")
        assert reward_sg_section(pred, gt) == 1.0

    def test_both_have_sg_partial_overlap(self):
        # Two Sg sections each, one matches.
        gt = parse_markush("<markush><cxsmi>CC|Sg:n:1,2:ht,Sg:n:3,4:eu</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CC|Sg:n:1,2:ht,Sg:n:5,6:eu</cxsmi></markush>")
        score = reward_sg_section(pred, gt)
        # One section matches out of 3 unique → 1/3.
        assert 0.0 < score < 1.0


class TestRewardMSection:
    def test_neither_has_m(self):
        pred = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        gt = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert reward_m_section(pred, gt) == 1.0

    def test_gt_has_m_pred_missing(self):
        gt = parse_markush("<markush><cxsmi>CC|m:1:2.3</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        assert reward_m_section(pred, gt) == 0.0

    def test_both_match(self):
        gt = parse_markush("<markush><cxsmi>CC|m:1:2.3</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CC|m:1:2.3</cxsmi></markush>")
        assert reward_m_section(pred, gt) == 1.0

    def test_both_have_but_different(self):
        gt = parse_markush("<markush><cxsmi>CC|m:1:2.3</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CC|m:5:6.7</cxsmi></markush>")
        # Both have m-section but content differs → 0.5 partial credit.
        assert reward_m_section(pred, gt) == 0.5

    def test_extra_m_in_pred_partial(self):
        gt = parse_markush("<markush><cxsmi>CC</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi>CC|m:1:2.3</cxsmi></markush>")
        # GT no m, pred has m → 0.5 partial penalty.
        assert reward_m_section(pred, gt) == 0.5


class TestRewardStable:
    def test_perfect_match(self):
        pred = {"stable": "R0:methyl<n>ethyl<ns>R1:hydroxyl"}
        gt = {"stable": "R0:methyl<n>ethyl<ns>R1:hydroxyl"}
        assert reward_stable(pred, gt) == 1.0

    def test_alternatives_set_unordered(self):
        # parse_stable uses sets, so order is irrelevant.
        pred = {"stable": "R0:ethyl<n>methyl"}
        gt = {"stable": "R0:methyl<n>ethyl"}
        assert reward_stable(pred, gt) == 1.0

    def test_neither_has_stable(self):
        assert reward_stable({"stable": ""}, {"stable": ""}) == 1.0

    def test_gt_no_stable_pred_has(self):
        # Documented behavior: extras get partial 0.5.
        assert reward_stable({"stable": "R0:methyl"}, {"stable": ""}) == 0.5

    def test_gt_has_stable_pred_missing_zero(self):
        assert reward_stable({"stable": ""}, {"stable": "R0:methyl"}) == 0.0

    def test_partial_match_fraction(self):
        pred = {"stable": "R0:methyl<ns>R1:wrong"}
        gt = {"stable": "R0:methyl<ns>R1:hydroxyl"}
        # 1 of 2 labels matches exactly.
        assert reward_stable(pred, gt) == 0.5


# ----------------------------------------------------------------------------
# compute_reward total
# ----------------------------------------------------------------------------


class TestComputeReward:
    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_perfect_prediction_high_score(self):
        gt = "<markush><cxsmi><r>R1</r>C1=CC=CC=C1|Sg:n:1,2:ht</cxsmi><stable>R1:methyl<n>ethyl<ns></stable></markush>"
        # Perfect duplicate of GT — should be high.
        score = compute_reward(gt, gt)
        assert score >= 6.0

    def test_garbage_returns_zero(self):
        gt = "<markush><cxsmi>CC</cxsmi></markush>"
        # Garbage text should NOT get vacuous credit for "missing optional sections".
        assert compute_reward("```json {}", gt) == 0.0

    def test_empty_pred_returns_zero(self):
        assert compute_reward("", "<markush><cxsmi>CC</cxsmi></markush>") == 0.0

    def test_returns_finite_for_malformed(self):
        # Various malformed inputs must produce finite (non-NaN, non-inf) values.
        gt = "<markush><cxsmi>CC</cxsmi></markush>"
        for bad in ["<markush><cxsmi>", "{}", "<r>R1</r>", "<markush></markush>"]:
            score = compute_reward(bad, gt)
            assert math.isfinite(score)
            assert score >= 0.0

    def test_weights_respected(self):
        gt = "<markush><cxsmi>CC</cxsmi></markush>"
        pred = "<markush><cxsmi>CC</cxsmi></markush>"
        weights = {
            "valid": 0.0,
            "backbone": 0.0,
            "r_groups": 0.0,
            "sg": 0.0,
            "m": 0.0,
            "stable": 0.0,
        }
        # All zero weights -> total reward 0.
        assert compute_reward(pred, gt, weights=weights) == 0.0

    def test_weight_override_changes_total(self):
        gt = "<markush><cxsmi>CC</cxsmi></markush>"
        pred = "<markush><cxsmi>CC</cxsmi></markush>"
        baseline = compute_reward(pred, gt)
        bumped = compute_reward(
            pred,
            gt,
            weights={
                "valid": 100.0,
                "backbone": 0.0,
                "r_groups": 0.0,
                "sg": 0.0,
                "m": 0.0,
                "stable": 0.0,
            },
        )
        # 100 * reward_valid_parse should dominate when the rest is zeroed out.
        assert bumped >= 50.0  # gives ample headroom (RDKit valid → r1=1.0 → 100)
        # And the baseline (smaller weights) is much smaller than 100.
        assert baseline < bumped

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_extra_sg_is_penalized_by_default_reward(self):
        gt = "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
        exact = "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
        extra_sg = "<markush><cxsmi>CCO|Sg:n:0,1,2:ht</cxsmi><stable></stable></markush>"

        exact_breakdown = compute_reward_breakdown(exact, gt)
        extra_breakdown = compute_reward_breakdown(extra_sg, gt)

        assert exact_breakdown["structure"] == 1.0
        assert extra_breakdown["structure"] == 0.0
        assert extra_breakdown["sg_extra"] == 1.0
        assert extra_breakdown["total"] < exact_breakdown["total"]

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_extra_sg_is_penalized_by_strict_graph_sg_precision_weights(self):
        weights = {
            "valid": 0.5,
            "backbone": 1.0,
            "r_groups": 0.5,
            "sg": 1.0,
            "sg_extra": -3.0,
            "m": 0.5,
            "stable": 0.5,
            "structure": 6.0,
            "mg2_cxsmi": 0.0,
            "mg2_stable": 0.0,
            "mg2_full": 0.0,
        }
        gt = "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
        exact = "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
        extra_sg = "<markush><cxsmi>CCO|Sg:n:0,1,2:ht</cxsmi><stable></stable></markush>"

        exact_breakdown = compute_reward_breakdown(exact, gt, weights=weights)
        extra_breakdown = compute_reward_breakdown(extra_sg, gt, weights=weights)

        assert exact_breakdown["position_graph_equal"] == 1.0
        assert extra_breakdown["position_graph_equal"] == 0.0
        assert extra_breakdown["sg_extra"] == 1.0
        assert exact_breakdown["total"] > extra_breakdown["total"]

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_mg2_exact_bonus_can_target_primary_metric(self):
        gt = "<markush><cxsmi><r>R1</r>CC</cxsmi><stable>R1:methyl</stable></markush>"
        weights = {
            "valid": 0.0,
            "backbone": 0.0,
            "r_groups": 0.0,
            "sg": 0.0,
            "m": 0.0,
            "stable": 0.0,
            "mg2_cxsmi": 3.0,
            "mg2_stable": 1.0,
            "mg2_full": 2.0,
        }
        breakdown = compute_reward_breakdown(gt, gt, weights=weights)
        assert breakdown["mg2_cxsmi"] == 1.0
        assert breakdown["mg2_stable"] == 1.0
        assert breakdown["mg2_full"] == 1.0
        assert breakdown["total"] == pytest.approx(6.0)

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit required")
    def test_mg2_exact_scores_zero_for_mismatched_cxsmi(self):
        gt = "<markush><cxsmi>CCO</cxsmi><stable></stable></markush>"
        pred = "<markush><cxsmi>CCN</cxsmi><stable></stable></markush>"
        scores = reward_mg2_exact_scores(pred, gt)
        assert scores["mg2_cxsmi"] == 0.0

    def test_returns_finite_when_gt_is_invalid(self):
        # GT garbage — function must still produce a number, not crash.
        pred = "<markush><cxsmi>CC</cxsmi></markush>"
        score = compute_reward(pred, "garbage")
        assert math.isfinite(score)

    def test_perfect_pred_invalid_gt_xml_still_finite(self):
        # GT lacks the XML wrapper entirely. We should not crash; pred is treated
        # as valid, but most cross-checks against gt will be zero-valued.
        pred = "<markush><cxsmi>CC</cxsmi></markush>"
        score = compute_reward(pred, "")
        assert math.isfinite(score)
        # Pred valid_parse contributes positive credit, but backbone is 0 since gt empty.
        assert score >= 0.0


# ----------------------------------------------------------------------------
# exact structure component
# ----------------------------------------------------------------------------


class TestRewardStructureMatch:
    def test_serialization_variant_matches(self):
        gt = parse_markush("<markush><cxsmi>Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6</cxsmi></markush>")
        pred = parse_markush(
            "<markush><cxsmi><r>Ym</r>C.NC1=CC=CC=C1|m:1:3.4.5.6.7.8</cxsmi></markush>"
        )

        assert reward_structure_match(pred, gt) == 1.0

    def test_label_mismatch_does_not_match(self):
        gt = parse_markush("<markush><cxsmi><r>R1</r>CC</cxsmi></markush>")
        pred = parse_markush("<markush><cxsmi><r>R2</r>CC</cxsmi></markush>")

        assert reward_structure_match(pred, gt) == 0.0
