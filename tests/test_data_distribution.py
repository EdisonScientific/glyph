from __future__ import annotations

import random

from glyph.markush.data.distribution import (
    compare_profiles,
    features_from_record,
    profile_records,
    target_counts_from_profile,
)
from glyph.markush.data.pipeline import (
    DecorationConfig,
    decorate_scaffold,
    decoration_config_for_profile,
    generate_decorated_samples,
)


def test_features_from_chat_record_detects_eval_like_markush_features():
    record = {
        "messages": [
            {"role": "user", "content": "<image>\nExtract."},
            {
                "role": "assistant",
                "content": (
                    "<markush><cxsmi>C<r>R'</r>.<r>L30</r>c1ccccc1"
                    "|Sg:n:1,2,3:n:ht,m:0:1.2.3</cxsmi>"
                    "<stable>R':alkyl<ns>L30:SO2</stable></markush>"
                ),
            },
        ]
    }

    features = features_from_record(record)

    assert features["rgroup_count"] == 2
    assert features["fragment"] is True
    assert features["has_sg"] is True
    assert features["has_m"] is True
    assert features["stable_row_count"] == 2
    assert "lettered_r" in features["label_categories"]
    assert "letter_number" in features["label_categories"]


def test_profile_and_compare_records_reports_histogram_distance():
    target = profile_records(
        [
            {"cxsmiles_opt": "<r>R1</r>CC"},
            {"cxsmiles_opt": "C<r>R2</r>.<r>R3</r>CC|m:0:1.2.3"},
        ]
    )
    candidate = profile_records([{"cxsmiles_opt": "<r>R1</r>CC"}])

    comparison = compare_profiles(candidate, target)

    assert target["records"] == 2
    assert target["rates"]["fragment"] == 0.5
    assert comparison["total_variation"]["feature_combo"] > 0


def test_target_counts_from_profile_uses_largest_remainders():
    profile = profile_records(
        [
            {"cxsmiles_opt": "<r>R1</r>CC"},
            {"cxsmiles_opt": "<r>R2</r>CCC"},
            {"cxsmiles_opt": "C<r>R3</r>.<r>R4</r>CC"},
        ]
    )

    counts = target_counts_from_profile(profile, 6, histogram="rgroup_bin")

    assert counts == {"1": 4, "2": 2}


def test_decoration_profiles_broaden_eval_aligned_generation():
    cfg = decoration_config_for_profile("eval_aligned")

    assert cfg.zero_rgroup_probability > 0.1
    assert cfg.fragment_probability > 0.3
    assert cfg.high_rgroup_max >= 20


def test_generate_decorated_samples_accepts_target_profile():
    scaffolds = ["c1ccccc1CCO", "COc1ccccc1CCN"]
    target_profile = profile_records(
        [
            {"cxsmiles_opt": "<r>R1</r>CC"},
            {"cxsmiles_opt": "<r>R2</r>CCC"},
        ]
    )
    cfg = DecorationConfig(
        min_rgroups=1,
        max_rgroups=1,
        label_style="R_only",
        sg_probability=0.0,
        m_probability=0.0,
        zero_rgroup_probability=0.0,
        fragment_probability=0.0,
    )

    samples, stats = generate_decorated_samples(
        scaffolds,
        num_samples=3,
        seed=5,
        config=cfg,
        target_profile=target_profile,
        profile_histogram="rgroup_bin",
    )

    assert len(samples) == 3
    assert stats["profile_target_bins"] == 1


def test_label_pool_can_generate_eval_like_r_labels():
    cfg = DecorationConfig(
        min_rgroups=3,
        max_rgroups=3,
        sg_probability=0.0,
        m_probability=0.0,
        zero_rgroup_probability=0.0,
    )
    sample = decorate_scaffold("c1ccccc1CCCO", config=cfg, rng=random.Random(1))

    assert sample is not None
    assert len(sample.rgroup_labels) == 3
