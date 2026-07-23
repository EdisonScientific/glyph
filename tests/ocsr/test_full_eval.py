"""Tests for the canonical OCSR eval module (``glyph.ocsr.full_eval``).

These use only synthetic SMILES and dummy report payloads; they exercise the
grading/report/write path and the opt-in majority-vote fold without needing a
checkpoint or GPU. The grading math itself lives in ``glyph.ocsr.eval`` and is
not re-implemented here.
"""

from __future__ import annotations

import json
from pathlib import Path

from glyph.ocsr.benchmark import SOURCE_HF, SOURCE_MOLSCRIBE, SOURCE_NONE
from glyph.ocsr.eval import canonical_smiles
from glyph.ocsr.full_eval import (
    format_summary,
    grade_majority_vote,
    grade_uspto_predictions,
    parse_vote_ks,
    write_eval_report,
    write_per_row_predictions,
)

# A four-row synthetic benchmark that exercises all three conventions and both
# denominators. Row sources mirror the real complete-label CSV.
_LABEL_ROWS = [
    # HF subset row: cis/trans differs (A miss), chirality kept (B hit), graph hit.
    {"image_id": "hf-1", "smiles": r"C/C=C\[C@H](F)Cl", "source_of_label": SOURCE_HF},
    # HF subset row: chirality differs (A + B miss), graph hit.
    {"image_id": "hf-2", "smiles": "C[C@@H](N)C(=O)O", "source_of_label": SOURCE_HF},
    # MolScribe-recovered row (outside the HF subset): exact match under A/B/C.
    {"image_id": "ms-1", "smiles": "OCC", "source_of_label": SOURCE_MOLSCRIBE},
    # Unscoreable ground truth: excluded from the numerator, counted in the total.
    {"image_id": "none-1", "smiles": "not_a_smiles", "source_of_label": SOURCE_NONE},
]
_PREDICTIONS = ["C/C=C/[C@H](F)Cl", "C[C@H](N)C(=O)O", "CCO", "CCO"]


def test_grade_uspto_predictions_two_denominators_and_conventions():
    report = grade_uspto_predictions(_LABEL_ROWS, _PREDICTIONS)

    full = report["full_5719"]
    assert full["denominator"] == 4
    assert full["n_scoreable_gt"] == 3
    assert full["n_unscoreable_gt"] == 1
    assert full["A_canon"]["n_correct"] == 1  # only ms-1
    assert full["B_chirality"]["n_correct"] == 2  # hf-1 + ms-1
    assert full["C_graph"]["n_correct"] == 3  # all three scoreable rows
    assert full["A_canon"]["pct"] == 25.0
    assert full["B_chirality"]["pct"] == 50.0
    assert full["C_graph"]["pct"] == 75.0

    subset = report["parseable_5704"]
    assert subset["denominator"] == 2  # only the two HF rows
    assert subset["A_canon"]["n_correct"] == 0
    assert subset["B_chirality"]["n_correct"] == 1
    assert subset["C_graph"]["n_correct"] == 2

    counts = report["counts"]
    assert counts["total_rows"] == 4
    assert counts["parseable_subset_rows"] == 2
    assert counts["scoreable_gt_rows"] == 3
    assert counts["unscoreable_gt_rows"] == 1


def test_write_eval_report_creates_missing_parent_dirs(tmp_path: Path):
    """Regression for the fresh-GLYPH_DATA_ROOT output-dir bug (docs-audit #5)."""

    report = grade_uspto_predictions(_LABEL_ROWS, _PREDICTIONS)
    nested = tmp_path / "does" / "not" / "exist" / "report.json"
    assert not nested.parent.exists()

    written = write_eval_report(report, nested)

    assert written == nested
    assert nested.is_file()
    assert json.loads(nested.read_text())["counts"]["total_rows"] == 4


def test_write_per_row_predictions_creates_missing_parent_dirs(tmp_path: Path):
    """Regression for the fresh-GLYPH_DATA_ROOT per-row output-dir bug."""

    nested = tmp_path / "fresh" / "root" / "eval" / "per_row.jsonl"
    assert not nested.parent.exists()

    written = write_per_row_predictions(_LABEL_ROWS, _PREDICTIONS, nested)

    assert written == nested
    assert nested.is_file()
    lines = [json.loads(line) for line in nested.read_text().splitlines()]
    assert len(lines) == len(_LABEL_ROWS)
    assert lines[0]["image_id"] == "hf-1"
    assert lines[0]["source_of_label"] == SOURCE_HF


def test_parse_vote_ks_parses_and_validates():
    assert parse_vote_ks("8") == [8]
    assert parse_vote_ks("16,4,8") == [4, 8, 16]  # sorted + de-duplicated
    assert parse_vote_ks(None) == []
    assert parse_vote_ks("") == []
    try:
        parse_vote_ks("0")
    except ValueError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("expected ValueError for non-positive K")


def test_grade_majority_vote_falls_back_to_greedy_and_reuses_grader():
    label_rows = [
        {"image_id": "hf-a", "smiles": "OCC", "source_of_label": SOURCE_HF},
        {"image_id": "hf-b", "smiles": "c1ccccc1", "source_of_label": SOURCE_HF},
    ]
    greedy_predictions = ["CCC", "CCC"]  # both wrong; used only as fallback
    candidates_by_row = [
        # Modal valid canonical form wins the vote (matches ethanol GT).
        [
            {"canonical": canonical_smiles("CCO"), "mean_logprob": -0.2},
            {"canonical": canonical_smiles("CCO"), "mean_logprob": -0.3},
        ],
        # All-invalid row: non-oracle fallback to the (wrong) greedy prediction.
        [
            {"canonical": None, "mean_logprob": -1.0},
            {"canonical": None, "mean_logprob": -1.1},
        ],
    ]

    payload = grade_majority_vote(label_rows, greedy_predictions, candidates_by_row, [2])

    assert set(payload) == {"2"}
    block = payload["2"]
    assert block["full_5719"]["A_canon"]["n_correct"] == 1  # only the voted ethanol row
    assert block["parseable_5704"]["A_canon"]["n_correct"] == 1
    diagnostics = block["vote_diagnostics"]
    assert diagnostics["k"] == 2
    assert diagnostics["all_invalid_count"] == 1
    assert diagnostics["tie_count"] == 0


def test_format_summary_includes_majority_vote_block():
    report = grade_uspto_predictions(_LABEL_ROWS, _PREDICTIONS)
    report["majority_vote"] = grade_majority_vote(
        [row for row in _LABEL_ROWS if row["source_of_label"] == SOURCE_HF],
        ["CCC", "CCC"],
        [
            [{"canonical": canonical_smiles("CCO"), "mean_logprob": -0.2}],
            [{"canonical": None, "mean_logprob": -1.0}],
        ],
        [8],
    )

    summary = format_summary(report)

    # Greedy A/B/C headings still render exactly as before...
    assert "FULL USPTO OCSR EVAL -- three stereochemistry conventions" in summary
    assert "A canonical/full-stereo" in summary
    # ...and the MV block is appended, not substituted.
    assert "MAJORITY VOTE  MV@8" in summary
