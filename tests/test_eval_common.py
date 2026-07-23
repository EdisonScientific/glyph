#!/usr/bin/env python3
"""Tests for ``glyph.markush.eval.common`` — shared parsing primitives.

``glyph.markush.eval.common`` is the single source of truth for tag
extraction, ``<r>`` stripping, ``<stable>`` parsing, and prediction cleanup.

The tests focus on observable behaviour:

* tag extraction (presence / absence / case / multi-line)
* chat-artifact stripping
* empty-string / whitespace / partial-output handling
* idempotency (applying twice == applying once)
* the composite ``prediction_to_cxsmiles_opt`` helper
"""

import pytest

from glyph.markush.eval.common import (
    clean_prediction,
    prediction_to_cxsmiles_opt,
)
from glyph.markush.eval.predict import (
    extract_cxsmi,
    extract_stable,
)

# ---------------------------------------------------------------------------
# extract_cxsmi
# ---------------------------------------------------------------------------


class TestExtractCxsmi:
    def test_basic_extraction(self):
        assert extract_cxsmi("<cxsmi>c1ccccc1</cxsmi>") == "c1ccccc1"

    def test_strips_inner_whitespace(self):
        assert extract_cxsmi("<cxsmi>   c1ccccc1\n</cxsmi>") == "c1ccccc1"

    def test_case_insensitive_tags(self):
        # The canonical extractor in eval.common is case-insensitive;
        # predict re-exports it and so must agree.
        assert extract_cxsmi("<CXSMI>c1ccccc1</CXSMI>") == "c1ccccc1"
        assert extract_cxsmi("<CxSmi>c1ccccc1</cxSMI>") == "c1ccccc1"

    def test_multiline_content(self):
        text = "<cxsmi>c1cc\nccc1</cxsmi>"
        # Newline is preserved inside the captured group.
        assert extract_cxsmi(text) == "c1cc\nccc1"

    def test_returns_none_when_missing(self):
        assert extract_cxsmi("plain text, no tags") is None

    def test_returns_none_on_empty_input(self):
        assert extract_cxsmi("") is None

    def test_returns_first_match(self):
        text = "<cxsmi>first</cxsmi> noise <cxsmi>second</cxsmi>"
        assert extract_cxsmi(text) == "first"

    def test_unclosed_tag_returns_none(self):
        # Partial / truncated model output: we want graceful failure.
        assert extract_cxsmi("<cxsmi>c1ccccc1") is None


# ---------------------------------------------------------------------------
# extract_stable
# ---------------------------------------------------------------------------


class TestExtractStable:
    def test_basic_extraction(self):
        assert extract_stable("<stable>R1:methyl</stable>") == "R1:methyl"

    def test_case_insensitive(self):
        assert extract_stable("<STABLE>R1:methyl</STABLE>") == "R1:methyl"

    def test_returns_none_when_missing(self):
        assert extract_stable("no stable tag here") is None

    def test_returns_none_on_empty(self):
        assert extract_stable("") is None

    def test_preserves_nested_markup(self):
        # The substitution table may legitimately contain other tag-like
        # tokens (e.g. <n>ethyl</n>); extraction should not eat them.
        text = "<stable>R1:methyl<n>ethyl</n></stable>"
        assert extract_stable(text) == "R1:methyl<n>ethyl</n>"


# ---------------------------------------------------------------------------
# clean_prediction
# ---------------------------------------------------------------------------


class TestCleanPrediction:
    def test_empty_string(self):
        assert clean_prediction("") == ""

    def test_whitespace_only(self):
        assert clean_prediction("   \n\t ") == ""

    def test_strips_thinking_block(self):
        assert clean_prediction("<think>reasoning</think>final") == "final"

    def test_strips_thinking_block_multiline(self):
        text = "<think>line 1\nline 2\nline 3</think>\nfinal output"
        assert clean_prediction(text) == "final output"

    def test_strips_thinking_block_case_insensitive(self):
        assert clean_prediction("<THINK>x</THINK>out") == "out"

    def test_strips_multiple_thinking_blocks(self):
        text = "<think>a</think>middle<think>b</think>end"
        assert clean_prediction(text) == "middleend"

    def test_truncates_at_im_end(self):
        assert clean_prediction("answer<|im_end|>garbage") == "answer"

    def test_truncates_at_endoftext(self):
        assert clean_prediction("answer<|endoftext|>noise") == "answer"

    def test_truncates_at_im_start(self):
        assert clean_prediction("answer<|im_start|>noise") == "answer"

    def test_trims_surrounding_whitespace(self):
        assert clean_prediction("   hello\n") == "hello"

    def test_no_modifications_for_clean_input(self):
        text = "<markush><cxsmi>c1ccccc1</cxsmi></markush>"
        assert clean_prediction(text) == text

    def test_idempotent(self):
        # Applying clean_prediction twice must equal applying it once.
        raw = "<think>reasoning</think>answer<|im_end|>junk"
        once = clean_prediction(raw)
        twice = clean_prediction(once)
        assert once == twice == "answer"


# ---------------------------------------------------------------------------
# prediction_to_cxsmiles_opt
# ---------------------------------------------------------------------------


class TestPredictionToCxsmilesOpt:
    def test_extracts_from_full_envelope(self):
        text = "<markush><cxsmi>c1ccccc1</cxsmi></markush>"
        assert prediction_to_cxsmiles_opt(text) == "c1ccccc1"

    def test_strips_thinking_then_extracts(self):
        text = "<think>thinking</think><cxsmi>c1ccccc1</cxsmi>"
        assert prediction_to_cxsmiles_opt(text) == "c1ccccc1"

    def test_strips_im_end_then_extracts(self):
        text = "<cxsmi>c1ccccc1</cxsmi><|im_end|>followup"
        assert prediction_to_cxsmiles_opt(text) == "c1ccccc1"

    def test_falls_back_to_cleaned_text_when_no_cxsmi(self):
        # Model emitted only the raw smiles, no XML envelope.
        text = "<think>reasoning</think>c1ccccc1"
        assert prediction_to_cxsmiles_opt(text) == "c1ccccc1"

    def test_empty_input_returns_empty_string(self):
        # clean_prediction("") -> ""; extract returns None; fallback is "".
        assert prediction_to_cxsmiles_opt("") == ""

    def test_idempotent_when_only_smiles(self):
        # Running the helper on its own output should be stable.
        once = prediction_to_cxsmiles_opt("<cxsmi>c1ccccc1</cxsmi>")
        twice = prediction_to_cxsmiles_opt(once)
        assert once == twice == "c1ccccc1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
