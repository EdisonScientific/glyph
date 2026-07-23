#!/usr/bin/env python3
"""Tests for glyph.markush.eval.predict module.

Tests cover the core utility functions:
- canonicalize_smiles: SMILES canonicalization with R-group handling
- canonical_match: Canonical comparison of two CXSMILES strings
- backbone_inchi_match: InChI-based backbone comparison
- extract_r_labels: R-group label extraction from CXSMILES
"""

import pytest

from glyph.markush.eval.predict import (
    backbone_inchi_match,
    canonical_match,
    canonicalize_smiles,
    clean_prediction,
    extract_cxsmi,
    extract_r_labels,
    extract_stable,
    has_valid_markush_xml,
)

# -----------------------------------------------------------------------------
# extract_r_labels tests
# -----------------------------------------------------------------------------


class TestExtractRLabels:
    """Tests for extract_r_labels function."""

    def test_single_r_label(self):
        """Extract a single R-label from CXSMILES."""
        cxsmiles = "<r>R1</r>c1ccccc1"
        labels = extract_r_labels(cxsmiles)
        assert labels == {"R1"}

    def test_multiple_r_labels(self):
        """Extract multiple R-labels from CXSMILES."""
        cxsmiles = "<r>R1</r>c1ccc(<r>R2</r>)cc1"
        labels = extract_r_labels(cxsmiles)
        assert labels == {"R1", "R2"}

    def test_no_r_labels(self):
        """Return empty set when no R-labels present."""
        cxsmiles = "c1ccccc1"
        labels = extract_r_labels(cxsmiles)
        assert labels == set()

    def test_empty_string(self):
        """Return empty set for empty string."""
        labels = extract_r_labels("")
        assert labels == set()

    def test_case_insensitive_tags(self):
        """Tags should be matched case-insensitively."""
        cxsmiles = "<R>R1</R>c1ccc(<r>R2</r>)cc1"
        labels = extract_r_labels(cxsmiles)
        assert labels == {"R1", "R2"}

    def test_various_r_label_names(self):
        """Handle various R-label naming conventions."""
        cxsmiles = "<r>R1</r>c1ccc(<r>X2</r>)c(<r>Z</r>)c1<r>Rn</r>"
        labels = extract_r_labels(cxsmiles)
        assert labels == {"R1", "X2", "Z", "Rn"}


# -----------------------------------------------------------------------------
# canonicalize_smiles tests
# -----------------------------------------------------------------------------


class TestCanonicalizeSmiles:
    """Tests for canonicalize_smiles function."""

    def test_simple_benzene(self):
        """Canonicalize simple benzene ring."""
        result = canonicalize_smiles("c1ccccc1")
        assert result is not None
        assert result == "c1ccccc1"

    def test_r_groups_replaced(self):
        """R-groups should be replaced with wildcards and canonicalized."""
        smiles = "<r>R1</r>c1ccccc1"
        result = canonicalize_smiles(smiles)
        assert result is not None
        assert "*" in result

    def test_extensions_stripped(self):
        """Extensions after | should be stripped."""
        smiles = "c1ccccc1|Sg:n:1,2:ht"
        result = canonicalize_smiles(smiles)
        assert result is not None
        assert "|" not in result

    def test_invalid_smiles(self):
        """Return None for invalid SMILES."""
        result = canonicalize_smiles("not_a_smiles_XYZ123")
        assert result is None

    def test_empty_string(self):
        """Return None for empty string."""
        result = canonicalize_smiles("")
        assert result is None

    def test_none_input(self):
        """Return None for None input."""
        result = canonicalize_smiles(None)  # type: ignore
        assert result is None


# -----------------------------------------------------------------------------
# canonical_match tests
# -----------------------------------------------------------------------------


class TestCanonicalMatch:
    """Tests for canonical_match function."""

    def test_identical_smiles(self):
        """Identical SMILES should match."""
        assert canonical_match("c1ccccc1", "c1ccccc1") is True

    def test_equivalent_smiles(self):
        """Equivalent SMILES with different representations should match."""
        # Both are benzene
        assert canonical_match("c1ccccc1", "C1=CC=CC=C1") is True

    def test_different_molecules(self):
        """Different molecules should not match."""
        assert canonical_match("c1ccccc1", "CCO") is False

    def test_different_r_group_labels_do_not_match(self):
        """R labels are atom-bound and label identity matters."""
        s1 = "<r>R1</r>c1ccccc1"
        s2 = "<r>R2</r>c1ccccc1"  # Different label but same backbone
        assert canonical_match(s1, s2) is False

    def test_serialization_variant_with_same_r_group_matches(self):
        """Equivalent atom ordering with the same label should match."""
        s1 = "Nc1ccccc1.C<r>Ym</r>|m:7:1.2.3.4.5.6"
        s2 = "<r>Ym</r>C.NC1=CC=CC=C1|m:1:3.4.5.6.7.8"
        assert canonical_match(s1, s2) is True

    def test_invalid_smiles_no_match(self):
        """Invalid SMILES should not match."""
        assert canonical_match("invalid", "c1ccccc1") is False


# -----------------------------------------------------------------------------
# backbone_inchi_match tests
# -----------------------------------------------------------------------------


class TestBackboneInchiMatch:
    """Tests for backbone_inchi_match function."""

    def test_identical_backbones(self):
        """Identical backbones should match via InChI."""
        result = backbone_inchi_match("c1ccccc1", "c1ccccc1")
        assert result is True

    def test_r_groups_as_carbon(self):
        """R-groups should be replaced with C for InChI generation."""
        s1 = "<r>R1</r>c1ccccc1"
        s2 = "Cc1ccccc1"  # Methyl group in place of R1
        result = backbone_inchi_match(s1, s2)
        # Both should have same backbone after R→C substitution
        assert result is True

    def test_different_backbones(self):
        """Different backbones should not match."""
        result = backbone_inchi_match("c1ccccc1", "c1ccncc1")
        assert result is False

    def test_invalid_returns_none(self):
        """Invalid SMILES should return None."""
        result = backbone_inchi_match("invalid_smiles", "c1ccccc1")
        assert result is None


# -----------------------------------------------------------------------------
# XML extraction tests
# -----------------------------------------------------------------------------


class TestXmlExtraction:
    """Tests for XML extraction functions."""

    def test_extract_cxsmi(self):
        """Extract cxsmi content from XML."""
        text = "<markush><cxsmi>c1ccccc1</cxsmi><stable></stable></markush>"
        result = extract_cxsmi(text)
        assert result == "c1ccccc1"

    def test_extract_cxsmi_case_insensitive(self):
        """cxsmi extraction should be case-insensitive."""
        text = "<CXSMI>c1ccccc1</CXSMI>"
        result = extract_cxsmi(text)
        assert result == "c1ccccc1"

    def test_extract_stable(self):
        """Extract stable content from XML."""
        text = "<stable>R1:methyl<n>ethyl</stable>"
        result = extract_stable(text)
        assert result == "R1:methyl<n>ethyl"

    def test_extract_stable_case_insensitive(self):
        """stable extraction should be case-insensitive."""
        text = "<STABLE>R1:test</STABLE>"
        result = extract_stable(text)
        assert result == "R1:test"

    def test_has_valid_markush_xml(self):
        """Check valid markush XML structure."""
        valid = "<markush><cxsmi>c1ccccc1</cxsmi></markush>"
        invalid = "just text"
        assert has_valid_markush_xml(valid) is True
        assert has_valid_markush_xml(invalid) is False

    def test_has_valid_markush_xml_empty_cxsmi(self):
        """Empty cxsmi should be invalid."""
        text = "<markush><cxsmi></cxsmi></markush>"
        assert has_valid_markush_xml(text) is False


# -----------------------------------------------------------------------------
# clean_prediction tests
# -----------------------------------------------------------------------------


class TestCleanPrediction:
    """Tests for clean_prediction function."""

    def test_removes_thinking_blocks(self):
        """Remove <think>...</think> blocks."""
        text = "<think>reasoning here</think>actual output"
        result = clean_prediction(text)
        assert result == "actual output"

    def test_removes_chat_artifacts(self):
        """Remove chat template artifacts."""
        text = "output<|im_end|>extra"
        result = clean_prediction(text)
        assert result == "output"

    def test_handles_empty_string(self):
        """Handle empty string."""
        assert clean_prediction("") == ""

    def test_thinking_case_insensitive(self):
        """Thinking block removal should be case-insensitive."""
        text = "<THINK>reasoning</THINK>output"
        result = clean_prediction(text)
        assert result == "output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
