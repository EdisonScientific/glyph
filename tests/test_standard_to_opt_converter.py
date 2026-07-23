"""Direct round-trip coverage for the shipped ``opt<->standard`` CXSMILES converters.

These converters (``glyph.markush.data.pipeline.standard_to_opt_cxsmiles`` and
``opt_to_standard_cxsmiles``) are the shared primitive used by ``prepare_sft``,
target preparation, the MolParser conversion, and the SFT mix. This is a minimal
direct test that both directions are structure-preserving (Markush-graph equal)
and round-trip back to the source.
"""

from __future__ import annotations

import pytest

from glyph.markush.data.pipeline import opt_to_standard_cxsmiles, standard_to_opt_cxsmiles
from glyph.markush.eval.markush_graph import markush_graph_equal

# (standard |$...$| CXSMILES, equivalent <r>LABEL</r> opt CXSMILES) pairs.
PAIRS = [
    (
        "*CC(*)O |$R1;;;R2;$|",
        "<r>R1</r>CC(<r>R2</r>)O",
    ),
    (
        "[1*]C1(C)CC2CC1C1(CCC)CC(C)C21 |$X;;;;;;;;;;;;;;$,|",
        "<r>X</r>C1(C)CC2CC1C1(CCC)CC(C)C21",
    ),
    (
        "*[N+]([H])(*O[H])*O[H] |$R4;;;R5;;;R5;;$,,,Sg:n:6,7:x:ht:::,Sg:n:3,4:y:ht:::,|",
        "<r>R4</r>[N+]([H])(<r>R5</r>O[H])<r>R5</r>O[H]|Sg:n:6,7:x:ht:::,Sg:n:3,4:y:ht:::",
    ),
]


def _assert_graph_equal(left: str, right: str) -> None:
    result = markush_graph_equal(left, right)
    assert result.valid_gt, result.gt_error
    assert result.valid_pred, result.pred_error
    assert result.equal is True, f"{left!r} != {right!r}"


@pytest.mark.parametrize(("standard", "opt"), PAIRS)
def test_standard_to_opt_matches_reference(standard: str, opt: str) -> None:
    assert standard_to_opt_cxsmiles(standard) == opt


@pytest.mark.parametrize(("standard", "opt"), PAIRS)
def test_opt_to_standard_is_structure_preserving(standard: str, opt: str) -> None:
    converted = opt_to_standard_cxsmiles(opt)
    _assert_graph_equal(converted, standard)


@pytest.mark.parametrize(("standard", "opt"), PAIRS)
def test_converters_round_trip(standard: str, opt: str) -> None:
    _assert_graph_equal(opt_to_standard_cxsmiles(standard_to_opt_cxsmiles(standard)), standard)
    _assert_graph_equal(standard_to_opt_cxsmiles(opt_to_standard_cxsmiles(opt)), opt)
