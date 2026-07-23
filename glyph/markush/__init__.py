"""Markush structure recognition from patent images."""

from glyph.markush.fto import (
    ConcreteMatchResult,
    MarkushClaim,
    extract_components,
    match_concrete_smiles,
    normalize_claim,
    parse_claim,
    strict_equal,
)
from glyph.markush.inference import MarkushPrediction, MarkushPredictor
from glyph.markush.targets import (
    ConverterCheckResult,
    TargetNormalizationResult,
    check_cxsmiles_round_trip,
    normalize_markush_target,
)

Predictor = MarkushPredictor

__version__ = "0.1.0"

__all__ = [
    "ConcreteMatchResult",
    "ConverterCheckResult",
    "MarkushClaim",
    "MarkushPrediction",
    "MarkushPredictor",
    "Predictor",
    "TargetNormalizationResult",
    "__version__",
    "check_cxsmiles_round_trip",
    "extract_components",
    "match_concrete_smiles",
    "normalize_claim",
    "normalize_markush_target",
    "parse_claim",
    "strict_equal",
]
