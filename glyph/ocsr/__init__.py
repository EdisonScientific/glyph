"""Public API for the dedicated optical chemical structure recognizer."""

__version__ = "0.1.0"

__all__ = ["OCSRPredictor", "Predictor", "__version__"]


def __getattr__(name: str):
    if name in {"OCSRPredictor", "Predictor"}:
        from .predict import OCSRPredictor

        return OCSRPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
