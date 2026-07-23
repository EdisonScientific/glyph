"""Data generation and preparation utilities for Markush training."""

__all__ = [
    "DecoratedMarkush",
    "DecorationConfig",
    "ScaffoldFilterConfig",
    "ValidationResult",
    "build_scaffold_pool",
    "decorate_scaffold",
    "generate_decorated_samples",
    "load_scaffold_pool",
    "opt_to_standard_cxsmiles",
    "prepare_training_jsonl_from_manifest",
    "standard_to_opt_cxsmiles",
    "validate_cxsmiles",
]


def __getattr__(name: str):
    if name in __all__:
        from glyph.markush.data import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'glyph.markush.data' has no attribute {name!r}")
