"""Benchmark prompt presets and default local paths for Markush evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from glyph.data.layout import markush_benchmark_dir
from glyph.markush.data.example_modes import (
    PATENT_IMAGE_MARKUSH_PROMPT,
    STRUCTURE_ONLY_MARKUSH_PROMPT,
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Prompt and path metadata for one Markush benchmark."""

    name: str
    prompt: str
    image_has_text: bool
    description: str

    @property
    def subdir(self) -> str:
        """Filesystem-safe (underscore) benchmark name used on disk."""

        return self.name.replace("-", "_")


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "ip5m": BenchmarkSpec(
        name="ip5m",
        prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
        image_has_text=False,
        description="IP5-M structure-only benchmark",
    ),
    "m2s": BenchmarkSpec(
        name="m2s",
        prompt=PATENT_IMAGE_MARKUSH_PROMPT,
        image_has_text=True,
        description="M2S structure-plus-text benchmark",
    ),
    "uspto-markush": BenchmarkSpec(
        name="uspto-markush",
        prompt=STRUCTURE_ONLY_MARKUSH_PROMPT,
        image_has_text=False,
        description="USPTO-Markush structure-only benchmark",
    ),
}

_ALIASES = {
    "ip5": "ip5m",
    "ip5-m": "ip5m",
    "ip5_markush": "ip5m",
    "ip5-markush": "ip5m",
    "uspto": "uspto-markush",
    "uspto_m": "uspto-markush",
    "uspto-m": "uspto-markush",
    "uspto_markush": "uspto-markush",
}


def benchmark_choices() -> tuple[str, ...]:
    """Return canonical benchmark names for argparse choices."""

    return tuple(BENCHMARKS)


def normalize_benchmark_name(value: str) -> str:
    """Normalize a user-facing benchmark name or alias."""

    key = value.strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    if key not in BENCHMARKS:
        valid = ", ".join(benchmark_choices())
        raise ValueError(f"Unknown benchmark {value!r}; expected one of: {valid}")
    return key


def get_benchmark_spec(value: str) -> BenchmarkSpec:
    """Return the canonical spec for a benchmark name or alias."""

    return BENCHMARKS[normalize_benchmark_name(value)]


def benchmark_eval_jsonl(benchmark: str, *, root: str | Path | None = None) -> Path:
    """Return the canonical eval JSONL path for a benchmark.

    Resolves to the ``glyph.data.layout`` layout that ``glyph data prepare
    --markush ... benchmarks`` writes and that
    :func:`glyph.markush.eval.benchmark_data.materialize_benchmark` unpacks:
    ``<root>/benchmarks/markush/<subdir>/eval_<subdir>.jsonl`` (``<subdir>`` is
    the filesystem-safe underscore name). ``root`` overrides the data root;
    when omitted the configured ``$GLYPH_DATA_ROOT`` is used. The historical
    flat ``data/eval_<bench>.jsonl`` paths never shipped in a clean checkout.
    """

    subdir = get_benchmark_spec(benchmark).subdir
    return markush_benchmark_dir(subdir, root=root) / f"eval_{subdir}.jsonl"


def resolve_benchmark_path(
    benchmark: str,
    *,
    explicit_path: str | Path | None = None,
    root: str | Path | None = None,
) -> Path:
    """Resolve the eval JSONL path for a benchmark.

    ``explicit_path`` wins.  Otherwise the benchmark's canonical
    ``glyph.data.layout`` path is returned (see :func:`benchmark_eval_jsonl`);
    ``root`` overrides the data root, defaulting to ``$GLYPH_DATA_ROOT``.
    """

    if explicit_path is not None:
        return Path(explicit_path).expanduser()
    return benchmark_eval_jsonl(benchmark, root=root)
