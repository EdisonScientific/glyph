"""Prepare Markush eval-benchmark JSONLs (``benchmarks/markush/{ip5m,m2s,uspto_markush}``).

Reads the unified JSONL an importer wrote for one DS4SD benchmark and emits an
``eval_<name>.jsonl`` with the shared prompt/target contract used by
``glyph markush eval``.

Image references in the emitted rows are normalized to ``$GLYPH_DATA_ROOT``-
relative paths: a ref already under ``data_root`` is rewritten relative to it; a
bare ``images/...`` ref is kept as-is; anything else is left untouched and flagged
in the returned stats (``absolute_image_rows``). No absolute paths are written.

Because the refs are ``$GLYPH_DATA_ROOT``-relative (``images/<benchmark>/<split>/…``),
the eval loader must be pointed at ``$GLYPH_DATA_ROOT`` as its image root. See
``glyph/data/README.md`` for the full loader invocation:

    python -m glyph.markush.eval.fast_eval \\
        --eval-data "$GLYPH_DATA_ROOT/benchmarks/markush/ip5m/eval_ip5m.jsonl" \\
        --image-root "$GLYPH_DATA_ROOT" ...
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from glyph.markush.data.example_modes import PATENT_IMAGE_MARKUSH, STRUCTURE_ONLY_MARKUSH
from glyph.markush.data.prepare_sft import (
    prompt_with_target_instruction,
    target_spec_for_record,
)
from glyph.markush.eval import common
from glyph.markush.eval.benchmarks import get_benchmark_spec
from glyph.paths import data_root as resolve_data_root


@dataclass
class BenchmarkPrepStats:
    """Summary of one prepared benchmark."""

    benchmark: str
    source_path: str
    output_path: str
    input_rows: int = 0
    written_rows: int = 0
    skipped_rows: int = 0
    absolute_image_rows: int = 0
    sha256: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "input_rows": self.input_rows,
            "written_rows": self.written_rows,
            "skipped_rows": self.skipped_rows,
            "absolute_image_rows": self.absolute_image_rows,
            "sha256": self.sha256,
            "samples": self.samples,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_image_ref(row: dict[str, Any]) -> str:
    """Return the source image reference for one unified row."""

    image = common.sft_image_path(row)
    if image:
        return image
    for key in ("image", "image_path", "path", "png_path", "rendered_path", "file_name"):
        value = row.get(key)
        if isinstance(value, dict):
            value = value.get("path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def relativize_image_ref(image: str, *, root: Path) -> tuple[str, bool]:
    """Return ``(ref, was_absolute)`` with ``ref`` made ``root``-relative if possible.

    A path already under ``root`` is rewritten relative to it. A relative ref is
    returned unchanged. An absolute ref outside ``root`` cannot be made portable
    and is returned unchanged with ``was_absolute=True`` so the caller can flag it.
    """

    if not image:
        return image, False
    path = Path(image)
    if not path.is_absolute():
        return image, False
    root = root.expanduser().resolve()
    try:
        return path.resolve().relative_to(root).as_posix(), False
    except ValueError:
        return image, True


def prepared_eval_row(
    row: dict[str, Any],
    *,
    benchmark: str,
    data_root: Path,
    canonicalize_markush_targets: bool,
    require_canonical_markush_targets: bool,
) -> tuple[dict[str, Any] | None, bool]:
    """Return ``(prepared_row_or_None, image_was_absolute)`` for one unified row."""

    spec = get_benchmark_spec(benchmark)
    target = target_spec_for_record(
        row,
        canonicalize_markush=canonicalize_markush_targets,
        require_canonical=require_canonical_markush_targets,
    )
    if target is None:
        return None, False
    cxsmiles, stable = common.extract_cxsmiles_and_stable(target.content, require_wrapper=False)
    if not cxsmiles:
        return None, False

    image = eval_image_ref(row)
    if not image:
        return None, False
    image, was_absolute = relativize_image_ref(image, root=data_root)

    prepared_prompt = prompt_with_target_instruction(spec.prompt, target)
    mode = PATENT_IMAGE_MARKUSH if spec.image_has_text else STRUCTURE_ONLY_MARKUSH
    prepared = dict(row)
    prepared["task"] = "markush"
    prepared["mode"] = mode
    prepared["source"] = prepared.get("source") or spec.name
    prepared["split"] = "eval"
    prepared["benchmark"] = spec.name
    prepared["image"] = image
    prepared["images"] = [image]
    prepared["prompt"] = prepared_prompt
    prepared["messages"] = [
        {"role": "user", "content": f"<image>\n{prepared_prompt}"},
        {"role": "assistant", "content": target.content},
    ]
    prepared["annotation"] = target.content
    prepared["ground_truth"] = target.content
    prepared["cxsmiles_opt"] = cxsmiles
    prepared["target_cxsmiles_format"] = target.format_label
    prepared["target_source"] = target.source
    prepared["target_format"] = target.format_label
    prepared["target_canonicalization_status"] = target.canonicalization_status
    prepared["target_canonicalization_error"] = target.canonicalization_error
    prepared["stable"] = stable
    return prepared, was_absolute


def prepare_benchmark(
    *,
    source_jsonl: Path,
    output_jsonl: Path,
    benchmark: str,
    data_root: Path | None = None,
    canonicalize_markush_targets: bool = False,
    require_canonical_markush_targets: bool = False,
    sample_limit: int = 5,
) -> BenchmarkPrepStats:
    """Prepare one benchmark's eval JSONL from its imported unified JSONL."""

    source_jsonl = source_jsonl.expanduser()
    output_jsonl = output_jsonl.expanduser()
    root = resolve_data_root(data_root)
    if not source_jsonl.exists():
        raise FileNotFoundError(source_jsonl)

    rows = load_jsonl(source_jsonl)
    prepared_rows: list[dict[str, Any]] = []
    stats = BenchmarkPrepStats(
        benchmark=benchmark,
        source_path=str(source_jsonl),
        output_path=str(output_jsonl),
        input_rows=len(rows),
    )
    for row in rows:
        prepared, was_absolute = prepared_eval_row(
            row,
            benchmark=benchmark,
            data_root=root,
            canonicalize_markush_targets=canonicalize_markush_targets,
            require_canonical_markush_targets=require_canonical_markush_targets,
        )
        if prepared is None:
            stats.skipped_rows += 1
            continue
        if was_absolute:
            stats.absolute_image_rows += 1
        prepared_rows.append(prepared)
        if len(stats.samples) < sample_limit:
            stats.samples.append(
                {
                    "id": prepared.get("id"),
                    "benchmark": prepared.get("benchmark"),
                    "image": prepared.get("image"),
                    "prompt": prepared.get("prompt"),
                    "target": prepared.get("annotation"),
                }
            )

    write_jsonl(prepared_rows, output_jsonl)
    stats.written_rows = len(prepared_rows)
    stats.sha256 = sha256_file(output_jsonl)
    return stats
