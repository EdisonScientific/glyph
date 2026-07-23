"""Deterministic stable-scaling SFT mixer (``build_stable_scaling_sft_mix`` home).

Backs the ``scripts/build_stable_scaling_sft_mix.py`` CLI. It combines
already-prepared SFT JSONL partitions in priority order, dedupes across
partitions, records per-row partition provenance, and writes a composition
manifest.

Image-reference portability: by default the mixer *absolutizes* image references
(and ``require_absolute`` gates on that). An opt-in ``relativize_to_root`` instead
rewrites every image reference to a ``$GLYPH_DATA_ROOT``-relative path on emit --
the mode the release orchestrator (:mod:`glyph.data.markush`) uses so no absolute
paths leak into the prepared mix.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVAL_SOURCES = {
    "ip5m",
    "ip5-markush",
    "ip5_markush",
    "m2s",
    "uspto-markush",
    "uspto_markush",
}

#: Row keys that may hold an image reference (scalar or ``{"path": ...}``).
IMAGE_KEYS = ("image", "image_path", "local_image_path", "page_path", "png_path", "rendered_path")


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    count: int
    path: Path


@dataclass(frozen=True)
class Candidate:
    key: str
    line_no: int
    row: dict[str, Any]


def parse_partition(value: str) -> PartitionSpec:
    """Parse ``NAME=COUNT:PATH``."""

    if "=" not in value or ":" not in value:
        raise argparse.ArgumentTypeError(f"Invalid partition {value!r}; expected NAME=COUNT:PATH")
    name, rest = value.split("=", 1)
    raw_count, raw_path = rest.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Partition name cannot be empty")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid count in {value!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("Partition count cannot be negative")
    path = Path(raw_path)
    if not path:
        raise argparse.ArgumentTypeError("Partition path cannot be empty")
    return PartitionSpec(name=name, count=count, path=path)


def parse_name_path(value: str) -> tuple[str, Path]:
    """Parse ``NAME=PATH``."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid name/path {value!r}; expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Name cannot be empty")
    return name, Path(raw_path)


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append((line_no, row))
    return rows


def row_key(row: dict[str, Any], line_no: int) -> str:
    source = str(row.get("source") or row.get("benchmark") or "unknown")
    source_id = str(row.get("source_id") or row.get("id") or row.get("sample_id") or line_no)
    mode = str(row.get("mode") or row.get("effective_mode") or "")
    return f"{source}\0{source_id}\0{mode}"


def is_eval_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "").strip().lower()
    benchmark = str(row.get("benchmark") or "").strip().lower()
    metadata = row.get("metadata")
    metadata_eval_only = isinstance(metadata, dict) and metadata.get("eval_only") is True
    return source in EVAL_SOURCES or benchmark in EVAL_SOURCES or metadata_eval_only


def _resolve_image_value(value: Any, image_root: Path | None) -> Any:
    if isinstance(value, str) and value.strip():
        image = Path(value)
        if image.is_absolute() or image_root is None:
            return value
        return str(image_root / image)
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        resolved = dict(value)
        resolved["path"] = _resolve_image_value(value["path"], image_root)
        return resolved
    return value


def resolve_images(row: dict[str, Any], image_root: Path | None) -> dict[str, Any]:
    resolved = copy.deepcopy(row)
    images = resolved.get("images")
    if isinstance(images, list):
        resolved["images"] = [_resolve_image_value(image, image_root) for image in images]
    for key in ("image", "image_path", "local_image_path", "page_path"):
        if key in resolved:
            resolved[key] = _resolve_image_value(resolved[key], image_root)
    return resolved


def relativize_image_ref(image: str, *, root: Path) -> tuple[str, bool]:
    """Return ``(ref, unresolved)`` with ``ref`` made ``root``-relative if possible.

    Resolution order (pure-path, no filesystem access):

    1. an already-relative ref is returned unchanged;
    2. an absolute ref under ``root`` is rewritten relative to ``root``;
    3. an absolute cluster ref that contains an ``images/`` segment is rewritten
       from that segment (``.../images/<source>/<split>/<file>`` ->
       ``images/<source>/<split>/<file>``), matching the portable §3 layout;
    4. anything else is returned unchanged with ``unresolved=True`` so the caller
       can flag it (a leaked absolute path).
    """

    if not image:
        return image, False
    path = Path(image)
    if not path.is_absolute():
        return image, False
    root = Path(root)
    if not root.is_absolute():
        root = Path.cwd() / root
    try:
        return path.relative_to(root).as_posix(), False
    except ValueError:
        pass
    parts = path.parts
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        return Path(*parts[idx:]).as_posix(), False
    return image, True


def _relativize_image_value(value: Any, root: Path, flags: list[bool]) -> Any:
    if isinstance(value, str) and value.strip():
        ref, unresolved = relativize_image_ref(value, root=root)
        flags.append(unresolved)
        return ref
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        rewritten = dict(value)
        rewritten["path"] = _relativize_image_value(value["path"], root, flags)
        return rewritten
    return value


def relativize_row_images(row: dict[str, Any], *, root: Path) -> tuple[dict[str, Any], bool]:
    """Rewrite every image ref in ``row`` to a ``root``-relative path.

    Returns ``(row, any_unresolved)`` where ``any_unresolved`` is True if any ref
    could not be made portable (still absolute).
    """

    rewritten = copy.deepcopy(row)
    flags: list[bool] = []
    images = rewritten.get("images")
    if isinstance(images, list):
        rewritten["images"] = [_relativize_image_value(image, root, flags) for image in images]
    for key in IMAGE_KEYS:
        if key in rewritten:
            rewritten[key] = _relativize_image_value(rewritten[key], root, flags)
    return rewritten, any(flags)


def require_absolute_images(row: dict[str, Any], *, context: str) -> None:
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{context}: row has no images list")
    for image in images:
        value = image.get("path") if isinstance(image, dict) else image
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context}: row has an empty image path")
        if not Path(value).is_absolute():
            raise ValueError(f"{context}: image path is not absolute: {value!r}")


def candidate_pool(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    used_keys: set[str],
    allow_eval_sources: bool,
) -> tuple[list[Candidate], dict[str, Any]]:
    candidates: list[Candidate] = []
    seen_in_partition: set[str] = set()
    stats = {
        "input_rows": len(rows),
        "eval_source_rows": 0,
        "duplicate_keys_within_partition": 0,
        "excluded_cross_partition_duplicates": 0,
    }
    for line_no, row in rows:
        key = row_key(row, line_no)
        if not allow_eval_sources and is_eval_row(row):
            stats["eval_source_rows"] += 1
            continue
        if key in seen_in_partition:
            stats["duplicate_keys_within_partition"] += 1
            continue
        seen_in_partition.add(key)
        if key in used_keys:
            stats["excluded_cross_partition_duplicates"] += 1
            continue
        candidates.append(Candidate(key=key, line_no=line_no, row=row))
    stats["eligible_rows"] = len(candidates)
    return candidates, stats


def select_candidates(
    candidates: list[Candidate],
    *,
    count: int,
    rng: random.Random,
    oversample: bool,
) -> tuple[list[Candidate], int]:
    if count <= 0:
        return [], 0
    if not candidates:
        return [], count
    if count <= len(candidates):
        return rng.sample(candidates, count), 0
    if not oversample:
        return list(candidates), count - len(candidates)
    selected = list(candidates)
    selected.extend(rng.choice(candidates) for _ in range(count - len(candidates)))
    return selected, 0


def add_partition_metadata(
    row: dict[str, Any],
    *,
    partition: PartitionSpec,
    line_no: int,
    repeat_index: int,
    partition_source_jsonl: str | None = None,
) -> dict[str, Any]:
    enriched = copy.deepcopy(row)
    enriched["stable_scaling_mix"] = {
        "partition": partition.name,
        "partition_source_jsonl": (
            partition_source_jsonl if partition_source_jsonl is not None else str(partition.path)
        ),
        "source_line": line_no,
        "repeat_index": repeat_index,
    }
    return enriched


def _portable_partition_source(path: Path, *, root: Path) -> str:
    """Return a ``root``-relative partition-source ref, else just its filename.

    ``stable_scaling_mix.partition_source_jsonl`` is rewritten ``root``-relative
    (or reduced to its filename) so no absolute paths are recorded.
    """

    ref, unresolved = relativize_image_ref(str(path), root=root)
    if unresolved:
        return Path(path).name
    return ref


def counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stable Scaling SFT Mix",
        "",
        f"- Created: `{manifest['created_at']}`",
        f"- Output: `{manifest['output_jsonl']}`",
        f"- Rows: `{manifest['written_rows']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Oversample: `{manifest['oversample']}`",
        f"- Shuffle: `{manifest['shuffle']}`",
        f"- Require absolute images: `{manifest['require_absolute_images']}`",
        "",
        "## Partitions",
        "",
        "| partition | source | requested | selected | shortage | repeated | cross-partition duplicates |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for partition in manifest["partitions"]:
        lines.append(
            "| {name} | `{path}` | {requested} | {selected} | {shortage} | {repeated} | {cross} |".format(
                name=partition["name"],
                path=partition["path"],
                requested=partition["requested_count"],
                selected=partition["selected_count"],
                shortage=partition["shortage"],
                repeated=partition["repeated_rows"],
                cross=partition["excluded_cross_partition_duplicates"],
            )
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            "```json",
            json.dumps(
                {
                    "partition_counts": manifest["partition_counts"],
                    "source_counts": manifest["source_counts"],
                    "mode_counts": manifest["mode_counts"],
                    "target_format_counts": manifest["target_format_counts"],
                    "target_source_counts": manifest["target_source_counts"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## Validation",
            "",
            "Validate this output and the public eval dir with "
            "`glyph.data.markush_check.check_artifacts` before training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_mix(
    *,
    partitions: list[PartitionSpec],
    output_jsonl: Path,
    manifest_json: Path,
    report_md: Path | None,
    image_roots: dict[str, Path],
    seed: int,
    oversample: bool,
    shuffle: bool,
    require_absolute: bool,
    allow_eval_sources: bool,
    argv: list[str],
    relativize_to_root: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic partition mix and write JSONL + manifest.

    ``relativize_to_root`` (release mode): when set, every emitted image ref is
    rewritten ``root``-relative instead of the default absolutize +
    ``require_absolute`` gate.
    """

    rng = random.Random(seed)
    used_keys: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    partition_summaries: list[dict[str, Any]] = []
    unresolved_image_rows = 0

    for partition in partitions:
        rows = load_jsonl(partition.path)
        pool, pool_stats = candidate_pool(
            rows,
            used_keys=used_keys,
            allow_eval_sources=allow_eval_sources,
        )
        selected, shortage = select_candidates(
            pool,
            count=partition.count,
            rng=rng,
            oversample=oversample,
        )
        repeats_by_key: dict[str, int] = defaultdict(int)
        image_root = image_roots.get(partition.name)
        partition_source = (
            _portable_partition_source(partition.path, root=relativize_to_root)
            if relativize_to_root is not None
            else None
        )
        selected_rows: list[dict[str, Any]] = []
        for candidate in selected:
            repeat_index = repeats_by_key[candidate.key]
            repeats_by_key[candidate.key] += 1
            row = resolve_images(candidate.row, image_root)
            if relativize_to_root is not None:
                row, unresolved = relativize_row_images(row, root=relativize_to_root)
                if unresolved:
                    unresolved_image_rows += 1
            elif require_absolute:
                require_absolute_images(
                    row,
                    context=f"{partition.path}:{candidate.line_no}",
                )
            row = add_partition_metadata(
                row,
                partition=partition,
                line_no=candidate.line_no,
                repeat_index=repeat_index,
                partition_source_jsonl=partition_source,
            )
            selected_rows.append(row)
        used_keys.update(candidate.key for candidate in pool)
        output_rows.extend(selected_rows)
        partition_summaries.append(
            {
                "name": partition.name,
                "path": str(partition.path),
                "requested_count": partition.count,
                "selected_count": len(selected_rows),
                "shortage": shortage,
                "repeated_rows": sum(max(0, count - 1) for count in repeats_by_key.values()),
                "image_root": str(image_root) if image_root else None,
                **pool_stats,
                "source_counts": counter_by(selected_rows, "source"),
                "mode_counts": counter_by(selected_rows, "mode"),
                "target_format_counts": counter_by(selected_rows, "target_format"),
            }
        )

    if shuffle:
        rng.shuffle(output_rows)

    write_jsonl(output_jsonl, output_rows)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": argv,
        "output_jsonl": str(output_jsonl),
        "seed": seed,
        "oversample": oversample,
        "shuffle": shuffle,
        "require_absolute_images": require_absolute,
        "relativize_to_root": str(relativize_to_root) if relativize_to_root else None,
        "unresolved_image_rows": unresolved_image_rows,
        "allow_eval_sources": allow_eval_sources,
        "written_rows": len(output_rows),
        "partitions": partition_summaries,
        "partition_counts": counter_by(
            [row["stable_scaling_mix"] for row in output_rows],
            "partition",
        ),
        "source_counts": counter_by(output_rows, "source"),
        "mode_counts": counter_by(output_rows, "mode"),
        "target_format_counts": counter_by(output_rows, "target_format"),
        "target_source_counts": counter_by(output_rows, "target_source"),
        "target_canonicalization_status_counts": counter_by(
            output_rows,
            "target_canonicalization_status",
        ),
    }
    write_manifest(manifest_json, manifest)
    if report_md:
        write_report(report_md, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic stable-scaling SFT mix.")
    parser.add_argument(
        "--partition",
        action="append",
        type=parse_partition,
        required=True,
        help="Prepared partition as NAME=COUNT:PATH. Repeat in priority order.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument(
        "--image-root",
        action="append",
        type=parse_name_path,
        default=[],
        help="Resolve relative image paths for one partition as NAME=PATH.",
    )
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--require-absolute-images", action="store_true")
    parser.add_argument(
        "--relativize-to-root",
        type=Path,
        default=None,
        help="Rewrite every emitted image ref relative to this root (release §3 mode).",
    )
    parser.add_argument("--allow-eval-sources", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    image_roots = dict(args.image_root)
    unknown_roots = set(image_roots) - {partition.name for partition in args.partition}
    if unknown_roots:
        names = ", ".join(sorted(unknown_roots))
        raise SystemExit(f"--image-root specified for unknown partition(s): {names}")
    manifest = build_mix(
        partitions=args.partition,
        output_jsonl=args.output_jsonl,
        manifest_json=args.manifest_json,
        report_md=args.report_md,
        image_roots=image_roots,
        seed=args.seed,
        oversample=not args.no_oversample,
        shuffle=not args.no_shuffle,
        require_absolute=args.require_absolute_images,
        allow_eval_sources=args.allow_eval_sources,
        argv=raw_argv,
        relativize_to_root=args.relativize_to_root,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
