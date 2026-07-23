"""Render OCSR-paper sources to unified OCSR JSONL.

Backs the ``scripts/materialize_ocsr_paper_sources.py`` CLI (a thin wrapper
over this module). Combines the OCSR patent source (MolScribe/USPTO,
image-backed) with curated SMILES-only sources and renders the SMILES-only
rows to PNG (RDKit, or the vendored Indigo renderer) so the Markush prep path
can convert them to no-extension Markush XML.

This renderer feeds only the appendix synthetic-augmentation arm; the headline
mix uses public MG1 rows, not local renders (see ``glyph/data/README.md``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from glyph.markush.data.ocsr import canonicalize_ocsr_smiles_checked

T = TypeVar("T")

_PSEUDO_MARKUSH_SMILES_RE = re.compile(r"\*|\[(?:Ar|Y)\]")


@dataclass(frozen=True)
class RenderJob:
    source_name: str
    source_path: str
    source_row_index: int
    sample_index: int
    smiles: str
    output_path: str
    row: dict[str, Any]


@dataclass(frozen=True)
class RenderResult:
    ok: bool
    source_name: str
    source_path: str
    source_row_index: int
    sample_index: int
    smiles: str
    canonical_smiles: str
    output_path: str
    render_backend: str = "rdkit"
    error: str = ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_count_path(value: str) -> tuple[str, int, Path]:
    """Parse ``NAME=COUNT:PATH``."""

    if "=" not in value or ":" not in value:
        raise argparse.ArgumentTypeError(f"Expected NAME=COUNT:PATH, got {value!r}")
    name, rest = value.split("=", 1)
    raw_count, raw_path = rest.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Source name cannot be empty")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid count in {value!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("Count cannot be negative")
    return name, count, Path(raw_path)


def sample_items(items: list[T], count: int, rng: random.Random, *, label: str) -> list[T]:
    if count == 0:
        return []
    if not items:
        raise ValueError(f"No rows available for {label}")
    if count <= len(items):
        return rng.sample(items, count)
    return [rng.choice(items) for _ in range(count)]


def pseudo_markush_smiles_reason(smiles: str) -> str:
    """Return a reason when an OCSR SMILES contains Markush-like placeholders."""

    if "*" in smiles:
        return "pseudo_markush_dummy_atom"
    if "[Ar]" in smiles or "[Y]" in smiles:
        return "pseudo_markush_placeholder_atom"
    if _PSEUDO_MARKUSH_SMILES_RE.search(smiles):
        return "pseudo_markush_placeholder"
    return ""


def row_image(row: dict[str, Any]) -> str:
    for key in ("image", "image_path", "path", "png_path", "rendered_path", "file_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    images = row.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    return ""


def normalize_patent_row(
    row: dict[str, Any], *, source_name: str, sample_index: int
) -> dict[str, Any]:
    image = row_image(row)
    smiles = str(row.get("smiles") or "").strip()
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "ocsr_paper_source": source_name,
            "ocsr_source_family": "real_uspto_tif",
            "materializer": Path(__file__).name,
            "materializer_sample_index": sample_index,
        }
    )
    return {
        "id": f"{source_name}_{sample_index:07d}_{row.get('id') or sample_index}",
        "source": source_name,
        "split": "train",
        "image": image,
        "image_has_text": False,
        "task": "ocsr",
        "cxsmiles_opt": None,
        "smiles": smiles,
        "annotation": None,
        "cxsmiles_dataset": None,
        "cxsmiles_standard": None,
        "description": None,
        "cells": None,
        "patent_id": row.get("patent_id") or row.get("id"),
        "patent_office": row.get("patent_office") or "USPTO",
        "metadata": metadata,
    }


def valid_patent_rows(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
    count: int,
    rng: random.Random,
    exclude_pseudo_markush_smiles: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample valid image-backed patent rows and canonicalize their SMILES."""

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    for source_index in indices:
        row = rows[source_index]
        image = row_image(row)
        raw_smiles = str(row.get("smiles") or "").strip()
        reason = ""
        canonical_smiles = None
        if not image:
            reason = "missing_image"
        elif not raw_smiles:
            reason = "missing_smiles"
        elif exclude_pseudo_markush_smiles and (
            pseudo_reason := pseudo_markush_smiles_reason(raw_smiles)
        ):
            reason = pseudo_reason
        else:
            result = canonicalize_ocsr_smiles_checked(raw_smiles)
            canonical_smiles = result.value
            if canonical_smiles is None:
                reason = result.error or "invalid_smiles"
        if reason:
            skipped.append(
                {
                    "source": source_name,
                    "source_index": source_index,
                    "id": row.get("id"),
                    "reason": reason,
                    "smiles": raw_smiles,
                }
            )
            continue
        normalized = dict(row)
        normalized["smiles"] = canonical_smiles
        selected.append(normalized)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(
            f"Only found {len(selected)} valid rows for {source_name}; requested {count}"
        )
    return selected, skipped


def smiles_from_csv_row(row: dict[str, str]) -> str:
    for key in ("SMILES", "smiles", "raw_SMILES", "canonical_smiles", "target"):
        value = row.get(key)
        if value and value.strip():
            return value.strip()
    return ""


def load_csv_smiles(path: Path) -> list[tuple[int, dict[str, str], str]]:
    rows: list[tuple[int, dict[str, str], str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            smiles = smiles_from_csv_row(row)
            if smiles:
                rows.append((index, dict(row), smiles))
    return rows


def stable_render_name(source_name: str, sample_index: int, smiles: str) -> str:
    digest = hashlib.sha1(f"{source_name}\0{sample_index}\0{smiles}".encode()).hexdigest()[:16]
    return f"{source_name}_{sample_index:07d}_{digest}.png"


def _render_smiles_with_rdkit(smiles: str, output: Path, *, width: int, height: int) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse canonical SMILES")
    rdDepictor.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.padding = 0.06
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    output.write_bytes(drawer.GetDrawingText())


def _ensure_indigo_pythonpath(indigo_pythonpath: str | None) -> None:
    if not indigo_pythonpath:
        return
    path = str(Path(indigo_pythonpath).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _render_smiles_with_indigo(
    smiles: str,
    output: Path,
    *,
    width: int,
    height: int,
    indigo_pythonpath: str | None,
) -> None:
    if width != height:
        raise ValueError("Indigo renderer backend currently requires square render dimensions")
    _ensure_indigo_pythonpath(indigo_pythonpath)
    try:
        from glyph.ocsr.indigo_renderer import IndigoRendererBoundary
    except Exception as exc:  # pragma: no cover - depends on optional Indigo install
        raise RuntimeError(
            "Could not import glyph.ocsr.indigo_renderer. Install the OCSR extra "
            "(`pip install glyph[ocsr]`) on Linux x86-64 so the vendored Indigo "
            "renderer is available, or pass --indigo-pythonpath pointing at a "
            "checkout that contains the glyph package."
        ) from exc

    renderer = IndigoRendererBoundary(image_size=width)
    output.write_bytes(renderer.render_smiles_to_png(smiles))


def render_smiles_job(
    job: RenderJob,
    *,
    width: int,
    height: int,
    force: bool,
    backend: str,
    indigo_pythonpath: str | None = None,
) -> RenderResult:
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment failure
        return RenderResult(
            ok=False,
            source_name=job.source_name,
            source_path=job.source_path,
            source_row_index=job.source_row_index,
            sample_index=job.sample_index,
            smiles=job.smiles,
            canonical_smiles="",
            output_path=job.output_path,
            render_backend=backend,
            error=f"rdkit_import_failed: {exc}",
        )

    try:
        canonical = canonicalize_ocsr_smiles_checked(job.smiles)
        if canonical.value is None:
            raise ValueError(canonical.error or "RDKit could not canonicalize SMILES")
        output = Path(job.output_path)
        if force or not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            if backend == "rdkit":
                _render_smiles_with_rdkit(canonical.value, output, width=width, height=height)
            elif backend == "indigo":
                _render_smiles_with_indigo(
                    canonical.value,
                    output,
                    width=width,
                    height=height,
                    indigo_pythonpath=indigo_pythonpath,
                )
            else:
                raise ValueError(f"Unsupported render backend: {backend}")
        return RenderResult(
            ok=True,
            source_name=job.source_name,
            source_path=job.source_path,
            source_row_index=job.source_row_index,
            sample_index=job.sample_index,
            smiles=job.smiles,
            canonical_smiles=canonical.value,
            output_path=str(output),
            render_backend=backend,
        )
    except Exception as exc:
        return RenderResult(
            ok=False,
            source_name=job.source_name,
            source_path=job.source_path,
            source_row_index=job.source_row_index,
            sample_index=job.sample_index,
            smiles=job.smiles,
            canonical_smiles="",
            output_path=job.output_path,
            render_backend=backend,
            error=str(exc),
        )


def render_smiles_jobs(
    jobs: list[RenderJob],
    *,
    width: int,
    height: int,
    workers: int,
    force: bool,
    backend: str,
    indigo_pythonpath: str | None,
) -> list[RenderResult]:
    if workers <= 1:
        return [
            render_smiles_job(
                job,
                width=width,
                height=height,
                force=force,
                backend=backend,
                indigo_pythonpath=indigo_pythonpath,
            )
            for job in jobs
        ]
    results: list[RenderResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                render_smiles_job,
                job,
                width=width,
                height=height,
                force=force,
                backend=backend,
                indigo_pythonpath=indigo_pythonpath,
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: (item.source_name, item.sample_index))


def render_result_to_row(result: RenderResult, row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row)
    metadata.update(
        {
            "ocsr_paper_source": result.source_name,
            "ocsr_source_family": "rendered_curated_smiles",
            "source_path": result.source_path,
            "source_row_index": result.source_row_index,
            "render_backend": result.render_backend,
            "materializer": Path(__file__).name,
            "materializer_sample_index": result.sample_index,
        }
    )
    return {
        "id": f"{result.source_name}_{result.sample_index:07d}",
        "source": result.source_name,
        "split": "train",
        "image": result.output_path,
        "image_has_text": False,
        "task": "ocsr",
        "cxsmiles_opt": None,
        "smiles": result.canonical_smiles,
        "annotation": None,
        "cxsmiles_dataset": None,
        "cxsmiles_standard": None,
        "description": None,
        "cells": None,
        "patent_id": row.get("pubchem_cid") or row.get("source") or result.source_name,
        "patent_office": "synthetic",
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--render-backend", choices=("rdkit", "indigo"), default="rdkit")
    parser.add_argument(
        "--indigo-pythonpath",
        type=Path,
        help=(
            "Optional path to prepend before importing glyph.ocsr.indigo_renderer, "
            "for example a checkout root that contains the glyph package."
        ),
    )
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument(
        "--exclude-pseudo-markush-smiles",
        action="store_true",
        help=(
            "Exclude otherwise RDKit-parseable OCSR rows whose SMILES contain "
            "Markush-like placeholders such as dummy atoms, [Ar], or [Y]. Use "
            "this when the downstream target contract treats OCSR as ordinary "
            "no-extension Markush."
        ),
    )
    parser.add_argument(
        "--patent-source",
        type=parse_count_path,
        action="append",
        default=[],
        metavar="NAME=COUNT:JSONL",
        help="Image-backed unified OCSR JSONL source, usually USPTO TIF rows.",
    )
    parser.add_argument(
        "--render-source",
        type=parse_count_path,
        action="append",
        default=[],
        metavar="NAME=COUNT:CSV",
        help="SMILES CSV source to render with the selected backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or output_dir / "unified_ocsr_paper_sources.jsonl"
    manifest_path = args.manifest or output_dir / "manifest.json"
    render_dir = output_dir / "renders"
    rng = random.Random(args.seed)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    source_inputs: dict[str, dict[str, Any]] = {}

    for source_name, count, path in args.patent_source:
        if not path.exists():
            raise FileNotFoundError(path)
        source_rows = load_jsonl(path)
        selected, patent_skipped = valid_patent_rows(
            source_rows,
            source_name=source_name,
            count=count,
            rng=rng,
            exclude_pseudo_markush_smiles=args.exclude_pseudo_markush_smiles,
        )
        skipped.extend(patent_skipped)
        for index, row in enumerate(selected):
            normalized = normalize_patent_row(row, source_name=source_name, sample_index=index)
            rows.append(normalized)
            source_counts[source_name] += 1
        source_inputs[source_name] = {
            "kind": "patent_jsonl",
            "path": str(path),
            "requested": count,
            "available": len(source_rows),
            "invalid_or_missing_before_sample_count": len(patent_skipped),
            "invalid_or_missing_before_sample_preview": patent_skipped[:20],
            "written": source_counts[source_name],
        }

    render_jobs: list[RenderJob] = []
    render_job_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source_name, count, path in args.render_source:
        if not path.exists():
            raise FileNotFoundError(path)
        raw_source_rows = load_csv_smiles(path)
        source_rows: list[tuple[int, dict[str, str], str]] = []
        render_filtered: list[dict[str, Any]] = []
        for source_row_index, csv_row, smiles in raw_source_rows:
            if args.exclude_pseudo_markush_smiles and (
                pseudo_reason := pseudo_markush_smiles_reason(smiles)
            ):
                render_filtered.append(
                    {
                        "source": source_name,
                        "source_index": source_row_index,
                        "id": csv_row.get("id") or csv_row.get("pubchem_cid"),
                        "reason": pseudo_reason,
                        "smiles": smiles,
                    }
                )
                continue
            source_rows.append((source_row_index, csv_row, smiles))
        skipped.extend(render_filtered)
        selected = sample_items(source_rows, count, rng, label=source_name)
        for sample_index, (source_row_index, csv_row, smiles) in enumerate(selected):
            output_path = (
                render_dir / source_name / stable_render_name(source_name, sample_index, smiles)
            )
            job = RenderJob(
                source_name=source_name,
                source_path=str(path),
                source_row_index=source_row_index,
                sample_index=sample_index,
                smiles=smiles,
                output_path=str(output_path),
                row=csv_row,
            )
            render_jobs.append(job)
            render_job_rows[(source_name, sample_index)] = csv_row
        source_inputs[source_name] = {
            "kind": "rendered_smiles_csv",
            "path": str(path),
            "requested": count,
            "available": len(raw_source_rows),
            "eligible": len(source_rows),
            "excluded_pseudo_markush_count": len(render_filtered),
            "excluded_pseudo_markush_preview": render_filtered[:20],
        }

    render_results = render_smiles_jobs(
        render_jobs,
        width=args.render_width,
        height=args.render_height,
        workers=args.workers,
        force=args.force_render,
        backend=args.render_backend,
        indigo_pythonpath=str(args.indigo_pythonpath) if args.indigo_pythonpath else None,
    )
    for result in render_results:
        if not result.ok:
            skipped.append(asdict(result))
            continue
        source_row = render_job_rows[(result.source_name, result.sample_index)]
        rows.append(render_result_to_row(result, source_row))
        source_counts[result.source_name] += 1

    rng.shuffle(rows)
    write_jsonl(rows, output_jsonl)
    manifest = {
        "artifact_root": str(output_dir),
        "output_jsonl": str(output_jsonl),
        "render_dir": str(render_dir),
        "seed": args.seed,
        "workers": args.workers,
        "render_size": [args.render_width, args.render_height],
        "render_backend": args.render_backend,
        "indigo_pythonpath": str(args.indigo_pythonpath) if args.indigo_pythonpath else None,
        "total_rows": len(rows),
        "source_counts": dict(source_counts),
        "source_inputs": source_inputs,
        "skipped_rows": len(skipped),
        "skipped_preview": skipped[:100],
        "target_contract": "unified OCSR rows; downstream builder converts to no-extension Markush XML",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
