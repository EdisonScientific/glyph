#!/usr/bin/env python3
"""Print greedy generations for a small mixed Markush/OCSR checkpoint sample.

This is intentionally lightweight: it reads either SFT JSONL rows
(``messages``/``images``) or unified/eval-style rows, selects a few examples per
mode, runs checkpoint generation, prints target/prediction pairs, and writes the
same details to JSON for later inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from glyph.markush.eval.checkpoint import EvalRecord, generate_one, load_model_and_processor

MARKUSH_MODES = {
    "patent_image_markush",
    "structure_only_markush",
    "text_conditioned_markush",
    "patent_image",
    "structure_only",
}
SMILES_MODES = {"molecule_ocsr", "ocsr", "smiles"}


@dataclass(slots=True)
class Sample:
    id: str
    mode: str
    image: str
    prompt: str
    target: str


@dataclass(slots=True)
class Result:
    id: str
    mode: str
    image: str
    prompt: str
    target: str
    prediction: str
    exact_match: bool
    valid_smiles: bool | None = None
    canonical_match: bool | None = None
    target_canonical: str | None = None
    prediction_canonical: str | None = None
    has_markush: bool | None = None
    has_cxsmi: bool | None = None
    closed_cxsmi: bool | None = None
    closed_markush: bool | None = None
    has_coordinate_block: bool | None = None


def _iter_jsonl(path: Path, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        yielded = 0
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            yielded += 1
            if limit is not None and yielded >= limit:
                break


def _strip_image_token(text: str) -> str:
    return text.replace("<image>\n", "").replace("<image>", "").strip()


def _target_from_cxsmiles(cxsmiles_opt: str, annotation: str = "") -> str:
    if annotation.strip().lower().startswith("<markush"):
        return annotation.strip()
    return f"<markush><cxsmi>{cxsmiles_opt}</cxsmi></markush>"


def _resolve_sample_image(image: str, image_root: Path | None) -> str:
    candidate = Path(image).expanduser()
    if candidate.is_absolute() or image_root is None:
        return str(candidate)

    root = image_root.expanduser()
    rooted = root / candidate
    data_rooted = root / "data" / candidate
    if rooted.exists():
        return str(rooted)
    if candidate.parts and candidate.parts[0] != "data" and data_rooted.exists():
        return str(data_rooted)
    if root.name == "data" or (candidate.parts and candidate.parts[0] == "data"):
        return str(rooted)
    if (root / "data").is_dir():
        return str(data_rooted)
    return str(rooted)


def _sample_from_row(row: dict[str, Any], image_root: Path | None = None) -> Sample | None:
    if row.get("messages") and row.get("images"):
        messages = row["messages"]
        mode = str(row.get("mode") or row.get("effective_mode") or "")
        image = _resolve_sample_image(str(row["images"][0]), image_root)
        return Sample(
            id=str(row.get("id", "")),
            mode=mode,
            image=image,
            prompt=_strip_image_token(str(messages[0]["content"])),
            target=str(messages[1]["content"]),
        )

    image = row.get("image") or row.get("image_path") or row.get("path") or row.get("png_path")
    if not image:
        images = row.get("images")
        if isinstance(images, list) and images:
            image = images[0]
    if not image:
        return None

    raw_mode = str(row.get("mode") or row.get("effective_mode") or row.get("variant") or "")
    task = str(row.get("task") or "").lower()
    if task == "ocsr" or row.get("smiles"):
        mode = "molecule_ocsr"
        target = str(row.get("smiles") or row.get("target") or "")
    else:
        mode = raw_mode or "markush"
        target = str(
            row.get("annotation")
            or _target_from_cxsmiles(str(row.get("cxsmiles_opt") or row.get("target") or ""))
        )

    return Sample(
        id=str(row.get("id") or row.get("sample_id") or ""),
        mode=mode,
        image=_resolve_sample_image(str(image), image_root),
        prompt=str(row.get("prompt") or ""),
        target=target,
    )


def _is_smiles_mode(mode: str) -> bool:
    return mode in SMILES_MODES or "ocsr" in mode or "smiles" in mode


def _select_samples(
    rows: Iterable[dict[str, Any]],
    max_per_mode: int,
    image_root: Path | None = None,
) -> list[Sample]:
    selected: list[Sample] = []
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for row in rows:
        sample = _sample_from_row(row, image_root=image_root)
        if sample is None or not sample.target or not sample.image:
            continue
        if sample.id in seen_ids:
            continue
        if counts.get(sample.mode, 0) >= max_per_mode:
            continue
        selected.append(sample)
        seen_ids.add(sample.id)
        counts[sample.mode] = counts.get(sample.mode, 0) + 1
    return selected


def _canonical_smiles(value: str) -> str | None:
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    return str(Chem.MolToSmiles(mol, canonical=True))


def _clean_smiles_prediction(prediction: str) -> str:
    prediction = prediction.strip().replace("`", "")
    for line in prediction.splitlines():
        line = line.strip()
        if line:
            return re.sub(r"^<[^>]+>", "", line).strip().split()[0]
    return ""


def _markush_flags(prediction: str) -> dict[str, bool]:
    lower = prediction.lower()
    return {
        "has_markush": "<markush" in lower,
        "has_cxsmi": "<cxsmi" in lower,
        "closed_cxsmi": "</cxsmi>" in lower,
        "closed_markush": "</markush>" in lower,
        "has_coordinate_block": "|(" in prediction,
    }


def _evaluate_prediction(sample: Sample, prediction: str) -> Result:
    result = Result(
        id=sample.id,
        mode=sample.mode,
        image=sample.image,
        prompt=sample.prompt,
        target=sample.target,
        prediction=prediction,
        exact_match=prediction.strip() == sample.target.strip(),
    )
    if _is_smiles_mode(sample.mode):
        pred_smiles = _clean_smiles_prediction(prediction)
        result.target_canonical = _canonical_smiles(sample.target)
        result.prediction_canonical = _canonical_smiles(pred_smiles)
        result.valid_smiles = result.prediction_canonical is not None
        result.canonical_match = result.target_canonical == result.prediction_canonical
    else:
        for key, value in _markush_flags(prediction).items():
            setattr(result, key, value)
    return result


def _print_result(index: int, total: int, result: Result, max_chars: int) -> None:
    print("\n" + "=" * 88)
    print(f"[{index}/{total}] {result.mode} | {result.id}")
    print(f"PROMPT: {result.prompt[:max_chars].replace(chr(10), ' | ')}")
    print(f"TARGET: {result.target[:max_chars]}")
    print(f"PRED  : {result.prediction[:max_chars]}")
    print(f"EXACT : {result.exact_match}")
    if _is_smiles_mode(result.mode):
        print(
            "SMILES:",
            f"valid={result.valid_smiles}",
            f"canonical_match={result.canonical_match}",
            f"target_can={result.target_canonical}",
            f"pred_can={result.prediction_canonical}",
        )
    else:
        print(
            "MARKUSH_FLAGS:",
            f"has_markush={result.has_markush}",
            f"has_cxsmi={result.has_cxsmi}",
            f"closed_cxsmi={result.closed_cxsmi}",
            f"closed_markush={result.closed_markush}",
            f"coordinate_block={result.has_coordinate_block}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Base Qwen-VL model path.")
    parser.add_argument("--checkpoint", required=True, help="LoRA checkpoint directory.")
    parser.add_argument("--samples", required=True, type=Path, help="SFT/unified JSONL samples.")
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Root for relative image paths in SFT/unified rows.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for inference.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32", "auto"),
        help="Model dtype.",
    )
    parser.add_argument("--max-per-mode", type=int, default=2, help="Samples per mode.")
    parser.add_argument(
        "--scan-limit",
        type=int,
        help="Maximum JSONL rows to scan before running generations.",
    )
    parser.add_argument("--max-new-tokens-markush", type=int, default=384)
    parser.add_argument("--max-new-tokens-smiles", type=int, default=128)
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        help="Minimum processor image pixels; omitted uses model default.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=1_048_576,
        help="Maximum processor image pixels.",
    )
    parser.add_argument("--max-print-chars", type=int, default=1000)
    parser.add_argument("--no-merge", action="store_true", help="Keep LoRA unmerged.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.scan_limit is not None and args.scan_limit <= 0:
        raise SystemExit("--scan-limit must be positive")
    rows = _iter_jsonl(args.samples.expanduser(), limit=args.scan_limit)
    samples = _select_samples(
        rows,
        max_per_mode=args.max_per_mode,
        image_root=args.image_root.expanduser() if args.image_root else None,
    )
    if not samples:
        raise SystemExit(f"No usable samples found in {args.samples}")

    print(f"Loading model: base={args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")
    model, processor = load_model_and_processor(
        args.base_model,
        args.checkpoint,
        args.device,
        dtype=args.dtype,
        merge_adapter=not args.no_merge,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )

    results: list[Result] = []
    for index, sample in enumerate(samples, 1):
        max_new_tokens = (
            args.max_new_tokens_smiles
            if _is_smiles_mode(sample.mode)
            else args.max_new_tokens_markush
        )
        record = EvalRecord(
            sample_id=sample.id,
            image_path=sample.image,
            cxsmiles_opt=sample.target,
            annotation=sample.target,
            prompt=sample.prompt,
        )
        prediction = generate_one(
            model,
            processor,
            record,
            max_new_tokens=max_new_tokens,
            device=args.device,
        )
        result = _evaluate_prediction(sample, prediction)
        results.append(result)
        _print_result(index, len(samples), result, args.max_print_chars)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(result) for result in results]
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
