#!/usr/bin/env python3
"""Rescore Markush predictions with DS4SD MarkushGrapher-2's public scorer.

This module is intentionally a thin adapter around
``markushgrapher.utils.ocsr.utils_evaluation.get_scores``.  The public scorer is
imported from vendored checkouts (or explicit paths) and is never modified.

Two prediction-conversion / ground-truth contracts are supported:

``official`` (default)
    Reproduces the paper's Markush headline exactly.  Predictions are decoded
    with ``CXSMILESTokenizer(training_dataset="mdu", condense_labels=True)
    .convert_opt_to_out`` followed by ``fix_cxsmiles``; a prediction that fails
    to convert or does not parse in RDKit is treated as ``None`` (incorrect).
    Ground truth is taken from DS4SD's ``gt_cxsmiles`` field (via ``--gt-jsonl``
    or a ``gt_cxsmiles`` field on the prediction samples), retaining unparseable
    rows so the full denominator is preserved.  Scoring uses
    ``get_scores(cxsmiles=True, markush=False, remove_stereo=True)``: the exact
    pinned call from the official out-of-repo pipeline.

``generic``
    The generic ``<r>``-tag adapter.  It only normalizes input convention before
    calling MG2 (unwraps ``<markush><cxsmi>...</cxsmi>``, optionally converts the
    local inline ``<r>R1</r>`` convention to RDKit CXSMILES atom labels, and
    inserts the RDKit-required space before CX extensions such as ``|Sg:...|``),
    takes ground truth from the model-format field, and scores with
    ``get_scores(cxsmiles=True, markush=True)``.  It does not filter invalid
    ground-truth rows before calling the scorer.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VENDOR_GRAPHER = ROOT / "tools" / "mg2_scorer" / "vendor" / "MarkushGrapher"
DEFAULT_VENDOR_GENERATOR = ROOT / "tools" / "mg2_scorer" / "vendor" / "MarkushGenerator"

#: The last vendor checkout paths actually configured. Once a caller overrides
#: the defaults (e.g. via ``--vendor-*``), later no-argument
#: ``configure_public_mg2_imports()`` calls from deep helpers reuse those paths
#: instead of silently re-applying the (possibly absent) defaults.
_CONFIGURED_GRAPHER: Path | None = None
_CONFIGURED_GENERATOR: Path | None = None

_RE_CXSMI = re.compile(r"<cxsmi>(.*?)</cxsmi>", re.DOTALL)
_RE_R_TAG = re.compile(r"<r>(.*?)</r>", re.DOTALL)


def configure_public_mg2_imports(
    *,
    grapher_path: Path | None = None,
    generator_path: Path | None = None,
) -> None:
    """Put vendored public MG2 repositories on ``sys.path``.

    The function is idempotent and intentionally does not import MG2 until after
    paths are configured, so callers can override the checkout locations. The
    effective paths are remembered module-side: once a caller passes explicit
    ``grapher_path``/``generator_path`` (e.g. threaded from ``--vendor-*``),
    subsequent no-argument calls reuse them rather than re-validating the
    defaults, which may be absent when only custom vendor dirs are present.
    """

    global _CONFIGURED_GRAPHER, _CONFIGURED_GENERATOR

    grapher = grapher_path or _CONFIGURED_GRAPHER or DEFAULT_VENDOR_GRAPHER
    generator = generator_path or _CONFIGURED_GENERATOR or DEFAULT_VENDOR_GENERATOR

    for path in (generator, grapher):
        if not path.exists():
            raise FileNotFoundError(
                f"Public MG2 dependency checkout not found: {path}. "
                "Clone DS4SD/MarkushGrapher and DS4SD/MarkushGenerator under "
                "tools/mg2_scorer/vendor or pass explicit --vendor-* paths."
            )
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    _CONFIGURED_GRAPHER = grapher
    _CONFIGURED_GENERATOR = generator


def _load_public_get_scores(
    *,
    grapher_path: Path | None = None,
    generator_path: Path | None = None,
):
    configure_public_mg2_imports(grapher_path=grapher_path, generator_path=generator_path)
    module = importlib.import_module("markushgrapher.utils.ocsr.utils_evaluation")
    return module.get_scores


def _load_rdkit_chem(
    *,
    grapher_path: Path | None = None,
    generator_path: Path | None = None,
):
    configure_public_mg2_imports(grapher_path=grapher_path, generator_path=generator_path)
    module = importlib.import_module("rdkit.Chem")
    return module


def load_official_conversion_tools(
    *,
    grapher_path: Path | None = None,
    generator_path: Path | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Load the vendored tools for the official prediction-conversion contract.

    Returns ``(cxsmiles_tokenizer, abbreviation, fix_cxsmiles, rdkit_chem)``:

    * ``cxsmiles_tokenizer``: ``CXSMILESTokenizer(training_dataset="mdu",
      condense_labels=True)``, whose ``convert_opt_to_out`` decodes the inline
      optimized CXSMILES the model emits;
    * ``abbreviation``: DS4SD's ``Abbreviation`` loaded from the vendored
      ``abbreviations_2.json`` (used by ``fix_cxsmiles`` to expand groups);
    * ``fix_cxsmiles``: the DS4SD post-conversion normalizer; and
    * ``rdkit_chem``: ``rdkit.Chem`` for the final validity check.

    These are the exact objects the official out-of-repo pipeline uses; nothing
    here modifies the vendored scorer.
    """

    from importlib.resources import files

    configure_public_mg2_imports(grapher_path=grapher_path, generator_path=generator_path)
    cxsmiles_tokenizer_cls = importlib.import_module(
        "markushgenerator.cxsmiles_tokenizer"
    ).CXSMILESTokenizer
    abbreviation_cls = importlib.import_module(
        "markushgrapher.utils.ocsr.abbreviation"
    ).Abbreviation
    fix_cxsmiles = importlib.import_module(
        "markushgrapher.utils.ocsr.utils_evaluation"
    ).fix_cxsmiles
    chem = _load_rdkit_chem(grapher_path=grapher_path, generator_path=generator_path)

    tokenizer = cxsmiles_tokenizer_cls(training_dataset="mdu", condense_labels=True)
    abbreviations_path = files("markushgenerator").joinpath("data", "abbreviations_2.json")
    with open(str(abbreviations_path), encoding="utf-8") as handle:
        abbreviation = abbreviation_cls(json.load(handle))
    return tokenizer, abbreviation, fix_cxsmiles, chem


def convert_prediction_official(
    raw_prediction: Any,
    *,
    tokenizer: Any,
    abbreviation: Any,
    fix_cxsmiles: Any,
    chem: Any,
) -> dict[str, Any]:
    """Convert one raw model prediction under the official contract.

    Mirrors the official pipeline exactly: extract the ``<cxsmi>...</cxsmi>``
    payload (a closing tag is required), strip all spaces, decode with
    ``convert_opt_to_out``, run ``fix_cxsmiles``, and require the result to parse
    in RDKit.  Any failure leaves ``final_prediction`` as ``None`` and records a
    ``failure`` reason; ``None`` predictions score as incorrect downstream.
    """

    record: dict[str, Any] = {
        "cxsmiles_opt": None,
        "converted_before_fix": None,
        "fixed_prediction": None,
        "final_prediction": None,
        "failure": None,
    }
    try:
        match = _RE_CXSMI.search("" if raw_prediction is None else str(raw_prediction))
        if match is None:
            record["failure"] = "missing_cxsmi_payload"
            return record
        opt = match.group(1).replace(" ", "")
        record["cxsmiles_opt"] = opt
        converted = tokenizer.convert_opt_to_out(opt)
        record["converted_before_fix"] = converted
        fixed = fix_cxsmiles(converted, abbreviation)
        record["fixed_prediction"] = fixed
        if fixed is None or chem.MolFromSmiles(fixed) is None:
            record["failure"] = "invalid_after_official_conversion"
            return record
        record["final_prediction"] = fixed
        return record
    except Exception as exc:  # match the official pipeline's blanket failure path
        record["failure"] = f"{type(exc).__name__}: {exc}"
        return record


def load_gt_mapping(
    path: Path,
    *,
    gt_field: str = "gt_cxsmiles",
    id_field: str = "id",
) -> dict[str, Any]:
    """Load a DS4SD ground-truth file into an ``{id: gt_cxsmiles}`` mapping.

    Accepts the DS4SD ground-truth bundle (shipped as ``ds4sd_mg2_predictions_878.jsonl``;
    DS4SD's own eval writes it as ``predictions_878.jsonl``) or any JSON/JSONL carrying an
    id field and a ``gt_cxsmiles`` field).  ``None`` ground-truth values are
    retained on purpose: they must count as incorrect while remaining in the
    denominator.
    """

    _, rows = load_prediction_payload(path)
    mapping: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get(id_field)
        if row_id is None:
            continue
        mapping[str(row_id)] = row.get(gt_field)
    return mapping


def extract_cxsmiles_payload(value: Any) -> str | None:
    """Extract a raw CXSMILES string from known prediction wrappers."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return ""
    match = _RE_CXSMI.search(text)
    if match:
        return match.group(1).strip()
    return text


def normalize_cxsmiles_spacing(cxsmiles: str | None) -> str | None:
    """Normalize RDKit CXSMILES spacing without changing scorer semantics."""

    if cxsmiles is None:
        return None
    text = cxsmiles.strip()
    if not text or "|" not in text:
        return text
    if text.count("|") % 2 == 1:
        text = f"{text}|"
    first_pipe = text.find("|")
    if first_pipe > 0 and not text[first_pipe - 1].isspace():
        text = f"{text[:first_pipe]} {text[first_pipe:]}"
    return text


def convert_local_r_tags_to_rdkit_cxsmiles(cxsmiles: str) -> str:
    """Convert local inline ``<r>LABEL</r>`` tags to RDKit atom-label CXSMILES.

    MG2's public scorer expects RDKit-style CXSMILES, where Markush labels are
    atom properties in the CX extension.  Local eval artifacts often store the
    same labels inline as XML-ish ``<r>...</r>`` tokens.  This conversion is an
    input adapter only: after conversion, scoring is delegated to MG2 unchanged.
    """

    chem = _load_rdkit_chem()
    labels: list[str] = []

    def replace_tag(match: re.Match[str]) -> str:
        labels.append(match.group(1).strip())
        return "*"

    tagged = _RE_R_TAG.sub(replace_tag, cxsmiles)
    tagged = normalize_cxsmiles_spacing(tagged) or ""
    if "|" in tagged:
        core, extension = tagged.split("|", 1)
        core = core.strip()
        extension = extension.rsplit("|", 1)[0].strip()
    else:
        core = tagged.strip()
        extension = ""

    parser_params = chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    mol = chem.MolFromSmiles(core, parser_params)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse local-r-tag Markush core after conversion: {core!r}"
        )

    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) < len(labels):
        raise ValueError(
            f"Expected at least {len(labels)} R atoms, found {len(dummy_atoms)} after RDKit parse"
        )
    atom_labels = [""] * mol.GetNumAtoms()
    for atom, label in zip(dummy_atoms, labels, strict=False):
        atom_labels[atom.GetIdx()] = label
    label_section = "$" + ";".join(atom_labels) + "$"
    combined_extension = f"{label_section},{extension}" if extension else label_section
    return f"{core} |{combined_extension}|"


def prepare_for_public_mg2(value: Any, *, convert_local_r_tags: bool = True) -> str | None:
    """Prepare a single artifact value for MG2's public scorer."""

    cxsmiles = extract_cxsmiles_payload(value)
    if cxsmiles is None:
        return None
    if convert_local_r_tags and _RE_R_TAG.search(cxsmiles):
        return convert_local_r_tags_to_rdkit_cxsmiles(cxsmiles)
    return normalize_cxsmiles_spacing(cxsmiles)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _json_default(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_json(v) for v in value]
    if hasattr(value, "item"):
        return _sanitize_json(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_prediction_payload(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a JSON/JSONL prediction artifact and return ``(payload, samples)``."""

    if path.suffix == ".jsonl":
        samples = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        return {"samples": samples}, samples

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"samples": payload}, payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a samples[] list")
    dict_samples = [sample for sample in samples if isinstance(sample, dict)]
    return payload, dict_samples


def _first_field(sample: dict[str, Any], field_names: Sequence[str]) -> Any:
    for field_name in field_names:
        if field_name in sample and sample[field_name] not in (None, ""):
            return sample[field_name]
    return None


def parse_field_list(value: str) -> list[str]:
    fields = [part.strip() for part in value.split(",") if part.strip()]
    if not fields:
        raise ValueError("field list must not be empty")
    return fields


def _exact_counts(details: dict[int, dict[str, Any] | None]) -> dict[str, Any]:
    values = [
        score.get("cxsmi_equality")
        for score in details.values()
        if score is not None and score.get("cxsmi_equality") is not None
    ]
    count = sum(1 for value in values if bool(value))
    total = len(values)
    return {
        "cxsmi_equality_count": count,
        "cxsmi_equality_total": total,
        "cxsmi_equality_accuracy": (count / total) if total else None,
        "cxsmi_equality_full_accuracy": (count / len(details)) if details else None,
        "invalid_gt_count": sum(score is None for score in details.values()),
        "size": len(details),
    }


def score_samples_with_public_mg2(
    samples: Sequence[dict[str, Any]],
    *,
    contract: str = "official",
    gt_fields: Sequence[str] = ("ground_truth", "gt_cxsmiles", "cxsmiles"),
    pred_fields: Sequence[str] = ("prediction_cxsmiles", "prediction", "cxsmiles"),
    id_fields: Sequence[str] = ("sample_id", "id", "index"),
    convert_local_r_tags: bool = True,
    training_smiles: Sequence[str] = (),
    limit: int | None = None,
    markush: bool | None = None,
    remove_stereo: bool = True,
    gt_by_id: dict[str, Any] | None = None,
    conversion_tools: tuple[Any, Any, Any, Any] | None = None,
    get_scores_fn: Any | None = None,
    grapher_path: Path | None = None,
    generator_path: Path | None = None,
) -> dict[str, Any]:
    """Run MG2 public ``get_scores`` on artifact samples.

    ``contract`` selects the prediction-conversion / ground-truth contract
    (``"official"`` (default) or ``"generic"``).  When ``markush`` is ``None``
    it defaults to the contract's pinned value (``True`` for generic, ``False``
    for official).

    ``grapher_path``/``generator_path`` override the vendored MG2 checkout
    locations (threaded from ``--vendor-*``). They are configured once up front
    so every downstream loader — including deep helpers such as
    :func:`prepare_for_public_mg2` — reuses them instead of re-applying the
    (possibly absent) defaults.
    """

    if contract not in ("generic", "official"):
        raise ValueError(f"unknown contract: {contract!r} (expected 'generic' or 'official')")
    if markush is None:
        markush = contract == "generic"

    # Configure vendor paths once so every downstream (no-argument) loader call
    # reuses these locations rather than re-validating the defaults.
    configure_public_mg2_imports(grapher_path=grapher_path, generator_path=generator_path)

    selected = list(samples[:limit] if limit is not None else samples)
    prepared_gt: list[str | None] = []
    prepared_pred: list[str | None] = []
    rows: list[dict[str, Any]] = []

    if contract == "official":
        tokenizer, abbreviation, fix_cxsmiles, chem = conversion_tools or (
            load_official_conversion_tools(grapher_path=grapher_path, generator_path=generator_path)
        )
        for idx, sample in enumerate(selected):
            sample_id = _first_field(sample, id_fields)
            raw_pred = _first_field(sample, pred_fields)
            conversion = convert_prediction_official(
                raw_pred,
                tokenizer=tokenizer,
                abbreviation=abbreviation,
                fix_cxsmiles=fix_cxsmiles,
                chem=chem,
            )
            if gt_by_id is not None:
                if sample_id is None or str(sample_id) not in gt_by_id:
                    raise KeyError(
                        f"sample id {sample_id!r} (index {idx}) not found in the "
                        "ground-truth mapping supplied via --gt-jsonl"
                    )
                raw_gt = gt_by_id[str(sample_id)]
            else:
                raw_gt = _first_field(sample, gt_fields)
            row: dict[str, Any] = {
                "index": sample.get("index", idx),
                "sample_id": sample_id,
                "ground_truth": raw_gt,
                "prediction": raw_pred,
                "cxsmiles_opt": conversion["cxsmiles_opt"],
                "conversion_failure": conversion["failure"],
                "mg2_ground_truth": raw_gt,
                "mg2_prediction": conversion["final_prediction"],
            }
            prepared_gt.append(raw_gt)
            prepared_pred.append(conversion["final_prediction"])
            rows.append(row)
    else:
        for idx, sample in enumerate(selected):
            raw_gt = _first_field(sample, gt_fields)
            raw_pred = _first_field(sample, pred_fields)
            row = {
                "index": sample.get("index", idx),
                "sample_id": _first_field(sample, id_fields),
                "ground_truth": raw_gt,
                "prediction": raw_pred,
                "preparation_errors": [],
            }
            try:
                mg2_gt = prepare_for_public_mg2(raw_gt, convert_local_r_tags=convert_local_r_tags)
            except (
                Exception
            ) as exc:  # preserve MG2 GT-failure behavior by passing raw value through
                mg2_gt = normalize_cxsmiles_spacing(extract_cxsmiles_payload(raw_gt))
                row["preparation_errors"].append(f"ground_truth: {type(exc).__name__}: {exc}")
            try:
                mg2_pred = prepare_for_public_mg2(
                    raw_pred, convert_local_r_tags=convert_local_r_tags
                )
            except Exception as exc:
                mg2_pred = normalize_cxsmiles_spacing(extract_cxsmiles_payload(raw_pred))
                row["preparation_errors"].append(f"prediction: {type(exc).__name__}: {exc}")
            row["mg2_ground_truth"] = mg2_gt
            row["mg2_prediction"] = mg2_pred
            prepared_gt.append(mg2_gt)
            prepared_pred.append(mg2_pred)
            rows.append(row)

    get_scores = get_scores_fn or _load_public_get_scores(
        grapher_path=grapher_path, generator_path=generator_path
    )
    aggregate, details = get_scores(
        prepared_gt,
        prepared_pred,
        list(training_smiles),
        cxsmiles=True,
        markush=markush,
        get_unreduced_scores=True,
        remove_stereo=remove_stereo,
    )

    typed_details: dict[int, dict[str, Any] | None] = dict(details)
    for idx, row in enumerate(rows):
        row["score"] = typed_details.get(idx)
        source_score = selected[idx].get("scores") if isinstance(selected[idx], dict) else None
        if isinstance(source_score, dict):
            row["source_scores"] = source_score

    exact_counts = _exact_counts(typed_details)
    if contract == "official":
        exact_counts["conversion_failure_count"] = sum(
            1 for row in rows if row["conversion_failure"] is not None
        )

    return {
        "aggregate": _sanitize_json(aggregate),
        "exact_counts": exact_counts,
        "samples": _sanitize_json(rows),
    }


def build_output_payload(
    *,
    input_path: Path,
    source_payload: dict[str, Any],
    score_payload: dict[str, Any],
    args: argparse.Namespace,
    gt_fields: Sequence[str],
    pred_fields: Sequence[str],
    markush: bool,
) -> dict[str, Any]:
    return {
        "config": {
            "scorer": "DS4SD/MarkushGrapher public get_scores",
            "contract": args.contract,
            "input_json": str(input_path),
            "gt_jsonl": str(args.gt_jsonl) if args.gt_jsonl is not None else None,
            "gt_fields": list(gt_fields),
            "pred_fields": list(pred_fields),
            "sample_id_fields": parse_field_list(args.sample_id_field),
            "convert_local_r_tags": args.convert_local_r_tags,
            "markush": markush,
            "remove_stereo": True,
            "limit": args.limit,
            "vendor_grapher": str(args.vendor_grapher),
            "vendor_generator": str(args.vendor_generator),
        },
        "source_aggregate_both_protocols": source_payload.get("aggregate_both_protocols"),
        "public_mg2": score_payload,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--contract",
        choices=("official", "generic"),
        default="official",
        help=(
            "Prediction-conversion / ground-truth contract. 'official' (default) "
            "reproduces the paper headline via CXSMILESTokenizer.convert_opt_to_out "
            "+ DS4SD gt_cxsmiles; 'generic' is the `<r>`-tag adapter."
        ),
    )
    parser.add_argument(
        "--gt-jsonl",
        type=Path,
        default=None,
        help=(
            "Official contract only: JSON/JSONL of DS4SD ground truth (e.g. "
            "ds4sd_mg2_predictions_878.jsonl) joined to prediction samples by id. If omitted, "
            "the ground truth is read from a gt_cxsmiles field on the prediction samples."
        ),
    )
    parser.add_argument(
        "--gt-id-field",
        default="id",
        help="Id field used to join --gt-jsonl rows to prediction samples (default: id).",
    )
    parser.add_argument(
        "--gt-field",
        default=None,
        help=(
            "Comma-separated ground-truth field priority. Defaults to 'gt_cxsmiles' "
            "for the official contract and 'ground_truth,gt_cxsmiles,cxsmiles' for generic."
        ),
    )
    parser.add_argument(
        "--pred-field",
        default=None,
        help=(
            "Comma-separated prediction field priority. Defaults to 'prediction' for the "
            "official contract and 'prediction_cxsmiles,prediction,cxsmiles' for generic."
        ),
    )
    parser.add_argument("--sample-id-field", default="sample_id,id,index")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vendor-grapher", type=Path, default=DEFAULT_VENDOR_GRAPHER)
    parser.add_argument("--vendor-generator", type=Path, default=DEFAULT_VENDOR_GENERATOR)
    parser.add_argument(
        "--convert-local-r-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generic contract only: convert local <r>LABEL</r> tags to RDKit CXSMILES "
            "atom labels before scoring."
        ),
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.contract == "generic" and args.gt_jsonl is not None:
        raise SystemExit("--gt-jsonl is only supported with --contract official")

    if args.gt_field is not None:
        gt_fields = parse_field_list(args.gt_field)
    elif args.contract == "official":
        gt_fields = ["gt_cxsmiles"]
    else:
        gt_fields = ["ground_truth", "gt_cxsmiles", "cxsmiles"]

    if args.pred_field is not None:
        pred_fields = parse_field_list(args.pred_field)
    elif args.contract == "official":
        pred_fields = ["prediction"]
    else:
        pred_fields = ["prediction_cxsmiles", "prediction", "cxsmiles"]

    markush = args.contract == "generic"

    configure_public_mg2_imports(
        grapher_path=args.vendor_grapher,
        generator_path=args.vendor_generator,
    )
    source_payload, samples = load_prediction_payload(args.input_json)

    gt_by_id = None
    if args.contract == "official" and args.gt_jsonl is not None:
        gt_by_id = load_gt_mapping(
            args.gt_jsonl,
            gt_field=gt_fields[0],
            id_field=args.gt_id_field,
        )

    score_payload = score_samples_with_public_mg2(
        samples,
        contract=args.contract,
        gt_fields=gt_fields,
        pred_fields=pred_fields,
        id_fields=parse_field_list(args.sample_id_field),
        convert_local_r_tags=args.convert_local_r_tags,
        markush=markush,
        gt_by_id=gt_by_id,
        limit=args.limit,
        grapher_path=args.vendor_grapher,
        generator_path=args.vendor_generator,
    )
    output = build_output_payload(
        input_path=args.input_json,
        source_payload=source_payload,
        score_payload=score_payload,
        args=args,
        gt_fields=gt_fields,
        pred_fields=pred_fields,
        markush=markush,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=args.indent, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["public_mg2"]["exact_counts"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
