"""Check prepared Markush train/eval artifacts for target-contract consistency.

Library home for the artifact-validation logic (``check_artifacts`` and its
helpers): given prepared train/eval JSONL, it verifies the target-format
contract, canonicalization metadata, and (optionally) that referenced images
resolve. ``glyph.data.markush_multitask`` uses it to validate the multitask
mix before it is written.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from glyph.markush.data.canonicalize import canonicalize_cxsmiles_checked
from glyph.markush.data.example_modes import MOLECULE_OCSR, MOLECULE_OCSR_PROMPT
from glyph.markush.data.ocsr import (
    OCSR_CANONICAL_STATUSES,
    OCSR_TARGET_FORMAT,
    RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX,
    canonicalize_ocsr_smiles_checked,
    row_declares_ocsr_smiles,
)
from glyph.markush.data.structured_transcripts import (
    SCHEMA_VERSION as STRUCTURED_TRANSCRIPT_SCHEMA_VERSION,
)
from glyph.markush.data.structured_transcripts import validate_structured_transcript_content
from glyph.markush.eval import common
from glyph.markush.eval.benchmarks import get_benchmark_spec, normalize_benchmark_name

DEFAULT_TARGET_FORMAT = "canonicalized cxsmiles_opt"
RAW_TARGET_FORMAT = "cxsmiles_opt"
STRUCTURED_TRANSCRIPT_TARGET_FORMAT = "structured transcript + final canonicalized cxsmiles_opt"
CANONICAL_SOURCE_SUFFIX = "->canonicalized_cxsmiles_opt"
STRUCTURED_TRANSCRIPT_SOURCE_SUFFIX = f"{CANONICAL_SOURCE_SUFFIX}->structured_transcript"
CANONICAL_STATUSES = {"canonicalized", "already_canonical"}
OCSR_AS_MARKUSH_NO_EXTENSION_MODE = "molecule_ocsr_as_markush_no_extension"
SCALAR_IMAGE_KEYS = ("image", "image_path", "path", "png_path", "rendered_path", "file_name")


@dataclass(frozen=True)
class CheckFailure:
    path: str
    line: int
    field: str
    message: str


@dataclass(frozen=True)
class CanonicalPayloadJob:
    path: str
    line: int
    cxsmiles: str


@dataclass(frozen=True)
class StructuredTranscriptJob:
    path: str
    line: int
    content: str
    require_canonicalized: bool


@dataclass(frozen=True)
class OCSRSmilesJob:
    path: str
    line: int
    smiles: str


@dataclass(frozen=True)
class RowText:
    prompt: str
    target: str
    prompt_from_messages: bool
    target_from_messages: bool


def target_instruction(expected_target_format: str) -> str:
    return (
        "Target format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; "
        f"the <cxsmi> value must use {expected_target_format}."
    )


def ocsr_smiles_instruction() -> str:
    return "Target format: return a SMILES string."


def structured_transcript_instruction(final_target_format: str = DEFAULT_TARGET_FORMAT) -> str:
    return (
        "Target format: first return one compact JSON object using schema "
        f"{STRUCTURED_TRANSCRIPT_SCHEMA_VERSION}, then return "
        "<markush><cxsmi>...</cxsmi><stable>...</stable></markush>; the final <cxsmi> "
        f"value must use {final_target_format}. Do not include explanations."
    )


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append((line_no, value))
    return rows


def image_exists(path: str, *, jsonl_path: Path) -> bool:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = jsonl_path.parent / image_path
    return image_path.is_file()


def _image_path_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("path")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def images_list_paths(row: dict[str, Any]) -> list[str]:
    images = row.get("images")
    if not isinstance(images, list):
        return []
    paths: list[str] = []
    for image in images:
        value = _image_path_value(image)
        if value and value not in paths:
            paths.append(value)
    return paths


def scalar_image_paths(row: dict[str, Any]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for key in SCALAR_IMAGE_KEYS:
        value = _image_path_value(row.get(key))
        if value:
            paths.append((key, value))
    return paths


def row_image_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for _key, value in scalar_image_paths(row):
        paths.append(value)
        break
    images = row.get("images")
    if isinstance(images, list):
        for image in images:
            value = _image_path_value(image)
            if value and value not in paths:
                paths.append(value)
    return paths


def eval_benchmark_from_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("eval_"):
        raise ValueError(f"eval JSONL name must start with eval_: {path.name}")
    return normalize_benchmark_name(stem.removeprefix("eval_"))


def ocsr_eval_benchmark_from_path(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("eval_ocsr_"):
        raise ValueError(f"OCSR eval JSONL name must start with eval_ocsr_: {path.name}")
    return stem.removeprefix("eval_ocsr_").strip().lower().replace("-", "_")


def expected_eval_prompt(benchmark: str, expected_target_format: str) -> str:
    spec = get_benchmark_spec(benchmark)
    return f"{spec.prompt}\n{target_instruction(expected_target_format)}"


def expected_target_format_for_row(
    row_target_format: Any,
    *,
    expected_target_format: str,
    require_canonicalized: bool,
) -> str:
    if require_canonicalized:
        return expected_target_format
    if isinstance(row_target_format, str) and row_target_format in {
        expected_target_format,
        RAW_TARGET_FORMAT,
    }:
        return row_target_format
    return expected_target_format


def add_failure(
    failures: list[CheckFailure],
    path: Path,
    line: int,
    field: str,
    message: str,
) -> None:
    failures.append(CheckFailure(str(path), line, field, message))


def message_content(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def strip_image_prefix(prompt: str) -> str:
    prompt = str(prompt or "")
    if prompt.startswith("<image>\n"):
        return prompt.removeprefix("<image>\n").strip()
    if prompt.startswith("<image>"):
        return prompt.removeprefix("<image>").strip()
    return prompt.strip()


def expected_ocsr_prompt() -> str:
    return f"{MOLECULE_OCSR_PROMPT}\n{ocsr_smiles_instruction()}"


def declared_task(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "").lower()
    mode = str(row.get("mode") or "").lower()
    target_format = row.get("target_format")
    if task == "ocsr" or mode == MOLECULE_OCSR or target_format == OCSR_TARGET_FORMAT:
        return "ocsr"
    return "markush"


def declares_ocsr_as_markush_no_extension(row: dict[str, Any]) -> bool:
    return str(row.get("mode") or "") == OCSR_AS_MARKUSH_NO_EXTENSION_MODE


def row_text(
    row: dict[str, Any],
    *,
    path: Path,
    line: int,
    require_messages: bool,
    failures: list[CheckFailure],
) -> RowText:
    user_message = message_content(row, "user")
    assistant_message = message_content(row, "assistant")
    prompt_field = row.get("prompt")
    annotation = row.get("annotation")
    ground_truth = row.get("ground_truth")

    if require_messages and not user_message:
        add_failure(failures, path, line, "messages", "row has no user message")
    if require_messages and not assistant_message:
        add_failure(failures, path, line, "messages", "row has no assistant target message")

    if user_message and not user_message.startswith("<image>\n"):
        add_failure(failures, path, line, "messages", "user message must start with <image>")
    if (
        user_message
        and isinstance(prompt_field, str)
        and prompt_field.strip()
        and strip_image_prefix(user_message) != prompt_field.strip()
    ):
        add_failure(
            failures,
            path,
            line,
            "prompt",
            "legacy prompt field does not match messages[0].content",
        )

    prompt = user_message or (prompt_field if isinstance(prompt_field, str) else "")
    target = assistant_message
    if not target and isinstance(annotation, str):
        target = annotation
    if not target and isinstance(ground_truth, str):
        target = ground_truth
    if not target and declared_task(row) == "ocsr":
        smiles = row.get("smiles")
        target = smiles if isinstance(smiles, str) else ""

    if assistant_message:
        if isinstance(annotation, str) and annotation and annotation != assistant_message:
            add_failure(
                failures,
                path,
                line,
                "annotation",
                "legacy annotation field does not match messages[1].content",
            )
        if isinstance(ground_truth, str) and ground_truth and ground_truth != assistant_message:
            add_failure(
                failures,
                path,
                line,
                "ground_truth",
                "legacy ground_truth field does not match messages[1].content",
            )
        smiles = row.get("smiles")
        if declared_task(row) == "ocsr" and isinstance(smiles, str) and smiles != assistant_message:
            add_failure(
                failures,
                path,
                line,
                "smiles",
                "legacy smiles field does not match messages[1].content",
            )

    return RowText(
        prompt=prompt,
        target=target,
        prompt_from_messages=bool(user_message),
        target_from_messages=bool(assistant_message),
    )


def add_summary_counts(summary: dict[str, Any], row: dict[str, Any]) -> None:
    summary["task_counts"][declared_task(row)] += 1
    summary["mode_counts"][row.get("mode")] += 1
    summary["target_format_counts"][row.get("target_format")] += 1
    summary["target_source_counts"][row.get("target_source")] += 1
    summary["source_counts"][row.get("source")] += 1
    summary["benchmark_counts"][row.get("benchmark")] += 1


def artifact_summary(
    path: Path, *, kind: str, rows: int, expected_benchmark: str = ""
) -> dict[str, Any]:
    return {
        "path": str(path),
        "kind": kind,
        "rows": rows,
        "expected_benchmark": expected_benchmark,
        "task_counts": Counter(),
        "mode_counts": Counter(),
        "target_format_counts": Counter(),
        "target_source_counts": Counter(),
        "source_counts": Counter(),
        "benchmark_counts": Counter(),
        "missing_images": 0,
    }


def check_row_images(
    row: dict[str, Any],
    *,
    path: Path,
    line: int,
    check_images: bool,
    require_absolute_images: bool,
    require_images_list: bool,
    failures: list[CheckFailure],
) -> int:
    missing = 0
    images = row.get("images")
    image_list_paths = images_list_paths(row)
    image_field = "images" if isinstance(images, list) or require_images_list else "image"
    if require_images_list and (not isinstance(images, list) or not images):
        add_failure(failures, path, line, "images", "row has no images list")
        missing += 1
    if isinstance(images, list):
        for index, image in enumerate(images):
            if not isinstance(image, str):
                add_failure(
                    failures,
                    path,
                    line,
                    "images",
                    f"images[{index}] must be a string path, found {image!r}",
                )
                missing += 1

    for key, scalar_path in scalar_image_paths(row):
        if image_list_paths and scalar_path not in image_list_paths:
            add_failure(
                failures,
                path,
                line,
                key,
                "legacy image field does not match the images list",
            )

    image_paths = row_image_paths(row)
    if not image_paths:
        add_failure(failures, path, line, image_field, "row has no image path")
        return missing + 1

    if not check_images:
        return missing

    for image in image_paths:
        if require_absolute_images and not Path(image).expanduser().is_absolute():
            add_failure(
                failures,
                path,
                line,
                image_field,
                f"image path is not absolute: {image!r}",
            )
            missing += 1
        elif not image_exists(image, jsonl_path=path):
            add_failure(failures, path, line, image_field, f"missing image {image!r}")
            missing += 1
    return missing


def _canonical_payload_problem(cxsmiles: str) -> str:
    result = canonicalize_cxsmiles_checked(cxsmiles)
    if result.value is None:
        return result.error or "canonicalizer could not parse target payload"
    if result.fallback:
        return result.error or "canonicalizer fell back to the original target"
    if result.value != cxsmiles:
        return f"payload is not canonicalized; canonical form is {result.value!r}"
    return ""


def _ocsr_smiles_problem(smiles: str) -> str:
    result = canonicalize_ocsr_smiles_checked(smiles)
    if result.value is None:
        return result.error or "RDKit could not parse OCSR SMILES target"
    if result.value != smiles:
        return f"payload is not RDKit-canonical isomeric SMILES; canonical form is {result.value!r}"
    return ""


def _structured_final_payload(content: str) -> tuple[str, str]:
    if not isinstance(content, str) or "\n" not in content:
        return "", ""
    _transcript_json, final_xml = content.split("\n", 1)
    return common.extract_cxsmiles_and_stable(final_xml, require_wrapper=True)


def _structured_transcript_problem(job: StructuredTranscriptJob) -> str:
    ok, error = validate_structured_transcript_content(job.content)
    if not ok:
        return error
    cxsmiles, _stable = _structured_final_payload(job.content)
    if not cxsmiles:
        return "structured transcript final XML has no CXSMILES payload"
    if job.require_canonicalized:
        return _canonical_payload_problem(cxsmiles)
    return ""


def run_structured_transcript_checks(
    jobs: list[StructuredTranscriptJob],
    *,
    workers: int,
) -> list[CheckFailure]:
    if not jobs:
        return []
    if workers <= 1:
        problems = [_structured_transcript_problem(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            problems = list(pool.map(_structured_transcript_problem, jobs))
    return [
        CheckFailure(job.path, job.line, "target_payload", problem)
        for job, problem in zip(jobs, problems, strict=True)
        if problem
    ]


def run_canonical_payload_checks(
    jobs: list[CanonicalPayloadJob],
    *,
    workers: int,
) -> list[CheckFailure]:
    if not jobs:
        return []
    if workers <= 1:
        problems = [_canonical_payload_problem(job.cxsmiles) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            problems = list(pool.map(_canonical_payload_problem, (job.cxsmiles for job in jobs)))
    return [
        CheckFailure(job.path, job.line, "target_payload", problem)
        for job, problem in zip(jobs, problems, strict=True)
        if problem
    ]


def run_ocsr_smiles_checks(
    jobs: list[OCSRSmilesJob],
    *,
    workers: int,
) -> list[CheckFailure]:
    if not jobs:
        return []
    if workers <= 1:
        problems = [_ocsr_smiles_problem(job.smiles) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            problems = list(pool.map(_ocsr_smiles_problem, (job.smiles for job in jobs)))
    return [
        CheckFailure(job.path, job.line, "target_payload", problem)
        for job, problem in zip(jobs, problems, strict=True)
        if problem
    ]


def is_ocsr_smiles_row(row: dict[str, Any]) -> bool:
    """Return whether a prepared train row declares the OCSR SMILES contract."""

    return row_declares_ocsr_smiles(row)


def check_ocsr_smiles_contract(
    row: dict[str, Any],
    *,
    path: Path,
    line: int,
    prompt: str,
    target: str,
    allow_ocsr_smiles_targets: bool,
    failures: list[CheckFailure],
    ocsr_smiles_jobs: list[OCSRSmilesJob],
) -> None:
    if not allow_ocsr_smiles_targets:
        add_failure(
            failures,
            path,
            line,
            "target_format",
            "OCSR SMILES targets are not allowed by this check invocation",
        )
        return
    task = row.get("task")
    if task is not None and str(task).lower() != "ocsr":
        add_failure(
            failures,
            path,
            line,
            "task",
            f"SMILES target rows must use task 'ocsr', found {task!r}",
        )
    if row.get("mode") != MOLECULE_OCSR:
        add_failure(
            failures,
            path,
            line,
            "mode",
            f"SMILES target rows must use mode {MOLECULE_OCSR!r}",
        )
    if row.get("target_format") != OCSR_TARGET_FORMAT:
        add_failure(
            failures,
            path,
            line,
            "target_format",
            f"expected {OCSR_TARGET_FORMAT!r}, found {row.get('target_format')!r}",
        )
    if strip_image_prefix(prompt) != expected_ocsr_prompt():
        add_failure(
            failures,
            path,
            line,
            "prompt",
            "OCSR prompt does not match the canonical SMILES extraction prompt",
        )
    if not target:
        add_failure(failures, path, line, "target_payload", "OCSR row has an empty SMILES target")
    elif "<markush" in target.lower() or "<cxsmi" in target.lower():
        add_failure(
            failures,
            path,
            line,
            "target_payload",
            "OCSR row must use bare SMILES, not Markush XML",
        )
    else:
        ocsr_smiles_jobs.append(OCSRSmilesJob(str(path), line, target))

    source = row.get("target_source")
    if not isinstance(source, str) or not source.endswith(
        RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX
    ):
        add_failure(
            failures,
            path,
            line,
            "target_source",
            "expected OCSR target_source ending in "
            f"{RDKIT_CANONICAL_ISOMERIC_SMILES_SOURCE_SUFFIX!r}, found {source!r}",
        )
    status = row.get("target_canonicalization_status")
    if status not in OCSR_CANONICAL_STATUSES:
        add_failure(
            failures,
            path,
            line,
            "target_canonicalization_status",
            f"expected one of {sorted(OCSR_CANONICAL_STATUSES)!r}, found {status!r}",
        )
    error = row.get("target_canonicalization_error")
    if error:
        add_failure(
            failures,
            path,
            line,
            "target_canonicalization_error",
            f"expected empty canonicalization error, found {error!r}",
        )


def check_target_contract(
    row: dict[str, Any],
    *,
    path: Path,
    line: int,
    prompt: str,
    target: str,
    expected_target_format: str,
    require_canonicalized: bool,
    allow_structured_transcript_targets: bool,
    failures: list[CheckFailure],
    canonical_jobs: list[CanonicalPayloadJob],
    structured_transcript_jobs: list[StructuredTranscriptJob],
) -> tuple[str, str]:
    target_format = row.get("target_format")
    contract_target_format = expected_target_format_for_row(
        target_format,
        expected_target_format=expected_target_format,
        require_canonicalized=require_canonicalized,
    )
    is_structured_transcript = (
        allow_structured_transcript_targets and target_format == STRUCTURED_TRANSCRIPT_TARGET_FORMAT
    )
    if target_format != contract_target_format and not is_structured_transcript:
        add_failure(
            failures,
            path,
            line,
            "target_format",
            f"expected {contract_target_format!r}"
            + (
                f" or {STRUCTURED_TRANSCRIPT_TARGET_FORMAT!r}"
                if allow_structured_transcript_targets
                else ""
            )
            + f", found {target_format!r}",
        )

    target_cxsmiles_format = row.get("target_cxsmiles_format")
    if (
        target_cxsmiles_format is not None
        and target_cxsmiles_format != contract_target_format
        and not is_structured_transcript
    ):
        add_failure(
            failures,
            path,
            line,
            "target_cxsmiles_format",
            f"expected {contract_target_format!r}, found {target_cxsmiles_format!r}",
        )

    expected_instruction = (
        structured_transcript_instruction(contract_target_format)
        if is_structured_transcript
        else target_instruction(contract_target_format)
    )
    if expected_instruction not in prompt:
        add_failure(
            failures,
            path,
            line,
            "prompt",
            "prompt does not include the expected target-format instruction",
        )

    if is_structured_transcript:
        cxsmiles, stable = _structured_final_payload(target)
        structured_transcript_jobs.append(
            StructuredTranscriptJob(
                str(path),
                line,
                target,
                require_canonicalized=require_canonicalized,
            )
        )
    else:
        cxsmiles, stable = common.extract_cxsmiles_and_stable(target, require_wrapper=False)
        if not cxsmiles:
            add_failure(
                failures,
                path,
                line,
                "target_payload",
                "target does not contain a non-empty <cxsmi> payload",
            )
        elif require_canonicalized:
            canonical_jobs.append(CanonicalPayloadJob(str(path), line, cxsmiles))

    if require_canonicalized:
        source = row.get("target_source")
        expected_suffix = (
            STRUCTURED_TRANSCRIPT_SOURCE_SUFFIX
            if is_structured_transcript
            else CANONICAL_SOURCE_SUFFIX
        )
        if not isinstance(source, str) or not source.endswith(expected_suffix):
            add_failure(
                failures,
                path,
                line,
                "target_source",
                f"expected a source ending in {expected_suffix!r}, found {source!r}",
            )
        status = row.get("target_canonicalization_status")
        if status not in CANONICAL_STATUSES:
            add_failure(
                failures,
                path,
                line,
                "target_canonicalization_status",
                f"expected one of {sorted(CANONICAL_STATUSES)!r}, found {status!r}",
            )
        error = row.get("target_canonicalization_error")
        if error:
            add_failure(
                failures,
                path,
                line,
                "target_canonicalization_error",
                f"expected empty canonicalization error, found {error!r}",
            )

    return cxsmiles, stable


def check_prepared_row(
    row: dict[str, Any],
    *,
    path: Path,
    line: int,
    kind: str,
    expected_target_format: str,
    expected_markush_prompt: str,
    require_canonicalized: bool,
    allow_structured_transcript_targets: bool,
    allow_ocsr_smiles_targets: bool,
    require_messages: bool,
    check_images: bool,
    require_absolute_images: bool,
    require_images_list: bool,
    failures: list[CheckFailure],
    canonical_jobs: list[CanonicalPayloadJob],
    structured_transcript_jobs: list[StructuredTranscriptJob],
    ocsr_smiles_jobs: list[OCSRSmilesJob],
) -> int:
    text = row_text(
        row,
        path=path,
        line=line,
        require_messages=require_messages,
        failures=failures,
    )
    if declared_task(row) == "ocsr" or is_ocsr_smiles_row(row):
        check_ocsr_smiles_contract(
            row,
            path=path,
            line=line,
            prompt=text.prompt,
            target=text.target,
            allow_ocsr_smiles_targets=allow_ocsr_smiles_targets,
            failures=failures,
            ocsr_smiles_jobs=ocsr_smiles_jobs,
        )
    else:
        if expected_markush_prompt:
            if not text.prompt:
                add_failure(failures, path, line, "prompt", "eval row has no prompt")
            elif strip_image_prefix(text.prompt) != expected_markush_prompt:
                add_failure(
                    failures,
                    path,
                    line,
                    "prompt",
                    "eval prompt does not match the expected benchmark prompt "
                    "with target-format instruction",
                )
        if kind == "eval" and not text.target_from_messages:
            annotation = row.get("annotation")
            ground_truth = row.get("ground_truth")
            if annotation != ground_truth:
                add_failure(
                    failures,
                    path,
                    line,
                    "ground_truth",
                    "eval annotation and ground_truth differ",
                )

        cxsmiles, stable = check_target_contract(
            row,
            path=path,
            line=line,
            prompt=text.prompt,
            target=text.target,
            expected_target_format=expected_target_format,
            require_canonicalized=require_canonicalized,
            allow_structured_transcript_targets=allow_structured_transcript_targets,
            failures=failures,
            canonical_jobs=canonical_jobs,
            structured_transcript_jobs=structured_transcript_jobs,
        )

        if kind == "eval":
            if row.get("cxsmiles_opt") is not None and row.get("cxsmiles_opt") != cxsmiles:
                add_failure(
                    failures,
                    path,
                    line,
                    "cxsmiles_opt",
                    "eval cxsmiles_opt does not match messages[1].content <cxsmi> payload",
                )
            if row.get("stable") is not None and row.get("stable") != stable:
                add_failure(
                    failures,
                    path,
                    line,
                    "stable",
                    "eval stable field does not match messages[1].content <stable> payload",
                )

    return check_row_images(
        row,
        path=path,
        line=line,
        check_images=check_images,
        require_absolute_images=require_absolute_images,
        require_images_list=require_images_list,
        failures=failures,
    )


def check_train_jsonl(
    path: Path,
    *,
    expected_target_format: str,
    require_canonicalized: bool,
    allow_structured_transcript_targets: bool,
    allow_ocsr_smiles_targets: bool,
    check_images: bool,
    require_absolute_images: bool,
    failures: list[CheckFailure],
    canonical_jobs: list[CanonicalPayloadJob],
    structured_transcript_jobs: list[StructuredTranscriptJob],
    ocsr_smiles_jobs: list[OCSRSmilesJob],
) -> dict[str, Any]:
    rows = load_jsonl(path)
    summary = artifact_summary(path, kind="train", rows=len(rows))
    if not rows:
        add_failure(failures, path, 0, "rows", "training JSONL is empty")

    for line_no, row in rows:
        add_summary_counts(summary, row)
        summary["missing_images"] += check_prepared_row(
            row,
            path=path,
            line=line_no,
            kind="train",
            expected_target_format=expected_target_format,
            expected_markush_prompt="",
            require_canonicalized=require_canonicalized,
            allow_structured_transcript_targets=allow_structured_transcript_targets,
            allow_ocsr_smiles_targets=allow_ocsr_smiles_targets,
            require_messages=True,
            check_images=check_images,
            require_absolute_images=require_absolute_images,
            require_images_list=True,
            failures=failures,
            canonical_jobs=canonical_jobs,
            structured_transcript_jobs=structured_transcript_jobs,
            ocsr_smiles_jobs=ocsr_smiles_jobs,
        )

    return summary


def check_eval_jsonl(
    path: Path,
    *,
    expected_target_format: str,
    require_canonicalized: bool,
    allow_ocsr_smiles_targets: bool,
    check_images: bool,
    require_absolute_images: bool,
    failures: list[CheckFailure],
    canonical_jobs: list[CanonicalPayloadJob],
    ocsr_smiles_jobs: list[OCSRSmilesJob],
) -> dict[str, Any]:
    rows = load_jsonl(path)
    is_ocsr_eval = path.stem.startswith("eval_ocsr_")
    try:
        if is_ocsr_eval:
            expected_benchmark = ocsr_eval_benchmark_from_path(path)
        else:
            expected_benchmark = eval_benchmark_from_path(path)
    except ValueError as exc:
        expected_benchmark = ""
        add_failure(failures, path, 0, "path", str(exc))
    summary = artifact_summary(
        path,
        kind="eval",
        rows=len(rows),
        expected_benchmark=expected_benchmark,
    )
    if not rows:
        message = (
            "OCSR eval JSONL is empty; materialization produced zero OCSR samples"
            if is_ocsr_eval
            else "eval JSONL is empty"
        )
        add_failure(failures, path, 0, "rows", message)

    for line_no, row in rows:
        add_summary_counts(summary, row)
        if expected_benchmark and row.get("benchmark") != expected_benchmark:
            add_failure(
                failures,
                path,
                line_no,
                "benchmark",
                f"expected benchmark {expected_benchmark!r} from filename, found {row.get('benchmark')!r}",
            )

        row_task = declared_task(row)
        is_ocsr_as_markush = declares_ocsr_as_markush_no_extension(row)
        if is_ocsr_eval and row_task != "ocsr" and not is_ocsr_as_markush:
            add_failure(
                failures,
                path,
                line_no,
                "task",
                "eval_ocsr_*.jsonl rows must declare the OCSR SMILES task contract "
                "or the explicit no-extension Markush OCSR mode",
            )
        if not is_ocsr_eval and row_task == "ocsr":
            add_failure(
                failures,
                path,
                line_no,
                "task",
                "OCSR eval rows must be staged in eval_ocsr_*.jsonl artifacts",
            )

        require_messages = row_task == "ocsr" or isinstance(row.get("messages"), list)
        require_images_list = (
            row_task == "ocsr" or require_messages or isinstance(row.get("images"), list)
        )
        expected_markush_prompt = ""
        if not is_ocsr_eval and row_task == "markush" and expected_benchmark:
            expected_markush_prompt = expected_eval_prompt(
                expected_benchmark,
                expected_target_format_for_row(
                    row.get("target_format"),
                    expected_target_format=expected_target_format,
                    require_canonicalized=require_canonicalized,
                ),
            )
        summary["missing_images"] += check_prepared_row(
            row,
            path=path,
            line=line_no,
            kind="eval",
            expected_target_format=expected_target_format,
            expected_markush_prompt=expected_markush_prompt if row_task == "markush" else "",
            require_canonicalized=require_canonicalized,
            allow_structured_transcript_targets=False,
            allow_ocsr_smiles_targets=allow_ocsr_smiles_targets,
            require_messages=require_messages,
            check_images=check_images,
            require_absolute_images=require_absolute_images,
            require_images_list=require_images_list,
            failures=failures,
            canonical_jobs=canonical_jobs,
            structured_transcript_jobs=[],
            ocsr_smiles_jobs=ocsr_smiles_jobs,
        )

    if (
        is_ocsr_eval
        and rows
        and summary["task_counts"].get("ocsr", 0) == 0
        and summary["mode_counts"].get(OCSR_AS_MARKUSH_NO_EXTENSION_MODE, 0) == 0
    ):
        add_failure(
            failures,
            path,
            0,
            "task",
            "OCSR eval JSONL loaded zero OCSR samples",
        )

    return summary


def counter_to_dict(value: Any) -> Any:
    if isinstance(value, Counter):
        return {
            str(key): count for key, count in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, dict):
        return {key: counter_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [counter_to_dict(item) for item in value]
    return value


def eval_jsonls_from_dir(eval_dir: Path) -> list[Path]:
    return sorted(path for path in eval_dir.glob("eval_*.jsonl") if path.is_file())


def check_artifacts(
    *,
    train_jsonls: list[Path],
    eval_dirs: list[Path],
    expected_target_format: str = DEFAULT_TARGET_FORMAT,
    require_canonicalized: bool = True,
    allow_structured_transcript_targets: bool = False,
    allow_ocsr_smiles_targets: bool = True,
    check_images: bool = False,
    require_absolute_images: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    failures: list[CheckFailure] = []
    canonical_jobs: list[CanonicalPayloadJob] = []
    structured_transcript_jobs: list[StructuredTranscriptJob] = []
    ocsr_smiles_jobs: list[OCSRSmilesJob] = []
    files: list[dict[str, Any]] = []

    for train_jsonl in train_jsonls:
        if not train_jsonl.exists():
            add_failure(failures, train_jsonl, 0, "path", "training JSONL does not exist")
            continue
        files.append(
            check_train_jsonl(
                train_jsonl,
                expected_target_format=expected_target_format,
                require_canonicalized=require_canonicalized,
                allow_structured_transcript_targets=allow_structured_transcript_targets,
                allow_ocsr_smiles_targets=allow_ocsr_smiles_targets,
                check_images=check_images,
                require_absolute_images=require_absolute_images,
                failures=failures,
                canonical_jobs=canonical_jobs,
                structured_transcript_jobs=structured_transcript_jobs,
                ocsr_smiles_jobs=ocsr_smiles_jobs,
            )
        )

    for eval_dir in eval_dirs:
        if not eval_dir.exists():
            add_failure(failures, eval_dir, 0, "path", "eval directory does not exist")
            continue
        eval_paths = eval_jsonls_from_dir(eval_dir)
        if not eval_paths:
            add_failure(failures, eval_dir, 0, "path", "eval directory has no eval_*.jsonl files")
        for eval_jsonl in eval_paths:
            files.append(
                check_eval_jsonl(
                    eval_jsonl,
                    expected_target_format=expected_target_format,
                    require_canonicalized=require_canonicalized,
                    allow_ocsr_smiles_targets=allow_ocsr_smiles_targets,
                    check_images=check_images,
                    require_absolute_images=require_absolute_images,
                    failures=failures,
                    canonical_jobs=canonical_jobs,
                    ocsr_smiles_jobs=ocsr_smiles_jobs,
                )
            )

    failures.extend(run_canonical_payload_checks(canonical_jobs, workers=workers))
    failures.extend(run_structured_transcript_checks(structured_transcript_jobs, workers=workers))
    failures.extend(run_ocsr_smiles_checks(ocsr_smiles_jobs, workers=workers))
    total_rows = sum(int(file_summary["rows"]) for file_summary in files)
    aggregate_counts = {
        "task_counts": Counter(),
        "mode_counts": Counter(),
        "target_format_counts": Counter(),
        "target_source_counts": Counter(),
        "source_counts": Counter(),
        "benchmark_counts": Counter(),
    }
    missing_images = 0
    for file_summary in files:
        for key, counter in aggregate_counts.items():
            counter.update(file_summary.get(key, Counter()))
        missing_images += int(file_summary.get("missing_images") or 0)
    report = {
        "ok": not failures,
        "expected_target_format": expected_target_format,
        "require_canonicalized": require_canonicalized,
        "allow_structured_transcript_targets": allow_structured_transcript_targets,
        "allow_ocsr_smiles_targets": allow_ocsr_smiles_targets,
        "check_images": check_images,
        "require_absolute_images": require_absolute_images,
        "workers": workers,
        "total_rows": total_rows,
        "canonical_payload_checks": len(canonical_jobs),
        "structured_transcript_checks": len(structured_transcript_jobs),
        "ocsr_smiles_checks": len(ocsr_smiles_jobs),
        "task_counts": counter_to_dict(aggregate_counts["task_counts"]),
        "mode_counts": counter_to_dict(aggregate_counts["mode_counts"]),
        "target_format_counts": counter_to_dict(aggregate_counts["target_format_counts"]),
        "target_source_counts": counter_to_dict(aggregate_counts["target_source_counts"]),
        "source_counts": counter_to_dict(aggregate_counts["source_counts"]),
        "benchmark_counts": counter_to_dict(aggregate_counts["benchmark_counts"]),
        "missing_images": missing_images,
        "files": [counter_to_dict(file_summary) for file_summary in files],
        "failure_count": len(failures),
        "failures": [asdict(failure) for failure in failures[:100]],
    }
    if len(failures) > 100:
        report["truncated_failures"] = len(failures) - 100
    return report
