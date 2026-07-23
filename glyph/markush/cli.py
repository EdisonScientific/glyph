"""Top-level Markush command line interface."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from glyph.markush.data.cli import app as data_app
from glyph.markush.eval.checkpoint import main as checkpoint_main
from glyph.markush.eval.cli import app as eval_app
from glyph.markush.grading import Protocol, grade_files
from glyph.markush.inference import (
    DEFAULT_MARKUSH_BASE_MODEL,
    DEFAULT_MARKUSH_PROMPT,
    DEFAULT_VOTE_K,
    MarkushPrediction,
    MarkushPredictor,
)
from glyph.markush.targets import SourceFormat, normalize_markush_target
from glyph.markush.train.cli import app as train_app

app = typer.Typer(no_args_is_help=True, help="Markush structure prediction tools.")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")


class OutputFormat(StrEnum):
    """CXSMILES representation selected for ``--text`` output."""

    cxsmiles = "cxsmiles"
    cxsmiles_opt = "cxsmiles_opt"


def _format_text(result: MarkushPrediction, output_format: OutputFormat) -> str:
    """Return the CXSMILES text for the requested ``--format``."""

    if output_format is OutputFormat.cxsmiles_opt:
        return result.cxsmiles_opt
    return result.cxsmiles


def _forward_to_entrypoint(command: str, args: list[str], main_func: Callable[[], None]) -> None:
    original_argv = sys.argv
    sys.argv = [command, *args]
    try:
        main_func()
    finally:
        sys.argv = original_argv


def _download_released_adapter() -> str:
    """Download the released Markush adapter and return its local directory."""

    from glyph.markush.inference import _resolve_adapter

    return _resolve_adapter(None)


@app.command("predict")
def predict(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    adapter: Annotated[
        str | None,
        typer.Option(
            "--adapter",
            "-a",
            "--checkpoint",
            help=(
                "LoRA adapter path or Hugging Face repo id. "
                "Omit to download EdisonScientific/MarkushGlyph."
            ),
        ),
    ] = None,
    base_model: Annotated[
        str | None,
        typer.Option(
            "--base-model",
            help="Base model path or Hugging Face repo id. Omit to download Qwen/Qwen3.5-2B-Base.",
        ),
    ] = None,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    dtype: Annotated[str, typer.Option("--dtype")] = "bfloat16",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 1024,
    merge_adapter: Annotated[bool, typer.Option("--merge-adapter/--no-merge-adapter")] = True,
    prompt: Annotated[str, typer.Option("--prompt")] = DEFAULT_MARKUSH_PROMPT,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help=(
                "CXSMILES form for --text output: 'cxsmiles' (standard |$...$|, "
                "default) or 'cxsmiles_opt' (the <r>LABEL</r> model form). "
                "--json always carries both."
            ),
        ),
    ] = OutputFormat.cxsmiles,
    majority_vote: Annotated[
        bool,
        typer.Option(
            "--majority-vote/--no-majority-vote",
            help=(
                "Sample --vote-k candidates and return the modal consensus "
                "(reproduces the paper MV@8). Off by default (greedy decode)."
            ),
        ),
    ] = False,
    vote_k: Annotated[
        int,
        typer.Option("--vote-k", help="Number of sampled candidates for --majority-vote."),
    ] = DEFAULT_VOTE_K,
    json_output: Annotated[bool, typer.Option("--json/--text")] = False,
) -> None:
    """Predict Markush structure for one image."""

    resolved_adapter = adapter or _download_released_adapter()
    resolved_base_model = (
        base_model or os.environ.get("MARKUSH_BASE_MODEL") or DEFAULT_MARKUSH_BASE_MODEL
    )
    predictor = MarkushPredictor(
        checkpoint=resolved_adapter,
        base_model=resolved_base_model,
        device=device,
        dtype=dtype,
        merge_adapter=merge_adapter,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
    )
    try:
        result = predictor.predict(image, majority_vote=majority_vote, vote_k=vote_k)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(result.asdict(), indent=2, sort_keys=True))
    else:
        typer.echo(_format_text(result, output_format))


@app.command("predict-batch")
def predict_batch(
    images: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    checkpoint: Annotated[
        str | None,
        typer.Option("--checkpoint", help="LoRA adapter path or Hugging Face repo id."),
    ] = None,
    base_model: Annotated[str | None, typer.Option("--base-model")] = None,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    dtype: Annotated[str, typer.Option("--dtype")] = "bfloat16",
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens")] = 1024,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Images per greedy model.generate call; ignored with --majority-vote.",
        ),
    ] = 8,
    merge_adapter: Annotated[bool, typer.Option("--merge-adapter/--no-merge-adapter")] = True,
    prompt: Annotated[str, typer.Option("--prompt")] = DEFAULT_MARKUSH_PROMPT,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help=(
                "Accepted for parity with `predict`. Batch JSONL rows always "
                "carry both `cxsmiles` (standard) and `cxsmiles_opt`."
            ),
        ),
    ] = OutputFormat.cxsmiles,
    majority_vote: Annotated[
        bool,
        typer.Option(
            "--majority-vote/--no-majority-vote",
            help=(
                "Sample --vote-k candidates per image and return the modal "
                "consensus (reproduces the paper MV@8). Off by default."
            ),
        ),
    ] = False,
    vote_k: Annotated[
        int,
        typer.Option("--vote-k", help="Number of sampled candidates for --majority-vote."),
    ] = DEFAULT_VOTE_K,
) -> None:
    """Predict Markush structure for multiple images and emit JSONL."""

    if not images:
        typer.echo("Error: at least one image is required", err=True)
        raise typer.Exit(code=2)

    resolved_base_model = base_model or os.environ.get("MARKUSH_BASE_MODEL")
    predictor = MarkushPredictor(
        checkpoint=checkpoint,
        base_model=resolved_base_model,
        device=device,
        dtype=dtype,
        merge_adapter=merge_adapter,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
    )

    lines: list[str] = []
    try:
        if majority_vote:
            results = [
                predictor.predict(image, majority_vote=True, vote_k=vote_k) for image in images
            ]
        else:
            results = predictor.predict_many(
                list(images),
                batch_size=batch_size,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        lines.extend(json.dumps(result.asdict(), sort_keys=True) for result in results)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = "\n".join(lines) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Saved predictions to {output}")
    else:
        typer.echo(payload, nl=False)


@app.command("normalize-target")
def normalize_target_command(
    target: Annotated[str, typer.Argument()],
    source_format: Annotated[
        SourceFormat,
        typer.Option("--source-format", help="Input target format."),
    ] = "auto",
    canonicalize: Annotated[
        bool,
        typer.Option("--canonicalize/--no-canonicalize"),
    ] = True,
    require_canonical: Annotated[
        bool,
        typer.Option("--require-canonical/--allow-noncanonical"),
    ] = False,
    check_roundtrip: Annotated[
        bool,
        typer.Option("--check-roundtrip/--no-check-roundtrip"),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json/--text")] = False,
) -> None:
    """Normalize one Markush XML/CXSMILES target to the paper target contract."""

    try:
        result = normalize_markush_target(
            target,
            source_format=source_format,
            canonicalize=canonicalize,
            require_canonical=require_canonical,
            check_roundtrip=check_roundtrip,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(result.asdict(), indent=2, sort_keys=True, default=str))
    else:
        typer.echo(result.markush_xml or result.error)
    if not result.valid:
        raise typer.Exit(code=1)


@app.command("grade")
def grade_command(
    ground_truth: Annotated[Path, typer.Option("--ground-truth", exists=True, dir_okay=False)],
    predictions: Annotated[Path, typer.Option("--predictions", exists=True, dir_okay=False)],
    protocol: Annotated[Protocol, typer.Option("--protocol")] = "ours",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    include_missing: Annotated[
        bool,
        typer.Option("--include-missing/--no-include-missing"),
    ] = True,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Grade prediction JSONL against ground-truth JSONL."""

    try:
        payload = grade_files(
            predictions,
            ground_truth,
            protocol=protocol,
            include_missing=include_missing,
            verbose=verbose,
            output_path=output,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output is not None:
        typer.echo(f"Saved metrics to {output}")
    else:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@app.command(
    "eval-checkpoint",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def eval_checkpoint(ctx: typer.Context) -> None:
    """Run checkpoint evaluation from the top-level CLI."""

    _forward_to_entrypoint("markush eval-checkpoint", list(ctx.args), checkpoint_main)


def main() -> None:
    """Run the Markush CLI."""

    app()


if __name__ == "__main__":
    main()
