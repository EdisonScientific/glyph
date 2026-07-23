"""Typer interface for OCSR inference, evaluation, and quantization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

app = typer.Typer(no_args_is_help=True, help="Optical chemical structure recognition tools.")

Device = Literal["cuda", "cpu"]
Precision = Literal["fp32", "fp16", "bf16", "auto"]
QuantizationMode = Literal["full", "decoder"]
PostprocessMode = Literal["light", "none"]


@app.command("predict")
def predict(
    images: Annotated[
        list[Path],
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Molecule image file(s)."),
    ],
    checkpoint: Annotated[
        str | None,
        typer.Option(
            "--checkpoint",
            "-c",
            help=(
                "Path to a .pth checkpoint or Hugging Face repo id. "
                "Omit to download EdisonScientific/OCSRGlyph. "
                "Security: deserialized with torch (weights_only=False) and may "
                "execute code from a malicious checkpoint — only use trusted sources."
            ),
        ),
    ] = None,
    device: Annotated[Device | None, typer.Option("--device")] = None,
    quantize: Annotated[
        QuantizationMode | None,
        typer.Option("--quantize", "-q", help="INT8 quantization mode."),
    ] = None,
    precision: Annotated[
        Precision,
        typer.Option("--precision", help="Model precision; auto selects for the device."),
    ] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    postprocess: Annotated[
        bool,
        typer.Option(
            "--postprocess/--no-postprocess",
            help="Apply the deterministic reproduced-evaluation SMILES postprocessor.",
        ),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json", help="Output JSON lines.")] = False,
) -> None:
    """Predict SMILES from one or more molecule images."""

    from .predict import OCSRPredictor

    predictor = OCSRPredictor(
        checkpoint,
        device=device,
        quantize=quantize,
        precision=None if precision == "auto" else precision,
    )
    results = predictor.predict_batch(
        images,
        batch_size=batch_size,
        postprocess=postprocess,
    )
    for path, smiles in zip(images, results, strict=True):
        if json_output:
            typer.echo(json.dumps({"image": str(path), "smiles": smiles}))
        elif len(results) == 1:
            typer.echo(smiles)
        else:
            typer.echo(f"{path}\t{smiles}")


@app.command("eval")
def evaluate(
    checkpoint: Annotated[
        str | None,
        typer.Option(
            "--checkpoint",
            "-c",
            help=(
                "Path to a .pth checkpoint or Hugging Face repo id. "
                "Omit to download EdisonScientific/OCSRGlyph. "
                "Security: deserialized with torch (weights_only=False) and may "
                "execute code from a malicious checkpoint — only use trusted sources."
            ),
        ),
    ] = None,
    benchmark: Annotated[
        Path | None,
        typer.Option(
            "--benchmark",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional complete-label CSV override (image_id, smiles, source_of_label). "
            "Default: score directly from the self-contained benchmark parquet.",
        ),
    ] = None,
    benchmark_parquet: Annotated[
        Path | None,
        typer.Option(
            "--benchmark-parquet",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional benchmark parquet override (images + labels). "
            "Default: the hosted self-contained parquet (auto-downloaded).",
        ),
    ] = None,
    device: Annotated[Device | None, typer.Option("--device")] = None,
    precision: Annotated[Precision, typer.Option("--precision")] = "fp32",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    postprocess: Annotated[PostprocessMode, typer.Option("--postprocess")] = "light",
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    majority_vote: Annotated[
        bool,
        typer.Option(
            "--majority-vote/--no-majority-vote",
            help="Also run opt-in self-consistency majority-vote decoding (MV@K). "
            "Never changes the greedy headline numbers.",
        ),
    ] = False,
    vote_k: Annotated[
        str,
        typer.Option(
            "--vote-k",
            help="Comma-separated K value(s) for --majority-vote, e.g. '8' or '4,8,16'.",
        ),
    ] = "8",
    temperature: Annotated[float, typer.Option("--temperature", min=0.0)] = 0.7,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 0.95,
    sample_seed: Annotated[int, typer.Option("--sample-seed")] = 20260625,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    per_row_out: Annotated[Path | None, typer.Option("--per-row-out", dir_okay=False)] = None,
) -> None:
    """Run the full USPTO benchmark evaluation."""

    from .full_eval import (
        format_summary,
        parse_vote_ks,
        run_full_uspto_eval,
        write_eval_report,
        write_per_row_predictions,
    )

    vote_ks = parse_vote_ks(vote_k) if majority_vote else None

    report, label_rows, predictions = run_full_uspto_eval(
        checkpoint=checkpoint,
        benchmark=benchmark,
        benchmark_parquet=benchmark_parquet,
        device=device,
        precision=precision,
        batch_size=batch_size,
        postprocess=postprocess,
        limit=limit,
        vote_k=vote_ks,
        temperature=temperature,
        top_p=top_p,
        sample_seed=sample_seed,
    )
    typer.echo(format_summary(report))

    if output is not None:
        output_path = write_eval_report(report, output)
        typer.echo(f"[out] wrote {output_path}")
    if per_row_out is not None:
        per_row_path = write_per_row_predictions(label_rows, predictions, per_row_out)
        typer.echo(f"[out] wrote {per_row_path}")


@app.command("quantize")
def quantize(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            dir_okay=False,
            readable=True,
            help=(
                "Checkpoint to quantize. Security: deserialized with torch "
                "(weights_only=False) and may execute code from a malicious "
                "checkpoint — only use trusted sources."
            ),
        ),
    ],
    output_path: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    mode: Annotated[QuantizationMode, typer.Option("--mode", "-m")] = "full",
) -> None:
    """INT8-quantize an OCSR checkpoint."""

    from .predict import OCSRPredictor

    stats = OCSRPredictor.quantize_checkpoint(
        str(input_path),
        str(output_path),
        mode=mode,
    )
    typer.echo(json.dumps(stats, indent=2))


def main() -> None:
    """Run the OCSR CLI."""

    app()


if __name__ == "__main__":
    main()
