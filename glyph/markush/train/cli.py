"""Typer entry point for supervised Markush fine-tuning."""

from __future__ import annotations

import sys

import typer

app = typer.Typer(
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run supervised fine-tuning with the existing training options.",
)


@app.callback(invoke_without_command=True)
def train(ctx: typer.Context) -> None:
    """Forward options to the existing Markush SFT entry point."""

    from glyph.markush.train.sft import build_argument_parser
    from glyph.markush.train.sft import main as sft_main

    parser = build_argument_parser()
    original_argv = sys.argv
    sys.argv = ["glyph markush train", *ctx.args]
    try:
        parser.parse_args(list(ctx.args))
        sft_main()
    finally:
        sys.argv = original_argv


def main() -> None:
    """Run the Markush training CLI."""

    app()


if __name__ == "__main__":
    main()
