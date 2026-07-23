"""Unified Typer command-line entry point for Glyph model families."""

from __future__ import annotations

import typer

from glyph.data.cli import app as data_app
from glyph.markush.cli import app as markush_app
from glyph.ocsr.cli import app as ocsr_app

app = typer.Typer(
    no_args_is_help=True,
    help="Molecular and Markush structure recognition tools.",
)
app.add_typer(ocsr_app, name="ocsr")
app.add_typer(markush_app, name="markush")
app.add_typer(data_app, name="data")


def main() -> None:
    """Run the unified Glyph CLI."""

    app()


if __name__ == "__main__":
    main()
