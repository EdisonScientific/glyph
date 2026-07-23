"""CLI tests for the ``glyph data prepare`` sub-app."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from glyph.cli import app

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Drop ANSI color codes so assertions ignore rich's decoration.

    Under CI (Typer/rich honour ``FORCE_COLOR``) error output is colorized,
    which splits option tokens like ``--ocsr`` with escape sequences. Matching
    on the decorated form is environment-dependent and brittle, so we assert on
    the plain text instead.
    """
    return _ANSI_SGR.sub("", text)


def test_top_level_cli_exposes_data_subapp() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "data" in _strip_ansi(result.output)


def test_data_prepare_requires_a_family() -> None:
    result = CliRunner().invoke(app, ["data", "prepare"])
    assert result.exit_code != 0
    assert "at least one of --ocsr" in _strip_ansi(result.output)


def test_data_prepare_ocsr_dry_run(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["data", "prepare", "--ocsr", "--dry-run", "--data-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "planned" in result.output
    assert "pubchem_1m" in result.output
    assert "uspto_ocsr_5719" in result.output
    assert not (tmp_path / "datasets").exists()


def test_data_prepare_markush_dry_run(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["data", "prepare", "--markush", "--dry-run", "--data-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "planned" in result.output
    assert "mg1_synthetic" in result.output
    assert not (tmp_path / "datasets").exists()


def test_data_prepare_markush_refuses_large_by_default(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["data", "prepare", "--markush", "--data-root", str(tmp_path)])
    # Refused (large-download guard) is a clean exit, not an error.
    assert result.exit_code == 0, result.output
    assert "refused" in result.output
    assert "--allow-large" in result.output


def test_data_prepare_ocsr_json_output(tmp_path: Path) -> None:
    import json

    result = CliRunner().invoke(
        app, ["data", "prepare", "--ocsr", "--dry-run", "--json", "--data-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["family"] == "ocsr"
    assert {step["name"] for step in payload["steps"]} >= {"pubchem_1m", "uspto_ocsr_5719"}
