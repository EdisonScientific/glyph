"""CLI regression tests for OCSR postprocessing and Typer wiring."""

from typer.testing import CliRunner

from glyph.ocsr import cli
from glyph.ocsr.full_eval import build_parser
from glyph.ocsr.postprocess import postprocess_smiles


def test_full_uspto_eval_defaults_postprocess_on():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "dummy.pth"])
    assert args.postprocess == "light"


def test_full_uspto_eval_allows_disabling_postprocess():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "dummy.pth", "--postprocess", "none"])
    assert args.postprocess == "none"


def test_eval_cli_wrapper_defaults_postprocess_on(monkeypatch):
    captured: dict[str, object] = {}

    def fake_eval(**kwargs: object):
        captured.update(kwargs)
        return ({"counts": {}}, [], [])

    monkeypatch.setattr("glyph.ocsr.full_eval.run_full_uspto_eval", fake_eval)
    monkeypatch.setattr("glyph.ocsr.full_eval.format_summary", lambda _report: "summary")

    result = CliRunner().invoke(cli.app, ["eval", "--checkpoint", "dummy.pth"])

    assert result.exit_code == 0, result.output
    assert captured["postprocess"] == "light"


def test_eval_cli_wrapper_forwards_postprocess_flag(monkeypatch):
    captured: dict[str, object] = {}

    def fake_eval(**kwargs: object):
        captured.update(kwargs)
        return ({"counts": {}}, [], [])

    monkeypatch.setattr("glyph.ocsr.full_eval.run_full_uspto_eval", fake_eval)
    monkeypatch.setattr("glyph.ocsr.full_eval.format_summary", lambda _report: "summary")

    result = CliRunner().invoke(
        cli.app,
        ["eval", "--checkpoint", "dummy.pth", "--postprocess", "none"],
    )

    assert result.exit_code == 0, result.output
    assert captured["postprocess"] == "none"


def test_default_postprocess_cleans_hh_suffix():
    """The default postprocess path strips hallucinated [HH] fragments."""
    assert postprocess_smiles("CCO.[HH]") == "CCO"
