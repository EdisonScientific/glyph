from __future__ import annotations

from pathlib import Path

from glyph.paths import (
    DATA_ROOT_ENV,
    DEFAULT_DATA_ROOT,
    GLYPH_DATA_ROOT,
    STANDARD_SUBDIRS,
    checkpoints_dir,
    data_path,
    data_root,
    datasets_dir,
    eval_out_dir,
    evals_dir,
    expand_config_vars,
    models_dir,
    prepared_dir,
)


def test_data_root_uses_cli_then_env_then_cwd_default(tmp_path: Path, monkeypatch) -> None:
    assert GLYPH_DATA_ROOT == DATA_ROOT_ENV == "GLYPH_DATA_ROOT"
    monkeypatch.chdir(tmp_path)
    assert data_root("/cli/root", env={DATA_ROOT_ENV: "/env/root"}) == Path("/cli/root")
    assert data_root(env={DATA_ROOT_ENV: "/env/root"}) == Path("/env/root")
    assert data_root(env={}) == tmp_path / DEFAULT_DATA_ROOT


def test_standard_subpath_helpers_share_one_root() -> None:
    root = "/tmp/glyph"
    assert STANDARD_SUBDIRS == (
        "models",
        "datasets",
        "prepared",
        "checkpoints",
        "evals",
        "eval_out",
    )
    assert models_dir("model", root=root) == Path(root, "models", "model")
    assert datasets_dir("source", root=root) == Path(root, "datasets", "source")
    assert prepared_dir("train.jsonl", root=root) == Path(root, "prepared", "train.jsonl")
    assert checkpoints_dir("run", root=root) == Path(root, "checkpoints", "run")
    assert evals_dir("suite", root=root) == Path(root, "evals", "suite")
    assert eval_out_dir("results.json", root=root) == Path(root, "eval_out", "results.json")
    assert data_path("other", root=root) == Path(root, "other")


def test_expand_config_vars_recurses_and_supplies_data_root_default() -> None:
    value = {
        "model": "${GLYPH_DATA_ROOT}/models/base",
        "nested": ["$GLYPH_DATA_ROOT/prepared/train.jsonl", "${HOME}/cache"],
    }

    expanded = expand_config_vars(value, root="/tmp/glyph")

    assert expanded["model"] == "/tmp/glyph/models/base"
    assert expanded["nested"][0] == "/tmp/glyph/prepared/train.jsonl"
    assert "${HOME}" not in expanded["nested"][1]
