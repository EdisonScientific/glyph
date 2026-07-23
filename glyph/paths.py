"""Portable filesystem locations shared by all Glyph models."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GLYPH_DATA_ROOT = "GLYPH_DATA_ROOT"
DATA_ROOT_ENV = GLYPH_DATA_ROOT
DEFAULT_DATA_ROOT = Path("./data")
STANDARD_SUBDIRS = (
    "models",
    "datasets",
    "prepared",
    "checkpoints",
    "evals",
    "eval_out",
)


def data_root(value: str | Path | None = None, *, env: Mapping[str, str] | None = None) -> Path:
    """Return the configured Glyph data root without changing the process CWD."""

    environ = os.environ if env is None else env
    configured = value if value is not None else environ.get(GLYPH_DATA_ROOT, DEFAULT_DATA_ROOT)
    path = Path(os.path.expandvars(str(configured))).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def data_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    """Return a path below the configured data root."""

    return data_root(root).joinpath(*parts)


def models_dir(*parts: str | Path, root: str | Path | None = None) -> Path:
    return data_path("models", *parts, root=root)


def datasets_dir(*parts: str | Path, root: str | Path | None = None) -> Path:
    return data_path("datasets", *parts, root=root)


def prepared_dir(*parts: str | Path, root: str | Path | None = None) -> Path:
    return data_path("prepared", *parts, root=root)


def checkpoints_dir(*parts: str | Path, root: str | Path | None = None) -> Path:
    return data_path("checkpoints", *parts, root=root)


def evals_dir(*parts: str | Path, root: str | Path | None = None) -> Path:
    return data_path("evals", *parts, root=root)


def eval_out_dir(*parts: str | Path, root: str | Path | None = None) -> Path:
    return data_path("eval_out", *parts, root=root)


def expand_config_vars(value: Any, *, root: str | Path | None = None) -> Any:
    """Recursively expand environment variables, including the data-root default."""

    if isinstance(value, str):
        configured_root = str(data_root(root))
        value = value.replace(f"${{{GLYPH_DATA_ROOT}}}", configured_root)
        value = value.replace(f"${GLYPH_DATA_ROOT}", configured_root)
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_config_vars(item, root=root) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config_vars(item, root=root) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_config_vars(item, root=root) for item in value)
    return value
