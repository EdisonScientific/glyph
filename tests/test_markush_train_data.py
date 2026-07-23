"""Tests for the frozen MarkushGlyph flat-index resolver (network-free tiers)."""

from __future__ import annotations

from pathlib import Path

from glyph.markush.train.data import (
    MARKUSH_TRAIN_JSONL_PATH,
    resolve_markush_train_jsonl,
)


def test_resolve_prefers_explicit_override(tmp_path: Path):
    explicit = tmp_path / "custom_train.jsonl"
    explicit.write_text("{}\n")
    assert resolve_markush_train_jsonl(explicit=explicit) == explicit


def test_resolve_prefers_mirror_env(tmp_path: Path, monkeypatch):
    mirror = tmp_path / "mirror"
    target = mirror / MARKUSH_TRAIN_JSONL_PATH
    target.parent.mkdir(parents=True)
    target.write_text("{}\n")
    monkeypatch.setenv("GLYPH_MARKUSH_TRAIN_ROOT", str(mirror))
    assert resolve_markush_train_jsonl() == target


def test_resolve_uses_glyph_data_root_snapshot(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GLYPH_MARKUSH_TRAIN_ROOT", raising=False)
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    snapshot = tmp_path / "datasets" / MARKUSH_TRAIN_JSONL_PATH
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}\n")
    assert resolve_markush_train_jsonl() == snapshot


def test_bare_hub_falls_back_to_default_repo_path(tmp_path: Path, monkeypatch):
    mirror = tmp_path / "mirror"
    target = mirror / MARKUSH_TRAIN_JSONL_PATH
    target.parent.mkdir(parents=True)
    target.write_text("{}\n")
    monkeypatch.setenv("GLYPH_MARKUSH_TRAIN_ROOT", str(mirror))
    # An empty repo_path (bare ``hub:``) resolves the default flat index.
    assert resolve_markush_train_jsonl("") == target
    assert resolve_markush_train_jsonl(None) == target
