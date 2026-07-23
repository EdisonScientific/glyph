"""Tests for the self-contained MarkushGlyph benchmark resolver + materializer.

These use only synthetic parquets built in ``tmp_path``; no network access and
no Hugging Face download are exercised (the explicit/env/snapshot tiers all
return before ``hf_hub_download``).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from glyph.markush.eval.benchmark_data import (
    MARKUSH_BENCHMARK_NAMES,
    load_benchmark_records,
    materialize_benchmark,
    normalize_benchmark_name,
    resolve_benchmark_gt,
    resolve_benchmark_parquet,
)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_synthetic_parquet(path: Path, benchmark: str, n: int = 2) -> None:
    ids, images, benchmarks, records = [], [], [], []
    for i in range(n):
        rel = f"images/{benchmark}/test/{i}.png"
        record = {
            "id": str(i),
            "image": rel,
            "images": [rel],
            "benchmark": benchmark,
            "prompt": "Extract the Markush structure from this image.",
            "messages": [
                {"role": "user", "content": f"<image>\nExtract row {i}."},
                {"role": "assistant", "content": "<markush><cxsmi>C</cxsmi></markush>"},
            ],
            "cxsmiles_opt": "C",
            "annotation": "<markush><cxsmi>C</cxsmi><stable></stable></markush>",
        }
        ids.append(str(i))
        images.append({"bytes": _png_bytes((i * 10, 0, 0)), "path": rel})
        benchmarks.append(benchmark)
        records.append(json.dumps(record, ensure_ascii=False))
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "id": pa.array(ids, pa.string()),
            "image": pa.array(images, image_type),
            "benchmark": pa.array(benchmarks, pa.string()),
            "record": pa.array(records, pa.string()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_normalize_benchmark_name_aliases():
    assert normalize_benchmark_name("IP5-M") == "ip5m"
    assert normalize_benchmark_name("ip5") == "ip5m"
    assert normalize_benchmark_name("uspto_markush") == "uspto_markush"
    assert normalize_benchmark_name("uspto-markush") == "uspto_markush"
    assert normalize_benchmark_name("uspto") == "uspto_markush"
    assert normalize_benchmark_name("m2s") == "m2s"
    assert set(MARKUSH_BENCHMARK_NAMES) == {"ip5m", "m2s", "uspto_markush"}


def test_normalize_benchmark_name_unknown_raises():
    with pytest.raises(ValueError):
        normalize_benchmark_name("not-a-benchmark")


def test_resolve_parquet_prefers_env_mirror(tmp_path: Path, monkeypatch):
    mirror = tmp_path / "mirror"
    target = mirror / "markush-eval" / "ip5m" / "ip5m_benchmark.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"parquet")
    monkeypatch.setenv("GLYPH_MARKUSH_EVAL_ROOT", str(mirror))
    assert resolve_benchmark_parquet("ip5m") == target


def test_resolve_parquet_uses_glyph_data_root_snapshot(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GLYPH_MARKUSH_EVAL_ROOT", raising=False)
    monkeypatch.setenv("GLYPH_DATA_ROOT", str(tmp_path))
    snapshot = tmp_path / "datasets" / "markush-eval" / "m2s" / "m2s_benchmark.parquet"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"parquet")
    assert resolve_benchmark_parquet("m2s") == snapshot


def test_resolve_gt_paths(tmp_path: Path, monkeypatch):
    mirror = tmp_path / "mirror"
    monkeypatch.setenv("GLYPH_MARKUSH_EVAL_ROOT", str(mirror))
    cases = {
        "ip5m": "markush-eval/ip5m/ds4sd_mg2_predictions_878.jsonl",
        "m2s": "markush-eval/m2s/ds4sd_mg2_gt_103.jsonl",
        "uspto_markush": "markush-eval/uspto_markush/ds4sd_mg2_gt_74.jsonl",
    }
    for bench, rel in cases.items():
        target = mirror / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
        assert resolve_benchmark_gt(bench) == target


def test_resolve_explicit_override_wins(tmp_path: Path):
    explicit = tmp_path / "custom.parquet"
    explicit.write_bytes(b"parquet")
    assert resolve_benchmark_parquet("ip5m", explicit=explicit) == explicit


def test_load_benchmark_records(tmp_path: Path):
    parquet = tmp_path / "ip5m_benchmark.parquet"
    _write_synthetic_parquet(parquet, "ip5m", n=3)
    records = load_benchmark_records(parquet)
    assert len(records) == 3
    assert records[0]["id"] == "0"
    assert records[0]["image_path"] == "images/ip5m/test/0.png"
    assert isinstance(records[0]["image_bytes"], bytes)
    assert records[0]["record"]["cxsmiles_opt"] == "C"


def test_materialize_benchmark_roundtrip(tmp_path: Path):
    parquet = tmp_path / "src" / "m2s_benchmark.parquet"
    _write_synthetic_parquet(parquet, "m2s", n=2)
    dest = tmp_path / "out"
    eval_jsonl, image_root = materialize_benchmark("m2s", dest, explicit_parquet=parquet)

    assert eval_jsonl == dest / "benchmarks" / "markush" / "m2s" / "eval_m2s.jsonl"
    assert image_root == dest
    rows = [json.loads(line) for line in eval_jsonl.read_text().splitlines() if line]
    assert len(rows) == 2
    for i, row in enumerate(rows):
        assert row["image"] == f"images/m2s/test/{i}.png"
        assert (image_root / row["image"]).is_file()
        # A valid PNG was written (decodes without error).
        with Image.open(image_root / row["image"]) as image:
            image.load()


def test_materialize_benchmark_idempotent(tmp_path: Path):
    parquet = tmp_path / "src" / "uspto_markush_benchmark.parquet"
    _write_synthetic_parquet(parquet, "uspto_markush", n=2)
    dest = tmp_path / "out"
    first, _ = materialize_benchmark("uspto_markush", dest, explicit_parquet=parquet)
    mtime = first.stat().st_mtime_ns
    second, _ = materialize_benchmark("uspto_markush", dest, explicit_parquet=parquet)
    assert second == first
    # Idempotent reuse leaves the already-written JSONL untouched.
    assert second.stat().st_mtime_ns == mtime


def test_benchmark_command_builds_frozen_greedy_argv(tmp_path: Path, monkeypatch):
    """The `benchmark` CLI command bakes the frozen greedy recipe into fast_eval."""

    from typer.testing import CliRunner

    from glyph.markush.eval import benchmark_data, fast_eval
    from glyph.markush.eval import cli as eval_cli

    eval_jsonl = tmp_path / "eval.jsonl"
    eval_jsonl.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_data,
        "materialize_benchmark",
        lambda *args, **kwargs: (eval_jsonl, tmp_path),
    )

    captured: dict[str, list[str]] = {}

    def fake_main() -> None:
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(fast_eval, "main", fake_main)

    result = CliRunner().invoke(
        eval_cli.app,
        [
            "benchmark",
            "--benchmark",
            "IP5-M",
            "--checkpoint",
            "ckpt",
            "--base-model",
            "base",
            "--output",
            str(tmp_path / "preds.json"),
            "--batch-size",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    argv = captured["argv"]

    def opt(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert opt("--markush-prompt-policy") == "visible_sg"
    assert opt("--seed") == "20260625"
    assert opt("--max-new-tokens") == "1024"
    assert opt("--multi-image-mode") == "single"
    assert "--skip-mcs" in argv
    assert opt("--eval-data") == str(eval_jsonl)
    assert opt("--image-root") == str(tmp_path)
    assert opt("--checkpoint") == "ckpt"
    assert opt("--base-model") == "base"
    assert opt("--batch-size") == "4"
    # Greedy: no stochastic-sampling flags leak in.
    assert "--do-sample" not in argv
    assert "--majority-vote" not in argv


def test_benchmark_command_requires_a_name():
    from typer.testing import CliRunner

    from glyph.markush.eval import cli as eval_cli

    result = CliRunner().invoke(eval_cli.app, ["benchmark"])
    assert result.exit_code == 2
