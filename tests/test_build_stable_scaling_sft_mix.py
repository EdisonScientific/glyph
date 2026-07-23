from __future__ import annotations

import json
from pathlib import Path

import pytest

from glyph.data.markush_mix import build_mix, parse_partition


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    row_id: str,
    *,
    source: str = "public",
    source_id: str | None = None,
    mode: str = "structure_only_markush",
    image: str = "/tmp/image.png",
) -> dict:
    return {
        "id": row_id,
        "source": source,
        "source_id": source_id or row_id,
        "mode": mode,
        "effective_mode": mode,
        "target_source": "cxsmiles_opt->canonicalized_cxsmiles_opt",
        "target_format": "canonicalized cxsmiles_opt",
        "target_canonicalization_status": "already_canonical",
        "target_canonicalization_error": "",
        "messages": [
            {
                "role": "user",
                "content": "<image>\nExtract the Markush structure.\nTarget format: return <markush><cxsmi>...</cxsmi><stable>...</stable></markush>; the <cxsmi> value must use canonicalized cxsmiles_opt.",
            },
            {
                "role": "assistant",
                "content": "<markush><cxsmi>CC</cxsmi><stable></stable></markush>",
            },
        ],
        "images": [image],
    }


def test_build_mix_dedupes_earlier_partitions_and_oversamples_later(tmp_path: Path) -> None:
    hard = tmp_path / "hard.jsonl"
    base = tmp_path / "base.jsonl"
    _write_jsonl(hard, [_row("shared"), _row("hard-only")])
    _write_jsonl(base, [_row("shared"), _row("base-only")])

    manifest = build_mix(
        partitions=[
            parse_partition(f"hard=2:{hard}"),
            parse_partition(f"base=3:{base}"),
        ],
        output_jsonl=tmp_path / "train.jsonl",
        manifest_json=tmp_path / "manifest.json",
        report_md=tmp_path / "report.md",
        image_roots={},
        seed=3,
        oversample=True,
        shuffle=False,
        require_absolute=True,
        allow_eval_sources=False,
        argv=[],
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["written_rows"] == 5
    assert manifest["partition_counts"] == {"base": 3, "hard": 2}
    assert manifest["partitions"][1]["excluded_cross_partition_duplicates"] == 1
    assert manifest["partitions"][1]["repeated_rows"] == 2
    assert {
        row["source_id"] for row in rows if row["stable_scaling_mix"]["partition"] == "base"
    } == {"base-only"}


def test_build_mix_resolves_relative_images_per_partition(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_row("rel", image="renders/rel.png")])

    build_mix(
        partitions=[parse_partition(f"eval_like=1:{source}")],
        output_jsonl=tmp_path / "train.jsonl",
        manifest_json=tmp_path / "manifest.json",
        report_md=None,
        image_roots={"eval_like": Path("/data/synth")},
        seed=3,
        oversample=True,
        shuffle=False,
        require_absolute=True,
        allow_eval_sources=False,
        argv=[],
    )

    row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8"))
    assert row["images"] == ["/data/synth/renders/rel.png"]


def test_build_mix_rejects_eval_sources_by_default(tmp_path: Path) -> None:
    source = tmp_path / "eval.jsonl"
    _write_jsonl(source, [_row("eval", source="ip5m")])

    manifest = build_mix(
        partitions=[parse_partition(f"bad=1:{source}")],
        output_jsonl=tmp_path / "train.jsonl",
        manifest_json=tmp_path / "manifest.json",
        report_md=None,
        image_roots={},
        seed=3,
        oversample=False,
        shuffle=False,
        require_absolute=True,
        allow_eval_sources=False,
        argv=[],
    )

    assert manifest["written_rows"] == 0
    assert manifest["partitions"][0]["eval_source_rows"] == 1
    assert manifest["partitions"][0]["shortage"] == 1


def test_build_mix_requires_absolute_images_when_requested(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_row("rel", image="renders/rel.png")])

    with pytest.raises(ValueError, match="image path is not absolute"):
        build_mix(
            partitions=[parse_partition(f"base=1:{source}")],
            output_jsonl=tmp_path / "train.jsonl",
            manifest_json=tmp_path / "manifest.json",
            report_md=None,
            image_roots={},
            seed=3,
            oversample=True,
            shuffle=False,
            require_absolute=True,
            allow_eval_sources=False,
            argv=[],
        )
