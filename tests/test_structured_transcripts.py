from __future__ import annotations

import json

from glyph.markush.data.structured_transcripts import (
    SCHEMA_VERSION,
    structured_transcript_for_markush_target,
    validate_structured_transcript_content,
    validate_structured_transcript_payload,
)
from glyph.markush.eval import common


def test_structured_transcript_is_deterministic_and_eval_compatible():
    target = (
        "<markush><cxsmi><r>R1</r>CC|Sg:n:1,2:n:ht</cxsmi>"
        "<stable>R1:methyl&lt;n&gt;ethyl&lt;ns&gt;</stable></markush>"
    )

    first = structured_transcript_for_markush_target(target)
    second = structured_transcript_for_markush_target(target)

    assert first.valid, first.error
    assert second.valid, second.error
    assert first.transcript_json == second.transcript_json
    assert first.content == second.content

    transcript_json, final_xml = first.content.split("\n", 1)
    payload = json.loads(transcript_json)
    assert payload == first.payload
    assert final_xml == first.final_xml
    assert common.prediction_to_cxsmiles_opt(first.content) == "<r>R1</r>CC|Sg:n:1,2:n:ht"

    ok, error = validate_structured_transcript_payload(payload)
    assert ok, error
    ok, error = validate_structured_transcript_content(first.content)
    assert ok, error


def test_structured_transcript_payload_captures_markush_extensions():
    result = structured_transcript_for_markush_target("<r>R1</r>CC|m:0:1.2,Sg:n:1,2:n:ht")

    assert result.valid, result.error
    assert result.payload is not None
    assert result.payload["schema_version"] == SCHEMA_VERSION
    assert result.payload["target_format"] == "canonicalized cxsmiles_opt"
    assert result.payload["graph"]["atom_count"] == 3
    assert result.payload["graph"]["fragment_count"] == 1
    assert result.payload["extensions"]["r_labels"] == [{"atom": 0, "label": "R1"}]
    assert result.payload["extensions"]["m_sections"] == [{"connector": 0, "atoms": [1, 2]}]
    assert result.payload["extensions"]["sg_sections"] == [
        {"kind": "n", "atoms": [1, 2], "fields": ["n", "ht"]}
    ]
    assert result.final_xml == (
        "<markush><cxsmi><r>R1</r>CC|m:0:1.2,Sg:n:1,2:n:ht</cxsmi><stable></stable></markush>"
    )


def test_structured_transcript_rejects_invalid_targets():
    result = structured_transcript_for_markush_target(
        "<markush><cxsmi><r>R1</r>bad</cxsmi><stable></stable></markush>"
    )

    assert result.valid is False
    assert result.payload is None
    assert result.error


def test_structured_transcript_validation_rejects_tampered_counts():
    result = structured_transcript_for_markush_target("<r>R1</r>CC")

    assert result.valid, result.error
    assert result.payload is not None
    payload = json.loads(result.transcript_json)
    payload["graph"]["atom_count"] = 999

    ok, error = validate_structured_transcript_payload(payload)
    assert ok is False
    assert error == "atom_count does not match parsed graph"


def test_structured_transcript_content_validation_rejects_tampered_final_xml():
    result = structured_transcript_for_markush_target("<r>R1</r>CC")

    assert result.valid, result.error
    content = (
        result.transcript_json + "\n<markush><cxsmi><r>R2</r>CC</cxsmi><stable></stable></markush>"
    )

    ok, error = validate_structured_transcript_content(content)
    assert ok is False
    assert error == "final XML CXSMILES does not match transcript final.cxsmiles_opt"
