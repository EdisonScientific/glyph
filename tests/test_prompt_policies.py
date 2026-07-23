from __future__ import annotations

import pytest

from glyph.markush.data.prompt_policies import (
    MARKUSH_PROMPT_POLICY_DEFAULT,
    MARKUSH_PROMPT_POLICY_VISIBLE_SG,
    VISIBLE_SG_INSTRUCTION,
    apply_markush_prompt_policy,
    normalize_markush_prompt_policy,
)


def test_default_prompt_policy_preserves_text() -> None:
    prompt = "Extract the Markush structure from this image.\nTarget format: return XML."

    assert apply_markush_prompt_policy(prompt, MARKUSH_PROMPT_POLICY_DEFAULT) == prompt


def test_visible_sg_policy_inserts_before_target_format() -> None:
    prompt = "<image>\nExtract the Markush structure from this image.\nTarget format: return XML."

    updated = apply_markush_prompt_policy(prompt, MARKUSH_PROMPT_POLICY_VISIBLE_SG)

    assert updated.splitlines() == [
        "<image>",
        "Extract the Markush structure from this image.",
        VISIBLE_SG_INSTRUCTION,
        "Target format: return XML.",
    ]
    assert apply_markush_prompt_policy(updated, MARKUSH_PROMPT_POLICY_VISIBLE_SG) == updated


def test_prompt_policy_normalizes_hyphenated_names() -> None:
    assert normalize_markush_prompt_policy("visible-sg") == MARKUSH_PROMPT_POLICY_VISIBLE_SG
    with pytest.raises(ValueError, match="Unknown Markush prompt policy"):
        normalize_markush_prompt_policy("not-a-policy")
