"""Opt-in prompt policies for Markush SFT and evaluation prompts."""

from __future__ import annotations

MARKUSH_PROMPT_POLICY_DEFAULT = "default"
MARKUSH_PROMPT_POLICY_VISIBLE_SG = "visible_sg"
MARKUSH_PROMPT_POLICIES = (
    MARKUSH_PROMPT_POLICY_DEFAULT,
    MARKUSH_PROMPT_POLICY_VISIBLE_SG,
)

VISIBLE_SG_INSTRUCTION = (
    "Do not include an Sg section unless the source image explicitly shows a "
    "repeating group or polymer bracket."
)


def normalize_markush_prompt_policy(value: str | None) -> str:
    policy = str(value or MARKUSH_PROMPT_POLICY_DEFAULT).strip().lower().replace("-", "_")
    if policy not in MARKUSH_PROMPT_POLICIES:
        valid = ", ".join(MARKUSH_PROMPT_POLICIES)
        raise ValueError(f"Unknown Markush prompt policy {value!r}; expected one of: {valid}")
    return policy


def apply_markush_prompt_policy(prompt: str, policy: str | None) -> str:
    """Apply an opt-in Markush prompt policy without changing target contracts.

    The policy sentence is inserted before the ``Target format:`` line when one
    is present, so target-format instructions stay last and easy to inspect.
    """

    normalized = normalize_markush_prompt_policy(policy)
    text = str(prompt or "")
    if normalized == MARKUSH_PROMPT_POLICY_DEFAULT or not text.strip():
        return text
    if VISIBLE_SG_INSTRUCTION in text:
        return text

    prefix = ""
    body = text
    if body.startswith("<image>\n"):
        prefix = "<image>\n"
        body = body.removeprefix(prefix)
    elif body.startswith("<image>"):
        prefix = "<image>"
        body = body.removeprefix(prefix).lstrip("\n")

    lines = body.splitlines()
    if not lines:
        return prefix + VISIBLE_SG_INSTRUCTION

    target_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().lower().startswith("target format:")
        ),
        len(lines),
    )
    lines.insert(target_index, VISIBLE_SG_INSTRUCTION)
    return prefix + "\n".join(lines)
