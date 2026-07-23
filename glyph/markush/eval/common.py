#!/usr/bin/env python3
"""Shared parsing primitives for Markush evaluation modules.

This module is the single source of truth for the Markush XML/CXSMILES
parsing helpers used across the evaluation and reward code
(``glyph.markush.eval.metrics_string``, ``glyph.markush.eval.metrics_mcs``,
``glyph.markush.eval.reward``, and ``glyph.markush.eval.predict``): the regex
constants for ``<cxsmi>``/``<stable>``/``<r>``, ``_strip_r_tags``,
``_extract_cxsmiles_and_stable``, ``_parse_stable``, and ``clean_prediction``.
Every caller imports from here and selects its semantics via explicit keyword
options (``require_wrapper``, ``permissive``, etc.), so all evaluators score
the same model output identically.

Regex flag policy
-----------------
All XML-tag regexes use ``re.IGNORECASE | re.DOTALL``, so both lower- and
upper-case tags (``<markush>``/``<MARKUSH>``, ``<cxsmi>``, ``<stable>``,
``<r>``, ``<think>``) match across single- and multi-line inputs.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any

#: Shared default generation budget (``max_new_tokens``) for every Markush eval
#: entry point (CLI, fast_eval, checkpoint, predict). This is the turnkey /
#: headline value: the documented benchmark reproduces ip5m greedy 511/878 at
#: 1024 new tokens. Keeping one source of truth prevents the entry points from
#: silently disagreeing (they historically defaulted to 256/512/1024).
DEFAULT_MAX_NEW_TOKENS = 1024


class _FlashLinearAttentionNoiseFilter(logging.Filter):
    """Drop the ``flash-linear-attention`` availability chatter from any logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        return "flash-linear-attention" not in message and "flash_linear_attention" not in message


def silence_dependency_logging() -> None:
    """Quiet third-party log/warning noise that never changes eval output.

    Loading the Qwen base can emit a ``flash-linear-attention`` availability
    warning, and Hugging Face downloads emit a per-request ``httpx`` INFO line.
    Neither affects predictions, so keep the predictor's stderr focused on
    evaluation progress. (The ``torch_dtype`` deprecation notice is avoided at
    the model-loading call site by passing ``dtype=`` rather than
    ``torch_dtype=``.)

    Safe to call more than once: the httpx level bump is idempotent and the
    log filter is attached to each root handler at most once.
    """

    warnings.filterwarnings("ignore", message=r".*flash.linear.attention.*")
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    root = logging.getLogger()
    noise_filter = _FlashLinearAttentionNoiseFilter()
    for handler in root.handlers:
        if not any(
            isinstance(existing, _FlashLinearAttentionNoiseFilter) for existing in handler.filters
        ):
            handler.addFilter(noise_filter)


# ---------------------------------------------------------------------------
# Compiled regex constants
#
# Every XML-tag regex uses IGNORECASE | DOTALL: IGNORECASE so that
# uppercase emissions from some checkpoints (e.g. ``<CXSMI>``) match the
# same as the canonical lowercase form, and DOTALL so that multi-line
# tag bodies (newlines inside the captured group) are matched correctly.
# ---------------------------------------------------------------------------

_XML_FLAGS = re.DOTALL | re.IGNORECASE

#: Full ``<markush>...</markush>`` envelope with optional ``<stable>`` body.
#: Groups: 1 = ``<cxsmi>`` body, 2 = optional ``<stable>`` body (may be
#: ``None`` when ``<stable>`` is absent).
RE_MARKUSH = re.compile(
    r"<markush>\s*<cxsmi>(.*?)</cxsmi>\s*(?:<stable>(.*?)</stable>)?\s*</markush>",
    _XML_FLAGS,
)

#: Lone ``<cxsmi>...</cxsmi>`` tag (no wrapper requirement).
RE_CXSMI = re.compile(r"<cxsmi>(.*?)</cxsmi>", _XML_FLAGS)

#: Lone ``<stable>...</stable>`` tag.
RE_STABLE = re.compile(r"<stable>(.*?)</stable>", _XML_FLAGS)

#: ``<r>LABEL</r>`` R-group tag.  Captures the label.
RE_R_TAG = re.compile(r"<r>(.*?)</r>", _XML_FLAGS)

#: ``<think>...</think>`` reasoning block + trailing whitespace.  Used by
#: :func:`clean_prediction` to strip out chain-of-thought before
#: extraction.
RE_THINKING = re.compile(r"<think>.*?</think>\s*", _XML_FLAGS)

#: CXSMILES positional-variation ``m:<connector>:<ring.ring>`` section.
#: Anchored at start-of-string or after a ``,`` to avoid matching the
#: ``m`` in unrelated payloads (e.g. ``Sg:n:1,2:ht,m:1:2.3``).
RE_M_SECTION = re.compile(r"(?:^|,)m:(\d+):([0-9.]+)")

# CXSMILES extension section-start markers used to bound variable-length
# extension payloads.  Anchor on real section starts rather than raw letters so
# the ``m`` in an Sg payload such as ``Sg:n:1,2:ht,m:1:2.3`` is handled as a new
# section, while unrelated payload text is not.
SECTION_MARKERS: tuple[str, ...] = (
    "atomProp:",
    "SgD:",
    "SgH:",
    "SgN:",
    "ctu:",
    "wU:",
    "wD:",
    "Sg:",
    "LN:",
    "m:",
    "c:",
    "t:",
    "f:",
    "w:",
    "S:",
    "r:",
    "@@:",
    "@:",
    "$",
    "(",
)


# Chat-template control tokens that some checkpoints emit alongside the
# Markush output.  ``clean_prediction`` truncates at the first occurrence.
_CHAT_TEMPLATE_TOKENS: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_cxsmiles_and_stable(
    text: str | None, *, require_wrapper: bool = True
) -> tuple[str, str]:
    """Extract ``(cxsmi_body, stable_body)`` from a Markush prediction string.

    Parameters
    ----------
    text:
        The raw prediction or annotation string.  ``None`` and non-string
        inputs return ``("", "")``.
    require_wrapper:
        If ``True`` (default for ``metrics_string`` semantics), only
        match content inside a complete ``<markush>...</markush>``
        envelope.  Inputs containing ``<cxsmi>`` *without* the wrapper
        return ``("", "")`` so downstream lenient-mode parsing can
        decide what to do with them.
        If ``False`` (``metrics_mcs`` / ``predict`` / ``reward``
        semantics), ``<cxsmi>`` and ``<stable>`` are matched
        independently anywhere in the text.

    Returns
    -------
    Tuple of ``(cxsmi_body, stable_body)``; either entry may be the
    empty string.
    """
    if not isinstance(text, str) or not text:
        return "", ""

    if require_wrapper:
        m = RE_MARKUSH.search(text)
        if m:
            return m.group(1).strip(), (m.group(2) or "").strip()
        return "", ""

    cxsmi_match = RE_CXSMI.search(text)
    stable_match = RE_STABLE.search(text)
    return (
        cxsmi_match.group(1).strip() if cxsmi_match else "",
        stable_match.group(1).strip() if stable_match else "",
    )


def chat_message_content(row: dict[str, Any], role: str) -> str:
    """Return the first non-empty chat message content for ``role``."""

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != role:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def sft_user_prompt(row: dict[str, Any]) -> str:
    """Return the SFT user prompt text without image placeholders."""

    user = chat_message_content(row, "user")
    if not user:
        return ""
    return user.replace("<image>\n", "").replace("<image>", "").strip()


def sft_assistant_target(row: dict[str, Any]) -> str:
    """Return the SFT assistant target, if present."""

    return chat_message_content(row, "assistant")


def sft_image_path(row: dict[str, Any]) -> str:
    """Return the first image path from an SFT ``images`` list."""

    images = row.get("images")
    if isinstance(images, list) and images:
        value = images[0]
        if isinstance(value, dict):
            value = value.get("path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def strip_r_tags(text: str | None) -> str:
    """Replace every ``<r>LABEL</r>`` with ``*`` (wildcard atom).

    Case-insensitive (``<R>...</R>`` is treated the same as
    ``<r>...</r>``).  ``None`` is normalised to ``""`` so the function
    is safe to call on possibly-missing fields.
    """
    if not text:
        return ""
    return RE_R_TAG.sub("*", text)


def extract_r_labels(text: str | None) -> list[str]:
    """Return the ordered list of R-group labels from ``<r>...</r>`` tags.

    Duplicates are preserved in the order they appear; deduplication is
    intentionally left to the caller so per-occurrence metrics
    (e.g. ordinal-aware MCS scoring) can still use this helper.
    """
    if not text:
        return []
    return RE_R_TAG.findall(text)


def split_cxsmiles(text: str | None, *, strip_trailing_pipe: bool = False) -> tuple[str, str]:
    """Split CXSMILES on the first ``|`` into ``(core, extension)``.

    Parameters
    ----------
    text:
        CXSMILES string; may already have ``<r>`` tags stripped or not.
    strip_trailing_pipe:
        If ``True`` (``metrics_mcs`` semantics), strip a trailing ``|``
        from the extension (handles the common ``CC |Sg:...|``
        formatting where RDKit emits/expects a closing pipe).
        If ``False`` (default — preserves ``metrics_string`` semantics),
        keep the extension verbatim after the first ``|``.
    """
    if not text:
        return "", ""
    if "|" not in text:
        return text.strip(), ""
    core, ext = text.split("|", 1)
    core = core.strip()
    ext = ext.strip()
    if strip_trailing_pipe:
        ext = ext.rstrip("|").strip()
    return core, ext


def parse_extension_sections(extensions: str | None) -> list[tuple[str, str]]:
    """Parse CXSMILES extension sections as ``(marker, payload)`` pairs.

    CXSMILES sections are comma-separated, but several payloads also contain
    commas. A comma is a boundary only when the following text starts with a
    known extension marker.
    """

    if not extensions:
        return []

    results: list[tuple[str, str]] = []
    i = 0
    n = len(extensions)
    while i < n:
        if extensions[i] == ",":
            i += 1
            continue

        matched_marker: str | None = None
        for marker in SECTION_MARKERS:
            if extensions.startswith(marker, i):
                matched_marker = marker
                break
        if matched_marker is None:
            comma = extensions.find(",", i)
            if comma == -1:
                break
            i = comma + 1
            continue

        if matched_marker == "$":
            close = extensions.find("$", i + len(matched_marker))
            if close == -1:
                results.append((matched_marker, extensions[i + len(matched_marker) :]))
                break
            results.append((matched_marker, extensions[i + len(matched_marker) : close + 1]))
            i = close + 1
            continue

        section_start = i + len(matched_marker)
        section_end = n
        j = section_start
        while j < n:
            if extensions[j] == ",":
                k = j + 1
                while k < n and extensions[k] == ",":
                    k += 1
                if k >= n:
                    section_end = j
                    break
                for marker in SECTION_MARKERS:
                    if extensions.startswith(marker, k):
                        section_end = j
                        break
                if section_end != n:
                    break
            j += 1

        results.append((matched_marker, extensions[section_start:section_end]))
        i = section_end + 1 if section_end < n else n

    return results


def _parse_extension_sections(extensions: str | None, target_marker: str) -> list[str]:
    return [
        payload
        for marker, payload in parse_extension_sections(extensions)
        if marker == target_marker
    ]


def parse_sg_sections(extensions: str | None) -> list[str]:
    """Parse CXSMILES ``Sg:`` extension payloads in order.

    Each returned item excludes the leading ``Sg:`` prefix.  Sections are
    bounded by the next recognized CXSMILES section marker at a comma boundary
    so ``Sg`` payloads do not swallow following ``m`` or coordinate sections.
    """

    return _parse_extension_sections(extensions, "Sg:")


def parse_m_sections(extensions: str | None) -> list[str]:
    """Parse CXSMILES positional-variation ``m:`` extension payloads in order."""

    return _parse_extension_sections(extensions, "m:")


def extract_m_section(extensions: str | None) -> str | None:
    """Extract the first CXSMILES positional-variation ``m:`` section payload."""

    sections = parse_m_sections(extensions)
    return sections[0] if sections else None


def parse_stable(stable_raw: str | None, *, permissive: bool = False) -> dict[str, list[str]]:
    """Parse a ``<stable>`` payload into ``{label: [alternatives]}``.

    Format
    ------
    ``R0:hydroxyl<n>carboxyl<ns>R1:methyl``

    * ``<ns>`` separates rows (one row per R label).
    * ``<n>`` separates alternatives within a row.
    * ``:`` separates the label from its alternatives.

    Parameters
    ----------
    stable_raw:
        Body of the ``<stable>`` tag (already extracted), or ``None``.
    permissive:
        If ``False`` (``metrics_string`` semantics): only ``:`` is a
        valid label separator.  Rows without a colon are kept as a
        labelled empty list.
        If ``True`` (``metrics_mcs`` semantics): ``=`` is also accepted
        as a label separator for mildly malformed model outputs; if a
        row contains both, the one that appears first wins.
    """
    result: dict[str, list[str]] = {}
    if not stable_raw:
        return result

    for entry in stable_raw.split("<ns>"):
        entry = entry.strip()
        if not entry:
            continue

        colon = entry.find(":")
        if permissive:
            equals = entry.find("=")
            if colon == -1 and equals == -1:
                sep = -1
            elif colon == -1:
                sep = equals
            elif equals == -1:
                sep = colon
            else:
                sep = min(colon, equals)
        else:
            sep = colon

        if sep == -1:
            # No recognized separator — keep the label with no alternatives
            # rather than silently dropping it, so false-positive rows
            # are visible to downstream metrics.
            result[entry] = []
            continue

        label = entry[:sep].strip()
        alternatives = [part.strip() for part in entry[sep + 1 :].split("<n>") if part.strip()]
        if permissive:
            # metrics_mcs's behavior: only store a row when a non-empty
            # label is present, otherwise drop.
            if label:
                result[label] = alternatives
        else:
            # metrics_string's behavior: always store, even when label
            # is empty (the previous code path used ``result[label]`` directly).
            result[label] = alternatives

    return result


def clean_prediction(text: str | None) -> str:
    """Strip ``<think>`` blocks and chat-template artifacts from raw model output.

    Steps
    -----
    1. Remove every ``<think>...</think>`` reasoning block (case-insensitive,
       spanning newlines).
    2. Truncate the string at the first occurrence of any chat-template
       control token (``<|im_end|>``, ``<|endoftext|>``, ``<|im_start|>``).
       Everything after the first such token (and the token itself) is
       discarded.
    3. Trim surrounding whitespace.

    The function is idempotent: ``clean_prediction(clean_prediction(x)) ==
    clean_prediction(x)``.
    """
    if not text:
        return ""

    text = RE_THINKING.sub("", text)
    for tok in _CHAT_TEMPLATE_TOKENS:
        text = text.split(tok)[0]
    return text.strip()


def prediction_to_cxsmiles_opt(text: str | None) -> str:
    """Convert a raw model prediction into a bare ``cxsmiles_opt`` string.

    Composition of :func:`clean_prediction` and the ``<cxsmi>`` extractor,
    with a graceful fallback to the cleaned text when no ``<cxsmi>`` tag
    is present (some checkpoints emit only the SMILES core without the
    XML envelope).

    Returns ``""`` on empty input.
    """
    cleaned = clean_prediction(text)
    if not cleaned:
        return ""
    m = RE_CXSMI.search(cleaned)
    if m:
        return m.group(1).strip()
    return cleaned


__all__ = [
    "RE_CXSMI",
    "RE_MARKUSH",
    "RE_M_SECTION",
    "RE_R_TAG",
    "RE_STABLE",
    "RE_THINKING",
    "SECTION_MARKERS",
    "chat_message_content",
    "clean_prediction",
    "extract_cxsmiles_and_stable",
    "extract_m_section",
    "extract_r_labels",
    "parse_extension_sections",
    "parse_m_sections",
    "parse_sg_sections",
    "parse_stable",
    "prediction_to_cxsmiles_opt",
    "sft_assistant_target",
    "sft_image_path",
    "sft_user_prompt",
    "split_cxsmiles",
    "strip_r_tags",
]
