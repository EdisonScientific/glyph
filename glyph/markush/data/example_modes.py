"""Canonical mixed-data example modes for Markush SFT preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STRUCTURE_ONLY_MARKUSH = "structure_only_markush"
PATENT_IMAGE_MARKUSH = "patent_image_markush"
TEXT_CONDITIONED_MARKUSH = "text_conditioned_markush"
MOLECULE_OCSR = "molecule_ocsr"
MARKUSH_COMPONENTS = "markush_components"
MOLPARSER_ATOM_LABEL = "molparser_atom_label"
MOLPARSER_NATIVE_MARKUSH = "molparser_native_markush"
MOLPARSER_CONVERTED_MARKUSH = "molparser_converted_markush"
STRUCTURED_TRANSCRIPT_MARKUSH = "structured_transcript_markush"

STRUCTURE_ONLY_MARKUSH_PROMPT = "Extract the Markush structure from this image."
PATENT_IMAGE_MARKUSH_PROMPT = "Extract the Markush structure from this patent image."
TEXT_CONDITIONED_MARKUSH_PROMPT = (
    "Patent text: {description}\nExtract the Markush structure from this image."
)
MOLECULE_OCSR_PROMPT = "Extract the molecular structure from this image as SMILES."
MARKUSH_COMPONENTS_PROMPT = (
    "Extract the Markush components from this image as compact JSON with keys "
    "r_labels, has_sg, sg_sections, has_m, m_sections, stable, and fragment_count."
)
MOLPARSER_ATOM_LABEL_PROMPT = (
    "Extract the atom-labeled molecular structure from this image in MolParser format."
)
MOLPARSER_NATIVE_MARKUSH_PROMPT = "Extract the Markush structure from this MolParser-Markush image."
MOLPARSER_CONVERTED_MARKUSH_PROMPT = (
    "Extract the Markush structure from this MolParser-Markush image."
)
STRUCTURED_TRANSCRIPT_MARKUSH_PROMPT = "Extract the Markush structure from this image."


@dataclass(frozen=True)
class ExampleMode:
    """A named, validated combination of task, image context, text context, and target."""

    name: str
    task: str
    image_context: str
    text_context: str
    target_format: str
    prompt_template: str
    description: str

    def prompt(self, *, description: str | None = None) -> str:
        """Return the user prompt body, without the leading image token."""

        if "{description}" not in self.prompt_template:
            return self.prompt_template
        return self.prompt_template.format(description=description or "")


EXAMPLE_MODES: dict[str, ExampleMode] = {
    STRUCTURE_ONLY_MARKUSH: ExampleMode(
        name=STRUCTURE_ONLY_MARKUSH,
        task="markush",
        image_context="structure_only",
        text_context="none",
        target_format="markush_xml",
        prompt_template=STRUCTURE_ONLY_MARKUSH_PROMPT,
        description="Structure-only Markush image to CXSMILES and substituent table XML.",
    ),
    PATENT_IMAGE_MARKUSH: ExampleMode(
        name=PATENT_IMAGE_MARKUSH,
        task="markush",
        image_context="patent_page",
        text_context="visible_in_image",
        target_format="markush_xml",
        prompt_template=PATENT_IMAGE_MARKUSH_PROMPT,
        description="Patent crop with visible structure/text to CXSMILES and substituent table XML.",
    ),
    TEXT_CONDITIONED_MARKUSH: ExampleMode(
        name=TEXT_CONDITIONED_MARKUSH,
        task="markush",
        image_context="structure_only",
        text_context="prompt_patent_text",
        target_format="markush_xml",
        prompt_template=TEXT_CONDITIONED_MARKUSH_PROMPT,
        description="Structure-only Markush image plus patent text prompt to XML.",
    ),
    MOLECULE_OCSR: ExampleMode(
        name=MOLECULE_OCSR,
        task="ocsr",
        image_context="structure_only",
        text_context="none",
        target_format="smiles",
        prompt_template=MOLECULE_OCSR_PROMPT,
        description="Standard molecule image to canonical SMILES.",
    ),
    MARKUSH_COMPONENTS: ExampleMode(
        name=MARKUSH_COMPONENTS,
        task="markush",
        image_context="any",
        text_context="optional",
        target_format="component_json",
        prompt_template=MARKUSH_COMPONENTS_PROMPT,
        description="Auxiliary Markush image to structured R/Sg/m/stable components.",
    ),
    MOLPARSER_ATOM_LABEL: ExampleMode(
        name=MOLPARSER_ATOM_LABEL,
        task="molparser",
        image_context="structure_only",
        text_context="none",
        target_format="molparser_atom_label",
        prompt_template=MOLPARSER_ATOM_LABEL_PROMPT,
        description="Auxiliary real structure crop to MolParser atom-label target.",
    ),
    MOLPARSER_NATIVE_MARKUSH: ExampleMode(
        name=MOLPARSER_NATIVE_MARKUSH,
        task="markush",
        image_context="structure_only",
        text_context="none",
        target_format="molparser_esmiles",
        prompt_template=MOLPARSER_NATIVE_MARKUSH_PROMPT,
        description="MolParser-Markush image to native MolParser E-SMILES.",
    ),
    MOLPARSER_CONVERTED_MARKUSH: ExampleMode(
        name=MOLPARSER_CONVERTED_MARKUSH,
        task="markush",
        image_context="structure_only",
        text_context="none",
        target_format="markush_xml",
        prompt_template=MOLPARSER_CONVERTED_MARKUSH_PROMPT,
        description=(
            "MolParser-Markush image with validated E-SMILES converted to shared CXSMILES."
        ),
    ),
    STRUCTURED_TRANSCRIPT_MARKUSH: ExampleMode(
        name=STRUCTURED_TRANSCRIPT_MARKUSH,
        task="markush",
        image_context="structure_only",
        text_context="none",
        target_format="structured_transcript_plus_markush_xml",
        prompt_template=STRUCTURED_TRANSCRIPT_MARKUSH_PROMPT,
        description=(
            "Auxiliary Markush image to deterministic component transcript plus final "
            "canonicalized CXSMILES_OPT XML."
        ),
    ),
}

DEFAULT_MODE_WEIGHTS: dict[str, float] = {
    STRUCTURE_ONLY_MARKUSH: 0.30,
    PATENT_IMAGE_MARKUSH: 0.25,
    TEXT_CONDITIONED_MARKUSH: 0.25,
    MOLECULE_OCSR: 0.20,
    MOLPARSER_ATOM_LABEL: 0.0,
    MOLPARSER_NATIVE_MARKUSH: 0.0,
    MOLPARSER_CONVERTED_MARKUSH: 0.0,
    STRUCTURED_TRANSCRIPT_MARKUSH: 0.0,
}


def mode_choices() -> tuple[str, ...]:
    """Return canonical example mode names."""

    return tuple(EXAMPLE_MODES)


def normalize_mode_name(value: str) -> str:
    """Normalize a descriptive mode name without accepting one-letter aliases."""

    key = value.strip().lower().replace("-", "_")
    if key not in EXAMPLE_MODES:
        valid = ", ".join(mode_choices())
        raise ValueError(f"Unknown example mode {value!r}; expected one of: {valid}")
    return key


def get_mode(value: str) -> ExampleMode:
    """Return an example mode by canonical descriptive name."""

    return EXAMPLE_MODES[normalize_mode_name(value)]


def mode_prompt(value: str, *, description: str | None = None) -> str:
    """Return the prompt body for an example mode."""

    return get_mode(value).prompt(description=description)


def modes_for_unified_record(record: dict[str, Any]) -> set[str]:
    """Return canonical example modes allowed by a unified dataset row."""

    task = str(record.get("task") or "markush").lower()
    if task == "ocsr":
        return {MOLECULE_OCSR}
    if task == "molparser":
        return {MOLPARSER_ATOM_LABEL}
    if task != "markush":
        return set()
    if str(record.get("source") or "").lower() == "molparser":
        return {MOLPARSER_CONVERTED_MARKUSH, MOLPARSER_NATIVE_MARKUSH}
    if record.get("image_has_text"):
        return {PATENT_IMAGE_MARKUSH}
    modes = {STRUCTURE_ONLY_MARKUSH}
    if record.get("description"):
        modes.add(TEXT_CONDITIONED_MARKUSH)
    return modes
