"""Feature profiling for Markush data distributions.

The generator should cover broad patent structure space while still matching
the benchmark distributions we care about.  This module keeps that measurable:
it extracts compact, renderer-agnostic features from seed JSONL, render
manifests, unified records, and chat-style SFT JSONL.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CXSMI_TAG_RE = re.compile(r"<cxsmi>(.*?)</cxsmi>", flags=re.IGNORECASE | re.DOTALL)
RGROUP_TAG_RE = re.compile(r"<r>([^<]+)</r>")
WILDCARD_RE = re.compile(r"\[(?:\d+)?\*\]|\*")
CHEMICAL_LABELS = {"OR", "CF3", "CN", "Me", "OH", "NH", "OMe", "NR", "SCH3"}

HISTOGRAMS = (
    "rgroup_count",
    "rgroup_bin",
    "label_category",
    "feature_combo",
    "sg_count",
    "m_count",
    "stable_row_count",
)


def iter_json_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Load records from JSONL files, JSON lists, or directories containing them."""

    files: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
            files.extend(sorted(path.glob("*/unified.jsonl")))
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)

    records: list[dict[str, Any]] = []
    for path in sorted(dict.fromkeys(files)):
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            continue
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(row for row in data if isinstance(row, dict))
    return records


def _assistant_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def extract_markush_text(record: dict[str, Any]) -> str:
    """Return the best available CXSMILES/Markush target text from a record."""

    for key in ("cxsmiles_opt", "cxsmiles", "cxsmiles_standard", "cxsmiles_input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    annotation = record.get("annotation")
    if isinstance(annotation, str) and annotation.strip():
        match = CXSMI_TAG_RE.search(annotation)
        return match.group(1).strip() if match else annotation.strip()

    text = _assistant_text(record)
    if text:
        match = CXSMI_TAG_RE.search(text)
        return match.group(1).strip() if match else text.strip()
    return ""


def _stable_text(record: dict[str, Any]) -> str:
    annotation = record.get("annotation")
    if isinstance(annotation, str) and annotation.strip():
        text = annotation
    else:
        text = _assistant_text(record)
    match = re.search(r"<stable>(.*?)</stable>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _core_and_extension(text: str) -> tuple[str, str]:
    if "|" not in text:
        return text, ""
    core, rest = text.split("|", 1)
    extension = rest.rsplit("|", 1)[0] if "|" in rest else rest
    return core.strip(), extension.strip()


def _labels_from_standard_extension(extension: str) -> list[str]:
    match = re.search(r"\$([^$]*)\$", extension)
    if not match:
        return []
    return [label for label in match.group(1).split(";") if label]


def rgroup_bin(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    if count <= 12:
        return "9-12"
    return "13+"


def label_category(label: str) -> str:
    value = label.strip()
    if value == "_AP":
        return "attachment"
    if value in CHEMICAL_LABELS:
        return "chemical"
    match = re.fullmatch(r"R(\d+)", value)
    if match:
        return "high_index_r" if int(match.group(1)) >= 21 else "standard_r"
    if value == "R":
        return "standard_r"
    if re.fullmatch(r"R[a-z]+|R'+", value):
        return "lettered_r"
    if re.fullmatch(r"R\d+[a-z]+", value):
        return "subscripted_r"
    if re.fullmatch(r"[A-Z]", value):
        return "single_letter"
    if re.fullmatch(r"[A-Z]+\d+", value):
        return "letter_number"
    return "other"


def features_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Extract distribution features from a Markush-like record."""

    text = extract_markush_text(record)
    core, extension = _core_and_extension(text)
    labels = RGROUP_TAG_RE.findall(text)
    if not labels:
        labels = _labels_from_standard_extension(extension)
    r_count = len(labels) if labels else len(WILDCARD_RE.findall(core))
    has_sg = "Sg:" in extension or "Sg:" in text
    has_m = re.search(r"(^|[,|])m:", extension) is not None
    sg_count = len(re.findall(r"(^|,)Sg:", extension))
    m_count = len(re.findall(r"(^|,)m:", extension))
    fragment = "." in core
    stable = _stable_text(record)
    stable_rows = 0
    if stable:
        stable_rows = len([row for row in stable.split("<ns>") if row.strip()])

    return {
        "rgroup_count": r_count,
        "rgroup_bin": rgroup_bin(r_count),
        "rgroup_labels": labels,
        "label_categories": [label_category(label) for label in labels],
        "has_sg": has_sg,
        "has_m": has_m,
        "sg_count": sg_count,
        "m_count": m_count,
        "fragment": fragment,
        "stable_row_count": stable_rows,
        "feature_combo": (
            f"r={rgroup_bin(r_count)}|sg={int(has_sg)}|m={int(has_m)}|frag={int(fragment)}"
        ),
    }


def _normalize_counts(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counter.items())}


def profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a JSON-serializable feature profile for Markush records."""

    histograms: dict[str, Counter[str]] = {name: Counter() for name in HISTOGRAMS}
    rates = Counter()
    profiled = 0
    skipped = 0
    for record in records:
        if str(record.get("task") or "markush").lower() == "ocsr":
            skipped += 1
            continue
        features = features_from_record(record)
        if not extract_markush_text(record):
            skipped += 1
            continue
        profiled += 1
        histograms["rgroup_count"][str(features["rgroup_count"])] += 1
        histograms["rgroup_bin"][str(features["rgroup_bin"])] += 1
        histograms["feature_combo"][str(features["feature_combo"])] += 1
        histograms["sg_count"][str(features["sg_count"])] += 1
        histograms["m_count"][str(features["m_count"])] += 1
        histograms["stable_row_count"][str(features["stable_row_count"])] += 1
        for category in features["label_categories"]:
            histograms["label_category"][category] += 1
        rates["has_sg"] += int(features["has_sg"])
        rates["has_m"] += int(features["has_m"])
        rates["fragment"] += int(features["fragment"])
        rates["zero_rgroup"] += int(features["rgroup_count"] == 0)

    rate_values = {
        key: (value / profiled if profiled else 0.0) for key, value in sorted(rates.items())
    }
    return {
        "version": 1,
        "records": profiled,
        "skipped_records": skipped,
        "rates": rate_values,
        "histograms": {
            name: {
                "counts": dict(sorted(counter.items())),
                "proportions": _normalize_counts(counter),
            }
            for name, counter in histograms.items()
        },
    }


def profile_paths(paths: list[Path]) -> dict[str, Any]:
    return profile_records(iter_json_records(paths))


def load_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def histogram_proportions(profile: dict[str, Any], histogram: str) -> dict[str, float]:
    hist = profile.get("histograms", {}).get(histogram, {})
    if "proportions" in hist:
        return {str(key): float(value) for key, value in hist["proportions"].items()}
    counts = Counter({str(key): int(value) for key, value in hist.get("counts", {}).items()})
    return _normalize_counts(counts)


def target_counts_from_profile(
    profile: dict[str, Any],
    total: int,
    *,
    histogram: str = "feature_combo",
) -> dict[str, int]:
    """Convert a profile histogram into deterministic integer target counts."""

    proportions = histogram_proportions(profile, histogram)
    if total <= 0 or not proportions:
        return {}
    raw = {key: value * total for key, value in proportions.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    for key in sorted(raw, key=lambda item: raw[item] - counts[item], reverse=True)[:remainder]:
        counts[key] += 1
    return {key: value for key, value in counts.items() if value > 0}


def profile_key_for_record(record: dict[str, Any], histogram: str = "feature_combo") -> str:
    features = features_from_record(record)
    value = features.get(histogram)
    if value is None:
        raise ValueError(f"Unsupported profile histogram: {histogram}")
    return str(value)


def compare_profiles(candidate: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Compare two profiles using per-histogram total variation distance."""

    distances: dict[str, float] = {}
    for histogram in HISTOGRAMS:
        left = histogram_proportions(candidate, histogram)
        right = histogram_proportions(target, histogram)
        keys = set(left) | set(right)
        distances[histogram] = 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
    core_histograms = [name for name in HISTOGRAMS if name != "stable_row_count"]
    return {
        "candidate_records": candidate.get("records", 0),
        "target_records": target.get("records", 0),
        "total_variation": distances,
        "mean_total_variation": (
            sum(distances.values()) / len(distances) if distances else 0.0
        ),
        "mean_core_total_variation": (
            sum(distances[name] for name in core_histograms) / len(core_histograms)
            if core_histograms
            else 0.0
        ),
    }
