"""Public helpers for Markush target normalization and converter checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from glyph.markush.data.canonicalize import canonicalize_cxsmiles_checked
from glyph.markush.data.pipeline import opt_to_standard_cxsmiles, standard_to_opt_cxsmiles
from glyph.markush.eval import common
from glyph.markush.eval.markush_graph import markush_graph_equal

SourceFormat = Literal["auto", "cxsmiles_opt", "standard"]


@dataclass(slots=True)
class ConverterCheckResult:
    """Standard/opt converter round-trip check for one target."""

    valid: bool
    cxsmiles_opt: str = ""
    standard: str = ""
    roundtrip_opt: str = ""
    graph_equal: bool = False
    error: str = ""

    def asdict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(slots=True)
class TargetNormalizationResult:
    """Normalized Markush target plus provenance and converter diagnostics."""

    valid: bool
    input: str
    cxsmiles_opt: str = ""
    markush_xml: str = ""
    stable: str = ""
    source_format: SourceFormat = "auto"
    target_source: str = ""
    target_format: str = ""
    canonicalization_status: str = ""
    canonicalization_error: str = ""
    converter_check: ConverterCheckResult | None = None
    error: str = ""

    def asdict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""

        payload = asdict(self)
        if self.converter_check is not None:
            payload["converter_check"] = self.converter_check.asdict()
        return payload


def _wrap_markush(cxsmiles_opt: str, stable: str = "") -> str:
    return f"<markush><cxsmi>{cxsmiles_opt}</cxsmi><stable>{stable}</stable></markush>"


def _extract_target(value: str) -> tuple[str, str]:
    cxsmiles, stable = common.extract_cxsmiles_and_stable(value, require_wrapper=False)
    if cxsmiles:
        return cxsmiles, stable
    return value.strip(), ""


def _to_opt(cxsmiles: str, source_format: SourceFormat) -> tuple[str, str, str]:
    if source_format == "cxsmiles_opt":
        return cxsmiles, "cxsmiles_opt", "cxsmiles_opt"
    if source_format == "standard":
        return standard_to_opt_cxsmiles(cxsmiles), "CXSMILES->cxsmiles_opt", "cxsmiles_opt"
    if "<r>" in cxsmiles.lower():
        return cxsmiles, "cxsmiles_opt", "cxsmiles_opt"
    converted = standard_to_opt_cxsmiles(cxsmiles)
    return converted, "CXSMILES->cxsmiles_opt", "cxsmiles_opt"


def check_cxsmiles_round_trip(cxsmiles_opt: str) -> ConverterCheckResult:
    """Check that opt -> standard -> opt preserves the Markush graph."""

    try:
        standard = opt_to_standard_cxsmiles(cxsmiles_opt)
        roundtrip_opt = standard_to_opt_cxsmiles(standard)
        equality = markush_graph_equal(cxsmiles_opt, roundtrip_opt)
    except Exception as exc:
        return ConverterCheckResult(
            valid=False,
            cxsmiles_opt=cxsmiles_opt,
            error=str(exc),
        )
    return ConverterCheckResult(
        valid=equality.equal,
        cxsmiles_opt=cxsmiles_opt,
        standard=standard,
        roundtrip_opt=roundtrip_opt,
        graph_equal=equality.equal,
        error=equality.error or equality.pred_error or equality.gt_error,
    )


def normalize_markush_target(
    value: str,
    *,
    source_format: SourceFormat = "auto",
    canonicalize: bool = True,
    require_canonical: bool = False,
    check_roundtrip: bool = True,
) -> TargetNormalizationResult:
    """Normalize a Markush XML or CXSMILES target to the paper ``cxsmiles_opt`` contract."""

    raw = value.strip()
    if source_format not in {"auto", "cxsmiles_opt", "standard"}:
        raise ValueError(
            f"source_format must be one of auto, cxsmiles_opt, standard; got {source_format!r}"
        )
    if not raw:
        return TargetNormalizationResult(valid=False, input=value, error="empty target")

    cxsmiles, stable = _extract_target(raw)
    try:
        opt_cxsmiles, target_source, target_format = _to_opt(cxsmiles, source_format)
    except Exception as exc:
        return TargetNormalizationResult(
            valid=False,
            input=value,
            stable=stable,
            source_format=source_format,
            error=f"conversion to cxsmiles_opt failed: {exc}",
        )
    if not opt_cxsmiles:
        return TargetNormalizationResult(
            valid=False,
            input=value,
            stable=stable,
            source_format=source_format,
            error="conversion to cxsmiles_opt produced an empty target",
        )

    if opt_cxsmiles != cxsmiles:
        equality = markush_graph_equal(cxsmiles, opt_cxsmiles)
        if not equality.equal:
            return TargetNormalizationResult(
                valid=False,
                input=value,
                cxsmiles_opt=opt_cxsmiles,
                stable=stable,
                source_format=source_format,
                target_source=target_source,
                target_format=target_format,
                error=equality.error or equality.pred_error or equality.gt_error,
            )

    canonicalization_status = ""
    canonicalization_error = ""
    if canonicalize:
        result = canonicalize_cxsmiles_checked(opt_cxsmiles)
        canonicalization_status = result.status
        canonicalization_error = result.error or ""
        if result.value is None or result.fallback:
            if require_canonical:
                return TargetNormalizationResult(
                    valid=False,
                    input=value,
                    cxsmiles_opt=opt_cxsmiles,
                    stable=stable,
                    source_format=source_format,
                    target_source=target_source,
                    target_format=target_format,
                    canonicalization_status=canonicalization_status,
                    canonicalization_error=canonicalization_error,
                    error="canonicalization was not accepted",
                )
        else:
            opt_cxsmiles = result.value
            target_source = f"{target_source}->canonicalized_cxsmiles_opt"
            target_format = "canonicalized cxsmiles_opt"

    converter_check = check_cxsmiles_round_trip(opt_cxsmiles) if check_roundtrip else None
    if converter_check is not None and not converter_check.valid:
        return TargetNormalizationResult(
            valid=False,
            input=value,
            cxsmiles_opt=opt_cxsmiles,
            markush_xml=_wrap_markush(opt_cxsmiles, stable),
            stable=stable,
            source_format=source_format,
            target_source=target_source,
            target_format=target_format,
            canonicalization_status=canonicalization_status,
            canonicalization_error=canonicalization_error,
            converter_check=converter_check,
            error="standard/opt converter round-trip failed",
        )

    return TargetNormalizationResult(
        valid=True,
        input=value,
        cxsmiles_opt=opt_cxsmiles,
        markush_xml=_wrap_markush(opt_cxsmiles, stable),
        stable=stable,
        source_format=source_format,
        target_source=target_source,
        target_format=target_format,
        canonicalization_status=canonicalization_status,
        canonicalization_error=canonicalization_error,
        converter_check=converter_check,
    )


__all__ = [
    "ConverterCheckResult",
    "SourceFormat",
    "TargetNormalizationResult",
    "check_cxsmiles_round_trip",
    "normalize_markush_target",
]
