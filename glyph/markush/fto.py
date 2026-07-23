"""Conservative freedom-to-operate helpers for Markush outputs.

These helpers expose validated parsing and matching building blocks.  They do
not implement complete patent-claim coverage semantics.  When a feature is
outside the supported subset, the matcher returns structured uncertainty
instead of making a silent coverage decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from glyph.markush.eval import common
from glyph.markush.eval.markush_graph import markush_graph_equal, parse_markush_graph


@dataclass(frozen=True, slots=True)
class MarkushClaim:
    """Parsed Markush claim representation."""

    raw: str
    markush_xml: str
    cxsmiles_opt: str
    stable_raw: str
    stable: dict[str, list[str]]
    backbone_smiles: str
    extension: str
    r_labels: tuple[str, ...]
    m_sections: tuple[str, ...]
    sg_sections: tuple[str, ...]
    valid_graph: bool
    parse_error: str

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConcreteMatchResult:
    """Result of matching a concrete molecule against a supported claim subset."""

    supported: bool
    matches: bool | None
    protocol: str
    unsupported_reasons: tuple[str, ...] = ()
    query_smiles: str = ""
    molecule_smiles: str = ""
    error: str = ""

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_xml(cxsmiles: str, stable: str) -> str:
    if not cxsmiles:
        return ""
    return f"<markush><cxsmi>{cxsmiles}</cxsmi><stable>{stable}</stable></markush>"


def parse_claim(raw: str) -> MarkushClaim:
    """Parse a Markush XML/CXSMILES claim into reusable components."""

    cleaned = common.clean_prediction(raw)
    cxsmiles, stable_raw = common.extract_cxsmiles_and_stable(cleaned, require_wrapper=False)
    if not cxsmiles:
        cxsmiles = common.prediction_to_cxsmiles_opt(cleaned)
        stable_raw = ""
    core, extension = common.split_cxsmiles(cxsmiles, strip_trailing_pipe=True)
    r_labels = tuple(common.extract_r_labels(core))
    m_sections = tuple(common.parse_m_sections(extension))
    sg_sections = tuple(common.parse_sg_sections(extension))

    valid_graph = False
    parse_error = ""
    if cxsmiles:
        try:
            parsed = parse_markush_graph(cxsmiles)
            valid_graph = parsed.valid
            parse_error = parsed.error
        except Exception as exc:  # pragma: no cover - defensive around RDKit/parser failures
            parse_error = str(exc)
    else:
        parse_error = "empty claim"

    return MarkushClaim(
        raw=cleaned,
        markush_xml=_normalize_xml(cxsmiles, stable_raw),
        cxsmiles_opt=cxsmiles,
        stable_raw=stable_raw,
        stable=common.parse_stable(stable_raw, permissive=True),
        backbone_smiles=common.strip_r_tags(core),
        extension=extension,
        r_labels=r_labels,
        m_sections=m_sections,
        sg_sections=sg_sections,
        valid_graph=valid_graph,
        parse_error=parse_error,
    )


def extract_components(raw: str) -> dict[str, Any]:
    """Return parsed Markush components as a JSON-serializable dict."""

    return parse_claim(raw).asdict()


def normalize_claim(raw: str) -> str:
    """Return the normalized Markush XML envelope used by package APIs."""

    return parse_claim(raw).markush_xml


def strict_equal(left: str, right: str) -> bool:
    """Return strict graph-isomorphism equality for two Markush claims."""

    result = markush_graph_equal(left, right)
    return bool(result.equal and result.valid_gt and result.valid_pred)


def _unsupported_simple_match_reasons(claim: MarkushClaim) -> list[str]:
    reasons: list[str] = []
    if not claim.valid_graph:
        reasons.append("claim_does_not_parse_as_valid_markush_graph")
    if claim.m_sections:
        reasons.append("m_sections_not_supported_by_simple_matcher")
    if claim.sg_sections:
        reasons.append("sg_sections_not_supported_by_simple_matcher")
    if any(values for values in claim.stable.values()):
        reasons.append("specific_stable_substituent_constraints_not_supported")
    return reasons


def _canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def match_concrete_smiles(smiles: str, claim_raw: str) -> ConcreteMatchResult:
    """Match a concrete molecule against a conservative supported claim subset.

    Supported cases:

    - no R labels: exact canonical RDKit SMILES equality;
    - R labels with no stable substituent constraints: RDKit wildcard
      substructure query after replacing ``<r>...</r>`` labels with ``*``.

    Unsupported cases return ``supported=False`` with reasons instead of a
    coverage verdict.
    """

    claim = parse_claim(claim_raw)
    reasons = _unsupported_simple_match_reasons(claim)
    if reasons:
        return ConcreteMatchResult(
            supported=False,
            matches=None,
            protocol="unsupported",
            unsupported_reasons=tuple(reasons),
            query_smiles=claim.backbone_smiles,
            molecule_smiles=smiles,
            error=claim.parse_error,
        )

    from rdkit import Chem

    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid molecule SMILES: {smiles}")

        if not claim.r_labels:
            claim_smiles = _canonical_smiles(claim.backbone_smiles)
            molecule_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            return ConcreteMatchResult(
                supported=True,
                matches=molecule_smiles == claim_smiles,
                protocol="rdkit_exact_canonical_smiles",
                query_smiles=claim_smiles,
                molecule_smiles=molecule_smiles,
            )

        query = Chem.MolFromSmarts(claim.backbone_smiles)
        if query is None:
            raise ValueError(f"unsupported wildcard query: {claim.backbone_smiles}")
        return ConcreteMatchResult(
            supported=True,
            matches=bool(molecule.HasSubstructMatch(query)),
            protocol="rdkit_wildcard_substructure_unconstrained_rgroups",
            query_smiles=claim.backbone_smiles,
            molecule_smiles=Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        )
    except Exception as exc:
        return ConcreteMatchResult(
            supported=False,
            matches=None,
            protocol="error",
            unsupported_reasons=("rdkit_matching_error",),
            query_smiles=claim.backbone_smiles,
            molecule_smiles=smiles,
            error=str(exc),
        )


__all__ = [
    "ConcreteMatchResult",
    "MarkushClaim",
    "extract_components",
    "match_concrete_smiles",
    "normalize_claim",
    "parse_claim",
    "strict_equal",
]
