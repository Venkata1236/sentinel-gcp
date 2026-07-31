"""
Rule definitions — the actual compliance checks the rule engine runs.

Every rule's condition operates on ALREADY-VALIDATED, CANONICAL fields
only (ProtocolExtraction + jurisdiction) — never on raw document text or
document-specific labels. This is what makes a rule written once work
identically regardless of whether the source document called something
"Protocol Number" or "Study Code" — see ARCHITECTURE.md §3.

Adding a new rule means adding a new Rule object to RULES below — nothing
else needs to change (engine.py just iterates this list).

RULE-007 (added after real-document testing): ARCT-165-01, a real
protocol tested in this project, surfaced the "unknown" jurisdiction
case — neither an IND number nor a EudraCT number was found, so
jurisdiction cannot be determined. Paired with pinecone_store.py's fix
scoping retrieval to ICH-only in that case, this rule makes the
uncertainty a visible, high-severity finding in every report, rather
than a silent retrieval-scoping decision a reviewer would never see.
"""
from dataclasses import dataclass
from typing import Callable

from sentinel_gcp.schema.extraction import ProtocolExtraction


@dataclass
class Rule:
    rule_id: str
    description: str
    condition: Callable[[ProtocolExtraction, str], bool]  # (extraction, jurisdiction) -> True if VIOLATED
    regulation_reference: str
    severity: str  # "low" | "medium" | "high"
    impact: str
    recommendation: str


def _ind_missing_for_fda(extraction: ProtocolExtraction, jurisdiction: str) -> bool:
    return jurisdiction == "FDA" and (
        extraction.metadata.ind_number is None
        or not extraction.metadata.ind_number.value
    )


def _eudract_missing_for_ema(extraction: ProtocolExtraction, jurisdiction: str) -> bool:
    return jurisdiction == "EMA" and (
        extraction.metadata.eudract_number is None
        or not extraction.metadata.eudract_number.value
    )


def _sae_timeline_missing(extraction: ProtocolExtraction, jurisdiction: str) -> bool:
    return (
        extraction.sae_reporting_timeline is None
        or not extraction.sae_reporting_timeline.value
    )


def _no_inclusion_criteria(extraction: ProtocolExtraction, jurisdiction: str) -> bool:
    return len(extraction.inclusion_criteria) == 0


def _no_exclusion_criteria(extraction: ProtocolExtraction, jurisdiction: str) -> bool:
    return len(extraction.exclusion_criteria) == 0


def _no_primary_endpoint(extraction: ProtocolExtraction, jurisdiction: str) -> bool:
    return not extraction.primary_endpoint


def _jurisdiction_unknown(extraction: ProtocolExtraction, jurisdiction: str) -> bool:
    return jurisdiction == "unknown"


RULES: list[Rule] = [
    Rule(
        rule_id="RULE-001",
        description="Missing IND number for FDA-jurisdiction trial",
        condition=_ind_missing_for_fda,
        regulation_reference="21 CFR 312.23",
        severity="high",
        impact="An FDA-regulated trial without a documented IND number may not be legally authorized to proceed.",
        recommendation="Confirm the IND number is documented in the protocol, or verify this trial genuinely falls outside IND requirements.",
    ),
    Rule(
        rule_id="RULE-002",
        description="Missing EudraCT number for EU-jurisdiction trial",
        condition=_eudract_missing_for_ema,
        regulation_reference="EU Clinical Trials Directive",
        severity="high",
        impact="An EU-regulated trial without a documented EudraCT number may not meet EU registration requirements.",
        recommendation="Confirm the EudraCT number is documented, or verify jurisdiction classification is correct.",
    ),
    Rule(
        rule_id="RULE-003",
        description="No SAE reporting timeline extracted",
        condition=_sae_timeline_missing,
        regulation_reference="21 CFR 312.32 / ICH-GCP E6(R3)",
        severity="medium",
        impact="Without a documented SAE reporting timeline, compliance with expedited safety reporting requirements cannot be verified.",
        recommendation="Manually locate and verify the SAE/AE reporting timeline section — extraction may have missed it, or it may genuinely be absent from the protocol.",
    ),
    Rule(
        rule_id="RULE-004",
        description="No inclusion criteria extracted",
        condition=_no_inclusion_criteria,
        regulation_reference="ICH-GCP E6(R3) — protocol content requirements",
        severity="high",
        impact="A protocol must define who is eligible to participate; an empty extraction here is a strong signal of an extraction failure, not an actual protocol gap.",
        recommendation="Manually verify inclusion criteria are present in the source document — this is very likely an extraction miss, not a real compliance gap.",
    ),
    Rule(
        rule_id="RULE-005",
        description="No exclusion criteria extracted",
        condition=_no_exclusion_criteria,
        regulation_reference="ICH-GCP E6(R3) — protocol content requirements",
        severity="high",
        impact="Same reasoning as RULE-004 — likely an extraction failure rather than a genuine absence.",
        recommendation="Manually verify exclusion criteria are present in the source document.",
    ),
    Rule(
        rule_id="RULE-006",
        description="No primary endpoint extracted",
        condition=_no_primary_endpoint,
        regulation_reference="ICH-GCP E6(R3) — protocol content requirements",
        severity="high",
        impact="Every interventional trial must define a primary endpoint; a missing extraction here needs verification before trusting anything else in the report.",
        recommendation="Manually verify the primary endpoint is stated in the source document.",
    ),
    Rule(
        rule_id="RULE-007",
        description="Jurisdiction could not be determined from extracted IND/EudraCT fields",
        condition=_jurisdiction_unknown,
        regulation_reference="N/A — process gap, not a regulatory citation",
        severity="high",
        impact=(
            "Compliance checks were limited to jurisdiction-agnostic ICH-GCP content only. "
            "FDA- or EMA-specific requirements were NOT checked, since which framework "
            "applies could not be determined from the extracted IND/EudraCT fields."
        ),
        recommendation="Manually confirm trial jurisdiction before relying on this report's compliance findings.",
    ),
]