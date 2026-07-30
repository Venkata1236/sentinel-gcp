"""
Rule execution engine — iterates RULES (definitions.py) against a given
extraction + jurisdiction, producing RuleResult objects for every rule
(pass or fail), not just the failures. See ARCHITECTURE.md §3.
"""
from sentinel_gcp.rules.definitions import RULES
from sentinel_gcp.schema.extraction import ProtocolExtraction
from sentinel_gcp.schema.compliance import ComplianceFlag, RuleResult


def run_rules(extraction: ProtocolExtraction, jurisdiction: str) -> list[RuleResult]:
    """Runs every rule in RULES against the given extraction+jurisdiction.
    Returns one RuleResult per rule, whether it passed or failed —
    this is what lets generate_report later say '6/6 checks passed, 0
    flags' rather than only reporting failures with no visibility into
    what was actually checked."""
    results: list[RuleResult] = []

    for rule in RULES:
        violated = rule.condition(extraction, jurisdiction)

        if violated:
            flag = ComplianceFlag(
                flag_id=rule.rule_id,
                source="rule_engine",
                issue=rule.description,
                evidence=None,  # rule_engine flags check field presence, not specific document text
                regulation_reference=rule.regulation_reference,
                retrieved_chunk_id=None,  # must stay None per ComplianceFlag's model_validator
                impact=rule.impact,
                recommendation=rule.recommendation,
                severity=rule.severity,
                extraction_confidence=None,
                retrieval_score=None,
                llm_certainty=None,  # must stay None per ComplianceFlag's model_validator
                final_confidence=1.0,  # deterministic — no uncertainty to weigh
            )
            results.append(RuleResult(rule_id=rule.rule_id, passed=False, flag=flag))
        else:
            results.append(RuleResult(rule_id=rule.rule_id, passed=True, flag=None))

    return results