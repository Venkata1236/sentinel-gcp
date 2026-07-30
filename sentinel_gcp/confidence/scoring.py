"""
compute_confidence — the weighted confidence formula referenced
throughout ARCHITECTURE.md §4. Deliberately a plain function, not a
graph node or an LLM call — confidence is arithmetic over values already
computed upstream (extraction confidence, retrieval score, LLM
certainty), never a bare model-asserted number.

Kept separate from generate_report.py so this formula is independently
unit-testable — e.g. confirming a rule_engine flag (no llm_certainty,
no retrieval_score) still produces a sensible confidence, versus an
Agent 2 flag with all three inputs present.
"""
from sentinel_gcp.schema.compliance import ComplianceFlag
from sentinel_gcp.config import settings


def compute_confidence(flag: ComplianceFlag) -> float:
    """Rule-engine flags are deterministic — no uncertainty to weigh, so
    they always resolve to 1.0 regardless of the configured weights.
    Agent 2 flags combine extraction confidence, retrieval score, and
    LLM certainty using the weights from config (defaults: 0.3/0.3/0.4,
    per ARCHITECTURE.md §4) — any missing input defaults to a
    conservative 0.5 rather than crashing or silently zeroing out."""
    if flag.source == "rule_engine":
        return 1.0

    extraction_confidence = flag.extraction_confidence if flag.extraction_confidence is not None else 0.5
    retrieval_score = flag.retrieval_score if flag.retrieval_score is not None else 0.5
    llm_certainty = flag.llm_certainty if flag.llm_certainty is not None else 0.5

    return (
        settings.CONFIDENCE_WEIGHT_EXTRACTION * extraction_confidence
        + settings.CONFIDENCE_WEIGHT_RETRIEVAL * retrieval_score
        + settings.CONFIDENCE_WEIGHT_LLM_CERTAINTY * llm_certainty
    )