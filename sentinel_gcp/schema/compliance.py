"""
Compliance schema — shared output contract for the rule engine (node 8,
deterministic) and Agent 2 / compliance_check (node 10, LLM).

Both sources produce the SAME ComplianceFlag shape, distinguished only by
`source`. This matters: it's what lets generate_report (and a human
reviewer) tell "the system is certain this is missing" (source=rule_engine,
confidence=1.0) apart from "the model made a judgment call about ambiguous
wording" (source=agent_2, confidence=weighted/uncertain) — see
ARCHITECTURE.md §4 for the confidence formula this feeds into.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class ComplianceFlag(BaseModel):
    flag_id: str                                    # e.g. "RULE-001" or an Agent 2-generated id
    source: Literal["rule_engine", "agent_2"]
    issue: str                                        # human-readable description of the gap
    evidence: Optional[str] = None                    # quoted/paraphrased text the flag is based on
    regulation_reference: Optional[str] = None         # e.g. "21 CFR 312.32"
    retrieved_chunk_id: Optional[str] = None           # traces back to the exact Pinecone chunk used (agent_2 flags only)
    impact: Optional[str] = None
    recommendation: Optional[str] = None
    severity: Literal["low", "medium", "high"] = "medium"

    # Confidence inputs — kept separate so compute_confidence() (sentinel_gcp/confidence/scoring.py)
    # can combine them per ARCHITECTURE.md §4, rather than trusting one bare number
    extraction_confidence: Optional[float] = None
    retrieval_score: Optional[float] = None
    llm_certainty: Optional[float] = None              # None for rule_engine flags — they're deterministic (implicitly 1.0)
    final_confidence: Optional[float] = None           # computed, filled in at generate_report time


class RuleResult(BaseModel):
    """One evaluation of a single rule against the current GraphState.
    Distinct from ComplianceFlag: a RuleResult exists even when the rule
    PASSES (no flag raised) — useful for the report's 'rule-engine checks
    run' summary (per the NEOD001 trace: '2/2 passed, 0 flags')."""
    rule_id: str
    passed: bool
    flag: Optional[ComplianceFlag] = None              # populated only if passed == False


class ContradictionFinding(BaseModel):
    """Output of both contradiction_check (early) and deep_contradiction_check (late).
    Kept as its own model rather than reusing ComplianceFlag — a contradiction
    is 'protocol vs itself', not 'protocol vs regulation', so it doesn't
    naturally carry a regulation_reference."""
    description: str
    section_refs: List[str] = Field(default_factory=list)   # e.g. ["9.6", "12.3"]
    severity: Literal["low", "medium", "high"] = "medium"
    check_stage: Literal["early", "deep"]