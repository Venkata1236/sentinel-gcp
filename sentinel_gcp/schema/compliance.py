"""
Compliance schema — shared output contract for the rule engine (node 8,
deterministic) and Agent 2 / compliance_check (node 10, LLM).

Both sources produce the SAME ComplianceFlag shape, distinguished only by
`source`. This matters: it's what lets generate_report (and a human
reviewer) tell "the system is certain this is missing" (source=rule_engine,
confidence=1.0) apart from "the model made a judgment call about ambiguous
wording" (source=agent_2, confidence=weighted/uncertain) — see
ARCHITECTURE.md §4 for the confidence formula this feeds into.

A model_validator enforces that the two sources can't produce
cross-contaminated fields — a rule_engine flag can never carry an
llm_certainty (it has none, by definition), and an agent_2 flag must
carry both llm_certainty and a retrieved_chunk_id, since an LLM finding
with no cited evidence source is exactly the kind of ungrounded claim
this whole schema exists to prevent.
"""
from pydantic import BaseModel, Field, model_validator
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
    llm_certainty: Optional[float] = None              # rule_engine flags must NOT set this — enforced below
    final_confidence: Optional[float] = None           # computed, filled in at generate_report time
    insufficient_evidence: bool = False
    
    @model_validator(mode="after")
    def check_source_consistency(self) -> "ComplianceFlag":
        if self.source == "rule_engine":
            if self.llm_certainty is not None:
                raise ValueError(
                    "rule_engine flags are deterministic and must not carry llm_certainty "
                    "(implicitly 1.0 — there's no model judgment to score)"
                )
            if self.retrieved_chunk_id is not None:
                raise ValueError(
                    "rule_engine flags check canonical fields directly, not retrieved "
                    "regulation chunks — retrieved_chunk_id should be unset"
                )
        elif self.source == "agent_2":
            if self.llm_certainty is None:
                raise ValueError(
                    "agent_2 flags must report llm_certainty — an LLM finding with no "
                    "stated certainty can't be scored by compute_confidence()"
                )
            if self.retrieved_chunk_id is None:
                raise ValueError(
                    "agent_2 flags must cite a retrieved_chunk_id — an LLM finding with "
                    "no evidence source is exactly the ungrounded-claim case this schema "
                    "exists to prevent (see ARCHITECTURE.md §6, groundedness)"
                )
        return self


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