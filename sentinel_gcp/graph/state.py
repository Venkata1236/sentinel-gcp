"""
GraphState — the shared state object every LangGraph node reads from
and writes to as a protocol moves through the 14-stage pipeline.

This is intentionally one flat object, not 14 separate ones — each node
only writes the fields it's responsible for, but every node can read
anything written by an earlier node. LangGraph passes this same object
through the whole run; it's what makes conditional routing possible
(e.g. validate_schema checking extraction_errors before deciding whether
to route to retry_extraction or contradiction_check).

Field ownership by node (per ARCHITECTURE.md §2.2):
  run_id                 ← set once, at POST /analyze time
  raw_pdf_path            ← set once, at graph invocation
  document_structure       ← parse_pdf (node 1)
  extraction_discovery      ← extract_discovery (node 2)
  extraction                 ← extract_fill (node 3)
  extraction_errors           ← validate_schema (node 4)
  retry_count                  ← retry_extraction (node 5)
  early_contradiction_findings  ← contradiction_check (node 6)
  jurisdiction                   ← determine_jurisdiction (node 7)
  rule_results                     ← rule_engine (node 8)
  retrieved_chunks                  ← retrieve (node 9)
  agent_2_flags                      ← compliance_check (node 10)
  deep_contradiction_findings          ← deep_contradiction_check (node 11)
  human_decisions                       ← human_review_gate (node 12) / API layer
  final_report                            ← generate_report (node 14)
  status                                   ← updated by nearly every node
"""
from typing import TypedDict, Optional, List, Literal

from sentinel_gcp.schema.document_structure import DocumentStructure
from sentinel_gcp.schema.extraction import ProtocolExtraction
from sentinel_gcp.schema.compliance import ComplianceFlag, RuleResult, ContradictionFinding


GraphStatus = Literal[
    "extracting",
    "validating",
    "checking",
    "retrieving",
    "reasoning",
    "reviewing",
    "complete",
    "needs_human",
]


class GraphState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    run_id: str
    raw_pdf_path: str

    # ── Stage 1: parse_pdf ────────────────────────────────
    document_structure: Optional[DocumentStructure]

    # ── Stages 2-3: extract_discovery, extract_fill ───────
    extraction_discovery: Optional[dict]
    extraction: Optional[ProtocolExtraction]

    # ── Stage 4-5: validate_schema, retry_extraction ──────
    extraction_errors: List[str]
    retry_count: int

    # ── Stage 6: contradiction_check (early) ───────────────
    early_contradiction_findings: List[ContradictionFinding]

    # ── Stage 7: determine_jurisdiction ─────────────────────
    jurisdiction: Optional[Literal["FDA", "EMA", "both", "unknown"]]

    # ── Stage 8: rule_engine ─────────────────────────────────
    rule_results: List[RuleResult]

    # ── Stage 9: retrieve ─────────────────────────────────────
    retrieved_chunks: List[dict]

    # ── Stage 10: compliance_check (Agent 2) ───────────────────
    agent_2_flags: List[ComplianceFlag]

    # ── Stage 10B: evidence_filter ──────────────────────────────
    # Audit trail — findings evidence_filter rejected, with WHY. Kept
    # separate from agent_2_flags (which only holds survivors) so a
    # dropped finding isn't just silently gone; a human/report can see
    # what got filtered and the specific groundedness/applicability
    # reasoning behind it, not just a smaller final count.
    evidence_filter_dropped: List[dict]

    # ── Stage 11: deep_contradiction_check ──────────────────────
    deep_contradiction_findings: List[ContradictionFinding]

    # ── Stage 12-13: human_review_gate, record_feedback ─────────
    human_decisions: List[dict]

    # ── Stage 14: generate_report ────────────────────────────────
    final_report: Optional[dict]

    # ── Overall run status, updated throughout ───────────────────
    status: GraphStatus


def initial_state(raw_pdf_path: str, run_id: str) -> GraphState:
    """Factory for a fresh GraphState at the start of a run —
    every list starts empty, every optional starts None, so no node
    has to guess whether a field exists yet or handle a KeyError."""
    return GraphState(
        run_id=run_id,
        raw_pdf_path=raw_pdf_path,
        document_structure=None,
        extraction_discovery=None,
        extraction=None,
        extraction_errors=[],
        retry_count=0,
        early_contradiction_findings=[],
        jurisdiction=None,
        rule_results=[],
        retrieved_chunks=[],
        agent_2_flags=[],
        evidence_filter_dropped=[],
        deep_contradiction_findings=[],
        human_decisions=[],
        final_report=None,
        status="extracting",
    )