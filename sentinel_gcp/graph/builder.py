"""
graph/builder.py — assembles all 14 nodes into a runnable LangGraph
StateGraph, with the conditional routing that's been referenced
throughout every node's docstring in this build:
  - validate_schema failure routes to retry_extraction, but only once
    (retry_count check), otherwise routes to a needs_human terminal path
  - human_review_gate is where the graph genuinely pauses via
    interrupt_before, checkpointed to Postgres

This file makes real what every node docstring described as "the
graph's routing logic decides this, not this node" — that deferred
decision-making lives here.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings

from sentinel_gcp.graph.nodes.parse_pdf import parse_pdf
from sentinel_gcp.graph.nodes.extract_discovery import extract_discovery
from sentinel_gcp.graph.nodes.extract_fill import extract_fill
from sentinel_gcp.graph.nodes.validate_schema import validate_schema
from sentinel_gcp.graph.nodes.retry_extraction import retry_extraction
from sentinel_gcp.graph.nodes.contradiction_check import contradiction_check
from sentinel_gcp.graph.nodes.determine_jurisdiction import determine_jurisdiction
from sentinel_gcp.graph.nodes.rule_engine import rule_engine
from sentinel_gcp.graph.nodes.retrieve import retrieve
from sentinel_gcp.graph.nodes.compliance_check import compliance_check
from sentinel_gcp.graph.nodes.evidence_filter import evidence_filter
from sentinel_gcp.graph.nodes.deep_contradiction_check import deep_contradiction_check
from sentinel_gcp.graph.nodes.human_review_gate import human_review_gate
from sentinel_gcp.graph.nodes.record_feedback import record_feedback
from sentinel_gcp.graph.nodes.generate_report import generate_report


def _route_after_validation(state: GraphState) -> str:
    """The conditional edge every validate_schema docstring pointed to.
    Reads extraction_errors + retry_count to decide: proceed (valid),
    retry once (invalid, first attempt), or give up (invalid, already
    retried once)."""
    if not state["extraction_errors"]:
        return "contradiction_check"
    if state["retry_count"] < settings.MAX_EXTRACTION_RETRIES:
        return "retry_extraction"
    return "needs_human_exit"


def _needs_human_exit(state: GraphState) -> GraphState:
    """Terminal node for the early-exit path — a schema-invalid
    extraction that failed even after one retry. Distinct from
    human_review_gate (node 12): that node reviews COMPLIANCE FINDINGS
    on a successfully-extracted protocol; this path means extraction
    itself never succeeded, so there's nothing downstream to review —
    a human needs to look at the raw document directly."""
    state["status"] = "needs_human"
    return state


def build_graph():
    graph = StateGraph(GraphState)

    # ── Register all nodes ──────────────────────────────────
    graph.add_node("parse_pdf", parse_pdf)
    graph.add_node("extract_discovery", extract_discovery)
    graph.add_node("extract_fill", extract_fill)
    graph.add_node("validate_schema", validate_schema)
    graph.add_node("retry_extraction", retry_extraction)
    graph.add_node("needs_human_exit", _needs_human_exit)
    graph.add_node("contradiction_check", contradiction_check)
    graph.add_node("determine_jurisdiction", determine_jurisdiction)
    graph.add_node("rule_engine", rule_engine)
    graph.add_node("retrieve", retrieve)
    graph.add_node("compliance_check", compliance_check)
    graph.add_node("evidence_filter", evidence_filter)
    graph.add_node("deep_contradiction_check", deep_contradiction_check)
    graph.add_node("human_review_gate", human_review_gate)
    graph.add_node("record_feedback", record_feedback)
    graph.add_node("generate_report", generate_report)

    # ── Linear edges (nodes 1-3): parsing through first extraction attempt ──
    graph.set_entry_point("parse_pdf")
    graph.add_edge("parse_pdf", "extract_discovery")
    graph.add_edge("extract_discovery", "extract_fill")
    graph.add_edge("extract_fill", "validate_schema")

    # ── Conditional routing after validation (the retry/escalate branch) ──
    graph.add_conditional_edges(
        "validate_schema",
        _route_after_validation,
        {
            "contradiction_check": "contradiction_check",
            "retry_extraction": "retry_extraction",
            "needs_human_exit": "needs_human_exit",
        },
    )
    # retry_extraction always routes BACK to validate_schema for a second
    # check — this is the loop every retry_extraction docstring described
    graph.add_edge("retry_extraction", "validate_schema")
    graph.add_edge("needs_human_exit", END)

    # ── Linear edges (nodes 6-11): checking through deep contradiction ──
    graph.add_edge("contradiction_check", "determine_jurisdiction")
    graph.add_edge("determine_jurisdiction", "rule_engine")
    graph.add_edge("rule_engine", "retrieve")
    graph.add_edge("retrieve", "compliance_check")
    graph.add_edge("compliance_check", "evidence_filter")
    graph.add_edge("evidence_filter", "deep_contradiction_check")
    graph.add_edge("deep_contradiction_check", "human_review_gate")

    # ── Human-in-the-loop pause point (node 12) ──
    # This is interrupt_before, not a regular edge — LangGraph halts
    # execution BEFORE running human_review_gate's node function again
    # on resume, since by the time we resume, the human's decision is
    # already in state['human_decisions'] and there's nothing left for
    # this node to compute — it already ran once to produce the summary.
    graph.add_edge("human_review_gate", "record_feedback")
    graph.add_edge("record_feedback", "generate_report")
    graph.add_edge("generate_report", END)

    return graph


def compile_graph(checkpointer: PostgresSaver):
    """Compiles the graph with a real Postgres checkpointer and the
    interrupt configuration that makes human_review_gate a genuine,
    durable pause — not just a node that happens to log a summary."""
    graph = build_graph()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["record_feedback"],
        # Interrupting BEFORE record_feedback (not human_review_gate
        # itself) means human_review_gate still RUNS and produces its
        # summary + sets status='reviewing', then the graph halts —
        # exactly matching human_review_gate's own docstring: "this
        # node prepares what the human will see, then the graph pauses."
    )