"""
human_review_gate — Node 12 of the Sentinel-GCP pipeline.

Interrupt node — not an LLM call, not deterministic logic in the usual
sense. This node's job is to assemble everything a human reviewer needs
to see, then LangGraph's interrupt_before mechanism (configured in
graph/builder.py, not here) actually pauses execution at this point.

The pause is durable: state is checkpointed to Postgres via
sentinel_gcp/persistence/checkpointer.py, so the pipeline can sit paused
for hours or days and resume correctly even if the API server restarts
in the meantime — this is why Postgres checkpointing (not in-memory
state) was a required architectural choice, not just a nice-to-have.

This node does NOT decide anything — it only packages the current
findings into a reviewable summary. The actual decision comes back via
POST /review/{run_id}, handled by record_feedback (node 13), not here.
"""
import logging

from sentinel_gcp.graph.state import GraphState

logger = logging.getLogger(__name__)


def human_review_gate(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads everything accumulated in state
    so far, writes a 'review_summary' the API layer's GET /review/{id}
    endpoint will serve to a human reviewer. Sets status to 'reviewing' —
    the graph.builder's interrupt_before=['human_review_gate'] config is
    what actually halts execution here; this node just prepares what
    the human will see when they check in."""
    rule_results = state["rule_results"]
    agent_2_flags = state["agent_2_flags"]
    early_findings = state["early_contradiction_findings"]
    deep_findings = state["deep_contradiction_findings"]

    rule_flags_only = [r.flag for r in rule_results if not r.passed]
    all_flags = rule_flags_only + agent_2_flags
    all_contradictions = early_findings + deep_findings

    summary = {
        "trial_identifier": state["extraction"].metadata.trial_identifier.value if state["extraction"] else None,
        "jurisdiction": state["jurisdiction"],
        "retry_count": state["retry_count"],
        "rule_checks_run": len(rule_results),
        "rule_checks_passed": len([r for r in rule_results if r.passed]),
        "total_flags": len(all_flags),
        "flags_by_source": {
            "rule_engine": len([f for f in all_flags if f.source == "rule_engine"]),
            "agent_2": len([f for f in all_flags if f.source == "agent_2"]),
        },
        "flags_by_severity": {
            "high": len([f for f in all_flags if f.severity == "high"]),
            "medium": len([f for f in all_flags if f.severity == "medium"]),
            "low": len([f for f in all_flags if f.severity == "low"]),
        },
        "contradiction_findings": len(all_contradictions),
        "flags": [f.model_dump() for f in all_flags],
        "contradictions": [c.model_dump() for c in all_contradictions],
    }

    state["status"] = "reviewing"
    # review_summary isn't a GraphState field per se — it's derived and
    # served directly by the API layer's GET /review/{id} route, which
    # reads the checkpointed state and reconstructs this same summary.
    # Logging it here for visibility into what a paused run looks like.
    logger.info(
        f"human_review_gate: PAUSED for review — {summary['total_flags']} flag(s) "
        f"({summary['flags_by_severity']}), {summary['contradiction_findings']} contradiction(s)"
    )
    logger.info(f"human_review_gate: full summary: {summary}")
    return state