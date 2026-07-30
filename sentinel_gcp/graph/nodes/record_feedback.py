"""
record_feedback — Node 13 of the Sentinel-GCP pipeline.

Deterministic (no LLM call). Runs immediately after the graph resumes
from human_review_gate's pause. Reads whatever decisions came in via
POST /review/{run_id} (state['human_decisions'], populated by the API
layer before resuming the graph — not by this node), and writes each
one to the eval store as a permanent labeled example.
"""
import logging

from sentinel_gcp.persistence.eval_store import EvalStore
from sentinel_gcp.graph.state import GraphState

logger = logging.getLogger(__name__)

_eval_store = EvalStore()


def record_feedback(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['human_decisions'] (a list
    of dicts, one per flag the human reviewed — populated by the API
    route handling POST /review/{run_id} BEFORE resuming the graph, not
    by this node itself). Writes each to the eval store."""
    decisions = state["human_decisions"]

    if not decisions:
        logger.warning(
            "record_feedback: no human_decisions found in state — "
            "this node should only run after a real review decision was submitted"
        )
        state["status"] = "complete"
        return state

    trial_identifier = (
        state["extraction"].metadata.trial_identifier.value if state["extraction"] else None
    )
    run_id = state.get("run_id", "unknown")  # NOTE: run_id isn't yet a GraphState field —
                                                # flagged below, needs adding

    recorded_count = 0
    for decision in decisions:
        _eval_store.record_decision(
            run_id=run_id,
            trial_identifier=trial_identifier,
            flag_id=decision["flag_id"],
            flag_snapshot=decision["flag_snapshot"],
            human_decision=decision["decision"],
            human_comment=decision.get("comment"),
        )
        recorded_count += 1

    state["status"] = "complete"
    logger.info(f"record_feedback: {recorded_count} decision(s) recorded to eval store")
    return state