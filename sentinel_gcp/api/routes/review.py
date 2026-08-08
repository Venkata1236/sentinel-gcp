"""
api/routes/review.py — GET/POST /review/{run_id} endpoints.

GET: reads a run's current workflow status (via RunStatusStore) and, if
paused, the compliance findings a human needs to review (reconstructed
from LangGraph's checkpointed state via the graph's own state snapshot).

POST: accepts the human's decision(s), writes them into the graph's state,
and resumes execution — this is what actually continues the pipeline
past human_review_gate's interrupt point, using the SAME thread_id
(run_id) established back in analyze.py so LangGraph resumes THIS
specific paused run.

ASYNC STATE ACCESS REQUIRED: confirmed via real testing — with
AsyncPostgresSaver (see persistence/checkpointer.py), the synchronous
graph.get_state()/update_state() raise asyncio.InvalidStateError from
the main thread ("Synchronous calls to AsyncPostgresSaver are only
allowed from a different thread"). Must use graph.aget_state() and
graph.aupdate_state() throughout — same async-only requirement that
drove the earlier switch to AsyncPostgresSaver in the first place.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from sentinel_gcp.api.dependencies import get_graph
from sentinel_gcp.persistence.run_status_store import RunStatusStore

logger = logging.getLogger(__name__)
router = APIRouter()

_status_store = RunStatusStore()


class FlagDecision(BaseModel):
    flag_id: str
    decision: str  # "approve" | "reject" | "comment"
    comment: str | None = None


class ReviewSubmission(BaseModel):
    decisions: list[FlagDecision]


@router.get("/review/{run_id}")
async def get_review(run_id: str, graph=Depends(get_graph)):
    status_entry = _status_store.get_latest_status(run_id)
    if status_entry is None:
        raise HTTPException(status_code=404, detail=f"No run found with run_id={run_id}")

    workflow_status = status_entry["status"]

    if workflow_status == "FAILED":
        return {
            "run_id": run_id,
            "status": "FAILED",
            "detail": status_entry.get("detail"),
            "flags": [],
        }

    if workflow_status in ("PENDING", "RUNNING"):
        return {
            "run_id": run_id,
            "status": workflow_status,
            "message": "Still processing — check back shortly.",
            "flags": [],
        }

    if workflow_status == "COMPLETED":
        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "message": "Review already completed. See GET /report/{run_id} for the final report.",
            "flags": [],
        }

    # PAUSED — the real case this endpoint exists for. Read the graph's
    # checkpointed state directly to get the actual findings.
    config = {"configurable": {"thread_id": run_id}}
    snapshot = await graph.aget_state(config)

    if snapshot is None or snapshot.values is None:
        raise HTTPException(
            status_code=500,
            detail=f"Run {run_id} status is PAUSED but no checkpoint state was found — "
                   f"this indicates an inconsistency between RunStatusStore and the "
                   f"Postgres checkpointer that needs investigation",
        )

    state = snapshot.values
    rule_results = state.get("rule_results", [])
    agent_2_flags = state.get("agent_2_flags", [])
    rule_flags_only = [r.flag for r in rule_results if not r.passed]
    all_flags = rule_flags_only + agent_2_flags

    return {
        "run_id": run_id,
        "status": "PAUSED",
        "trial_identifier": state["extraction"].metadata.trial_identifier.value if state.get("extraction") else None,
        "jurisdiction": state.get("jurisdiction"),
        "flags": [f.model_dump() if hasattr(f, "model_dump") else f for f in all_flags],
        "contradictions": [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in state.get("early_contradiction_findings", []) + state.get("deep_contradiction_findings", [])
        ],
    }


@router.post("/review/{run_id}")
async def submit_review(run_id: str, submission: ReviewSubmission, graph=Depends(get_graph)):
    status_entry = _status_store.get_latest_status(run_id)
    if status_entry is None:
        raise HTTPException(status_code=404, detail=f"No run found with run_id={run_id}")
    if status_entry["status"] != "PAUSED":
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is not awaiting review (current status: {status_entry['status']})",
        )

    config = {"configurable": {"thread_id": run_id}}
    snapshot = await graph.aget_state(config)
    if snapshot is None:
        raise HTTPException(status_code=500, detail=f"No checkpoint state found for {run_id}")

    # Build the flag_id -> full flag snapshot map, needed so
    # record_feedback (node 13) has the complete flag content, not just
    # the human's raw decision, when it writes to EvalStore.
    state = snapshot.values
    rule_flags = {r.flag.flag_id: r.flag for r in state.get("rule_results", []) if not r.passed}
    agent_2_flags = {f.flag_id: f for f in state.get("agent_2_flags", [])}
    all_flags_by_id = {**rule_flags, **agent_2_flags}

    human_decisions = []
    for decision in submission.decisions:
        flag = all_flags_by_id.get(decision.flag_id)
        if flag is None:
            raise HTTPException(
                status_code=400,
                detail=f"flag_id '{decision.flag_id}' does not exist on run {run_id}",
            )
        human_decisions.append({
            "flag_id": decision.flag_id,
            "flag_snapshot": flag.model_dump() if hasattr(flag, "model_dump") else flag,
            "decision": decision.decision,
            "comment": decision.comment,
        })

    # Update state with the human's decisions, then resume the graph
    # from exactly where it paused — LangGraph's update_state + None
    # input to ainvoke is the documented pattern for resuming after
    # an interrupt_before pause.
    await graph.aupdate_state(config, {"human_decisions": human_decisions})

    _status_store.set_status(run_id, "RUNNING", detail="Resumed after human review decision")
    result = await graph.ainvoke(None, config=config)

    final_status = result.get("status", "unknown")
    if final_status == "complete":
        _status_store.set_status(run_id, "COMPLETED")
    else:
        _status_store.set_status(run_id, "FAILED", detail=f"Unexpected post-resume status: {final_status}")

    logger.info(f"submit_review: {run_id} resumed and reached status={final_status}")

    return {
        "run_id": run_id,
        "status": final_status,
        "message": "Review submitted. See GET /report/{run_id} for the final report." if final_status == "complete" else "Unexpected outcome after resume — check logs.",
    }