"""
api/routes/report.py — GET /report/{run_id} endpoint.

Serves the final report for a COMPLETED run — reconstructs it from
LangGraph's checkpointed state, the same underlying data
generate_report (node 14) computed and logged, since that node doesn't
currently write the report anywhere queryable itself (see the note below).
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from sentinel_gcp.api.dependencies import get_graph
from sentinel_gcp.persistence.run_status_store import RunStatusStore

logger = logging.getLogger(__name__)
router = APIRouter()

_status_store = RunStatusStore()


@router.get("/report/{run_id}")
async def get_report(run_id: str, graph=Depends(get_graph)):
    status_entry = _status_store.get_latest_status(run_id)
    if status_entry is None:
        raise HTTPException(status_code=404, detail=f"No run found with run_id={run_id}")

    if status_entry["status"] == "FAILED":
        raise HTTPException(
            status_code=422,
            detail=f"Run {run_id} failed and has no report: {status_entry.get('detail')}",
        )
    if status_entry["status"] in ("PENDING", "RUNNING", "PAUSED"):
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is not yet complete (current status: {status_entry['status']}). "
                   f"Check GET /review/{run_id} for current progress.",
        )

    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if snapshot is None or snapshot.values is None:
        raise HTTPException(
            status_code=500,
            detail=f"Run {run_id} status is COMPLETED but no checkpoint state was found — "
                   f"inconsistency between RunStatusStore and the Postgres checkpointer",
        )

    state = snapshot.values
    extraction = state.get("extraction")
    rule_results = state.get("rule_results", [])
    agent_2_flags = state.get("agent_2_flags", [])
    human_decisions = state.get("human_decisions", [])

    rule_flags_only = [r.flag for r in rule_results if not r.passed]
    all_flags = rule_flags_only + agent_2_flags
    decisions_by_flag = {d["flag_id"]: d for d in human_decisions}

    report = {
        "run_id": run_id,
        "trial_identifier": extraction.metadata.trial_identifier.value if extraction else None,
        "phase": extraction.metadata.phase_raw if extraction else None,
        "sponsor": extraction.metadata.sponsor.value if extraction else None,
        "jurisdiction": state.get("jurisdiction"),
        "retry_count": state.get("retry_count", 0),
        "rule_engine_summary": {
            "checks_run": len(rule_results),
            "checks_passed": len([r for r in rule_results if r.passed]),
            "flags_raised": len(rule_flags_only),
        },
        "agent_2_summary": {
            "flags_raised": len(agent_2_flags),
        },
        "contradiction_summary": {
            "early_check_findings": len(state.get("early_contradiction_findings", [])),
            "deep_check_findings": len(state.get("deep_contradiction_findings", [])),
        },
        "flags": [
            {
                **(f.model_dump() if hasattr(f, "model_dump") else f),
                "human_decision": decisions_by_flag.get(
                    f.flag_id if hasattr(f, "flag_id") else f.get("flag_id"), {}
                ).get("decision", "not_reviewed"),
            }
            for f in all_flags
        ],
        "contradictions": [
            c.model_dump() if hasattr(c, "model_dump") else c
            for c in state.get("early_contradiction_findings", []) + state.get("deep_contradiction_findings", [])
        ],
    }

    logger.info(f"get_report: served final report for {run_id}")
    return report