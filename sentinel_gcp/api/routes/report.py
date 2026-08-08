"""
api/routes/report.py — GET /report/{run_id} endpoint.

Serves the final report for a COMPLETED run by reading
state['final_report'] directly — generate_report (node 14) is the ONLY
place that computes this report; this endpoint no longer duplicates
that logic (see the fix applied after code review flagged the
duplication risk).
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
    snapshot = await graph.aget_state(config)
    if snapshot is None or snapshot.values is None:
        raise HTTPException(
            status_code=500,
            detail=f"Run {run_id} status is COMPLETED but no checkpoint state was found",
        )

    final_report = snapshot.values.get("final_report")
    if final_report is None:
        raise HTTPException(
            status_code=500,
            detail=f"Run {run_id} is COMPLETED but final_report was never written to state — "
                   f"this indicates generate_report ran without producing output, needs investigation",
        )

    logger.info(f"get_report: served final report for {run_id}")
    return final_report