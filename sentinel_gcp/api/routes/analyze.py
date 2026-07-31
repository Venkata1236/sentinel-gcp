"""
api/routes/analyze.py — POST /analyze endpoint.

Accepts a PDF upload, saves it to disk, generates a run_id, and kicks off
an async LangGraph run. Per ARCHITECTURE.md, the response returns
immediately with the run_id — the actual pipeline execution (including
the eventual human_review_gate pause) happens in the background, not
within this request/response cycle.

Workflow status (PENDING/RUNNING/PAUSED/FAILED/COMPLETED) is tracked via
RunStatusStore, independent of LangGraph's own checkpoint state — this
is what lets GET /review/{run_id} answer reliably even if the background
task died with an unhandled exception (see RunStatusStore's docstring).
"""
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks

from sentinel_gcp.graph.state import initial_state
from sentinel_gcp.api.dependencies import get_graph
from sentinel_gcp.persistence.run_status_store import RunStatusStore

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_status_store = RunStatusStore()


@router.post("/analyze")
async def analyze_protocol(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    graph=Depends(get_graph),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted — see ARCHITECTURE.md's documented "
                   "scope boundary on supported input types",
        )

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    saved_path = UPLOAD_DIR / f"{run_id}.pdf"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"analyze_protocol: {run_id} — saved upload as {saved_path}")
    _status_store.set_status(run_id, "PENDING", detail="Upload received, queued for processing")

    state = initial_state(raw_pdf_path=str(saved_path))
    state["run_id"] = run_id

    background_tasks.add_task(_run_graph_background, graph, state, run_id)

    return {
        "run_id": run_id,
        "status": "PENDING",
        "message": "Analysis started. Poll GET /review/{run_id} for status and findings.",
    }


async def _run_graph_background(graph, state: dict, run_id: str):
    """Invokes the graph with a thread_id matching run_id — this is what
    lets LangGraph's Postgres checkpointer later resume THIS SPECIFIC run
    when a human decision arrives via POST /review/{run_id}, rather than
    starting a fresh, disconnected run.

    Wraps execution with explicit RunStatusStore transitions, including
    the FAILED case that was previously only logged and invisible to
    any client polling for status."""
    config = {"configurable": {"thread_id": run_id}}
    _status_store.set_status(run_id, "RUNNING")

    try:
        result = await graph.ainvoke(state, config=config)
        final_status = result.get("status", "unknown")

        if final_status == "reviewing":
            _status_store.set_status(run_id, "PAUSED", detail="Awaiting human review decision")
        elif final_status == "complete":
            _status_store.set_status(run_id, "COMPLETED")
        elif final_status == "needs_human":
            _status_store.set_status(run_id, "FAILED", detail="Extraction failed validation after retry — needs manual document review")
        else:
            _status_store.set_status(run_id, "FAILED", detail=f"Graph ended in unexpected state: {final_status}")

        logger.info(f"_run_graph_background: {run_id} reached status={final_status}")

    except Exception as e:
        _status_store.set_status(run_id, "FAILED", detail=f"Unhandled exception: {str(e)}")
        logger.exception(f"_run_graph_background: {run_id} failed with an unhandled exception")