"""
api/routes/analyze.py — POST /analyze endpoint.

Accepts a PDF upload, saves it to disk, generates a run_id, and kicks off
an async LangGraph run. Per ARCHITECTURE.md, the response returns
immediately with the run_id — the actual pipeline execution (including
the eventual human_review_gate pause) happens in the background, not
within this request/response cycle.
"""
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks

from sentinel_gcp.graph.state import initial_state
from sentinel_gcp.api.dependencies import get_graph

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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

    state = initial_state(raw_pdf_path=str(saved_path))
    state["run_id"] = run_id

    # Run the graph in the background — this request returns immediately
    # with the run_id; the caller polls GET /review/{run_id} to check
    # progress, since a full run (including the eventual human pause)
    # is far too long-lived for a single synchronous HTTP request.
    background_tasks.add_task(_run_graph_background, graph, state, run_id)

    return {
        "run_id": run_id,
        "status": "extracting",
        "message": "Analysis started. Poll GET /review/{run_id} for status and findings.",
    }


async def _run_graph_background(graph, state: dict, run_id: str):
    """Invokes the graph with a thread_id matching run_id — this is what
    lets LangGraph's Postgres checkpointer later resume THIS SPECIFIC run
    when a human decision arrives via POST /review/{run_id}, rather than
    starting a fresh, disconnected run."""
    config = {"configurable": {"thread_id": run_id}}
    try:
        await graph.ainvoke(state, config=config)
        logger.info(f"_run_graph_background: {run_id} reached a pause or completion point")
    except Exception:
        logger.exception(f"_run_graph_background: {run_id} failed with an unhandled exception")
        # NOTE: a real production system would want to write a failure
        # status somewhere queryable here (e.g. a dedicated status table),
        # not just log it — flagged as a gap, not fixed in this pass.