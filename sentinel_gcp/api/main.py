"""
api/main.py — FastAPI application entry point for Sentinel-GCP.

Wires together the LangGraph pipeline (graph/builder.py) with the four
endpoints defined in ARCHITECTURE.md §2.1: POST /analyze, GET/POST
/review/{run_id}, GET /report/{run_id}. Route logic itself lives in
api/routes/ — this file just assembles the FastAPI app and includes them.

WINDOWS EVENT LOOP FIX — must run before anything else in this file,
before ANY async code executes: psycopg's async driver (used by
AsyncPostgresSaver, see persistence/checkpointer.py) cannot run under
Windows' default ProactorEventLoop — confirmed via real testing, raises
psycopg.InterfaceError at connection time. WindowsSelectorEventLoopPolicy
is the documented fix. This only exists on Windows (referencing it on
Linux/Mac would raise AttributeError), hence the platform guard. Set at
module level so it takes effect at IMPORT time, before uvicorn creates
the event loop it'll actually serve requests on — setting it later
(e.g. inside lifespan()) would be too late, the loop already exists by then.
"""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager

from sentinel_gcp.persistence.checkpointer import get_checkpointer
from sentinel_gcp.graph.builder import compile_graph
from sentinel_gcp.api.routes import analyze, review, report
from sentinel_gcp.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level holder for the compiled graph — set during app startup
# (lifespan below), read by route handlers via api/dependencies.py.
# Not a global in the naive sense: FastAPI's lifespan pattern is the
# documented way to manage a resource (here, the Postgres-backed
# compiled graph) that needs setup/teardown around the app's lifetime.
compiled_graph_holder: dict = {"graph": None, "checkpointer_cm": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Sentinel-GCP API — initializing Postgres checkpointer")
    checkpointer_cm = get_checkpointer()
    checkpointer = await checkpointer_cm.__aenter__()
    compiled_graph_holder["graph"] = compile_graph(checkpointer)
    compiled_graph_holder["checkpointer_cm"] = checkpointer_cm
    logger.info("Sentinel-GCP API ready")

    yield  # app runs here

    logger.info("Shutting down Sentinel-GCP API")
    if compiled_graph_holder["checkpointer_cm"]:
        await compiled_graph_holder["checkpointer_cm"].__aexit__(None, None, None)


app = FastAPI(
    title="Sentinel-GCP",
    description="AI-powered clinical trial protocol compliance analyzer",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(analyze.router, tags=["analyze"])
app.include_router(review.router, tags=["review"])
app.include_router(report.router, tags=["report"])


@app.get("/health")
def health_check():
    """Basic liveness check — does NOT verify Postgres/Pinecone
    connectivity, just confirms the process is up. A real production
    deployment would want a deeper readiness check too, not built yet."""
    return {"status": "ok", "graph_initialized": compiled_graph_holder["graph"] is not None}