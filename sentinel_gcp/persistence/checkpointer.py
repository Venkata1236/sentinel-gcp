"""
persistence/checkpointer.py — real Postgres checkpointing for LangGraph.

This is what makes human_review_gate's pause (node 12) genuinely durable —
not an in-memory wait, but state actually persisted to Postgres, so the
pipeline can sit paused for hours or days and resume correctly even if
the API server restarts in between. Required by graph/builder.py's
compile_graph(), which was written assuming this file would exist.

Uses LangGraph's built-in AsyncPostgresSaver rather than a hand-rolled
table — it already handles the checkpoint serialization/versioning
LangGraph needs internally; we're not reinventing that.

ASYNC, NOT SYNC — confirmed via real testing: api/routes/analyze.py and
review.py exclusively call `await graph.ainvoke(...)`, which requires the
checkpointer to implement LangGraph's ASYNC interface (aget_tuple, aput,
etc.). The synchronous PostgresSaver class does NOT implement these — its
base class's default async methods just raise NotImplementedError. This
surfaced as a real failure the first time the compiled graph was actually
invoked via ainvoke() through the FastAPI layer: `NotImplementedError` at
AsyncPregelLoop.__aenter__ -> checkpointer.aget_tuple. Neither
test_db_connection.py (calls sync setup() directly) nor
run_nodes_manual.py (calls node functions directly, never touches the
compiled graph at all) ever exercised this path, which is exactly why it
went unnoticed until the API was tested end-to-end for the first time.
"""
import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_checkpointer():
    """Async context manager wrapping AsyncPostgresSaver's own connection
    lifecycle. Usage:
        async with get_checkpointer() as checkpointer:
            graph = compile_graph(checkpointer)
            result = await graph.ainvoke(...)

    On first-ever run against a fresh database, .setup() creates the
    checkpoint tables LangGraph needs — safe to call every time, it's a
    no-op if the tables already exist."""
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set — persistence/checkpointer.py requires "
            "a real Postgres connection string in .env. See .env.example."
        )

    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        await checkpointer.setup()
        logger.info("Postgres checkpointer initialized (async)")
        yield checkpointer