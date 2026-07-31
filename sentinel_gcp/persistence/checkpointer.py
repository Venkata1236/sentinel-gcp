"""
persistence/checkpointer.py — real Postgres checkpointing for LangGraph.

This is what makes human_review_gate's pause (node 12) genuinely durable —
not an in-memory wait, but state actually persisted to Postgres, so the
pipeline can sit paused for hours or days and resume correctly even if
the API server restarts in between. Required by graph/builder.py's
compile_graph(), which was written assuming this file would exist.

Uses LangGraph's built-in PostgresSaver rather than a hand-rolled table —
it already handles the checkpoint serialization/versioning LangGraph
needs internally; we're not reinventing that.
"""
import logging
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver

from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)


@contextmanager
def get_checkpointer():
    """Context manager wrapping PostgresSaver's own connection lifecycle.
    Usage:
        with get_checkpointer() as checkpointer:
            graph = compile_graph(checkpointer)
            result = graph.invoke(...)

    On first-ever run against a fresh database, .setup() creates the
    checkpoint tables LangGraph needs — safe to call every time, it's a
    no-op if the tables already exist."""
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set — persistence/checkpointer.py requires "
            "a real Postgres connection string in .env. See .env.example."
        )

    with PostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        checkpointer.setup()
        logger.info("Postgres checkpointer initialized")
        yield checkpointer