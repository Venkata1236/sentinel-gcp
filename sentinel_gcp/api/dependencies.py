"""
api/dependencies.py — FastAPI dependency injection helpers.

Route handlers (api/routes/*.py) use get_graph() to access the compiled
LangGraph instance that api/main.py's lifespan sets up at startup —
rather than importing compiled_graph_holder directly, keeping routes
decoupled from main.py's internal structure.
"""
from fastapi import HTTPException


def get_graph():
    from sentinel_gcp.api.main import compiled_graph_holder

    graph = compiled_graph_holder["graph"]
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not yet initialized — server may still be starting up",
        )
    return graph