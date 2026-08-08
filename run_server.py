"""
run_server.py — entrypoint for running the Sentinel-GCP API on Windows.

WHY THIS FILE EXISTS instead of just `uvicorn sentinel_gcp.api.main:app`:
Setting asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
at the top of api/main.py was NOT sufficient — confirmed via real
testing, psycopg still raised the same ProactorEventLoop error even
after that fix was in place and confirmed loaded (traceback line
numbers matched the updated file). The uvicorn CLI's own app-loading
and event-loop setup sequence does not reliably respect a policy change
made inside the app module it dynamically imports.

The fix that's actually documented to work for psycopg's async driver +
uvicorn + Windows: set the policy as the very FIRST thing that happens
in the process — before uvicorn itself is even imported — then call
uvicorn programmatically rather than via its CLI. This guarantees
nothing else gets a chance to establish a loop/policy first.

Run this instead of the bare `uvicorn` CLI command:
    python run_server.py
"""
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "sentinel_gcp.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # reload spawns a subprocess that wouldn't inherit
                        # this policy setting — keep off until this is
                        # confirmed working, revisit separately if reload
                        # is wanted for active development
    )