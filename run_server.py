"""
run_server.py — entrypoint for running the Sentinel-GCP API on Windows.

WHY THIS FILE EXISTS instead of just `uvicorn sentinel_gcp.api.main:app`:
psycopg's async driver cannot run under Windows' default
ProactorEventLoop. Two earlier attempts at fixing this both failed in
real testing:
  1. Setting asyncio.set_event_loop_policy() at the top of api/main.py
     — the CLI still used ProactorEventLoop.
  2. Setting the policy here, before `import uvicorn`, then calling
     uvicorn.run() — SAME error persisted, meaning something between
     that policy call and psycopg's connection attempt was not
     respecting it (uvicorn.run()/asyncio.run()'s implicit loop
     creation, exact mechanism unconfirmed).

This version stops relying on the global policy being respected at
all: it creates the event loop EXPLICITLY and runs the server on that
loop directly via loop.run_until_complete(), rather than going through
asyncio.run()/uvicorn.run()'s own loop-creation logic. This is more
surgical — we own the loop object from the start, so there's no
opportunity for something else to substitute a ProactorEventLoop
before psycopg needs it.

Run this instead of the bare `uvicorn` CLI command:
    python run_server.py
"""
import sys
import asyncio

import uvicorn


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print(f"Using event loop: {type(loop).__name__}")  # sanity check —
    # should print SelectorEventLoop, not ProactorEventLoop, before the
    # server starts. If this still says ProactorEventLoop, the policy
    # call itself isn't taking effect even at this explicit level —
    # that would mean the fix needs to go a layer deeper still.

    config = uvicorn.Config("sentinel_gcp.api.main:app", host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)

    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    main()