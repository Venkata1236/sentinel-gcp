"""
test_db_connection.py — one-off connectivity check for the
Postgres/Supabase checkpointer. Confirms DATABASE_URL in .env is
correct and LangGraph's required tables get created successfully.
Not part of the permanent pipeline — a throwaway verification script.

Uses the ASYNC checkpointer (same as the real API path via
graph.ainvoke()) rather than a separate sync-only smoke test — after a
real NotImplementedError was found when the sync PostgresSaver was
paired with FastAPI's async invocation path, this now tests the exact
same code the API actually runs, not a similar-but-different path.
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

from sentinel_gcp.persistence.checkpointer import get_checkpointer


async def main():
    print("Attempting to connect to database...")
    async with get_checkpointer() as checkpointer:
        print("SUCCESS: Connected and LangGraph tables initialized correctly.")


asyncio.run(main())