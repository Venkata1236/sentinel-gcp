"""
test_db_connection.py — one-off connectivity check for the
Postgres/Supabase checkpointer. Confirms DATABASE_URL in .env is
correct and LangGraph's required tables get created successfully.
Not part of the permanent pipeline — a throwaway verification script.
"""
import logging

logging.basicConfig(level=logging.INFO)

from sentinel_gcp.persistence.checkpointer import get_checkpointer

print("Attempting to connect to database...")

with get_checkpointer() as checkpointer:
    print("SUCCESS: Connected and LangGraph tables initialized correctly.")