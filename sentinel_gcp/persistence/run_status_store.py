"""
RunStatusStore — persists a lightweight workflow status per run_id,
independent of LangGraph's own checkpointed state.

Why this exists separately from GraphState's 'status' field: GraphState
only reflects status WHILE a graph run is actively executing or paused
correctly. It has no way to represent "the background task died with an
unhandled exception" — at that point, nothing ever wrote a final status
into the checkpoint at all. This store exists specifically to catch that
case, plus give api/routes/* a fast, simple lookup that doesn't require
reading/reconstructing full LangGraph checkpoint state just to answer
"is this run still going?"

Same lightweight-JSONL-then-migrate-to-Postgres-later philosophy as
EvalStore (see persistence/eval_store.py) — deliberately not
over-engineered before it's proven necessary.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

WorkflowStatus = Literal["PENDING", "RUNNING", "PAUSED", "FAILED", "COMPLETED"]

DEFAULT_STATUS_STORE_PATH = Path("data/run_status.jsonl")


class RunStatusStore:
    def __init__(self, path: Path = DEFAULT_STATUS_STORE_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def set_status(
        self,
        run_id: str,
        status: WorkflowStatus,
        detail: Optional[str] = None,
    ) -> None:
        """Appends a new status entry. Deliberately append-only (like
        EvalStore) rather than update-in-place — this gives a free
        audit trail of every status transition a run went through,
        which is useful debugging signal on its own (e.g. seeing a run
        went PENDING -> RUNNING -> PAUSED -> RUNNING -> FAILED tells you
        it failed specifically after a human resumed it, not during
        the initial pass)."""
        entry = {
            "run_id": run_id,
            "status": status,
            "detail": detail,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"RunStatusStore: {run_id} -> {status}" + (f" ({detail})" if detail else ""))

    def get_latest_status(self, run_id: str) -> Optional[dict]:
        """Scans the log for the most recent entry matching run_id.
        Fine for now at low volume; if this file grows large enough to
        matter, this is a natural place to migrate to an indexed
        Postgres table instead — same interface, different backend,
        same pattern as the FAISS->Pinecone and EvalStore migration path."""
        if not self._path.exists():
            return None

        latest = None
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry["run_id"] == run_id:
                    latest = entry
        return latest