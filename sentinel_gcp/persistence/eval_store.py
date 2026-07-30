"""
EvalStore — writes human review decisions as labeled ground-truth
examples, growing the evaluation set with every real run. This is the
feedback loop referenced throughout ARCHITECTURE.md — corrections aren't
a dead end, they become future eval data.

Kept separate from record_feedback.py (the graph node) so the actual
storage logic is testable without any GraphState/LangGraph involved —
same separation-of-concerns pattern used for rules/engine.py.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# For now, a simple append-only JSONL file — one labeled example per line.
# Matches eval/ground_truth/'s existing structure (per the file-structure
# plan) rather than requiring Postgres to be stood up before this can
# work at all. Migrating to a real Postgres table later is a storage-layer
# swap, same interface, same reasoning as the FAISS->Pinecone abstraction.
DEFAULT_EVAL_STORE_PATH = Path("eval/ground_truth/feedback_log.jsonl")


class EvalStore:
    def __init__(self, path: Path = DEFAULT_EVAL_STORE_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record_decision(
        self,
        run_id: str,
        trial_identifier: str | None,
        flag_id: str,
        flag_snapshot: dict,
        human_decision: str,  # "approve" | "reject" | "comment"
        human_comment: str | None,
    ) -> str:
        """Writes one labeled example — a single flag plus what a human
        decided about it. One human review of a run can produce multiple
        calls to this (one per flag reviewed)."""
        entry_id = f"feedback-{uuid.uuid4().hex[:12]}"
        entry = {
            "entry_id": entry_id,
            "run_id": run_id,
            "trial_identifier": trial_identifier,
            "flag_id": flag_id,
            "flag_snapshot": flag_snapshot,
            "human_decision": human_decision,
            "human_comment": human_comment,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"EvalStore: recorded {entry_id} — {human_decision} on flag {flag_id}")
        return entry_id