"""
dev_checkpoint.py — development-only state persistence for
run_nodes_manual.py. NOT part of the runtime pipeline (that's what
Postgres checkpointing via graph/builder.py is for, once built) — this
is purely a cost/time optimization for manual node-by-node testing.

Real cost evidence that motivated this: repeatedly rerunning the full
script from Stage 1 to debug a downstream node (compliance_check,
retrieve, etc.) re-paid extract_fill's ~41,005-token full-document call
every single time — roughly $0.28 EACH rerun, purely to re-reach a node
that hadn't changed. Estimated 8-15+ full reruns in one session ($2.30-
$4.20 from extract_fill alone), compounding with repeated retrieve/
compliance_check calls to the ~$8 total reported.

Usage: save state after any expensive stage completes; load it back
when testing a LATER node, skipping every earlier (already-verified,
already-paid-for) stage entirely.
"""
import pickle
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("dev_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def save_checkpoint(state: dict, label: str):
    """Saves state under a human-readable label, e.g. 'after_stage4',
    'after_stage8'. Overwrites any existing checkpoint with the same
    label — checkpoints are meant to represent 'current best known-good
    state at this stage', not a full history."""
    path = CHECKPOINT_DIR / f"{label}.pkl"
    with open(path, "wb") as f:
        pickle.dump(state, f)
    logger.info(f"dev_checkpoint: saved '{label}' -> {path}")
    print(f"  💾 Checkpoint saved: {label}")


def load_checkpoint(label: str) -> dict | None:
    """Returns the saved state, or None if no checkpoint with this
    label exists yet — caller should fall back to running the real
    pipeline stages in that case."""
    path = CHECKPOINT_DIR / f"{label}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    logger.info(f"dev_checkpoint: loaded '{label}' from {path}")
    print(f"  📂 Checkpoint loaded: {label} (skipping earlier stages — $0 spent)")
    return state


def list_checkpoints():
    """Shows what's available, with save time — useful when you're not
    sure which checkpoint is current/fresh vs. stale from a while ago."""
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pkl"))
    if not checkpoints:
        print("  No checkpoints saved yet.")
        return
    print("  Available checkpoints:")
    for cp in checkpoints:
        mtime = datetime.fromtimestamp(cp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"    - {cp.stem} (saved {mtime})")