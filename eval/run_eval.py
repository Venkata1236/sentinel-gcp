"""
eval/run_eval.py — the actual evaluation entrypoint. Runs
extraction_metrics, retrieval_metrics, groundedness, applicability, and
calibration against real ground truth / real run output, producing one
consolidated report.

This is NOT part of the runtime pipeline — it's a separate, offline
process you run periodically (per ARCHITECTURE.md's evaluation
philosophy: after real usage accumulates, not on every single run) to
answer "is the system actually working well," with real numbers instead
of anecdotal spot-checks.

HONEST STATE as of writing this file: ground_truth/*.json files
(hand-labeled correct extractions for real protocols) don't exist yet —
only the process for creating them has been discussed. This script is
built and ready to run the moment those files exist; it will report
clearly what's missing rather than silently producing an empty or
misleading report.
"""
import json
import logging
from pathlib import Path

from eval.evaluators.extraction_metrics import evaluate_extraction
from eval.evaluators.retrieval_metrics import evaluate_retrieval_suite
from eval.evaluators.groundedness import evaluate_groundedness_suite
from eval.evaluators.applicability import evaluate_applicability_suite
from eval.evaluators.calibration import compute_calibration, load_calibration_data_from_eval_store
from eval.failure_taxonomy import summarize_failures

from sentinel_gcp.schema.extraction import ProtocolExtraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROUND_TRUTH_DIR = Path("eval/ground_truth")


def run_extraction_eval() -> dict:
    """Compares real extraction output against hand-labeled ground
    truth files. Requires PAIRS of files: {name}.json (ground truth)
    and a corresponding real extraction run's output — the latter isn't
    automated yet (would come from dev_checkpoints/ or a saved run log),
    so this currently reports what ground truth EXISTS, not a live
    comparison, until that wiring is built."""
    ground_truth_files = list(GROUND_TRUTH_DIR.glob("*.json"))

    if not ground_truth_files:
        logger.warning(
            "run_extraction_eval: no ground truth files found in "
            f"{GROUND_TRUTH_DIR} — extraction accuracy cannot be measured yet. "
            "See ARCHITECTURE.md's ground-truth creation process."
        )
        return {"status": "no_ground_truth", "results": []}

    logger.info(f"run_extraction_eval: found {len(ground_truth_files)} ground truth file(s)")
    # NOTE: actual comparison against a real extraction run is not yet
    # wired here — this reports availability, not live scores, until
    # dev_checkpoints/ (or a similar saved-run mechanism) is connected.
    return {
        "status": "ground_truth_available_not_yet_compared",
        "ground_truth_files": [f.stem for f in ground_truth_files],
    }


def run_calibration_eval() -> dict:
    """The one evaluator with a fully real, working data path right now
    — reads directly from EvalStore's feedback log."""
    pairs = load_calibration_data_from_eval_store()
    report = compute_calibration(pairs)
    return {
        "status": "insufficient_data" if report.insufficient_data else "ok",
        "total_flags_analyzed": report.total_flags_analyzed,
        "overall_calibration_error": report.overall_calibration_error,
        "buckets": [b.__dict__ for b in report.buckets],
    }


def run_failure_taxonomy_summary() -> dict:
    return summarize_failures()


def run_full_eval() -> dict:
    """Runs every evaluator that has real data available right now, and
    reports HONESTLY on the ones that don't yet — rather than silently
    skipping them or producing misleading placeholder numbers."""
    logger.info("run_eval: starting full evaluation pass")

    report = {
        "extraction": run_extraction_eval(),
        "calibration": run_calibration_eval(),
        "failure_taxonomy": run_failure_taxonomy_summary(),
        # retrieval, groundedness, applicability evaluators exist and are
        # tested (see their own modules), but require a curated set of
        # (query, expected_relevant_chunks) or (flag, chunk_text) pairs
        # that hasn't been assembled yet — same honest gap as extraction
        # ground truth. Listed explicitly so the report shows what's
        # NOT yet measured, not just what is.
        "retrieval": {"status": "not_yet_wired — needs curated relevance ground truth"},
        "groundedness": {"status": "not_yet_wired — needs real Agent 2 flags + their retrieved chunks saved from a run"},
        "applicability": {"status": "not_yet_wired — needs real Agent 2 flags + their source ProtocolExtraction saved from a run"},
    }

    logger.info("run_eval: evaluation pass complete")
    return report


if __name__ == "__main__":
    result = run_full_eval()
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))