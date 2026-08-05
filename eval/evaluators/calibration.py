"""
eval/evaluators/calibration.py — confidence calibration analysis.

Checks whether compute_confidence()'s scores (sentinel_gcp/confidence/
scoring.py) are actually MEANINGFUL, not just numbers. A well-calibrated
system's "90% confidence" flags should be agreed-with by a human
reviewer roughly 90% of the time — if flags rated 0.9 are only agreed
with 60% of the time, the confidence score is overstating certainty,
which is arguably worse than not having a confidence score at all,
since it creates false trust.

Ground truth here comes from record_feedback's output (persistence/
eval_store.py) — real human approve/reject/comment decisions on real
flags, accumulated over actual usage. This evaluator is only meaningful
once a real corpus of reviewed flags exists; on day one with zero
reviewed flags, it has nothing to calibrate against.
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Confidence values get bucketed into these ranges for the reliability
# diagram — narrow enough to be informative, wide enough that each
# bucket has enough flags to be statistically meaningful once you have
# real volume (not meaningful yet with only a handful of flags).
CONFIDENCE_BUCKETS = [
    (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)
]


@dataclass
class CalibrationBucket:
    range_low: float
    range_high: float
    predicted_avg_confidence: float
    actual_agreement_rate: float
    sample_count: int
    calibration_gap: float  # |predicted - actual| — 0.0 is perfectly calibrated


@dataclass
class CalibrationReport:
    buckets: list[CalibrationBucket] = field(default_factory=list)
    overall_calibration_error: float = 0.0  # mean absolute gap across all buckets
    total_flags_analyzed: int = 0
    insufficient_data: bool = False


def compute_calibration(
    flag_confidence_and_agreement: list[tuple[float, bool]],
) -> CalibrationReport:
    """Takes (final_confidence, human_agreed) pairs — human_agreed is
    True if the human's decision (approve) matched the flag being a
    real finding, False if the human rejected it as incorrect/not
    applicable. Bucketed comparison of predicted vs. actual reliability."""
    if len(flag_confidence_and_agreement) < 10:
        logger.warning(
            f"compute_calibration: only {len(flag_confidence_and_agreement)} "
            f"reviewed flag(s) available — too few for meaningful calibration "
            f"analysis. Need real usage volume via record_feedback before this "
            f"evaluator produces trustworthy output."
        )
        return CalibrationReport(insufficient_data=True, total_flags_analyzed=len(flag_confidence_and_agreement))

    buckets = []
    total_gap = 0.0
    bucket_count = 0

    for low, high in CONFIDENCE_BUCKETS:
        bucket_items = [
            (conf, agreed) for conf, agreed in flag_confidence_and_agreement
            if low <= conf < high
        ]
        if not bucket_items:
            continue

        predicted_avg = sum(conf for conf, _ in bucket_items) / len(bucket_items)
        actual_rate = sum(1 for _, agreed in bucket_items if agreed) / len(bucket_items)
        gap = abs(predicted_avg - actual_rate)

        buckets.append(CalibrationBucket(
            range_low=low, range_high=high,
            predicted_avg_confidence=round(predicted_avg, 3),
            actual_agreement_rate=round(actual_rate, 3),
            sample_count=len(bucket_items),
            calibration_gap=round(gap, 3),
        ))
        total_gap += gap
        bucket_count += 1

    overall_error = total_gap / bucket_count if bucket_count else 0.0

    if overall_error > 0.15:
        logger.warning(
            f"compute_calibration: overall calibration error {overall_error:.3f} is high — "
            f"confidence scores are meaningfully overstating or understating real reliability. "
            f"Consider reweighting compute_confidence()'s CONFIDENCE_WEIGHT_* settings."
        )

    return CalibrationReport(
        buckets=buckets,
        overall_calibration_error=round(overall_error, 3),
        total_flags_analyzed=len(flag_confidence_and_agreement),
        insufficient_data=False,
    )


def load_calibration_data_from_eval_store(eval_store_path: str = "eval/ground_truth/feedback_log.jsonl") -> list[tuple[float, bool]]:
    """Reads real human decisions from EvalStore's JSONL log (see
    persistence/eval_store.py) and converts them into the
    (confidence, agreed) pairs this module needs. 'agreed' is True when
    the human approved the flag as a real finding, False on reject."""
    import json
    from pathlib import Path

    path = Path(eval_store_path)
    if not path.exists():
        logger.warning(f"load_calibration_data_from_eval_store: {path} not found — no feedback recorded yet")
        return []

    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            flag_snapshot = entry.get("flag_snapshot", {})
            confidence = flag_snapshot.get("final_confidence")
            decision = entry.get("human_decision")
            if confidence is not None and decision in ("approve", "reject"):
                pairs.append((confidence, decision == "approve"))
    return pairs