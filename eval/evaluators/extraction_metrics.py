"""
eval/evaluators/extraction_metrics.py — precision/recall/F1 for Agent 1
extraction, measured against hand-labeled ground truth.

Compares a real ProtocolExtraction output against a ground-truth
ProtocolExtraction field by field, using MICRO-averaging — every
individual comparison (each scalar field, each list item) contributes
one true-positive/false-positive/false-negative count to a SHARED pool,
rather than computing scalar accuracy and list accuracy as two separate
scores and blending them equally. This means fields/items are weighted
by how many of them actually exist, not by category — a protocol with
20 inclusion criteria and 3 scalar fields correctly has the list
comparisons dominate the overall score, not get diluted 50/50 against
just 3 scalars. This was a genuine design fix, not stylistic — the
prior 50/50 category-average approach under-weighted list fields
whenever they had more items than there were scalar fields.

List-field comparison uses Counter (multiset), not set — duplicate
predicted or expected items are counted correctly rather than collapsed
to one. If Agent 1 hallucinates the same inclusion criterion twice, that
now shows up as a real precision hit instead of being silently absorbed.

KNOWN LIMITATION, still not fixed (needs real protocol-testing evidence
before deciding it's worth the complexity): field/list-item comparison
is exact string match after light normalization, not semantic. A
predicted criterion that's substantively correct but differently worded
than ground truth counts as wrong on both sides.
"""
import logging
from collections import Counter
from dataclasses import dataclass, field

from sentinel_gcp.schema.extraction import ProtocolExtraction

logger = logging.getLogger(__name__)

_SCALAR_FIELD_PATHS = [
    ("metadata.trial_identifier", True),
    ("metadata.sponsor", True),
    ("metadata.phase_raw", False),
    ("metadata.ind_number", True),
    ("metadata.eudract_number", True),
    ("primary_endpoint", False),
    ("sae_reporting_timeline", True),
]

_LIST_FIELD_PATHS = [
    "inclusion_criteria",
    "exclusion_criteria",
    "secondary_endpoints",
]


@dataclass
class FieldResult:
    field_path: str
    correct: bool
    predicted: str | None
    expected: str | None


@dataclass
class ExtractionMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    field_results: list[FieldResult] = field(default_factory=list)
    list_field_details: dict[str, dict] = field(default_factory=dict)


def evaluate_extraction(
    predicted: ProtocolExtraction,
    expected: ProtocolExtraction,
) -> ExtractionMetrics:
    """Compares predicted (real Agent 1 output) against expected
    (hand-labeled ground truth). All scalar and list-item comparisons
    feed one shared TP/FP/FN pool (micro-averaging) — see module
    docstring for why this replaced the earlier category-average design."""
    total_tp, total_fp, total_fn = 0, 0, 0
    field_results: list[FieldResult] = []

    for path, has_provenance in _SCALAR_FIELD_PATHS:
        pred_val = _normalize(_get_field_value(predicted, path, has_provenance))
        exp_val = _normalize(_get_field_value(expected, path, has_provenance))

        if pred_val is None and exp_val is None:
            continue  # neither side has this field — not counted, no signal either way
        elif pred_val == exp_val:
            total_tp += 1
            field_results.append(FieldResult(path, True, pred_val, exp_val))
        else:
            # Mismatch: counts as BOTH a false positive (wrong/extra value
            # predicted) and a false negative (correct value not produced) —
            # a single wrong scalar genuinely costs you on both axes.
            if pred_val is not None:
                total_fp += 1
            if exp_val is not None:
                total_fn += 1
            field_results.append(FieldResult(path, False, pred_val, exp_val))

    list_field_details = {}
    for path in _LIST_FIELD_PATHS:
        pred_list = getattr(predicted, path, [])
        exp_list = getattr(expected, path, [])
        tp, fp, fn, details = _compare_lists_multiset(pred_list, exp_list)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        list_field_details[path] = details

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ExtractionMetrics(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        field_results=field_results,
        list_field_details=list_field_details,
    )


def _get_field_value(extraction: ProtocolExtraction, path: str, has_provenance: bool) -> str | None:
    obj = extraction
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    if has_provenance:
        return getattr(obj, "value", None)
    return obj


def _normalize(value: str | None) -> str | None:
    return value.strip().lower() if value else None


def _compare_lists_multiset(predicted: list[str], expected: list[str]) -> tuple[int, int, int, dict]:
    """Counter-based (multiset) comparison — duplicates are counted
    correctly instead of collapsed by set(). If Agent 1 predicts the
    same criterion twice but it only appears once in ground truth,
    that extra duplicate now correctly counts as a false positive."""
    pred_counts = Counter(_normalize(p) for p in predicted)
    exp_counts = Counter(_normalize(e) for e in expected)

    true_positives = sum((pred_counts & exp_counts).values())  # per-item min(pred, expected)
    false_positives = sum(pred_counts.values()) - true_positives
    false_negatives = sum(exp_counts.values()) - true_positives

    precision = true_positives / sum(pred_counts.values()) if pred_counts else 0.0
    recall = true_positives / sum(exp_counts.values()) if exp_counts else 0.0

    details = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "predicted_count": len(predicted),
        "expected_count": len(expected),
        "matched_count": true_positives,
    }
    return true_positives, false_positives, false_negatives, details