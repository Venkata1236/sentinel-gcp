"""
eval/evaluators/extraction_metrics.py — precision/recall/F1 for Agent 1
extraction, measured against hand-labeled ground truth.

Compares a real ProtocolExtraction output against a ground-truth
ProtocolExtraction (hand-labeled from a real protocol — see
eval/ground_truth/) field by field. Two kinds of fields need different
comparison logic:
  - Scalar fields (trial_identifier, phase_raw, sponsor, etc.) — exact
    match on .value, ignoring provenance (page/section/confidence aren't
    "correctness," they're metadata)
  - List fields (inclusion_criteria, exclusion_criteria, etc.) — treated
    as a set-comparison problem, since exact list-order matching is too
    strict for extracted text that may be phrased slightly differently
    even when substantively correct
"""
import logging
from dataclasses import dataclass, field

from sentinel_gcp.schema.extraction import ProtocolExtraction

logger = logging.getLogger(__name__)

# Fields compared as scalars — exact match on .value (or the bare value
# for non-FieldWithProvenance fields like phase_raw, primary_endpoint)
_SCALAR_FIELD_PATHS = [
    ("metadata.trial_identifier", True),   # True = FieldWithProvenance (has .value)
    ("metadata.sponsor", True),
    ("metadata.phase_raw", False),          # False = plain field
    ("metadata.ind_number", True),
    ("metadata.eudract_number", True),
    ("primary_endpoint", False),
    ("sae_reporting_timeline", True),
]

# Fields compared as sets of strings
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
    field_results: list[FieldResult] = field(default_factory=list)
    list_field_details: dict[str, dict] = field(default_factory=dict)  # per-list-field precision/recall


def evaluate_extraction(
    predicted: ProtocolExtraction,
    expected: ProtocolExtraction,
) -> ExtractionMetrics:
    """Compares predicted (real Agent 1 output) against expected
    (hand-labeled ground truth). Returns overall precision/recall/F1
    plus a per-field breakdown — the breakdown matters more than the
    single aggregate number for actually debugging WHICH kinds of
    fields are unreliable."""
    field_results: list[FieldResult] = []

    for path, has_provenance in _SCALAR_FIELD_PATHS:
        pred_val = _get_field_value(predicted, path, has_provenance)
        exp_val = _get_field_value(expected, path, has_provenance)
        correct = _normalize(pred_val) == _normalize(exp_val)
        field_results.append(FieldResult(path, correct, pred_val, exp_val))

    list_field_details = {}
    for path in _LIST_FIELD_PATHS:
        pred_list = getattr(predicted, path, [])
        exp_list = getattr(expected, path, [])
        list_field_details[path] = _compare_lists(pred_list, exp_list)

    correct_scalars = sum(1 for r in field_results if r.correct)
    total_scalars = len(field_results)

    # Aggregate precision/recall combines scalar-field accuracy with
    # list-field set-comparison — a simple average across both
    # categories, not weighted by field count, so one very long list
    # field doesn't dominate the score.
    scalar_accuracy = correct_scalars / total_scalars if total_scalars else 0.0
    list_precisions = [d["precision"] for d in list_field_details.values()]
    list_recalls = [d["recall"] for d in list_field_details.values()]
    avg_list_precision = sum(list_precisions) / len(list_precisions) if list_precisions else 0.0
    avg_list_recall = sum(list_recalls) / len(list_recalls) if list_recalls else 0.0

    precision = (scalar_accuracy + avg_list_precision) / 2
    recall = (scalar_accuracy + avg_list_recall) / 2
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ExtractionMetrics(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
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
    """Lowercase + strip whitespace before comparing — a scalar field
    that's substantively correct but differs in casing/whitespace
    shouldn't count as a miss."""
    return value.strip().lower() if value else None


def _compare_lists(predicted: list[str], expected: list[str]) -> dict:
    """Set-based comparison with normalized strings — order doesn't
    matter, and minor phrasing differences are tolerated via
    normalization, but this is still a strict comparison (no fuzzy/
    semantic matching). A known limitation: a predicted criterion
    that's substantively correct but worded differently from the
    ground truth will count as both a false positive AND a false
    negative — acceptable for now, worth revisiting with fuzzy matching
    if this turns out to understate real accuracy once run against
    real protocols."""
    pred_set = {_normalize(p) for p in predicted}
    exp_set = {_normalize(e) for e in expected}

    true_positives = len(pred_set & exp_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(exp_set) if exp_set else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "predicted_count": len(predicted),
        "expected_count": len(expected),
        "matched_count": true_positives,
    }