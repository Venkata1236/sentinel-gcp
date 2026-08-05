"""
eval/failure_taxonomy.py — categorized failure logging across pipeline
runs. Turns "something went wrong" into a structured, countable record
of WHAT kind of thing went wrong, so patterns are visible across many
runs instead of only remembered anecdotally.

Categories below are drawn directly from real failures encountered
during this project's actual development (not speculative).

IMPROVEMENTS (per code review — production engineering utility, not
part of the inference pipeline):
1. severity field (info/warning/error/critical) — prioritizes which
   failures actually need attention vs. informational noise.
2. exception_type + truncated traceback — faster debugging than a
   category label alone.
3. model/SDK version metadata — makes a failure like the Pinecone
   hits-vs-matches bug traceable to exactly which SDK version caused
   it, useful when a dependency upgrade reintroduces something similar.
4. Filtering (by node/category/date) via summarize_failures()'s new
   filter parameters — a lightweight CLI substitute, not a full
   dashboard (out of scope for this project's size).
5. Trend computation — summarize_failures() now supports comparing two
   time windows, e.g. "before/after a specific fix" (the exact Pinecone
   SDK fix example from the review).
"""
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

FAILURE_LOG_PATH = Path("eval/failure_taxonomy_log.jsonl")

TRACEBACK_TRUNCATE_CHARS = 2000  # full tracebacks can be huge; keep enough to debug, not the whole log file


class FailureSeverity(str, Enum):
    INFO = "info"           # expected/handled gracefully, logged for visibility only
    WARNING = "warning"     # degraded but recovered (e.g. fallback path used)
    ERROR = "error"         # this specific run's output is compromised
    CRITICAL = "critical"   # pipeline-breaking, needs immediate attention


class FailureCategory(str, Enum):
    PARSE_TIMEOUT_OR_HANG = "parse_timeout_or_hang"
    PARSE_TABLE_MALFORMED = "parse_table_malformed"
    PARSE_OCR_MISS = "parse_ocr_miss"

    EXTRACTION_INVALID_JSON = "extraction_invalid_json"
    EXTRACTION_SCHEMA_VALIDATION_FAILED = "extraction_schema_validation_failed"
    EXTRACTION_MISSING_CRITICAL_FIELD = "extraction_missing_critical_field"

    LLM_REFUSAL = "llm_refusal"
    LLM_EMPTY_RESPONSE = "llm_empty_response"
    LLM_RATE_LIMIT = "llm_rate_limit"

    RETRIEVAL_ZERO_RESULTS = "retrieval_zero_results"
    RETRIEVAL_WRONG_JURISDICTION = "retrieval_wrong_jurisdiction"

    COMPLIANCE_UNGROUNDED_CITATION = "compliance_ungrounded_citation"
    COMPLIANCE_ABSENCE_VS_NOT_EXTRACTED = "compliance_absence_vs_not_extracted"
    COMPLIANCE_REPORTING_RELATIONSHIP_CONFLATED = "compliance_reporting_relationship_conflated"

    INFRA_PINECONE_SDK_MISMATCH = "infra_pinecone_sdk_mismatch"
    INFRA_ENCODING_MOJIBAKE = "infra_encoding_mojibake"
    INFRA_WINDOWS_PERMISSIONS = "infra_windows_permissions"

    OTHER = "other"


@dataclass
class FailureRecord:
    run_id: str
    node_name: str
    category: FailureCategory
    description: str
    severity: FailureSeverity = FailureSeverity.ERROR
    document_name: str | None = None
    exception_type: str | None = None
    traceback_excerpt: str | None = None
    model_or_sdk_versions: dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: str | None = None
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def record_failure(
    run_id: str,
    node_name: str,
    category: FailureCategory,
    description: str,
    severity: FailureSeverity = FailureSeverity.ERROR,
    document_name: str | None = None,
    exception: Exception | None = None,
    model_or_sdk_versions: dict[str, str] | None = None,
    resolved: bool = False,
    resolution_notes: str | None = None,
) -> FailureRecord:
    """Append-only log. Pass the actual exception object (not just a
    string) when available — exception_type and a truncated traceback
    are extracted automatically, giving much faster debugging context
    than a category label alone."""
    exception_type = type(exception).__name__ if exception else None
    traceback_excerpt = None
    if exception is not None:
        full_tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        traceback_excerpt = full_tb[:TRACEBACK_TRUNCATE_CHARS]

    record = FailureRecord(
        run_id=run_id,
        node_name=node_name,
        category=category,
        description=description,
        severity=severity,
        document_name=document_name,
        exception_type=exception_type,
        traceback_excerpt=traceback_excerpt,
        model_or_sdk_versions=model_or_sdk_versions or {},
        resolved=resolved,
        resolution_notes=resolution_notes,
    )
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.__dict__, default=str) + "\n")
    logger.info(f"failure_taxonomy: recorded [{severity.value}] {category.value} in {node_name} (run={run_id})")
    return record


def _load_records(log_path: Path = FAILURE_LOG_PATH) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def summarize_failures(
    log_path: Path = FAILURE_LOG_PATH,
    node_filter: str | None = None,
    category_filter: FailureCategory | None = None,
    severity_filter: FailureSeverity | None = None,
    since: str | None = None,   # ISO date string, e.g. "2026-08-01"
) -> dict:
    """Aggregates the failure log into counts per category, with
    optional filters — a lightweight CLI substitute rather than a full
    dashboard (out of scope for this project's size, per review)."""
    records = _load_records(log_path)

    if node_filter:
        records = [r for r in records if r["node_name"] == node_filter]
    if category_filter:
        records = [r for r in records if r["category"] == category_filter.value]
    if severity_filter:
        records = [r for r in records if r["severity"] == severity_filter.value]
    if since:
        records = [r for r in records if r["recorded_at"] >= since]

    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    unresolved = 0
    for r in records:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
        by_severity[r.get("severity", "error")] = by_severity.get(r.get("severity", "error"), 0) + 1
        if not r.get("resolved", False):
            unresolved += 1

    return {
        "total": len(records),
        "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
        "by_severity": dict(sorted(by_severity.items(), key=lambda x: -x[1])),
        "unresolved_count": unresolved,
    }


def compare_trend(
    category: FailureCategory,
    before_date: str,
    after_date: str,
    log_path: Path = FAILURE_LOG_PATH,
) -> dict:
    """Compares failure counts for one category across two EXPLICIT
    time windows:
      before window: before_date <= recorded_at < after_date
      after window:  recorded_at >= after_date
    e.g. confirming 'retrieval failures dropped after the Pinecone SDK
    fix' with real numbers instead of a general impression. Previously
    before_date was accepted but never actually used, silently treating
    'before' as 'everything before after_date' with no lower bound —
    fixed to match the documented two-window comparison."""
    records = _load_records(log_path)
    category_records = [r for r in records if r["category"] == category.value]

    before_count = sum(
        1 for r in category_records
        if before_date <= r["recorded_at"] < after_date
    )
    after_count = sum(
        1 for r in category_records
        if r["recorded_at"] >= after_date
    )

    if before_count == 0:
        pct_change = None
    else:
        pct_change = round(((after_count - before_count) / before_count) * 100, 1)

    return {
        "category": category.value,
        "before_window": f"{before_date} to {after_date}",
        "after_window": f"{after_date} onward",
        "before_count": before_count,
        "after_count": after_count,
        "percent_change": pct_change,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = summarize_failures()
    print(f"\nFailure taxonomy summary:")
    print(f"  Total recorded: {summary['total']}")
    print(f"  Unresolved: {summary['unresolved_count']}")
    print(f"  By severity: {summary['by_severity']}")
    print(f"  By category:")
    for cat, count in summary["by_category"].items():
        print(f"    {cat}: {count}")