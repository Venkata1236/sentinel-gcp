"""
eval/failure_taxonomy.py — categorized failure logging across pipeline
runs. Turns "something went wrong" into a structured, countable record
of WHAT kind of thing went wrong, so patterns are visible across many
runs instead of only remembered anecdotally.

Categories below are drawn directly from real failures encountered
during this project's actual development (not speculative) — see
ARCHITECTURE.md's variance-coverage discussion and today's real
debugging session (Pinecone response-shape bug, Claude refusals,
markdown-fence JSON, absence-vs-not-extracted flag quality issue).
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

FAILURE_LOG_PATH = Path("eval/failure_taxonomy_log.jsonl")


class FailureCategory(str, Enum):
    # Parsing-stage failures
    PARSE_TIMEOUT_OR_HANG = "parse_timeout_or_hang"           # e.g. NEOD001's slow table
    PARSE_TABLE_MALFORMED = "parse_table_malformed"             # e.g. integer column keys
    PARSE_OCR_MISS = "parse_ocr_miss"                            # scanned page not caught

    # Extraction-stage failures
    EXTRACTION_INVALID_JSON = "extraction_invalid_json"          # markdown fences, truncation
    EXTRACTION_SCHEMA_VALIDATION_FAILED = "extraction_schema_validation_failed"
    EXTRACTION_MISSING_CRITICAL_FIELD = "extraction_missing_critical_field"

    # LLM call failures (any node)
    LLM_REFUSAL = "llm_refusal"                                    # stop_reason=refusal
    LLM_EMPTY_RESPONSE = "llm_empty_response"                       # empty content, non-refusal
    LLM_RATE_LIMIT = "llm_rate_limit"

    # Retrieval-stage failures
    RETRIEVAL_ZERO_RESULTS = "retrieval_zero_results"
    RETRIEVAL_WRONG_JURISDICTION = "retrieval_wrong_jurisdiction"    # filter leaked wrong content

    # Compliance-reasoning quality issues (not hard failures — output quality)
    COMPLIANCE_UNGROUNDED_CITATION = "compliance_ungrounded_citation"
    COMPLIANCE_ABSENCE_VS_NOT_EXTRACTED = "compliance_absence_vs_not_extracted"
    COMPLIANCE_REPORTING_RELATIONSHIP_CONFLATED = "compliance_reporting_relationship_conflated"

    # Infrastructure
    INFRA_PINECONE_SDK_MISMATCH = "infra_pinecone_sdk_mismatch"
    INFRA_ENCODING_MOJIBAKE = "infra_encoding_mojibake"
    INFRA_WINDOWS_PERMISSIONS = "infra_windows_permissions"        # symlink/WDAC-class issues

    OTHER = "other"


@dataclass
class FailureRecord:
    run_id: str
    node_name: str
    category: FailureCategory
    description: str
    document_name: str | None = None
    resolved: bool = False
    resolution_notes: str | None = None
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def record_failure(
    run_id: str,
    node_name: str,
    category: FailureCategory,
    description: str,
    document_name: str | None = None,
    resolved: bool = False,
    resolution_notes: str | None = None,
) -> FailureRecord:
    """Append-only log, same pattern as EvalStore/RunStatusStore. Call
    this from any node's exception handler / warning path to build a
    real, queryable history instead of only console logs that scroll
    away."""
    record = FailureRecord(
        run_id=run_id,
        node_name=node_name,
        category=category,
        description=description,
        document_name=document_name,
        resolved=resolved,
        resolution_notes=resolution_notes,
    )
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.__dict__) + "\n")
    logger.info(f"failure_taxonomy: recorded {category.value} in {node_name} (run={run_id})")
    return record


def summarize_failures(log_path: Path = FAILURE_LOG_PATH) -> dict:
    """Aggregates the failure log into counts per category — the actual
    output worth looking at periodically to see whether one category is
    disproportionately common (a signal that specific fix deserves more
    investment than a one-off patch)."""
    if not log_path.exists():
        return {"total": 0, "by_category": {}, "unresolved_count": 0}

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    by_category: dict[str, int] = {}
    unresolved = 0
    for r in records:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
        if not r.get("resolved", False):
            unresolved += 1

    return {
        "total": len(records),
        "by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
        "unresolved_count": unresolved,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = summarize_failures()
    print(f"\nFailure taxonomy summary:")
    print(f"  Total recorded: {summary['total']}")
    print(f"  Unresolved: {summary['unresolved_count']}")
    print(f"  By category:")
    for cat, count in summary["by_category"].items():
        print(f"    {cat}: {count}")