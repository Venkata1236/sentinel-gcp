"""
validate_schema — Node 4 of the Sentinel-GCP pipeline.

Deterministic (no LLM call). Takes the raw dict extract_fill (node 3)
produced and attempts to parse it into an actual ProtocolExtraction
instance. This is the hard gate referenced throughout ARCHITECTURE.md —
a malformed extraction NEVER silently proceeds to contradiction_check,
rule_engine, or Agent 2. It either becomes a validated object, or the
graph routes to retry_extraction (node 5).

This node does not decide whether to retry — it only records what went
wrong. The graph's conditional edge (built in graph/builder.py) reads
state['extraction_errors'] after this node runs to decide the route.
"""
import logging

from pydantic import ValidationError

from sentinel_gcp.schema.extraction import ProtocolExtraction
from sentinel_gcp.graph.state import GraphState

logger = logging.getLogger(__name__)


def validate_schema(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['extraction'] (a raw dict
    from extract_fill), writes either a validated ProtocolExtraction
    back into state['extraction'], or populates state['extraction_errors']
    and leaves state['extraction'] as None for the retry/escalation path
    to handle."""
    raw_extraction = state["extraction"]

    if raw_extraction is None:
        # extract_fill already failed to produce parseable JSON at all —
        # no point attempting Pydantic validation on nothing.
        state["extraction_errors"] = ["extract_fill produced no parseable output (invalid JSON from the model)"]
        state["extraction"] = None
        state["status"] = "needs_human" if state["retry_count"] >= 1 else "validating"
        logger.warning("validate_schema: no extraction to validate — extract_fill returned invalid JSON")
        return state

    try:
        validated = ProtocolExtraction(**raw_extraction)
        state["extraction"] = validated
        state["extraction_errors"] = []
        state["status"] = "checking"  # ready to move on to contradiction_check
        logger.info(
            f"validate_schema: PASSED — trial_identifier="
            f"{validated.metadata.trial_identifier.value}"
        )
    except ValidationError as e:
        errors = [_format_validation_error(err) for err in e.errors()]
        state["extraction_errors"] = errors
        state["extraction"] = None  # explicitly clear — a partially-valid object is not safe to use downstream
        state["status"] = "needs_human" if state["retry_count"] >= 1 else "validating"
        logger.warning(f"validate_schema: FAILED — {len(errors)} error(s): {errors}")

    return state


def _format_validation_error(err: dict) -> str:
    """Turns a Pydantic error dict into a readable string — used both
    for logging and, critically, fed back into retry_extraction's
    stricter prompt so the model knows exactly what to fix, not just
    that something was wrong."""
    field_path = ".".join(str(loc) for loc in err["loc"])
    return f"{field_path}: {err['msg']}"