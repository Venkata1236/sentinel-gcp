"""
determine_jurisdiction — Node 7 of the Sentinel-GCP pipeline.

Deterministic (no LLM call). Classifies a validated extraction as
FDA / EMA / both / unknown, based purely on whether ind_number and/or
eudract_number were found — no interpretation needed, just presence checks
on already-validated canonical fields.

This is the exact node that fixed the bug found during design against
OEV-125: a naive rule engine without jurisdiction gating would incorrectly
flag EVERY EU-only trial for "missing IND number" — this node exists so
rule_engine (node 8) and retrieve (node 9) can scope correctly instead.
"""
import logging

from sentinel_gcp.graph.state import GraphState

logger = logging.getLogger(__name__)


def determine_jurisdiction(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['extraction'] (a validated
    ProtocolExtraction), writes state['jurisdiction']."""
    extraction = state["extraction"]
    if extraction is None:
        raise ValueError(
            "determine_jurisdiction requires a validated ProtocolExtraction — "
            "this node should only be reached after validate_schema passes"
        )

    has_ind = _has_value(extraction.metadata.ind_number)
    has_eudract = _has_value(extraction.metadata.eudract_number)

    if has_ind and has_eudract:
        jurisdiction = "both"
    elif has_ind:
        jurisdiction = "FDA"
    elif has_eudract:
        jurisdiction = "EMA"
    else:
        jurisdiction = "unknown"

    state["jurisdiction"] = jurisdiction
    logger.info(
        f"determine_jurisdiction: {jurisdiction} "
        f"(ind_present={has_ind}, eudract_present={has_eudract})"
    )
    return state


def _has_value(field) -> bool:
    """A FieldWithProvenance object being non-None isn't enough on its
    own — extract_fill could have returned an IND field object with
    every sub-field (including .value) set to null. Only a real,
    non-empty .value counts as 'present'."""
    return field is not None and field.value is not None and field.value.strip() != ""