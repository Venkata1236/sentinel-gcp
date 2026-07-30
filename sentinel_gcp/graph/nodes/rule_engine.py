"""
rule_engine — Node 8 of the Sentinel-GCP pipeline.

Deterministic (no LLM call). Thin LangGraph wrapper around
sentinel_gcp/rules/engine.py's run_rules() — kept separate so the actual
rule execution logic is independently unit-testable without any
GraphState/LangGraph machinery involved.
"""
import logging

from sentinel_gcp.rules.engine import run_rules
from sentinel_gcp.graph.state import GraphState

logger = logging.getLogger(__name__)


def rule_engine(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['extraction'] and
    state['jurisdiction'], writes state['rule_results']."""
    extraction = state["extraction"]
    jurisdiction = state["jurisdiction"]

    if extraction is None:
        raise ValueError("rule_engine requires a validated ProtocolExtraction")
    if jurisdiction is None:
        raise ValueError("rule_engine requires jurisdiction to be set (determine_jurisdiction must run first)")

    results = run_rules(extraction, jurisdiction)
    flagged = [r for r in results if not r.passed]

    state["rule_results"] = results
    logger.info(
        f"rule_engine: {len(results)} rule(s) checked, "
        f"{len(flagged)} flag(s) raised: {[r.rule_id for r in flagged]}"
    )
    return state