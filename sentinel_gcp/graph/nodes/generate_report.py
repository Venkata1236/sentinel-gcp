"""
generate_report — Node 14 of the Sentinel-GCP pipeline (final node).

Deterministic (no LLM call). Assembles everything accumulated across the
whole run into the final report, served via GET /report/{run_id}.
Computes real final_confidence for every flag using compute_confidence()
(sentinel_gcp/confidence/scoring.py) — this is the point where each
flag's final_confidence field (left None everywhere upstream) actually
gets filled in.
"""
import logging
from datetime import datetime, timezone

from sentinel_gcp.confidence.scoring import compute_confidence
from sentinel_gcp.graph.state import GraphState

logger = logging.getLogger(__name__)


def generate_report(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads the full accumulated state,
    writes state['status'] = 'complete'. The actual report document
    is assembled here and returned/logged — GET /report/{run_id} (API
    layer, not yet built) will read the same underlying checkpointed
    state and reconstruct this same report shape."""
    extraction = state["extraction"]
    rule_results = state["rule_results"]
    agent_2_flags = state["agent_2_flags"]
    early_findings = state["early_contradiction_findings"]
    deep_findings = state["deep_contradiction_findings"]
    human_decisions = state["human_decisions"]

    rule_flags_only = [r.flag for r in rule_results if not r.passed]
    all_flags = rule_flags_only + agent_2_flags

    # This is where every flag's final_confidence actually gets computed —
    # left as None throughout the rest of the pipeline per ComplianceFlag's
    # design (see schema/compliance.py) specifically so this is the single
    # place confidence gets finalized, not scattered across multiple nodes.
    for flag in all_flags:
        flag.final_confidence = compute_confidence(flag)

    decisions_by_flag = {d["flag_id"]: d for d in human_decisions}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trial_identifier": extraction.metadata.trial_identifier.value if extraction else None,
        "phase": extraction.metadata.phase_raw if extraction else None,
        "jurisdiction": state["jurisdiction"],
        "sponsor": extraction.metadata.sponsor.value if extraction else None,
        "retry_count": state["retry_count"],
        "rule_engine_summary": {
            "checks_run": len(rule_results),
            "checks_passed": len([r for r in rule_results if r.passed]),
            "flags_raised": len(rule_flags_only),
        },
        "agent_2_summary": {
            "flags_raised": len(agent_2_flags),
        },
        "contradiction_summary": {
            "early_check_findings": len(early_findings),
            "deep_check_findings": len(deep_findings),
        },
        "flags": [
            {
                **flag.model_dump(),
                "human_decision": decisions_by_flag.get(flag.flag_id, {}).get("decision", "not_reviewed"),
            }
            for flag in all_flags
        ],
        "contradictions": [f.model_dump() for f in early_findings + deep_findings],
    }

    state["status"] = "complete"
    logger.info(
        f"generate_report: COMPLETE — {report['trial_identifier']}, "
        f"{len(all_flags)} total flag(s), "
        f"{len([f for f in all_flags if f.final_confidence and f.final_confidence > 0.9])} high-confidence"
    )
    logger.info(f"generate_report: full report: {report}")
    return state