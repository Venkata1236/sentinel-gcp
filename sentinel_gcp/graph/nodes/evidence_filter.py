"""
evidence_filter — new node, inserted between compliance_check (node 10)
and deep_contradiction_check (node 11).

Wires eval/evaluators/groundedness.py and eval/evaluators/applicability.py
into the LIVE pipeline — previously both existed only as standalone
evaluators callable from eval/run_eval.py, never invoked during an
actual graph run. state['agent_2_flags'] reached generate_report
without either check ever running against it.

Only source='agent_2' flags are checked — rule_engine flags are
deterministic (no LLM judgment involved, so neither "is this grounded"
nor "does this conclusion follow" applies) and insufficient_evidence
notes are explicitly NOT compliance conclusions (see compliance_check.py),
so they're passed through unfiltered too.

Policy: a flag is DROPPED if either check fails —
  - not grounded  -> the citation doesn't actually say what the flag claims
  - not applicable -> the regulation doesn't actually pertain to this
                       protocol's own content (real citation, wrong target)
Dropped flags are written to state['evidence_filter_dropped'] with a
structured reason (not just logged), so a human reviewing pipeline
output — or generate_report.py — can see what got filtered and why,
rather than just a smaller final count with no trail.
"""
import logging

from sentinel_gcp.graph.state import GraphState
from eval.evaluators.groundedness import evaluate_groundedness
from eval.evaluators.applicability import evaluate_applicability

logger = logging.getLogger(__name__)


def evidence_filter(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['agent_2_flags'],
    state['retrieved_chunks'], and state['extraction']; rewrites
    state['agent_2_flags'] with ungrounded/inapplicable agent_2
    findings removed, and writes state['evidence_filter_dropped']
    with the structured reason for each removal."""
    flags = state["agent_2_flags"]
    retrieved_chunks = state["retrieved_chunks"]
    extraction = state["extraction"]

    if not flags:
        logger.info("evidence_filter: no agent_2_flags to check — skipping")
        state["evidence_filter_dropped"] = []
        return state

    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in retrieved_chunks}

    kept = []
    dropped = []
    for flag in flags:
        # rule_engine flags and insufficient-evidence notes aren't
        # LLM conclusions about evidence — nothing for either evaluator
        # to check (see module docstring).
        if flag.source != "agent_2" or flag.insufficient_evidence:
            kept.append(flag)
            continue

        chunk_text = chunk_text_by_id.get(flag.retrieved_chunk_id)
        if chunk_text is None:
            # Shouldn't happen — compliance_check already validates
            # retrieved_chunk_id against deduped_chunks before building
            # the flag — but fail safe (drop, don't assume grounded)
            # rather than crash if state ever gets out of sync.
            logger.warning(
                f"evidence_filter: flag {flag.flag_id} cites unknown "
                f"chunk_id={flag.retrieved_chunk_id!r} — dropping (fail-safe)"
            )
            dropped.append({
                "flag_id": flag.flag_id, "issue": flag.issue,
                "reason_type": "invalid_citation",
                "reasoning": f"cited chunk_id {flag.retrieved_chunk_id!r} not found among retrieved chunks",
            })
            continue

        groundedness = evaluate_groundedness(flag, chunk_text)
        if not groundedness.grounded:
            dropped.append({
                "flag_id": flag.flag_id, "issue": flag.issue,
                "reason_type": "groundedness_failed",
                "reasoning": groundedness.reasoning,
            })
            continue
        flag.grounded = True

        applicability = evaluate_applicability(flag, extraction)
        if not applicability.applicable:
            dropped.append({
                "flag_id": flag.flag_id, "issue": flag.issue,
                "reason_type": "applicability_failed",
                "reasoning": applicability.reasoning,
            })
            continue
        flag.applicable = True

        kept.append(flag)

    if dropped:
        logger.warning(
            f"evidence_filter: dropped {len(dropped)}/{len(flags)} agent_2 finding(s) "
            f"— " + "; ".join(f"[{d['flag_id']}] {d['reason_type']}: {d['reasoning']}" for d in dropped)
        )
    logger.info(f"evidence_filter: {len(kept)}/{len(flags)} finding(s) passed groundedness + applicability")

    state["agent_2_flags"] = kept
    state["evidence_filter_dropped"] = dropped
    return state