"""
compliance_check — Node 10 of the Sentinel-GCP pipeline (Agent 2).

LLM call. Reasons over retrieved_chunks against the validated extraction
to produce ComplianceFlags for genuinely nuanced judgment calls — NOT for
anything rule_engine (node 8) already covers mechanically. Every flag
produced here MUST cite a specific retrieved_chunk_id and report
llm_certainty — enforced by ComplianceFlag's model_validator, so an
ungrounded flag literally cannot be constructed (see ARCHITECTURE.md §6,
groundedness).
"""
import json
import logging
import uuid

from anthropic import Anthropic

from sentinel_gcp.schema.compliance import ComplianceFlag
from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

COMPLIANCE_SYSTEM_PROMPT = """You are a regulatory compliance reviewer for clinical trial protocols. \
You have been given extracted protocol data and RETRIEVED REGULATION TEXT relevant to specific \
compliance topics. Your job is to identify genuinely nuanced compliance concerns — NOT simple \
presence/absence checks (those are already handled separately by deterministic rules).

Only raise a flag when there's a real judgment call to make — for example, wording that's \
ambiguous relative to what the regulation requires, or a description that technically exists \
but may not substantively satisfy the regulation's intent. Do NOT flag something just because \
you can — if the extracted content clearly and adequately addresses a retrieved regulation's \
requirement, do not manufacture a flag.

For EVERY flag you raise, you MUST:
1. Cite the EXACT chunk_id of the retrieved chunk your flag is based on — never make a claim \
without pointing to specific retrieved text
2. Report your own certainty (0.0-1.0) — be honest; a genuinely ambiguous case should get a \
lower certainty than a fairly clear-cut one

Return ONLY a JSON array (no other text) of flags in this shape:
[
  {
    "issue": "<plain-language description of the concern>",
    "evidence": "<the specific extracted text this concern is based on>",
    "chunk_id": "<the exact chunk_id from RETRIEVED CHUNKS this flag cites>",
    "regulation_reference": "<e.g. '21 CFR 312.32'>",
    "impact": "<why this matters>",
    "recommendation": "<what a human reviewer should check or do>",
    "severity": "low" | "medium" | "high",
    "llm_certainty": <float 0.0-1.0>
  }
]
Return [] if nothing raises a genuine concern needing human judgment."""


def compliance_check(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['extraction'] and
    state['retrieved_chunks'], writes state['agent_2_flags']."""
    extraction = state["extraction"]
    retrieved_chunks = state["retrieved_chunks"]

    if extraction is None:
        raise ValueError("compliance_check requires a validated ProtocolExtraction")

    if not retrieved_chunks:
        logger.info("compliance_check: no retrieved chunks — skipping, nothing to reason over")
        state["agent_2_flags"] = []
        return state

    logger.info(f"compliance_check: reasoning over {len(retrieved_chunks)} retrieved chunk(s)")

    context = _build_context(extraction, retrieved_chunks)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=COMPLIANCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    raw_text = response.content[0].text
    try:
        raw_flags = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(f"compliance_check: model did not return valid JSON, got: {raw_text[:300]}")
        raw_flags = []

    valid_chunk_ids = {c["chunk_id"] for c in retrieved_chunks}
    flags = []
    for raw in raw_flags:
        cited_chunk_id = raw.get("chunk_id")
        if cited_chunk_id not in valid_chunk_ids:
            # The model cited a chunk_id that wasn't actually in what we
            # retrieved — this would fail ComplianceFlag's implicit trust
            # in retrieved_chunk_id being real. Drop the flag rather than
            # construct it with a fabricated-looking citation.
            logger.warning(
                f"compliance_check: model cited unknown chunk_id '{cited_chunk_id}' — dropping flag"
            )
            continue
        try:
            flag = ComplianceFlag(
                flag_id=f"AGENT2-{uuid.uuid4().hex[:8]}",
                source="agent_2",
                issue=raw["issue"],
                evidence=raw.get("evidence"),
                regulation_reference=raw.get("regulation_reference"),
                retrieved_chunk_id=cited_chunk_id,
                impact=raw.get("impact"),
                recommendation=raw.get("recommendation"),
                severity=raw.get("severity", "medium"),
                llm_certainty=raw["llm_certainty"],
                extraction_confidence=_get_extraction_confidence(extraction),
                retrieval_score=_get_retrieval_score(retrieved_chunks, cited_chunk_id),
            )
            flags.append(flag)
        except Exception as e:
            # ComplianceFlag's model_validator (source-consistency check)
            # or a missing required field would raise here — a malformed
            # Agent 2 output is dropped, not silently coerced into a
            # partially-valid flag.
            logger.warning(f"compliance_check: dropped malformed flag ({e}): {raw}")

    state["agent_2_flags"] = flags
    logger.info(f"compliance_check: {len(flags)} valid flag(s) produced")
    return state


def _build_context(extraction, retrieved_chunks: list[dict]) -> str:
    chunks_text = "\n\n".join(
        f"[chunk_id: {c['chunk_id']}] (topic: {c['topic']}, source: {c['regulation_source']})\n{c['text']}"
        for c in retrieved_chunks
    )
    return (
        f"EXTRACTED PROTOCOL DATA:\n{extraction.model_dump_json(indent=2)}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_text}"
    )


def _get_extraction_confidence(extraction) -> float | None:
    """Rough average of available per-field confidences, feeding
    compute_confidence() later — see sentinel_gcp/confidence/scoring.py
    (not yet built)."""
    confidences = [
        f.confidence
        for f in [extraction.metadata.trial_identifier, extraction.metadata.sponsor]
        if f and f.confidence is not None
    ]
    return sum(confidences) / len(confidences) if confidences else None


def _get_retrieval_score(retrieved_chunks: list[dict], chunk_id: str) -> float | None:
    match = next((c for c in retrieved_chunks if c["chunk_id"] == chunk_id), None)
    return match["score"] if match else None