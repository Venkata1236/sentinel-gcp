"""
compliance_check — Node 10 of the Sentinel-GCP pipeline (Agent 2).

LLM call. Reasons over retrieved_chunks against the validated extraction
to produce ComplianceFlags for genuinely nuanced judgment calls — NOT for
anything rule_engine (node 8) already covers mechanically. Every flag
produced here MUST cite a specific retrieved_chunk_id and report
llm_certainty — enforced by ComplianceFlag's model_validator, so an
ungrounded flag literally cannot be constructed (see ARCHITECTURE.md §6,
groundedness).

IMPROVEMENTS applied after real testing against OEV-125:
1. Chunk deduplication before prompting.
2. Section references surfaced in the formatted chunk context.
3. Self-check requirement: every flag must include a supporting_quote —
   the exact sentence from the cited chunk. Flags without one are dropped.
4. "Not extracted" vs "not present" fix: the model reasons over
   EXTRACTED FIELDS (a structured subset), not the full source document.
   A real code review of live output caught the model treating "not in
   my extracted fields" as "absent from the protocol" — an evidence-of-
   absence vs absence-of-evidence problem. The prompt now explicitly
   requires an "insufficient_evidence" flag instead of a firm finding
   when this distinction is unclear.
5. Reporting-relationship precision: the same review caught a flag
   conflating the protocol's Investigator->Sponsor SAE timeline with a
   retrieved regulation's Sponsor->Authority timeline — different
   parties, not directly comparable. The prompt now explicitly requires
   checking both sides govern the same reporting relationship before
   treating a timeline mismatch as a compliance issue.
"""
import logging
import uuid

from anthropic import Anthropic

from sentinel_gcp.schema.compliance import ComplianceFlag
from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.utils.json_parsing import parse_claude_json
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

COMPLIANCE_SYSTEM_PROMPT = """You are performing a ROUTINE REGULATORY COMPLIANCE REVIEW of a \
publicly registered clinical trial protocol (registered on ClinicalTrials.gov, a US government \
database of clinical trials). This is standard due-diligence work performed by clinical research \
organizations and regulatory affairs teams — comparing protocol documentation against publicly \
available regulatory text (FDA, EMA, ICH-GCP) to identify documentation gaps before submission.

You have been given extracted protocol data and RETRIEVED REGULATION TEXT relevant to specific \
compliance topics. Your job is to identify genuinely nuanced compliance concerns — NOT simple \
presence/absence checks (those are already handled separately by deterministic rules).

Only raise a flag when there's a real judgment call to make — for example, wording that's \
ambiguous relative to what the regulation requires, or a description that technically exists \
but may not substantively satisfy the regulation's intent. Do NOT flag something just because \
you can — if the extracted content clearly and adequately addresses a retrieved regulation's \
requirement, do not manufacture a flag.

CRITICAL DISTINCTION — extracted fields vs. full document: you are reasoning over EXTRACTED \
FIELDS from the protocol, NOT the full source document. If something isn't in the extracted \
data, that means "not captured by extraction" — it does NOT mean "absent from the protocol." \
Only raise an absence-based flag when the retrieved regulation requires something that the \
EXTRACTED FIELDS actively contradict, or when a required value IS present but wrong (e.g. a \
numeric threshold that's incorrect). If you genuinely cannot tell whether something is missing \
from the protocol or simply wasn't extracted, set "insufficient_evidence": true instead of \
raising a firm compliance flag.

REPORTING-RELATIONSHIP CHECK: when comparing timelines/obligations between the protocol and a \
retrieved regulation, verify they govern the SAME reporting relationship — e.g. \
Investigator-to-Sponsor and Sponsor-to-Regulatory-Authority are DIFFERENT obligations between \
different parties. Do not treat a timeline as mismatched with a regulation unless both sides \
actually govern the same parties and the same reporting step.

SELF-CHECK REQUIREMENT before returning any flag: you must be able to quote the EXACT sentence \
or phrase from the cited chunk that supports your claim. If you cannot locate a specific \
supporting quote in the retrieved text — only a general sense that the topic is "relevant" — \
do NOT raise the flag; use "insufficient_evidence": true instead. This is especially important \
for ABSENCE claims — these are inherently harder to ground than direct conflicts, so be more \
conservative with them.

For EVERY flag you raise, you MUST:
1. Cite the EXACT chunk_id of the retrieved chunk your flag is based on — never make a claim \
without pointing to specific retrieved text
2. Include "supporting_quote": the exact sentence/phrase from that chunk supporting your claim
3. Set "insufficient_evidence" to true if you cannot confidently distinguish "missing from the \
protocol" from "not captured by extraction," or false if you have a genuine, groundable finding
4. Report your own certainty (0.0-1.0) — be honest; a genuinely ambiguous case (especially any \
absence claim) should get a lower certainty than a direct, clearly-worded conflict

Return ONLY a JSON array (no other text) of flags in this shape:
[
  {
    "issue": "<plain-language description of the concern>",
    "evidence": "<the specific extracted text this concern is based on>",
    "chunk_id": "<the exact chunk_id from RETRIEVED CHUNKS this flag cites>",
    "supporting_quote": "<exact quote from the cited chunk supporting this flag>",
    "regulation_reference": "<e.g. '21 CFR 312.32' or 'Regulation (EU) No 536/2014, Article 42'>",
    "impact": "<why this matters>",
    "recommendation": "<what a human reviewer should check or do>",
    "severity": "low" | "medium" | "high",
    "llm_certainty": <float 0.0-1.0>,
    "insufficient_evidence": <true|false>
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

    deduped_chunks = _deduplicate_chunks(retrieved_chunks)
    logger.info(
        f"compliance_check: reasoning over {len(deduped_chunks)} unique chunk(s) "
        f"(deduplicated from {len(retrieved_chunks)} retrieved)"
    )

    context = _build_context(extraction, deduped_chunks)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=COMPLIANCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    if not response.content:
        logger.warning(f"compliance_check: Claude returned empty content, stop_reason={response.stop_reason}")
        state["agent_2_flags"] = []
        return state

    raw_text = response.content[0].text
    raw_flags = parse_claude_json(raw_text, "compliance_check")
    if raw_flags is None:
        raw_flags = []

    valid_chunk_ids = {c["chunk_id"] for c in deduped_chunks}
    flags = []
    for raw in raw_flags:
        cited_chunk_id = raw.get("chunk_id")
        if cited_chunk_id not in valid_chunk_ids:
            logger.warning(
                f"compliance_check: model cited unknown chunk_id '{cited_chunk_id}' — dropping flag"
            )
            continue

        supporting_quote = raw.get("supporting_quote")
        if not supporting_quote:
            logger.warning(
                f"compliance_check: flag missing supporting_quote (self-check failed) — "
                f"dropping flag: {raw.get('issue', '')[:100]}"
            )
            continue

        is_insufficient = raw.get("insufficient_evidence", False)
        if is_insufficient:
            logger.info(
                f"compliance_check: insufficient_evidence flag (not a firm finding): "
                f"{raw.get('issue', '')[:100]}"
            )

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
                retrieval_score=_get_retrieval_score(deduped_chunks, cited_chunk_id),
                insufficient_evidence=is_insufficient,
            )
            flags.append(flag)
        except Exception as e:
            logger.warning(f"compliance_check: dropped malformed flag ({e}): {raw}")

    state["agent_2_flags"] = flags
    logger.info(f"compliance_check: {len(flags)} valid flag(s) produced")
    return state


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Drops chunks with duplicate chunk_id (the same underlying chunk
    retrieved by multiple topic queries)."""
    seen_ids = set()
    deduped = []
    for chunk in chunks:
        if chunk["chunk_id"] not in seen_ids:
            seen_ids.add(chunk["chunk_id"])
            deduped.append(chunk)
    if len(deduped) < len(chunks):
        logger.info(f"compliance_check: deduplicated {len(chunks) - len(deduped)} duplicate chunk(s)")
    return deduped


def _build_context(extraction, retrieved_chunks: list[dict]) -> str:
    chunks_text = "\n\n".join(
        f"[chunk_id: {c['chunk_id']}] (topic: {c['topic']}, source: {c['regulation_source']}, "
        f"section: {c.get('section_ref') or 'unspecified'})\n{c['text']}"
        for c in retrieved_chunks
    )
    return (
        f"EXTRACTED PROTOCOL DATA (note: this is a STRUCTURED SUBSET of the full protocol, "
        f"not the complete document):\n{extraction.model_dump_json(indent=2)}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_text}"
    )


def _get_extraction_confidence(extraction) -> float | None:
    confidences = [
        f.confidence
        for f in [extraction.metadata.trial_identifier, extraction.metadata.sponsor]
        if f and f.confidence is not None
    ]
    return sum(confidences) / len(confidences) if confidences else None


def _get_retrieval_score(retrieved_chunks: list[dict], chunk_id: str) -> float | None:
    match = next((c for c in retrieved_chunks if c["chunk_id"] == chunk_id), None)
    return match["score"] if match else None