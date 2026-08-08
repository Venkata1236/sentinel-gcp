"""
compliance_check — Node 10 of the Sentinel-GCP pipeline (Agent 2).

...(existing docstring, plus:)

IMPROVEMENT (round 2): insufficient_evidence findings are now a
STRUCTURALLY SEPARATE category, not a boolean flag alongside a normal
severity — a downstream consumer reading only `severity` can no longer
mistake an "I can't tell" finding for an actual violation. The prompt
is also now given the EXPLICIT list of fields ProtocolExtraction
actually captures, so absence-claims can only be raised about concepts
genuinely outside the schema when the model correctly recognizes them
as such — rather than guessing at what "extracted data" might contain.
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

MAX_REFUSAL_RETRIES = 2

# Hedged language is a direct signal the model itself isn't confident this
# is a real violation — not something to trust as-is at "medium"/"high"
# severity. Checked against `issue` ONLY, not `evidence` — evidence quotes
# or paraphrases the source regulatory chunk, which legitimately contains
# words like "may" in its own conditional structure (e.g. "SUSARs may
# require expedited reporting") without that reflecting the MODEL's
# confidence at all. Scanning evidence too caused a real false-positive
# demotion (SAE-reporting-relationship finding, previously confirmed
# grounded at 0.95 confidence, got rerouted here for a hedge word that
# was almost certainly in the quoted source text, not the model's claim).
import re

_SPECULATIVE_LANGUAGE_PATTERN = re.compile(
    r"\b(may|might|appears?|could|likely|seems?|possibly)\b", re.IGNORECASE
)


def _has_speculative_language(raw: dict) -> bool:
    return bool(_SPECULATIVE_LANGUAGE_PATTERN.search(raw.get("issue", "")))

# The ACTUAL fields ProtocolExtraction captures — given to the model
# explicitly so it can distinguish "this concept isn't even in our
# extraction schema" from "this field exists in the schema but came
# back empty for this document." Keep this in sync with
# sentinel_gcp/schema/extraction.py if fields are added/removed.
EXTRACTED_SCHEMA_FIELDS = """
- trial_identifier, sponsor, phase, ind_number, eudract_number
- study_arms (cohort name, participant count, randomization ratio, population description)
- inclusion_criteria, exclusion_criteria (as lists)
- primary_endpoint, secondary_endpoints
- sae_reporting_timeline
"""

def _build_compliance_system_prompt(jurisdiction: str | None) -> str:
    jurisdiction_label = jurisdiction or "UNDETERMINED"
    return f"""You are performing a ROUTINE REGULATORY COMPLIANCE REVIEW of a \
publicly registered clinical trial protocol (registered on ClinicalTrials.gov, a US government \
database of clinical trials). This is standard due-diligence work performed by clinical research \
organizations and regulatory affairs teams — comparing protocol documentation against publicly \
available regulatory text (FDA, EMA, ICH-GCP) to identify documentation gaps before submission.

You have been given extracted protocol data and RETRIEVED REGULATION TEXT relevant to specific \
compliance topics. Your job is to identify genuinely nuanced compliance concerns — NOT simple \
presence/absence checks (those are already handled separately by deterministic rules).

JURISDICTION SCOPE: this protocol's determined regulatory jurisdiction is {jurisdiction_label}. \
The RETRIEVED CHUNKS below have already been filtered to {jurisdiction_label}-relevant and \
ICH-GCP sources (ICH-GCP applies globally regardless of jurisdiction) — do not reason about a \
DIFFERENT jurisdiction's specific regulatory mechanisms (e.g. FDA IND numbers, 21 CFR \
requirements) from your own general knowledge if {jurisdiction_label} is not FDA, unless the \
extracted protocol data ITSELF explicitly indicates a submission under that other jurisdiction. \
Every finding or note must trace to a chunk actually present in RETRIEVED CHUNKS below — if a \
regulatory concept occurs to you but isn't grounded in one of those chunks, do not raise it.

Only raise a COMPLIANCE FINDING when there's a real judgment call to make, based on POSITIVE \
EVIDENCE — content that IS present in the extracted data but is ambiguous, incomplete, or \
substantively doesn't satisfy what the regulation requires. Do NOT flag something just because \
you can — if the extracted content clearly and adequately addresses a retrieved regulation's \
requirement, do not manufacture a finding.

THE EXTRACTION SCHEMA ONLY CAPTURES THESE FIELDS:
{EXTRACTED_SCHEMA_FIELDS}
Any regulatory concept NOT in this list (e.g. withdrawal procedures, data monitoring committee \
composition, statistical analysis plan details) is OUTSIDE THE EXTRACTION SCHEMA ENTIRELY — the \
extraction pipeline was never designed to capture it. Do NOT raise a finding claiming such a \
concept is "missing from the protocol" — that would be a claim about the WHOLE PROTOCOL based \
on a schema that never attempted to capture it. This is different from a concept that IS in the \
schema above but came back empty/null for THIS document — that MAY be worth an insufficient-
evidence note (see below), since it's at least plausible the field is genuinely absent.

REPORTING-RELATIONSHIP CHECK: when comparing timelines/obligations between the protocol and a \
retrieved regulation, verify they govern the SAME reporting relationship — e.g. \
Investigator-to-Sponsor and Sponsor-to-Regulatory-Authority are DIFFERENT obligations between \
different parties. Do not treat a timeline as mismatched with a regulation unless both sides \
actually govern the same parties and the same reporting step.

SELF-CHECK REQUIREMENT before returning any finding: you must be able to quote the EXACT sentence \
or phrase from the cited chunk that supports your claim. If you cannot, do not raise it.

TWO SEPARATE OUTPUT CATEGORIES — return findings in the correct one:

1. "compliance_findings" — POSITIVE-EVIDENCE findings only. The extracted data contains \
something that conflicts with, or is ambiguous relative to, what the retrieved regulation \
requires. severity must be "medium" or "high" — reserved for findings with real, positive \
textual support, never for absence-based suspicions.

2. "insufficient_evidence_notes" — for a field that IS in the extraction schema above, is \
empty/null or ambiguously stated for THIS document, AND the retrieved regulation suggests it \
should contain something specific. These are NOT compliance violations — they are notes that a \
human reviewer should check the full source document, since the extraction pipeline may simply \
not have captured it. Never assign "medium" or "high" severity to these; use "low" only, and \
only as an indicator of review priority, not violation severity.

Anything that is neither a genuine compliance finding nor an insufficient-evidence note — a \
retrieved chunk that's simply not relevant, or a field that's clearly and adequately addressed — \
should not appear in your output AT ALL. There is no third "ignore" category to populate; \
silence on a topic already means "nothing to report."

For every item in EITHER category, include: issue, evidence, chunk_id (from RETRIEVED CHUNKS), \
supporting_quote (exact text from that chunk), regulation_reference, impact, recommendation, \
llm_certainty (0.0-1.0).

Return ONLY a JSON object (no other text) in this shape:
{{
  "compliance_findings": [
    {{"issue": "...", "evidence": "...", "chunk_id": "...", "supporting_quote": "...",
      "regulation_reference": "...", "impact": "...", "recommendation": "...",
      "severity": "medium" | "high", "llm_certainty": <float>}}
  ],
  "insufficient_evidence_notes": [
    {{"issue": "...", "evidence": "...", "chunk_id": "...", "supporting_quote": "...",
      "regulation_reference": "...", "impact": "...", "recommendation": "...",
      "llm_certainty": <float>}}
  ]
}}
Return empty arrays for either/both if nothing genuinely applies."""


def compliance_check(state: GraphState) -> GraphState:
    extraction = state["extraction"]
    retrieved_chunks = state["retrieved_chunks"]
    jurisdiction = state["jurisdiction"]

    if extraction is None:
        raise ValueError("compliance_check requires a validated ProtocolExtraction")

    if not retrieved_chunks:
        logger.info("compliance_check: no retrieved chunks — skipping")
        state["agent_2_flags"] = []
        return state

    deduped_chunks = _deduplicate_chunks(retrieved_chunks)
    logger.info(f"compliance_check: reasoning over {len(deduped_chunks)} unique chunk(s), jurisdiction={jurisdiction}")

    context = _build_context(extraction, deduped_chunks)
    system_prompt = _build_compliance_system_prompt(jurisdiction)

    response = None
    for attempt in range(1, MAX_REFUSAL_RETRIES + 2):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": context}],
        )
        if response.content:
            if attempt > 1:
                logger.info(f"compliance_check: succeeded on attempt {attempt}")
            break
        logger.warning(f"compliance_check: attempt {attempt} refused, stop_reason={response.stop_reason}")

    if not response or not response.content:
        logger.warning(f"compliance_check: refusal persisted after {MAX_REFUSAL_RETRIES + 1} attempts")
        state["agent_2_flags"] = []
        return state

    raw_text = response.content[0].text
    parsed = parse_claude_json(raw_text, "compliance_check")
    if parsed is None:
        parsed = {}

    valid_chunk_ids = {c["chunk_id"] for c in deduped_chunks}
    flags = []

    # Category 1: real compliance findings — medium/high severity only,
    # UNLESS the finding itself hedges (may/might/appears/...), in which
    # case the model's own wording says it isn't confident — reroute to
    # insufficient_evidence rather than trust the stated severity.
    for raw in parsed.get("compliance_findings", []):
        if _has_speculative_language(raw):
            logger.info(
                f"compliance_check: rerouting speculative-language finding to "
                f"insufficient_evidence — issue text: {raw.get('issue', '')!r}"
            )
            raw["severity"] = "low"
            flag = _build_flag(raw, valid_chunk_ids, extraction, deduped_chunks, insufficient=True)
        else:
            flag = _build_flag(raw, valid_chunk_ids, extraction, deduped_chunks, insufficient=False)
        if flag:
            flags.append(flag)

    # Category 2: insufficient-evidence notes — always "low", never a violation
    for raw in parsed.get("insufficient_evidence_notes", []):
        raw["severity"] = "low"  # forced, regardless of what the model returned
        flag = _build_flag(raw, valid_chunk_ids, extraction, deduped_chunks, insufficient=True)
        if flag:
            flags.append(flag)

    state["agent_2_flags"] = flags
    logger.info(
        f"compliance_check: {len(flags)} total item(s) "
        f"({sum(1 for f in flags if not f.insufficient_evidence)} finding(s), "
        f"{sum(1 for f in flags if f.insufficient_evidence)} insufficient-evidence note(s))"
    )
    return state


def _build_flag(raw, valid_chunk_ids, extraction, deduped_chunks, insufficient: bool) -> ComplianceFlag | None:
    cited_chunk_id = raw.get("chunk_id")
    if cited_chunk_id not in valid_chunk_ids:
        logger.warning(f"compliance_check: unknown chunk_id '{cited_chunk_id}' — dropping")
        return None
    if not raw.get("supporting_quote"):
        logger.warning(f"compliance_check: missing supporting_quote — dropping: {raw.get('issue', '')[:100]}")
        return None
    try:
        return ComplianceFlag(
            flag_id=f"AGENT2-{uuid.uuid4().hex[:8]}",
            source="agent_2",
            issue=raw["issue"],
            evidence=raw.get("evidence"),
            supporting_quote=raw.get("supporting_quote"),
            regulation_reference=raw.get("regulation_reference"),
            retrieved_chunk_id=cited_chunk_id,
            impact=raw.get("impact"),
            recommendation=raw.get("recommendation"),
            severity=raw.get("severity", "low"),
            llm_certainty=raw["llm_certainty"],
            extraction_confidence=_get_extraction_confidence(extraction),
            retrieval_score=_get_retrieval_score(deduped_chunks, cited_chunk_id),
            insufficient_evidence=insufficient,
        )
    except Exception as e:
        logger.warning(f"compliance_check: dropped malformed item ({e}): {raw}")
        return None


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
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
        f"EXTRACTED PROTOCOL DATA:\n{extraction.model_dump_json(indent=2)}\n\n"
        f"RETRIEVED CHUNKS:\n{chunks_text}"
    )


def _get_extraction_confidence(extraction) -> float | None:
    confidences = [
        f.confidence for f in [extraction.metadata.trial_identifier, extraction.metadata.sponsor]
        if f and f.confidence is not None
    ]
    return sum(confidences) / len(confidences) if confidences else None


def _get_retrieval_score(retrieved_chunks: list[dict], chunk_id: str) -> float | None:
    match = next((c for c in retrieved_chunks if c["chunk_id"] == chunk_id), None)
    return match["score"] if match else None