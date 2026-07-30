"""
extract_fill — Node 3 of the Sentinel-GCP pipeline.

Agent 1, Pass 2 (LLM call). Takes the label map from extract_discovery
(node 2) plus the full DocumentStructure, and populates the canonical
ProtocolExtraction schema (sentinel_gcp/schema/extraction.py) — with
per-field provenance (page, section, confidence) on every extracted value.

This does NOT validate the output — that's validate_schema (node 4).
This node's job is purely to attempt extraction; a malformed or
incomplete result here is expected to be caught downstream, not
prevented here.
"""
import json
import logging

from anthropic import Anthropic

from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

EXTRACTION_SYSTEM_PROMPT = """You are extracting structured data from a clinical trial protocol document.

You have been given a LABEL MAP describing what this specific document calls \
common fields (from a prior analysis pass), plus the document's section \
headings and relevant text. Use the label map to know WHERE to look, but \
extract the ACTUAL VALUES from the document text provided.

For every field you extract, you MUST also report:
- the page number it came from
- the section ID it came from (if identifiable)
- a confidence score from 0.0 to 1.0 for how certain you are

If a field genuinely cannot be found in the provided text, return null for \
its value — do NOT guess or infer a plausible-sounding answer. A null field \
is the correct, honest output when information isn't present; an invented \
value is a serious error.

Phase handling: report the phase exactly as written (e.g. "Phase 3", "2", \
"1/2") in phase_raw, AND set the boolean flags for every phase number the \
trial includes — a "1/2" combined trial should have BOTH \
phase_includes_1 and phase_includes_2 set to true, not just one.

Return ONLY a JSON object matching this exact shape (no other text):
{
  "metadata": {
    "trial_identifier": {"value": str|null, "label_used": str|null, "verbatim": str|null, "page": int|null, "section": str|null, "confidence": float|null},
    "sponsor": {"value": str|null, "label_used": str|null, "verbatim": str|null, "page": int|null, "section": str|null, "confidence": float|null},
    "phase_raw": str|null,
    "phase_includes_1": bool,
    "phase_includes_2": bool,
    "phase_includes_3": bool,
    "phase_includes_4": bool,
    "ind_number": {"value": str|null, "label_used": str|null, "verbatim": str|null, "page": int|null, "section": str|null, "confidence": float|null} | null,
    "eudract_number": {"value": str|null, "label_used": str|null, "verbatim": str|null, "page": int|null, "section": str|null, "confidence": float|null} | null
  },
  "study_arms": [
    {"cohort_name": str|null, "n_participants": int|null, "randomization_ratio": str|null, "population_description": str|null}
  ],
  "inclusion_criteria": [str],
  "exclusion_criteria": [str],
  "primary_endpoint": str|null,
  "secondary_endpoints": [str],
  "sae_reporting_timeline": {"value": str|null, "label_used": str|null, "verbatim": str|null, "page": int|null, "section": str|null, "confidence": float|null} | null,
  "additional_metadata": []
}"""


def extract_fill(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['document_structure'] and
    state['extraction_discovery'], writes state['extraction'] as a raw
    dict — validate_schema (node 4) is responsible for parsing it into
    an actual ProtocolExtraction instance and catching malformed output."""
    doc_structure = state["document_structure"]
    label_map = state["extraction_discovery"]

    if doc_structure is None:
        raise ValueError("extract_fill requires document_structure to be set")
    if label_map is None:
        raise ValueError("extract_fill requires extraction_discovery to have run first")

    logger.info("extract_fill: starting Pass 2 structured extraction")

    context = _build_extraction_context(doc_structure, label_map)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    raw_text = response.content[0].text
    try:
        extracted_dict = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(
            f"extract_fill: model did not return valid JSON, got: {raw_text[:300]}"
        )
        extracted_dict = None  # validate_schema will catch this as a failure and route to retry

    # NOTE: state['extraction'] holds a raw dict here, not a validated
    # ProtocolExtraction instance yet. GraphState's type hint says
    # Optional[ProtocolExtraction] for downstream clarity, but the actual
    # Pydantic parsing + validation happens in validate_schema (node 4) —
    # keeping this node's only job as "attempt extraction," nothing more.
    state["extraction"] = extracted_dict
    state["status"] = "validating"
    logger.info("extract_fill: extraction attempt complete, handing off to validate_schema")
    return state


def _build_extraction_context(doc_structure, label_map: dict) -> str:
    """Full document text + the label map from discovery. Unlike
    extract_discovery's deliberately narrow context, extraction genuinely
    needs the whole document — inclusion/exclusion criteria, endpoints,
    and the SAE section can be anywhere, not just the front matter."""
    all_pages_text = "\n---PAGE BREAK---\n".join(
        f"[Page {p}]\n{text}"
        for p, text in sorted(doc_structure.raw_text_by_page.items())
    )
    headings = "\n".join(
        f"- {s.section_id or '?'}: {s.heading}" for s in doc_structure.sections
    )
    tables_summary = "\n".join(
        f"- Table on page {t.page_start}"
        + (f" ('{t.name}')" if t.name else "")
        + f", confidence={t.confidence:.2f}"
        for t in doc_structure.tables
    )

    return (
        f"LABEL MAP FROM PRIOR ANALYSIS:\n{json.dumps(label_map, indent=2)}\n\n"
        f"DOCUMENT SECTION HEADINGS:\n{headings}\n\n"
        f"DOCUMENT TABLES DETECTED:\n{tables_summary}\n\n"
        f"FULL DOCUMENT TEXT:\n{all_pages_text}"
    )