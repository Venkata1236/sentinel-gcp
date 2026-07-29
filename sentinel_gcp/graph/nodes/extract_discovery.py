"""
extract_discovery — Node 2 of the Sentinel-GCP pipeline.

Agent 1, Pass 1 (LLM call). Reads the parsed DocumentStructure and asks
Claude to inventory what labels/structure THIS specific document uses —
e.g. "trial ID is called 'Protocol Number'" vs "Study Code" vs
"PROTOCOL NO." — before any attempt is made to fill the canonical schema.

This is what makes extract_fill (node 3) robust across structurally
different protocols instead of hardcoded to one document's wording —
see ARCHITECTURE.md §2.2 and §7 (variance coverage).

Output is an intermediate free-form label map (dict), not yet validated
against ProtocolExtraction — that validation happens after extract_fill,
not here.
"""
import json
import logging

from anthropic import Anthropic

from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

DISCOVERY_SYSTEM_PROMPT = """You are analyzing a clinical trial protocol document to identify its \
labeling conventions before structured extraction begins.

Your job is NOT to extract values yet. Your job is to identify what LABELS \
this specific document uses for common concepts, since different sponsors \
and jurisdictions word things differently. For example:
- The trial's unique identifier might be labeled "Protocol Number", \
"Study Code", "Protocol No.", or "PROTOCOL NO."
- A US IND-related field might be labeled "IND Number", "US IND Number", \
or not be present at all (EU-only trials use EudraCT instead)
- The phase might be labeled "Phase", "Development Phase", or \
"Phase of development", and its VALUE might be a single phase ("Phase 3"), \
a plain number ("2"), or a combined phase ("1/2")

Return ONLY a JSON object (no other text) with this shape:
{
  "trial_identifier_label": "<the exact label used, or null if not found>",
  "trial_identifier_location": "<page/section where found, or null>",
  "sponsor_label": "<exact label, or null>",
  "phase_label": "<exact label, or null>",
  "phase_value_format": "<brief description, e.g. 'single phase string' or 'combined X/Y'>",
  "has_ind_number": <true/false>,
  "ind_label": "<exact label if present, else null>",
  "has_eudract_number": <true/false>,
  "eudract_label": "<exact label if present, else null>",
  "sae_timeline_section_id": "<section ID if identifiable, e.g. '9.6', else null>",
  "notable_structural_features": ["<any other structural notes worth flagging, "
                                    "e.g. 'multi-cohort design with sub-cohorts A1/A2/B'>"]
}"""


def extract_discovery(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['document_structure'],
    writes state['extraction_discovery']."""
    doc_structure = state["document_structure"]
    if doc_structure is None:
        raise ValueError("extract_discovery requires document_structure to be set (parse_pdf must run first)")

    logger.info("extract_discovery: starting Pass 1 label discovery")

    # Use the first several pages + section headings as the discovery context —
    # trial identifier, phase, sponsor, and IND/EudraCT fields are reliably
    # on the cover page/synopsis, so we don't need the full document here.
    discovery_context = _build_discovery_context(doc_structure)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=DISCOVERY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": discovery_context}],
    )

    raw_text = response.content[0].text
    try:
        label_map = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(f"extract_discovery: model did not return valid JSON, got: {raw_text[:200]}")
        label_map = _empty_label_map()

    state["extraction_discovery"] = label_map
    logger.info(f"extract_discovery: found trial_identifier_label={label_map.get('trial_identifier_label')}")
    return state


def _build_discovery_context(doc_structure) -> str:
    """First 3 pages of raw text + all section headings — enough to spot
    labeling conventions without spending tokens on the full document."""
    early_pages = sorted(doc_structure.raw_text_by_page.keys())[:3]
    early_text = "\n---PAGE BREAK---\n".join(
        doc_structure.raw_text_by_page[p] for p in early_pages
    )
    headings = "\n".join(
        f"- {s.section_id or '?'}: {s.heading}" for s in doc_structure.sections
    )
    return (
        f"DOCUMENT — FIRST PAGES:\n{early_text}\n\n"
        f"DOCUMENT — SECTION HEADINGS:\n{headings}"
    )


def _empty_label_map() -> dict:
    """Fallback when the model's response isn't parseable JSON — extract_fill
    (node 3) will still attempt extraction directly from raw text in this
    case, just without the label hints. validate_schema (node 4) is the
    real safety net if this degrades extraction quality."""
    return {
        "trial_identifier_label": None,
        "sponsor_label": None,
        "phase_label": None,
        "has_ind_number": False,
        "has_eudract_number": False,
        "notable_structural_features": [],
    }