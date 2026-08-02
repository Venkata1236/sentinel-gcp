"""
extract_discovery — Node 2 of the Sentinel-GCP pipeline.

Agent 1, Pass 1 (LLM call). Reads the parsed DocumentStructure and asks
Claude to inventory what labels/structure THIS specific document uses —
e.g. "trial ID is called 'Protocol Number'" vs "Study Code" vs
"PROTOCOL NO." — before any attempt is made to fill the canonical schema.

Two-tier context strategy (Approach 1 + 8 from design review):
  Tier 1 (fast path, ~95% of documents): first 3 pages + all section
    headings. Covers front-loaded cover/synopsis fields per ICH-GCP
    convention.
  Tier 2 (fallback, only when Tier 1 leaves critical fields null):
    a deterministic keyword scan (plain Python, no LLM call) finds
    which pages mention "Sponsor", "Protocol Number", "Phase", etc.,
    and those specific pages get added to the context for a second,
    targeted discovery call — instead of ever sending the full document.

This is deliberately NOT the same retry mechanism as retry_extraction
(node 5) — that one retries extract_fill after a Pydantic validation
failure on the final schema. This retry happens earlier and for a
different reason: the label MAP itself came back incomplete, before
extraction has even been attempted.
"""
import json
import logging
import re

from anthropic import Anthropic

from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings
from sentinel_gcp.utils.json_parsing import parse_claude_json

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

# If discovery comes back null on any of these, it's worth a targeted
# second pass — these are the fields extract_fill (node 3) genuinely
# cannot proceed well without.
CRITICAL_DISCOVERY_FIELDS = ["trial_identifier_label", "sponsor_label", "phase_label"]

# Plain-Python keyword scan vocabulary — deliberately simple substring
# matching, not fuzzy/semantic search. Good enough to locate candidate
# pages; the LLM still does the actual reading/labeling on those pages.
DISCOVERY_KEYWORDS = [
    "Sponsor", "Sponsor Name", "Study Sponsor", "Trial Sponsor",
    "Protocol Number", "Protocol No", "PROTOCOL NO", "Study Code",
    "Study Identifier", "Clinical Trial Identifier", "Trial ID",
    "Phase", "Development Phase", "Phase of Development",
    "IND Number", "US IND", "EudraCT",
]

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

    logger.info("extract_discovery: starting Pass 1 label discovery (Tier 1: fast path)")

    fast_context = _build_fast_context(doc_structure)
    label_map = _call_discovery(fast_context)

    missing_critical = [f for f in CRITICAL_DISCOVERY_FIELDS if not label_map.get(f)]
    if missing_critical:
        logger.info(
            f"extract_discovery: Tier 1 left critical fields null ({missing_critical}); "
            f"escalating to Tier 2 (keyword-scoped second pass)"
        )
        keyword_pages = _keyword_scan_pages(doc_structure)
        if keyword_pages:
            expanded_context = _build_fast_context(doc_structure, extra_pages=keyword_pages)
            second_pass_map = _call_discovery(expanded_context)
            # Merge: prefer second-pass values for fields that were missing,
            # keep first-pass values for anything that already succeeded
            label_map = _merge_label_maps(label_map, second_pass_map)
        else:
            logger.warning(
                "extract_discovery: keyword scan found no candidate pages; "
                "proceeding with Tier 1 result, extract_fill will attempt "
                "direct extraction without label hints for missing fields"
            )

    state["extraction_discovery"] = label_map
    logger.info(f"extract_discovery: final trial_identifier_label={label_map.get('trial_identifier_label')}")
    return state


def _call_discovery(context: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=DISCOVERY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )
    if not response.content:
        logger.warning(f"extract_discovery: Claude returned empty content, stop_reason={response.stop_reason}")
        return _empty_label_map()
    raw_text = response.content[0].text
    result = parse_claude_json(raw_text, "extract_discovery")
    return result if result is not None else _empty_label_map()


def _build_fast_context(doc_structure, extra_pages: list[int] | None = None) -> str:
    """Tier 1: first 3 pages (cover/synopsis-level fields, per ICH-GCP
    convention) + ALL section headings (spans the whole document, covers
    deep fields like sae_timeline_section_id).

    Tier 2 (when extra_pages is provided): the same base context, PLUS
    the specific pages a keyword scan flagged as likely containing a
    still-missing critical field — e.g. a sponsor name that only appears
    in a page-60 signature block rather than the cover page."""
    page_numbers = sorted(doc_structure.raw_text_by_page.keys())[:3]
    if extra_pages:
        page_numbers = sorted(set(page_numbers) | set(extra_pages))

    pages_text = "\n---PAGE BREAK---\n".join(
        f"[Page {p}]\n{doc_structure.raw_text_by_page[p]}"
        for p in page_numbers
        if p in doc_structure.raw_text_by_page
    )
    headings = "\n".join(
        f"- {s.section_id or '?'}: {s.heading}" for s in doc_structure.sections
    )
    return (
        f"DOCUMENT — SELECTED PAGES:\n{pages_text}\n\n"
        f"DOCUMENT — ALL SECTION HEADINGS (full document scope):\n{headings}"
    )


def _keyword_scan_pages(doc_structure, max_pages: int = 5) -> list[int]:
    """Deterministic, no-LLM keyword scan across every page — finds
    candidate pages likely to contain a critical field that Tier 1's
    first-3-pages assumption missed (e.g. sponsor buried in a page-60
    signature block). Returns at most `max_pages` matches to keep the
    Tier 2 LLM call cheap and targeted, not a full-document dump."""
    matches: list[int] = []
    for page_no, text in sorted(doc_structure.raw_text_by_page.items()):
        if any(re.search(re.escape(kw), text, re.IGNORECASE) for kw in DISCOVERY_KEYWORDS):
            matches.append(page_no)
        if len(matches) >= max_pages:
            break
    return matches


def _merge_label_maps(first_pass: dict, second_pass: dict) -> dict:
    """Second pass only overrides fields the first pass left null/empty —
    never overwrites a value Tier 1 already found correctly."""
    merged = dict(first_pass)
    for key, value in second_pass.items():
        if not merged.get(key) and value:
            merged[key] = value
    return merged


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