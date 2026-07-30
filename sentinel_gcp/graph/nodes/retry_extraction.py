"""
retry_extraction — Node 5 of the Sentinel-GCP pipeline.

LLM call, conditional — only runs when validate_schema (node 4) reports
extraction_errors AND retry_count is still 0. Re-attempts extraction with
a stricter prompt that includes the EXACT validation errors from the
first attempt, so the model corrects specific known problems rather than
guessing blind a second time.

Capped at one retry (per MAX_EXTRACTION_RETRIES in config.py / .env) —
per ARCHITECTURE.md, not every validation failure is a document problem;
one retry catches single-attempt model misses cheaply, but a second
consecutive failure escalates to needs_human rather than retrying
indefinitely.
"""
import json
import logging

from anthropic import Anthropic

from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

RETRY_SYSTEM_PROMPT = """You are re-attempting structured extraction from a clinical trial protocol \
document. Your PREVIOUS attempt failed schema validation with specific errors listed below. \
Fix these exact problems — do not change fields that were already correct.

Common causes of validation failure and how to fix them:
- A required nested field (like metadata.trial_identifier) was missing entirely —
  ensure every FieldWithProvenance object has at least a "value" key, even if
  other keys like page/section are null
- A field expected to be a list was returned as a single value or omitted —
  ensure inclusion_criteria, exclusion_criteria, study_arms, secondary_endpoints
  are always JSON arrays, even if empty ([])
- A boolean field (phase_includes_1, etc.) was returned as a string ("true")
  instead of an actual boolean (true)
- Extra/unexpected fields were added that aren't part of the schema

Return ONLY a corrected JSON object in the exact same shape as before —
no explanation, no markdown formatting, just the raw corrected JSON."""


def retry_extraction(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['document_structure'],
    state['extraction_discovery'], and state['extraction_errors'] (the
    specific failures from validate_schema), writes a corrected raw dict
    back to state['extraction']. Increments state['retry_count'] so the
    graph's routing knows this was already attempted."""
    doc_structure = state["document_structure"]
    label_map = state["extraction_discovery"]
    errors = state["extraction_errors"]

    logger.info(f"retry_extraction: attempt with {len(errors)} known error(s) to fix: {errors}")

    context = _build_retry_context(doc_structure, label_map, errors)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=RETRY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    raw_text = response.content[0].text
    try:
        extracted_dict = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(f"retry_extraction: retry ALSO returned invalid JSON: {raw_text[:300]}")
        extracted_dict = None

    state["extraction"] = extracted_dict
    state["retry_count"] = state["retry_count"] + 1
    state["status"] = "validating"  # routes back to validate_schema for a second check
    logger.info(f"retry_extraction: attempt complete, retry_count now {state['retry_count']}")
    return state


def _build_retry_context(doc_structure, label_map: dict, errors: list[str]) -> str:
    """Same full-document context as extract_fill, PLUS the specific
    validation errors from the failed first attempt — this is the whole
    point of a stricter retry rather than just re-running the same prompt
    and hoping for a different result."""
    all_pages_text = "\n---PAGE BREAK---\n".join(
        f"[Page {p}]\n{text}"
        for p, text in sorted(doc_structure.raw_text_by_page.items())
    )
    headings = "\n".join(
        f"- {s.section_id or '?'}: {s.heading}" for s in doc_structure.sections
    )
    errors_text = "\n".join(f"- {e}" for e in errors)

    return (
        f"YOUR PREVIOUS EXTRACTION ATTEMPT FAILED VALIDATION WITH THESE ERRORS:\n{errors_text}\n\n"
        f"LABEL MAP FROM PRIOR ANALYSIS:\n{json.dumps(label_map, indent=2)}\n\n"
        f"DOCUMENT SECTION HEADINGS:\n{headings}\n\n"
        f"FULL DOCUMENT TEXT:\n{all_pages_text}"
    )