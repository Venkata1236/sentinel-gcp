"""
extract_fill — Node 3 of the Sentinel-GCP pipeline.

Agent 1, Pass 2 (LLM call). Takes the label map from extract_discovery
(node 2) plus the full DocumentStructure, and populates the canonical
ProtocolExtraction schema (sentinel_gcp/schema/extraction.py) — with
per-field provenance (page, section, confidence) on every extracted value.

PROMPT CACHING: the full document text is sent here AND again in
deep_contradiction_check (node 11), within the same graph run — both
calls typically land well inside Claude's 5-minute cache window. The
document text is marked cache_control="ephemeral" as its own content
block, separate from the (per-call-varying) label map / instructions,
so the second full-document send in deep_contradiction_check gets the
~90% cached-input discount instead of being billed at full price twice.
See ARCHITECTURE.md cost analysis — this was the single largest cost
driver identified per-protocol (the two full-document-text calls
accounted for ~95% of total run cost before this fix).

CACHE PREFIX REQUIREMENT (round 2 fix): Anthropic's prompt cache keys
on the ENTIRE prefix up to and including the cache_control breakpoint —
that includes `system`, not just the message content blocks. The
document_text_block being byte-identical between this node and
deep_contradiction_check is necessary but NOT sufficient: if `system`
differs between the two calls, the shared block still misses, because
it's no longer at the same position in an identical prefix. This is
why SHARED_DOCUMENT_SYSTEM_PROMPT below is deliberately generic and
imported verbatim by deep_contradiction_check.py rather than each node
defining its own `system` string — the task-specific instructions that
used to live in `system` now live in an UNCACHED message block that
comes AFTER the cached document block, where they're free to differ
per node without breaking the cache.

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
from sentinel_gcp.utils.json_parsing import parse_claude_json

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

# Deliberately generic and IDENTICAL across every node that sends the
# cached document_text_block (currently: this node and
# deep_contradiction_check, which imports this constant verbatim rather
# than defining its own). Task-specific instructions live in a separate,
# uncached message block instead — see module docstring "CACHE PREFIX
# REQUIREMENT" above for why this split is required, not cosmetic.
SHARED_DOCUMENT_SYSTEM_PROMPT = (
    "You are analyzing a clinical trial protocol document, provided to you "
    "as extracted text below. Follow the specific task instructions given "
    "in the user message."
)

EXTRACTION_TASK_INSTRUCTIONS = """You are extracting structured data from a clinical trial protocol document.

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

    document_text_block = _build_document_text_block(doc_structure)
    instructions_block = _build_instructions_block(label_map)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system=SHARED_DOCUMENT_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": document_text_block,
                    "cache_control": {"type": "ephemeral"},
                    # This exact block is sent again, unchanged, by
                    # deep_contradiction_check (node 11) later in the
                    # same run — marking it here lets that later call
                    # reuse the cache instead of paying full price again.
                    # Requires `system` above to also match exactly,
                    # which is why it's the shared generic prompt, not
                    # this node's task-specific instructions.
                },
                {
                    "type": "text",
                    "text": EXTRACTION_TASK_INSTRUCTIONS,
                    # Task-specific — lives AFTER the cache breakpoint,
                    # so it's free to differ from deep_contradiction_check's
                    # own task instructions without invalidating the
                    # shared cache on the block above.
                },
                {
                    "type": "text",
                    "text": instructions_block,
                    # NOT cached — varies per document via the label map,
                    # so caching it would never hit anyway.
                },
            ],
        }],
    )

    raw_text = response.content[0].text
    extracted_dict = parse_claude_json(raw_text, "extract_fill")

    state["extraction"] = extracted_dict
    state["status"] = "validating"

    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    logger.info(
        f"extract_fill: extraction attempt complete — "
        f"input_tokens={usage.input_tokens}, cache_write={cache_write}, cache_read={cache_read}"
    )
    return state


def _build_document_text_block(doc_structure) -> str:
    """The large, reused-across-calls part — kept as its own function
    and its own content block specifically so it can be cache-marked
    independently of the instructions/label-map block below."""
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
        f"DOCUMENT SECTION HEADINGS:\n{headings}\n\n"
        f"DOCUMENT TABLES DETECTED:\n{tables_summary}\n\n"
        f"FULL DOCUMENT TEXT:\n{all_pages_text}"
    )


def _build_instructions_block(label_map: dict) -> str:
    """The small, per-call-varying part — deliberately NOT cached,
    since the label map differs by document and caching it would never
    produce a hit."""
    return f"LABEL MAP FROM PRIOR ANALYSIS:\n{json.dumps(label_map, indent=2)}"