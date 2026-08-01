"""
contradiction_check — Node 6 of the Sentinel-GCP pipeline (EARLY pass).

LLM call. Checks whether Agent 1's own extracted fields are internally
consistent with each other — e.g. does the primary endpoint reference a
condition (cardiac hospitalization) that the inclusion criteria never
actually screens for? This is deliberately cheap and early, running
BEFORE jurisdiction detection, the rule engine, or any retrieval —
per ARCHITECTURE.md §5, this catches summary-level mismatches that don't
need retrieval context to detect.

Distinct from deep_contradiction_check (node 11), which runs much later
and has access to raw document sections, rule-engine findings, and Agent 2
findings — this early check only ever sees the validated ProtocolExtraction
object, nothing else.
"""
import json
import logging

from anthropic import Anthropic

from sentinel_gcp.schema.compliance import ContradictionFinding
from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings
from sentinel_gcp.utils.json_parsing import parse_claude_json

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

CONTRADICTION_SYSTEM_PROMPT = """You are checking a clinical trial protocol's EXTRACTED SUMMARY FIELDS \
for internal inconsistencies — NOT checking against any regulation, just checking whether the \
extracted fields agree with each other.

Examples of what to look for:
- The primary endpoint mentions a specific condition or outcome (e.g. "cardiac \
hospitalization") that the inclusion/exclusion criteria give no indication the \
trial population would experience or be screened for
- Study arm descriptions that contradict the stated randomization structure
- A phase marked as Phase 1 (safety-focused, typically small n) with study arms \
showing hundreds of participants, which would be unusual and worth flagging

Do NOT flag minor stylistic differences or things that are merely incomplete —
only flag things that are actually LOGICALLY INCONSISTENT with each other.
If you find nothing inconsistent, return an empty list — do not manufacture
a finding just to have something to report.

Return ONLY a JSON array (no other text) of findings in this shape:
[
  {
    "description": "<what's inconsistent, in plain language>",
    "section_refs": ["<section IDs involved, if identifiable, else empty list>"]
  }
]
Return [] if nothing inconsistent was found."""


def contradiction_check(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['extraction'] (must already
    be a validated ProtocolExtraction, not a raw dict — this only runs
    after validate_schema has passed), writes
    state['early_contradiction_findings']."""
    extraction = state["extraction"]
    if extraction is None:
        raise ValueError(
            "contradiction_check requires a validated ProtocolExtraction — "
            "this node should only be reached after validate_schema passes"
        )

    logger.info("contradiction_check: starting early self-consistency check")

    context = _build_context(extraction)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=CONTRADICTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    raw_text = response.content[0].text
    raw_findings = parse_claude_json(raw_text, "contradiction_check")
    if raw_findings is None:
        raw_findings = []

    findings = [
        ContradictionFinding(
            description=f["description"],
            section_refs=f.get("section_refs", []),
            check_stage="early",
        )
        for f in raw_findings
    ]

    state["early_contradiction_findings"] = findings
    logger.info(f"contradiction_check: found {len(findings)} inconsistenc(y/ies)")
    return state


def _build_context(extraction) -> str:
    """Serializes just the extracted summary fields — deliberately NOT
    the full document text, keeping this check cheap. Uses Pydantic's
    own JSON serialization so the model sees exactly what was extracted,
    not a hand-written summary that might drop something relevant."""
    return (
        "EXTRACTED PROTOCOL SUMMARY FIELDS:\n"
        f"{extraction.model_dump_json(indent=2)}"
    )