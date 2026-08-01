"""
deep_contradiction_check — Node 11 of the Sentinel-GCP pipeline.

LLM call. Runs AFTER compliance_check, with access to raw document
sections, rule_engine findings, and Agent 2 findings — context the
early contradiction_check (node 6) deliberately doesn't have, since it
runs cheaply and early on summary fields only.

PROMPT CACHING: this node sends the SAME full document text block as
extract_fill (node 3), built via the identical
_build_document_text_block() function imported from that module —
byte-for-byte identical content is required for a cache hit. Running
within the same graph execution (well inside the 5-minute cache
window), this call should hit the cache extract_fill already wrote,
getting ~90% off the input-token cost for that block instead of paying
full price for the full document text a second time.

Catches a genuinely different error class: cross-section contradictions
within the SOURCE DOCUMENT ITSELF — e.g. one section states an SAE
reporting window of 24 hours, a different section (perhaps an older,
technically-superseded amendment reference) states 7 days. This requires
seeing actual document text from multiple sections simultaneously, which
the early check's extraction-summary-only view cannot provide.

See ARCHITECTURE.md §5 for why this is a SEPARATE node from the early
check, not a relocation of it.
"""
import json
import logging

from anthropic import Anthropic

from sentinel_gcp.schema.compliance import ContradictionFinding
from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.graph.nodes.extract_fill import _build_document_text_block
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

DEEP_CONTRADICTION_SYSTEM_PROMPT = """You are checking a clinical trial protocol for INTERNAL \
CONTRADICTIONS across its full document text — cases where different sections of the SAME \
document state conflicting information about the same topic.

You have been given: the full document's section text, the deterministic rule engine's findings, \
and the compliance reviewer's (Agent 2) findings. Use ALL of this context — a rule-engine or \
Agent 2 finding might reference the same topic a contradiction touches, and cross-referencing \
them can reveal or resolve an apparent conflict (e.g. an amendment log might explain that an \
older reference to "7 days" was superseded by a later "24 hours" requirement — that would NOT \
be a live contradiction).

Only flag GENUINE, UNRESOLVED contradictions — not differences that are explained by amendment \
history, versioning, or clearly distinct contexts (e.g. one timeline for SAEs, a different one \
for non-serious AEs is NOT a contradiction).

Return ONLY a JSON array (no other text) of findings in this shape:
[
  {
    "description": "<what conflicts, in plain language>",
    "section_refs": ["<section IDs involved>"]
  }
]
Return [] if nothing is genuinely, unresolvedly contradictory."""


def deep_contradiction_check(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['document_structure'],
    state['rule_results'], and state['agent_2_flags'], writes
    state['deep_contradiction_findings']."""
    doc_structure = state["document_structure"]
    rule_results = state["rule_results"]
    agent_2_flags = state["agent_2_flags"]

    if doc_structure is None:
        raise ValueError("deep_contradiction_check requires document_structure to be set")

    logger.info("deep_contradiction_check: starting deep cross-section consistency check")

    # Reuses the EXACT same function extract_fill used — byte-for-byte
    # identical output for the same doc_structure input is what makes
    # this a cache hit, not a cache miss.
    document_text_block = _build_document_text_block(doc_structure)
    findings_block = _build_findings_block(rule_results, agent_2_flags)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2048,
        system=DEEP_CONTRADICTION_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": document_text_block,
                    "cache_control": {"type": "ephemeral"},
                    # Same content as extract_fill's cached block —
                    # should hit that cache if this runs within ~5 min
                    # of it, which it does in a normal graph execution.
                },
                {
                    "type": "text",
                    "text": findings_block,
                    # NOT cached — unique to this call (rule/Agent 2 findings).
                },
            ],
        }],
    )

    raw_text = response.content[0].text
    try:
        raw_findings = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(f"deep_contradiction_check: model did not return valid JSON, got: {raw_text[:200]}")
        raw_findings = []

    findings = [
        ContradictionFinding(
            description=f["description"],
            section_refs=f.get("section_refs", []),
            check_stage="deep",
        )
        for f in raw_findings
    ]

    state["deep_contradiction_findings"] = findings

    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    logger.info(
        f"deep_contradiction_check: found {len(findings)} unresolved contradiction(s) — "
        f"input_tokens={usage.input_tokens}, cache_write={cache_write}, cache_read={cache_read}"
    )
    if cache_read == 0:
        logger.warning(
            "deep_contradiction_check: cache_read_input_tokens is 0 — the document "
            "text cache from extract_fill was NOT hit. Check timing (5-min window) "
            "or confirm _build_document_text_block() output is byte-identical."
        )
    return state


def _build_findings_block(rule_results, agent_2_flags) -> str:
    rule_findings_text = "\n".join(
        f"- {r.rule_id}: {'FLAGGED — ' + r.flag.issue if not r.passed else 'passed'}"
        for r in rule_results
    )
    agent_2_findings_text = "\n".join(
        f"- {f.issue} (evidence: {f.evidence})" for f in agent_2_flags
    ) or "(none)"

    return (
        f"RULE ENGINE FINDINGS:\n{rule_findings_text}\n\n"
        f"AGENT 2 COMPLIANCE FINDINGS:\n{agent_2_findings_text}"
    )