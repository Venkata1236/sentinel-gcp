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

ROUND 2 FIX: byte-identical document text was NOT sufficient on its
own — the cache prefix also includes `system`, and this node used to
send its own DEEP_CONTRADICTION_SYSTEM_PROMPT as `system` while
extract_fill sent a different string, which broke the match even
though the cached block's content was identical. Both nodes now send
SHARED_DOCUMENT_SYSTEM_PROMPT (imported verbatim from extract_fill.py,
not redefined here) as `system`, and this node's actual task
instructions live in a separate uncached message block instead. See
extract_fill.py's module docstring, "CACHE PREFIX REQUIREMENT".

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
import re

from anthropic import Anthropic

from sentinel_gcp.schema.compliance import ContradictionFinding
from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.graph.nodes.extract_fill import (
    _build_document_text_block,
    SHARED_DOCUMENT_SYSTEM_PROMPT,
)
from sentinel_gcp.config import settings
from sentinel_gcp.utils.json_parsing import parse_claude_json

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

# Prompt-only enforcement of the CORE TEST is not fully reliable — same
# lesson already learned in compliance_check.py's speculative-language
# filter. These phrases are strong signals the model is describing
# ambiguity/missing-information, not an actual "Section A says X, Section
# B says NOT X" conflict, REGARDLESS of what contradiction_type it
# assigned. Checked against `description` for every finding; a match
# drops the finding entirely rather than downgrading it — if the
# description itself says "does not explicitly," the model has already
# told us this isn't a real conflict, so there's nothing to keep.
_AMBIGUITY_LANGUAGE_PATTERN = re.compile(
    r"does not explicitly|could be interpreted|not (explicitly )?specified|"
    r"\bambiguity\b|\bambiguous\b|does not (state|mandate|clarify|resolve)",
    re.IGNORECASE,
)


def _describes_ambiguity_not_contradiction(f: dict) -> bool:
    return bool(_AMBIGUITY_LANGUAGE_PATTERN.search(f.get("description", "")))

DEEP_CONTRADICTION_TASK_INSTRUCTIONS = """You are checking a clinical trial protocol for INTERNAL \
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

CORE TEST — apply this before flagging anything: you must be able to state the conflict as \
"Section A says X. Section B says NOT X" (or something logically incompatible with X) about the \
SAME topic. If you cannot state it that way — if what you actually have is scope that's merely \
UNCLEAR, a detail that's simply MISSING, or a definition that's UNDERSPECIFIED, where both \
statements COULD be true simultaneously under some reasonable reading — that is NOT a \
contradiction, even a "possible" one. Do not report it here. Ambiguity and missing detail are \
real issues, but they are a DIFFERENT kind of issue than a contradiction, and this check is \
scoped specifically to contradictions.

SIMULTANEOUS-TRUTH TEST (apply this too, separately): if BOTH statements can be true at the \
same time without either one being violated, this is NOT a contradiction — return nothing. \
"Day 15" stated in an overview/summary passage and "Day 15 (±2)" stated in the detailed \
schedule/table are NOT a contradiction — a summary section stating a rule less precisely than a \
detailed section stating the SAME rule is not a conflict, it's normal document structure \
(headline vs. detail). Only treat this as a contradiction if the summary statement is worded as \
an actual EXCLUSION or LIMIT (e.g. "only Day 15 is acceptable, no exceptions") that the detailed \
section then violates — not merely "the summary happens not to repeat every qualifier."

  - NOT a contradiction: "the document doesn't specify whether X applies in case Y" (a gap)
  - NOT a contradiction: "section A defines X one way; the document doesn't clarify how this \
    interacts with unrelated section B" (underspecification, not conflicting statements)
  - NOT a contradiction: a summary/overview passage states a rule in less detail than a table or \
    detailed section states the SAME rule (e.g. "Day 15" vs. "Day 15 (±2)") — this is normal \
    summary-vs-detail structure, not a conflict, UNLESS the summary explicitly excludes what the \
    detail permits
  - IS a contradiction: "section A requires X; section B requires NOT-X (or Z, incompatible with \
    X), for the same topic, same scope, same conditions"

MERGE, DON'T DUPLICATE: if you find multiple observations that describe the SAME underlying \
conflict — e.g. the same two requirements are restated or cross-referenced in three different \
sections — return ONE finding covering it, with ALL relevant section_refs combined, rather than \
several near-duplicate findings about the same root issue. Only return separate findings when \
they are genuinely about DIFFERENT topics or DIFFERENT conflicting statements, even if they \
happen to share a section reference (a single section can validly contain multiple unrelated \
requirements — sharing a section_ref alone does not make two issues the same conflict).

For every contradiction you flag (having already passed the CORE TEST above), classify it as \
exactly one of:
- "hard": a definite, unresolved conflict — the same topic, genuinely incompatible statements, \
  no plausible reading that reconciles them. E.g. one section requires a 24-hour SAE report, \
  another requires 7 days, for the same reporting relationship, with no amendment history \
  explaining the difference.
- "possible": a genuine conflict IS present (passes the CORE TEST), but a specific alternative \
  reading — one you can actually state — could resolve it. This is still X vs. NOT-X; it's just \
  not certain which reading is correct. This is NOT for "the document doesn't say" — that fails \
  the CORE TEST entirely and shouldn't be reported.
- "editorial": the statements ARE technically inconsistent in wording (not merely underspecified), \
  but resolving it one way or the other would not change what the protocol actually requires — \
  e.g. one section says "Day 1" and another says "Visit 2" for what is clearly, unambiguously the \
  same study day.

Return ONLY a JSON array (no other text) of findings in this shape:
[
  {
    "description": "<what conflicts, in plain language>",
    "section_refs": ["<section IDs involved>"],
    "contradiction_type": "hard" | "possible" | "editorial",
    "confidence": <float 0.0-1.0, your confidence that this is a genuine, unresolved contradiction>
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
        system=SHARED_DOCUMENT_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": document_text_block,
                    "cache_control": {"type": "ephemeral"},
                    # Same content as extract_fill's cached block, AND
                    # `system` above now matches extract_fill's exactly —
                    # both were required for a real cache hit (see
                    # module docstring, "ROUND 2 FIX").
                },
                {
                    "type": "text",
                    "text": DEEP_CONTRADICTION_TASK_INSTRUCTIONS,
                    # Task-specific — after the cache breakpoint, so it
                    # can differ from extract_fill's instructions without
                    # invalidating the shared cached block above.
                },
                {
                    "type": "text",
                    "text": findings_block,
                    # NOT cached — unique to this call (rule/Agent 2 findings).
                },
            ],
        }],
    )

    if not response.content:
        logger.warning(f"deep_contradiction_check: Claude returned empty content, stop_reason={response.stop_reason}")
        state["deep_contradiction_findings"] = []
        return state
    raw_text = response.content[0].text
    raw_findings = parse_claude_json(raw_text, "deep_contradiction_check")
    if raw_findings is None:
        raw_findings = []

    _SEVERITY_BY_TYPE = {"hard": "high", "possible": "medium", "editorial": "low"}

    findings = []
    for f in raw_findings:
        if _describes_ambiguity_not_contradiction(f):
            logger.warning(
                f"deep_contradiction_check: dropping finding despite CORE TEST instruction — "
                f"description contains ambiguity language, not a real conflict: "
                f"{f.get('description', '')!r}"
            )
            continue

        contradiction_type = f.get("contradiction_type")
        if contradiction_type not in _SEVERITY_BY_TYPE:
            logger.warning(
                f"deep_contradiction_check: missing/invalid contradiction_type "
                f"{contradiction_type!r} for finding {f.get('description', '')[:80]!r} "
                f"— leaving unclassified, severity stays at model default"
            )
            contradiction_type = None
        findings.append(ContradictionFinding(
            description=f["description"],
            section_refs=f.get("section_refs", []),
            check_stage="deep",
            contradiction_type=contradiction_type,
            llm_confidence=f.get("confidence"),
            # Severity was previously never set for deep findings at all
            # (always fell through to the schema default of "medium"
            # regardless of actual severity) — now derived from the
            # classification the model just gave us, when we have one.
            **({"severity": _SEVERITY_BY_TYPE[contradiction_type]} if contradiction_type else {}),
        ))

    state["deep_contradiction_findings"] = findings

    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    type_counts = {}
    for f in findings:
        key = f.contradiction_type or "unclassified"
        type_counts[key] = type_counts.get(key, 0) + 1
    logger.info(
        f"deep_contradiction_check: found {len(findings)} unresolved contradiction(s) "
        f"({type_counts}) — "
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