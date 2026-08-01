"""
eval/evaluators/groundedness.py — checks whether an Agent 2 ComplianceFlag's
evidence/issue actually follows from the retrieved regulation chunk it
cites, rather than drifting into an unsupported claim.

This is DIFFERENT from the schema-level enforcement already in
ComplianceFlag's model_validator (sentinel_gcp/schema/compliance.py) —
that validator only confirms a retrieved_chunk_id was cited at all (a
structural check). This evaluator checks the SEMANTIC relationship: does
the cited chunk's actual text support what the flag claims, or did the
model cite a real chunk but then say something the chunk doesn't
actually support?

Uses an LLM-as-judge approach — a separate, cheap Claude call asking a
narrow yes/no question, not a full compliance review. This is standard
practice for groundedness evaluation and matches ARCHITECTURE.md §6's
"how do you know Agent 2 isn't inventing citations" interview answer.
"""
import json
import logging
from dataclasses import dataclass

from anthropic import Anthropic

from sentinel_gcp.schema.compliance import ComplianceFlag
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

GROUNDEDNESS_JUDGE_PROMPT = """You are checking whether a CLAIM is actually supported by a SOURCE TEXT.

This is NOT a compliance review — you are only checking factual grounding: \
does the source text actually say what the claim asserts, or does the claim \
go beyond, contradict, or misrepresent what the source says?

A claim can be considered GROUNDED even if it paraphrases the source, as \
long as the paraphrase is accurate. A claim is NOT grounded if it asserts \
something the source doesn't actually support, even if the claim sounds \
plausible or is regulatorily reasonable.

Return ONLY a JSON object (no other text):
{
  "grounded": true|false,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation, 1-2 sentences>"
}"""


@dataclass
class GroundednessResult:
    flag_id: str
    grounded: bool
    judge_confidence: float
    reasoning: str


def evaluate_groundedness(flag: ComplianceFlag, retrieved_chunk_text: str) -> GroundednessResult:
    """Checks a single ComplianceFlag against the actual text of the
    chunk it cites. Only meaningful for source='agent_2' flags — a
    rule_engine flag has no retrieved_chunk_id at all (enforced by
    ComplianceFlag's model_validator), so there's nothing to check
    groundedness against."""
    if flag.source != "agent_2":
        raise ValueError(
            f"evaluate_groundedness only applies to agent_2 flags, "
            f"got source='{flag.source}' for flag_id={flag.flag_id}"
        )

    claim = f"{flag.issue} (evidence cited: {flag.evidence or 'none stated'})"

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=GROUNDEDNESS_JUDGE_PROMPT,
        messages=[{
            "role": "user",
            "content": f"CLAIM:\n{claim}\n\nSOURCE TEXT (the retrieved chunk this claim cites):\n{retrieved_chunk_text}",
        }],
    )

    raw_text = response.content[0].text
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(
            f"evaluate_groundedness: judge did not return valid JSON for flag {flag.flag_id}, "
            f"got: {raw_text[:200]} — treating as UNGROUNDED (fail-safe, not fail-open)"
        )
        # Deliberate fail-safe: if the judge itself fails to respond
        # parseably, treat the flag as ungrounded rather than assuming
        # it's fine — consistent with "null/honest failure over silent
        # false confidence" used throughout this project.
        return GroundednessResult(
            flag_id=flag.flag_id, grounded=False, judge_confidence=0.0,
            reasoning="Groundedness judge returned unparseable output — flagged as ungrounded by default",
        )

    return GroundednessResult(
        flag_id=flag.flag_id,
        grounded=result.get("grounded", False),
        judge_confidence=result.get("confidence", 0.0),
        reasoning=result.get("reasoning", ""),
    )


def evaluate_groundedness_suite(
    flags_with_chunks: list[tuple[ComplianceFlag, str]],
) -> dict:
    """Runs evaluate_groundedness() across multiple (flag, chunk_text)
    pairs — what run_eval.py (not yet built) will call across a full
    set of Agent 2 findings from real runs. Reports the groundedness
    RATE, which is the direct empirical metric behind the
    hallucination-handling interview answer."""
    results = [evaluate_groundedness(flag, chunk_text) for flag, chunk_text in flags_with_chunks]

    grounded_count = sum(1 for r in results if r.grounded)
    total = len(results)
    groundedness_rate = grounded_count / total if total else 0.0

    ungrounded = [r for r in results if not r.grounded]
    if ungrounded:
        logger.warning(
            f"evaluate_groundedness_suite: {len(ungrounded)}/{total} flag(s) UNGROUNDED — "
            f"{[r.flag_id for r in ungrounded]}"
        )

    return {
        "groundedness_rate": round(groundedness_rate, 3),
        "grounded_count": grounded_count,
        "total_count": total,
        "ungrounded_flags": [r.flag_id for r in ungrounded],
        "per_flag_results": results,
    }