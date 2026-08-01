"""
eval/evaluators/applicability.py — checks whether a compliance flag's
CONCLUSION actually follows from the protocol's own content, not just
whether the cited regulation text is real (that's groundedness.py's job).

Distinct failure mode from citation grounding, per real example:
  Regulation: "Pregnant women must be excluded."
  Protocol:   Does not enroll pregnant women at all.
  Agent 2:    "Protocol violates pregnancy exclusion requirement."

That flag would PASS groundedness.py (the cited regulation text is real
and accurately quoted) while being substantively WRONG — the regulation
doesn't actually apply to what the protocol says. groundedness.py cannot
catch this because it never sees the protocol's own extracted content,
only the claim and the regulation chunk.

Kept as a SEPARATE evaluator from groundedness.py, not merged into it —
mixing "is the citation real" and "does the conclusion follow" into one
judge call would make a failure ambiguous (which of the two problems
occurred?). Separate scores mean each failure mode is independently
diagnosable.
"""
import json
import logging
from dataclasses import dataclass

from anthropic import Anthropic

from sentinel_gcp.schema.compliance import ComplianceFlag
from sentinel_gcp.schema.extraction import ProtocolExtraction
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

APPLICABILITY_JUDGE_PROMPT = """You are checking whether a COMPLIANCE CONCLUSION actually follows \
from the PROTOCOL CONTENT it claims to be about — NOT whether the cited regulation text is real \
(that's checked separately). Assume the citation is accurate; your only job is to check whether \
the regulation, as cited, actually APPLIES to what the protocol itself says.

A conclusion is APPLICABLE if the protocol's actual content genuinely triggers the cited \
regulatory requirement. A conclusion is NOT applicable if the regulation is real but doesn't \
actually pertain to this protocol's specific situation — for example, flagging a pregnancy \
exclusion requirement violation on a protocol that already excludes pregnant women entirely.

Return ONLY a JSON object (no other text):
{
  "applicable": true|false,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<brief explanation, 1-2 sentences>"
}"""


@dataclass
class ApplicabilityResult:
    flag_id: str
    applicable: bool
    judge_confidence: float
    reasoning: str


def evaluate_applicability(
    flag: ComplianceFlag,
    protocol_extraction: ProtocolExtraction,
) -> ApplicabilityResult:
    """Checks whether a flag's conclusion is actually triggered by the
    protocol's own extracted content — a different question from
    groundedness.py's 'is the citation real'. Only meaningful for
    source='agent_2' flags, same reasoning as groundedness.py."""
    if flag.source != "agent_2":
        raise ValueError(
            f"evaluate_applicability only applies to agent_2 flags, "
            f"got source='{flag.source}' for flag_id={flag.flag_id}"
        )

    conclusion = f"{flag.issue} (citing: {flag.regulation_reference or 'unspecified'})"
    protocol_summary = protocol_extraction.model_dump_json(indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=APPLICABILITY_JUDGE_PROMPT,
        messages=[{
            "role": "user",
            "content": f"COMPLIANCE CONCLUSION:\n{conclusion}\n\nPROTOCOL CONTENT:\n{protocol_summary}",
        }],
    )

    raw_text = response.content[0].text
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning(
            f"evaluate_applicability: judge did not return valid JSON for flag {flag.flag_id}, "
            f"got: {raw_text[:200]} — treating as NOT APPLICABLE (fail-safe, not fail-open)"
        )
        return ApplicabilityResult(
            flag_id=flag.flag_id, applicable=False, judge_confidence=0.0,
            reasoning="Applicability judge returned unparseable output — flagged as not applicable by default",
        )

    return ApplicabilityResult(
        flag_id=flag.flag_id,
        applicable=result.get("applicable", False),
        judge_confidence=result.get("confidence", 0.0),
        reasoning=result.get("reasoning", ""),
    )


def evaluate_applicability_suite(
    flags_with_extraction: list[tuple[ComplianceFlag, ProtocolExtraction]],
) -> dict:
    """Aggregate version, mirroring evaluate_groundedness_suite()'s
    shape. A flag should ideally pass BOTH this and groundedness for a
    genuinely trustworthy compliance report — run_eval.py should report
    both rates side by side, not blend them into one number."""
    results = [evaluate_applicability(flag, extraction) for flag, extraction in flags_with_extraction]

    applicable_count = sum(1 for r in results if r.applicable)
    total = len(results)
    applicability_rate = applicable_count / total if total else 0.0

    not_applicable = [r for r in results if not r.applicable]
    if not_applicable:
        logger.warning(
            f"evaluate_applicability_suite: {len(not_applicable)}/{total} flag(s) NOT APPLICABLE — "
            f"{[r.flag_id for r in not_applicable]}"
        )

    return {
        "applicability_rate": round(applicability_rate, 3),
        "applicable_count": applicable_count,
        "total_count": total,
        "not_applicable_flags": [r.flag_id for r in not_applicable],
        "per_flag_results": results,
    }