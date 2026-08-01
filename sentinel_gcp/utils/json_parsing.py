"""
Shared JSON-parsing helper for Claude API responses. Models sometimes
wrap JSON output in markdown code fences (```json ... ```) despite
explicit prompt instructions not to — confirmed via real testing
(OEV-125 extract_fill run). Centralized here so every node that parses
a JSON response from Claude gets this fix once, not five separate
copy-pasted implementations.
"""
import json
import logging

logger = logging.getLogger(__name__)


def parse_claude_json(raw_text: str, node_name: str) -> dict | list | None:
    """Strips markdown code fences if present, then parses JSON.
    Returns None on failure (never raises) — callers already handle
    None as 'extraction/response failed, route accordingly' per the
    honesty-over-guessing principle used throughout this pipeline."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"{node_name}: model did not return valid JSON, got: {text[:300]}")
        return None