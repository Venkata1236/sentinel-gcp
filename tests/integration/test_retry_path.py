"""
Integration test for the retry_extraction path — validate_schema fails,
retry_extraction attempts once with a stricter prompt, validate_schema
checks again. Uses MOCKED Anthropic responses (unittest.mock), not real
API calls — this tests the CONTROL FLOW (does a failure correctly
trigger a retry, does the retry's result get re-validated), not the
model's actual extraction quality, so there's no reason to spend real
API cost testing it.
"""
import pytest
from unittest.mock import MagicMock, patch

from sentinel_gcp.graph.state import initial_state
from sentinel_gcp.graph.nodes.validate_schema import validate_schema
from sentinel_gcp.graph.nodes.retry_extraction import retry_extraction
from sentinel_gcp.schema.document_structure import DocumentStructure, ParsingCoverage, Section


def _make_test_state_with_raw_extraction(raw_extraction: dict | None) -> dict:
    """Builds a minimal GraphState as if extract_fill just ran, with a
    given raw (unvalidated) extraction dict already in state — skips
    actually running parse_pdf/extract_discovery/extract_fill, since
    this test is about the validate->retry->validate LOOP, not those
    earlier stages."""
    state = initial_state(raw_pdf_path="fake.pdf", run_id="test-retry-run")
    state["document_structure"] = DocumentStructure(
        sections=[Section(heading="Test Section", page_start=1, text="test")],
        parsing_coverage=ParsingCoverage(total_pages=1),
    )
    state["extraction_discovery"] = {"trial_identifier_label": "Protocol Number"}
    state["extraction"] = raw_extraction
    return state


def _make_mock_response(text: str):
    """Builds a fake Anthropic response object matching what
    response.content[0].text expects, without a real API call."""
    mock_content_block = MagicMock()
    mock_content_block.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    mock_response.stop_reason = "end_turn"
    return mock_response


_VALID_EXTRACTION_JSON = """{
  "metadata": {
    "trial_identifier": {"value": "TEST-001", "label_used": "Protocol Number"},
    "sponsor": {"value": "Test Sponsor"},
    "phase_raw": "Phase 2",
    "phase_includes_2": true
  },
  "inclusion_criteria": ["test criterion"],
  "exclusion_criteria": [],
  "study_arms": [],
  "secondary_endpoints": [],
  "additional_metadata": []
}"""

# Deliberately malformed — missing the required nested "metadata" object entirely
_INVALID_EXTRACTION_JSON = """{
  "inclusion_criteria": ["test criterion"]
}"""


def test_valid_extraction_passes_validation_without_retry():
    """Baseline — a well-formed extraction should pass validate_schema
    on the first attempt, with retry_count staying at 0."""
    import json
    state = _make_test_state_with_raw_extraction(json.loads(_VALID_EXTRACTION_JSON))
    state = validate_schema(state)

    assert state["extraction_errors"] == []
    assert state["extraction"] is not None
    assert state["extraction"].metadata.trial_identifier.value == "TEST-001"


def test_malformed_extraction_fails_validation_with_real_error_messages():
    """A malformed extraction (missing required metadata) should fail
    validation with a readable error, not silently pass or crash."""
    import json
    state = _make_test_state_with_raw_extraction(json.loads(_INVALID_EXTRACTION_JSON))
    state = validate_schema(state)

    assert len(state["extraction_errors"]) > 0
    assert state["extraction"] is None


@patch("sentinel_gcp.graph.nodes.retry_extraction.client")
def test_retry_extraction_uses_validation_errors_in_its_prompt(mock_client):
    """Confirms retry_extraction actually INCLUDES the specific errors
    from the failed validation attempt in what it sends to Claude —
    this is the whole point of a 'stricter' retry, not just a blind
    second attempt with the same prompt."""
    mock_client.messages.create.return_value = _make_mock_response(_VALID_EXTRACTION_JSON)

    state = _make_test_state_with_raw_extraction(None)
    state["extraction_errors"] = ["metadata: field required"]

    state = retry_extraction(state)

    # Confirm the mock was actually called, and inspect what was sent
    assert mock_client.messages.create.called
    call_kwargs = mock_client.messages.create.call_args.kwargs
    sent_content = call_kwargs["messages"][0]["content"]
    assert "metadata: field required" in sent_content, (
        "retry_extraction must include the specific validation errors "
        "in its prompt — a retry that doesn't tell the model what was "
        "wrong is just a blind second guess, not a targeted correction"
    )


@patch("sentinel_gcp.graph.nodes.retry_extraction.client")
def test_retry_extraction_increments_retry_count(mock_client):
    """retry_count must increment — this is what graph/builder.py's
    _route_after_validation() checks to decide whether a SECOND retry
    is allowed (it isn't, per MAX_EXTRACTION_RETRIES=1) or whether to
    route to needs_human_exit instead."""
    mock_client.messages.create.return_value = _make_mock_response(_VALID_EXTRACTION_JSON)

    state = _make_test_state_with_raw_extraction(None)
    state["extraction_errors"] = ["some error"]
    assert state["retry_count"] == 0

    state = retry_extraction(state)
    assert state["retry_count"] == 1


@patch("sentinel_gcp.graph.nodes.retry_extraction.client")
def test_full_retry_loop_succeeds_on_second_attempt(mock_client):
    """End-to-end (mocked) confirmation of the actual designed behavior:
    first extraction fails validation, retry_extraction is called, its
    output THIS TIME is well-formed, and validate_schema running again
    on the retry's output passes cleanly. This is the complete loop
    graph/builder.py's routing is designed to produce."""
    import json

    # First attempt: malformed
    state = _make_test_state_with_raw_extraction(json.loads(_INVALID_EXTRACTION_JSON))
    state = validate_schema(state)
    assert state["extraction_errors"] != []  # confirms it genuinely failed first

    # Retry: mock returns a WELL-FORMED response this time
    mock_client.messages.create.return_value = _make_mock_response(_VALID_EXTRACTION_JSON)
    state = retry_extraction(state)
    assert state["retry_count"] == 1

    # Re-validate the retry's output
    state = validate_schema(state)
    assert state["extraction_errors"] == []
    assert state["extraction"].metadata.trial_identifier.value == "TEST-001"


@patch("sentinel_gcp.graph.nodes.retry_extraction.client")
def test_retry_that_also_fails_leaves_extraction_errors_populated(mock_client):
    """If the retry ALSO produces malformed output, validate_schema
    should fail it again cleanly — this is the case graph/builder.py's
    routing sends to needs_human_exit, since MAX_EXTRACTION_RETRIES=1
    means no further automatic attempts happen."""
    mock_client.messages.create.return_value = _make_mock_response(_INVALID_EXTRACTION_JSON)

    state = _make_test_state_with_raw_extraction(None)
    state["extraction_errors"] = ["initial error"]

    state = retry_extraction(state)
    state = validate_schema(state)

    assert state["extraction_errors"] != []
    assert state["extraction"] is None