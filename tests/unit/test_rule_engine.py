"""
Unit tests for the rule engine — no LLM call, no LangGraph, no PDF
parsing needed. Pure Python logic tests against the Rule definitions
and execution engine.

The OEV-125 test below is the single most important regression test in
this project: it's the exact bug found during design (an EudraCT-only
trial with no IND number would incorrectly trigger RULE-001 if
jurisdiction gating were missing or broken). If this test ever starts
failing, it means the jurisdiction-gating fix has regressed.
"""
import pytest

from sentinel_gcp.rules.engine import run_rules
from sentinel_gcp.schema.extraction import (
    ProtocolExtraction,
    TrialMetadata,
    FieldWithProvenance,
)


def _make_extraction(
    ind_value: str | None = None,
    eudract_value: str | None = None,
    sae_value: str | None = "within 24 hours",
    inclusion: list[str] | None = None,
    exclusion: list[str] | None = None,
    endpoint: str | None = "some endpoint",
) -> ProtocolExtraction:
    """Small factory to build a minimal-but-valid ProtocolExtraction for
    each test case, without needing a full real document."""
    return ProtocolExtraction(
        metadata=TrialMetadata(
            trial_identifier=FieldWithProvenance(value="TEST-001"),
            sponsor=FieldWithProvenance(value="Test Sponsor"),
            phase_raw="Phase 2",
            phase_includes_2=True,
            ind_number=FieldWithProvenance(value=ind_value) if ind_value else FieldWithProvenance(value=None),
            eudract_number=FieldWithProvenance(value=eudract_value) if eudract_value else FieldWithProvenance(value=None),
        ),
        inclusion_criteria=inclusion if inclusion is not None else ["age 18-50"],
        exclusion_criteria=exclusion if exclusion is not None else ["pregnancy"],
        primary_endpoint=endpoint,
        sae_reporting_timeline=FieldWithProvenance(value=sae_value) if sae_value else None,
    )


def test_oev125_regression_ema_trial_never_triggers_ind_rule():
    """THE regression test. An EMA-jurisdiction trial (EudraCT present,
    no IND — exactly OEV-125's real shape) must NEVER trigger RULE-001,
    even though ind_number.value is None. This is the bug that was found
    and fixed during design — see ARCHITECTURE.md §3 and CHECKPOINTS.md."""
    extraction = _make_extraction(
        ind_value=None,
        eudract_value="2021-001541-13",  # OEV-125's real EudraCT number
    )
    results = run_rules(extraction, jurisdiction="EMA")

    rule_001_result = next(r for r in results if r.rule_id == "RULE-001")
    assert rule_001_result.passed is True, (
        "RULE-001 (missing IND for FDA) incorrectly fired on an EMA-only trial — "
        "jurisdiction gating is broken"
    )

    rule_002_result = next(r for r in results if r.rule_id == "RULE-002")
    assert rule_002_result.passed is True, (
        "RULE-002 (missing EudraCT for EMA) incorrectly fired even though "
        "a EudraCT number IS present"
    )


def test_fda_trial_with_missing_ind_correctly_flags():
    """The inverse case — an FDA-jurisdiction trial genuinely missing its
    IND number SHOULD trigger RULE-001. Confirms the gate isn't just
    disabling the rule entirely, only correctly scoping it."""
    extraction = _make_extraction(ind_value=None, eudract_value=None)
    results = run_rules(extraction, jurisdiction="FDA")

    rule_001_result = next(r for r in results if r.rule_id == "RULE-001")
    assert rule_001_result.passed is False
    assert rule_001_result.flag is not None
    assert rule_001_result.flag.severity == "high"
    assert rule_001_result.flag.source == "rule_engine"


def test_fda_trial_with_ind_present_passes():
    """NEOD001-CL002's real shape — FDA trial WITH an IND number present
    should pass RULE-001 cleanly."""
    extraction = _make_extraction(ind_value="122,912", eudract_value=None)
    results = run_rules(extraction, jurisdiction="FDA")

    rule_001_result = next(r for r in results if r.rule_id == "RULE-001")
    assert rule_001_result.passed is True


def test_unknown_jurisdiction_skips_both_jurisdiction_rules():
    """Neither IND nor EudraCT found — jurisdiction is 'unknown'. Neither
    RULE-001 nor RULE-002 should fire, since we can't determine which
    applies (per determine_jurisdiction.py's design note)."""
    extraction = _make_extraction(ind_value=None, eudract_value=None)
    results = run_rules(extraction, jurisdiction="unknown")

    rule_001_result = next(r for r in results if r.rule_id == "RULE-001")
    rule_002_result = next(r for r in results if r.rule_id == "RULE-002")
    assert rule_001_result.passed is True
    assert rule_002_result.passed is True


def test_missing_sae_timeline_flags():
    extraction = _make_extraction(sae_value=None)
    results = run_rules(extraction, jurisdiction="FDA")

    rule_003_result = next(r for r in results if r.rule_id == "RULE-003")
    assert rule_003_result.passed is False
    assert rule_003_result.flag.severity == "medium"


def test_empty_inclusion_criteria_flags_as_likely_extraction_failure():
    extraction = _make_extraction(inclusion=[])
    results = run_rules(extraction, jurisdiction="FDA")

    rule_004_result = next(r for r in results if r.rule_id == "RULE-004")
    assert rule_004_result.passed is False
    assert "extraction" in rule_004_result.flag.recommendation.lower()


def test_all_rules_pass_on_well_formed_fda_trial():
    """A NEOD001-CL002-shaped trial — everything present — should pass
    every single rule, 0 flags. Matches the real document-trace
    walkthrough result from earlier in the project."""
    extraction = _make_extraction(ind_value="122,912", eudract_value=None)
    results = run_rules(extraction, jurisdiction="FDA")

    failed = [r for r in results if not r.passed]
    assert failed == [], f"Expected 0 flags on a well-formed trial, got: {[r.rule_id for r in failed]}"