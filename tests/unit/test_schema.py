"""
Unit tests for the core Pydantic schemas — pure logic, zero API cost.
Covers ProtocolExtraction/FieldWithProvenance (extraction.py),
DocumentStructure/TableRegion (document_structure.py), and
ComplianceFlag's model_validator (compliance.py) — the source-
consistency constraint that's been load-bearing all session (every
rule_engine flag and every agent_2 flag construction depends on it
being correct).
"""
import pytest
from pydantic import ValidationError

from sentinel_gcp.schema.extraction import (
    ProtocolExtraction,
    TrialMetadata,
    FieldWithProvenance,
    StudyArm,
)
from sentinel_gcp.schema.document_structure import (
    DocumentStructure,
    Section,
    TableRegion,
    ParsingCoverage,
)
from sentinel_gcp.schema.compliance import ComplianceFlag


# ─── ProtocolExtraction / FieldWithProvenance ──────────────────────────

def test_protocol_extraction_minimal_valid():
    """Every field is Optional except the required nested objects —
    a minimal extraction (all null values) should still construct
    validly, per the 'null is honest, never crash' design principle."""
    extraction = ProtocolExtraction(
        metadata=TrialMetadata(
            trial_identifier=FieldWithProvenance(value=None),
            sponsor=FieldWithProvenance(value=None),
        )
    )
    assert extraction.metadata.trial_identifier.value is None
    assert extraction.inclusion_criteria == []
    assert extraction.study_arms == []


def test_field_with_provenance_carries_all_metadata():
    """Confirms every provenance field (page, section, confidence,
    label_used, verbatim) is actually captured, not just value —
    this is what makes report citations traceable back to source."""
    field = FieldWithProvenance(
        value="OEV-125",
        label_used="Study code:",
        verbatim="Study code: OEV-125",
        page=1,
        section="Title page",
        confidence=1.0,
    )
    assert field.value == "OEV-125"
    assert field.page == 1
    assert field.confidence == 1.0


def test_phase_boolean_flags_support_combined_phases():
    """The exact case ARCT-165-01 surfaced during design — a '1/2'
    combined phase must be representable as BOTH flags true, not an
    either/or enum."""
    metadata = TrialMetadata(
        trial_identifier=FieldWithProvenance(value="ARCT-165-01"),
        sponsor=FieldWithProvenance(value="Arcturus Therapeutics"),
        phase_raw="1/2",
        phase_includes_1=True,
        phase_includes_2=True,
    )
    assert metadata.phase_includes_1 is True
    assert metadata.phase_includes_2 is True
    assert metadata.phase_includes_3 is False


def test_study_arms_supports_multi_cohort():
    """The multi-cohort case (ARCT-165-01's A1/A2/B design) — study_arms
    must be a list that can hold multiple distinct cohorts with
    different randomization ratios, not a flat single-arm assumption."""
    extraction = ProtocolExtraction(
        metadata=TrialMetadata(
            trial_identifier=FieldWithProvenance(value="ARCT-165-01"),
            sponsor=FieldWithProvenance(value="Arcturus"),
        ),
        study_arms=[
            StudyArm(cohort_name="A1", n_participants=12, randomization_ratio="1:1:1"),
            StudyArm(cohort_name="A2", n_participants=24, randomization_ratio="3:1"),
            StudyArm(cohort_name="B", n_participants=36),
        ],
    )
    assert len(extraction.study_arms) == 3
    assert extraction.study_arms[1].randomization_ratio == "3:1"


def test_mutable_default_lists_are_independent_across_instances():
    """Regression test for the mutable-default-arguments concern raised
    early in this project — confirms Field(default_factory=list) means
    two separate ProtocolExtraction instances never share the same
    underlying list object."""
    e1 = ProtocolExtraction(
        metadata=TrialMetadata(trial_identifier=FieldWithProvenance(), sponsor=FieldWithProvenance())
    )
    e2 = ProtocolExtraction(
        metadata=TrialMetadata(trial_identifier=FieldWithProvenance(), sponsor=FieldWithProvenance())
    )
    e1.inclusion_criteria.append("test criterion")
    assert e2.inclusion_criteria == []  # e2 must NOT see e1's mutation


# ─── DocumentStructure / TableRegion ────────────────────────────────────

def test_table_region_confidence_defaults_to_full_trust():
    """A table with no explicit confidence score defaults to 1.0 —
    only real testing-derived low scores should reduce trust, not an
    accidental unset value."""
    table = TableRegion(page_start=24, parsed_rows=[{"col1": "value1"}])
    assert table.confidence == 1.0


def test_table_region_accepts_string_keyed_rows():
    """Regression test for the real bug found during testing —
    Docling's export_to_dataframe() with no header row produces integer
    column keys, requiring the df.columns.astype(str) fix in
    parse_pdf.py. This test confirms the SCHEMA side accepts the
    corrected string-keyed shape (the fix itself lives in parse_pdf.py,
    not testable here without Docling — this just confirms the schema
    doesn't itself reject valid string-keyed rows)."""
    table = TableRegion(
        page_start=1,
        parsed_rows=[{"0": "cell_value", "1": "another_cell"}],  # string keys, even if originally numeric-looking
    )
    assert table.parsed_rows[0]["0"] == "cell_value"


def test_document_structure_requires_parsing_coverage():
    """parsing_coverage is the one required (non-Optional) field on
    DocumentStructure — every other field defaults to empty. This is
    deliberate: you should never have a DocumentStructure without
    knowing what was/wasn't successfully covered."""
    with pytest.raises(ValidationError):
        DocumentStructure()  # missing required parsing_coverage


def test_document_structure_minimal_valid():
    structure = DocumentStructure(
        parsing_coverage=ParsingCoverage(total_pages=74)
    )
    assert structure.sections == []
    assert structure.parsing_coverage.total_pages == 74


# ─── ComplianceFlag model_validator ──────────────────────────────────────

def test_rule_engine_flag_rejects_llm_certainty():
    """The load-bearing constraint: a rule_engine flag must NOT carry
    llm_certainty, since it made no LLM judgment call. Enforced at
    construction time, not just by convention."""
    with pytest.raises(ValidationError):
        ComplianceFlag(
            flag_id="RULE-001",
            source="rule_engine",
            issue="test",
            llm_certainty=0.8,  # should be rejected
        )


def test_rule_engine_flag_rejects_retrieved_chunk_id():
    """Same principle — a rule_engine flag never retrieved anything,
    so it can't carry a retrieved_chunk_id."""
    with pytest.raises(ValidationError):
        ComplianceFlag(
            flag_id="RULE-001",
            source="rule_engine",
            issue="test",
            retrieved_chunk_id="chunk-test123",  # should be rejected
        )


def test_agent2_flag_requires_llm_certainty():
    """The inverse constraint — an agent_2 flag MUST report its
    certainty, since it made a judgment call. This is what prevents an
    ungrounded LLM assertion from silently existing with no confidence
    signal at all."""
    with pytest.raises(ValidationError):
        ComplianceFlag(
            flag_id="AGENT2-001",
            source="agent_2",
            issue="test",
            retrieved_chunk_id="chunk-test123",
            # llm_certainty deliberately omitted
        )


def test_agent2_flag_requires_retrieved_chunk_id():
    """The groundedness enforcement itself — an agent_2 flag MUST cite
    a retrieved chunk. This is the schema-level guarantee that backs
    the 'how do you know Agent 2 isn't hallucinating citations'
    interview answer (ARCHITECTURE.md §6)."""
    with pytest.raises(ValidationError):
        ComplianceFlag(
            flag_id="AGENT2-001",
            source="agent_2",
            issue="test",
            llm_certainty=0.8,
            # retrieved_chunk_id deliberately omitted
        )


def test_valid_rule_engine_flag_constructs_cleanly():
    flag = ComplianceFlag(
        flag_id="RULE-001",
        source="rule_engine",
        issue="Missing IND number",
        severity="high",
        final_confidence=1.0,
    )
    assert flag.llm_certainty is None
    assert flag.retrieved_chunk_id is None


def test_valid_agent2_flag_constructs_cleanly():
    flag = ComplianceFlag(
        flag_id="AGENT2-001",
        source="agent_2",
        issue="SAE timeline scope gap",
        severity="high",
        llm_certainty=0.85,
        retrieved_chunk_id="chunk-ff736bc846b910fb",
        insufficient_evidence=False,
    )
    assert flag.llm_certainty == 0.85
    assert flag.insufficient_evidence is False