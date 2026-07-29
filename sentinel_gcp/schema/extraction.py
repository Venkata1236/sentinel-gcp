"""
Core+extension schema for Agent 1 (Protocol Extraction Agent) —
Sentinel-GCP

Design principles (validated against 3 structurally distinct real protocols:
NEOD001-CL002 [Phase 3, single 2-arm, has US IND], OEV-125 [Phase 2, EU/EudraCT
only, no IND], ARCT-165-01 [Phase 1/2 combined, multi-cohort with sub-cohorts]):

1. Every field is Optional — a missing field is `null`, never a crash or a
   hallucinated value.
2. Identifying fields carry provenance (label_used, verbatim, page, section)
   so a downstream validator can flag "the model used a label we've never
   seen before" instead of silently trusting it.
3. Phase is stored as both the raw string (handles "1/2", "IIb", etc.) AND
   as boolean flags (phase_includes_1/2/3/4) so compliance rules check
   `phase_includes_1 == True` instead of a brittle `phase == "Phase 1"`
   string match that silently misses combined-phase trials.
4. study_arms is always a list (empty for single-arm trials) — never a
   single flat set of criteria — so multi-cohort/sub-cohort designs
   (e.g. Cohort A1/A2/B with different populations and randomization
   ratios) can be represented without merging or dropping data.
5. additional_metadata is a catch-all list for anything encountered that
   doesn't fit the core fields — new variance gets captured here first,
   then promoted to a core field once it recurs across enough documents.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class FieldWithProvenance(BaseModel):
    value: Optional[str] = None
    label_used: Optional[str] = None   # exact label as it appears in the source doc
    verbatim: Optional[str] = None     # exact quoted snippet, for audit trail
    page: Optional[int] = None         # source page number
    section: Optional[str] = None      # source section ID (e.g. "9.6")
    confidence: Optional[float] = None # 0.0-1.0, per-field extraction confidence


class StudyArm(BaseModel):
    cohort_name: Optional[str] = None
    n_participants: Optional[int] = None
    randomization_ratio: Optional[str] = None
    population_description: Optional[str] = None


class TrialMetadata(BaseModel):
    trial_identifier: FieldWithProvenance
    sponsor: FieldWithProvenance
    phase_raw: Optional[str] = None
    phase_includes_1: bool = False
    phase_includes_2: bool = False
    phase_includes_3: bool = False
    phase_includes_4: bool = False
    ind_number: Optional[FieldWithProvenance] = None
    eudract_number: Optional[FieldWithProvenance] = None


class ProtocolExtraction(BaseModel):
    metadata: TrialMetadata
    study_arms: List[StudyArm] = Field(default_factory=list)
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    primary_endpoint: Optional[str] = None
    secondary_endpoints: List[str] = Field(default_factory=list)
    sae_reporting_timeline: Optional[FieldWithProvenance] = None
    additional_metadata: List[dict] = Field(default_factory=list)