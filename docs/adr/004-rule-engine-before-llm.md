# ADR 004: Deterministic Rule Engine Runs Before Any LLM Compliance Reasoning

**Status:** Accepted, implemented, verified against real data (10/10 unit tests, live OEV-125 run)

## Context

Many compliance checks are mechanical presence/absence questions ("is an IND
number present for an FDA trial") that don't require language understanding.
Routing every check through an LLM would add unnecessary cost, latency, and
hallucination risk to checks that have a single deterministically correct
answer.

## Decision

A plain-Python rule engine (`rules/definitions.py` + `rules/engine.py`) runs
before Agent 2 (the LLM compliance reasoner), checking only canonical,
already-validated extraction fields — never raw document text or
document-specific labels. Currently 7 rules: IND/EudraCT presence
(jurisdiction-gated), SAE timeline presence, inclusion/exclusion/endpoint
presence, and jurisdiction-unknown detection (RULE-007).

## Consequences

- **The single most important regression test in this project** exists
  because of this design: an EMA-jurisdiction trial (OEV-125, no IND,
  EudraCT present) must never trigger the IND-missing rule. Verified both
  in isolated unit tests (`test_rule_engine.py`, 10/10 passing) and live
  against real extracted OEV-125 data (RULE-001 correctly passed).
- Rules operate on canonical fields only, meaning a rule written once works
  identically regardless of whether the source document called something
  "Protocol Number" or "Study Code" — this is what ADR 001's two-pass
  extraction design makes possible.
- Real production value observed: on OEV-125, the rule engine passed 7/7
  checks cleanly, meaning Agent 2's live API call budget was spent entirely
  on the genuinely nuanced question (SAE timeline scope), not re-verifying
  presence checks the rule engine had already settled for free.
- RULE-007 (jurisdiction-unknown) is paired with a retrieval-scoping design
  decision (ADR 002): when jurisdiction can't be determined, retrieval is
  deliberately narrowed to ICH-GCP only, and RULE-007 makes that narrowing
  a visible, high-severity finding in the report rather than a silent
  scoping decision a reviewer would never see.