# ADR 001: Two-Pass Extraction (Discovery + Fill)

**Status:** Accepted, implemented, verified against real data (OEV-125)

## Context

Real clinical trial protocols use inconsistent labeling for the same concepts.
Across the three protocols sourced for this project:

| Concept | NEOD001-CL002 | OEV-125 | ARCT-165-01 |
|---|---|---|---|
| Trial identifier label | "Protocol Number" | "Study code:" | "PROTOCOL NO." |
| Phase notation | "Phase 3" | "Phase 2" | "1/2" (combined) |
| Jurisdiction identifier | US IND Number | EudraCT No. | Neither present |

A single-pass extraction prompt ("extract these fields") either has to hardcode
assumptions about labeling, or risk missing fields whose labels don't match
what the prompt expects.

## Decision

Split extraction into two LLM calls:
1. **Discovery** — a cheap pass over the first few pages + headings, asking
   the model to report what THIS document calls each concept.
2. **Fill** — the full extraction pass, given the discovery pass's label map
   as guidance, populating the canonical schema regardless of source wording.

## Consequences

- Two LLM calls per protocol instead of one, but discovery's call is small
  (~2-3K tokens) — a minor cost addition relative to fill's ~40K+ token cost.
- Extraction correctly handled all three test protocols' different labeling
  without any document-specific code.
- Verified in production use: on OEV-125, discovery correctly identified
  "Study code:" as the identifier label and "EudraCT No." before fill ran,
  producing a fully valid extraction on the first attempt.