# ADR 003: Two Separate Contradiction-Check Nodes, Not One Relocated Node

**Status:** Accepted, implemented; deep check verified structurally (mocked), early check verified against real data

## Context

A code review suggested moving the early contradiction check (node 6) to run
after compliance_check, reasoning that it would then have richer context
(regulations + Agent 2 findings) available to detect contradictions.

## Decision

Keep both checks as separate nodes rather than relocating the early one:

- **Node 6 (early)** — cheap, runs on Agent 1's summarized extracted fields
  only, before any retrieval cost is spent. Catches summary-level
  inconsistency (e.g. endpoint references a condition inclusion criteria
  don't screen for).
- **Node 11 (deep)** — expensive, runs after compliance_check, with access to
  raw document section text plus rule-engine and Agent 2 findings. Catches
  cross-section contradictions within the source document itself (e.g. one
  section states a 24-hour SAE window, an older amendment reference states
  7 days) — a genuinely different error class the early check structurally
  cannot see, since it never has raw section text.

## Consequences

- The early check's cost (real, verified): 0 findings on OEV-125,
  cheap and fast, ran before any Pinecone retrieval cost.
- The deep check explicitly distinguishes genuine live contradictions from
  differences explained by amendment/version history — required prompt
  engineering specifically for this, since a naive contradiction check
  would false-positive on every amended protocol.
- Both checks reuse `_build_document_text_block()` from `extract_fill.py`
  identically (deep check imports it directly rather than reimplementing),
  which is also what enables prompt-cache reuse between extract_fill and
  deep_contradiction_check within the same run.