# Sentinel-GCP

**AI-powered clinical trial protocol compliance analyzer.**

Sentinel-GCP automates first-pass regulatory compliance review of clinical trial protocols — surfacing gaps for human clinical scientist review before submission. It never auto-approves or auto-rejects; the system's role is to surface what a human should look at, not to replace regulatory judgment.

> Full architecture, design rationale, and every major decision explained in [ARCHITECTURE.md](ARCHITECTURE.md). Individual decisions documented in [docs/adr/](docs/adr/).

---

## Why this exists

Pharmaceutical sponsors spend weeks manually cross-checking a 100+ page protocol against FDA, EMA, and ICH-GCP requirements before submission. Sentinel-GCP automates the mechanical part of that review — freeing human reviewers to focus on the judgment calls that actually need them.

**What makes this different from a generic "AI reads your document" tool:**

- **Deterministic checks never touch an LLM.** A rule engine handles every mechanical presence/absence question (is an IND number present, is a SAE timeline extracted) in plain Python — zero cost, zero hallucination risk. The LLM is reserved for genuinely nuanced judgment.
- **Every AI-generated compliance flag must cite real, retrieved regulatory text.** This isn't a prompt instruction you have to trust — it's enforced at the data-schema level. A flag citing text that wasn't actually retrieved literally cannot be constructed.
- **Jurisdiction-aware, honestly.** If a trial's jurisdiction can't be confidently determined from the extracted data, the system doesn't guess — it narrows retrieval to jurisdiction-agnostic content only and raises a dedicated flag saying so.

---

## Status

**Proven, not just built.** 9 of 14 pipeline nodes are verified end-to-end against real clinical trial protocols with live Claude API and Pinecone calls — not unit tests against synthetic fixtures, actual runs against real documents.

| Layer | Status |
|---|---|
| PDF parsing (Docling + OCR fallback) | Verified — 5.5x speedup found and fixed via real profiling |
| Two-pass extraction (discovery + fill) | Verified against 3 structurally distinct real protocols |
| Schema validation + retry | Verified |
| Rule engine (7 deterministic checks) | Verified — 10/10 unit tests, live regression confirmed |
| Jurisdiction-aware retrieval (Pinecone) | Verified — 212-chunk regulatory corpus, live |
| Agent 2 compliance reasoning | Verified — grounded, cited findings on real data |
| Deep contradiction check | Built, not yet run live |
| Human review, report generation | Built, not yet run live |
| Full compiled graph + Postgres checkpointing | Not yet invoked |

## A real bug, found and fixed

During development, `retrieve()` was silently returning zero results despite a healthy 212-vector Pinecone index and successful `200 OK` responses on every call. Root cause, found through systematic diagnostic isolation rather than guesswork: a Pinecone SDK version change returned `SearchRecordsResponse(result=SearchResult(hits=[...]))` instead of the documented `{"matches": [...]}` shape — the old parsing code was reading a key that no longer existed, failing silently instead of throwing an error.

This is one of six distinct real bugs found and fixed through live testing during this project (see [ARCHITECTURE.md](ARCHITECTURE.md) and the ADRs for the rest) — the kind of thing that only surfaces when you actually run a system against real infrastructure, not when you review the code.

---

## Architecture at a glance

```
PDF -> Parse (Docling, OCR fallback)
    -> Two-pass extraction (discover labels -> fill schema)
    -> Schema validation (retry once, then escalate to human)
    -> Early self-consistency check
    -> Jurisdiction detection -> Deterministic rule engine
    -> Jurisdiction-scoped Pinecone retrieval
    -> Agent 2 compliance reasoning (grounded, cited)
    -> Deep cross-section contradiction check
    -> Human review (pipeline pauses, persists, resumes)
    -> Report generation
```

Full diagrams in `docs/diagrams/`.

---

## Tech stack

`LangGraph` - `Claude API (Sonnet)` - `Docling` / `PyMuPDF` / `Tesseract` - `Pydantic` - `Pinecone` (FAISS local fallback) - `FastAPI` - `Postgres` (Supabase) - `LangSmith`

---

## Project structure

```
sentinel_gcp/        runtime package -- schemas, graph nodes, rules, retrieval, API
ingestion/            one-time/periodic regulatory corpus builder (fetch -> chunk -> upsert)
eval/                  evaluators (extraction, retrieval, groundedness, applicability,
                        confidence calibration), failure taxonomy, ground truth
tests/                  unit + integration tests
docs/adr/                architecture decision records
```

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # fill in real API keys
```

**Windows note:** enable Developer Mode (Settings -> Privacy & security -> For developers) before installing. Docling's model download uses symlinks that a standard Windows account can't create otherwise -- this cost real debugging time before being traced to that specific OS restriction.

## Build the regulatory corpus

```powershell
python ingestion/fetch_regulations.py       # FDA (eCFR), ICH-GCP, EU CTR
python -m ingestion.chunk_and_embed          # section-aware chunking
python -m ingestion.upsert_to_pinecone       # live index
```

## Run pipeline nodes manually

`run_nodes_manual.py` runs individual nodes against a real protocol PDF, with dev-checkpointing (`dev_checkpoint.py`) so iterating on one downstream node doesn't re-pay for `extract_fill`'s expensive full-document call every time.

## Tests

```powershell
pytest tests/ -v
```

---

## Key design decisions

| Decision | Why | Details |
|---|---|---|
| Two-pass extraction | Sponsors label the same field differently -- "Protocol Number" vs "Study Code" vs "PROTOCOL NO." | [ADR 001](docs/adr/001-two-pass-extraction.md) |
| Pinecone over FAISS | Native metadata filtering makes jurisdiction-scoped retrieval a first-class query, not app-level post-filtering | [ADR 002](docs/adr/002-pinecone-over-faiss.md) |
| Two contradiction checks, not one | An early cheap check and a late context-rich check catch genuinely different error classes | [ADR 003](docs/adr/003-dual-contradiction-checks.md) |
| Rule engine before any LLM call | Mechanical checks don't need -- and shouldn't risk -- an LLM's judgment | [ADR 004](docs/adr/004-rule-engine-before-llm.md) |

---

## What this project demonstrates

- Designing multi-agent AI systems for regulated domains, where traceability and graceful failure matter as much as raw model capability
- Building data schemas that survive real-world variance -- discovered by sourcing and testing against structurally different real documents, not assumed upfront
- Separating deterministic and probabilistic reasoning explicitly, using each only where it's actually the right tool
- Debugging real infrastructure (SDK version drift, OS-level permission issues, LLM safety-classifier variability) through systematic isolation rather than guesswork