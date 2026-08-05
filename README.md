# Sentinel-GCP

AI-powered clinical trial protocol compliance analyzer. Automates first-pass
regulatory compliance review of clinical trial protocols — surfacing gaps for
human clinical scientist review, never auto-approving or auto-rejecting.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design and
rationale, and [docs/adr/](docs/adr/) for individual architectural decisions.

## Status

Core pipeline: parsing, two-pass extraction, schema validation, rule engine,
retrieval, and Agent 2 compliance reasoning (9 of 14 nodes) are proven working
end-to-end against real clinical trial protocols with live Claude API and
Pinecone calls — not just unit-tested in isolation.

Full ingestion pipeline (FDA + ICH-GCP + EU CTR regulatory corpus: fetch,
chunk, embed, upsert) is live and verified in a production Pinecone index.

Remaining: `deep_contradiction_check` through `generate_report` need live
testing; the full compiled LangGraph with real Postgres/Supabase
checkpointing has not yet been invoked end-to-end.

## Tech Stack

LangGraph · Claude API (Sonnet) · Docling/PyMuPDF/Tesseract · Pydantic ·
Pinecone (FAISS local fallback) · FastAPI · Postgres (Supabase) · LangSmith

## Project Structure

```
sentinel_gcp/       # runtime package — schemas, graph nodes, rules, retrieval, API
ingestion/           # one-time/periodic corpus builder (fetch/chunk/upsert regulations)
eval/                 # evaluators (extraction, retrieval, groundedness, applicability,
                       # calibration), failure taxonomy, ground truth
tests/                 # unit + integration tests
docs/adr/               # architecture decision records
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in real API keys
```

Requires Windows Developer Mode enabled (Settings → Privacy & security → For
developers) before installing — Docling's model download uses symlinks that
regular Windows accounts cannot create otherwise.

## Running the Ingestion Pipeline

```powershell
python ingestion/fetch_regulations.py
python -m ingestion.chunk_and_embed
python -m ingestion.upsert_to_pinecone
```

## Manual Pipeline Testing

`run_nodes_manual.py` runs pipeline nodes individually against a real
protocol PDF, with dev-checkpointing (`dev_checkpoint.py`) to avoid
re-paying for expensive stages (especially `extract_fill`'s full-document
call) while iterating on a single downstream node.

## Tests

```powershell
pytest tests/ -v
```

## Key Design Decisions

- **Two-pass extraction** — a document-labeling discovery pass before the
  real extraction pass, so field labels varying by sponsor/jurisdiction
  don't require hardcoded assumptions. See [ADR 001](docs/adr/001-two-pass-extraction.md).
- **Deterministic rule engine before any LLM compliance call** — mechanical
  presence/absence checks never touch an LLM; only genuinely nuanced
  judgment calls reach Agent 2. See [ADR 004](docs/adr/004-rule-engine-before-llm.md).
- **Jurisdiction-aware retrieval** — FDA/EMA/ICH-GCP content is scoped via
  Pinecone's native metadata filtering; an unresolved jurisdiction narrows
  retrieval to jurisdiction-agnostic ICH-GCP content only, rather than
  guessing. See [ADR 002](docs/adr/002-pinecone-over-faiss.md).
- **Every Agent 2 flag must cite real, retrieved evidence** — enforced at
  the Pydantic schema level (`ComplianceFlag`'s model_validator), not just
  by prompt instruction.