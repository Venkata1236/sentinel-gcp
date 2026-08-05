# ADR 002: Pinecone as Primary Vector Store, FAISS as Local Fallback

**Status:** Accepted, implemented, verified — 212 real chunks live in production index

## Context

The rule engine already distinguishes FDA, EMA, ICH-GCP, and "unknown"
jurisdiction. Regulatory retrieval needs to respect that distinction — an
FDA-only trial shouldn't retrieve EMA-specific procedural law, and vice versa,
while ICH-GCP (jurisdiction-agnostic) should be retrievable by either.

## Decision

Use Pinecone as the primary vector store, behind a `VectorStore` abstract
interface, with FAISS available as a local, no-API-key dev fallback via the
same interface.

**Why Pinecone specifically:** native metadata filtering makes jurisdiction
scoping a first-class query parameter (`filter={"jurisdiction": {"$in": [...]}}`)
rather than something the application layer has to post-filter in code.

## Consequences

- Real jurisdiction bug caught and fixed BECAUSE of this design: initial
  `$eq` filter logic would have made ICH-GCP content invisible to
  jurisdiction-specific queries entirely (fixed to `$in`, tested against
  real EMA and FDA cases).
- A second real jurisdiction case (ARCT-165-01: neither IND nor EudraCT
  found) drove a further refinement — "unknown" jurisdiction retrieval
  scoped to ICH-GCP only, never FDA or EMA specifics, to avoid a
  confidently-wrong citation on a trial whose jurisdiction was never
  confirmed. This is RULE-007's paired design.
- A real SDK version mismatch was found and fixed during production
  upsert/query testing: this Pinecone SDK version returns
  `SearchRecordsResponse(result=SearchResult(hits=[...]))`, not the
  documented-elsewhere `{"matches": [...]}` shape — found via systematic
  diagnostic isolation (confirmed upsert succeeded, confirmed index stats,
  confirmed raw response shape) rather than assumption.
- FAISS's `_embed()` remains unimplemented (`NotImplementedError`) — the
  fallback path is structurally ready but was never exercised, since
  Pinecone's hosted embeddings made a separate embedding provider
  unnecessary for the primary path.