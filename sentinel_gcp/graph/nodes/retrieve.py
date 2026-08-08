"""
retrieve — Node 9 of the Sentinel-GCP pipeline.

Deterministic + embeddings (no reasoning LLM call — the embedding model
call isn't "reasoning," it's a vector lookup). Runs one PER-TOPIC query
per compliance-relevant field (SAE timeline, eligibility, endpoints —
NOT one blanket per-document query), jurisdiction-scoped via the
VectorStore interface. Backend (Pinecone vs FAISS) is chosen via
VECTOR_STORE_BACKEND in config, never hardcoded here.
"""
import logging

from sentinel_gcp.retrieval.vector_store import VectorStore
from sentinel_gcp.retrieval.pinecone_store import PineconeStore
from sentinel_gcp.retrieval.faiss_store import FAISSStore
from sentinel_gcp.graph.state import GraphState
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

# Not uniform — SAE reporting and eligibility criteria each cover multiple
# distinct regulatory sub-rules (reporting-relationship variants, timeline
# thresholds; inclusion vs. exclusion vs. withdrawal-adjacent requirements
# respectively) so a single top-3 retrieval is more likely to miss a
# relevant chunk than for endpoints, which is usually governed by one or
# two closely-related regulatory passages. protocol_amendments kept at
# the prior default (3) — no evidence yet either way; revisit once
# eval/evaluators/retrieval_metrics.py is actually run against real
# queries rather than guessed from topic shape alone.
_TOPIC_TOP_K = {
    "sae_reporting": 5,
    "eligibility": 4,
    "protocol_amendments": 3,
    "endpoints": 2,
}
_DEFAULT_TOP_K = 3

# Re-ranking: overfetch more candidates than we'll keep, then re-score
# combining Pinecone's vector similarity with a simple keyword-overlap
# signal against the topic query, keeping only the final top_k. No new
# dependency (no cross-encoder, no reranking API) — deliberately the
# cheapest option, chosen as the default absent a specific preference for
# a cross-encoder or Cohere Rerank. Revisit if eval/evaluators/
# retrieval_metrics.py run against real queries shows this isn't enough.
_OVERFETCH_MULTIPLIER = 3
_RERANK_SCORE_WEIGHT = 0.7   # Pinecone vector similarity
_RERANK_KEYWORD_WEIGHT = 0.3  # keyword overlap with the topic query

_STOPWORDS = {
    "a", "an", "the", "for", "of", "to", "and", "or", "in", "on", "with",
    "requirements", "requirement",  # near-universal in every topic query here — no discriminating power
}


def _keyword_overlap_score(query_text: str, chunk_text: str) -> float:
    """Fraction of the query's meaningful words that actually appear in
    the chunk. Overlap coefficient relative to query length (not
    Jaccard) — a long chunk containing all the query's words shouldn't
    score worse than a short one just for also containing other text."""
    query_words = {w for w in query_text.lower().split() if w not in _STOPWORDS and len(w) > 2}
    if not query_words:
        return 0.0
    chunk_words = set(chunk_text.lower().split())
    return len(query_words & chunk_words) / len(query_words)


def _get_vector_store() -> VectorStore:
    if settings.VECTOR_STORE_BACKEND == "pinecone":
        return PineconeStore()
    return FAISSStore()


def retrieve(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['extraction'] and
    state['jurisdiction'], writes state['retrieved_chunks'] — a flat
    list across all topic queries, each RetrievedChunk still tagged
    with which topic query produced it via a synthetic 'topic' field
    added at this layer (not part of RetrievedChunk itself, added here
    so compliance_check can group by topic)."""
    extraction = state["extraction"]
    jurisdiction = state["jurisdiction"]

    if extraction is None:
        raise ValueError("retrieve requires a validated ProtocolExtraction")

    store = _get_vector_store()
    all_chunks = []

    topic_queries = _build_topic_queries(extraction)
    for topic, query_text in topic_queries.items():
        if query_text is None:
            continue  # nothing extracted for this topic — skip the query, don't waste a call
        top_k = _TOPIC_TOP_K.get(topic, _DEFAULT_TOP_K)
        overfetch_k = top_k * _OVERFETCH_MULTIPLIER
        candidates = store.query(query_text, jurisdiction_filter=jurisdiction, top_k=overfetch_k)

        reranked = sorted(
            candidates,
            key=lambda c: (
                _RERANK_SCORE_WEIGHT * c.score
                + _RERANK_KEYWORD_WEIGHT * _keyword_overlap_score(query_text, c.text)
            ),
            reverse=True,
        )[:top_k]

        for chunk in reranked:
            all_chunks.append({"topic": topic, **chunk.model_dump()})

    state["retrieved_chunks"] = all_chunks
    per_topic_counts = {}
    for c in all_chunks:
        per_topic_counts[c["topic"]] = per_topic_counts.get(c["topic"], 0) + 1
    logger.info(
        f"retrieve: {len(topic_queries)} topic quer(y/ies) issued, "
        f"{len(all_chunks)} total chunk(s) retrieved ({per_topic_counts})"
    )
    return state


def _build_topic_queries(extraction) -> dict[str, str | None]:
    """One query per compliance-relevant topic — deliberately NOT one
    query for the whole document. Per ARCHITECTURE.md, smaller targeted
    retrieval improves reasoning quality over blanket document-level
    retrieval."""
    return {
        "sae_reporting": (
            f"SAE reporting timeline requirements: {extraction.sae_reporting_timeline.value}"
            if extraction.sae_reporting_timeline and extraction.sae_reporting_timeline.value
            else None
        ),
        "eligibility": (
            "eligibility criteria requirements for clinical trial protocols"
            if extraction.inclusion_criteria or extraction.exclusion_criteria
            else None
        ),
        "protocol_amendments": "protocol amendment procedure requirements",
        "endpoints": (
            f"primary endpoint definition requirements: {extraction.primary_endpoint}"
            if extraction.primary_endpoint
            else None
        ),
    }