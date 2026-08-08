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
        chunks = store.query(query_text, jurisdiction_filter=jurisdiction, top_k=top_k)
        for chunk in chunks:
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