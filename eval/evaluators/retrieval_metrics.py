"""
eval/evaluators/retrieval_metrics.py — retrieval@k for Agent 2's
regulatory retrieval, measured against hand-labeled relevant chunks.

Given a query (e.g. "SAE reporting timeline requirement") and the set
of chunk_ids a human labeled as genuinely relevant, checks how many of
the top-k retrieved chunks actually match — retrieval@k, the standard
IR metric. Distinct from extraction_metrics.py: this measures whether
retrieve.py (node 9) is finding the RIGHT regulation text, not whether
Agent 1's extraction was correct.

Ground truth here is a mapping: query -> set of chunk_ids that SHOULD
be retrieved. Building this requires knowing the actual chunk_ids in
your Pinecone index — see build_relevance_ground_truth() below for how
to bootstrap this from a real upsert run.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    query: str
    precision_at_k: float
    recall_at_k: float
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    matched_chunk_ids: list[str]


def evaluate_retrieval(
    query: str,
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
) -> RetrievalMetrics:
    """Core retrieval@k computation. retrieved_chunk_ids is ordered
    (as returned by the vector store, best-match first) but this
    computes unordered precision/recall@k — NOT rank-sensitive metrics
    like MRR or NDCG. That's a deliberate scope decision: unordered
    precision/recall@k is simpler to hand-label ground truth for (you
    only need to know WHICH chunks are relevant, not their ideal rank
    order) and is sufficient for validating that retrieve.py's
    jurisdiction filtering + query construction are working — rank-
    sensitive metrics would be the next layer of rigor if unordered
    precision/recall@k turns out insufficient once run against real data."""
    retrieved_set = set(retrieved_chunk_ids)
    matched = retrieved_set & relevant_chunk_ids

    precision = len(matched) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(matched) / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0

    return RetrievalMetrics(
        query=query,
        precision_at_k=round(precision, 3),
        recall_at_k=round(recall, 3),
        retrieved_chunk_ids=retrieved_chunk_ids,
        relevant_chunk_ids=sorted(relevant_chunk_ids),
        matched_chunk_ids=sorted(matched),
    )


def evaluate_retrieval_suite(
    test_cases: list[dict],
) -> dict:
    """Runs evaluate_retrieval() across multiple query/ground-truth pairs
    and aggregates. test_cases shape:
        [{"query": str, "retrieved_chunk_ids": list[str], "relevant_chunk_ids": set[str]}, ...]
    This is what run_eval.py (not yet built) will call across your
    full ground-truth query set, not just one query at a time."""
    results = [evaluate_retrieval(**case) for case in test_cases]

    avg_precision = sum(r.precision_at_k for r in results) / len(results) if results else 0.0
    avg_recall = sum(r.recall_at_k for r in results) / len(results) if results else 0.0

    zero_recall_queries = [r.query for r in results if r.recall_at_k == 0.0]
    if zero_recall_queries:
        logger.warning(
            f"evaluate_retrieval_suite: {len(zero_recall_queries)} quer(y/ies) "
            f"retrieved ZERO relevant chunks: {zero_recall_queries} — likely "
            f"jurisdiction filtering or missing corpus content, not a scoring bug"
        )

    return {
        "average_precision_at_k": round(avg_precision, 3),
        "average_recall_at_k": round(avg_recall, 3),
        "per_query_results": results,
        "zero_recall_queries": zero_recall_queries,
    }


def build_relevance_ground_truth_template(chunk_ids_by_source: dict[str, list[str]]) -> dict:
    """Helper for BOOTSTRAPPING ground truth, not for running eval itself.
    Given the real chunk_ids currently in your Pinecone index (grouped
    by regulation_source, e.g. from a debug query), produces a starter
    template you fill in by hand — marking which chunks are actually
    relevant to which known compliance-check queries. This doesn't
    replace human judgment; it just saves you from hand-typing chunk_ids
    that only exist after a real upsert run.

    Example usage after running upsert_to_pinecone.py:
        # query Pinecone directly, group results by regulation_source
        chunk_ids_by_source = {"21 CFR 312.32": ["chunk-a1b2...", "chunk-c3d4..."], ...}
        template = build_relevance_ground_truth_template(chunk_ids_by_source)
        # then hand-edit template to mark true relevance per query
    """
    return {
        "sae_reporting": {
            "candidate_chunk_ids_from_312_32": chunk_ids_by_source.get("21 CFR 312.32", []),
            "relevant_chunk_ids": [],  # FILL IN BY HAND after reviewing candidates
        },
        "protocol_amendments": {
            "candidate_chunk_ids_from_312_30": chunk_ids_by_source.get("21 CFR 312.30", []),
            "relevant_chunk_ids": [],
        },
        "eligibility": {
            "candidate_chunk_ids_ich_gcp": chunk_ids_by_source.get("ICH E6(R3) GCP", []),
            "relevant_chunk_ids": [],
        },
    }