"""
PineconeStore — production vector store implementation.

Uses Pinecone's native metadata filtering for jurisdiction scoping —
a query with jurisdiction_filter="FDA" only searches chunks tagged
jurisdiction=FDA at the INDEX level, not via application-side
post-filtering. This is the specific reason Pinecone was chosen over
FAISS for production (see ARCHITECTURE.md's Pinecone rationale).
"""
import logging

from pinecone import Pinecone

from sentinel_gcp.retrieval.vector_store import VectorStore, RetrievedChunk
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):
    def __init__(self):
        self._client = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index = self._client.Index(settings.PINECONE_INDEX_NAME)

    def query(
        self,
        query_text: str,
        jurisdiction_filter: str | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        pinecone_filter = None
        if jurisdiction_filter and jurisdiction_filter not in ("both", "unknown"):
            # FIX: use $in, not $eq — a chunk tagged jurisdiction-agnostic
            # ("ICH") must ALSO match a specific-jurisdiction query (FDA or
            # EMA), since ICH-GCP applies across both. An exact-match filter
            # would silently exclude ICH-GCP content from every jurisdiction-
            # specific query, which is wrong — ICH-GCP is exactly the kind
            # of broadly-applicable guidance a compliance check needs
            # regardless of which specific jurisdiction a trial falls under.
            pinecone_filter = {"jurisdiction": {"$in": [jurisdiction_filter, "ICH"]}}

        results = self._index.search(
            namespace="",
            query={"inputs": {"text": query_text}, "top_k": top_k},
            filter=pinecone_filter,
        )
        chunks = [
            RetrievedChunk(
                chunk_id=match["id"],
                text=match["metadata"]["text"],
                regulation_source=match["metadata"].get("regulation_source", "unknown"),
                jurisdiction=match["metadata"].get("jurisdiction", "unknown"),
                score=match["score"],
            )
            for match in results.get("matches", [])
        ]
        logger.info(f"PineconeStore: query returned {len(chunks)} chunk(s), filter={pinecone_filter}")
        return chunks