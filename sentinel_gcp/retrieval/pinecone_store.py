"""
PineconeStore — production vector store implementation.

Jurisdiction-scoped retrieval, resolved per real-document testing
(ARCT-165-01 surfaced the "unknown" jurisdiction case — neither IND nor
EudraCT found):

  FDA     -> search FDA + ICH content
  EMA     -> search EMA + ICH content
  both    -> search FDA + EMA + ICH (a CONFIRMED dual-filed trial —
             both identifiers were actually found, this is a fact,
             not uncertainty)
  unknown -> search ICH ONLY (neither identifier found — genuine
             uncertainty). Deliberately does NOT fall back to searching
             FDA+EMA: citing a specific national regulation (e.g. 21 CFR
             312.32) on a trial whose jurisdiction was never confirmed
             risks a confidently-wrong citation, not just an imprecise
             one — the one failure mode this whole project is built to
             prevent. ICH-GCP is always safe to include since it applies
             regardless of which national framework governs.
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
        allowed_jurisdictions = self._resolve_allowed_jurisdictions(jurisdiction_filter)
        pinecone_filter = (
            {"jurisdiction": {"$in": allowed_jurisdictions}}
            if allowed_jurisdictions is not None
            else None
        )

        # FIX: this Pinecone SDK version rejects the old nested query={...}
        # dict combined with a separate filter= argument — real testing
        # error: "received both 'query=' and 'filter'. Pass either the
        # legacy query=SearchQuery(...) form OR the new flat keyword
        # arguments, not both." Switched to the flat kwarg form.
        results = self._index.search(
            namespace="default",
            inputs={"text": query_text},
            top_k=top_k,
            filter=pinecone_filter,
        )

        print(f"DEBUG RAW PINECONE RESPONSE: {results}")  # TEMPORARY — remove once bug is found
        print(f"DEBUG RAW RESPONSE TYPE: {type(results)}")  # TEMPORARY

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
        logger.info(
            f"PineconeStore: query returned {len(chunks)} chunk(s), "
            f"jurisdiction_filter={jurisdiction_filter}, allowed={allowed_jurisdictions}"
        )
        return chunks

    @staticmethod
    def _resolve_allowed_jurisdictions(jurisdiction_filter: str | None) -> list[str] | None:
        """Maps a trial's determined jurisdiction to the set of
        regulation-jurisdiction tags that should be searched. Returns
        None for no filtering at all (used only if jurisdiction_filter
        itself is None, i.e. not yet determined at all)."""
        if jurisdiction_filter is None:
            return None
        if jurisdiction_filter == "FDA":
            return ["FDA", "ICH"]
        if jurisdiction_filter == "EMA":
            return ["EMA", "ICH"]
        if jurisdiction_filter == "both":
            return ["FDA", "EMA", "ICH"]
        # "unknown" — neither IND nor EudraCT found. Deliberately narrow,
        # not broad: ICH-GCP only, per module docstring above.
        return ["ICH"]