"""
VectorStore — abstract interface for regulatory-text retrieval.

Both PineconeStore (production) and FAISSStore (local dev, no API key
needed) implement this same interface. retrieve.py (the graph node)
only ever talks to this interface, never to Pinecone or FAISS directly —
this is what lets VECTOR_STORE_BACKEND in .env swap the backend without
touching any pipeline code, per ARCHITECTURE.md §8.
"""
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    regulation_source: str      # e.g. "21 CFR 312.32"
    jurisdiction: str           # "FDA" | "EMA" | "both"
    section_ref: str | None = None
    score: float                # similarity/relevance score, 0.0-1.0


class VectorStore(ABC):
    @abstractmethod
    def query(
        self,
        query_text: str,
        jurisdiction_filter: Optional[str] = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Returns the top_k most relevant chunks for query_text,
        optionally filtered to a specific jurisdiction. jurisdiction_filter
        of None means no filtering (used for jurisdiction == 'both' or
        'unknown' cases)."""
        raise NotImplementedError