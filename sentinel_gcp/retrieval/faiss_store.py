"""
FAISSStore — local dev/test fallback. No API key needed. Loads a small
pre-built local index (built by ingestion/chunk_and_embed.py) and does
in-memory similarity search, with jurisdiction filtering applied
AFTER retrieval (application-side), since FAISS has no native metadata
filtering the way Pinecone does — this is the concrete tradeoff behind
the "why Pinecone" interview answer.
"""
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from sentinel_gcp.retrieval.vector_store import VectorStore, RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path("data/regulations/faiss_index.bin")
DEFAULT_METADATA_PATH = Path("data/regulations/faiss_metadata.pkl")


class FAISSStore(VectorStore):
    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ):
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path} — run "
                f"ingestion/chunk_and_embed.py first to build the local index"
            )
        self._index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            self._metadata = pickle.load(f)  # list of dicts, aligned by row index to the FAISS index

    def query(
        self,
        query_text: str,
        jurisdiction_filter: str | None = None,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        query_embedding = self._embed(query_text)

        # Over-fetch when a jurisdiction filter is set, since FAISS can't
        # filter natively — we filter AFTER getting results back, so we
        # need extra candidates to still end up with top_k after filtering
        fetch_k = top_k * 5 if jurisdiction_filter else top_k
        distances, indices = self._index.search(np.array([query_embedding]), fetch_k)

        chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            if jurisdiction_filter and jurisdiction_filter not in ("both", "unknown"):
                if meta.get("jurisdiction") != jurisdiction_filter:
                    continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    regulation_source=meta.get("regulation_source", "unknown"),
                    jurisdiction=meta.get("jurisdiction", "unknown"),
                    score=1.0 / (1.0 + float(dist)),  # convert L2 distance to a rough 0-1 score
                )
            )
            if len(chunks) >= top_k:
                break

        logger.info(f"FAISSStore: query returned {len(chunks)} chunk(s) after jurisdiction filtering")
        return chunks

    def _embed(self, text: str) -> np.ndarray:
        # NOTE: embedding model choice is an open decision flagged back in
        # the Pinecone pricing discussion — this needs a real embedding
        # call (e.g. via Anthropic, or a local sentence-transformers model)
        # to actually produce a usable vector. Placeholder raises clearly
        # rather than silently returning garbage.
        raise NotImplementedError(
            "FAISSStore._embed needs a real embedding model wired in — "
            "see the open embedding-model decision from earlier in the project"
        )