"""
ingestion/upsert_to_pinecone.py — loads chunked regulation text into
the live Pinecone index. Last step of the ingestion pipeline:
fetch_regulations.py -> chunk_and_embed.py -> upsert_to_pinecone.py.

Uses Pinecone's hosted embedding model at upsert time (via upsert_records,
not raw vector upsert) — matching the embedding-model decision resolved
in chunk_and_embed.py: no separate embedding API call needed, Pinecone
embeds the text itself.

Chunk IDs are deterministic (see chunk_and_embed.py's
_make_deterministic_chunk_id) — same logical chunk always resolves to
the same ID, so re-running this script after a corpus refresh overwrites
existing vectors in place rather than duplicating them.
"""
import logging

from pinecone import Pinecone

from ingestion.chunk_and_embed import chunk_all_regulations, chunk_all_pdf_regulations
from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 96  # Pinecone's upsert_records has a practical batch limit;
                   # chunking the upsert avoids a single oversized request


def upsert_all_chunks():
    client = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = client.Index(settings.PINECONE_INDEX_NAME)

    xml_chunks = chunk_all_regulations()
    pdf_chunks = chunk_all_pdf_regulations()  # covers ICH (PDF) + EU CTR (HTML) — see chunk_and_embed.py
    all_chunks = xml_chunks + pdf_chunks

    if not all_chunks:
        logger.warning("No chunks to upsert — run fetch_regulations.py first")
        return

    logger.info(f"Upserting {len(all_chunks)} chunk(s) to Pinecone index '{settings.PINECONE_INDEX_NAME}'")

    records = [
        {
            "_id": chunk.chunk_id,
            "text": chunk.text,               # Pinecone embeds this field automatically
            "regulation_source": chunk.regulation_source,
            "jurisdiction": chunk.jurisdiction,
            "section_ref": chunk.section_ref or "",
        }
        for chunk in all_chunks
    ]

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        index.upsert_records(namespace="", records=batch)
        logger.info(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} records)")

    by_jurisdiction = {}
    for chunk in all_chunks:
        by_jurisdiction[chunk.jurisdiction] = by_jurisdiction.get(chunk.jurisdiction, 0) + 1

    logger.info(f"Upsert complete. Breakdown by jurisdiction: {by_jurisdiction}")
    return by_jurisdiction


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = upsert_all_chunks()
    print(f"\nUpsert complete: {result}")