"""
ingestion/chunk_and_embed.py — section-aware chunking + embedding prep.

Two source paths, one shared chunking core:
  - chunk_all_regulations() — processes eCFR XML files (FDA sources)
  - chunk_pdf_regulation() — processes a single PDF (ICH-GCP, and any
    future EU-specific source), reusing parse_pdf.py's Docling-based
    parser (genuine code reuse, not a second parsing implementation)

Both paths converge on _group_paragraphs_into_chunks() — the actual
chunking decision (merge short paragraphs, split oversized ones with
overlap) is identical regardless of whether the source was XML or PDF.

Embedding itself happens at Pinecone's end (see upsert_to_pinecone.py)
using Pinecone's hosted embedding model — resolves the open embedding-
model decision flagged back when faiss_store.py was written.

Chunking strategy:
  - PRIMARY split boundary: natural subsection structure (eCFR's <P>
    tags, or Docling's detected Section objects for PDFs) — NOT raw
    token count
  - Target size: ~300-500 tokens per chunk. A single paragraph under
    that cap becomes one chunk; several short consecutive paragraphs
    may be merged up to the target; a single paragraph OVER the cap
    gets split further, with overlap applied only in that fallback case
  - Overlap: ~50-75 tokens (~15-20% of target), applied only when a
    chunk had to be split mid-section — never introduced between
    already-distinct subsections, since that would blur clause boundaries
"""
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/regulations")
CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
# Rough heuristic: ~4 characters per token for English regulatory text —
# avoids pulling in a real tokenizer dependency just for chunk sizing.
CHARS_PER_TOKEN_ESTIMATE = 4
CHUNK_TARGET_CHARS = CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN_ESTIMATE
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN_ESTIMATE


@dataclass
class RegulationChunk:
    chunk_id: str
    text: str
    regulation_source: str      # e.g. "21 CFR 312.32" or "ICH E6(R3)"
    jurisdiction: str            # "FDA" | "ICH" | "EMA"
    section_ref: str | None = None   # e.g. "(c)(1)" — provenance within the source section


# ─────────────────────────────────────────────────────────────────
# XML path (eCFR / FDA sources)
# ─────────────────────────────────────────────────────────────────

def chunk_all_regulations(input_dir: Path = INPUT_DIR) -> list[RegulationChunk]:
    """Processes every .xml file in input_dir, returns all chunks across
    all documents. upsert_to_pinecone.py consumes this list directly."""
    all_chunks: list[RegulationChunk] = []
    xml_files = sorted(input_dir.glob("*.xml"))

    for xml_path in xml_files:
        logger.info(f"Chunking {xml_path.name}")
        chunks = _chunk_one_xml_document(xml_path)
        all_chunks.extend(chunks)
        logger.info(f"  -> {len(chunks)} chunk(s) from {xml_path.name}")

    logger.info(f"chunk_all_regulations: {len(all_chunks)} total chunk(s) across {len(xml_files)} XML document(s)")
    return all_chunks


def _chunk_one_xml_document(xml_path: Path) -> list[RegulationChunk]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    citation = _extract_citation(root, xml_path)
    jurisdiction = "FDA"  # all current XML sources are FDA/eCFR

    paragraphs = _extract_paragraph_texts_from_xml(root)
    return _group_paragraphs_into_chunks(paragraphs, citation, jurisdiction)


def _extract_citation(root, xml_path: Path) -> str:
    """Pulls the citation (e.g. '21 CFR 312.32') from the hierarchy_metadata
    attribute eCFR embeds on the root DIV8 element."""
    hierarchy_meta = root.get("hierarchy_metadata", "")
    match = re.search(r'"citation"\s*:\s*"([^"]+)"', hierarchy_meta)
    return match.group(1) if match else xml_path.stem.replace("_", " ").upper()


def _extract_paragraph_texts_from_xml(root) -> list[tuple[str, str]]:
    """Returns (section_ref, text) pairs, one per <P> tag — this IS the
    section-aware split, since eCFR already delivers content pre-divided
    at the subsection level via <P> tags."""
    results = []
    para_counter = 0
    for p in root.iter("P"):
        text = "".join(p.itertext()).strip()
        if not text:
            continue
        para_counter += 1
        section_ref = _guess_section_ref(text, fallback=f"para-{para_counter}")
        results.append((section_ref, text))
    return results


# ─────────────────────────────────────────────────────────────────
# PDF path (ICH-GCP, and any future EU-specific source)
# ─────────────────────────────────────────────────────────────────

def chunk_pdf_regulation(pdf_path: Path, citation: str, jurisdiction: str) -> list[RegulationChunk]:
    """Reuses parse_pdf.py's Docling-based parsing (already built and
    tested for trial protocols) against a regulation PDF instead — same
    tool, different input. DocumentStructure.sections gives us the same
    kind of section-aware boundary the XML path gets from <P> tags."""
    from sentinel_gcp.graph.nodes.parse_pdf import _parse_with_docling

    logger.info(f"Chunking PDF {pdf_path.name} (jurisdiction={jurisdiction})")
    structure = _parse_with_docling(pdf_path)

    paragraphs = [
        (s.section_id or f"para-{i}", s.text)
        for i, s in enumerate(structure.sections)
        if s.text.strip()
    ]
    chunks = _group_paragraphs_into_chunks(paragraphs, citation, jurisdiction)
    logger.info(f"  -> {len(chunks)} chunk(s) from {pdf_path.name}")
    return chunks


def chunk_all_pdf_regulations(input_dir: Path = INPUT_DIR) -> list[RegulationChunk]:
    """Convenience wrapper matching chunk_all_regulations()'s shape, for
    the known PDF sources fetched by fetch_regulations.py. Currently
    just ICH-GCP; EU-specific sources would be added here once fetched."""
    all_chunks: list[RegulationChunk] = []

    ich_path = input_dir / "ich_e6_r3_gcp.pdf"
    if ich_path.exists():
        all_chunks.extend(
            chunk_pdf_regulation(ich_path, citation="ICH E6(R3) GCP", jurisdiction="ICH")
        )
    else:
        logger.warning(f"{ich_path} not found — run fetch_regulations.py's fetch_all_ich_sources() first")

    return all_chunks


# ─────────────────────────────────────────────────────────────────
# Shared chunking core — used by both XML and PDF paths
# ─────────────────────────────────────────────────────────────────

def _guess_section_ref(text: str, fallback: str) -> str:
    """Best-effort extraction of a subsection label like '(c)(1)' from
    the start of a paragraph's text — falls back to a generic paragraph
    number if the text doesn't start with a recognizable label."""
    match = re.match(r"^\(([a-z0-9]+)\)(\(([a-z0-9]+)\))?", text)
    return match.group(0) if match else fallback


def _group_paragraphs_into_chunks(
    paragraphs: list[tuple[str, str]],
    citation: str,
    jurisdiction: str,
) -> list[RegulationChunk]:
    """Merges consecutive short paragraphs up to CHUNK_TARGET_CHARS;
    splits any single paragraph that alone exceeds the target. Overlap
    is only applied in that split case, never between distinct subsections."""
    chunks: list[RegulationChunk] = []
    buffer_text = ""
    buffer_refs: list[str] = []

    def flush_buffer():
        if buffer_text.strip():
            chunks.append(
                RegulationChunk(
                    chunk_id=f"chunk-{uuid.uuid4().hex[:10]}",
                    text=buffer_text.strip(),
                    regulation_source=citation,
                    jurisdiction=jurisdiction,
                    section_ref=", ".join(buffer_refs) if buffer_refs else None,
                )
            )

    for section_ref, text in paragraphs:
        if len(text) > CHUNK_TARGET_CHARS:
            flush_buffer()
            buffer_text, buffer_refs = "", []
            chunks.extend(_split_long_paragraph(text, section_ref, citation, jurisdiction))
            continue

        if len(buffer_text) + len(text) > CHUNK_TARGET_CHARS:
            flush_buffer()
            buffer_text, buffer_refs = "", []

        buffer_text += ("\n\n" if buffer_text else "") + text
        buffer_refs.append(section_ref)

    flush_buffer()
    return chunks


def _split_long_paragraph(text: str, section_ref: str, citation: str, jurisdiction: str) -> list[RegulationChunk]:
    """Fallback splitter for a single paragraph too long to be one chunk
    on its own. Overlap IS applied here, since we're cutting mid-content,
    not at a natural subsection boundary."""
    pieces = []
    start = 0
    while start < len(text):
        end = start + CHUNK_TARGET_CHARS
        piece = text[start:end]
        pieces.append(
            RegulationChunk(
                chunk_id=f"chunk-{uuid.uuid4().hex[:10]}",
                text=piece.strip(),
                regulation_source=citation,
                jurisdiction=jurisdiction,
                section_ref=f"{section_ref} (part {len(pieces) + 1})",
            )
        )
        start = end - CHUNK_OVERLAP_CHARS
    return pieces


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    xml_chunks = chunk_all_regulations()
    pdf_chunks = chunk_all_pdf_regulations()
    all_chunks = xml_chunks + pdf_chunks

    print(f"\nProduced {len(xml_chunks)} chunk(s) from XML (FDA) + {len(pdf_chunks)} chunk(s) from PDF (ICH) = {len(all_chunks)} total.")
    if all_chunks:
        print(f"\nSample chunk:\n  source: {all_chunks[0].regulation_source}")
        print(f"  jurisdiction: {all_chunks[0].jurisdiction}")
        print(f"  section_ref: {all_chunks[0].section_ref}")
        print(f"  text (first 200 chars): {all_chunks[0].text[:200]}")