"""
ingestion/chunk_and_embed.py — section-aware chunking + embedding prep.

Three source paths, one shared chunking core:
  - chunk_all_regulations() — processes eCFR XML files (FDA sources)
  - chunk_pdf_regulation() — processes a regulation PDF (ICH-GCP),
    reusing parse_pdf.py's Docling-based parser (genuine code reuse)
  - chunk_html_regulation() — processes a regulation HTML page (EU CTR),
    using Python's stdlib html.parser — no new dependency needed

All three converge on _group_paragraphs_into_chunks() — the actual
chunking decision (merge short paragraphs, split oversized ones with
overlap) is identical regardless of source format.

Embedding itself happens at Pinecone's end (see upsert_to_pinecone.py)
using Pinecone's hosted embedding model.

Chunking strategy:
  - PRIMARY split boundary: natural subsection structure (eCFR's <P>
    tags, Docling's detected Sections for PDFs, or paragraph-level tags
    for HTML) — NOT raw token count
  - Target size: ~300-500 tokens per chunk, merging short consecutive
    paragraphs, splitting rare oversized ones with overlap applied only
    in that fallback case
  - Overlap: ~50-75 tokens (~15-20% of target)
"""
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/regulations")
CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
CHARS_PER_TOKEN_ESTIMATE = 4
CHUNK_TARGET_CHARS = CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN_ESTIMATE
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN_ESTIMATE

# Maps a fetched PDF filename to (citation, jurisdiction). New PDF
# sources just need an entry added here.
_PDF_SOURCE_METADATA: dict[str, tuple[str, str]] = {
    "ich_e6_r3_gcp.pdf": ("ICH E6(R3) GCP", "ICH"),
}

# Same idea for HTML sources.
_HTML_SOURCE_METADATA: dict[str, tuple[str, str]] = {
    "eu_ctr_536_2014.html": ("Regulation (EU) No 536/2014", "EMA"),
}


@dataclass
class RegulationChunk:
    chunk_id: str
    text: str
    regulation_source: str      # e.g. "21 CFR 312.32" or "ICH E6(R3) GCP"
    jurisdiction: str            # "FDA" | "ICH" | "EMA"
    section_ref: str | None = None


# ─────────────────────────────────────────────────────────────────
# XML path (eCFR / FDA sources)
# ─────────────────────────────────────────────────────────────────

def chunk_all_regulations(input_dir: Path = INPUT_DIR) -> list[RegulationChunk]:
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
    jurisdiction = "FDA"
    paragraphs = _extract_paragraph_texts_from_xml(root)
    return _group_paragraphs_into_chunks(paragraphs, citation, jurisdiction)


def _extract_citation(root, xml_path: Path) -> str:
    hierarchy_meta = root.get("hierarchy_metadata", "")
    match = re.search(r'"citation"\s*:\s*"([^"]+)"', hierarchy_meta)
    return match.group(1) if match else xml_path.stem.replace("_", " ").upper()


def _extract_paragraph_texts_from_xml(root) -> list[tuple[str, str]]:
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
# PDF path (ICH-GCP)
# ─────────────────────────────────────────────────────────────────

def chunk_pdf_regulation(pdf_path: Path, citation: str, jurisdiction: str) -> list[RegulationChunk]:
    """Reuses parse_pdf.py's Docling-based parsing — same tool already
    built and tested for trial protocols, different input."""
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


# ─────────────────────────────────────────────────────────────────
# HTML path (EU CTR) — stdlib html.parser, no new dependency
# ─────────────────────────────────────────────────────────────────

class _ParagraphExtractor(HTMLParser):
    """Minimal HTML->paragraph extractor. Groups text by <p>/<div>/<li>
    boundaries — good enough for pulling plain regulatory text out of
    EUR-Lex's page structure without pulling in BeautifulSoup."""
    def __init__(self):
        super().__init__()
        self.paragraphs: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "div", "li") and self._current:
            self._flush()

    def handle_data(self, data):
        text = data.strip()
        if text:
            self._current.append(text)

    def handle_endtag(self, tag):
        if tag in ("p", "div", "li"):
            self._flush()

    def _flush(self):
        if self._current:
            self.paragraphs.append(" ".join(self._current))
            self._current = []


def _extract_paragraph_texts_from_html(html_text: str) -> list[tuple[str, str]]:
    parser = _ParagraphExtractor()
    parser.feed(html_text)
    results = []
    for i, text in enumerate(parser.paragraphs):
        if len(text) < 20:  # skip nav/boilerplate fragments
            continue
        section_ref = _guess_section_ref(text, fallback=f"para-{i}")
        results.append((section_ref, text))
    return results


def chunk_html_regulation(html_path: Path, citation: str, jurisdiction: str) -> list[RegulationChunk]:
    html_text = html_path.read_text(encoding="utf-8")
    paragraphs = _extract_paragraph_texts_from_html(html_text)
    logger.info(f"Chunking HTML {html_path.name} — {len(paragraphs)} paragraph(s) extracted")
    return _group_paragraphs_into_chunks(paragraphs, citation, jurisdiction)


# ─────────────────────────────────────────────────────────────────
# Combined PDF + HTML entrypoint
# ─────────────────────────────────────────────────────────────────

def chunk_all_pdf_regulations(input_dir: Path = INPUT_DIR) -> list[RegulationChunk]:
    """Despite the name (kept for compatibility with upsert_to_pinecone.py's
    existing call), this now covers BOTH PDF and HTML non-XML sources —
    ICH-GCP (PDF) and EU CTR (HTML). Any file present on disk but missing
    from either metadata map is flagged with a warning, not silently skipped."""
    all_chunks: list[RegulationChunk] = []

    for filename, (citation, jurisdiction) in _PDF_SOURCE_METADATA.items():
        pdf_path = input_dir / filename
        if pdf_path.exists():
            all_chunks.extend(chunk_pdf_regulation(pdf_path, citation, jurisdiction))
        else:
            logger.warning(f"{pdf_path} not found — run fetch_regulations.py first")

    for filename, (citation, jurisdiction) in _HTML_SOURCE_METADATA.items():
        html_path = input_dir / filename
        if html_path.exists():
            all_chunks.extend(chunk_html_regulation(html_path, citation, jurisdiction))
        else:
            logger.warning(f"{html_path} not found — run fetch_regulations.py first")

    known_filenames = set(_PDF_SOURCE_METADATA.keys()) | set(_HTML_SOURCE_METADATA.keys())
    for path in list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.html")):
        if path.name not in known_filenames:
            logger.warning(
                f"{path.name} exists on disk but has no metadata entry — "
                f"it will NOT be chunked or upserted until a mapping is added"
            )

    return all_chunks


# ─────────────────────────────────────────────────────────────────
# Shared chunking core
# ─────────────────────────────────────────────────────────────────

def _guess_section_ref(text: str, fallback: str) -> str:
    match = re.match(r"^\(([a-z0-9]+)\)(\(([a-z0-9]+)\))?", text)
    return match.group(0) if match else fallback


def _group_paragraphs_into_chunks(
    paragraphs: list[tuple[str, str]],
    citation: str,
    jurisdiction: str,
) -> list[RegulationChunk]:
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
    pdf_html_chunks = chunk_all_pdf_regulations()
    all_chunks = xml_chunks + pdf_html_chunks

    print(
        f"\nProduced {len(xml_chunks)} chunk(s) from XML (FDA) + "
        f"{len(pdf_html_chunks)} chunk(s) from PDF/HTML (ICH + EU) = {len(all_chunks)} total."
    )
    if all_chunks:
        by_jurisdiction = {}
        for c in all_chunks:
            by_jurisdiction[c.jurisdiction] = by_jurisdiction.get(c.jurisdiction, 0) + 1
        print(f"Breakdown by jurisdiction: {by_jurisdiction}")
        print(f"\nSample chunk:\n  source: {all_chunks[0].regulation_source}")
        print(f"  jurisdiction: {all_chunks[0].jurisdiction}")
        print(f"  section_ref: {all_chunks[0].section_ref}")
        print(f"  text (first 200 chars): {all_chunks[0].text[:200]}")