"""
parse_pdf — Node 1 of the Sentinel-GCP pipeline.

Deterministic (no LLM call). Wraps Docling to turn a raw PDF into a
DocumentStructure — sections with page/section provenance, table regions
with parsed rows + a confidence score, and figure regions for anything
that isn't text (charts/graphs — see sentinel_gcp/vision/figure_description.py
for how those get described later).

Falls back to PyMuPDF if Docling fails outright on a document, and falls
back to Tesseract OCR per-page if a page's native text extraction comes
back empty (i.e. it's a scanned image, not real text) — per
ARCHITECTURE.md's parsing robustness discussion.
"""
import logging
from pathlib import Path

from docling.document_converter import DocumentConverter
import fitz  # PyMuPDF, fallback parser
import pytesseract
from PIL import Image

from sentinel_gcp.schema.document_structure import (
    DocumentStructure,
    Section,
    TableRegion,
    FigureRegion,
    ParsingCoverage,
)
from sentinel_gcp.graph.state import GraphState

logger = logging.getLogger(__name__)

OCR_TEXT_LENGTH_THRESHOLD = 20  # below this many chars, treat the page as image-only


def parse_pdf(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['raw_pdf_path'], writes
    state['document_structure']."""
    pdf_path = Path(state["raw_pdf_path"])
    logger.info(f"parse_pdf: starting on {pdf_path.name}")

    try:
        structure = _parse_with_docling(pdf_path)
    except Exception as e:
        logger.warning(f"Docling failed on {pdf_path.name} ({e}); falling back to PyMuPDF")
        structure = _parse_with_pymupdf_fallback(pdf_path)

    state["document_structure"] = structure
    state["status"] = "extracting"
    logger.info(
        f"parse_pdf: done — {len(structure.sections)} sections, "
        f"{len(structure.tables)} tables, {len(structure.figures)} figures, "
        f"{len(structure.parsing_coverage.ocr_fallback_pages)} OCR-fallback pages"
    )
    return state


def _parse_with_docling(pdf_path: Path) -> DocumentStructure:
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    sections: list[Section] = []
    tables: list[TableRegion] = []
    figures: list[FigureRegion] = []
    raw_text_by_page: dict[int, str] = {}
    ocr_fallback_pages: list[int] = []
    low_confidence_table_pages: list[int] = []

    # Walk Docling's layout-classified elements
    for item, level in doc.iterate_items():
        item_type = type(item).__name__

        if item_type == "SectionHeaderItem" or item_type == "TextItem":
            page_no = _get_page_number(item)
            text = getattr(item, "text", "") or ""
            raw_text_by_page[page_no] = raw_text_by_page.get(page_no, "") + "\n" + text

            if item_type == "SectionHeaderItem":
                sections.append(
                    Section(
                        heading=text,
                        section_id=_extract_section_id(text),
                        page_start=page_no,
                        text=text,
                    )
                )

        elif item_type == "TableItem":
            page_no = _get_page_number(item)
            parsed_rows, confidence = _table_to_rows(item)
            if confidence < 0.7:
                low_confidence_table_pages.append(page_no)
            tables.append(
                TableRegion(
                    name=_nearby_heading(item, sections),
                    page_start=page_no,
                    parsed_rows=parsed_rows,
                    confidence=confidence,
                    raw_bbox=_get_bbox(item),
                )
            )

        elif item_type == "PictureItem":
            page_no = _get_page_number(item)
            figures.append(
                FigureRegion(
                    page=page_no,
                    nearby_caption_text=_nearby_caption(item),
                    raw_bbox=_get_bbox(item),
                )
            )

    total_pages = doc.num_pages if hasattr(doc, "num_pages") else len(raw_text_by_page)

    # OCR fallback pass — any page with near-empty extracted text is likely scanned
    for page_no in range(1, total_pages + 1):
        text = raw_text_by_page.get(page_no, "")
        if len(text.strip()) < OCR_TEXT_LENGTH_THRESHOLD:
            ocr_text = _ocr_page(pdf_path, page_no)
            if ocr_text:
                raw_text_by_page[page_no] = ocr_text
                ocr_fallback_pages.append(page_no)

    coverage = ParsingCoverage(
        total_pages=total_pages,
        ocr_fallback_pages=ocr_fallback_pages,
        figures_detected=len(figures),
        tables_flagged_low_confidence=low_confidence_table_pages,
    )

    return DocumentStructure(
        sections=sections,
        tables=tables,
        figures=figures,
        raw_text_by_page=raw_text_by_page,
        parsing_coverage=coverage,
    )


def _parse_with_pymupdf_fallback(pdf_path: Path) -> DocumentStructure:
    """No layout/table/figure detection here — just gets raw text per page
    so the pipeline can still proceed (with a validate_schema likely to
    catch downstream issues) rather than crashing outright."""
    doc = fitz.open(str(pdf_path))
    raw_text_by_page = {i + 1: page.get_text() for i, page in enumerate(doc)}
    total_pages = len(doc)
    doc.close()

    return DocumentStructure(
        sections=[],
        tables=[],
        figures=[],
        raw_text_by_page=raw_text_by_page,
        parsing_coverage=ParsingCoverage(
            total_pages=total_pages,
            ocr_fallback_pages=[],
            figures_detected=0,
            tables_flagged_low_confidence=[],
        ),
    )


def _ocr_page(pdf_path: Path, page_no: int) -> str:
    doc = fitz.open(str(pdf_path))
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return pytesseract.image_to_string(img)


def _extract_section_id(heading_text: str) -> str | None:
    """Pulls a section number like '9.6' off the front of a heading string,
    e.g. '9.6 SAE Reporting' -> '9.6'."""
    import re
    match = re.match(r"^(\d+(\.\d+)*)\s", heading_text)
    return match.group(1) if match else None


def _get_page_number(item) -> int:
    prov = getattr(item, "prov", None)
    if prov and len(prov) > 0:
        return prov[0].page_no
    return 1


def _get_bbox(item) -> list[float] | None:
    prov = getattr(item, "prov", None)
    if prov and len(prov) > 0 and hasattr(prov[0], "bbox"):
        b = prov[0].bbox
        return [b.l, b.t, b.r, b.b]
    return None


def _table_to_rows(table_item) -> tuple[list[dict], float]:
    """Converts Docling's table structure into a list of row dicts,
    and derives a confidence score from Docling's own structure-recognition
    output where available."""
    try:
        df = table_item.export_to_dataframe()
        rows = df.to_dict(orient="records")
        confidence = getattr(table_item, "confidence", 0.85)
        return rows, confidence
    except Exception:
        return [], 0.3  # couldn't parse structure at all — flag it low


def _nearby_heading(table_item, sections: list[Section]) -> str | None:
    page = _get_page_number(table_item)
    candidates = [s for s in sections if s.page_start <= page]
    return candidates[-1].heading if candidates else None


def _nearby_caption(picture_item) -> str | None:
    return getattr(picture_item, "caption", None)