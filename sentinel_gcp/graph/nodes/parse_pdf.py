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

NOTE: Docling's exact class names/import paths can shift between versions.
The isinstance() checks below assume docling_core's document item types —
verify these imports resolve correctly against your installed Docling
version on first real run, and adjust if the API has moved.

VISIBILITY NOTE: Docling's converter.convert() is ONE blocking call with
NO built-in per-page progress callback in its standard API — you cannot
hook into "now processing page 47 of 137" through normal usage. What
this file DOES provide instead, honestly: (1) the total page count
logged BEFORE Docling starts (via a cheap PyMuPDF page-count check), so
you know the scale of the task, and (2) a heartbeat log every 15 seconds
WHILE convert() is blocking, so "still working" is always visible
instead of an unbroken silent gap that's indistinguishable from a hang.

PERFORMANCE FIX (found via real testing — a 74-page document took
~1483s): Docling's default pipeline runs its OWN built-in OCR (typically
EasyOCR, a heavy neural model) on top of layout detection and table
recognition. This is REDUNDANT with this project's dedicated
_run_ocr_fallback() below, which already targets ONLY genuinely scanned
pages (native-text extraction came back empty) via lightweight
Tesseract. Disabling Docling's built-in OCR (pipeline_options.do_ocr =
False) removes this duplicate work without reducing coverage — the
existing fallback still runs unchanged and catches the same pages.
Table structure recognition is deliberately left untouched at full
accuracy — that's where compliance-relevant data actually lives, and
is not traded away for speed. This fix applies equally in production
(every real user's single upload benefits) not just dev iteration —
unlike a file-hash cache, which was considered and rejected since a
real deployment never re-parses the same document twice.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Fresh-machine fix: avoids requiring MSVC's cl.exe
import logging
import re
import threading
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import (
    SectionHeaderItem,
    TableItem,
    PictureItem,
    TextItem,
)
import fitz  # PyMuPDF, fallback parser + cheap page-count check
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

# Fewer than ~20 extracted characters, OR fewer than ~4 words, usually
# indicates an image-only/scanned page rather than real extracted text.
OCR_TEXT_LENGTH_THRESHOLD = 20
OCR_TEXT_WORD_THRESHOLD = 4

# Below this, a parsed table is flagged for downstream/human review
# rather than trusted outright.
LOW_TABLE_CONFIDENCE = 0.70

# How often the heartbeat logs while Docling is blocking on convert().
HEARTBEAT_INTERVAL_SECONDS = 15


def parse_pdf(state: GraphState) -> GraphState:
    """LangGraph node entrypoint. Reads state['raw_pdf_path'], writes
    state['document_structure']."""
    pdf_path = Path(state["raw_pdf_path"])
    logger.info(f"parse_pdf: starting on {pdf_path.name}")
    start_time = time.perf_counter()

    try:
        structure = _parse_with_docling(pdf_path)
    except Exception:
        logger.warning(
            f"Docling failed on {pdf_path.name}; falling back to PyMuPDF",
            exc_info=True,
        )
        structure = _parse_with_pymupdf_fallback(pdf_path)

    structure.parsing_coverage.parsing_duration_seconds = round(
        time.perf_counter() - start_time, 2
    )

    state["document_structure"] = structure
    state["status"] = "extracting"
    logger.info(
        f"parse_pdf: done in {structure.parsing_coverage.parsing_duration_seconds}s — "
        f"{len(structure.sections)} sections, {len(structure.tables)} tables, "
        f"{len(structure.figures)} figures, "
        f"{len(structure.parsing_coverage.ocr_fallback_pages)} OCR-fallback pages"
    )
    return state


def _get_page_count_cheaply(pdf_path: Path) -> int:
    """Quick page count via PyMuPDF BEFORE handing off to Docling — this
    is fast (milliseconds) regardless of document complexity, unlike
    waiting for Docling's full conversion just to learn how big the
    document is."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def _parse_with_docling(pdf_path: Path) -> DocumentStructure:
    total_page_count = _get_page_count_cheaply(pdf_path)
    logger.info(
        f"parse_pdf: {pdf_path.name} has {total_page_count} page(s) — "
        f"starting Docling conversion (this may take several minutes on "
        f"large/table-heavy documents; heartbeat logs every "
        f"{HEARTBEAT_INTERVAL_SECONDS}s below)"
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # redundant with _run_ocr_fallback() below —
                                       # see module docstring PERFORMANCE FIX note

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    # Docling's convert() is one blocking call with NO per-page progress
    # callback available through the standard API. This background
    # thread logs elapsed time periodically WHILE convert() blocks, so
    # "still working" is always visible rather than an ambiguous silent gap.
    stop_heartbeat = threading.Event()

    def _heartbeat():
        elapsed = 0
        while not stop_heartbeat.wait(timeout=HEARTBEAT_INTERVAL_SECONDS):
            elapsed += HEARTBEAT_INTERVAL_SECONDS
            logger.info(
                f"parse_pdf: Docling still processing {pdf_path.name} "
                f"({total_page_count} pages)... ({elapsed}s elapsed)"
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        result = converter.convert(str(pdf_path))
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)

    doc = result.document

    sections: list[Section] = []
    tables: list[TableRegion] = []
    figures: list[FigureRegion] = []
    raw_text_by_page: dict[int, str] = {}
    ocr_fallback_pages: list[int] = []
    low_confidence_table_pages: list[int] = []

    for item, level in doc.iterate_items():
        if isinstance(item, (SectionHeaderItem, TextItem)):
            page_no = _get_page_number(item)
            text = getattr(item, "text", "") or ""
            raw_text_by_page[page_no] = raw_text_by_page.get(page_no, "") + "\n" + text

            if isinstance(item, SectionHeaderItem):
                sections.append(
                    Section(
                        heading=text,
                        section_id=_extract_section_id(text),
                        page_start=page_no,
                        text=text,
                    )
                )

        elif isinstance(item, TableItem):
            page_no = _get_page_number(item)
            parsed_rows, confidence = _table_to_rows(item)
            if confidence < LOW_TABLE_CONFIDENCE:
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

        elif isinstance(item, PictureItem):
            page_no = _get_page_number(item)
            figures.append(
                FigureRegion(
                    page=page_no,
                    nearby_caption_text=_nearby_caption(item),
                    raw_bbox=_get_bbox(item),
                )
            )

    if hasattr(doc, "num_pages"):
        num_pages_attr = doc.num_pages
        total_pages = num_pages_attr() if callable(num_pages_attr) else num_pages_attr
    else:
        total_pages = len(raw_text_by_page)

    ocr_fallback_pages = _run_ocr_fallback(pdf_path, raw_text_by_page, total_pages)

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
    """No native table/figure detection here, but we DO attempt lightweight
    heading detection on the plain text — a lower-confidence version of
    what Docling gives us, better than losing all structure entirely."""
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info(f"parse_pdf: PyMuPDF fallback — {pdf_path.name} has {total_pages} page(s)")

    raw_text_by_page = {}
    for i, page in enumerate(doc):
        page_no = i + 1
        raw_text_by_page[page_no] = page.get_text()
        if page_no % 25 == 0 or page_no == total_pages:
            logger.info(f"parse_pdf: PyMuPDF fallback — extracted page {page_no}/{total_pages}")
    doc.close()

    sections: list[Section] = []
    for page_no, text in raw_text_by_page.items():
        for line in text.splitlines():
            section_id = _extract_section_id(line.strip())
            if section_id:
                sections.append(
                    Section(
                        heading=line.strip(),
                        section_id=section_id,
                        page_start=page_no,
                        text=line.strip(),
                    )
                )

    return DocumentStructure(
        sections=sections,
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


def _run_ocr_fallback(
    pdf_path: Path, raw_text_by_page: dict[int, str], total_pages: int
) -> list[int]:
    """Identifies which pages need OCR, then opens the PDF ONCE and
    processes all of them in that single session — not one open/close
    per page. This is the SOLE OCR mechanism now that Docling's built-in
    OCR is disabled — it must continue to correctly catch every
    genuinely scanned page (verify OCR-fallback page counts stay
    consistent after this change, e.g. still catching pages 16/43 on
    OEV-125)."""
    pages_needing_ocr = [
        page_no
        for page_no in range(1, total_pages + 1)
        if _looks_like_scanned_page(raw_text_by_page.get(page_no, ""))
    ]
    if not pages_needing_ocr:
        return []

    logger.info(f"OCR fallback needed on {len(pages_needing_ocr)} page(s): {pages_needing_ocr}")
    doc = fitz.open(str(pdf_path))
    ocr_fallback_pages: list[int] = []
    try:
        for idx, page_no in enumerate(pages_needing_ocr, start=1):
            logger.info(f"parse_pdf: OCR processing page {page_no} ({idx}/{len(pages_needing_ocr)})")
            page = doc[page_no - 1]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                raw_text_by_page[page_no] = ocr_text
                ocr_fallback_pages.append(page_no)
    finally:
        doc.close()

    return ocr_fallback_pages


def _looks_like_scanned_page(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < OCR_TEXT_LENGTH_THRESHOLD or len(stripped.split()) < OCR_TEXT_WORD_THRESHOLD


def _extract_section_id(heading_text: str) -> str | None:
    """Pulls a section number like '9.6' off a heading string, optionally
    preceded by a word like 'Section' or 'Appendix'. Tightened after real
    testing showed the looser version false-matching addresses/dates
    (e.g. '10628 Science Center Dr', '22 August 2022')."""
    heading_text = heading_text.strip()
    if len(heading_text) > 100:
        return None
    if re.match(r"^\d{1,5}\s+[A-Z]", heading_text):
        return None
    if re.match(r"^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)", heading_text):
        return None
    match = re.match(r"^(?:Section|Appendix)?\s*(\d+(\.\d+)+|[A-Z]\.\d+)\s+\S", heading_text)
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
    try:
        df = table_item.export_to_dataframe()
        df.columns = df.columns.astype(str)  # tables with no header row get an
                                                # integer RangeIndex — TableRegion
                                                # requires Dict[str, str] keys
        rows = df.to_dict(orient="records")
        confidence = getattr(table_item, "confidence", 0.85)
        return rows, confidence
    except Exception:
        logger.warning("Table structure parsing failed; flagging low confidence", exc_info=True)
        return [], 0.3


def _nearby_heading(table_item, sections: list[Section]) -> str | None:
    page = _get_page_number(table_item)
    candidates = [s for s in sections if s.page_start <= page]
    return candidates[-1].heading if candidates else None


def _nearby_caption(picture_item) -> str | None:
    return getattr(picture_item, "caption", None)


# TODO (future improvement, deferred): parallelize OCR across pages for
# large scanned documents (400+ pages). Needs thread-safe handling of the
# shared fitz.Document object. Not worth the added complexity until the
# sequential path is proven correct end-to-end.

# TODO (future improvement, deferred): hard timeout around converter.convert()
# with automatic fallback to PyMuPDF, so a genuinely pathological document
# (e.g. NEOD001's Schedule of Assessments table, which ran far slower than
# every other document tested) can't hang indefinitely. Not yet implemented —
# discussed but deferred in favor of trying this OCR fix first.