"""
DocumentStructure — the output of parse_pdf (node 1).

Preserves the document's actual layout — headings, page numbers, section IDs,
table regions — so downstream nodes can attach real provenance (page, section)
to extracted fields instead of the LLM guessing them from a flat text blob.

Table parsing confidence matters here: the Schedule of Assessments-style
tables (merged headers, footnote markers) are the known stress point for
naive PDF extraction — a low confidence score on a TableRegion signals that
a downstream node (or a human) should double-check that table's content.
"""
from pydantic import BaseModel
from typing import Optional, List, Dict


class Section(BaseModel):
    heading: str
    section_id: Optional[str] = None      # e.g. "9.6" — used for provenance & cross-referencing
    page_start: int
    page_end: Optional[int] = None
    text: str                              # the section's raw extracted text


class TableRegion(BaseModel):
    name: Optional[str] = None             # e.g. "Schedule of Events"
    page_start: int
    page_end: Optional[int] = None
    parsed_rows: List[Dict[str, str]] = []  # row data as extracted
    confidence: float = 1.0                 # Docling's structure-recognition confidence
    raw_bbox: Optional[List[float]] = None  # bounding box, for re-cropping if needed later


class FigureRegion(BaseModel):
    """Non-text visual element (chart, graph, diagram) detected but not
    parsed as text — see sentinel_gcp/vision/figure_description.py for
    how these get a vision-model description attached later."""
    page: int
    nearby_caption_text: Optional[str] = None
    raw_bbox: Optional[List[float]] = None


class ParsingCoverage(BaseModel):
    """Honesty layer — what fraction of the document was actually captured,
    so gaps are visible rather than silent."""
    total_pages: int
    ocr_fallback_pages: List[int] = []
    figures_detected: int = 0
    tables_flagged_low_confidence: List[int] = []  # page numbers


class DocumentStructure(BaseModel):
    sections: List[Section] = []
    tables: List[TableRegion] = []
    figures: List[FigureRegion] = []
    raw_text_by_page: Dict[int, str] = {}
    parsing_coverage: ParsingCoverage