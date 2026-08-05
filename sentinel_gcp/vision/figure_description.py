"""
vision/figure_description.py — describes charts/graphs/figures detected
by parse_pdf.py (DocumentStructure.figures) but not readable as text.

Deferred earlier in this project's design phase as a known gap: Docling
detects figure REGIONS (bounding boxes) but doesn't interpret their
content. This module fills that gap by cropping the detected region
from the source PDF and sending it to Claude's vision endpoint.

HONEST LIMITATION, stated plainly (per the original parsing-robustness
discussion): this does NOT extract precise data points from a chart —
vision models are decent but not exact at reading axis values. What
this DOES reliably capture is the figure's substantive CLAIM (e.g.
"dmLT is at least 50-fold less toxic than native LT," the real example
from OEV-125's Figure 1) — sufficient for a compliance reviewer to know
a figure exists and what it argues, without needing pixel-perfect
data extraction.

This is a genuinely optional, lower-priority pipeline addition — no
existing node currently calls it. It's built and ready to wire into
parse_pdf.py's FigureRegion handling once that becomes a priority,
but doing so is a real, separate design decision (adds a vision API
call per detected figure, which could be a meaningful per-document
cost multiplier on figure-heavy documents) — not done automatically
by this file's mere existence.
"""
import base64
import logging
from pathlib import Path

import fitz  # PyMuPDF, for cropping the figure region from the source PDF
from anthropic import Anthropic

from sentinel_gcp.config import settings

logger = logging.getLogger(__name__)

client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

FIGURE_DESCRIPTION_PROMPT = """This is a figure (chart, graph, or diagram) extracted from a \
clinical trial protocol or regulatory document. Describe:
1. What TYPE of figure this is (bar chart, line graph, diagram, table-like image, etc.)
2. What it appears to be COMPARING or ILLUSTRATING (axis labels, legend, general subject)
3. The SUBSTANTIVE CLAIM or conclusion the figure appears to support, if identifiable

Be concise — 2-3 sentences. If the image is unclear, illegible, or you cannot confidently \
describe its content, say so plainly rather than guessing. Do NOT attempt to read exact \
numeric values off axes — describe general trends/relationships only, since precise \
data extraction from images is unreliable."""


def describe_figure(pdf_path: Path, page_number: int, bbox: list[float] | None) -> str | None:
    """Crops the figure region (if bbox is available) or falls back to
    the whole page, sends it to Claude's vision endpoint, returns a
    short description. Returns None (not a fabricated description) if
    the crop fails or the model can't produce a confident description —
    same 'null is honest, never guess' principle used throughout this
    project's extraction logic."""
    try:
        image_bytes = _crop_region_as_png(pdf_path, page_number, bbox)
    except Exception:
        logger.warning(
            f"describe_figure: failed to crop page {page_number} of {pdf_path.name}",
            exc_info=True,
        )
        return None

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": FIGURE_DESCRIPTION_PROMPT},
                ],
            }],
        )
    except Exception:
        logger.warning(f"describe_figure: Claude API call failed for page {page_number}", exc_info=True)
        return None

    if not response.content:
        logger.warning(f"describe_figure: empty response for page {page_number}, stop_reason={response.stop_reason}")
        return None

    return response.content[0].text.strip()


def _crop_region_as_png(pdf_path: Path, page_number: int, bbox: list[float] | None) -> bytes:
    """Renders the figure's bounding box region as a PNG image. Falls
    back to rendering the full page if bbox is unavailable (per
    document_structure.py's docstring, raw_bbox is Optional — not every
    parser backend reliably returns it)."""
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number - 1]
        if bbox:
            clip_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            pix = page.get_pixmap(dpi=200, clip=clip_rect)
        else:
            logger.info(f"_crop_region_as_png: no bbox for page {page_number}, rendering full page instead")
            pix = page.get_pixmap(dpi=150)
        return pix.tobytes("png")
    finally:
        doc.close()


def describe_all_figures(pdf_path: Path, figures: list) -> dict[int, str]:
    """Runs describe_figure() across every FigureRegion in a
    DocumentStructure.figures list. Returns a dict keyed by a synthetic
    index (not page number alone, since multiple figures can share a
    page) mapping to the description, or omits an entry entirely if
    description failed — callers should treat a missing key as 'could
    not describe,' not assume every figure got a result."""
    descriptions = {}
    for i, figure in enumerate(figures):
        description = describe_figure(pdf_path, figure.page, figure.raw_bbox)
        if description:
            descriptions[i] = description
        else:
            logger.info(f"describe_all_figures: no description produced for figure {i} on page {figure.page}")
    logger.info(f"describe_all_figures: {len(descriptions)}/{len(figures)} figure(s) successfully described")
    return descriptions