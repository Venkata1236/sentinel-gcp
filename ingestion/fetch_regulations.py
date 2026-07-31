"""
ingestion/fetch_regulations.py — pulls raw regulatory source text.

This is a ONE-TIME / PERIODIC script, not part of the runtime graph —
run manually (or on a schedule) to build/refresh the local regulatory
corpus before chunk_and_embed.py processes it and upsert_to_pinecone.py
loads it into the vector store. Deliberately kept separate from
sentinel_gcp/ (the runtime package) per the original file-structure
decision.

Sources fetched (matching what the project's Phase 1 study covered):
  - FDA 21 CFR 312 (via eCFR)
  - ICH-GCP E6(R3) — NOTE: no reliable single authoritative machine-
    readable source was settled on during design; flagged below
  - EMA guidance — NOTE: same gap, flagged below

Each fetched document is tagged with its jurisdiction at THIS stage,
not later — this is what lets chunk_and_embed.py propagate jurisdiction
metadata all the way through to Pinecone, which is what makes
retrieve.py's jurisdiction-filtered queries possible at all.
"""
import logging
import time
from pathlib import Path
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/regulations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_SECONDS = 1.0  # basic politeness delay between requests


@dataclass
class RegulationSource:
    name: str
    url: str
    jurisdiction: str  # "FDA" | "EMA" | "ICH" (ICH-GCP applies broadly, not jurisdiction-exclusive)
    output_filename: str


# eCFR sections directly relevant to protocol content and safety reporting —
# same sections identified during the project's original Phase 1 study
# (see ARCHITECTURE.md's referenced study phase / interview-prep material)
FDA_SOURCES = [
    RegulationSource(
        name="21 CFR 312.23 — IND content and format",
        url="https://www.ecfr.gov/api/versioner/v1/full/current/title-21.xml?part=312&section=312.23",
        jurisdiction="FDA",
        output_filename="21_cfr_312_23.xml",
    ),
    RegulationSource(
        name="21 CFR 312.32 — IND safety reporting",
        url="https://www.ecfr.gov/api/versioner/v1/full/current/title-21.xml?part=312&section=312.32",
        jurisdiction="FDA",
        output_filename="21_cfr_312_32.xml",
    ),
    RegulationSource(
        name="21 CFR 312.30 — Protocol amendments",
        url="https://www.ecfr.gov/api/versioner/v1/full/current/title-21.xml?part=312&section=312.30",
        jurisdiction="FDA",
        output_filename="21_cfr_312_30.xml",
    ),
    RegulationSource(
        name="21 CFR 312.33 — Annual reports",
        url="https://www.ecfr.gov/api/versioner/v1/full/current/title-21.xml?part=312&section=312.33",
        jurisdiction="FDA",
        output_filename="21_cfr_312_33.xml",
    ),
]

# KNOWN GAP, not resolved in this pass: ICH-GCP E6(R3) and EMA guidance
# don't have an equivalent free, stable, machine-readable API like eCFR.
# ICH-GCP is typically distributed as PDF (e.g. via ich.org); EMA guidance
# similarly. A real implementation of this would need PDF fetching +
# the SAME parsing pipeline already built for protocols (parse_pdf.py) —
# genuinely reusable, but not wired up here. Placeholder list below
# documents the INTENT; actual fetch logic for these two is deferred.
ICH_EMA_SOURCES_TODO = [
    "ICH-GCP E6(R3) — needs PDF source URL + parse_pdf.py reuse",
    "EMA Clinical Trials Regulation guidance — needs PDF source URL + parse_pdf.py reuse",
]


def fetch_all_fda_sources() -> list[Path]:
    """Fetches every FDA source in FDA_SOURCES, writes raw XML to
    data/regulations/. Returns the list of file paths written, which
    chunk_and_embed.py will process next."""
    written_paths = []

    for source in FDA_SOURCES:
        output_path = OUTPUT_DIR / source.output_filename
        logger.info(f"Fetching {source.name} from {source.url}")

        try:
            response = requests.get(source.url, timeout=30)
            response.raise_for_status()
            output_path.write_text(response.text, encoding="utf-8")
            written_paths.append(output_path)
            logger.info(f"Saved {source.name} -> {output_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {source.name}: {e}")
            # Deliberately continue to the next source rather than
            # aborting the whole fetch run over one failed request —
            # partial corpus refresh is better than none.

        time.sleep(REQUEST_DELAY_SECONDS)

    if ICH_EMA_SOURCES_TODO:
        logger.warning(
            f"NOT FETCHED (known gap, deferred): {ICH_EMA_SOURCES_TODO} — "
            f"corpus is FDA-only until this is addressed. Jurisdiction-filtered "
            f"retrieval for EMA-jurisdiction trials will have no chunks to find."
        )

    return written_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = fetch_all_fda_sources()
    print(f"\nFetched {len(paths)} FDA source(s) to {OUTPUT_DIR}/")
    print(f"NOTE: ICH-GCP and EMA sources are NOT yet fetched — see ICH_EMA_SOURCES_TODO in this file.")