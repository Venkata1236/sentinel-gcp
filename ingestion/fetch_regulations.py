"""
ingestion/fetch_regulations.py — pulls raw regulatory source text.

This is a ONE-TIME / PERIODIC script, not part of the runtime graph —
run manually (or on a schedule) to build/refresh the local regulatory
corpus before chunk_and_embed.py processes it and upsert_to_pinecone.py
loads it into the vector store. Deliberately kept separate from
sentinel_gcp/ (the runtime package) per the original file-structure
decision.

Sources fetched (matching what the project's Phase 1 study covered):
  - FDA 21 CFR 312 (via eCFR, machine-readable API)
  - ICH-GCP E6(R3) — official EMA-hosted Step 5 PDF. Tagged jurisdiction
    "ICH" (not "EMA") since this guideline applies broadly across FDA
    and EMA trials, not exclusively European ones.
  - EU Clinical Trials Regulation (EU) No 536/2014 — official EUR-Lex
    consolidated PDF. Tagged jurisdiction "EMA", NOT "ICH" — this is
    EU-specific procedural law (submission timelines, CTIS, multi-
    Member-State coordination), distinct from ICH-GCP's cross-
    jurisdiction scientific/ethical standard.

Each fetched document is tagged with its jurisdiction at THIS stage,
not later — this is what lets chunk_and_embed.py propagate jurisdiction
metadata all the way through to Pinecone, and what makes
retrieval/pinecone_store.py's $in filter fix meaningful (ICH-tagged
content matches BOTH FDA and EMA queries; EMA-tagged content like the
EU CTR matches only EMA/both queries).

FIXES applied via real testing, not just code review:
  1. eCFR's versioner API rejects the literal word "current" in the URL
     path and requires an actual date; it also lags 1-2 business days
     behind the Federal Register, so requesting today's date can 404.
     _SAFE_ECFR_DATE uses 5 days back to safely clear weekends/holidays.
  2. eCFR responses were decoding with mojibake (Â§ instead of §) —
     requests was guessing the wrong charset. response.encoding is now
     forced to "utf-8" before reading .text.
"""
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import date, timedelta

import requests

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/regulations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_SECONDS = 1.0  # basic politeness delay between requests

# eCFR lags 1-2 business days behind the Federal Register — requesting
# today's date can 404 if today's snapshot doesn't exist yet. Using a
# date a few days back is safer than "today" or the literal word "current".
_SAFE_ECFR_DATE = (date.today() - timedelta(days=5)).isoformat()


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
        url=f"https://www.ecfr.gov/api/versioner/v1/full/{_SAFE_ECFR_DATE}/title-21.xml?part=312&section=312.23",
        jurisdiction="FDA",
        output_filename="21_cfr_312_23.xml",
    ),
    RegulationSource(
        name="21 CFR 312.32 — IND safety reporting",
        url=f"https://www.ecfr.gov/api/versioner/v1/full/{_SAFE_ECFR_DATE}/title-21.xml?part=312&section=312.32",
        jurisdiction="FDA",
        output_filename="21_cfr_312_32.xml",
    ),
    RegulationSource(
        name="21 CFR 312.30 — Protocol amendments",
        url=f"https://www.ecfr.gov/api/versioner/v1/full/{_SAFE_ECFR_DATE}/title-21.xml?part=312&section=312.30",
        jurisdiction="FDA",
        output_filename="21_cfr_312_30.xml",
    ),
    RegulationSource(
        name="21 CFR 312.33 — Annual reports",
        url=f"https://www.ecfr.gov/api/versioner/v1/full/{_SAFE_ECFR_DATE}/title-21.xml?part=312&section=312.33",
        jurisdiction="FDA",
        output_filename="21_cfr_312_33.xml",
    ),
]

# ICH-GCP — official EMA-hosted Step 5 (finalized) guideline. Tagged
# "ICH" deliberately, not "EMA" — see module docstring above.
ICH_SOURCES = [
    RegulationSource(
        name="ICH E6(R3) — Guideline for Good Clinical Practice (Step 5, EMA-hosted)",
        url="https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e6-r3-guideline-good-clinical-practice-gcp-step-5_en.pdf",
        jurisdiction="ICH",
        output_filename="ich_e6_r3_gcp.pdf",
    ),
]

# EU Clinical Trials Regulation — official EUR-Lex consolidated PDF.
# Tagged "EMA", NOT "ICH" — this is EU-specific procedural/legal
# framework (CTIS submission, multi-Member-State coordination,
# EudraVigilance safety reporting), distinct from ICH-GCP's broadly-
# applicable scientific/ethical standard.
EU_SOURCES = [
    RegulationSource(
        name="Regulation (EU) No 536/2014 — EU Clinical Trials Regulation",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014R0536",
        jurisdiction="EMA",
        output_filename="eu_ctr_536_2014.pdf",
    ),
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
            response.encoding = "utf-8"  # force correct encoding; eCFR sends UTF-8 but
                                           # requests sometimes guesses wrong without an
                                           # explicit charset in the response headers,
                                           # causing mojibake (Â§ instead of §)
            output_path.write_text(response.text, encoding="utf-8")
            written_paths.append(output_path)
            logger.info(f"Saved {source.name} -> {output_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {source.name}: {e}")
            # Deliberately continue to the next source rather than
            # aborting the whole fetch run over one failed request —
            # partial corpus refresh is better than none.

        time.sleep(REQUEST_DELAY_SECONDS)

    return written_paths


def fetch_all_ich_sources() -> list[Path]:
    """PDF sources fetch differently from eCFR's XML — raw binary write
    (response.content, not response.text), so there's no encoding
    concern here at all (that was an XML/text-specific issue)."""
    written_paths = []

    for source in ICH_SOURCES:
        output_path = OUTPUT_DIR / source.output_filename
        logger.info(f"Fetching {source.name} from {source.url}")

        try:
            response = requests.get(source.url, timeout=60)  # PDFs are larger; longer timeout
            response.raise_for_status()
            output_path.write_bytes(response.content)
            written_paths.append(output_path)
            logger.info(f"Saved {source.name} -> {output_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {source.name}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return written_paths


def fetch_all_eu_sources() -> list[Path]:
    """Same binary-write pattern as fetch_all_ich_sources(). A custom
    User-Agent header is included defensively — some EU government sites
    reject requests with no User-Agent at all; verify this is actually
    necessary once run for real, and remove if not."""
    written_paths = []

    for source in EU_SOURCES:
        output_path = OUTPUT_DIR / source.output_filename
        logger.info(f"Fetching {source.name} from {source.url}")

        try:
            response = requests.get(
                source.url,
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            output_path.write_bytes(response.content)
            written_paths.append(output_path)
            logger.info(f"Saved {source.name} -> {output_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {source.name}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return written_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fda_paths = fetch_all_fda_sources()
    ich_paths = fetch_all_ich_sources()
    eu_paths = fetch_all_eu_sources()

    print(
        f"\nFetched {len(fda_paths)} FDA + {len(ich_paths)} ICH + "
        f"{len(eu_paths)} EU source(s) to {OUTPUT_DIR}/"
    )