"""
run_nodes_manual.py — manually invokes pipeline nodes in sequence,
WITHOUT the full compiled graph (graph/builder.py) or Postgres
checkpointing. This is the first real end-to-end test of the extraction
side of the pipeline — parse_pdf through validate_schema — against a
real protocol PDF, with real Claude API calls.

Deliberately NOT using compile_graph()/PostgresSaver yet, since that
requires Docker Postgres to be running (a separate, untested
prerequisite) — this script isolates "does the extraction logic
actually work" from "does the checkpointing infrastructure work",
so a failure in one doesn't get confused with a failure in the other.
"""
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sentinel_gcp.graph.state import initial_state
from sentinel_gcp.graph.nodes.parse_pdf import parse_pdf
from sentinel_gcp.graph.nodes.extract_discovery import extract_discovery
from sentinel_gcp.graph.nodes.extract_fill import extract_fill
from sentinel_gcp.graph.nodes.validate_schema import validate_schema

# In run_nodes_manual.py, change:
PDF_PATH = "tests/fixtures/sample_protocols/oev125_etvax.pdf"

run_id = f"manual-run-{uuid.uuid4().hex[:8]}"
state = initial_state(raw_pdf_path=PDF_PATH, run_id=run_id)

print(f"\n{'='*60}\nSTAGE 1: parse_pdf\n{'='*60}")
state = parse_pdf(state)
structure = state["document_structure"]
print(f"  total_pages: {structure.parsing_coverage.total_pages}")
print(f"  sections: {len(structure.sections)}")
print(f"  tables: {len(structure.tables)}")
print(f"  parsing_duration: {structure.parsing_coverage.parsing_duration_seconds}s")

print(f"\n{'='*60}\nSTAGE 2: extract_discovery (REAL CLAUDE API CALL)\n{'='*60}")
state = extract_discovery(state)
print(f"  label_map: {state['extraction_discovery']}")

print(f"\n{'='*60}\nSTAGE 3: extract_fill (REAL CLAUDE API CALL — full document text, cached)\n{'='*60}")
state = extract_fill(state)
print(f"  raw extraction (first 500 chars): {str(state['extraction'])[:500]}")

print(f"\n{'='*60}\nSTAGE 4: validate_schema\n{'='*60}")
state = validate_schema(state)
if state["extraction_errors"]:
    print(f"  VALIDATION FAILED: {state['extraction_errors']}")
else:
    extraction = state["extraction"]
    print(f"  VALIDATION PASSED")
    print(f"  trial_identifier: {extraction.metadata.trial_identifier.value}")
    print(f"  sponsor: {extraction.metadata.sponsor.value}")
    print(f"  phase_raw: {extraction.metadata.phase_raw}")
    print(f"  ind_number: {extraction.metadata.ind_number.value if extraction.metadata.ind_number else None}")
    print(f"  inclusion_criteria count: {len(extraction.inclusion_criteria)}")
    print(f"  exclusion_criteria count: {len(extraction.exclusion_criteria)}")
    print(f"  primary_endpoint: {extraction.primary_endpoint}")

print(f"\n{'='*60}\nDONE — run_id: {run_id}\n{'='*60}")