"""
run_nodes_manual.py — manually invokes pipeline nodes in sequence,
WITHOUT the full compiled graph (graph/builder.py) or Postgres
checkpointing. Tests nodes 1-8: parse_pdf through retrieve, with
diagnostics for the persistent Pinecone 0-results bug.
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
from sentinel_gcp.graph.nodes.contradiction_check import contradiction_check
from sentinel_gcp.graph.nodes.determine_jurisdiction import determine_jurisdiction
from sentinel_gcp.graph.nodes.rule_engine import rule_engine
from sentinel_gcp.graph.nodes.retrieve import retrieve
from sentinel_gcp.rules.definitions import RULES
from sentinel_gcp.retrieval.pinecone_store import PineconeStore
from pinecone import Pinecone
from sentinel_gcp.config import settings

_RULE_DESCRIPTIONS = {rule.rule_id: rule.description for rule in RULES}

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

print(f"\n{'='*60}\nSTAGE 3: extract_fill (REAL CLAUDE API CALL)\n{'='*60}")
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
    print(f"  eudract_number: {extraction.metadata.eudract_number.value if extraction.metadata.eudract_number else None}")
    print(f"  inclusion_criteria count: {len(extraction.inclusion_criteria)}")
    print(f"  exclusion_criteria count: {len(extraction.exclusion_criteria)}")
    print(f"  primary_endpoint: {extraction.primary_endpoint}")

    print(f"\n{'='*60}\nSTAGE 5: contradiction_check (early)\n{'='*60}")
    state = contradiction_check(state)
    findings = state["early_contradiction_findings"]
    print(f"  {len(findings)} finding(s)")
    for f in findings:
        print(f"    - {f.description} (sections: {f.section_refs})")

    print(f"\n{'='*60}\nSTAGE 6: determine_jurisdiction\n{'='*60}")
    state = determine_jurisdiction(state)
    print(f"  jurisdiction: {state['jurisdiction']}")

    print(f"\n{'='*60}\nSTAGE 7: rule_engine\n{'='*60}")
    state = rule_engine(state)
    rule_results = state["rule_results"]
    flagged = [r for r in rule_results if not r.passed]
    passed = [r for r in rule_results if r.passed]
    print(f"  {len(rule_results)} rule(s) checked — {len(passed)} passed, {len(flagged)} flagged\n")
    for r in rule_results:
        description = _RULE_DESCRIPTIONS.get(r.rule_id, "unknown rule")
        if r.passed:
            print(f"  ✓ {r.rule_id}: {description}")
        else:
            print(f"  ✗ {r.rule_id}: {description}")
            print(f"      → {r.flag.issue} (severity={r.flag.severity})")

    print(f"\n{'='*60}\nSTAGE 8: retrieve — Pinecone diagnostics\n{'='*60}")

    print("  === DIAGNOSTIC A: raw index stats ===")
    _raw_client = Pinecone(api_key=settings.PINECONE_API_KEY)
    _raw_index = _raw_client.Index(settings.PINECONE_INDEX_NAME)
    stats = _raw_index.describe_index_stats()
    print(f"  total_vector_count: {stats.get('total_vector_count')}")
    print(f"  namespaces: {stats.get('namespaces')}")

    print("\n  === DIAGNOSTIC B: raw search response shape ===")
    _raw_response = _raw_index.search(
        namespace="default",
        inputs={"text": "SAE reporting timeline requirement"},
        top_k=3,
    )
    print(f"  type: {type(_raw_response)}")
    print(f"  raw response: {_raw_response}")

    print("\n  === DIAGNOSTIC C: our store, no filter ===")
    _debug_store = PineconeStore()
    _debug_results = _debug_store.query("SAE reporting timeline requirement", jurisdiction_filter=None, top_k=3)
    print(f"  chunks returned: {len(_debug_results)}")

    print("\n  === Real retrieve() call ===")
    state = retrieve(state)
    chunks = state["retrieved_chunks"]
    print(f"  {len(chunks)} chunk(s) retrieved (jurisdiction-filtered)")
    for c in chunks:
        print(f"    - [{c['topic']}] {c['regulation_source']} (score={c['score']:.3f})")

print(f"\n{'='*60}\nDONE — run_id: {run_id}\n{'='*60}")