"""
run_nodes_manual.py — manually invokes pipeline nodes in sequence.

DEV COST OPTIMIZATION: checkpoints state after Stage 4 (post-extraction,
the most expensive stage — extract_fill sends the full document, ~41K
tokens) and after Stage 8 (post-retrieval). When testing changes to
LATER nodes only (compliance_check, deep_contradiction_check, etc.),
set RESUME_FROM to skip straight past the already-paid-for stages
instead of re-running and re-paying for them.

NOTE: RESUME_FROM only works if the corresponding checkpoint file
already exists on THIS machine (checkpoints are not committed to git).
On a fresh clone / fresh machine with no dev_checkpoints/ yet, leave
this as None for the first run so both checkpoints get created here.
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
from sentinel_gcp.graph.nodes.compliance_check import compliance_check
from sentinel_gcp.graph.nodes.deep_contradiction_check import deep_contradiction_check
from sentinel_gcp.rules.definitions import RULES
from dev_checkpoint import save_checkpoint, load_checkpoint, list_checkpoints

_RULE_DESCRIPTIONS = {rule.rule_id: rule.description for rule in RULES}

PDF_PATH = "tests/fixtures/sample_protocols/oev125_etvax.pdf"

# ─── SET THIS to skip expensive already-verified stages ───────────────
# None            = run everything from scratch (REQUIRED on a machine
#                    with no existing dev_checkpoints/ — see note above)
# "after_stage4"  = skip Stages 1-4 (parse+discovery+fill+validate)
# "after_stage8"  = skip Stages 1-8 (also skip contradiction/jurisdiction/
#                    rules/retrieve) — use when testing compliance_check
#                    or deep_contradiction_check only, and ONLY once an
#                    "after_stage8" checkpoint already exists on this
#                    machine from a prior full run
RESUME_FROM = None
# ────────────────────────────────────────────────────────────────────

list_checkpoints()

state = None
if RESUME_FROM:
    state = load_checkpoint(RESUME_FROM)
    if state is None:
        logger.warning(
            "RESUME_FROM=%r was set but no matching checkpoint exists on "
            "this machine — falling back to a full run from Stage 1.",
            RESUME_FROM,
        )

# Track how far the loaded checkpoint actually got us, so the two stage
# blocks below don't silently skip work a stale/mismatched checkpoint
# never did. Previously this was inferred from RESUME_FROM's value alone,
# which broke when RESUME_FROM="after_stage8" but no such checkpoint
# existed yet (state stayed at post-Stage-4, but Stages 5-8 were still
# skipped) — that's what produced the empty compliance_check run.
resumed_stage = 0
if state is not None:
    resumed_stage = 8 if RESUME_FROM == "after_stage8" else 4

if state is None:
    run_id = f"manual-run-{uuid.uuid4().hex[:8]}"
    state = initial_state(raw_pdf_path=PDF_PATH, run_id=run_id)

    print(f"\n{'='*60}\nSTAGE 1: parse_pdf\n{'='*60}")
    state = parse_pdf(state)
    structure = state["document_structure"]
    print(f"  total_pages: {structure.parsing_coverage.total_pages}, "
          f"sections: {len(structure.sections)}, tables: {len(structure.tables)}, "
          f"parsing_duration: {structure.parsing_coverage.parsing_duration_seconds}s")

    print(f"\n{'='*60}\nSTAGE 2: extract_discovery (REAL CLAUDE API CALL)\n{'='*60}")
    state = extract_discovery(state)

    print(f"\n{'='*60}\nSTAGE 3: extract_fill (REAL CLAUDE API CALL — ~$0.28)\n{'='*60}")
    state = extract_fill(state)

    print(f"\n{'='*60}\nSTAGE 4: validate_schema\n{'='*60}")
    state = validate_schema(state)

    if state["extraction_errors"]:
        print(f"  VALIDATION FAILED: {state['extraction_errors']}")
        raise SystemExit("Stopping — validation failed, nothing to checkpoint")

    extraction = state["extraction"]
    print(f"  VALIDATION PASSED — {extraction.metadata.trial_identifier.value}")
    save_checkpoint(state, "after_stage4")
    resumed_stage = 4

if resumed_stage < 8:
    print(f"\n{'='*60}\nSTAGE 5: contradiction_check (early)\n{'='*60}")
    state = contradiction_check(state)
    print(f"  {len(state['early_contradiction_findings'])} finding(s)")

    print(f"\n{'='*60}\nSTAGE 6: determine_jurisdiction\n{'='*60}")
    state = determine_jurisdiction(state)
    print(f"  jurisdiction: {state['jurisdiction']}")

    print(f"\n{'='*60}\nSTAGE 7: rule_engine\n{'='*60}")
    state = rule_engine(state)
    rule_results = state["rule_results"]
    passed = [r for r in rule_results if r.passed]
    flagged = [r for r in rule_results if not r.passed]
    print(f"  {len(passed)} passed, {len(flagged)} flagged")
    for r in rule_results:
        desc = _RULE_DESCRIPTIONS.get(r.rule_id, "unknown")
        print(f"  {'✓' if r.passed else '✗'} {r.rule_id}: {desc}")

    print(f"\n{'='*60}\nSTAGE 8: retrieve (Pinecone — no LLM cost)\n{'='*60}")
    state = retrieve(state)
    chunks = state["retrieved_chunks"]
    print(f"  {len(chunks)} chunk(s) retrieved")

    save_checkpoint(state, "after_stage8")

print(f"\n{'='*60}\nSTAGE 9: compliance_check (Agent 2) — REAL CLAUDE API CALL\n{'='*60}")
state = compliance_check(state)
flags = state["agent_2_flags"]
findings = [f for f in flags if not f.insufficient_evidence]
notes = [f for f in flags if f.insufficient_evidence]
print(f"  {len(findings)} finding(s), {len(notes)} insufficient-evidence note(s)")
for f in flags:
    tag = "NOTE" if f.insufficient_evidence else f.severity.upper()
    print(f"    [{tag}] {f.issue}")

print(f"\n{'='*60}\nSTAGE 10: deep_contradiction_check — REAL CLAUDE API CALL\n{'='*60}")
state = deep_contradiction_check(state)
deep_findings = state["deep_contradiction_findings"]
print(f"  {len(deep_findings)} unresolved contradiction(s) found")
for f in deep_findings:
    print(f"    - {f.description} (sections: {f.section_refs})")

print(f"\n{'='*60}\nDONE\n{'='*60}")