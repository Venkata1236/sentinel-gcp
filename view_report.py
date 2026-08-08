"""
view_report.py — fetches a run's report from the live API and prints it
in a human-readable format, instead of the raw JSON GET /report returns.

Usage:
    python view_report.py <run_id>
    python view_report.py <run_id> --host http://localhost:8000

Requires the server (run_server.py) to be running.
"""
import sys
import json
import urllib.request
import urllib.error


def fetch(url: str) -> dict:
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} from {url}:\n{body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Could not reach {url} — is the server running? ({e})")
        sys.exit(1)


def print_report(report: dict):
    print(f"\n{'='*70}")
    print(f"  {report.get('trial_identifier', '?')} — {report.get('sponsor', '?')}")
    print(f"  Phase {report.get('phase', '?')} | Jurisdiction: {report.get('jurisdiction', '?')} | Run: {report.get('run_id', '?')}")
    print(f"  Generated: {report.get('generated_at', '?')}")
    print(f"{'='*70}")

    rule_summary = report.get("rule_engine_summary", {})
    print(f"\nRULE ENGINE — {rule_summary.get('checks_passed', 0)}/{rule_summary.get('checks_run', 0)} checks passed, "
          f"{rule_summary.get('flags_raised', 0)} flagged")
    passed = rule_summary.get("passed_rule_ids", [])
    if passed:
        print(f"  Passed: {', '.join(passed)}")

    agent_2_summary = report.get("agent_2_summary", {})
    print(f"\nAGENT 2 (COMPLIANCE) — {agent_2_summary.get('flags_raised', 0)} raised, "
          f"{agent_2_summary.get('dropped_by_evidence_filter', 0)} dropped by evidence filter")

    contradiction_summary = report.get("contradiction_summary", {})
    print(f"\nCONTRADICTIONS — {contradiction_summary.get('early_check_findings', 0)} early-stage, "
          f"{contradiction_summary.get('deep_check_findings', 0)} deep-check")

    flags = report.get("flags", [])
    if flags:
        print(f"\n{'-'*70}\nFINDINGS ({len(flags)})\n{'-'*70}")
        for f in flags:
            tag = "NOTE" if f.get("insufficient_evidence") else f.get("severity", "?").upper()
            decision = f.get("human_decision", "not_reviewed")
            decision_tag = f" [{decision.upper()}]" if decision != "not_reviewed" else ""
            print(f"\n  [{tag}]{decision_tag} {f.get('issue', '(no issue text)')}")
            if f.get("regulation_reference"):
                print(f"    Regulation: {f['regulation_reference']}")
            if f.get("recommendation"):
                print(f"    Recommendation: {f['recommendation']}")
            certainty = f.get("llm_certainty")
            confidence = f.get("final_confidence")
            if certainty is not None or confidence is not None:
                parts = []
                if certainty is not None:
                    parts.append(f"llm_certainty={certainty:.2f}")
                if confidence is not None:
                    parts.append(f"final_confidence={confidence:.2f}")
                print(f"    ({', '.join(parts)})")
    else:
        print("\nNo compliance findings.")

    dropped = report.get("evidence_filter_dropped", [])
    if dropped:
        print(f"\n{'-'*70}\nDROPPED BY EVIDENCE FILTER ({len(dropped)})\n{'-'*70}")
        for d in dropped:
            print(f"\n  [{d.get('reason_type', '?')}] {d.get('issue', '(no issue text)')}")
            print(f"    {d.get('reasoning', '')}")

    contradictions = report.get("contradictions", [])
    if contradictions:
        print(f"\n{'-'*70}\nCONTRADICTIONS ({len(contradictions)})\n{'-'*70}")
        for c in contradictions:
            ctype = (c.get("contradiction_type") or "unclassified").upper()
            conf = c.get("llm_confidence")
            conf_str = f" conf={conf:.2f}" if conf is not None else ""
            print(f"\n  [{ctype}{conf_str}] {c.get('description', '(no description)')}")
            if c.get("section_refs"):
                print(f"    Sections: {', '.join(c['section_refs'])}")

    print(f"\n{'='*70}\n")


def print_review_status(data: dict):
    status = data.get("status")
    print(f"\nRun {data.get('run_id', '?')}: status = {status}")

    if status in ("PENDING", "RUNNING"):
        print(f"  {data.get('message', '')}")
        return
    if status == "FAILED":
        print(f"  Detail: {data.get('detail', '(no detail)')}")
        return
    if status == "COMPLETED":
        print(f"  {data.get('message', '')}")
        return

    flags = data.get("flags", [])
    print(f"  Trial: {data.get('trial_identifier', '?')} | Jurisdiction: {data.get('jurisdiction', '?')}")
    print(f"  {len(flags)} flag(s) awaiting review:\n")
    for f in flags:
        tag = "NOTE" if f.get("insufficient_evidence") else f.get("severity", "?").upper()
        print(f"  [{tag}] flag_id={f.get('flag_id')}")
        print(f"    {f.get('issue', '(no issue text)')}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_report.py <run_id> [--host http://localhost:8000]")
        sys.exit(1)

    run_id = sys.argv[1]
    host = "http://localhost:8000"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]

    # Try the review endpoint first (works for PENDING/RUNNING/PAUSED/FAILED),
    # fall back to report if the run is COMPLETED.
    review_data = fetch(f"{host}/review/{run_id}")

    if review_data.get("status") == "COMPLETED":
        report = fetch(f"{host}/report/{run_id}")
        print_report(report)
    else:
        print_review_status(review_data)