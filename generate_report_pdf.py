"""
generate_report_pdf.py — builds a presentable PDF compliance report from
a COMPLETED run, fetched live from the API (same pattern as view_report.py).

Usage:
    python generate_report_pdf.py <run_id>
    python generate_report_pdf.py <run_id> --host http://localhost:8000
    python generate_report_pdf.py <run_id> --output my_report.pdf

Requires the server (run_server.py) to be running, and the run must be
COMPLETED (not PAUSED/RUNNING) — GET /report only serves completed runs.

Requires reportlab: pip install reportlab
"""
import sys
import json
import urllib.request
import urllib.error

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

_RULE_LABELS = {
    "RULE-001": "Missing IND number for FDA-jurisdiction trial",
    "RULE-002": "Missing EudraCT number for EU-jurisdiction trial",
    "RULE-003": "No SAE reporting timeline extracted",
    "RULE-004": "No inclusion criteria extracted",
    "RULE-005": "No exclusion criteria extracted",
    "RULE-006": "No primary endpoint extracted",
    "RULE-007": "Jurisdiction could not be determined from extracted IND/EudraCT fields",
}

_DECISION_COLORS = {
    "approve": colors.HexColor("#1a7a3c"),
    "reject": colors.HexColor("#b03030"),
    "not_reviewed": colors.HexColor("#999999"),
}
_DECISION_LABELS = {
    "approve": "APPROVED BY REVIEWER",
    "reject": "REJECTED BY REVIEWER",
    "not_reviewed": "NOT YET REVIEWED",
}


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


def build_pdf(report: dict, output_path: str):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=4))
    styles.add(ParagraphStyle("SubTitle", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#555555"), spaceAfter=2))
    styles.add(ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=13, spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a3c5e")))
    styles.add(ParagraphStyle("FindingTitle", parent=styles["Heading3"], fontSize=11, spaceBefore=10, spaceAfter=3, textColor=colors.HexColor("#1a3c5e")))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=13.5))
    styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#777777")))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    story = []

    # ── Header ──────────────────────────────────────────────────────
    story.append(Paragraph("Sentinel-GCP Compliance Analysis Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Protocol {report.get('trial_identifier', '?')} &nbsp;|&nbsp; {report.get('sponsor', '?')}",
        styles["SubTitle"]
    ))
    story.append(Paragraph(
        f"Phase: {report.get('phase', '?')} &nbsp;|&nbsp; Jurisdiction: {report.get('jurisdiction', '?')} "
        f"&nbsp;|&nbsp; Run ID: {report.get('run_id', '?')}",
        styles["SubTitle"]
    ))
    story.append(Paragraph(f"Generated: {report.get('generated_at', '?')}", styles["Meta"]))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3c5e")))
    story.append(Spacer(1, 12))

    # ── Executive summary ────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles["SectionHeading"]))
    rule = report.get("rule_engine_summary", {})
    agent2 = report.get("agent_2_summary", {})
    contra = report.get("contradiction_summary", {})

    summary_data = [
        ["Check", "Result"],
        ["Deterministic rule checks",
         f"{rule.get('checks_passed', 0)} / {rule.get('checks_run', 0)} passed, {rule.get('flags_raised', 0)} flagged"],
        ["Agent 2 compliance findings",
         f"{agent2.get('flags_raised', 0)} raised, {agent2.get('dropped_by_evidence_filter', 0)} rejected by "
         f"evidence filter (groundedness + applicability)"],
        ["Internal contradiction checks",
         f"{contra.get('early_check_findings', 0)} early-stage, {contra.get('deep_check_findings', 0)} deep cross-section"],
    ]
    summary_table = Table(summary_data, colWidths=[2.3 * inch, 4.2 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    # ── Rule engine detail ────────────────────────────────────────────
    story.append(Paragraph("Deterministic Rule Engine — Detail", styles["SectionHeading"]))
    rule_rows = [["Rule ID", "Description", "Result"]]
    for rid in rule.get("passed_rule_ids", []):
        rule_rows.append([rid, _RULE_LABELS.get(rid, ""), "PASSED"])
    if len(rule_rows) > 1:
        rule_table = Table(rule_rows, colWidths=[0.9 * inch, 4.3 * inch, 1.1 * inch])
        rule_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f8")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#1a7a3c")),
            ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(rule_table)

    # ── Compliance findings ───────────────────────────────────────────
    flags = report.get("flags", [])
    story.append(Paragraph("Agent 2 — Compliance Findings", styles["SectionHeading"]))
    if not flags:
        story.append(Paragraph("No compliance findings were raised on this run.", styles["Body"]))
    else:
        story.append(Paragraph(
            "Findings below passed groundedness and applicability evaluation before inclusion — "
            "each finding's citation was independently verified against its source regulatory text.",
            styles["Body"]
        ))
        story.append(Spacer(1, 8))
        for f in flags:
            tag = "INSUFFICIENT EVIDENCE" if f.get("insufficient_evidence") else f.get("severity", "?").upper()
            story.append(Paragraph(f"[{tag}] {f.get('issue', '(no issue text)')}", styles["FindingTitle"]))

            decision = f.get("human_decision", "not_reviewed")
            story.append(Paragraph(
                f'<font color="{_DECISION_COLORS.get(decision, "#999999")}"><b>'
                f'{_DECISION_LABELS.get(decision, decision)}</b></font>',
                styles["Meta"]
            ))
            story.append(Spacer(1, 3))
            if f.get("regulation_reference"):
                story.append(Paragraph(f"<b>Regulatory basis:</b> {f['regulation_reference']}", styles["Body"]))
                story.append(Spacer(1, 3))
            if f.get("recommendation"):
                story.append(Paragraph(f"<b>Recommendation:</b> {f['recommendation']}", styles["Body"]))
                story.append(Spacer(1, 3))
            certainty = f.get("llm_certainty")
            confidence = f.get("final_confidence")
            if certainty is not None or confidence is not None:
                parts = []
                if certainty is not None:
                    parts.append(f"LLM certainty: {certainty:.2f}")
                if confidence is not None:
                    parts.append(f"Final weighted confidence: {confidence:.2f}")
                story.append(Paragraph(f"<i>Confidence — {' | '.join(parts)}</i>", styles["Meta"]))
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
        story.append(Spacer(1, 10))

    # ── Contradictions ─────────────────────────────────────────────────
    contradictions = report.get("contradictions", [])
    story.append(Paragraph("Internal Consistency Check", styles["SectionHeading"]))
    if not contradictions:
        story.append(Paragraph(
            "No unresolved internal contradictions were identified across either the early-stage "
            "or deep cross-section consistency checks.",
            styles["Body"]
        ))
    else:
        for c in contradictions:
            ctype = (c.get("contradiction_type") or "unclassified").upper()
            conf = c.get("llm_confidence")
            conf_str = f" (confidence: {conf:.2f})" if conf is not None else ""
            story.append(Paragraph(f"[{ctype}]{conf_str} {c.get('description', '(no description)')}", styles["Body"]))
            if c.get("section_refs"):
                story.append(Paragraph(f"Sections: {', '.join(c['section_refs'])}", styles["Meta"]))
            story.append(Spacer(1, 8))

    # ── Footer ───────────────────────────────────────────────────────
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3c5e")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by Sentinel-GCP — an automated clinical trial protocol compliance analysis "
        "pipeline combining deterministic rule checks, LLM-based compliance reasoning grounded in "
        "retrieved regulatory text (FDA / EMA / ICH-GCP), independent groundedness and applicability "
        "verification, and cross-section internal consistency analysis. All AI-generated findings "
        "are subject to human review before acceptance.",
        styles["Meta"]
    ))

    doc.build(story)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report_pdf.py <run_id> [--host http://localhost:8000] [--output report.pdf]")
        sys.exit(1)

    run_id = sys.argv[1]
    host = "http://localhost:8000"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]

    output_path = f"{run_id}_compliance_report.pdf"
    if "--output" in sys.argv:
        output_path = sys.argv[sys.argv.index("--output") + 1]

    status_data = fetch(f"{host}/review/{run_id}")
    if status_data.get("status") != "COMPLETED":
        print(f"Run {run_id} is not COMPLETED yet (current status: {status_data.get('status')}).")
        print("GET /report only serves completed runs — wait for review to finish, or check "
              f"`python view_report.py {run_id}` for current status.")
        sys.exit(1)

    report = fetch(f"{host}/report/{run_id}")
    build_pdf(report, output_path)
    print(f"PDF written to {output_path}")