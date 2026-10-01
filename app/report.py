"""Report assembly + evidence graph + SHA-256 fingerprint (IMPLEMENTATION_PLAN.md §9).

report.py only consumes the authoritative result of the investigation: it
never re-runs checks, never re-scores and never renumbers findings. The
recommendation is written by code and always routes to human review.

Pipeline (run_pipeline):
  orchestrator (tools + safety net) → deterministic finalisation (manipulation,
  pattern, scoring — once) → safe finding view → explanation → notice →
  report → fingerprint
"""
import copy
import hashlib
import hmac
import json
import re
from collections.abc import Callable
from datetime import datetime, timezone

from app import checks, notice
from app.models import AgentEvent, Finding, Report
from app.orchestrator import Investigation

RECOMMENDATIONS = {
    "LOW": "No material inconsistency detected. Retain evidence and complete routine human compliance review.",
    "NEEDS_REVIEW": "Additional evidence review is recommended before proceeding. "
                    "Route to the compliance officer for review.",
    "HIGH": "Material inconsistencies or high-risk evidence were detected. "
            "Escalate to the compliance officer for investigation.",
}
OFFICER_NOTE = " Recommendation only — the final decision rests with the compliance officer."

SEVERITY_RANK = {"critical": 3, "major": 2, "minor": 1, "ok": 0}
COLOURS = {"red": "#dc2626", "amber": "#f59e0b", "green": "#16a34a", "neutral": "#9ca3af"}


def recommendation_for(verdict: str) -> str:
    return RECOMMENDATIONS[verdict] + OFFICER_NOTE


# --- Evidence graph ------------------------------------------------------------------
def _kg(value) -> str:
    return f"{value:,.0f} kg" if isinstance(value, (int, float)) else "no quantity"


def _colour(severity: str | None) -> tuple[str, str]:
    """(status, colour) for the worst linked severity; None = not covered by any check."""
    if severity is None:
        return "neutral", COLOURS["neutral"]
    if severity in ("critical", "major"):
        return "red", COLOURS["red"]
    if severity == "minor":
        return "amber", COLOURS["amber"]
    return "green", COLOURS["green"]


def _links(finding: Finding) -> tuple[list[str], list[str]]:
    """Nodes and edges a finding refers to. Fixed by code, from the finding's own check/evidence."""
    doc = finding.evidence.get("document")
    if finding.check == "registry":
        return ["recycler"], ["recycler-certificate"]
    if finding.check == "quantity":
        return {"epr_certificate": (["certificate"], ["claim-certificate"]),
                "invoice": (["invoice"], ["invoice-company"]),
                "transporter": (["transporter"], ["transporter-truck"])}.get(doc, ([], []))
    if finding.check == "capacity":
        return ["facility"], ["recycler-facility"]
    if finding.check == "timeline":
        if "distance_km" in finding.evidence:
            return ["transporter", "truck", "gps"], ["truck-facility", "gps-truck"]
        return ["weighbridge", "gps"], ["weighbridge-facility", "gps-truck"]
    if finding.check == "image":
        return ["facility"], []
    if finding.check == "manipulation":
        node = {"epr_certificate": "certificate", "invoice": "invoice",
                "weighbridge": "weighbridge", "transporter": "transporter"}.get(doc)
        return ([node] if node else []), []
    if finding.check == "pattern":
        return ["claim"], []
    return [], []


def build_graph(case: dict, extractions: dict, findings: list[Finding], registry: dict) -> dict:
    """Nodes/edges for vis-network. Labels use only case metadata, the registry and
    validated extractions; colours come only from the findings."""
    ex = {doc: extractions.get(doc) or {} for doc in checks.DOC_NAMES}
    reg_no = (ex["epr_certificate"].get("recycler_reg_no") or case.get("recycler_reg_no"))
    record = registry.get(reg_no, {})
    vehicle = ex["weighbridge"].get("vehicle_no") or ex["transporter"].get("vehicle_no")

    nodes = [
        ("company", "Company", case.get("producer", "Producer")),
        ("claim", "Claim", f"Claim {case['case_id']}"),
        ("recycler", "Recycler", f"{record.get('name', 'Unknown recycler')} ({reg_no or '?'})"),
        ("certificate", "Certificate", ex["epr_certificate"].get("certificate_id") or "EPR certificate"),
        ("invoice", "Invoice", "Invoice"),
        ("weighbridge", "Weighbridge", "Weighbridge ticket"),
        ("transporter", "Transporter", "Transporter document"),
        ("truck", "Truck", vehicle or "Truck"),
        ("gps", "GPS", "GPS log"),
        ("facility", "Facility", f"Facility ({record.get('state', '?')})"),
    ]
    time = lambda doc: f" at {ex[doc]['time']}" if ex[doc].get("time") else ""  # noqa: E731
    edges = [
        ("company-claim", "company", "claim", "EPR claim"),
        ("claim-certificate", "claim", "certificate", _kg(ex["epr_certificate"].get("quantity_kg"))),
        ("recycler-certificate", "recycler", "certificate", "issued"),
        ("invoice-company", "invoice", "company", _kg(ex["invoice"].get("quantity_kg"))),
        ("recycler-invoice", "recycler", "invoice", "billed"),
        ("truck-weighbridge", "truck", "weighbridge",
         _kg(ex["weighbridge"].get("quantity_kg")) + time("weighbridge")),
        ("weighbridge-facility", "weighbridge", "facility", "weighed at"),
        ("transporter-truck", "transporter", "truck",
         _kg(ex["transporter"].get("quantity_kg")) + time("transporter")),
        ("truck-facility", "truck", "facility", "delivered to"),
        ("gps-truck", "gps", "truck", "tracks"),
        ("recycler-facility", "recycler", "facility",
         f"{record['capacity_kg_per_day']:,} kg/day capacity" if record else "operates"),
    ]

    node_hits: dict[str, list[Finding]] = {n[0]: [] for n in nodes}
    edge_hits: dict[str, list[Finding]] = {e[0]: [] for e in edges}
    for f in findings:
        linked_nodes, linked_edges = _links(f)
        for n in linked_nodes:
            node_hits[n].append(f)
        for e in linked_edges:
            edge_hits[e].append(f)

    def worst(hits: list[Finding]) -> str | None:
        return max((f.severity for f in hits), key=SEVERITY_RANK.get, default=None)

    graph = {"nodes": [], "edges": []}
    for node_id, group, label in nodes:
        severity = worst(node_hits[node_id])
        status, colour = _colour(severity)
        graph["nodes"].append({"id": node_id, "group": group, "label": label, "severity": severity,
                               "status": status, "color": colour,
                               "finding_ids": [f.id for f in node_hits[node_id]]})
    for edge_id, src, dst, label in edges:
        severity = worst(edge_hits[edge_id])
        status, colour = _colour(severity)
        graph["edges"].append({"id": edge_id, "from": src, "to": dst, "label": label, "severity": severity,
                               "status": status, "color": colour,
                               "finding_ids": [f.id for f in edge_hits[edge_id]]})
    return graph


# --- Fingerprint ------------------------------------------------------------------------
def canonical_json(report: Report | dict) -> bytes:
    data = (report if isinstance(report, Report) else Report.model_validate(report)).model_dump(mode="json")
    data.pop("fingerprint_sha256", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def fingerprint(report: Report | dict) -> str:
    return hashlib.sha256(canonical_json(report)).hexdigest()


def verify_fingerprint(report: Report | dict, expected: str | None = None) -> bool:
    if expected is None:
        expected = report.fingerprint_sha256 if isinstance(report, Report) else report.get("fingerprint_sha256", "")
    return hmac.compare_digest(fingerprint(report), str(expected).lower())


# --- Public API view ------------------------------------------------------------------------
REDACTION_MARKER = "[instruction-like text withheld]"


def _injected_texts(report: dict) -> list[str]:
    texts = []
    for f in report.get("findings") or []:
        excerpt = (f.get("evidence") or {}).get("excerpt") if f.get("check") == "manipulation" else None
        if isinstance(excerpt, str):
            texts += [part.strip() for part in excerpt.split(" | ") if len(part.strip()) >= 8]
    return texts


def _scrub(value, texts: list[str]):
    if isinstance(value, str):
        for text in texts:
            value = re.sub(re.escape(text), REDACTION_MARKER, value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [_scrub(v, texts) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, texts) for k, v in value.items()}
    return value


def public_job_view(snapshot: dict) -> dict:
    """Copy of a job snapshot that is safe to serve over the API.

    The internal Report keeps the injected excerpt (audit trail, fingerprint).
    The public copy replaces each manipulation finding's excerpt with
    REDACTION_MARKER and scrubs that text from any other string. Everything
    else — IDs, severities, categories, confidences, ordinary evidence and the
    fingerprint of the internal report — is returned unchanged. The fingerprint
    therefore verifies against the internal report, not this view.
    """
    view = copy.deepcopy(snapshot)
    report = view.get("report")
    redactions: list[str] = []
    texts: list[str] = []
    if isinstance(report, dict):
        texts = _injected_texts(report)
        for f in report.get("findings") or []:
            evidence = f.get("evidence") or {}
            if f.get("check") == "manipulation" and isinstance(evidence.get("excerpt"), str):
                evidence["excerpt"] = REDACTION_MARKER
                redactions.append(f"{f.get('id')}.evidence.excerpt")
    view = _scrub(view, texts)
    view["redactions"] = redactions
    return view


# --- Report assembly ----------------------------------------------------------------------
def build_report(result: dict, generated_at: str | None = None) -> Report:
    """result: the investigation result plus explanation/notice (see run_pipeline)."""
    case = checks.load_case(result["case_id"])
    findings = [Finding.model_validate(f) for f in result["findings"]]
    scoring = result["scoring"]
    report = Report(
        claim_id=result["case_id"],
        verdict=scoring["verdict"],
        score=scoring["score"],
        score_breakdown=scoring["breakdown"],
        findings=findings,
        explanation=result["explanation"],
        recommendation=recommendation_for(scoring["verdict"]),
        clarification_notice=result.get("notice"),
        graph=build_graph(case, result["extractions"], findings, checks.load_registry()),
        fingerprint_sha256="",
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        score_band=scoring["score_band"],
        verdict_reason=scoring["verdict_reason"],
        escalated_by=scoring["escalated_by"],
        formula=scoring["formula"],
        policy=scoring["policy"],
        agent_summary=result.get("agent_summary", ""),
        planner=result.get("planner", ""),
        case_title=case.get("title", ""),
        extraction_sources=result.get("extraction_sources", {}),
        events=result.get("events", []),
        explanation_source=result.get("explanation_source", ""),
        notice_source=result.get("notice_source", ""),
    )
    report.fingerprint_sha256 = fingerprint(report)
    return report


def run_pipeline(case_id: str, emit: Callable[[AgentEvent], None] | None = None) -> Report:
    investigation = Investigation(case_id, emit)
    result = investigation.run()  # tools, safety net, manipulation, pattern, scoring: each once
    findings = [Finding.model_validate(f) for f in result["findings"]]
    scoring = result["scoring"]

    explanation = notice.write_explanation(findings, scoring, case_id)
    investigation.log_event(
        "Explanation Agent", "write_explanation()",
        "Put the code-produced findings into words; only valid finding IDs and supported numbers allowed",
        f"Explanation {'written by LLM and validated' if explanation.source.startswith('llm') else 'written by code'}"
        f" ({explanation.note})",
        "warning" if "rejected" in explanation.note else "ok")

    cert = result["extractions"].get("epr_certificate", {})
    reg_no = cert.get("recycler_reg_no")
    record = checks.load_registry().get(reg_no or "", {})
    claim = {"case_id": case_id, "recycler_reg_no": reg_no, "recycler_name": record.get("name"),
             "certificate_id": cert.get("certificate_id")}
    draft = notice.draft_notice(findings, scoring, claim)
    issues = sum(f.severity != "ok" for f in findings)
    investigation.log_event(
        "Notice Agent", "draft_notice()",
        "Draft a clarification request to the recycler for officer approval",
        (f"Draft notice covering {issues} finding(s) — requires officer approval ({draft.note})"
         if draft.text else draft.note),
        "warning" if "rejected" in draft.note else "ok")

    investigation.log_event(
        "Report Builder", "build_report()",
        "Assemble the authoritative findings, score, evidence graph and SHA-256 fingerprint",
        f"Report ready: score {scoring['score']}/100, verdict {scoring['verdict']} — "
        "recommendation for human review", "ok")

    return build_report({
        **result,
        "events": [e.model_dump() for e in investigation.events],
        "explanation": explanation.text,
        "explanation_source": explanation.source,
        "notice": draft.text,
        "notice_source": draft.source,
    })
