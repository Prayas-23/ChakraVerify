"""FastAPI app, routes, serves static/.

The /api routes are thin wrappers over the existing job store and case data:
the browser starts a verification job and polls it. All checks, scoring and
report assembly stay in the backend pipeline (app.report.run_pipeline).
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app import checks, jobs
from app import report as report_module
from app.report import public_job_view

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# Short display names for the demo-case selector, in demo order.
DEMO_CASE_LABELS = {
    "genuine": "Genuine",
    "borderline": "Borderline",
    "quantity_mismatch": "Quantity mismatch",
    "capacity": "Capacity mismatch",
    "timeline": "Timeline mismatch",
    "multi_small": "Multiple small anomalies",
    "injection": "Injection / manipulation",
}

# POST /api/agent (aiKart API-endpoint submission). The message is untrusted: it is
# only matched against the demo cases below and never reaches run_pipeline() or an LLM.
AGENT_MESSAGE_MAX_LEN = 200   # longest valid input is a ~25-character label; 200 leaves room, caps abuse
AGENT_TIMEOUT_S = 60          # DEMO_MODE finishes in well under a second; live LLM runs take longer

app = FastAPI(title="RecyVerify")


class VerifyRequest(BaseModel):
    case_id: str


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: StrictStr = Field(min_length=1, max_length=AGENT_MESSAGE_MAX_LEN)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def demo_case_aliases() -> dict[str, set[str]]:
    """Normalised alias → case IDs: canonical ID, display label and case number."""
    aliases: dict[str, set[str]] = {}
    for case_id in checks.list_case_ids():
        number = checks.load_case(case_id)["number"]
        for alias in (case_id, DEMO_CASE_LABELS.get(case_id, case_id), str(number)):
            aliases.setdefault(_normalise(alias), set()).add(case_id)
    return aliases


def match_case(message: str) -> str | None:
    """Exact (case- and whitespace-insensitive) match to one demo case; None if unmatched or ambiguous."""
    matches = demo_case_aliases().get(_normalise(message), set())
    return next(iter(matches)) if len(matches) == 1 else None


def agent_help_text() -> str:
    lines = ["SatyaSetu verifies the evidence chain of the demo EPR recycling claims below.",
             "Send one of them as the message — its number, ID or name, for example \"7\" or \"injection\":"]
    for case_id in checks.list_case_ids():
        number = checks.load_case(case_id)["number"]
        lines.append(f"{number}. {case_id} — {DEMO_CASE_LABELS.get(case_id, case_id)}")
    lines.append("Free-text claims cannot be verified here: every result must come from real evidence.")
    return "\n".join(lines)


def agent_output(report: dict) -> str:
    """Plain-text summary written by code from the redacted public report (no LLM)."""
    case_id = report["claim_id"]
    number = checks.load_case(case_id)["number"]
    issues = [f for f in report["findings"] if f["severity"] != "ok"]
    lines = [
        f"SatyaSetu verification — Case {number}: {DEMO_CASE_LABELS.get(case_id, case_id)} ({case_id})",
        f"Verdict: {report['verdict']}",
        f"Score: {report['score']}/100",
        f"Score band: {report['score_band']}",
        f"Reason: {report['verdict_reason']}",
    ]
    if issues:
        lines.append(f"Findings needing attention ({len(issues)} of {len(report['findings'])}):")
        lines += [f"- {f['id']} [{f['severity']}] {f['check']}: {f['message']}" for f in issues]
    else:
        lines.append(f"Findings: all {len(report['findings'])} checks returned ok.")
    lines.append(f"Recommendation: {report['recommendation']}")
    lines.append("Clarification notice: draft prepared — requires officer approval."
                 if report.get("clarification_notice") else "Clarification notice: none (verdict LOW).")
    lines.append(f"Report fingerprint (SHA-256): {report['fingerprint_sha256']}")
    lines.append("FINAL DECISION: HUMAN COMPLIANCE OFFICER. AI output is advisory; review the evidence "
                 "before making a compliance decision.")
    return "\n".join(lines)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/cases")
def list_cases() -> list[dict]:
    """Demo cases for the selector. The scenario text is not returned (case 7's
    scenario quotes the injected instruction)."""
    registry = checks.load_registry()
    cases = []
    for case_id in checks.list_case_ids():
        case = checks.load_case(case_id)
        reg_no = case["recycler_reg_no"]
        cases.append({
            "case_id": case_id,
            "label": DEMO_CASE_LABELS.get(case_id, case["title"]),
            "title": case["title"],
            "recycler_reg_no": reg_no,
            "recycler_name": registry.get(reg_no, {}).get("name"),
            "evidence": case["evidence"],
        })
    return cases


@app.post("/api/verify")
def verify(request: VerifyRequest) -> dict:
    if request.case_id not in checks.list_case_ids():
        raise HTTPException(status_code=404, detail="Unknown case_id")
    return {"job_id": jobs.submit(request.case_id).id}


@app.post("/api/agent")
def agent(request: AgentRequest):
    """aiKart endpoint: {"message": text} → {"output", "job_id", "report"} (synchronous).

    Only a matched case ID is passed on; the message itself goes nowhere. The report is
    the redacted public view; the score and verdict come from the deterministic pipeline.
    """
    case_id = match_case(request.message)
    if case_id is None:
        return {"output": agent_help_text(), "job_id": None, "report": None}

    job = jobs.submit(case_id)
    job.thread.join(timeout=AGENT_TIMEOUT_S)
    snapshot = job.snapshot()
    if snapshot["status"] == "done":
        report = public_job_view(snapshot)["report"]
        return {"output": agent_output(report), "job_id": job.id, "report": report}
    if snapshot["status"] == "failed":
        return JSONResponse(status_code=500, content={
            "output": "The verification could not be completed. No result is available; please try again.",
            "job_id": job.id, "report": None})
    return JSONResponse(status_code=504, content={
        "output": f"The verification is still running. Poll /api/jobs/{job.id} for the result.",
        "job_id": job.id, "report": None})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Public, redacted view: injected document text is withheld (see report.public_job_view)."""
    job = jobs.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return public_job_view(job.snapshot())


@app.get("/api/jobs/{job_id}/verify-fingerprint")
def verify_job_fingerprint(job_id: str) -> dict:
    """Recompute the SHA-256 fingerprint of the job's INTERNAL report and compare it with the
    stored one (report.fingerprint / report.verify_fingerprint, constant-time comparison).

    The internal report is used because the fingerprint covers the complete report,
    including evidence the public view withholds. Only hashes and a boolean are returned.
    404: unknown job. 409: job queued, running or failed (no report to verify).
    """
    job = jobs.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    snapshot = job.snapshot()
    if snapshot["status"] != "done" or snapshot["report"] is None:
        state = snapshot["status"]
        message = ("Job failed; there is no report to verify" if state == "failed"
                   else f"Job is {state}; no report to verify yet")
        raise HTTPException(status_code=409, detail=message)
    internal = snapshot["report"]
    stored = str(internal.get("fingerprint_sha256", ""))
    return {
        "job_id": job.id,
        "stored_fingerprint": stored,
        "computed_fingerprint": report_module.fingerprint(internal),
        "valid": report_module.verify_fingerprint(internal, stored),
    }
