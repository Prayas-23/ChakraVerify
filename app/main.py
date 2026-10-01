"""FastAPI app, routes, serves static/.

The /api routes are thin wrappers over the existing job store and case data:
the browser starts a verification job and polls it. All checks, scoring and
report assembly stay in the backend pipeline (app.report.run_pipeline).
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import checks, jobs
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

app = FastAPI(title="RecyVerify")


class VerifyRequest(BaseModel):
    case_id: str


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


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Public, redacted view: injected document text is withheld (see report.public_job_view)."""
    job = jobs.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return public_job_view(job.snapshot())
