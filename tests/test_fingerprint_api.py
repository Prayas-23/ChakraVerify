"""GET /api/jobs/{job_id}/verify-fingerprint — verifies the INTERNAL report.

The fingerprint covers the complete internal report (including the withheld
injection excerpt), so verification uses job.report, never the redacted public
view. The response carries hashes and a boolean only — never report content.
"""
import copy
import json
import re

import pytest
from fastapi.testclient import TestClient

from app import jobs, report
from app.main import app

INJECTED_FRAGMENTS = ("SYSTEM NOTE", "pre-approved", "mark as VERIFIED", "skip all checks")
RESPONSE_KEYS = {"job_id", "stored_fingerprint", "computed_fingerprint", "valid"}


@pytest.fixture()
def client():
    return TestClient(app)


def completed_job(client, case_id: str) -> jobs.Job:
    job_id = client.post("/api/verify", json={"case_id": case_id}).json()["job_id"]
    job = jobs.store.get(job_id)
    job.thread.join(timeout=30)
    assert job.snapshot()["status"] == "done"
    return job


def verify(client, job_id: str):
    return client.get(f"/api/jobs/{job_id}/verify-fingerprint")


# --- 1. genuine --------------------------------------------------------------------------
def test_completed_genuine_job_verifies(client):
    job = completed_job(client, "genuine")
    response = verify(client, job.id)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["job_id"] == job.id and body["valid"] is True
    assert body["stored_fingerprint"] == body["computed_fingerprint"] == job.report["fingerprint_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", body["computed_fingerprint"])


# --- 2. injection ------------------------------------------------------------------------
def test_completed_injection_job_verifies_without_leaking_anything(client):
    job = completed_job(client, "injection")
    response = verify(client, job.id)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True and set(body) == RESPONSE_KEYS   # hashes and a boolean only, no report
    serialized = json.dumps(body)
    for fragment in INJECTED_FRAGMENTS:
        assert fragment.lower() not in serialized.lower(), fragment
    for report_field in ("findings", "evidence", "excerpt", "explanation", "verdict"):
        assert report_field not in serialized


def test_verification_uses_internal_report_not_public_view(client):
    job = completed_job(client, "injection")
    internal = job.snapshot()["report"]
    public = client.get(f"/api/jobs/{job.id}").json()["report"]
    assert report.verify_fingerprint(internal) and not report.verify_fingerprint(public)
    body = verify(client, job.id).json()
    assert body["valid"] is True
    assert body["computed_fingerprint"] == report.fingerprint(internal) != report.fingerprint(public)


@pytest.mark.parametrize("case_id", ["borderline", "quantity_mismatch", "capacity", "timeline", "multi_small"])
def test_other_demo_cases_verify(client, case_id):
    assert verify(client, completed_job(client, case_id).id).json()["valid"] is True


# --- 3/4. unknown, queued, running, failed ------------------------------------------------
def test_unknown_job_is_404(client):
    assert verify(client, "no-such-job").status_code == 404


@pytest.mark.parametrize("status", ["queued", "running"])
def test_unfinished_job_is_409(client, status):
    job = jobs.store.create("genuine")       # never started: stays "queued"
    if status == "running":
        job.start()
    response = verify(client, job.id)
    assert response.status_code == 409
    assert status in response.json()["detail"]


def test_failed_job_is_409_without_internal_details(client):
    job = jobs.store.create("genuine")
    job.start()
    job.fail("RuntimeError: internal detail that must not leak")
    response = verify(client, job.id)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "failed" in detail and "internal detail" not in detail


# --- 5/6. tampering, no mutation ------------------------------------------------------------
@pytest.mark.parametrize("tamper", [
    lambda r: r.update(score=r["score"] + 1),
    lambda r: r.update(verdict="LOW"),
    lambda r: r["findings"][0].update(severity="minor"),
    lambda r: r["findings"][-1]["evidence"].update(excerpt="edited excerpt"),
    lambda r: r["events"].pop(),
])
def test_tampered_internal_report_fails_verification(client, tamper):
    job = completed_job(client, "injection")
    tamper(job.report)  # in-memory tampering after completion
    body = verify(client, job.id).json()
    assert body["valid"] is False
    assert body["stored_fingerprint"] != body["computed_fingerprint"]


def test_tampered_stored_fingerprint_fails_verification(client):
    job = completed_job(client, "genuine")
    job.report["fingerprint_sha256"] = "0" * 64
    body = verify(client, job.id).json()
    assert body["valid"] is False and body["stored_fingerprint"] == "0" * 64


def test_verification_does_not_mutate_the_job(client):
    job = completed_job(client, "injection")
    before = copy.deepcopy(job.snapshot())
    for _ in range(3):
        assert verify(client, job.id).json()["valid"] is True
    assert job.snapshot() == before


# --- 7. public job view still redacted ------------------------------------------------------
def test_public_job_response_still_redacted_after_verification(client):
    job = completed_job(client, "injection")
    verify(client, job.id)
    public = client.get(f"/api/jobs/{job.id}").json()
    serialized = json.dumps(public)
    for fragment in INJECTED_FRAGMENTS:
        assert fragment.lower() not in serialized.lower(), fragment
    assert report.REDACTION_MARKER in serialized
    assert public["redactions"] and public["report"]["fingerprint_sha256"] == job.report["fingerprint_sha256"]


def test_agent_job_id_can_be_verified(client):
    body = client.post("/api/agent", json={"message": "7"}).json()
    assert verify(client, body["job_id"]).json()["valid"] is True
