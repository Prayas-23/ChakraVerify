"""Bounded job-store retention: oldest finished jobs are evicted first; active jobs never."""
import threading

import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.main import app


def finished(store: jobs.JobStore, n: int, how: str = "done") -> list[jobs.Job]:
    out = []
    for _ in range(n):
        job = store.create("genuine")
        job.start()
        job.complete({}) if how == "done" else job.fail("RuntimeError: x")
        out.append(job)
    return out


def ids(store: jobs.JobStore) -> list[str]:
    return list(store._jobs)


# --- configuration ------------------------------------------------------------------------
def test_default_cap(monkeypatch):
    monkeypatch.delenv(jobs.MAX_JOBS_ENV, raising=False)
    assert jobs.configured_max_jobs() == jobs.DEFAULT_MAX_JOBS == 200
    assert jobs.JobStore().max_size == 200


def test_cap_from_environment(monkeypatch):
    monkeypatch.setenv(jobs.MAX_JOBS_ENV, " 25 ")
    assert jobs.configured_max_jobs() == 25 and jobs.JobStore().max_size == 25


@pytest.mark.parametrize("raw", ["", "abc", "0", "-3", "2.5", str(jobs.MAX_JOBS_LIMIT + 1)])
def test_invalid_cap_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv(jobs.MAX_JOBS_ENV, raw)
    assert jobs.configured_max_jobs() == jobs.DEFAULT_MAX_JOBS


def test_explicit_cap_overrides_environment(monkeypatch):
    monkeypatch.setenv(jobs.MAX_JOBS_ENV, "50")
    assert jobs.JobStore(max_size=3).max_size == 3


# --- eviction policy ------------------------------------------------------------------------
def test_jobs_accessible_until_cap_is_exceeded():
    store = jobs.JobStore(max_size=3)
    created = finished(store, 3)
    assert all(store.get(j.id) is j for j in created) and len(store) == 3


def test_completed_jobs_evicted_oldest_first():
    store = jobs.JobStore(max_size=3)
    a, b, c = finished(store, 3)
    d = store.create("genuine")
    assert store.get(a.id) is None
    assert ids(store) == [b.id, c.id, d.id]


def test_failed_jobs_evicted_oldest_first():
    store = jobs.JobStore(max_size=2)
    a, b = finished(store, 2, how="failed")
    c = store.create("genuine")
    assert store.get(a.id) is None and ids(store) == [b.id, c.id]


def test_eviction_is_deterministic_and_skips_active_jobs():
    store = jobs.JobStore(max_size=3)
    [a] = finished(store, 1)
    b = store.create("genuine")
    b.start()                                   # running
    [c] = finished(store, 1, how="failed")
    d = store.create("genuine")                 # over cap → oldest finished (a) goes
    assert ids(store) == [b.id, c.id, d.id]
    e = store.create("genuine")                 # over cap → next oldest finished (c); b is still running
    assert ids(store) == [b.id, d.id, e.id]


def test_active_jobs_are_never_evicted_even_over_the_cap():
    store = jobs.JobStore(max_size=2)
    queued = store.create("genuine")
    running = store.create("genuine")
    running.start()
    extra = store.create("genuine")
    assert ids(store) == [queued.id, running.id, extra.id] and len(store) == 3  # over cap, nothing dropped
    queued.start()
    queued.complete({})
    newest = store.create("genuine")            # now one finished job can go; active ones stay
    assert ids(store) == [running.id, extra.id, newest.id]


def test_multiple_finished_jobs_evicted_at_once_after_active_backlog_clears():
    store = jobs.JobStore(max_size=2)
    backlog = [store.create("genuine") for _ in range(4)]
    for job in backlog:
        job.start()
        job.complete({})
    newest = store.create("genuine")
    assert ids(store) == [backlog[3].id, newest.id]


def test_evicted_job_object_stays_usable_for_holders():
    store = jobs.JobStore(max_size=1)
    [a] = finished(store, 1)
    held = store.get(a.id)
    store.create("genuine")
    assert store.get(a.id) is None
    assert held.snapshot()["status"] == "done"  # a request already holding the job is unaffected


def test_concurrent_creation_is_safe_and_bounded():
    store = jobs.JobStore(max_size=10)
    errors = []

    def worker():
        try:
            for _ in range(50):
                job = store.create("genuine")
                job.start()
                job.complete({})
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.create("genuine")
    assert errors == [] and len(store) <= 10


# --- API behaviour with a small store ---------------------------------------------------------
@pytest.fixture()
def small_store(monkeypatch):
    store = jobs.JobStore(max_size=2)
    monkeypatch.setattr(jobs, "store", store)
    return store


@pytest.fixture()
def client():
    return TestClient(app)


def run_verify(client, case_id: str) -> str:
    job_id = client.post("/api/verify", json={"case_id": case_id}).json()["job_id"]
    jobs.store.get(job_id).thread.join(timeout=30)
    return job_id


def test_evicted_job_returns_safe_404(client, small_store):
    first = run_verify(client, "genuine")
    assert client.get(f"/api/jobs/{first}").json()["status"] == "done"   # accessible before eviction
    run_verify(client, "capacity")
    run_verify(client, "timeline")
    response = client.get(f"/api/jobs/{first}")
    assert response.status_code == 404 and response.json() == {"detail": "Unknown job"}


def test_fingerprint_of_evicted_job_returns_safe_404(client, small_store):
    first = run_verify(client, "injection")
    assert client.get(f"/api/jobs/{first}/verify-fingerprint").json()["valid"] is True
    run_verify(client, "genuine")
    run_verify(client, "genuine")
    response = client.get(f"/api/jobs/{first}/verify-fingerprint")
    assert response.status_code == 404 and response.json() == {"detail": "Unknown job"}


def test_retained_jobs_still_work_after_eviction(client, small_store):
    run_verify(client, "genuine")
    second = run_verify(client, "quantity_mismatch")
    third = run_verify(client, "timeline")
    for job_id, (score, verdict) in ((second, (70, "HIGH")), (third, (60, "HIGH"))):
        report = client.get(f"/api/jobs/{job_id}").json()["report"]
        assert (report["score"], report["verdict"]) == (score, verdict)
        assert client.get(f"/api/jobs/{job_id}/verify-fingerprint").json()["valid"] is True
    assert len(small_store) == 2


def test_agent_endpoint_intact_with_bounded_store(client, small_store):
    bodies = [client.post("/api/agent", json={"message": m}).json() for m in ("injection", "1", "capacity")]
    assert [(b["report"]["score"], b["report"]["verdict"]) for b in bodies] == [(60, "HIGH"), (0, "LOW"), (35, "HIGH")]
    assert "SYSTEM NOTE" not in str(bodies[0])
    assert client.get(f"/api/jobs/{bodies[0]['job_id']}").status_code == 404   # oldest evicted
    assert client.get(f"/api/jobs/{bodies[2]['job_id']}").status_code == 200
    help_body = client.post("/api/agent", json={"message": "unknown"}).json()
    assert help_body["job_id"] is None and len(small_store) == 2               # help creates no job


def test_store_size_never_leaks_through_api(client, small_store):
    job_id = run_verify(client, "genuine")
    body = client.get(f"/api/jobs/{job_id}").text
    assert "max_size" not in body and "JOB_STORE" not in body and "_jobs" not in body
