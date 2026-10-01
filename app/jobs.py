"""In-memory job store + event log.

Jobs live in process memory only (no database). A job runs in a background
thread; its snapshot() is what GET /api/jobs/{id} will return:
{"job_id", "case_id", "status", "events": [...], "report": {...} | None, "error"}.

Retention is bounded: at most JOB_STORE_MAX_SIZE jobs are kept (env var,
default 200). When a new job pushes the store over the cap, the oldest
finished (done/failed) jobs are evicted first, in creation order. Queued and
running jobs are never evicted, so the store may exceed the cap while that
many jobs are active. An evicted job is simply unknown (404).
"""
import os
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from app.models import AgentEvent

STATUSES = ("queued", "running", "done", "failed")
FINISHED_STATUSES = ("done", "failed")
DEFAULT_MAX_JOBS = 200        # each report is tens of KB; 200 keeps memory small and covers a demo session
MAX_JOBS_LIMIT = 10_000       # upper bound for the env setting
MAX_JOBS_ENV = "JOB_STORE_MAX_SIZE"


def configured_max_jobs() -> int:
    """JOB_STORE_MAX_SIZE if it is an integer in 1..MAX_JOBS_LIMIT, else DEFAULT_MAX_JOBS."""
    try:
        value = int(os.getenv(MAX_JOBS_ENV, "").strip())
    except ValueError:
        return DEFAULT_MAX_JOBS
    return value if 1 <= value <= MAX_JOBS_LIMIT else DEFAULT_MAX_JOBS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job:
    def __init__(self, case_id: str):
        self.id = uuid.uuid4().hex[:12]
        self.case_id = case_id
        self.status = "queued"
        self.events: list[AgentEvent] = []
        self.report: dict | None = None
        self.error: str | None = None
        self.created_at = _now()
        self.finished_at: str | None = None
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def add_event(self, event: AgentEvent) -> None:
        with self._lock:
            self.events.append(event)

    def start(self) -> None:
        with self._lock:
            self.status = "running"

    def complete(self, report: dict) -> None:
        with self._lock:
            self.report, self.status, self.finished_at = report, "done", _now()

    def fail(self, error: str) -> None:
        with self._lock:
            self.error, self.status, self.finished_at = error, "failed", _now()

    def is_finished(self) -> bool:
        with self._lock:
            return self.status in FINISHED_STATUSES

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.id,
                "case_id": self.case_id,
                "status": self.status,
                "events": [e.model_dump() for e in self.events],
                "report": self.report,
                "error": self.error,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
            }


class JobStore:
    def __init__(self, max_size: int | None = None):
        self._jobs: dict[str, Job] = {}   # insertion order = creation order (oldest first)
        self._lock = threading.Lock()
        self.max_size = max_size if max_size is not None else configured_max_jobs()

    def create(self, case_id: str) -> Job:
        job = Job(case_id)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_finished_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def _evict_finished_locked(self) -> None:
        """Drop the oldest finished jobs until within max_size; active jobs are never dropped.
        Lock order is always store → job, and Job never takes the store lock, so no deadlock."""
        excess = len(self._jobs) - self.max_size
        if excess <= 0:
            return
        for job_id in [jid for jid, job in self._jobs.items() if job.is_finished()][:excess]:
            del self._jobs[job_id]


store = JobStore()
Runner = Callable[[str, Callable[[AgentEvent], None]], dict]


def _run(job: Job, runner: Runner) -> None:
    job.start()
    try:
        job.complete(runner(job.case_id, job.add_event))
    except Exception as exc:  # report any failure on the job instead of killing the thread silently
        job.fail(f"{type(exc).__name__}: {exc}")


def run_report(case_id: str, emit: Callable[[AgentEvent], None]) -> dict:
    """Default runner: the full Phase 4 pipeline; the job's report is the Report as JSON."""
    from app.report import run_pipeline
    return run_pipeline(case_id, emit).model_dump(mode="json")


def submit(case_id: str, runner: Runner | None = None) -> Job:
    """Create a job and run it in a background thread."""
    if runner is None:
        runner = run_report
    job = store.create(case_id)
    job.thread = threading.Thread(target=_run, args=(job, runner), daemon=True, name=f"job-{job.id}")
    job.thread.start()
    return job
