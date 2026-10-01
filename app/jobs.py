"""In-memory job store + event log.

Jobs live in process memory only (no database). A job runs in a background
thread; its snapshot() is what GET /api/jobs/{id} will return:
{"job_id", "case_id", "status", "events": [...], "report": {...} | None, "error"}.
"""
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from app.models import AgentEvent

STATUSES = ("queued", "running", "done", "failed")


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
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, case_id: str) -> Job:
        job = Job(case_id)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


store = JobStore()
Runner = Callable[[str, Callable[[AgentEvent], None]], dict]


def _run(job: Job, runner: Runner) -> None:
    job.start()
    try:
        job.complete(runner(job.case_id, job.add_event))
    except Exception as exc:  # report any failure on the job instead of killing the thread silently
        job.fail(f"{type(exc).__name__}: {exc}")


def submit(case_id: str, runner: Runner | None = None) -> Job:
    """Create a job and run it in a background thread."""
    if runner is None:
        from app.orchestrator import run_investigation
        runner = run_investigation
    job = store.create(case_id)
    job.thread = threading.Thread(target=_run, args=(job, runner), daemon=True, name=f"job-{job.id}")
    job.thread.start()
    return job
