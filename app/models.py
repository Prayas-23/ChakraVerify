"""Pydantic models (Finding, Extraction, AgentEvent, Report)."""
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["ok", "minor", "major", "critical"]


class Finding(BaseModel):
    id: str                     # e.g. "F3"; assigned after all checks have run
    check: str                  # "quantity" | "capacity" | "timeline" | "registry" | "image" | "manipulation" | "pattern"
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)  # 1.0 direct evidence, 0.6 inferred
    message: str                # human-readable, produced by code
    evidence: dict              # the exact values compared


class Extraction(BaseModel):
    doc_type: Literal["epr_certificate", "invoice", "weighbridge", "transporter", "unknown"]
    recycler_name: str | None = None
    recycler_reg_no: str | None = None
    quantity_kg: float | None = None
    date: str | None = None             # ISO date
    time: str | None = None             # HH:MM
    vehicle_no: str | None = None
    certificate_id: str | None = None
    suspicious_content: bool = False    # true if the document contains instructions aimed at software/AI
    suspicious_excerpt: str | None = None


class AgentEvent(BaseModel):
    step: int
    agent: str                  # "Orchestrator", "Document Agent", "Capacity Agent", ...
    action: str                 # what it did
    reason: str                 # why it chose to do it (shown in UI)
    result_summary: str
    status: Literal["running", "ok", "warning", "alert"]


class Report(BaseModel):
    claim_id: str
    verdict: Literal["LOW", "NEEDS_REVIEW", "HIGH"]
    score: int
    score_breakdown: list[dict]  # per finding: id, severity, weight, confidence, points
    findings: list[Finding]
    explanation: str             # LLM-written, cites finding IDs only
    recommendation: str          # always routes to human review
    clarification_notice: str | None
    graph: dict                  # {nodes: [...], edges: [...]} for vis-network
    fingerprint_sha256: str
    generated_at: str
    # Authoritative scoring detail (copied from app.scoring, never recomputed)
    score_band: str = ""
    verdict_reason: str = ""
    escalated_by: list[str] = Field(default_factory=list)
    formula: str = ""
    policy: str = ""
    # Agent run
    agent_summary: str = ""
    planner: str = ""
    case_title: str = ""
    extraction_sources: dict[str, str] = Field(default_factory=dict)
    events: list[AgentEvent] = Field(default_factory=list)
    explanation_source: str = ""        # llm | llm-cache | fallback
    notice_source: str = ""             # llm | llm-cache | fallback | none
