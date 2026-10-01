"""Risk score formula (IMPLEMENTATION_PLAN.md §7). Pure code, no LLM.

Two separate steps:

1. Score (numerical formula, never altered by any policy):
       points(finding) = weight × confidence
       score           = min(100, round(sum(points)))
       score band      = 0–29 LOW, 30–59 NEEDS_REVIEW, ≥60 HIGH

2. Final-verdict policy (applied after the score is calculated):
       a critical finding backed by direct evidence (confidence 1.0) escalates
       the final verdict to HIGH. The score is left as is, so a report can
       legitimately show "Score 35/100 — Verdict HIGH".
"""
from app.models import Finding

WEIGHTS = {"critical": 35, "major": 25, "minor": 10, "ok": 0}
MAX_SCORE = 100
LOW_MAX = 29
NEEDS_REVIEW_MAX = 59

ESCALATION_SEVERITY = "critical"
ESCALATION_MIN_CONFIDENCE = 1.0
ESCALATION_REASON = "Critical direct-evidence finding detected"

FORMULA = (
    "points = weight × confidence (critical 35, major 25, minor 10, ok 0); "
    "score = min(100, round(sum(points))); "
    "0–29 LOW, 30–59 NEEDS_REVIEW, ≥60 HIGH"
)
POLICY = "Verdict policy: any critical finding with confidence 1.0 → HIGH (score unchanged)"


# --- Step 1: score ------------------------------------------------------------
def points(finding: Finding) -> float:
    return WEIGHTS[finding.severity] * finding.confidence


def breakdown(findings: list[Finding]) -> list[dict]:
    return [
        {
            "id": f.id,
            "severity": f.severity,
            "weight": WEIGHTS[f.severity],
            "confidence": f.confidence,
            "points": round(points(f), 2),
        }
        for f in findings
    ]


def compute_score(findings: list[Finding]) -> int:
    return min(MAX_SCORE, round(sum(points(f) for f in findings)))


def score_band(score: int) -> str:
    if score > NEEDS_REVIEW_MAX:
        return "HIGH"
    if score > LOW_MAX:
        return "NEEDS_REVIEW"
    return "LOW"


# --- Step 2: final-verdict policy --------------------------------------------
def escalating_findings(findings: list[Finding]) -> list[str]:
    return [
        f.id for f in findings
        if f.severity == ESCALATION_SEVERITY and f.confidence >= ESCALATION_MIN_CONFIDENCE
    ]


def final_verdict(score: int, findings: list[Finding]) -> tuple[str, str]:
    """Return (verdict, reason)."""
    band = score_band(score)
    escalators = escalating_findings(findings)
    if escalators and band != "HIGH":
        return "HIGH", f"{ESCALATION_REASON} ({', '.join(escalators)})"
    return band, f"Score {score}/100 falls in the {band} band"


def score_findings(findings: list[Finding]) -> dict:
    score = compute_score(findings)
    verdict, reason = final_verdict(score, findings)
    return {
        "score": score,
        "score_band": score_band(score),
        "verdict": verdict,
        "verdict_reason": reason,
        "escalated_by": escalating_findings(findings) if verdict != score_band(score) else [],
        "breakdown": breakdown(findings),
        "formula": FORMULA,
        "policy": POLICY,
    }
