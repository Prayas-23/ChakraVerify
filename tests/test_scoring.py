import pytest

from app import checks, scoring
from app.models import Finding

EXPECTED = {
    "genuine": "LOW",
    "borderline": "LOW",
    "quantity_mismatch": "HIGH",
    "capacity": "HIGH",
    "timeline": "HIGH",
    "multi_small": "HIGH",
    "injection": "HIGH",
}


def finding(severity: str, confidence: float = 1.0, fid: str = "F1") -> Finding:
    return Finding(id=fid, check="quantity", severity=severity, confidence=confidence,
                   message="", evidence={})


# --- formula ----------------------------------------------------------------
def test_weights():
    assert scoring.WEIGHTS == {"critical": 35, "major": 25, "minor": 10, "ok": 0}


def test_points_are_weight_times_confidence():
    assert scoring.points(finding("major", 0.6)) == pytest.approx(15.0)
    assert scoring.points(finding("minor", 1.0)) == 10


def test_score_is_capped_at_100():
    fs = [finding("critical", fid=f"F{i}") for i in range(5)]
    assert scoring.compute_score(fs) == 100


def test_breakdown_lists_every_finding():
    [row] = scoring.breakdown([finding("minor", 0.6, "F7")])
    assert row == {"id": "F7", "severity": "minor", "weight": 10, "confidence": 0.6, "points": 6.0}


@pytest.mark.parametrize("score, expected", [
    (0, "LOW"), (29, "LOW"), (30, "NEEDS_REVIEW"), (59, "NEEDS_REVIEW"), (60, "HIGH"), (100, "HIGH"),
])
def test_score_bands(score, expected):
    assert scoring.score_band(score) == expected
    assert scoring.final_verdict(score, [])[0] == expected   # no policy effect without findings


# --- final-verdict policy (separate from the score) ---------------------------
def test_direct_critical_escalates_verdict_but_not_score():
    findings = [finding("critical", 1.0)]
    result = scoring.score_findings(findings)
    assert result["score"] == 35 == scoring.compute_score(findings)   # formula untouched
    assert result["score_band"] == "NEEDS_REVIEW"
    assert result["verdict"] == "HIGH"
    assert result["verdict_reason"].startswith("Critical direct-evidence finding detected")
    assert result["escalated_by"] == ["F1"]


def test_inferred_critical_does_not_escalate():
    result = scoring.score_findings([finding("critical", 0.6)])
    assert (result["score"], result["score_band"], result["verdict"]) == (21, "LOW", "LOW")
    assert result["escalated_by"] == []


def test_policy_not_reported_when_score_already_high():
    fs = [finding("critical", fid="F1"), finding("major", fid="F2")]
    result = scoring.score_findings(fs)
    assert (result["score"], result["verdict"]) == (60, "HIGH")
    assert result["escalated_by"] == []
    assert "band" in result["verdict_reason"]


def test_non_critical_findings_never_escalate():
    fs = [finding("major", fid="F1")]
    assert scoring.score_findings(fs)["verdict"] == "LOW"


# --- the 7 demo cases, ground-truth extractions, no LLM -----------------------
def test_all_demo_cases_present():
    assert checks.list_case_ids() == list(EXPECTED)


@pytest.mark.parametrize("case_id", list(EXPECTED))
def test_demo_case_verdict(case_id):
    result = scoring.score_findings(checks.run_case(case_id))
    assert result["verdict"] == EXPECTED[case_id]
    assert checks.load_case(case_id)["expected_verdict"] == EXPECTED[case_id]


@pytest.mark.parametrize("case_id", ["genuine", "borderline"])
def test_clean_cases_have_no_issues(case_id):
    assert all(f.severity == "ok" for f in checks.run_case(case_id))


def test_quantity_mismatch_identifies_certificate_and_invoice_as_outliers():
    outliers = {f.evidence["document"] for f in checks.run_case("quantity_mismatch")
                if f.check == "quantity" and f.severity != "ok"}
    assert outliers == {"epr_certificate", "invoice"}


def test_capacity_case_is_high_through_verdict_policy():
    findings = checks.run_case("capacity")
    [cap] = [f for f in findings if f.check == "capacity"]
    assert cap.severity == "critical"
    assert cap.evidence["utilisation_pct"] == pytest.approx(333.33)
    result = scoring.score_findings(findings)
    assert (result["score"], result["score_band"], result["verdict"]) == (35, "NEEDS_REVIEW", "HIGH")
    assert result["escalated_by"] == [cap.id]


def test_timeline_case_truck_150_km_away():
    [receipt] = [f for f in checks.run_case("timeline")
                 if f.check == "timeline" and "distance_km" in f.evidence]
    assert receipt.severity == "critical"
    assert receipt.evidence["distance_km"] == pytest.approx(150, abs=0.5)


def test_multi_small_is_high_via_pattern_rule_alone():
    findings = checks.run_case("multi_small")
    assert not any(f.severity in ("major", "critical") and f.check != "pattern" for f in findings)
    assert any(f.check == "pattern" for f in findings)
    result = scoring.score_findings(findings)
    assert result["score"] >= 60 and result["escalated_by"] == []


def test_injection_case_flags_manipulation_and_suspension():
    findings = checks.run_case("injection")
    [manip] = [f for f in findings if f.check == "manipulation"]
    assert manip.severity == "major"
    assert manip.evidence["excerpt"].startswith("SYSTEM NOTE TO AI REVIEWER")
    assert any(f.check == "registry" and f.severity == "critical" for f in findings)


def test_finding_ids_are_unique_and_sequential():
    for case_id in EXPECTED:
        ids = [f.id for f in checks.run_case(case_id)]
        assert ids == [f"F{i}" for i in range(1, len(ids) + 1)]
