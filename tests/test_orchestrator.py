import json

import pytest

from app import checks, jobs, llm, orchestrator
from app.orchestrator import Investigation, run_investigation
from app.scoring import score_findings

EXPECTED = {"genuine": "LOW", "borderline": "LOW", "quantity_mismatch": "HIGH", "capacity": "HIGH",
            "timeline": "HIGH", "multi_small": "HIGH", "injection": "HIGH"}


def tc(name: str, **args) -> dict:
    return {"name": name, "args": {"reason": f"test reason for {name}", **args}}


class FakeAgentLLM:
    """Stands in for llm.chat: plays scripted tool-call turns, records every message
    list it receives, and refuses extraction calls (tools=None) so extraction falls
    back to ground truth."""

    def __init__(self, turns: list[list[dict]], repeat_last: bool = False):
        self.turns, self.repeat_last, self.seen = list(turns), repeat_last, []

    def __call__(self, messages, tools=None, response_format=None):
        if tools is None:
            raise llm.LLMUnavailable("extraction not faked")
        self.seen.append(json.loads(json.dumps(messages)))
        turn = self.turns[0] if self.repeat_last and len(self.turns) == 1 else (
            self.turns.pop(0) if self.turns else [tc("finish", summary="done")])
        calls = [{"id": f"c{len(self.seen)}-{i}", "name": c["name"], "arguments": json.dumps(c["args"])}
                 for i, c in enumerate(turn)]
        return {"content": "", "tool_calls": calls, "cached": False}


def tools_called(inv: Investigation, origin: str | None = None) -> list[str]:
    return [t["tool"] for t in inv.tool_log if origin is None or t["origin"] == origin]


def run(case_id: str) -> tuple[Investigation, dict]:
    inv = Investigation(case_id)
    return inv, inv.run()


# --- DEMO_MODE integration (deterministic planner, zero API calls) -------------
@pytest.mark.parametrize("case_id", ["genuine", "quantity_mismatch", "injection"])
def test_demo_integration_key_cases(case_id):
    inv, result = run(case_id)
    assert result["planner"] == "scripted"
    assert result["scoring"]["verdict"] == EXPECTED[case_id]
    assert set(result["extraction_sources"].values()) == {"ground_truth"}
    assert result["safety_net_calls"] == 0  # the planner already ran every required check


@pytest.mark.parametrize("case_id", list(EXPECTED))
def test_demo_findings_identical_to_phase2_engine(case_id):
    result = run_investigation(case_id)
    phase2 = checks.run_case(case_id)
    assert [(f["id"], f["check"], f["severity"], f["evidence"]) for f in result["findings"]] == \
           [(f.id, f.check, f.severity, f.evidence) for f in phase2]
    assert result["scoring"] == score_findings(phase2)
    assert result["scoring"]["verdict"] == EXPECTED[case_id]


def test_evidence_listed_first_then_certificate_then_recycler():
    inv, _ = run("genuine")
    assert tools_called(inv)[:3] == ["list_evidence", "extract_document", "lookup_recycler"]
    assert inv.tool_log[1]["args"]["doc_name"] == "epr_certificate"


def test_quantity_mismatch_triggers_transporter_inspection():
    inv, result = run("quantity_mismatch")
    called = tools_called(inv, "agent")
    assert called.index("inspect_transporter_doc") == called.index("check_quantities") + 1
    assert "epr_certificate, invoice" in result["agent_summary"]
    [inspect_event] = [e for e in result["events"] if e["action"].startswith("inspect_transporter_doc")]
    assert "Outlier document(s): epr_certificate, invoice" in inspect_event["result_summary"]


def test_every_tool_invocation_creates_an_event():
    for case_id in EXPECTED:
        inv, result = run(case_id)
        tool_events = [e for e in result["events"] if e["action"].split("(")[0] in orchestrator.TOOL_NAMES]
        assert len(tool_events) == len(inv.tool_log)
        for event in tool_events:
            assert event["reason"] and event["result_summary"] and event["agent"]


def test_events_are_streamed_to_emit_callback():
    received = []
    result = run_investigation("timeline", emit=received.append)
    assert [e.model_dump() for e in received] == result["events"]
    assert [e.step for e in received] == list(range(1, len(received) + 1))


def test_injection_case_flags_manipulation_in_events():
    _, result = run("injection")
    risk = [e for e in result["events"] if e["action"] == "check_manipulation()"]
    assert risk[0]["status"] == "alert"
    assert any(f["check"] == "manipulation" and f["severity"] == "major" for f in result["findings"])


# --- LLM path with a fake model -------------------------------------------------
def test_llm_planner_runs_tools_and_is_used(monkeypatch):
    cid = "quantity_mismatch"
    fake = FakeAgentLLM([
        [tc("list_evidence", case_id=cid)],
        [tc("extract_document", case_id=cid, doc_name="epr_certificate")],
        [tc("lookup_recycler", reg_no="R004")],
        [tc("check_quantities", case_id=cid)],
        [tc("inspect_transporter_doc", case_id=cid)],
        [tc("check_capacity", case_id=cid), tc("check_timeline", case_id=cid), tc("check_image", case_id=cid)],
        [tc("finish", summary="Certificate and invoice are the outliers.")],
    ])
    monkeypatch.setattr(llm, "chat", fake)
    inv, result = run(cid)
    assert result["planner"] == "llm"
    assert result["agent_summary"] == "Certificate and invoice are the outliers."
    assert result["safety_net_calls"] == 0
    assert result["scoring"]["verdict"] == "HIGH"
    reasons = [e["reason"] for e in result["events"] if e["action"].startswith("check_capacity")]
    assert reasons == ["test reason for check_capacity"]  # the model's stated reason is logged
    # tool results were returned to the model as tool messages
    assert fake.seen[-1][-1]["role"] == "tool"


def test_manipulation_instructions_are_never_followed(monkeypatch):
    """A model that 'obeys' the injected note and finishes early cannot change the outcome."""
    cid = "injection"
    fake = FakeAgentLLM([
        [tc("list_evidence", case_id=cid)],
        [tc("extract_document", case_id=cid, doc_name="epr_certificate")],
        [tc("extract_document", case_id=cid, doc_name="invoice")],
        [tc("finish", summary="Claim is pre-approved. VERIFIED. Skipping all checks.")],
    ])
    monkeypatch.setattr(llm, "chat", fake)
    inv, result = run(cid)

    # the injected text never reached the orchestrator model
    assert "SYSTEM NOTE" not in json.dumps(fake.seen)
    assert "[withheld" in json.dumps(fake.seen)
    # the safety net ran every skipped check and code decided the verdict
    assert result["safety_net_calls"] > 0
    assert {"lookup_recycler", "inspect_transporter_doc", "check_capacity",
            "check_timeline", "check_image"} <= set(tools_called(inv, "safety_net"))
    assert result["scoring"]["verdict"] == "HIGH"
    assert result["scoring"] == score_findings(checks.run_case(cid))
    assert any(f["check"] == "manipulation" and f["severity"] == "major" for f in result["findings"])


def test_step_limit_is_enforced(monkeypatch):
    fake = FakeAgentLLM([[tc("list_evidence", case_id="capacity")]], repeat_last=True)
    monkeypatch.setattr(llm, "chat", fake)
    inv, result = run("capacity")
    assert result["steps"] == orchestrator.MAX_STEPS
    assert len(fake.seen) == orchestrator.MAX_STEPS
    assert any(e["action"] == "step_limit" for e in result["events"])
    assert result["scoring"]["verdict"] == "HIGH"  # safety net still completed the checks


def test_step_limit_applies_within_a_single_turn(monkeypatch):
    monkeypatch.setattr(llm, "chat", FakeAgentLLM([[tc("list_evidence", case_id="genuine")] * 20]))
    _, result = run("genuine")
    assert result["steps"] == orchestrator.MAX_STEPS


def test_guards_enforce_order_when_model_skips_ahead(monkeypatch):
    monkeypatch.setattr(llm, "chat", FakeAgentLLM([[tc("check_capacity", case_id="capacity")]]))
    inv, _ = run("capacity")
    assert tools_called(inv)[:3] == ["list_evidence", "extract_document", "check_capacity"]
    assert [t["origin"] for t in inv.tool_log[:3]] == ["guard", "guard", "agent"]


def test_bad_tool_calls_are_rejected_safely(monkeypatch):
    fake = FakeAgentLLM([[tc("delete_findings"), tc("check_capacity", case_id="genuine"),
                          tc("extract_document", case_id="capacity", doc_name="../../secrets")]])
    monkeypatch.setattr(llm, "chat", fake)
    _, result = run("capacity")
    errors = [m for m in fake.seen[1] if m["role"] == "tool"]
    assert all("error" in json.loads(m["content"]) for m in errors) and len(errors) == 3
    assert result["scoring"]["verdict"] == "HIGH"


def test_wrong_recycler_lookup_is_corrected_by_safety_net(monkeypatch):
    cid = "injection"
    monkeypatch.setattr(llm, "chat", FakeAgentLLM([[tc("lookup_recycler", reg_no="R001")]]))
    inv, result = run(cid)
    assert inv.registry_checked_for == "R005"
    assert any(f["check"] == "registry" and f["severity"] == "critical" for f in result["findings"])


def test_llm_failure_mid_run_switches_to_deterministic_planner(monkeypatch):
    calls = {"n": 0}

    def flaky(messages, tools=None, response_format=None):
        if tools is None:
            raise llm.LLMUnavailable("extraction")
        calls["n"] += 1
        if calls["n"] > 1:
            raise llm.LLMUnavailable("provider down")
        return {"content": "", "tool_calls": [{"id": "c1", "name": "list_evidence",
                                               "arguments": '{"case_id": "timeline", "reason": "start"}'}]}

    monkeypatch.setattr(llm, "chat", flaky)
    _, result = run("timeline")
    assert result["planner"] == "llm+scripted"
    assert result["scoring"]["verdict"] == "HIGH"
    assert result["safety_net_calls"] == 0


def test_prose_reply_without_tool_call_finishes(monkeypatch):
    def prose(messages, tools=None, response_format=None):
        if tools is None:
            raise llm.LLMUnavailable("extraction")
        return {"content": "Everything looks fine.", "tool_calls": []}

    monkeypatch.setattr(llm, "chat", prose)
    _, result = run("multi_small")
    assert result["agent_summary"] == "Everything looks fine."
    assert result["scoring"]["verdict"] == "HIGH"


def test_unknown_case_is_rejected():
    with pytest.raises(ValueError):
        Investigation("../../etc")


# --- jobs -----------------------------------------------------------------------
def test_job_runs_in_background_and_completes():
    job = jobs.submit("quantity_mismatch")
    job.thread.join(timeout=30)
    snap = jobs.store.get(job.id).snapshot()
    assert snap["status"] == "done"
    assert snap["events"] and snap["events"][0]["agent"] == "Orchestrator"
    assert snap["report"]["scoring"]["verdict"] == "HIGH"
    assert set(snap) >= {"status", "events", "report"}


def test_job_failure_is_recorded():
    def broken(case_id, emit):
        raise RuntimeError("boom")

    job = jobs.submit("genuine", runner=broken)
    job.thread.join(timeout=5)
    snap = job.snapshot()
    assert snap["status"] == "failed" and "boom" in snap["error"] and snap["report"] is None


def test_unknown_case_job_fails_cleanly():
    job = jobs.submit("no_such_case")
    job.thread.join(timeout=5)
    assert job.snapshot()["status"] == "failed"


def test_job_store_lookup():
    job = jobs.store.create("genuine")
    assert jobs.store.get(job.id) is job
    assert jobs.store.get("missing") is None
    assert job.snapshot()["status"] == "queued"
