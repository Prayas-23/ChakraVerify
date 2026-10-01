"""Phase 5: API routes used by the UI, the static page, and the page's view logic.

The view logic (the <script id="satyasetu-core"> block in static/index.html)
is pure JavaScript; it is extracted and run in Node against real pipeline
reports. Those tests are skipped when Node is not installed.
"""
import copy
import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app import checks, jobs
from app.main import app
from app.report import run_pipeline

INDEX = (checks.DATA_DIR.parent / "static" / "index.html").read_text(encoding="utf-8")
CASE_LABELS = ["Genuine", "Borderline", "Quantity mismatch", "Capacity mismatch", "Timeline mismatch",
               "Multiple small anomalies", "Injection / manipulation"]
INJECTED = "SYSTEM NOTE TO AI REVIEWER"
EVIDENCE_FILES = ["epr_certificate.png", "invoice.png", "weighbridge.png", "transporter.png", "gps.csv", "photo.jpg"]


@pytest.fixture()
def client():
    return TestClient(app)


# --- API routes ------------------------------------------------------------------
def test_cases_route_lists_the_seven_demo_cases_in_order(client):
    cases = client.get("/api/cases").json()
    assert [c["label"] for c in cases] == CASE_LABELS
    assert [c["case_id"] for c in cases] == checks.list_case_ids()
    assert all(c["evidence"] == EVIDENCE_FILES for c in cases)


def test_cases_route_does_not_leak_scenarios_or_expected_verdicts(client):
    body = client.get("/api/cases").text
    assert INJECTED not in body
    assert "expected_verdict" not in body and "scenario" not in body


def test_verify_and_poll_job(client):
    job_id = client.post("/api/verify", json={"case_id": "timeline"}).json()["job_id"]
    jobs.store.get(job_id).thread.join(timeout=30)
    snap = client.get(f"/api/jobs/{job_id}").json()
    assert snap["status"] == "done"
    assert (snap["report"]["score"], snap["report"]["verdict"]) == (60, "HIGH")
    assert [e["step"] for e in snap["events"]] == list(range(1, len(snap["events"]) + 1))


def test_unknown_case_and_job_are_404(client):
    assert client.post("/api/verify", json={"case_id": "../etc"}).status_code == 404
    assert client.post("/api/verify", json={}).status_code == 422
    assert client.get("/api/jobs/nope").status_code == 404


def test_network_guard_still_blocks_external_hosts(offline):
    import socket
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="network access attempted"):
            sock.connect(("93.184.216.34", 80))
    finally:
        sock.close()
        offline.clear()  # expected attempt; keep the fixture's no-network assertion for everything else


def test_only_intended_routes_exist(client):
    api_paths = {r.path for r in app.routes if r.path.startswith("/api")}
    assert api_paths == {"/api/cases", "/api/verify", "/api/jobs/{job_id}", "/api/agent",
                         "/api/jobs/{job_id}/verify-fingerprint"}
    assert {"/health", "/"} <= {r.path for r in app.routes}


# --- Static page -----------------------------------------------------------------
def test_index_served_with_branding_and_human_review_messaging(client):
    html = client.get("/").text
    assert html == INDEX
    for text in ["SATYASETU", "AI-powered evidence verification for e-waste &amp; EPR claims",
                 "Don't trust the claim. Verify the entire evidence chain.",
                 "FINAL DECISION: HUMAN COMPLIANCE OFFICER",
                 "AI output is advisory. Review evidence before making a compliance decision.",
                 "DRAFT — Requires officer approval", "Report Integrity Fingerprint",
                 "Recommendation only — final decision by compliance officer."]:
        assert text in html, text


def test_page_uses_vis_network_and_never_builds_html_from_data():
    assert "vis-network" in INDEX and "vis.Network" in INDEX
    script = INDEX[INDEX.index("<script id=\"satyasetu-core\">"):]
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in script, sink


def test_page_has_no_decision_buttons_or_client_side_hashing():
    assert not re.search(r"<button[^>]*>\s*(approve|reject)", INDEX, re.IGNORECASE)
    assert "crypto.subtle" not in INDEX and "sha256(" not in INDEX.lower()


# --- View logic in Node ---------------------------------------------------------------
NODE = shutil.which("node")
RUNNER = r"""
const core = require(process.argv[2]);
const fixtures = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const files = %s;
const out = {};
for (const [name, r] of Object.entries(fixtures)) {
  const rows = core.findingRows(r);
  const events = (r && Array.isArray(r.events) ? r.events : []).map((e) => core.eventView(e, r));
  const shown = {
    rows, verdict: core.verdictView(r), explanation: core.explanationView(r), notice: core.noticeView(r),
    events, sources: core.sourcesView(r), breakdown: core.breakdownView(r),
  };
  out[name] = {
    verdict: core.verdictView(r), ids: rows.map((x) => x.id), issues: rows.filter((x) => x.isIssue).map((x) => x.id),
    rows, fp: core.fingerprintView(r), human: core.humanReview(r), notice: core.noticeView(r),
    kinds: events.map((e) => e.kind), shown: JSON.stringify(shown),
    status: Object.fromEntries(files.map((f) => [f, core.evidenceStatus(f, r)])),
  };
}
out.__guess = Object.fromEntries(["EPR_cert_March.pdf", "tax-invoice-0042.png", "WB_ticket.jpg",
  "lorry_receipt.pdf", "truck_gps.csv", "facility_photo.jpg", "notes.txt"].map((n) => [n, core.guessType(n)]));
console.log(JSON.stringify(out));
""" % json.dumps(EVIDENCE_FILES)


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    if NODE is None:
        pytest.skip("Node.js not installed; view-logic tests need it")
    core = re.search(r'<script id="satyasetu-core">(.*?)</script>', INDEX, re.DOTALL)[1]
    tmp = tmp_path_factory.mktemp("ui")
    (tmp / "core.js").write_text(core, encoding="utf-8")
    (tmp / "runner.js").write_text(RUNNER, encoding="utf-8")

    fixtures = {cid: run_pipeline(cid).model_dump(mode="json") for cid in checks.list_case_ids()}
    review = copy.deepcopy(fixtures["multi_small"])     # synthetic: render-only, fingerprint not re-verified
    review.update(verdict="NEEDS_REVIEW", score=45, score_band="NEEDS_REVIEW")
    fixtures["needs_review"] = review
    fixtures["empty"] = {}
    fixtures["none"] = None
    fixtures["partial"] = {"verdict": "HIGH", "findings": [{"id": "F1"}]}
    fixtures["bad_fingerprint"] = {**fixtures["genuine"], "fingerprint_sha256": "not-a-hash"}
    (tmp / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")

    result = subprocess.run([NODE, str(tmp / "runner.js"), str(tmp / "core.js"), str(tmp / "fixtures.json")],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), fixtures


@pytest.mark.parametrize("case_id, label, cls", [
    ("genuine", "LOW", "v-low"), ("needs_review", "NEEDS REVIEW", "v-review"), ("quantity_mismatch", "HIGH", "v-high"),
])
def test_verdict_rendering_for_each_level(views, case_id, label, cls):
    out, fixtures = views
    v = out[case_id]["verdict"]
    assert (v["label"], v["cls"]) == (label, cls)
    assert v["scoreText"] == f"{fixtures[case_id]['score']} / 100"
    assert v["reason"] == fixtures[case_id]["verdict_reason"]


def test_capacity_shows_escalation_from_backend(views):
    v = views[0]["capacity"]["verdict"]
    assert (v["score"], v["band"], v["label"], v["escalated"]) == (35, "NEEDS_REVIEW", "HIGH", True)
    assert v["escalatedBy"] == views[1]["capacity"]["escalated_by"]


@pytest.mark.parametrize("case_id", checks.list_case_ids())
def test_findings_render_their_backend_ids(views, case_id):
    out, fixtures = views
    assert out[case_id]["ids"] == [f["id"] for f in fixtures[case_id]["findings"]]
    assert out[case_id]["issues"] == [f["id"] for f in fixtures[case_id]["findings"] if f["severity"] != "ok"]
    for row, f in zip(out[case_id]["rows"], fixtures[case_id]["findings"]):
        assert (row["severity"], row["category"], row["confidence"]) == (f["severity"], f["check"], f["confidence"])


@pytest.mark.parametrize("case_id", checks.list_case_ids())
def test_fingerprint_comes_from_backend(views, case_id):
    out, fixtures = views
    fp = out[case_id]["fp"]
    assert fp["full"] == fixtures[case_id]["fingerprint_sha256"]
    assert fp["short"].startswith(fp["full"][:12]) and fp["short"].endswith(fp["full"][-8:])


@pytest.mark.parametrize("name", ["empty", "none", "partial", "bad_fingerprint"])
def test_missing_or_invalid_report_renders_nothing_fake(views, name):
    out = views[0][name]
    if name == "bad_fingerprint":
        assert out["fp"] is None and out["ids"]  # real findings still shown, no invented hash
        return
    assert out["verdict"] is None and out["ids"] == [] and out["fp"] is None and out["notice"] is None
    assert all(s["status"] == "Pending verification" for s in out["status"].values())


def test_injection_shows_manipulation_without_the_injected_text(views):
    out, fixtures = views
    assert INJECTED in json.dumps(fixtures["injection"])            # present in the backend data …
    assert INJECTED not in out["injection"]["shown"]                 # … but never in anything displayed
    assert "pre-approved" not in out["injection"]["shown"]
    [manip] = [r for r in out["injection"]["rows"] if r["category"] == "manipulation"]
    assert manip["severity"] == "major" and manip["source"] == "Invoice"
    assert any(e["withheld"] for e in manip["evidence"])
    assert out["injection"]["status"]["invoice.png"]["status"] == "Finding detected"


@pytest.mark.parametrize("case_id, draft", [("genuine", False), ("borderline", False), ("quantity_mismatch", True),
                                            ("capacity", True), ("timeline", True), ("multi_small", True), ("injection", True)])
def test_human_review_messaging(views, case_id, draft):
    human = views[0][case_id]["human"]
    assert human["decision"] == "FINAL DECISION: HUMAN COMPLIANCE OFFICER"
    assert human["advisory"] == "AI output is advisory. Review evidence before making a compliance decision."
    assert (human["draft"] == "DRAFT — Requires officer approval") is draft
    assert (views[0][case_id]["notice"] is not None) is draft


def test_evidence_card_status_follows_backend_graph(views):
    out = views[0]
    q = out["quantity_mismatch"]["status"]
    assert q["epr_certificate.png"]["status"] == q["invoice.png"]["status"] == "Finding detected"
    assert q["transporter.png"]["status"] == q["weighbridge.png"]["status"] == "Verified"
    assert all(s["status"] == "Verified" for s in out["genuine"]["status"].values())
    m = out["multi_small"]["status"]
    assert m["photo.jpg"]["status"] == "Finding detected" and m["photo.jpg"]["cls"] == "chip-minor"


@pytest.mark.parametrize("case_id", checks.list_case_ids())
def test_activity_order_and_phase4_events_last(views, case_id):
    out, fixtures = views
    kinds = out[case_id]["kinds"]
    assert len(kinds) == len(fixtures[case_id]["events"])
    assert kinds[-3:] == ["explanation", "notice", "report"]
    assert "explanation" not in kinds[:-3] and "report" not in kinds[:-3]


# --- Fingerprint verification button --------------------------------------------------------
FP_RUNNER = r"""
const core = require(process.argv[2]);
const H1 = "a".repeat(64), H2 = "b".repeat(64), JOB = "abc123";
const VALID = { job_id: JOB, stored_fingerprint: H1, computed_fingerprint: H1, valid: true };
function fakeFetch(status, body, reject) {
  const calls = [];
  const fn = (url, init) => {
    calls.push({ url, method: init && init.method });
    if (reject) return Promise.reject(new Error("ECONNREFUSED 10.1.2.3 internal secret"));
    return Promise.resolve({ status, json: () => (body === undefined ? Promise.reject(new Error("not json")) : Promise.resolve(body)) });
  };
  fn.calls = calls;
  return fn;
}
(async () => {
  const out = { urls: {
    ok: core.fingerprintVerifyUrl(JOB), nul: core.fingerprintVerifyUrl(null), undef: core.fingerprintVerifyUrl(undefined),
    empty: core.fingerprintVerifyUrl(""), traversal: core.fingerprintVerifyUrl("../x"), num: core.fingerprintVerifyUrl(5) } };
  const scenarios = {
    valid: [200, VALID],
    invalid: [200, { job_id: JOB, stored_fingerprint: H1, computed_fingerprint: H2, valid: false }],
    not_found: [404, { detail: "Unknown job" }],
    not_ready: [409, { detail: "Job is running; no report to verify yet" }],
    server_error: [500, { detail: "Traceback: internal secret" }],
    non_json_error: [502, undefined],
    malformed: [200, { job_id: JOB, valid: "yes" }],
    wrong_job: [200, Object.assign({}, VALID, { job_id: "other" })],
    valid_without_hashes: [200, { job_id: JOB, valid: true }],
  };
  for (const [name, [status, body]] of Object.entries(scenarios)) {
    const f = fakeFetch(status, body);
    const v = core.createFingerprintVerifier(f);
    out[name] = { view: await v.run(JOB, H1), calls: f.calls, busyAfter: v.busy };
  }
  { const f = fakeFetch(0, null, true); out.network = { view: await core.createFingerprintVerifier(f).run(JOB, H1), calls: f.calls }; }
  { const f = fakeFetch(200, VALID); out.no_job = { view: await core.createFingerprintVerifier(f).run(null, H1), calls: f.calls }; }
  { const f = fakeFetch(200, VALID); out.mismatch = { view: await core.createFingerprintVerifier(f).run(JOB, H2) }; }
  {
    let release; const gate = new Promise((r) => { release = r; }); const calls = [];
    const f = (url) => { calls.push(url); return gate.then(() => ({ status: 200, json: async () => VALID })); };
    const v = core.createFingerprintVerifier(f);
    const p1 = v.run(JOB, H1); const busyDuring = v.busy; const p2 = v.run(JOB, H1);
    const callsWhileInFlight = calls.length; const samePromise = p1 === p2;
    release();
    const [r1, r2] = await Promise.all([p1, p2]);
    const busyAfter = v.busy;
    await v.run(JOB, H1);
    out.double = { callsWhileInFlight, samePromise, busyDuring, busyAfter, results: [r1.state, r2.state], callsAfterRetry: calls.length };
  }
  out.messages = core.FINGERPRINT_CHECK;
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def fp(tmp_path_factory):
    if NODE is None:
        pytest.skip("Node.js not installed; view-logic tests need it")
    core = re.search(r'<script id="satyasetu-core">(.*?)</script>', INDEX, re.DOTALL)[1]
    tmp = tmp_path_factory.mktemp("fp")
    (tmp / "core.js").write_text(core, encoding="utf-8")
    (tmp / "runner.js").write_text(FP_RUNNER, encoding="utf-8")
    result = subprocess.run([NODE, str(tmp / "runner.js"), str(tmp / "core.js")],
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_verify_button_exists_in_page():
    dom = INDEX[INDEX.index("<!-- DOM rendering."):]
    assert '"Verify fingerprint"' in dom and "btn-verify" in dom
    assert "fingerprintVerifier.run(jobId, displayed)" in dom


def test_verify_button_only_rendered_with_a_job_id():
    dom = INDEX[INDEX.index("<!-- DOM rendering."):]
    assert "C.fingerprintVerifyUrl(state.jobId) ? fingerprintVerifyButton(state.jobId" in dom
    assert "state.jobId = job_id" in dom                       # set only when the job is done
    assert re.search(r"function resetResult\(\) \{\s*state\.report = null; state\.events = \[\]; state\.jobId = null;", dom)
    assert "state.report = null; state.jobId = null;" in dom   # cleared on error


def test_verify_url_requires_a_valid_job_id(fp):
    assert fp["urls"] == {"ok": "/api/jobs/abc123/verify-fingerprint", "nul": None, "undef": None,
                          "empty": None, "traversal": None, "num": None}


def test_no_job_id_makes_no_request(fp):
    assert fp["no_job"]["calls"] == [] and fp["no_job"]["view"]["state"] == "no_job"


def test_button_calls_the_verify_endpoint_with_current_job_id(fp):
    assert fp["valid"]["calls"] == [{"url": "/api/jobs/abc123/verify-fingerprint", "method": "GET"}]


def test_valid_response_shows_success(fp):
    view = fp["valid"]["view"]
    assert view["state"] == "valid" and view["message"].startswith("Fingerprint valid")
    assert view["stored"] == view["computed"] == "a" * 64 and view["matchesDisplayed"] is True


def test_invalid_response_shows_tampering(fp):
    view = fp["invalid"]["view"]
    assert view["state"] == "invalid"
    assert view["message"] == "Fingerprint verification failed — report may have been modified."
    assert (view["stored"], view["computed"]) == ("a" * 64, "b" * 64)


def test_404_handled_safely(fp):
    view = fp["not_found"]["view"]
    assert view["state"] == "not_found" and "no longer exists" in view["message"]


def test_409_handled_safely(fp):
    view = fp["not_ready"]["view"]
    assert view["state"] == "not_ready" and "not yet ready" in view["message"]


@pytest.mark.parametrize("case", ["server_error", "non_json_error", "network", "malformed", "wrong_job",
                                  "valid_without_hashes"])
def test_errors_handled_without_exposing_details(fp, case):
    view = fp[case]["view"]
    assert view["state"] == "error"
    assert view["message"] == "Fingerprint verification could not be completed. Please try again."
    assert "secret" not in json.dumps(view) and "Traceback" not in json.dumps(view)
    assert view["stored"] is None and view["computed"] is None


def test_displayed_fingerprint_mismatch_is_flagged(fp):
    assert fp["mismatch"]["view"]["state"] == "valid" and fp["mismatch"]["view"]["matchesDisplayed"] is False


def test_no_double_submit_while_request_in_flight(fp):
    d = fp["double"]
    assert d["callsWhileInFlight"] == 1 and d["samePromise"] is True
    assert d["busyDuring"] is True and d["busyAfter"] is False
    assert d["results"] == ["valid", "valid"]
    assert d["callsAfterRetry"] == 2  # a later click makes a new request
    assert all(fp[name]["busyAfter"] is False for name in ("valid", "invalid", "not_found", "server_error"))


def test_verification_ui_does_not_hash_or_show_report_content():
    assert "crypto.subtle" not in INDEX
    dom = INDEX[INDEX.index("function renderFingerprintCheck"):INDEX.index("function renderReport")]
    for forbidden in ("report", "findings", "excerpt", "innerHTML"):
        assert forbidden not in dom.replace("reload the report", ""), forbidden


def test_upload_type_guessing(views):
    assert views[0]["__guess"] == {
        "EPR_cert_March.pdf": "epr_certificate", "tax-invoice-0042.png": "invoice", "WB_ticket.jpg": "weighbridge",
        "lorry_receipt.pdf": "transporter", "truck_gps.csv": "gps", "facility_photo.jpg": "photo", "notes.txt": "unknown",
    }
