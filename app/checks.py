"""Deterministic check functions -> Findings (IMPLEMENTATION_PLAN.md §6).

Every number in a Finding is computed here, never by the LLM. Check functions
take plain values and return list[Finding] with id="" — IDs are assigned by
assign_ids() once all checks have run. The loaders at the bottom read case
data from data/ and run_case() runs every check on one case.
"""
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import piexif
from PIL import Image

from app.models import Extraction, Finding, Severity

# --- Thresholds -------------------------------------------------------------
QTY_OK_MAX = 0.02                  # relative diff vs weighbridge ≤ 2 % → ok
QTY_MINOR_MAX = 0.05               # ≤ 5 % → minor, above → major
CAPACITY_OK_MAX = 0.90             # utilisation ≤ 90 % → ok
CAPACITY_MINOR_MAX = 1.00          # ≤ 100 % → minor, above → critical
DISTANCE_MAJOR_KM = 5.0            # truck > 5 km from facility at receipt → major
DISTANCE_CRITICAL_KM = 20.0        # > 20 km → critical
ARRIVAL_RADIUS_KM = 1.0            # first GPS fix within this radius = arrival
WEIGHBRIDGE_EARLY_MAJOR_MIN = 30   # weighbridge before arrival: < 30 min minor, ≥ 30 major
GPS_MAX_GAP_MIN = 15               # nearest GPS fix further away in time → inferred
IMAGE_HASH_MAX_DISTANCE = 5        # average-hash Hamming distance counted as "same photo"
PATTERN_MIN_FINDINGS = 3
PATTERN_MIN_SOURCES = 3

CONF_DIRECT = 1.0
CONF_INFERRED = 0.6
EARTH_RADIUS_KM = 6371.0

# Independent evidence category behind each check (used by check_pattern).
EVIDENCE_SOURCE = {
    "registry": "registry",
    "quantity": "documents",
    "manipulation": "documents",
    "timeline": "logistics_gps",
    "capacity": "facility_capacity",
    "image": "image",
}

DOC_NAMES = ("epr_certificate", "invoice", "weighbridge", "transporter")
DOC_LABELS = {
    "epr_certificate": "EPR certificate",
    "invoice": "Invoice",
    "weighbridge": "Weighbridge ticket",
    "transporter": "Transporter document",
}


def _finding(check: str, severity: Severity, message: str, evidence: dict,
             confidence: float = CONF_DIRECT) -> Finding:
    return Finding(id="", check=check, severity=severity, confidence=confidence,
                   message=message, evidence=evidence)


def _kg(value: float) -> str:
    return f"{value:,.0f} kg"


def _norm_name(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", name.lower()).split())


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


# --- Checks -----------------------------------------------------------------
def check_registry(reg_no: str | None, name: str | None, registry: dict[str, dict]) -> list[Finding]:
    record = registry.get(reg_no) if reg_no else None
    if record is None:
        return [_finding("registry", "critical",
                         f"Registration number {reg_no!r} not found in the recycler registry",
                         {"reg_no": reg_no, "name_on_document": name})]

    findings = []
    if record["status"] != "registered":
        findings.append(_finding("registry", "critical",
                                 f"Recycler {reg_no} ({record['name']}) has registry status "
                                 f"'{record['status']}'",
                                 {"reg_no": reg_no, "registry_status": record["status"]}))
    if name and _norm_name(name) != _norm_name(record["name"]):
        findings.append(_finding("registry", "major",
                                 f"Name on document '{name}' does not match registry name "
                                 f"'{record['name']}' for {reg_no}",
                                 {"reg_no": reg_no, "name_on_document": name,
                                  "registry_name": record["name"]}))
    if not findings:
        findings.append(_finding("registry", "ok",
                                 f"Recycler {reg_no} ({record['name']}) is registered and the name matches",
                                 {"reg_no": reg_no, "registry_status": record["status"]}))
    return findings


def compare_quantities(cert: float | None, invoice: float | None, weighbridge: float | None,
                       transporter: float | None = None) -> list[Finding]:
    """One finding per document, each compared with the weighbridge net weight.

    When the transporter document independently agrees with the weighbridge
    (≤ 2 %), a document that is > 5 % off is escalated to critical: two
    physical weight records confirm it is the outlier.
    """
    if not weighbridge:
        return []

    transporter_agrees = (transporter is not None
                          and abs(transporter - weighbridge) / weighbridge <= QTY_OK_MAX)
    findings = []
    for doc, qty in (("epr_certificate", cert), ("invoice", invoice), ("transporter", transporter)):
        if qty is None:
            continue
        diff = abs(qty - weighbridge) / weighbridge
        severity: Severity = "ok" if diff <= QTY_OK_MAX else "minor" if diff <= QTY_MINOR_MAX else "major"
        evidence = {"document": doc, "document_kg": qty, "weighbridge_kg": weighbridge,
                    "relative_diff_pct": round(diff * 100, 2)}
        message = (f"{DOC_LABELS[doc]} states {_kg(qty)} vs weighbridge {_kg(weighbridge)} "
                   f"({diff * 100:.2f}% difference)")
        if severity == "major" and doc != "transporter" and transporter_agrees:
            severity = "critical"
            evidence["transporter_kg"] = transporter
            message += (f"; the transporter document independently records {_kg(transporter)}, "
                        f"so the {DOC_LABELS[doc].lower()} is the outlier")
        findings.append(_finding("quantity", severity, message, evidence))
    return findings


def check_capacity(reg_no: str | None, quantity_kg: float | None, period_days: float,
                   registry: dict[str, dict]) -> list[Finding]:
    record = registry.get(reg_no) if reg_no else None
    if record is None or quantity_kg is None or period_days <= 0:
        return []

    capacity = record["capacity_kg_per_day"]
    utilisation = round(quantity_kg / (capacity * period_days), 4)
    severity: Severity = ("ok" if utilisation <= CAPACITY_OK_MAX
                          else "minor" if utilisation <= CAPACITY_MINOR_MAX else "critical")
    return [_finding("capacity", severity,
                     f"Claimed {_kg(quantity_kg)} over {period_days:g} day(s) is "
                     f"{utilisation * 100:.1f}% of {reg_no}'s capacity "
                     f"({_kg(capacity)}/day)",
                     {"reg_no": reg_no, "quantity_kg": quantity_kg, "period_days": period_days,
                      "capacity_kg_per_day": capacity,
                      "utilisation_pct": round(utilisation * 100, 2)})]


def check_timeline(gps_rows: list[dict], facility_latlon: tuple[float, float],
                   receipt_time: datetime | None, weighbridge_time: datetime | None) -> list[Finding]:
    """gps_rows: [{"timestamp": datetime, "lat": float, "lon": float}, ...]"""
    if not gps_rows:
        return []
    rows = sorted(gps_rows, key=lambda r: r["timestamp"])
    findings = []

    if receipt_time is not None:
        row = min(rows, key=lambda r: abs((r["timestamp"] - receipt_time).total_seconds()))
        gap_min = abs((row["timestamp"] - receipt_time).total_seconds()) / 60
        distance = haversine_km((row["lat"], row["lon"]), facility_latlon)
        severity: Severity = ("critical" if distance > DISTANCE_CRITICAL_KM
                              else "major" if distance > DISTANCE_MAJOR_KM else "ok")
        findings.append(_finding(
            "timeline", severity,
            f"At delivery receipt time {receipt_time:%Y-%m-%d %H:%M} the truck was "
            f"{distance:.1f} km from the facility (GPS fix {row['timestamp']:%H:%M})",
            {"receipt_time": receipt_time.isoformat(), "gps_timestamp": row["timestamp"].isoformat(),
             "truck_latlon": [row["lat"], row["lon"]], "facility_latlon": list(facility_latlon),
             "distance_km": round(distance, 2), "gps_gap_min": round(gap_min, 1)},
            CONF_DIRECT if gap_min <= GPS_MAX_GAP_MIN else CONF_INFERRED))

    if weighbridge_time is not None:
        arrival = next((r for r in rows
                        if haversine_km((r["lat"], r["lon"]), facility_latlon) <= ARRIVAL_RADIUS_KM), None)
        if arrival is None:
            findings.append(_finding(
                "timeline", "major",
                f"Weighbridge ticket at {weighbridge_time:%H:%M} but the GPS log never reaches the facility",
                {"weighbridge_time": weighbridge_time.isoformat(), "gps_arrival": None}))
        else:
            early_min = (arrival["timestamp"] - weighbridge_time).total_seconds() / 60
            severity = ("ok" if early_min <= 0
                        else "minor" if early_min < WEIGHBRIDGE_EARLY_MAJOR_MIN else "major")
            message = (f"Weighbridge ticket at {weighbridge_time:%H:%M}, "
                       f"{early_min:.0f} min before GPS arrival at {arrival['timestamp']:%H:%M}"
                       if early_min > 0 else
                       f"Weighbridge ticket at {weighbridge_time:%H:%M} is after GPS arrival at "
                       f"{arrival['timestamp']:%H:%M}")
            findings.append(_finding(
                "timeline", severity, message,
                {"weighbridge_time": weighbridge_time.isoformat(),
                 "gps_arrival": arrival["timestamp"].isoformat(),
                 "minutes_before_arrival": round(max(early_min, 0), 1)}))
    return findings


def average_hash(photo_path: str | Path) -> str:
    """64-bit average hash: 8×8 greyscale, bit = pixel above mean."""
    with Image.open(photo_path) as img:
        pixels = list(img.convert("L").resize((8, 8), Image.Resampling.LANCZOS).tobytes())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if p > mean else "0" for p in pixels)
    return f"{int(bits, 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def check_image(photo_path: str | Path, all_photo_hashes: dict[str, str]) -> list[Finding]:
    """all_photo_hashes: {other_claim_id: average_hash} for photos of OTHER claims."""
    try:
        exif = piexif.load(str(photo_path))
    except Exception:
        exif = {}
    has_date = piexif.ExifIFD.DateTimeOriginal in exif.get("Exif", {})
    has_gps = piexif.GPSIFD.GPSLatitude in exif.get("GPS", {})

    findings = []
    if has_date and has_gps:
        findings.append(_finding("image", "ok", "Facility photo carries EXIF capture date and GPS",
                                 {"exif_date": True, "exif_gps": True}))
    else:
        missing = [k for k, present in (("date", has_date), ("GPS", has_gps)) if not present]
        findings.append(_finding("image", "minor",
                                 f"Facility photo has no EXIF {' or '.join(missing)} metadata",
                                 {"exif_date": has_date, "exif_gps": has_gps}, CONF_INFERRED))

    photo_hash = average_hash(photo_path)
    for other_id, other_hash in sorted(all_photo_hashes.items()):
        distance = hamming(photo_hash, other_hash)
        if distance <= IMAGE_HASH_MAX_DISTANCE:
            findings.append(_finding("image", "major",
                                     f"Facility photo matches the photo submitted for claim "
                                     f"'{other_id}' (hash distance {distance})",
                                     {"photo_hash": photo_hash, "other_claim": other_id,
                                      "other_hash": other_hash, "hamming_distance": distance}))
    return findings


def check_manipulation(extractions: dict[str, Extraction]) -> list[Finding]:
    findings = [
        _finding("manipulation", "major",
                 f"Possible document manipulation: {DOC_LABELS.get(doc, doc)} contains text "
                 f"addressed to software/AI reviewers",
                 {"document": doc, "excerpt": ex.suspicious_excerpt})
        for doc, ex in extractions.items() if ex.suspicious_content
    ]
    return findings or [_finding("manipulation", "ok",
                                 "No instructions aimed at software/AI found in any document",
                                 {"documents_checked": sorted(extractions)})]


def check_pattern(findings: list[Finding]) -> list[Finding]:
    """Counts independent evidence categories, not checks: e.g. a quantity
    finding and a manipulation finding both come from the documents."""
    non_ok = [f for f in findings if f.severity != "ok" and f.check in EVIDENCE_SOURCE]
    sources = sorted({EVIDENCE_SOURCE[f.check] for f in non_ok})
    if len(non_ok) >= PATTERN_MIN_FINDINGS and len(sources) >= PATTERN_MIN_SOURCES:
        return [_finding("pattern", "major",
                         f"Corroborated multi-source inconsistency: {len(non_ok)} non-ok findings "
                         f"across {len(sources)} independent evidence sources ({', '.join(sources)})",
                         {"finding_ids": [f.id for f in non_ok], "sources": sources})]
    return []


def assign_ids(findings: list[Finding]) -> list[Finding]:
    """Give unnumbered findings the next free IDs F1, F2, … (in place)."""
    used = [int(f.id[1:]) for f in findings if f.id]
    next_id = max(used, default=0) + 1
    for f in findings:
        if not f.id:
            f.id = f"F{next_id}"
            next_id += 1
    return findings


# --- Case data --------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CASES_DIR = DATA_DIR / "cases"
GROUND_TRUTH_DIR = DATA_DIR / "cache" / "ground_truth"


def load_registry() -> dict[str, dict]:
    records = json.loads((DATA_DIR / "recyclers.json").read_text(encoding="utf-8"))
    return {r["reg_no"]: r for r in records}


def list_case_ids() -> list[str]:
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in CASES_DIR.glob("*/case.json")]
    return [c["case_id"] for c in sorted(cases, key=lambda c: c["number"])]


def load_case(case_id: str) -> dict:
    return json.loads((CASES_DIR / case_id / "case.json").read_text(encoding="utf-8"))


def load_gps(case_id: str) -> list[dict]:
    path = CASES_DIR / case_id / "gps.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [{"timestamp": datetime.fromisoformat(r["timestamp"]),
                 "lat": float(r["lat"]), "lon": float(r["lon"])} for r in csv.DictReader(fh)]


def load_ground_truth(case_id: str) -> dict[str, Extraction]:
    raw = json.loads((GROUND_TRUTH_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    return {doc: Extraction.model_validate(data) for doc, data in raw.items()}


def photo_hashes() -> dict[str, str]:
    return {p.parent.name: average_hash(p) for p in sorted(CASES_DIR.glob("*/photo.jpg"))}


def event_time(extraction: Extraction | None) -> datetime | None:
    if extraction is None or not extraction.date or not extraction.time:
        return None
    return datetime.fromisoformat(f"{extraction.date}T{extraction.time}")


def run_checks(case: dict, extractions: dict[str, Extraction], registry: dict[str, dict],
               gps_rows: list[dict], photo_path: Path | None,
               other_photo_hashes: dict[str, str]) -> list[Finding]:
    """Run every deterministic check for one claim and return numbered findings."""
    cert = extractions.get("epr_certificate")
    qty = {doc: (ex.quantity_kg if ex else None)
           for doc, ex in ((d, extractions.get(d)) for d in DOC_NAMES)}
    reg_no = cert.recycler_reg_no if cert else case.get("recycler_reg_no")

    findings: list[Finding] = []
    findings += check_registry(reg_no, cert.recycler_name if cert else None, registry)
    findings += compare_quantities(qty["epr_certificate"], qty["invoice"],
                                   qty["weighbridge"], qty["transporter"])
    findings += check_capacity(reg_no, qty["epr_certificate"], case["claim"]["period_days"], registry)
    if reg_no in registry:
        facility = (registry[reg_no]["lat"], registry[reg_no]["lon"])
        findings += check_timeline(gps_rows, facility, event_time(extractions.get("transporter")),
                                   event_time(extractions.get("weighbridge")))
    if photo_path is not None and photo_path.exists():
        findings += check_image(photo_path, other_photo_hashes)
    findings += check_manipulation(extractions)
    assign_ids(findings)
    findings += check_pattern(findings)
    return assign_ids(findings)


def run_case(case_id: str, extractions: dict[str, Extraction] | None = None) -> list[Finding]:
    """Run all checks on a demo case; defaults to the ground-truth extractions (no LLM)."""
    hashes = photo_hashes()
    hashes.pop(case_id, None)
    return run_checks(load_case(case_id),
                      extractions if extractions is not None else load_ground_truth(case_id),
                      load_registry(), load_gps(case_id), CASES_DIR / case_id / "photo.jpg", hashes)
