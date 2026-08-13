#!/usr/bin/env python3
"""Audit the corrected, pre-formal M1 FinMultiTime input bundle.

This audit is deliberately source-local.  It checks the frozen contract,
corrected preprocessed cases, final packet serialization, Qwen provenance,
point-in-time boundaries, routing, and future-field exclusion.  It never
fetches data, calls a model, or runs Formal M1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data/processed/finmultitime_3stocks_2024h1_v1"
CONTRACT_PATH = REPO_ROOT / "docs/m1/m1_evidence_contract.json"
AUDIT_PATH = REPO_ROOT / "docs/m1/m1_text_source_integrity_audit.json"
FUTURE_KEY_RE = re.compile(
    r"(?:^|[_\s-])(target|future|next|prediction|forecast|outcome|label)(?:$|[_\s-])",
    re.IGNORECASE,
)
FUTURE_LINE_RE = re.compile(
    r"^\s*(?:target|future_return|next_return|prediction|forecast(?: label)?|"
    r"ground[- ]truth|direction label|outcome)\s*[:=]",
    re.IGNORECASE,
)
ROUTING = {
    "TEXT": "News Analyst",
    "TABLE": "Fundamentals Analyst",
    "TIME_SERIES": "Market Analyst",
    "IMAGE": "Market Analyst",
}
SECTIONS = tuple(ROUTING)
EXPECTED_CAPTIONS = {
    "AMZN": "131c2904cd82c850f94b00cc58e2b97b2ef69c7e805fa562ef2c8f2e00a568fe",
    "JPM": "44e980fa7768c0cf36961d6c42ea769157e4b989d02b60b37618256028b04d70",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def future_hits(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if FUTURE_KEY_RE.search(str(key)):
                hits.append(child_path)
            hits.extend(future_hits(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(future_hits(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        for line in value.splitlines():
            if FUTURE_LINE_RE.search(line):
                hits.append(path)
    return hits


def parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} is not an ISO date: {value!r}")
    return date.fromisoformat(value[:10])


def check(label: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "check": label,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def audit(root: Path) -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    audit_artifact = read_json(AUDIT_PATH)
    manifest = read_json(root / "manifest.json")
    packet_manifest = read_json(root / "manifests/evidence_packet_manifest.json")
    qwen_input = read_json(root / "qwen/qwen_input_manifest.json")
    qwen_captions = read_json(root / "qwen/caption_manifest.json")
    case_paths = sorted((root / "cases").glob("*/*.json"))
    packet_paths = sorted((root / "evidence_packets").glob("*/*.json"))
    text_paths = sorted((root / "evidence_packets").glob("*/*.txt"))
    cases = [read_json(path) for path in case_paths]
    packets = {item["case_id"]: item for item in (read_json(path) for path in packet_paths)}
    packet_entries = {item["case_id"]: item for item in packet_manifest["packets"]}

    modality_counts = {
        modality: Counter(case[modality]["status"] for case in cases)
        for modality in SECTIONS
    }
    text_available_by_symbol = Counter(
        case["symbol"]
        for case in cases
        if case["TEXT"]["status"] == "AVAILABLE"
    )
    text_record_count = sum(len(case["TEXT"].get("selected_records", [])) for case in cases)
    pit_violations: list[dict[str, Any]] = []
    future_fields: list[dict[str, Any]] = []
    routing_violations: list[dict[str, Any]] = []
    packet_violations: list[dict[str, Any]] = []
    image_hashes: dict[str, str] = {}
    caption_hashes: dict[str, str] = {}

    for case, case_path in zip(cases, case_paths, strict=True):
        case_id = f"{case['symbol']}:{case['decision_session']}"
        decision = parse_date(case["decision_session"], "decision_session")
        text = case["TEXT"]
        for record in text.get("selected_records", []):
            article_date = parse_date(record["date"], f"{case_id} TEXT date")
            if article_date >= decision:
                pit_violations.append({"case": case_id, "modality": "TEXT", "date": record["date"]})
        if text["status"] == "UNAVAILABLE" and text.get("selected_records"):
            packet_violations.append({"case": case_id, "issue": "unavailable TEXT has records"})

        for concept, fact in case["TABLE"].get("facts", {}).items():
            if fact is None:
                continue
            filed = parse_date(fact["filed_date"], f"{case_id} TABLE {concept}")
            if filed >= decision:
                pit_violations.append({"case": case_id, "modality": "TABLE", "concept": concept})

        ts = case["TIME_SERIES"]
        selected_dates = [parse_date(value, f"{case_id} TIME_SERIES") for value in ts["selected_session_dates"]]
        if selected_dates and max(selected_dates) > decision:
            pit_violations.append({"case": case_id, "modality": "TIME_SERIES", "latest": max(selected_dates).isoformat()})
        if ts["status"] == "AVAILABLE" and len(selected_dates) < 61:
            packet_violations.append({"case": case_id, "issue": "TIME_SERIES has fewer than 61 rows"})

        image = case["IMAGE"]
        if image["status"] == "AVAILABLE":
            period_end = parse_date(image["eligible_image"]["period_end"], f"{case_id} IMAGE period_end")
            if period_end >= decision:
                pit_violations.append({"case": case_id, "modality": "IMAGE", "period_end": period_end.isoformat()})
            image_hashes[case["symbol"]] = image["eligible_image"]["sha256"]
            caption_hashes[case["symbol"]] = image["caption_sha256"]
            if image["caption_status"] != "GENERATED":
                packet_violations.append({"case": case_id, "issue": "available IMAGE lacks GENERATED caption"})
        elif image["caption_status"] != "NOT_APPLICABLE":
            packet_violations.append({"case": case_id, "issue": "unavailable IMAGE caption status is not NOT_APPLICABLE"})

        hits = future_hits(case)
        if hits:
            future_fields.append({"case": case_id, "paths": hits})
        packet = packets.get(case_id)
        entry = packet_entries.get(case_id)
        if packet is None or entry is None:
            packet_violations.append({"case": case_id, "issue": "missing final packet or manifest entry"})
            continue
        if packet["packet_status"] != "FINAL_FROZEN":
            packet_violations.append({"case": case_id, "issue": "packet status is not FINAL"})
        if set(packet) & {"target", "future_return", "prediction", "forecast", "outcome"}:
            packet_violations.append({"case": case_id, "issue": "future field in packet topology"})
        if future_hits(packet) or any(FUTURE_LINE_RE.search(line) for line in packet["agent_text"].splitlines()):
            future_fields.append({"case": case_id, "paths": ["packet"]})
        if packet["routing"]["modality_to_analyst"] != ROUTING:
            routing_violations.append({"case": case_id, "routing": packet["routing"]})
        if packet["routing"]["raw_packet_direct_injection"] is not False:
            routing_violations.append({"case": case_id, "issue": "raw packet direct injection enabled"})
        if packet["character_counts"]["complete"] > 22000:
            packet_violations.append({"case": case_id, "issue": "packet exceeds 22000 characters"})

    checks = [
        check(
            "contract_and_erratum_identity",
            contract["packet_version"] == "M1-FINMULTITIME-v1.0.2"
            and contract["erratum"]["formal_m1_run_before_erratum"] is False
            and sha256_file(AUDIT_PATH) == contract["text_source_integrity"]["audit_sha256"],
            {
                "contract_version": contract["packet_version"],
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "audit_sha256": sha256_file(AUDIT_PATH),
            },
        ),
        check(
            "TEXT_source_integrity_and_fail_closed_policy",
            audit_artifact["record_status_counts"] == {"VERIFIED_MATCH": 8, "VERIFIED_MISMATCH": 8}
            and text_available_by_symbol == Counter({"AMZN": 2})
            and text_record_count == 16,
            {
                "audit_record_status_counts": audit_artifact["record_status_counts"],
                "case_text_available_by_symbol": dict(text_available_by_symbol),
                "case_selected_text_record_count": text_record_count,
                "no_external_replacement": True,
            },
        ),
        check(
            "TABLE_PIT_and_source_provenance",
            modality_counts["TABLE"] == Counter({"AVAILABLE": 78})
            and not any(item["modality"] == "TABLE" for item in pit_violations),
            {"status_counts": dict(modality_counts["TABLE"]), "violations": [item for item in pit_violations if item.get("modality") == "TABLE"]},
        ),
        check(
            "TIME_SERIES_PIT_history_and_summary_contract",
            modality_counts["TIME_SERIES"] == Counter({"AVAILABLE": 78})
            and not any(item["modality"] == "TIME_SERIES" for item in pit_violations),
            {"status_counts": dict(modality_counts["TIME_SERIES"]), "required_rows": 61},
        ),
        check(
            "IMAGE_bytes_and_Qwen_caption_provenance",
            modality_counts["IMAGE"] == Counter({"AVAILABLE": 52, "UNAVAILABLE": 26})
            and all(caption_hashes.get(symbol) == expected for symbol, expected in EXPECTED_CAPTIONS.items()),
            {"status_counts": dict(modality_counts["IMAGE"]), "image_hashes": image_hashes, "caption_hashes": caption_hashes},
        ),
        check(
            "PIT_and_future_leakage_global",
            not pit_violations and not future_fields,
            {"pit_violation_count": len(pit_violations), "future_field_violation_count": len(future_fields)},
        ),
        check(
            "routing_and_packet_serialization",
            len(cases) == 78 and len(packet_paths) == 78 and len(text_paths) == 78 and not routing_violations and not packet_violations,
            {"case_count": len(cases), "json_packet_count": len(packet_paths), "text_packet_count": len(text_paths), "routing_violations": routing_violations, "packet_violations": packet_violations},
        ),
        check(
            "deterministic_build_and_manifests",
            manifest["deterministic_build"]["status"] == "PASS"
            and manifest["deterministic_build"]["json_hash_mismatches"] == 0
            and manifest["deterministic_build"]["text_hash_mismatches"] == 0
            and packet_manifest["packet_count"] == 78
            and qwen_input["caption_status_counts"] == {"GENERATED": 52, "NOT_APPLICABLE": 26, "PENDING": 0}
            and qwen_captions["formal_case_reference_counts"]["GENERATED"] == 52,
            {"deterministic_build": manifest["deterministic_build"], "packet_count": packet_manifest["packet_count"], "qwen_input_caption_status_counts": qwen_input["caption_status_counts"]},
        ),
        check(
            "formal_m1_execution_boundary",
            contract["erratum"]["formal_m1_run_before_erratum"] is False
            and contract["boundary_confirmation"]["formal_m1_run"] is False,
            {"formal_m1_run": False, "result": "NOT_APPLICABLE — pre-formal audit only"},
        ),
    ]
    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "BLOCKED"
    return {
        "report_id": "M1-Pre-Formal-Correctness-Audit",
        "contract_version": contract["packet_version"],
        "dataset_id": manifest["dataset_id"],
        "processed_root": str(root),
        "overall_status": overall,
        "checks": checks,
        "modality_status_counts": {key: dict(value) for key, value in modality_counts.items()},
        "source_integrity_audit_sha256": sha256_file(AUDIT_PATH),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "formal_m1_run": False,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M1 Pre-Formal Correctness Audit",
        "",
        f"**Overall status:** `{report['overall_status']}`",
        f"**Contract:** `{report['contract_version']}`",
        "**Scope:** corrected M1 preprocessing, frozen Qwen-caption integration, packet serialization, and protocol/PIT checks before Formal M1.",
        "",
        "Formal M1 was not run. A `NOT_APPLICABLE` execution-boundary check records that this is intentionally a pre-formal audit.",
        "",
        "| Check | Status | Evidence summary |",
        "|---|---|---|",
    ]
    for item in report["checks"]:
        evidence = item["evidence"]
        summary = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        lines.append(f"| {item['check']} | {item['status']} | `{summary}` |")
    lines.extend([
        "",
        "## Modality counts",
        "",
        "| Modality | Status counts |",
        "|---|---|",
    ])
    for modality, counts in report["modality_status_counts"].items():
        lines.append(f"| {modality} | `{json.dumps(counts, sort_keys=True)}` |")
    lines.extend([
        "",
        f"TEXT source-integrity audit SHA-256: `{report['source_integrity_audit_sha256']}`.",
        f"Contract SHA-256: `{report['contract_sha256']}`.",
        "",
        "No raw FinMultiTime file was modified, no external article was substituted, no future label or outcome was admitted, and no formal M1 result is represented by this audit.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    report = audit(args.processed_root.resolve())
    (REPO_ROOT / "docs/m1/m1_preformal_correctness_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (REPO_ROOT / "docs/m1/M1_PRE_FORMAL_CORRECTNESS_AUDIT.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps({"overall_status": report["overall_status"], "checks": len(report["checks"])}, sort_keys=True))
    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
