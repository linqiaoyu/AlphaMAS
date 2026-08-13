#!/usr/bin/env python3
"""Build the isolated, out-of-window M1 AAPL pilot input bundle.

The pilot deliberately reuses the frozen M1 v1.0.2 raw loader, modality
selection functions, packet formatter, and future-field checks.  It changes
only the schedule, symbol allow-list, and bundle metadata.  It never writes
to the raw FinMultiTime release or to the formal processed bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.audit_finmultitime import (  # noqa: E402
    NEWS_ARCHIVE,
    NEWS_MEMBERS,
    TABLE_ARCHIVE,
    TABLE_MEMBER_RE,
    TS_ARCHIVE,
    TS_MEMBERS,
)
from scripts.finmultitime.build_m1_evidence_packets import (  # noqa: E402
    REQUIRED_SECTIONS,
    assert_no_future_fields,
    build_packet,
    canonical_write_bytes,
    sha256_bytes,
    sha256_file,
)
from scripts.finmultitime.design_m1_contract import (  # noqa: E402
    CONTRACT_VERSION,
    load_raw_data,
)
from scripts.finmultitime.preprocess_m1_inputs import (  # noqa: E402
    DEFAULT_CONTRACT_DIR,
    DEFAULT_RAW_ROOT,
    build_case_records,
    enforce_frozen_contract,
    json_safe,
    source_member_hash,
)

PILOT_DATASET_ID = "finmultitime_m1_pilot_aapl_2023q4_4w_v1"
PILOT_SYMBOL = "AAPL"
PILOT_SESSIONS = (
    "2023-10-06",
    "2023-10-13",
    "2023-10-20",
    "2023-10-27",
)
PILOT_CASE_COUNT = len(PILOT_SESSIONS)
PILOT_STATUS = "PILOT_FROZEN"
FROZEN_M0_BASE_SHA = "2535896c8b1070b19c06fa6a936663babb4356f7"
FORMAL_PROCESSED_ROOT = REPO_ROOT / "data/processed/finmultitime_3stocks_2024h1_v1"
NY_TZ = ZoneInfo("America/New_York")


class PilotInputError(ValueError):
    """Raised when the pilot cannot be proven to use frozen identities."""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity_hash(paths: dict[str, str]) -> str:
    payload = json.dumps(paths, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inventory(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def _next_weekday(value: date) -> date:
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _pilot_context() -> dict[str, Any]:
    events: list[dict[str, str]] = []
    decisions = [date.fromisoformat(value) for value in PILOT_SESSIONS]
    for decision in decisions:
        close_ny = datetime.combine(decision, time(16), tzinfo=NY_TZ)
        events.append({
            "symbol": PILOT_SYMBOL,
            "decision_session": decision.isoformat(),
            "decision_close_utc": close_ny.astimezone(timezone.utc).isoformat(),
            "decision_close_ny": close_ny.isoformat(),
            "execution_session": _next_weekday(decision).isoformat(),
        })
    return {
        "events": events,
        "decisions": decisions,
        "warmup": set(),
        "warmup_start": decisions[0],
        "warmup_end": decisions[0],
        "formal_start": decisions[0],
        "formal_end": decisions[-1],
        "formal_calendar_start": date(2023, 10, 1),
        "formal_calendar_end": date(2023, 10, 31),
        "final_valuation": decisions[-1],
    }


def _require_raw_root(raw_root: Path) -> None:
    required = (
        raw_root / NEWS_ARCHIVE,
        raw_root / TABLE_ARCHIVE,
        raw_root / TS_ARCHIVE,
        raw_root / "image/image",
    )
    if not raw_root.is_dir() or any(not path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise PilotInputError(
            "RAW DATASET PRE-FLIGHT BLOCKED — required FinMultiTime mount/source "
            f"is missing: {missing}"
        )


def _aapl_table_members(raw_root: Path) -> list[str]:
    import zipfile

    with zipfile.ZipFile(raw_root / TABLE_ARCHIVE) as archive:
        return sorted(
            name for name in archive.namelist() if TABLE_MEMBER_RE[PILOT_SYMBOL].match(name)
        )


def _frozen_aapl_source_identities(raw_root: Path) -> dict[str, Any]:
    """Compare AAPL raw identities against the frozen formal source manifest."""
    source_path = FORMAL_PROCESSED_ROOT / "manifests/source_manifest.json"
    if not source_path.is_file():
        raise PilotInputError(f"formal source provenance is missing: {source_path}")
    formal_records = [
        record for record in read_json(source_path).get("records", [])
        if record.get("symbol") == PILOT_SYMBOL
    ]
    expected_text = next(
        (record.get("member") for record in formal_records if record.get("modality") == "TEXT"),
        None,
    )
    expected_ts = next(
        (record.get("member") for record in formal_records if record.get("modality") == "TIME_SERIES"),
        None,
    )
    expected_table_record = next(
        (record for record in formal_records if record.get("modality") == "TABLE"),
        None,
    )
    expected_table = {
        item["member"]: item for item in (expected_table_record or {}).get("members", [])
    }
    actual_text = source_member_hash(raw_root, NEWS_ARCHIVE, NEWS_MEMBERS[PILOT_SYMBOL])
    actual_ts = source_member_hash(raw_root, TS_ARCHIVE, TS_MEMBERS[PILOT_SYMBOL])
    actual_table = {
        member: source_member_hash(raw_root, TABLE_ARCHIVE, member)
        for member in _aapl_table_members(raw_root)
    }
    if expected_text != actual_text or expected_ts != actual_ts or expected_table != actual_table:
        raise PilotInputError(
            "RAW SOURCE IDENTITY REVIEW REQUIRED — relevant AAPL raw source identity "
            "differs from the frozen formal provenance"
        )
    return {
        "formal_source_manifest": "data/processed/finmultitime_3stocks_2024h1_v1/manifests/source_manifest.json",
        "formal_source_manifest_sha256": sha256_file(source_path),
        "AAPL": {
            "TEXT": actual_text,
            "TABLE": [actual_table[key] for key in sorted(actual_table)],
            "TIME_SERIES": actual_ts,
            "IMAGE": {
                "formal_expected_coverage": "UNAVAILABLE",
                "raw_file_count": None,
                "policy": "AAPL remains UNAVAILABLE under frozen image audit",
            },
        },
    }


def _preflight_and_load(raw_root: Path, contract_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_raw_root(raw_root)
    contract = enforce_frozen_contract(contract_dir)
    aapl_policy = contract["text_integrity_policy"].get(PILOT_SYMBOL)
    if not aapl_policy or aapl_policy.get("formal_text_policy") != "UNAVAILABLE":
        raise PilotInputError("frozen AAPL TEXT source-integrity policy is not UNAVAILABLE")
    source_provenance = _frozen_aapl_source_identities(raw_root)
    raw_data = load_raw_data(raw_root)
    aapl_images = raw_data["images"].get(PILOT_SYMBOL, [])
    if aapl_images:
        raise PilotInputError(
            "RAW SOURCE IDENTITY REVIEW REQUIRED — raw data unexpectedly contains "
            "an AAPL image contrary to the frozen image audit"
        )
    source_provenance["AAPL"]["IMAGE"]["raw_file_count"] = len(aapl_images)
    return contract, raw_data, source_provenance


def _pilot_cases(raw_data: dict[str, Any], context: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, Any]]:
    records, _simulation, _ts_selections = build_case_records(raw_data, context, contract)
    cases = [json_safe(record) for record in records if record["symbol"] == PILOT_SYMBOL]
    cases.sort(key=lambda item: item["decision_session"])
    if [case["decision_session"] for case in cases] != list(PILOT_SESSIONS):
        raise PilotInputError("pilot case schedule is not exactly the frozen four-session schedule")
    if len(cases) != PILOT_CASE_COUNT or {case["symbol"] for case in cases} != {PILOT_SYMBOL}:
        raise PilotInputError("pilot case structure is not exactly AAPL-only with four cases")
    for case in cases:
        case["case_id"] = f"{case['symbol']}:{case['decision_session']}"
        case["_path"] = f"evidence_packets/{PILOT_SYMBOL}/{case['decision_session']}.json"
    return cases


def _pilot_packet(case: dict[str, Any], contract: dict[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    packet, text = build_packet(
        case,
        REPO_ROOT,
        {"evidence_contract_sha256": contract["contract_sha256"]},
        {},
    )
    packet["dataset_id"] = PILOT_DATASET_ID
    packet["packet_status"] = PILOT_STATUS
    packet["pilot_only"] = True
    packet["formal_eligible"] = False
    packet["source_provenance"]["pilot_only"] = True
    packet["source_provenance"]["formal_eligible"] = False
    assert_no_future_fields(packet)
    if packet["TEXT"]["status"] != "UNAVAILABLE":
        raise PilotInputError(f"AAPL TEXT unexpectedly available: {case['decision_session']}")
    if packet["IMAGE"]["status"] != "UNAVAILABLE" or packet["IMAGE"]["caption_status"] != "NOT_APPLICABLE":
        raise PilotInputError(f"AAPL IMAGE policy drifted: {case['decision_session']}")
    decision = date.fromisoformat(case["decision_session"])
    table_violations = [
        concept for concept, fact in case["TABLE"]["facts"].items()
        if fact is not None and fact.get("filed_date") >= decision.isoformat()
    ]
    ts_dates = case["TIME_SERIES"].get("selected_session_dates", [])
    ts_violations = [value for value in ts_dates if value > decision.isoformat()]
    if table_violations or len(ts_dates) != 61 or ts_violations:
        raise PilotInputError(
            f"pilot PIT validation failed for {case['symbol']}:{case['decision_session']}"
        )
    validation = {
        "case_id": case["case_id"],
        "TEXT": packet["TEXT"]["status"],
        "TABLE": packet["TABLE"]["status"],
        "TIME_SERIES": packet["TIME_SERIES"]["status"],
        "IMAGE": packet["IMAGE"]["status"],
        "selected_time_series_rows": len(ts_dates),
        "table_pit_violations": len(table_violations),
        "time_series_pit_violations": len(ts_violations),
        "future_field_violations": 0,
    }
    return packet, text.encode("utf-8"), validation


def _write_candidate(
    destination: Path,
    raw_data: dict[str, Any],
    contract: dict[str, Any],
    source_provenance: dict[str, Any],
    source_commit_sha: str,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    cases = _pilot_cases(raw_data, _pilot_context(), contract)
    entries: list[dict[str, Any]] = []
    packet_validations: list[dict[str, Any]] = []
    for case in cases:
        packet, text_bytes, validation = _pilot_packet(case, contract)
        json_bytes = canonical_write_bytes(packet)
        json_path = destination / "evidence_packets" / PILOT_SYMBOL / f"{case['decision_session']}.json"
        text_path = destination / "evidence_packets" / PILOT_SYMBOL / f"{case['decision_session']}.txt"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_bytes(json_bytes)
        text_path.write_bytes(text_bytes)
        packet_validations.append(validation)
        entries.append({
            "case_id": case["case_id"],
            "symbol": PILOT_SYMBOL,
            "decision_session": case["decision_session"],
            "json_path": json_path.relative_to(destination).as_posix(),
            "json_sha256": sha256_bytes(json_bytes),
            "text_path": text_path.relative_to(destination).as_posix(),
            "text_sha256": sha256_bytes(text_bytes),
            "packet_status": PILOT_STATUS,
            "pilot_only": True,
            "formal_eligible": False,
            "modality_statuses": {name: packet[name]["status"] for name in REQUIRED_SECTIONS},
            "route_sha256": {
                name: sha256_bytes(packet["routed_projections"][name]["text"].encode("utf-8"))
                for name in ("news_analyst", "fundamentals_analyst", "market_analyst", "social_analyst")
            },
        })

    packet_manifest = {
        "schema_version": "1.0",
        "record_type": "m1_finmultitime_pilot_packet_manifest",
        "contract_version": CONTRACT_VERSION,
        "dataset_id": PILOT_DATASET_ID,
        "bundle_scope": "PILOT",
        "pilot_only": True,
        "formal_eligible": False,
        "packet_status": PILOT_STATUS,
        "packet_count": PILOT_CASE_COUNT,
        "symbols": [PILOT_SYMBOL],
        "pilot_sessions": list(PILOT_SESSIONS),
        "packets": sorted(entries, key=lambda item: item["case_id"]),
    }
    packet_manifest_path = destination / "manifests/evidence_packet_manifest.json"
    write_json(packet_manifest_path, packet_manifest)
    source_provenance = {
        **source_provenance,
        "record_type": "m1_pilot_source_provenance",
        "bundle_scope": "PILOT",
        "pilot_only": True,
        "formal_eligible": False,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "text_policy": {
            "symbol": PILOT_SYMBOL,
            "status": contract["text_integrity_policy"][PILOT_SYMBOL]["formal_text_policy"],
            "reason": contract["text_integrity_policy"][PILOT_SYMBOL]["reason"],
            "audit_status": contract["text_integrity_policy"][PILOT_SYMBOL]["audit_status"],
        },
        "image_policy": {
            "symbol": PILOT_SYMBOL,
            "status": "UNAVAILABLE",
            "caption_status": "NOT_APPLICABLE",
            "qwen_calls": 0,
        },
        "raw_finmultitime_read_only": True,
    }
    source_path = destination / "manifests/source_provenance.json"
    write_json(source_path, source_provenance)
    validation = {
        "record_type": "m1_pilot_input_validation",
        "bundle_scope": "PILOT",
        "pilot_only": True,
        "formal_eligible": False,
        "case_count": PILOT_CASE_COUNT,
        "packet_count": len(entries),
        "cases": packet_validations,
        "PIT_violations": sum(
            item["table_pit_violations"] + item["time_series_pit_violations"]
            for item in packet_validations
        ),
        "future_field_violations": sum(
            item["future_field_violations"] for item in packet_validations
        ),
        "qwen_calls": 0,
        "deepseek_calls": 0,
        "paid_api_calls": 0,
        "agent_runs": 0,
    }
    validation_path = destination / "manifests/pilot_validation.json"
    write_json(validation_path, validation)
    (destination / "README.md").write_text(
        "# Isolated M1 AAPL out-of-window pilot input bundle\n\n"
        "**PILOT ONLY. NOT FORMAL M1 INPUT. NOT PART OF THE 78-CASE TEST SET. "
        "NOT ELIGIBLE FOR PERFORMANCE ANALYSIS.**\n\n"
        f"Bundle: `{PILOT_DATASET_ID}`\n\n"
        "This four-case bundle exists solely for pre-formal runtime correctness validation. "
        "It uses the frozen `M1-FINMULTITIME-v1.0.2` selection rules and the established "
        "out-of-window AAPL dates. It contains no raw dataset, no Qwen caption, no Agent run, "
        "and no performance outcome. The formal `experiments/M1/inputs/` archive is separate "
        "and must remain byte-identical.\n\n"
        "TEXT is UNAVAILABLE under the frozen AAPL source-integrity policy. IMAGE is "
        "UNAVAILABLE/NOT_APPLICABLE under the frozen AAPL image audit. TABLE and TIME_SERIES "
        "retain their point-in-time-safe source provenance.\n",
        encoding="utf-8",
    )
    processed_files = _inventory(
        destination,
        exclude={"manifest.json", "manifests/processed_sha256.json", "manifests/input_bundle_checksums.json"},
    )
    processed_path = destination / "manifests/processed_sha256.json"
    write_json(processed_path, {
        "schema_version": "1.0",
        "hash_algorithm": "SHA-256",
        "hash_scope": "all pilot research files except manifest and checksum inventories",
        "files": processed_files,
    })
    packet_manifest_sha = sha256_file(packet_manifest_path)
    source_sha = sha256_file(source_path)
    validation_sha = sha256_file(validation_path)
    processed_sha = sha256_file(processed_path)
    builder_path = Path(__file__).resolve()
    selection_path = REPO_ROOT / "scripts/finmultitime/design_m1_contract.py"
    bundle_manifest = {
        "schema_version": "1.0",
        "record_type": "frozen_m1_pilot_input_bundle_manifest",
        "bundle_scope": "PILOT",
        "pilot_only": True,
        "formal_eligible": False,
        "dataset": {
            "dataset_id": PILOT_DATASET_ID,
            "symbols": [PILOT_SYMBOL],
            "pilot_sessions": list(PILOT_SESSIONS),
            "case_count": PILOT_CASE_COUNT,
        },
        "contract": {
            "version": CONTRACT_VERSION,
            "sha256": contract["contract_sha256"],
            "source": "docs/m1/m1_evidence_contract.json",
        },
        "frozen_source_code": {
            "source_commit_sha": source_commit_sha,
            "pilot_builder_path": "scripts/finmultitime/build_m1_pilot_inputs.py",
            "pilot_builder_sha256": sha256_file(builder_path),
            "frozen_selection_source_path": "scripts/finmultitime/design_m1_contract.py",
            "frozen_selection_source_sha256": sha256_file(selection_path),
            "frozen_m0_base_sha": FROZEN_M0_BASE_SHA,
        },
        "source_provenance": {
            "path": source_path.relative_to(destination).as_posix(),
            "sha256": source_sha,
        },
        "validation": {
            "path": validation_path.relative_to(destination).as_posix(),
            "sha256": validation_sha,
            "PIT_violations": validation["PIT_violations"],
            "future_field_violations": validation["future_field_violations"],
        },
        "processed_checksums": {
            "path": processed_path.relative_to(destination).as_posix(),
            "sha256": processed_sha,
            "file_count": len(processed_files),
        },
        "packets": {
            "count": PILOT_CASE_COUNT,
            "manifest_path": packet_manifest_path.relative_to(destination).as_posix(),
            "manifest_sha256": packet_manifest_sha,
            "deterministic_build": {
                "status": "PENDING_DOUBLE_BUILD",
                "packets_compared": PILOT_CASE_COUNT,
                "json_hash_mismatches": 0,
                "text_hash_mismatches": 0,
            },
        },
        "status": {
            "input_bundle_frozen": True,
            "formal_m1_run": False,
            "pilot_only": True,
            "formal_eligible": False,
        },
    }
    bundle_manifest_path = destination / "manifests/input_bundle_manifest.json"
    write_json(bundle_manifest_path, bundle_manifest)
    bundle_manifest_sha = sha256_file(bundle_manifest_path)
    top_manifest = {
        "schema_version": "1.0",
        "record_type": "frozen_m1_pilot_input_bundle",
        "dataset_id": PILOT_DATASET_ID,
        "bundle_scope": "PILOT",
        "pilot_only": True,
        "formal_eligible": False,
        "target_symbols": [PILOT_SYMBOL],
        "pilot_sessions": list(PILOT_SESSIONS),
        "final_evidence_packet_count": PILOT_CASE_COUNT,
        "formal_case_count": 0,
        "pilot_case_count": PILOT_CASE_COUNT,
        "frozen_evidence_contract_version": CONTRACT_VERSION,
        "evidence_contract_sha256": contract["contract_sha256"],
        "input_bundle_frozen": True,
        "input_bundle_manifest": {
            "path": bundle_manifest_path.relative_to(destination).as_posix(),
            "sha256": bundle_manifest_sha,
        },
        "final_evidence_packet_manifest": {
            "path": packet_manifest_path.relative_to(destination).as_posix(),
            "sha256": packet_manifest_sha,
        },
        "processed_checksums": {
            "path": processed_path.relative_to(destination).as_posix(),
            "sha256": processed_sha,
            "file_count": len(processed_files),
        },
        "source_provenance": {
            "path": source_path.relative_to(destination).as_posix(),
            "sha256": source_sha,
        },
        "input_bundle_checksums": {
            "path": "manifests/input_bundle_checksums.json",
            "hash_algorithm": "SHA-256",
        },
        "deterministic_build": {
            "status": "PENDING_DOUBLE_BUILD",
            "packets_compared": PILOT_CASE_COUNT,
            "json_hash_mismatches": 0,
            "text_hash_mismatches": 0,
        },
        "status": {
            "input_bundle_frozen": True,
            "formal_m1_run": False,
            "pilot_only": True,
            "formal_eligible": False,
        },
    }
    top_path = destination / "manifest.json"
    write_json(top_path, top_manifest)
    checksum_files = _inventory(destination, exclude={"manifests/input_bundle_checksums.json"})
    checksum_path = destination / "manifests/input_bundle_checksums.json"
    write_json(checksum_path, {
        "schema_version": "1.0",
        "hash_algorithm": "SHA-256",
        "hash_scope": "every pilot bundle file except this checksum inventory",
        "bundle_manifest_path": bundle_manifest_path.relative_to(destination).as_posix(),
        "bundle_manifest_sha256": bundle_manifest_sha,
        "files": checksum_files,
    })
    return {
        "packets": entries,
        "packet_manifest_sha256": packet_manifest_sha,
        "text_hashes": {item["case_id"]: item["text_sha256"] for item in entries},
        "json_hashes": {item["case_id"]: item["json_sha256"] for item in entries},
        "source_provenance_sha256": source_sha,
        "bundle_manifest_sha256": bundle_manifest_sha,
        "top_manifest_sha256": sha256_file(top_path),
        "checksum_sha256": sha256_file(checksum_path),
        "processed_sha256": processed_sha,
        "checksum_file_count": len(checksum_files),
    }


def build(
    *,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path,
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
    verify_determinism: bool = True,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    contract_dir = contract_dir.resolve()
    source_commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    contract, raw_data, source_provenance = _preflight_and_load(raw_root, contract_dir)
    temp_parent = Path(tempfile.mkdtemp(prefix=f".{PILOT_DATASET_ID}-", dir=output_dir.parent))
    try:
        first = temp_parent / "first"
        first_result = _write_candidate(first, raw_data, contract, source_provenance, source_commit_sha)
        second_result = None
        if verify_determinism:
            second = temp_parent / "second"
            second_result = _write_candidate(second, raw_data, contract, source_provenance, source_commit_sha)
        json_mismatches = 0
        text_mismatches = 0
        if second_result is not None:
            json_mismatches = sum(
                first_result["json_hashes"][case_id] != second_result["json_hashes"][case_id]
                for case_id in first_result["json_hashes"]
            )
            text_mismatches = sum(
                first_result["text_hashes"][case_id] != second_result["text_hashes"][case_id]
                for case_id in first_result["text_hashes"]
            )
            if json_mismatches or text_mismatches:
                raise PilotInputError("deterministic double build produced hash mismatches")
        deterministic = {
            "status": "PASS" if second_result is not None else "NOT_REQUESTED",
            "packets_compared": PILOT_CASE_COUNT,
            "json_hash_mismatches": json_mismatches,
            "text_hash_mismatches": text_mismatches,
        }
        # Bind the deterministic verdict into the two bundle manifests after
        # the candidate comparison.  This is the only post-build metadata edit.
        for relative in ("manifests/input_bundle_manifest.json", "manifest.json"):
            path = first / relative
            value = read_json(path)
            value["deterministic_build"] = deterministic
            if relative.endswith("input_bundle_manifest.json"):
                value["packets"]["deterministic_build"] = deterministic
            write_json(path, value)
        finalized_bundle_manifest_sha = sha256_file(
            first / "manifests/input_bundle_manifest.json"
        )
        top_manifest_path = first / "manifest.json"
        top_manifest = read_json(top_manifest_path)
        top_manifest["input_bundle_manifest"]["sha256"] = finalized_bundle_manifest_sha
        write_json(top_manifest_path, top_manifest)
        # The checksum inventory is regenerated after the verdict is bound.
        checksum_path = first / "manifests/input_bundle_checksums.json"
        checksum_files = _inventory(first, exclude={"manifests/input_bundle_checksums.json"})
        checksum = read_json(checksum_path)
        checksum["bundle_manifest_sha256"] = finalized_bundle_manifest_sha
        checksum["files"] = checksum_files
        write_json(checksum_path, checksum)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
        result = {
            **first_result,
            "output_dir": str(output_dir),
            "dataset_id": PILOT_DATASET_ID,
            "bundle_scope": "PILOT",
            "pilot_only": True,
            "formal_eligible": False,
            "case_count": PILOT_CASE_COUNT,
            "deterministic_build": deterministic,
            "bundle_manifest_sha256": sha256_file(
                output_dir / "manifests/input_bundle_manifest.json"
            ),
            "checksum_sha256": sha256_file(
                output_dir / "manifests/input_bundle_checksums.json"
            ),
            "top_manifest_sha256": sha256_file(output_dir / "manifest.json"),
            "checksum_file_count": len(
                read_json(output_dir / "manifests/input_bundle_checksums.json")["files"]
            ),
            "bundle_identity": _identity_hash({
                "manifest.json": sha256_file(output_dir / "manifest.json"),
                "manifests/input_bundle_manifest.json": sha256_file(output_dir / "manifests/input_bundle_manifest.json"),
                "manifests/input_bundle_checksums.json": sha256_file(output_dir / "manifests/input_bundle_checksums.json"),
                "manifests/evidence_packet_manifest.json": sha256_file(output_dir / "manifests/evidence_packet_manifest.json"),
                "manifests/processed_sha256.json": sha256_file(output_dir / "manifests/processed_sha256.json"),
            }),
        }
        return result
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--no-double-build", action="store_true")
    args = parser.parse_args()
    result = build(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        contract_dir=args.contract_dir,
        verify_determinism=not args.no_double_build,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
