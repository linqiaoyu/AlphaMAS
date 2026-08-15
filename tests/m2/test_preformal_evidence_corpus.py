from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.finmultitime.design_m1_contract import SELECTED_TABLE_CONCEPTS
from scripts.m2.build_preformal_evidence_corpus import (
    DEFAULT_M1_ARCHIVE,
    EXPECTED_ROLE_COUNTS,
    PROMPT_SHA256,
    SCHEMA_SHA256,
    SYMBOLS,
    TIER_IDENTITIES,
    _m1_caption_map,
    audit_text_candidates,
    build_packet,
    case_list_sha,
    load_plan,
    _pit_violations,
)
from scripts.m2.run_preformal_qwen_batch import validate_environment

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_case_membership_roles_and_tiers() -> None:
    plan = load_plan()
    assert len(plan) == 96
    assert {row["symbol"] for row in plan} == set(SYMBOLS)
    assert Counter(row["split_role"] for row in plan) == EXPECTED_ROLE_COUNTS
    for tier, (count, identity) in TIER_IDENTITIES.items():
        rows = [row for row in plan if row[f"in_{tier.lower()}"]]
        assert len(rows) == count
        assert case_list_sha(rows) == identity


def test_role_protection_is_explicit() -> None:
    plan = load_plan()
    holdout = [row for row in plan if row["split_role"] == "FINAL_HOLDOUT"]
    assert len(holdout) == 16
    assert all(row["protected"] for row in holdout)
    assert all(not row["available_for_training"] for row in holdout)
    assert all(not row["available_for_validation"] for row in holdout)
    pilot = [row for row in plan if row["split_role"] == "E2E_PILOT"]
    assert all(row["engineering_only"] and not row["performance_for_selection"] for row in pilot)


def _raw_record(symbol: str, title: str = "Example Company Reports Results") -> dict[str, str]:
    return {
        "Date": "2023-05-04",
        "Article_title": title,
        "Article": "Example Company reports results and discusses its operations.",
        "Url": "https://example.test/example-company-reports-results",
        "Stock_symbol": symbol,
    }


def test_source_integrity_fails_closed_for_aapl() -> None:
    plan = [row for row in load_plan() if row["decision_session"] == "2023-05-05"]
    data = {"news": {symbol: None for symbol in SYMBOLS}}
    data["news"]["AAPL"] = {"records": [_raw_record("AAPL")]}
    data["news"]["ARR"] = {"records": [_raw_record("ARR")]}
    audit = audit_text_candidates(data, plan)
    by_symbol = {item["symbol"]: item for item in audit["records"]}
    assert by_symbol["AAPL"]["status"] == "REJECTED"
    assert "M1_FROZEN_SOURCE_INTEGRITY_POLICY" in by_symbol["AAPL"]["rejection_reasons"]
    assert by_symbol["ARR"]["status"] == "SAFE"
    assert audit["outcome_data_used"] is False
    assert audit["automatic_symbol_reselection"] is False


def _unavailable_packet_skeleton() -> dict[str, object]:
    sessions = [(date(2023, 5, 5) - timedelta(days=index)).isoformat() for index in range(60, -1, -1)]
    summary = {
        "cumulative_return_5d": 0.01,
        "cumulative_return_20d": 0.02,
        "cumulative_return_60d": 0.03,
        "realised_volatility_20d_annualised": 0.2,
        "high_low_range_20d": 0.1,
        "drawdown_from_60d_peak": -0.05,
        "relative_volume_vs_20d_mean": 1.1,
    }
    return {
        "case_id": "AAPL:2023-05-05",
        "symbol": "AAPL",
        "decision_session": "2023-05-05",
        "decision_time": "2023-05-05T20:00:00+00:00",
        "role": "TRAIN",
        "tier_membership": {"in_compact": False, "in_standard": False, "in_maximum": True},
        "protected": False,
        "available_for_training": True,
        "available_for_validation": False,
        "engineering_only": False,
        "performance_for_selection": False,
        "pit_cutoff": "2023-05-05",
        "routing": {},
        "sections": {
            "TEXT": {"status": "UNAVAILABLE", "reason": "frozen", "selected_records": [], "source_member": None},
            "TABLE": {"status": "UNAVAILABLE", "facts": {concept: None for concept in SELECTED_TABLE_CONCEPTS}, "source_members": []},
            "TIME_SERIES": {"status": "AVAILABLE", "selected_row_count": 61, "required_row_count": 61, "selected_session_dates": sessions, "summary": summary, "through_session": "2023-05-05", "source_member": "fixture.csv"},
            "IMAGE": {"status": "UNAVAILABLE", "caption_status": "NOT_APPLICABLE", "reason": "frozen"},
        },
    }


def test_packet_is_bounded_pit_safe_and_outcome_free() -> None:
    packet, rendered = build_packet(_unavailable_packet_skeleton(), {})
    assert packet["role"] == "TRAIN"
    assert packet["TIME_SERIES"]["through_session"] == "2023-05-05"
    assert "future_return" not in rendered
    assert len(rendered) <= 22_000
    normalized = {"case_id": packet["case_id"], "decision_session": packet["decision_session"], "sections": {name: packet[name] for name in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")}}
    assert _pit_violations([normalized]) == []


def test_packet_and_corpus_binding_inputs_are_deterministic() -> None:
    left, left_text = build_packet(_unavailable_packet_skeleton(), {})
    right, right_text = build_packet(_unavailable_packet_skeleton(), {})
    assert left == right
    assert left_text == right_text
    assert left["packet_sha256"] == right["packet_sha256"]


def test_frozen_m1_caption_reuse_is_exact_image_sha() -> None:
    captions = _m1_caption_map(DEFAULT_M1_ARCHIVE)
    assert "215c01f8a03dc55719558644992b14c28d8b9f604d44e3504ef7df0f99997bb8" in captions
    assert captions["215c01f8a03dc55719558644992b14c28d8b9f604d44e3504ef7df0f99997bb8"]["canonical_caption_sha256"] == "131c2904cd82c850f94b00cc58e2b97b2ef69c7e805fa562ef2c8f2e00a568fe"


def test_qwen_contract_identities_and_environment(tmp_path: Path) -> None:
    contract = json.loads((REPO_ROOT / "docs/m1/m1_evidence_contract.json").read_text())
    environment = json.loads((REPO_ROOT / "docs/m1/m1_qwen_aws_environment_freeze.json").read_text())
    snapshot = tmp_path / environment["qwen"]["revision"]
    snapshot.mkdir()
    validated = validate_environment(contract, environment, snapshot)
    assert validated["contract"]["prompt_sha256"] == PROMPT_SHA256
    assert validated["contract"]["schema_sha256"] == SCHEMA_SHA256
    assert validated["qwen"]["local_snapshot_path"] == str(snapshot)
