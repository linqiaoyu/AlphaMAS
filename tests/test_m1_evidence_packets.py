from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.finmultitime.build_m1_evidence_packets import (
    CONTRACT_VERSION,
    MAX_PACKET_CHARS,
    PacketValidationError,
    build_packet,
    build_sections,
    canonical_write_bytes,
    format_agent_text,
    load_caption_identities,
    normalize_table_fact,
    validate_packet,
    validate_text_source,
)


def _fact(concept: str = "Assets") -> dict:
    return {
        "concept": concept,
        "value": 100,
        "unit": "USD",
        "form": "10-Q",
        "fy": 2024,
        "fp": "Q1",
        "period_start": "2024-01-01",
        "period_end": "2024-03-31",
        "period_duration_days": 91,
        "period_duration_class": "quarterly",
        "filed_date": "2024-05-01",
        "accession_number": "000-test",
        "source_provenance_hash": "a" * 64,
    }


def _case(symbol: str = "AAPL", decision: str = "2024-06-28") -> dict:
    sessions = [
        (date.fromisoformat(decision) - timedelta(days=60 - index)).isoformat()
        for index in range(61)
    ]
    facts = dict.fromkeys(
        (
            "Assets",
            "Liabilities",
            "StockholdersEquity",
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInInvestingActivities",
            "NetCashProvidedByUsedInFinancingActivities",
        )
    )
    facts["Assets"] = _fact()
    return {
        "record_type": "preprocessed_case_record",
        "contract_version": CONTRACT_VERSION,
        "dataset_id": "finmultitime_3stocks_2024h1_v1",
        "symbol": symbol,
        "decision_session": decision,
        "decision_time_utc": f"{decision}T20:00:00+00:00",
        "source_hash_reference": "b" * 64,
        "_path": f"cases/{symbol}/{decision}.json",
        "TEXT": {
            "status": "UNAVAILABLE",
            "reason": "no eligible source record",
            "selected_records": [],
            "latest_safe_date": None,
        },
        "TABLE": {
            "status": "AVAILABLE",
            "facts": facts,
            "source_members": ["financial_reports/test.json"],
        },
        "TIME_SERIES": {
            "status": "AVAILABLE",
            "through_session": decision,
            "selected_row_count": 61,
            "required_row_count": 61,
            "selected_session_dates": sessions,
            "source_member": "time_series/test.csv",
            "summary": {
                "cumulative_return_5d": 0.1,
                "cumulative_return_20d": 0.2,
                "cumulative_return_60d": 0.3,
                "realised_volatility_20d_annualised": 0.4,
                "high_low_range_20d": 0.5,
                "drawdown_from_60d_peak": -0.1,
                "relative_volume_vs_20d_mean": 1.1,
            },
        },
        "IMAGE": {
            "status": "UNAVAILABLE",
            "caption_status": "NOT_APPLICABLE",
            "reason": "no completed image",
            "eligible_image": None,
        },
        "contract_projection": {"PIT_violation_count": 0},
    }


def _manifest() -> dict:
    return {
        "evidence_contract_sha256": "1" * 64,
    }


def _caption_identity() -> dict:
    return {
        "AMZN": {
            "caption_manifest_sha256": "2" * 64,
            "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "model_revision": "r" * 40,
            "aws_environment_freeze_sha256": "3" * 64,
            "package_freeze_sha256": "4" * 64,
            "captioning_code_sha": "5" * 40,
            "prompt_sha256": "6" * 64,
            "schema_sha256": "7" * 64,
            "caption_ref": "qwen/captions/amzn.json",
            "caption_sha256": "8" * 64,
            "image_filename": "amzn.png",
            "image_sha256": "9" * 64,
            "canonical_caption": {"trend": "upward"},
        },
        "JPM": {
            "caption_manifest_sha256": "2" * 64,
            "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "model_revision": "r" * 40,
            "aws_environment_freeze_sha256": "3" * 64,
            "package_freeze_sha256": "4" * 64,
            "captioning_code_sha": "5" * 40,
            "prompt_sha256": "6" * 64,
            "schema_sha256": "7" * 64,
            "caption_ref": "qwen/captions/jpm.json",
            "caption_sha256": "a" * 64,
            "image_filename": "jpm.png",
            "image_sha256": "b" * 64,
            "canonical_caption": {"trend": "upward"},
        },
    }


def test_final_packet_has_exactly_four_sections_and_explicit_unavailable_text() -> None:
    sections = build_sections(_case(), _caption_identity())
    assert set(sections) == {"TEXT", "TABLE", "TIME_SERIES", "IMAGE"}
    assert sections["TEXT"]["status"] == "UNAVAILABLE"
    rendered = format_agent_text("AAPL", "2024-06-28", sections)
    assert "[TEXT]" in rendered
    assert "Status: UNAVAILABLE" in rendered
    assert "Missingness reason: no eligible source record" in rendered


def test_table_preserves_duration_metadata() -> None:
    normalized = normalize_table_fact(_fact("NetCashProvidedByUsedInOperatingActivities"))
    assert normalized["period_duration_days"] == 91
    assert normalized["period_duration_class"] == "quarterly"
    assert normalized["filed_date"] == "2024-05-01"
    assert normalized["accession"] == "000-test"


def test_time_series_has_the_frozen_seven_summary_fields() -> None:
    section = build_sections(_case(), _caption_identity())["TIME_SERIES"]
    assert list(section["summaries"]) == [
        "cumulative_return_5d",
        "cumulative_return_20d",
        "cumulative_return_60d",
        "realised_volatility_20d_annualised",
        "high_low_range_20d",
        "drawdown_from_60d_peak",
        "relative_volume_vs_20d_mean",
    ]


def test_aapl_has_no_caption_and_amzn_jpm_reuse_caption_identity() -> None:
    identities = _caption_identity()
    aapl = build_sections(_case("AAPL"), identities)["IMAGE"]
    assert aapl["status"] == "UNAVAILABLE"
    assert aapl["caption_status"] == "NOT_APPLICABLE"

    for symbol in ("AMZN", "JPM"):
        case = _case(symbol)
        case["IMAGE"] = {
            "status": "AVAILABLE",
            "caption_status": "GENERATED",
            "evidence_age_calendar_days": 180,
            "eligible_image": {
                "period_start": "2023-07-01",
                "period_end": "2023-12-31",
                "processed_path": f"image/{symbol.lower()}.png",
                "sha256": identities[symbol]["image_sha256"],
            },
            "caption_ref": identities[symbol]["caption_ref"],
            "caption_sha256": identities[symbol]["caption_sha256"],
        }
        image = build_sections(case, identities)["IMAGE"]
        assert image["caption_sha256"] == identities[symbol]["caption_sha256"]


def test_packet_routing_and_character_ceiling() -> None:
    packet, text = build_packet(_case(), Path("."), _manifest(), _caption_identity())
    assert packet["routing"]["modality_to_analyst"] == {
        "TEXT": "News Analyst",
        "TABLE": "Fundamentals Analyst",
        "TIME_SERIES": "Market Analyst",
        "IMAGE": "Market Analyst",
    }
    assert packet["routed_projections"]["social_analyst"]["text"] == ""
    assert packet["character_counts"]["complete"] == len(text)
    assert packet["character_counts"]["complete"] <= MAX_PACKET_CHARS


def test_pit_rejection_for_same_day_news() -> None:
    case = _case()
    case["TEXT"] = {
        "status": "AVAILABLE",
        "selected_records": [
            {
                "date": "2024-06-28",
                "title": "same day",
                "body": "body",
                "url": "https://example.test",
                "record_hash": "a" * 64,
            }
        ],
        "latest_safe_date": "2024-06-28",
    }
    with pytest.raises(PacketValidationError, match="same-day or future"):
        validate_text_source(case, date(2024, 6, 28))


def test_future_field_prohibition() -> None:
    packet, _ = build_packet(_case(), Path("."), _manifest(), _caption_identity())
    tampered = copy.deepcopy(packet)
    tampered["target_label"] = "x"
    with pytest.raises(PacketValidationError, match="future-facing field path"):
        validate_packet(tampered, date(2024, 6, 28))

    tampered = copy.deepcopy(packet)
    tampered["agent_text"] += "\nPrediction: up\n"
    with pytest.raises(PacketValidationError, match="field label"):
        validate_packet(tampered, date(2024, 6, 28))


def test_packet_hash_serialization_is_deterministic() -> None:
    packet, _ = build_packet(_case(), Path("."), _manifest(), _caption_identity())
    assert canonical_write_bytes(packet) == canonical_write_bytes(copy.deepcopy(packet))


def test_caption_loader_refuses_pending_or_contract_mismatch(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    caption_manifest = {
        "evidence_contract_version": CONTRACT_VERSION,
        "evidence_contract_sha256": "1" * 64,
        "formal_case_reference_counts": {"GENERATED": 52},
        "captions": [],
    }
    (qwen / "caption_manifest.json").write_text(json.dumps(caption_manifest))
    pending = {
        "evidence_contract_sha256": "1" * 64,
        "caption_status_counts": {"GENERATED": 0, "NOT_APPLICABLE": 26, "PENDING": 52},
    }
    (qwen / "qwen_input_manifest.json").write_text(json.dumps(pending))
    with pytest.raises(PacketValidationError, match="52/26/0"):
        load_caption_identities(tmp_path, "1" * 64)

    pending["caption_status_counts"] = {"GENERATED": 52, "NOT_APPLICABLE": 26, "PENDING": 0}
    pending["evidence_contract_sha256"] = "9" * 64
    (qwen / "qwen_input_manifest.json").write_text(json.dumps(pending))
    with pytest.raises(PacketValidationError, match="contract hash"):
        load_caption_identities(tmp_path, "1" * 64)
