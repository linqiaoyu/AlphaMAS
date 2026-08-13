from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import date, timedelta

import pytest

import scripts.finmultitime.preprocess_m1_inputs as preprocess
from scripts.finmultitime.design_m1_contract import SELECTED_TABLE_CONCEPTS
from scripts.finmultitime.preprocess_m1_inputs import (
    DEFAULT_CONTRACT_DIR,
    NEWS_MEMBERS,
    TS_MEMBERS,
    enforce_frozen_contract,
    image_case_section,
    research_hashes,
    source_member_hash,
    table_case_section,
    text_case_section,
    ts_case_section,
    write_json,
)


def _news_record(day: str, *, title: str, url: str = "https://example.test/a") -> dict:
    return {
        "Date": day,
        "Article_title": title,
        "Url": url,
        "Article": "body\nwith whitespace",
        "Stock_symbol": "AAPL",
    }


def _table_row(concept: str, *, filed: str, value: int) -> dict:
    return {
        "member": "financial_reports/aapl/test.json",
        "taxonomy": "us-gaap",
        "concept": concept,
        "unit": "USD",
        "start": None,
        "end": date(2023, 12, 31),
        "filed": date.fromisoformat(filed),
        "raw_start": None,
        "raw_end": "2023-12-31",
        "raw_filed": filed,
        "val": value,
        "accn": f"0000000000-{concept}",
        "form": "10-K",
        "fy": 2023,
        "fp": "FY",
        "frame": None,
    }


def _series_rows(decision: date, count: int = 61) -> list[dict]:
    return [
        {
            "session_date": decision - timedelta(days=count - index - 1),
            "numeric": {
                "Close": 100.0 + index,
                "High": 101.0 + index,
                "Low": 99.0 + index,
                "Volume": 1000.0 + index,
            },
        }
        for index in range(count)
    ]


def test_text_preprocessing_deduplicates_and_rejects_same_day() -> None:
    decision = date(2024, 1, 10)
    first = _news_record("2024-01-09", title="kept")
    raw = {"records": [
        first,
        dict(first),
        _news_record("2024-01-05", title="older", url="https://example.test/b"),
        _news_record("2024-01-10", title="same day", url="https://example.test/c"),
        _news_record("2023-12-11", title="inclusive boundary", url="https://example.test/e"),
        _news_record("2023-12-10", title="exclusive boundary", url="https://example.test/f"),
        _news_record("2023-11-01", title="outside lookback", url="https://example.test/d"),
    ]}

    section = text_case_section(raw, "AAPL", decision)

    assert section["status"] == "AVAILABLE"
    assert len(section["selected_records"]) == 3
    assert section["dedup_removed_count"] == 1
    assert section["same_day_ambiguous_rejected_count"] == 1
    assert section["source_member"] == NEWS_MEMBERS["AAPL"]
    assert section["selected_records"][0]["duplicate_provenance"]
    assert all(len(item["title"]) <= 200 for item in section["selected_records"])
    assert all(len(item["body"]) <= 900 for item in section["selected_records"])


def test_table_preprocessing_is_pit_safe_and_exposes_unavailable_concepts() -> None:
    decision = date(2024, 1, 10)
    tables = {
        "AAPL": {
            "observations": [
                _table_row("Assets", filed="2024-01-09", value=100),
                _table_row("Liabilities", filed="2024-01-10", value=50),
            ]
        }
    }

    section = table_case_section(tables, "AAPL", decision)

    assert section["status"] == "AVAILABLE"
    assert section["facts"]["Assets"]["value"] == 100
    assert section["facts"]["Assets"]["filed_date"] == "2024-01-09"
    assert section["facts"]["Liabilities"] is None
    assert set(section["unavailable_concepts"]) == set(SELECTED_TABLE_CONCEPTS) - {"Assets"}
    assert section["same_day_ambiguous_rejected_count"] == 1


def test_time_series_preprocessing_requires_61_completed_rows() -> None:
    decision = date(2024, 1, 10)
    series = {"AAPL": {"rows": _series_rows(decision)}}

    section = ts_case_section(series, "AAPL", decision)

    assert section["status"] == "AVAILABLE"
    assert section["selected_row_count"] == 61
    assert section["required_row_count"] == 61
    assert section["through_session"] == decision
    assert section["source_member"] == TS_MEMBERS["AAPL"]
    assert section["summary"]["cumulative_return_60d"] > 0


def test_image_preprocessing_has_explicit_unavailable_and_pending_states() -> None:
    decision = date(2024, 1, 10)
    unavailable = image_case_section(None, decision)
    available = image_case_section(
        {
            "filename": "aapl_2023_H2_candlestick.png",
            "path": "/raw/aapl_2023_H2_candlestick.png",
            "period_start": date(2023, 7, 1),
            "period_end": date(2023, 12, 31),
            "bytes": 10,
            "sha256": "a" * 64,
        },
        decision,
    )

    assert unavailable["status"] == "UNAVAILABLE"
    assert unavailable["caption_status"] == "PENDING"
    assert available["status"] == "AVAILABLE"
    assert available["evidence_age_calendar_days"] == 10
    assert available["caption_status"] == "PENDING"


def test_all_four_modality_sections_preserve_unavailable_status() -> None:
    decision = date(2024, 1, 10)

    assert text_case_section(None, "JPM", decision)["status"] == "UNAVAILABLE"
    assert table_case_section({"JPM": None}, "JPM", decision)["status"] == "UNAVAILABLE"
    assert ts_case_section({"JPM": None}, "JPM", decision)["status"] == "UNAVAILABLE"
    assert image_case_section(None, decision)["status"] == "UNAVAILABLE"


def test_source_member_hash_records_archive_member_identity(tmp_path) -> None:
    archive_path = tmp_path / "source.zip"
    payload = b"deterministic source member\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("member.jsonl", payload)

    result = source_member_hash(tmp_path, "source.zip", "member.jsonl")

    assert result["source_kind"] == "zip_member"
    assert result["member_uncompressed_size"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_frozen_contract_mismatch_fails_closed(tmp_path) -> None:
    for name in (
        "m1_evidence_contract.json",
        "m1_evidence_contract_freeze.json",
        "m1_evidence_contract_case_simulation.csv",
    ):
        shutil.copy(DEFAULT_CONTRACT_DIR / name, tmp_path / name)
    contract_path = tmp_path / "m1_evidence_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["packet_version"] = "tampered"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="contract version mismatch"):
        enforce_frozen_contract(tmp_path)


def test_research_hashes_ignore_variable_metadata_and_are_stable(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        write_json(root / "cases/AAPL/2024-01-05.json", {"z": 3, "a": [1, 2]})
        write_json(root / "manifest.json", {"generated_at": "different"})
        write_json(root / "manifests/processed_sha256.json", {"generated_at": "different"})
        (root / "README.md").write_text("different", encoding="utf-8")

    assert research_hashes(left) == research_hashes(right)


def test_build_runs_two_staging_passes_for_determinism(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_build_to_directory(destination, raw_root, contract_dir, m0_snapshot_dir):
        nonlocal calls
        calls += 1
        write_json(destination / "research.json", {"stable": True})
        write_json(destination / "manifest.json", {"run": calls})
        return {
            "records": 78,
            "images": [],
            "equivalence": {"research_relevant_differences": 0},
            "research_hashes": research_hashes(destination),
        }

    monkeypatch.setattr(preprocess, "_build_to_directory", fake_build_to_directory)
    result = preprocess.build(
        output_dir=tmp_path / "processed",
        raw_root=tmp_path / "raw",
        contract_dir=tmp_path / "contract",
        m0_snapshot_dir=tmp_path / "m0",
        verify_determinism=True,
    )

    assert calls == 2
    assert result["deterministic_rerun"] == "PASS"
    assert (tmp_path / "processed/manifest.json").is_file()
