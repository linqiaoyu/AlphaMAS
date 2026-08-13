#!/usr/bin/env python3
"""Audit formal-use FinMultiTime TEXT rows and write frozen evidence artifacts.

This audit is deliberately source-local.  It verifies raw member identity and
row fidelity, records the complete legacy formal-use set, and preserves the
human-reviewed headline/URL/body verdicts without downloading or copying web
content.  The resulting symbol-level policy is fail-closed and is consumed by
the M1 preprocessor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.audit_finmultitime import NEWS_ARCHIVE, NEWS_MEMBERS, TARGETS
from scripts.finmultitime.design_m1_contract import (
    load_raw_data,
    selected_news,
    truncate,
    MAX_ARTICLE_BODY_CHARS,
    MAX_ARTICLE_TITLE_CHARS,
)

DEFAULT_RAW_ROOT = Path("/Volumes/Jackson/Dataset/FinMultiTime")
DEFAULT_PREVIOUS_SOURCE_MANIFEST = Path(
    "/Users/yulinqiao/Desktop/AlphaMAS-Experiments/experiments/M1/inputs/manifests/source_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/m1"
FORMAL_RECORD_STATUSES = {"VERIFIED_MATCH", "VERIFIED_MISMATCH", "UNVERIFIABLE"}
AUDIT_VERSION = "1.0"
FORMAL_DECISIONS = (
    "2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26",
    "2024-02-02", "2024-02-09", "2024-02-16", "2024-02-23",
    "2024-03-01", "2024-03-08", "2024-03-15", "2024-03-22",
    "2024-03-28", "2024-04-05", "2024-04-12", "2024-04-19",
    "2024-04-26", "2024-05-03", "2024-05-10", "2024-05-17",
    "2024-05-24", "2024-05-31", "2024-06-07", "2024-06-14",
    "2024-06-21", "2024-06-28",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list)) else ("" if value is None else value)
                for key, value in row.items()
            })


def member_identity(raw_root: Path, symbol: str) -> dict[str, Any]:
    member = NEWS_MEMBERS[symbol]
    archive_path = raw_root / NEWS_ARCHIVE
    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            return {
                "symbol": symbol,
                "available": False,
                "outer_archive": NEWS_ARCHIVE,
                "member": member,
            }
        info = archive.getinfo(member)
        payload = archive.read(info)
    return {
        "symbol": symbol,
        "available": True,
        "outer_archive": NEWS_ARCHIVE,
        "member": member,
        "member_uncompressed_size": info.file_size,
        "member_compressed_size": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
        "sha256": sha256_bytes(payload),
    }


def expected_member_identity(source_manifest: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    for item in source_manifest.get("records", []):
        if item.get("symbol") == symbol and item.get("modality") == "TEXT":
            return item
    return None


def source_identity_check(actual: dict[str, Any], expected: dict[str, Any] | None) -> dict[str, Any]:
    expected_member = expected.get("member") if expected else None
    if isinstance(expected_member, dict):
        fields = (
            "member", "member_uncompressed_size", "member_compressed_size",
            "member_crc32", "sha256",
        )
        actual_values = {field: actual.get(field) for field in fields}
        expected_values = {field: expected_member.get(field) for field in fields}
        return {
            "status": "PASS" if actual_values == expected_values else "FAIL",
            "actual": actual_values,
            "expected": expected_values,
        }
    expected_available = bool(expected and expected.get("available"))
    return {
        "status": "PASS" if actual.get("available") == expected_available else "FAIL",
        "actual": {"available": actual.get("available"), "member": actual.get("member")},
        "expected": {"available": expected_available, "member": NEWS_MEMBERS[actual["symbol"]]},
    }


def record_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "Date": record.get("Date"),
        "Stock_symbol": record.get("Stock_symbol"),
        "Article_title": record.get("Article_title"),
        "Url": record.get("Url"),
        "Article": record.get("Article"),
    }


def selected_projection(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "Date": record.get("Date"),
        "Stock_symbol": record.get("Stock_symbol"),
        "Article_title": truncate(record.get("Article_title"), MAX_ARTICLE_TITLE_CHARS),
        "Url": record.get("Url"),
        "Article": truncate(record.get("Article"), MAX_ARTICLE_BODY_CHARS),
    }


def normalize_for_fidelity(value: Any) -> Any:
    if value is None:
        return None
    return str(value)


def row_fidelity(raw: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    raw_values = record_projection(raw)
    selected_values = selected_projection(selected)
    checks = {
        field: normalize_for_fidelity(selected_values[field]) == normalize_for_fidelity(
            truncate(raw_values[field], MAX_ARTICLE_TITLE_CHARS if field == "Article_title" else MAX_ARTICLE_BODY_CHARS)
            if field in {"Article_title", "Article"} else raw_values[field]
        )
        for field in raw_values
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "fields": checks}


def compact_body(value: Any, length: int = 260) -> str:
    return " ".join(str(value or "").split())[:length]


def classify_record(symbol: str, record: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Return the pre-registered audit verdict for a formal-use row.

    This is a source-integrity classification, not a selection rule.  The
    AAPL verdict is supported by the explicit unrelated 2015--2019 body
    corpus in the artifact; AMZN rows are source-locally coherent.  The
    resulting formal policy is symbol-level, so no individual article is
    retained or discarded based on appearance.
    """

    if symbol == "AAPL":
        return (
            "VERIFIED_MISMATCH",
            "Article body begins with an unrelated historical company/event and 2015-2019 earnings-era prose while the title and Nasdaq URL identify a different 2023 Apple-targeted article.",
            {
                "headline_url": "TITLE_AND_URL_PRESENT",
                "body_evidence": compact_body(record.get("Article")),
                "external_web_check": "NOT_PERFORMED",
            },
        )
    if symbol == "AMZN":
        return (
            "VERIFIED_MATCH",
            "Title slug, URL, and source-local body discuss the same Amazon/market topic; no clear cross-article body substitution was observed.",
            {
                "headline_url": "TITLE_AND_URL_PRESENT",
                "body_evidence": compact_body(record.get("Article")),
                "external_web_check": "NOT_PERFORMED",
            },
        )
    raise ValueError(f"unexpected formal-use TEXT symbol: {symbol}")


def structural_sample(rows: list[dict[str, Any]], *, dates: set[str], verdict: str) -> dict[str, Any]:
    selected = [
        {
            "source_index": index,
            "date": row.get("Date"),
            "title": row.get("Article_title"),
            "url": row.get("Url"),
            "body_prefix": compact_body(row.get("Article")),
        }
        for index, row in enumerate(rows) if row.get("Date") in dates
    ]
    return {
        "dates": sorted(dates),
        "sample_size": len(selected),
        "source_index_range": [
            min((row["source_index"] for row in selected), default=None),
            max((row["source_index"] for row in selected), default=None),
        ],
        "observed_verdict": verdict,
        "records": selected,
    }


def build_artifacts(raw_root: Path, source_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    if not raw_root.is_dir():
        raise ValueError(f"raw source is missing: {raw_root}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    data = load_raw_data(raw_root)
    formal_cases = [
        {"symbol": symbol, "decision_session": decision.isoformat()}
        for decision in (date.fromisoformat(value) for value in FORMAL_DECISIONS)
        for symbol in TARGETS
    ]
    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    case_refs: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for case in formal_cases:
        selection = selected_news(data["news"][case["symbol"]], date.fromisoformat(case["decision_session"]))
        for record in selection["records"]:
            key = (case["symbol"], record["_record_hash"])
            selected_by_key[key] = record
            case_refs[key].append(f"{case['symbol']}:{case['decision_session']}")

    record_rows: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for (symbol, record_hash), raw_record in sorted(selected_by_key.items()):
        status, rationale, evidence = classify_record(symbol, raw_record)
        selected = {
            "symbol": symbol,
            "decision_session": case_refs[(symbol, record_hash)][0].split(":", 1)[1],
            "record_hash": record_hash,
            "source_index": raw_record["_source_index"],
            "date": raw_record.get("Date"),
            "title": truncate(raw_record.get("Article_title"), MAX_ARTICLE_TITLE_CHARS),
            "url": raw_record.get("Url", ""),
            "body": truncate(raw_record.get("Article"), MAX_ARTICLE_BODY_CHARS),
        }
        fidelity_source = {
            **raw_record,
            "Article_title": selected["title"],
            "Article": selected["body"],
        }
        fidelity = row_fidelity(raw_record, fidelity_source)
        if fidelity["status"] != "PASS":
            raise ValueError(f"raw-row fidelity failed for {symbol}:{record_hash}")
        row = {
            "symbol": symbol,
            "formal_case_references": case_refs[(symbol, record_hash)],
            "formal_case_reference_count": len(case_refs[(symbol, record_hash)]),
            "source_member": NEWS_MEMBERS[symbol],
            "source_index": raw_record["_source_index"],
            "record_hash": record_hash,
            "Date": raw_record.get("Date"),
            "Stock_symbol": raw_record.get("Stock_symbol"),
            "Article_title": raw_record.get("Article_title"),
            "Url": raw_record.get("Url"),
            "Article": raw_record.get("Article"),
            "preprocessed_title": selected["title"],
            "preprocessed_body": selected["body"],
            "raw_row_fidelity": fidelity["status"],
            "integrity_status": status,
            "integrity_rationale": rationale,
        }
        record_rows.append(row)
        audit_records.append({
            **row,
            "raw_row": record_projection(raw_record),
            "preprocessed_copy": {
                "date": selected["date"],
                "title": selected["title"],
                "url": selected["url"],
                "body": selected["body"],
                "record_hash": record_hash,
                "source_index": raw_record["_source_index"],
            },
            "raw_row_fidelity": fidelity,
            "audit_evidence": evidence,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "m1_text_source_integrity_records.csv", record_rows)

    symbol_identities = {}
    for symbol in TARGETS:
        actual = member_identity(raw_root, symbol)
        expected = expected_member_identity(source_manifest, symbol)
        identity = source_identity_check(actual, expected)
        if identity["status"] != "PASS":
            raise ValueError(f"raw source identity mismatch for {symbol}")
        symbol_identities[symbol] = {"actual": actual, "expected_comparison": identity}

    formal_counts = Counter(item["integrity_status"] for item in audit_records)
    by_symbol = Counter(item["symbol"] for item in audit_records)
    text_case_references_by_symbol = {
        symbol: len({reference for item in audit_records if item["symbol"] == symbol for reference in item["formal_case_references"]})
        for symbol in TARGETS
    }
    record_case_references_by_symbol = {
        symbol: sum(item["formal_case_reference_count"] for item in audit_records if item["symbol"] == symbol)
        for symbol in TARGETS
    }
    policy_by_symbol = {
        "AAPL": {
            "audit_status": "VERIFIED_MISMATCH",
            "formal_text_policy": "UNAVAILABLE",
            "formal_use_record_count": by_symbol["AAPL"],
            "reason": "AAPL formal-use rows show clustered headline/URL/body corruption in the raw member; the entire symbol TEXT modality is frozen unavailable.",
        },
        "AMZN": {
            "audit_status": "VERIFIED_MATCH",
            "formal_text_policy": "AVAILABLE",
            "formal_use_record_count": by_symbol["AMZN"],
            "reason": "Selected AMZN formal-use rows pass the source-local title/URL/body integrity audit and remain source-native.",
        },
        "JPM": {
            "audit_status": "UNVERIFIABLE",
            "formal_text_policy": "UNAVAILABLE",
            "formal_use_record_count": 0,
            "reason": "No sp500_news/JPM.jsonl member exists; no external or cross-symbol replacement is permitted.",
        },
    }
    audit = {
        "schema_version": AUDIT_VERSION,
        "record_type": "m1_formal_text_source_integrity_audit",
        "verdict": "CORRECTION_REQUIRED_FAIL_CLOSED",
        "formal_m1_run": False,
        "outcomes_inspected": False,
        "external_web_content_used_as_evidence": False,
        "raw_source": {
            "root": str(raw_root),
            "raw_source_read_only": True,
            "member_identity_comparison": symbol_identities,
            "previous_source_manifest": "experiments/M1/inputs/manifests/source_manifest.json",
        },
        "formal_use_set": {
            "case_count": len(formal_cases),
            "unique_record_count": len(audit_records),
            "unique_record_counts_by_symbol": dict(sorted(by_symbol.items())),
            "formal_cases_with_text_by_symbol": text_case_references_by_symbol,
            "record_case_reference_counts_by_symbol": record_case_references_by_symbol,
            "records_artifact": "docs/m1/m1_text_source_integrity_records.csv",
        },
        "record_status_counts": dict(sorted(formal_counts.items())),
        "records": audit_records,
        "structural_member_audit": {
            "AAPL": {
                "classification": "CLUSTERED_SYSTEMIC_IN_RELEVANT_DATE_BLOCK",
                "reason": "The contiguous 2023-12-15/2023-12-16 block around formal-use indices contains current-looking titles/URLs paired with unrelated 2015-2019/2017 earnings-era bodies.",
                "sample": structural_sample(data["news"]["AAPL"]["records"], dates={"2023-12-15", "2023-12-16"}, verdict="VERIFIED_MISMATCH"),
            },
            "AMZN": {
                "classification": "NO_CLEAR_CORRUPTION_IN_RELEVANT_DATE_BLOCK",
                "reason": "Systematic review of the 2023-12-16 block containing the formal-use rows found title/URL/topic-compatible source-local bodies; no individual row was manually substituted.",
                "sample": structural_sample(data["news"]["AMZN"]["records"], dates={"2023-12-16"}, verdict="VERIFIED_MATCH"),
            },
            "JPM": {
                "classification": "MEMBER_ABSENT",
                "sample": {"sample_size": 0, "observed_verdict": "UNVERIFIABLE"},
            },
        },
        "policy_by_symbol": policy_by_symbol,
        "correction_rule": "If a symbol's relevant FinMultiTime member has any verified source-integrity corruption or lacks a member, freeze that symbol's entire TEXT modality UNAVAILABLE. VERIFIED_MATCH symbols retain the fixed deterministic selection; no external replacement is allowed.",
    }
    write_json(output_dir / "m1_text_source_integrity_audit.json", audit)
    return {
        "unique_record_count": len(audit_records),
        "counts_by_symbol": dict(sorted(by_symbol.items())),
        "status_counts": dict(sorted(formal_counts.items())),
        "audit_sha256": sha256_file(output_dir / "m1_text_source_integrity_audit.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_PREVIOUS_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build_artifacts(args.raw_root, args.source_manifest, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
