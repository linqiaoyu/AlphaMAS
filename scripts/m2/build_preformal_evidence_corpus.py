#!/usr/bin/env python3
"""Build the deterministic M2-06 pre-Formal FinMultiTime evidence corpus.

The raw dataset is opened read-only.  Only the eight frozen M2 symbols and the
source members required by the 96 frozen cases are read.  This module reuses
the M1 pure selection and packet-formatting functions; it never executes an
Agent, reads an outcome, or calls a network service.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.audit_finmultitime import (  # noqa: E402
    IMAGE_RE,
    read_jsonl_news,
    read_table_files,
    read_time_series,
)
from scripts.finmultitime.build_m1_evidence_packets import (  # noqa: E402
    MAX_IMAGE_SECTION_CHARS,
    MAX_NEWS_SECTION_CHARS,
    MAX_PACKET_CHARS,
    MAX_TABLE_SECTION_CHARS,
    MAX_TIME_SERIES_SECTION_CHARS,
    assert_no_future_fields,
    format_agent_text,
    format_image_section,
    format_table_section,
    format_text_section,
    format_time_series_section,
    validate_table_source,
    validate_text_source,
    validate_time_series_source,
    validate_packet,
)
from scripts.finmultitime.design_m1_contract import (  # noqa: E402
    CONTRACT_VERSION,
    MAX_ARTICLE_BODY_CHARS,
    MAX_ARTICLE_TITLE_CHARS,
    MIN_TIME_SERIES_ROWS,
    SELECTED_TABLE_CONCEPTS,
    canonical_json,
    image_selection,
    selected_news,
    table_selection,
    time_series_selection,
    truncate,
    ts_summary,
)
from scripts.finmultitime.preprocess_m1_inputs import (  # noqa: E402
    enforce_frozen_contract,
    selected_text_record,
    sha256_file,
    source_member_hash,
)
from scripts.finmultitime.run_qwen_caption import (  # noqa: E402
    CAPTION_FIELDS,
    MAX_CAPTION_CHARS,
    MODEL_REPO,
    MODEL_REVISION,
    parse_canonical_caption,
    validate_caption_content,
)

TASK_ID = "M2-06"
SCHEMA_VERSION = "1.0"
DATASET_ID = "m2_preformal_evidence_v1"
STARTING_SOURCE_SHA = "bc67a5cda93e08b40a97a37fd88feee9e16441b7"
STARTING_EXPERIMENTS_SHA = "3a8e63a9abe89a80787b68aabb9b3523453966fc"
HISTORICAL_QWEN_CODE_SHA = "4ce6af13c6dd82218d1bb9a2600fdbf7f108bc8c"
PROMPT_SHA256 = "284c6e52763796a47f7d30fd2e44cfe68d9211db6831d89ec5ed436920c34df9"
SCHEMA_SHA256 = "bf8f04330ffb1bd8468b9bf01eb96bec6b35bb4bad29c8e3f1ad6c47cf0ca8e4"
SYNTHETIC_IMAGE_SHA256 = "6ffb3fd7da6e769117acc480911a7de1e7c81cf02a75b4ed14a9909a47f02f4b"
SYNTHETIC_RAW_SHA256 = "09807960d0f82b69d2f10da316d11f092f0b51adef7bc0dc0ca7f13bfaadf045"
SYNTHETIC_CANONICAL_SHA256 = "3549cf3cdce93a0088fc17b7dc88c10399f78114ddfa813e39ce0d75f91eff94"

SYMBOLS = ("AAPL", "AEMD", "AGI", "AMZN", "ARR", "EML", "JBSS", "JPM")
ROLES = ("TRAIN", "VALIDATION", "FINAL_HOLDOUT", "E2E_PILOT")
EXPECTED_ROLE_COUNTS = {"TRAIN": 56, "VALIDATION": 16, "FINAL_HOLDOUT": 16, "E2E_PILOT": 8}
TIER_IDENTITIES = {
    "COMPACT": (72, "f35488ed0f910f73b11713bb7eadf00191f13f5e5b711b309dc879aca8075d62"),
    "STANDARD": (80, "f1253ab0ed8e23d9ae5656fc4250d7dca63bd6f9342e2fd9a2d84eac8e377452"),
    "MAXIMUM": (96, "68fbc780cbb454f78d0ae719f91665737ed6e04928eba91f4c9c0530745d601f"),
}
NEWS_ARCHIVE = "text/sp500_news.zip"
TABLE_ARCHIVE = "table/SP500_tabular.zip"
TS_ARCHIVE = "time_series/S&P500_time_series.zip"
DEFAULT_RAW_ROOT = Path("/Volumes/Jackson/Dataset/FinMultiTime")
DEFAULT_PLAN = REPO_ROOT / "docs/m2/m2_preformal_semantic_case_plan.csv"
DEFAULT_PROTOCOL = REPO_ROOT / "docs/m2/m2_preformal_data_and_split_protocol.json"
DEFAULT_M1_ARCHIVE = Path(
    "/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-Experiments/experiments/M1"
)

FORBIDDEN_KEY_TOKENS = (
    "future_return", "future_price", "prediction_target", "realised_next_return",
    "ground_truth", "reward", "target", "label", "outcome", "formal_result",
)
ROUTING = {
    "TEXT": "News Analyst",
    "TABLE": "Fundamentals Analyst",
    "TIME_SERIES": "Market Analyst",
    "IMAGE": "Market Analyst",
}


class EvidenceBuildError(ValueError):
    """Fail-closed M2-06 correctness error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_value(value: str) -> bool:
    if value not in {"true", "false"}:
        raise EvidenceBuildError(f"invalid frozen boolean: {value!r}")
    return value == "true"


def load_plan(plan_path: Path = DEFAULT_PLAN, protocol_path: Path = DEFAULT_PROTOCOL) -> list[dict[str, Any]]:
    with plan_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    protocol = read_json(protocol_path)
    result = []
    for row in rows:
        item = dict(row)
        item["in_compact"] = bool_value(row["compact_included"])
        item["in_standard"] = bool_value(row["standard_included"])
        item["in_maximum"] = bool_value(row["maximum_included"])
        item["protected"] = row["split_role"] == "FINAL_HOLDOUT"
        item["available_for_training"] = row["split_role"] == "TRAIN"
        item["available_for_validation"] = row["split_role"] == "VALIDATION"
        item["engineering_only"] = row["split_role"] == "E2E_PILOT"
        item["performance_for_selection"] = False
        result.append(item)
    validate_plan(result, protocol)
    return result


def case_list_sha(rows: Iterable[dict[str, Any]]) -> str:
    lines = [
        f"{row['case_id']}|{row['maturity_session']}|{row['split_role']}"
        for row in sorted(rows, key=lambda item: item["case_id"])
    ]
    return sha256_bytes(("\n".join(lines) + "\n").encode())


def validate_plan(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    if len(rows) != 96 or len({row["case_id"] for row in rows}) != 96:
        raise EvidenceBuildError("frozen case plan is not 96 unique cases")
    if tuple(sorted({row["symbol"] for row in rows})) != tuple(sorted(SYMBOLS)):
        raise EvidenceBuildError("frozen symbol membership drift")
    if Counter(row["split_role"] for row in rows) != EXPECTED_ROLE_COUNTS:
        raise EvidenceBuildError("frozen role counts drift")
    if any(row["decision_session"] >= "2024-01-01" for row in rows):
        raise EvidenceBuildError("2024 decision entered pre-Formal plan")
    tier_sets = {}
    for tier, (count, expected_sha) in TIER_IDENTITIES.items():
        key = f"in_{tier.lower()}"
        selected = [row for row in rows if row[key]]
        if len(selected) != count or case_list_sha(selected) != expected_sha:
            raise EvidenceBuildError(f"{tier} identity drift")
        frozen = protocol["budget_tiers"][tier]
        if frozen["case_count"] != count or frozen["canonical_case_list_sha256"] != expected_sha:
            raise EvidenceBuildError(f"{tier} protocol identity drift")
        tier_sets[tier] = {row["case_id"] for row in selected}
    if not tier_sets["COMPACT"] < tier_sets["STANDARD"] < tier_sets["MAXIMUM"]:
        raise EvidenceBuildError("tiers are not strictly nested")


def _table_member_pattern(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"^financial_reports/{symbol.lower()}/[^/]+\.json$", re.IGNORECASE)


def _image_items(raw_root: Path, symbol: str) -> list[dict[str, Any]]:
    directory = raw_root / "image/image" / f"S&P500_image_{symbol[0].lower()}" / symbol.lower()
    result = []
    if not directory.is_dir():
        return result
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name.startswith("._"):
            continue
        match = IMAGE_RE.match(path.name)
        if not match or match.group("symbol").upper() != symbol:
            continue
        year, half = int(match.group("year")), int(match.group("half"))
        period_start = date(year, 1 if half == 1 else 7, 1)
        period_end = date(year, 6, 30) if half == 1 else date(year, 12, 31)
        result.append({
            "filename": path.name,
            "path": str(path),
            "period_start": period_start,
            "period_end": period_end,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return result


def load_raw_members(raw_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not raw_root.is_dir():
        raise EvidenceBuildError(f"raw root missing: {raw_root}")
    news: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    series: dict[str, Any] = {}
    source_records: list[dict[str, Any]] = []
    with zipfile.ZipFile(raw_root / NEWS_ARCHIVE) as archive:
        names = set(archive.namelist())
        for symbol in SYMBOLS:
            member = f"sp500_news/{symbol}.jsonl"
            available = member in names
            news[symbol] = read_jsonl_news(archive, member) if available else None
            source_records.append({
                "symbol": symbol, "modality": "TEXT", "available": available,
                "source": source_member_hash(raw_root, NEWS_ARCHIVE, member) if available else {
                    "source_kind": "absent_zip_member", "outer_archive": NEWS_ARCHIVE, "member": member,
                },
            })
    with zipfile.ZipFile(raw_root / TABLE_ARCHIVE) as archive:
        names = archive.namelist()
        for symbol in SYMBOLS:
            members = sorted(name for name in names if _table_member_pattern(symbol).match(name))
            tables[symbol] = read_table_files(archive, members) if members else None
            source_records.append({
                "symbol": symbol, "modality": "TABLE", "available": bool(members),
                "sources": [source_member_hash(raw_root, TABLE_ARCHIVE, member) for member in members],
            })
    with zipfile.ZipFile(raw_root / TS_ARCHIVE) as archive:
        names = set(archive.namelist())
        for symbol in SYMBOLS:
            member = f"S&P500_time_series/{symbol.lower()}.csv"
            if member not in names:
                raise EvidenceBuildError(f"required time-series member absent: {member}")
            series[symbol] = read_time_series(archive, member)
            source_records.append({
                "symbol": symbol, "modality": "TIME_SERIES", "available": True,
                "source": source_member_hash(raw_root, TS_ARCHIVE, member),
            })
    images = {symbol: _image_items(raw_root, symbol) for symbol in SYMBOLS}
    return {"news": news, "tables": tables, "series": series, "images": images}, {
        "schema_version": SCHEMA_VERSION,
        "raw_root": str(raw_root),
        "records": source_records,
    }


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) >= 3}


def audit_text_candidates(data: dict[str, Any], plan: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit every record reachable by any frozen case using source content only."""
    decisions = sorted({date.fromisoformat(row["decision_session"]) for row in plan})
    records = []
    safe_hashes: dict[str, set[str]] = {symbol: set() for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        candidates: dict[str, dict[str, Any]] = {}
        for decision in decisions:
            for record in selected_news(data["news"][symbol], decision)["records"]:
                candidates[record["_record_hash"]] = record
        for record_hash, record in sorted(candidates.items()):
            reasons = []
            title = record.get("Article_title")
            body = record.get("Article")
            url = record.get("Url")
            record_date = record.get("Date")
            source_symbol = str(record.get("Stock_symbol") or "").upper()
            if not all(isinstance(value, str) and value.strip() for value in (title, body, url, record_date)):
                reasons.append("STRUCTURALLY_INVALID_RECORD")
            if source_symbol and source_symbol != symbol:
                reasons.append("CROSS_SYMBOL_SOURCE_RECORD")
            parsed = urlparse(str(url or ""))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                reasons.append("INVALID_URL")
            title_tokens = _tokens(str(title or ""))
            slug_tokens = _tokens(unquote(parsed.path.replace("-", " ")))
            if title_tokens and len(title_tokens & slug_tokens) / len(title_tokens) < 0.5:
                reasons.append("URL_HEADLINE_MISMATCH")
            # AAPL is a previously frozen systemic headline/body corruption.
            if symbol == "AAPL":
                reasons.append("M1_FROZEN_SOURCE_INTEGRITY_POLICY")
            status = "REJECTED" if reasons else "SAFE"
            if status == "SAFE":
                safe_hashes[symbol].add(record_hash)
            records.append({
                "symbol": symbol,
                "source_member": f"sp500_news/{symbol}.jsonl",
                "source_index": record.get("_source_index"),
                "record_hash": record_hash,
                "date": record_date,
                "title": title,
                "url": url,
                "status": status,
                "rejection_reasons": sorted(set(reasons)),
                "audit_basis": "SOURCE_CONTENT_AND_PROVENANCE_ONLY",
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "candidate_record_count": len(records),
        "safe_record_count": sum(item["status"] == "SAFE" for item in records),
        "unsafe_record_count": sum(item["status"] == "REJECTED" for item in records),
        "outcome_data_used": False,
        "external_replacement": False,
        "automatic_symbol_reselection": False,
        "records": records,
        "safe_hashes": {symbol: sorted(values) for symbol, values in safe_hashes.items()},
    }


def safe_news_data(raw: dict[str, Any] | None, safe_hashes: set[str]) -> dict[str, Any] | None:
    if raw is None:
        return None
    # selected_news deterministically recomputes these same canonical hashes.
    from scripts.finmultitime.design_m1_contract import deduplicate_news
    deduped = deduplicate_news(raw["records"])
    return {
        **raw,
        "records": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in deduped["records"]
            if item["_record_hash"] in safe_hashes
        ],
    }


def text_section(raw: dict[str, Any] | None, symbol: str, decision: date, safe: set[str]) -> dict[str, Any]:
    if symbol == "AAPL":
        return {"status": "UNAVAILABLE", "reason": "M1 frozen AAPL source-integrity policy", "selected_records": [], "same_day_ambiguous_rejected": []}
    selected = selected_news(safe_news_data(raw, safe), decision)
    records = [selected_text_record(item, {}) for item in selected["records"]]
    return {
        "status": selected["status"],
        "reason": selected.get("reason", ""),
        "selected_records": records,
        "selected_record_hashes": [item["record_hash"] for item in records],
        "same_day_ambiguous_rejected": [
            {"record_hash": item["_record_hash"], "date": item.get("Date"), "reason": "AMBIGUOUS_REJECTED"}
            for item in selected.get("ambiguous_records", []) if item["_record_hash"] in safe
        ],
        "latest_safe_date": selected["latest_safe_date"].isoformat() if selected.get("latest_safe_date") else None,
        "source_member": f"sp500_news/{symbol}.jsonl" if raw is not None else None,
    }


def table_section(tables: dict[str, Any], symbol: str, decision: date) -> dict[str, Any]:
    selected = table_selection(tables, symbol, decision)
    return {
        "status": selected["status"],
        "reason": "" if selected["status"] == "AVAILABLE" else "no PIT-safe selected facts",
        "facts": {concept: selected["facts"].get(concept) for concept in SELECTED_TABLE_CONCEPTS},
        "unavailable_concepts": selected["unavailable_concepts"],
        "same_day_ambiguous_rejected_count": selected["same_day_count"],
        "latest_safe_filed_date": selected["latest_safe"].isoformat() if selected["latest_safe"] else None,
        "source_members": sorted({fact["source_member"] for fact in selected["facts"].values() if fact}),
    }


def time_series_section(series: dict[str, Any], symbol: str, decision: date) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = time_series_selection(series, symbol, decision)
    rows = selected["rows"]
    section = {
        "status": selected["status"],
        "reason": "" if selected["status"] == "AVAILABLE" else "fewer than 61 completed sessions",
        "through_session": selected["latest"].isoformat() if selected["latest"] else None,
        "selected_row_count": len(rows),
        "required_row_count": MIN_TIME_SERIES_ROWS,
        "selected_session_dates": [item["session_date"].isoformat() for item in rows],
        "summary": ts_summary(selected),
        "source_member": f"S&P500_time_series/{symbol.lower()}.csv",
    }
    return section, rows


def image_section(images: dict[str, Any], symbol: str, decision: date) -> dict[str, Any]:
    if symbol == "AAPL":
        return {"status": "UNAVAILABLE", "reason": "M1 frozen AAPL IMAGE policy", "caption_status": "NOT_APPLICABLE"}
    selected = image_selection(images, symbol, decision)
    item = selected.get("item")
    if item is None:
        return {"status": "UNAVAILABLE", "reason": selected["reason"], "caption_status": "NOT_APPLICABLE"}
    return {
        "status": "AVAILABLE",
        "reason": selected["reason"],
        "source_path": item["path"],
        "filename": item["filename"],
        "image_sha256": item["sha256"],
        "bytes": item["bytes"],
        "inferred_period": f"{item['period_start'].year}-H{1 if item['period_start'].month == 1 else 2}",
        "inferred_period_end": item["period_end"].isoformat(),
        "caption_status": "PENDING",
    }


def _m1_caption_map(m1_archive: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(m1_archive.glob("**/caption_manifest.json")):
        value = read_json(path)
        items = value.get("captions", value.get("images", [])) if isinstance(value, dict) else []
        if isinstance(items, dict):
            items = list(items.values())
        for item in items:
            image_sha = item.get("image_sha256") or item.get("sha256")
            caption = item.get("canonical_caption")
            if not caption and item.get("caption_ref"):
                candidates = [
                    path.parent / item["caption_ref"],
                    path.parent.parent / item["caption_ref"],
                ]
                caption_path = next((candidate for candidate in candidates if candidate.is_file()), None)
                if caption_path:
                    artifact = read_json(caption_path)
                    item = {**item, **artifact}
                    caption = artifact.get("canonical_caption")
            if image_sha and caption:
                result[image_sha] = item
    for path in sorted(m1_archive.glob("**/*.json")):
        if "evidence_packets" not in path.parts:
            continue
        value = read_json(path)
        image = value.get("IMAGE", {}) if isinstance(value, dict) else {}
        if image.get("status") == "AVAILABLE" and image.get("image_sha256") and image.get("canonical_caption"):
            result.setdefault(image["image_sha256"], {
                **image,
                "canonical_caption": json.dumps(image["canonical_caption"], ensure_ascii=False, separators=(",", ":")),
                "canonical_caption_sha256": image.get("caption_sha256"),
            })
    return result


def build_skeletons(data: dict[str, Any], plan: list[dict[str, Any]], text_audit: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    safe = {symbol: set(values) for symbol, values in text_audit["safe_hashes"].items()}
    cases = []
    union_rows: dict[str, dict[str, dict[str, Any]]] = {symbol: {} for symbol in SYMBOLS}
    for planned in plan:
        symbol = planned["symbol"]
        decision = date.fromisoformat(planned["decision_session"])
        ts, rows = time_series_section(data["series"], symbol, decision)
        for row in rows:
            union_rows[symbol][row["session_date"].isoformat()] = row
        sections = {
            "TEXT": text_section(data["news"][symbol], symbol, decision, safe[symbol]),
            "TABLE": table_section(data["tables"], symbol, decision),
            "TIME_SERIES": ts,
            "IMAGE": image_section(data["images"], symbol, decision),
        }
        cases.append({
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "dataset_id": DATASET_ID,
            "case_id": planned["case_id"],
            "symbol": symbol,
            "decision_session": planned["decision_session"],
            "decision_time": planned["decision_time"],
            "role": planned["split_role"],
            "tier_membership": {"in_compact": planned["in_compact"], "in_standard": planned["in_standard"], "in_maximum": True},
            "protected": planned["protected"],
            "available_for_training": planned["available_for_training"],
            "available_for_validation": planned["available_for_validation"],
            "engineering_only": planned["engineering_only"],
            "performance_for_selection": False,
            "pit_cutoff": planned["decision_session"],
            "routing": {"modality_to_analyst": ROUTING, "social_analyst": "NO_FINMULTITIME_SPECIFIC_EVIDENCE"},
            "sections": sections,
            "final_packet_status": "PENDING_QWEN" if sections["IMAGE"]["status"] == "AVAILABLE" else "READY_WITHOUT_IMAGE",
        })
    normalized_union = {
        symbol: [union_rows[symbol][key] for key in sorted(union_rows[symbol])]
        for symbol in SYMBOLS
    }
    return cases, normalized_union


def unique_image_manifest(cases: list[dict[str, Any]], m1_archive: Path) -> dict[str, Any]:
    refs: dict[str, list[str]] = defaultdict(list)
    images = {}
    for case in cases:
        image = case["sections"]["IMAGE"]
        if image["status"] == "AVAILABLE":
            refs[image["image_sha256"]].append(case["case_id"])
            images[image["image_sha256"]] = image
    frozen = _m1_caption_map(m1_archive)
    rows = []
    for index, image_sha in enumerate(sorted(images), 1):
        item = images[image_sha]
        reused = frozen.get(image_sha)
        rows.append({
            "image_id": f"M2IMG{index:03d}",
            "source_member": item["source_path"],
            "image_filename": item["filename"],
            "image_sha256": image_sha,
            "inferred_period": item["inferred_period"],
            "inferred_period_end": item["inferred_period_end"],
            "referenced_case_count": len(refs[image_sha]),
            "cases": sorted(refs[image_sha]),
            "symbols": sorted({case.split(":", 1)[0] for case in refs[image_sha]}),
            "qwen_status": "REUSED_FROZEN" if reused else "PENDING",
            "caption_source": "REUSED_FROZEN_M1" if reused else None,
            "reused_caption": reused,
        })
    return {"schema_version": SCHEMA_VERSION, "order": "ascending image_sha256", "images": rows}


def _serialize_source_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_date": row["session_date"].isoformat(),
        "raw": row.get("raw"),
        "numeric": row.get("numeric"),
    }


def write_pre_qwen_archive(output: Path, raw_root: Path, plan_path: Path, protocol_path: Path, m1_archive: Path) -> dict[str, Any]:
    contract = enforce_frozen_contract()
    if contract["packet_version"] != CONTRACT_VERSION:
        raise EvidenceBuildError("M1 contract version drift")
    data, provenance = load_raw_members(raw_root)
    plan = load_plan(plan_path, protocol_path)
    audit = audit_text_candidates(data, plan)
    cases, union_rows = build_skeletons(data, plan, audit)
    image_manifest = unique_image_manifest(cases, m1_archive)
    # Bind actually touched image bytes into provenance.
    for item in image_manifest["images"]:
        path = Path(item["source_member"])
        provenance["records"].append({
            "symbol": item["symbols"][0], "modality": "IMAGE", "available": True,
            "source": {"source_kind": "file", "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)},
        })
    if output.exists():
        raise EvidenceBuildError(f"output already exists: {output}")
    output.mkdir(parents=True)
    write_json(output / "audits/text_source_integrity.json", audit)
    write_json(output / "manifests/source_provenance.json", provenance)
    write_json(output / "manifests/unique_image_manifest.json", image_manifest)
    write_json(output / "manifests/tier_membership.json", {
        "budget_tier_selected": False,
        "status": "DEFERRED TO M2-07",
        "tiers": {tier: {"case_count": count, "canonical_case_list_sha256": sha} for tier, (count, sha) in TIER_IDENTITIES.items()},
        "cases": [{"case_id": row["case_id"], **row["tier_membership"]} for row in cases],
    })
    for case in cases:
        write_json(output / "inputs/pre_qwen_packet_skeletons" / case["role"].lower() / f"{case['case_id'].replace(':', '_')}.json", case)
    # Archive complete selected source subsets, never outcomes.
    selected_text = {}
    selected_facts = {}
    for case in cases:
        for record in case["sections"]["TEXT"]["selected_records"]:
            selected_text[(case["symbol"], record["record_hash"])] = {"symbol": case["symbol"], **record}
        for concept, fact in case["sections"]["TABLE"]["facts"].items():
            if fact:
                selected_facts[(case["symbol"], concept, fact["source_provenance_hash"])] = {"symbol": case["symbol"], **fact}
    text_path = output / "inputs/source_extracts/text_selected_records.jsonl"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("".join(json.dumps(selected_text[key], sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for key in sorted(selected_text)), encoding="utf-8")
    table_path = output / "inputs/source_extracts/table_selected_facts.jsonl"
    table_path.write_text("".join(json.dumps(selected_facts[key], sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for key in sorted(selected_facts)), encoding="utf-8")
    for symbol, rows in union_rows.items():
        write_json(output / f"inputs/source_extracts/time_series/{symbol}.json", {
            "symbol": symbol,
            "source_member": f"S&P500_time_series/{symbol.lower()}.csv",
            "rows": [_serialize_source_row(row) for row in rows],
        })
    for item in image_manifest["images"]:
        source = Path(item["source_member"])
        target = output / "inputs/images" / f"{item['image_sha256']}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != item["image_sha256"]:
            raise EvidenceBuildError("staged image hash mismatch")
        item["staged_path"] = str(target.relative_to(output))
    write_json(output / "manifests/unique_image_manifest.json", image_manifest)
    case_index = output / "inputs/case_index.csv"
    with case_index.open("w", newline="", encoding="utf-8") as handle:
        fields = ("case_id", "symbol", "decision_session", "role", "in_compact", "in_standard", "in_maximum", "protected", "engineering_only")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case["case_id"], "symbol": case["symbol"], "decision_session": case["decision_session"], "role": case["role"],
                **case["tier_membership"], "protected": str(case["protected"]).lower(), "engineering_only": str(case["engineering_only"]).lower(),
            })
    mutation = source_mutation_report(provenance)
    write_json(output / "audits/source_mutation_guard.json", mutation)
    validation = validate_pre_qwen(cases, image_manifest, mutation)
    write_json(output / "audits/pit_audit.json", validation["pit_audit"])
    write_json(output / "manifests/phase_a_sha256.json", file_manifest(output))
    (output / "README.md").write_text(
        "# M2-06 pre-Formal evidence inputs\n\n**PRE-FORMAL DEVELOPMENT — NOT FORMAL M2**\n\n"
        "This archive contains deterministic, outcome-free inputs for 96 frozen 2023 cases. The raw FinMultiTime source remained read-only. Budget tier selection is deferred to M2-07.\n",
        encoding="utf-8",
    )
    return {"cases": len(cases), "images": len(image_manifest["images"]), **validation}


def source_mutation_report(provenance: dict[str, Any]) -> dict[str, Any]:
    differences = []
    checked = 0
    raw_root = Path(provenance["raw_root"])
    for record in provenance["records"]:
        sources = record.get("sources", []) + ([record["source"]] if record.get("source") else [])
        for source in sources:
            if source["source_kind"] == "absent_zip_member":
                continue
            checked += 1
            if source["source_kind"] == "zip_member":
                actual = source_member_hash(raw_root, source["outer_archive"], source["member"])["sha256"]
            else:
                actual = sha256_file(Path(source["path"]))
            if actual != source["sha256"]:
                differences.append({"source": source, "actual_sha256": actual})
    return {"schema_version": SCHEMA_VERSION, "members_checked": checked, "differences": differences, "raw_dataset_mutation": "NO" if not differences else "YES", "status": "PASS" if not differences else "FAIL"}


def _pit_violations(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for case in cases:
        decision = date.fromisoformat(case["decision_session"])
        sections = case["sections"]
        for record in sections["TEXT"]["selected_records"]:
            if date.fromisoformat(record["date"]) >= decision:
                violations.append({"case_id": case["case_id"], "modality": "TEXT"})
        for fact in sections["TABLE"]["facts"].values():
            if fact and date.fromisoformat(fact["filed_date"]) >= decision:
                violations.append({"case_id": case["case_id"], "modality": "TABLE"})
        if any(date.fromisoformat(value) > decision for value in sections["TIME_SERIES"]["selected_session_dates"]):
            violations.append({"case_id": case["case_id"], "modality": "TIME_SERIES"})
        image = sections["IMAGE"]
        if image["status"] == "AVAILABLE" and date.fromisoformat(image["inferred_period_end"]) >= decision:
            violations.append({"case_id": case["case_id"], "modality": "IMAGE"})
    return violations


def validate_pre_qwen(cases: list[dict[str, Any]], images: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    if len(cases) != 96 or Counter(case["role"] for case in cases) != EXPECTED_ROLE_COUNTS:
        raise EvidenceBuildError("pre-Qwen population mismatch")
    if any(case["decision_session"] >= "2024-01-01" for case in cases):
        raise EvidenceBuildError("2024 case leakage")
    if mutation["status"] != "PASS":
        raise EvidenceBuildError("raw dataset mutation detected")
    violations = _pit_violations(cases)
    if violations:
        raise EvidenceBuildError("PIT violation detected")
    if any(item["qwen_status"] not in {"PENDING", "REUSED_FROZEN"} for item in images["images"]):
        raise EvidenceBuildError("invalid image scheduling status")
    assert_no_future_fields(cases)
    return {"pit_audit": {"violations": violations, "pit_violations": 0, "status": "PASS"}, "pre_qwen_status": "PASS"}


def load_caption_bundle(caption_root: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(caption_root / "caption_manifest.json")
    result = {}
    for item in manifest["captions"]:
        canonical = item["canonical_caption"]
        if sha256_bytes(canonical.encode()) != item["canonical_caption_sha256"]:
            raise EvidenceBuildError("canonical caption identity mismatch")
        if parse_canonical_caption(canonical) != canonical:
            raise EvidenceBuildError("caption is not in frozen canonical representation")
        validate_caption_content(canonical)
        if item["model_repo"] != MODEL_REPO or item["model_revision"] != MODEL_REVISION:
            raise EvidenceBuildError("Qwen model identity mismatch")
        result[item["image_sha256"]] = item
    return result


def _packet_sections(skeleton: dict[str, Any], captions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = json.loads(json.dumps(skeleton["sections"]))
    decision = date.fromisoformat(skeleton["decision_session"])
    sections = {
        "TEXT": validate_text_source({"TEXT": source["TEXT"]}, decision),
        "TABLE": validate_table_source({"TABLE": source["TABLE"]}, decision),
        "TIME_SERIES": validate_time_series_source({"TIME_SERIES": source["TIME_SERIES"]}, decision),
    }
    image_source = source["IMAGE"]
    if image_source["status"] == "AVAILABLE":
        caption = captions.get(image_source["image_sha256"])
        if caption is None:
            raise EvidenceBuildError(f"missing caption: {image_source['image_sha256']}")
        sections["IMAGE"] = {
            "status": "AVAILABLE",
            "caption_status": "GENERATED",
            "chart_period": image_source["inferred_period"],
            "inferred_period_end": image_source["inferred_period_end"],
            "evidence_age_calendar_days": (decision - date.fromisoformat(image_source["inferred_period_end"])).days,
            "source_identity": image_source["filename"],
            "image_sha256": image_source["image_sha256"],
            "canonical_caption": json.loads(caption["canonical_caption"]),
            "caption_sha256": caption["canonical_caption_sha256"],
            "provenance_hash_reference": caption["canonical_caption_sha256"],
            "caption_source": caption["caption_source"],
            "reason": "",
        }
    else:
        sections["IMAGE"] = {
            "status": "UNAVAILABLE", "caption_status": "NOT_APPLICABLE",
            "chart_period": None, "inferred_period_end": None,
            "evidence_age_calendar_days": None, "source_identity": "",
            "image_sha256": None, "canonical_caption": None,
            "caption_sha256": None, "provenance_hash_reference": "",
            "reason": image_source.get("reason", "no eligible completed image"),
        }
    return sections


def build_packet(skeleton: dict[str, Any], captions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    sections = _packet_sections(skeleton, captions)
    rendered = format_agent_text(skeleton["symbol"], skeleton["decision_session"], sections)
    routed = {
        "news_analyst": format_text_section(sections["TEXT"]),
        "fundamentals_analyst": format_table_section(sections["TABLE"]),
        "market_analyst": "\n\n".join((format_time_series_section(sections["TIME_SERIES"]), format_image_section(sections["IMAGE"]))),
        "social_analyst": "",
    }
    packet = {
        "record_type": "m1_finmultitime_evidence_packet",
        "contract_version": CONTRACT_VERSION,
        "dataset_id": DATASET_ID,
        "case_id": skeleton["case_id"],
        "symbol": skeleton["symbol"],
        "decision_session": skeleton["decision_session"],
        "decision_time": skeleton["decision_time"],
        "packet_status": "FINAL_FROZEN",
        "source_provenance": {"task_id": TASK_ID, "pit_cutoff": skeleton["pit_cutoff"]},
        "role": skeleton["role"],
        "tier_membership": skeleton["tier_membership"],
        "protected": skeleton["protected"],
        "available_for_training": skeleton["available_for_training"],
        "available_for_validation": skeleton["available_for_validation"],
        "engineering_only": skeleton["engineering_only"],
        "performance_for_selection": False,
        "m2_metadata": {
            "role": skeleton["role"], "tier_membership": skeleton["tier_membership"],
            "protected": skeleton["protected"],
            "available_for_training": skeleton["available_for_training"],
            "available_for_validation": skeleton["available_for_validation"],
            "engineering_only": skeleton["engineering_only"],
            "performance_for_selection": False,
        },
        "TEXT": sections["TEXT"], "TABLE": sections["TABLE"], "TIME_SERIES": sections["TIME_SERIES"], "IMAGE": sections["IMAGE"],
        "routing": {"modality_to_analyst": ROUTING, "social_analyst": "NO_FINMULTITIME_EVIDENCE", "raw_packet_direct_injection": False},
        "routed_projections": {
            "news_analyst": {"sections": ["TEXT"], "text": routed["news_analyst"]},
            "fundamentals_analyst": {"sections": ["TABLE"], "text": routed["fundamentals_analyst"]},
            "market_analyst": {"sections": ["TIME_SERIES", "IMAGE"], "text": routed["market_analyst"]},
            "social_analyst": {"sections": [], "text": ""},
        },
        "character_counts": {"complete": len(rendered), "news": len(routed["news_analyst"]), "fundamentals": len(routed["fundamentals_analyst"]), "market": len(routed["market_analyst"])},
        "agent_text": rendered,
        "section_sha256": {
            name: sha256_bytes(canonical_json(sections[name]).encode()) for name in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")
        },
    }
    packet["packet_sha256"] = sha256_bytes(canonical_bytes(packet))
    validate_packet(packet, date.fromisoformat(packet["decision_session"]))
    sizes = {
        "TEXT": len(format_text_section(packet["TEXT"])),
        "TABLE": len(format_table_section(packet["TABLE"])),
        "TIME_SERIES": len(format_time_series_section(packet["TIME_SERIES"])),
        "IMAGE": len(format_image_section(packet["IMAGE"])),
        "TOTAL": len(rendered),
    }
    limits = {"TEXT": MAX_NEWS_SECTION_CHARS, "TABLE": MAX_TABLE_SECTION_CHARS, "TIME_SERIES": MAX_TIME_SERIES_SECTION_CHARS, "IMAGE": MAX_IMAGE_SECTION_CHARS, "TOTAL": MAX_PACKET_CHARS}
    if any(sizes[key] > limits[key] for key in sizes):
        raise EvidenceBuildError(f"packet bound violation: {packet['case_id']}")
    return packet, rendered


def _walk_forbidden(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).casefold()
            if any(token == lowered or token in lowered for token in FORBIDDEN_KEY_TOKENS):
                found.append(key_path)
            found.extend(_walk_forbidden(child, key_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_forbidden(child, f"{path}[{index}]"))
    return found


def build_final_archive(pre_qwen_root: Path, caption_root: Path, output: Path) -> dict[str, Any]:
    captions = load_caption_bundle(caption_root)
    skeleton_paths = sorted((pre_qwen_root / "inputs/pre_qwen_packet_skeletons").glob("**/*.json"))
    skeletons = [read_json(path) for path in skeleton_paths]
    if len(skeletons) != 96:
        raise EvidenceBuildError("pre-Qwen skeleton count mismatch")
    if output.exists():
        raise EvidenceBuildError(f"output already exists: {output}")
    output.mkdir(parents=True)
    packets = []
    bounds = []
    for skeleton in sorted(skeletons, key=lambda item: item["case_id"]):
        packet, rendered = build_packet(skeleton, captions)
        packets.append(packet)
        role = packet["role"].lower()
        filename = f"{packet['case_id'].replace(':', '_')}.json"
        write_json(output / f"packets/structured/{role}/{filename}", packet)
        rendered_path = output / f"packets/rendered/{role}/{filename.removesuffix('.json')}.txt"
        rendered_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_path.write_text(rendered, encoding="utf-8")
        bounds.append({"case_id": packet["case_id"], "total_chars": len(rendered), "status": "PASS"})
    shutil.copytree(pre_qwen_root / "inputs", output / "inputs")
    shutil.copytree(pre_qwen_root / "audits", output / "audits", dirs_exist_ok=True)
    shutil.copytree(pre_qwen_root / "manifests", output / "manifests", dirs_exist_ok=True)
    shutil.copytree(caption_root, output / "captions")
    pit = _pit_violations([{"case_id": p["case_id"], "decision_session": p["decision_session"], "sections": {name: p[name] for name in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")}} for p in packets])
    forbidden = [{"case_id": packet["case_id"], "paths": _walk_forbidden(packet)} for packet in packets]
    forbidden = [item for item in forbidden if item["paths"]]
    if pit or forbidden:
        raise EvidenceBuildError("final PIT/future-label audit failed")
    corpus_binding = {
        "case_identities": [packet["case_id"] for packet in packets],
        "packet_sha256": {packet["case_id"]: packet["packet_sha256"] for packet in packets},
        "source_provenance_sha256": sha256_file(output / "manifests/source_provenance.json"),
        "text_integrity_sha256": sha256_file(output / "audits/text_source_integrity.json"),
        "image_manifest_sha256": sha256_file(output / "manifests/unique_image_manifest.json"),
        "caption_manifest_sha256": sha256_file(output / "captions/caption_manifest.json"),
        "tier_membership_sha256": sha256_file(output / "manifests/tier_membership.json"),
        "m1_contract_version": CONTRACT_VERSION,
        "m1_contract_sha256": enforce_frozen_contract()["contract_sha256"],
        "split_plan_sha256": sha256_file(DEFAULT_PLAN),
    }
    corpus_sha = sha256_bytes(canonical_bytes(corpus_binding))
    write_json(output / "audits/pit_audit.json", {"pit_violations": 0, "violations": [], "status": "PASS"})
    write_json(output / "audits/future_label_audit.json", {"future_label_violations": 0, "violations": [], "status": "PASS"})
    write_json(output / "audits/packet_validation.json", {"packet_count": 96, "packet_bound_violations": 0, "cases": bounds, "status": "PASS"})
    write_json(output / "manifests/corpus_manifest.json", {"schema_version": SCHEMA_VERSION, "packet_count": 96, "preformal_evidence_corpus_identity_sha256": corpus_sha, "binding": corpus_binding})
    write_json(output / "manifests/final_sha256.json", file_manifest(output))
    (output / "README.md").write_text(
        "# M2-06 frozen pre-Formal evidence corpus\n\n**PRE-FORMAL DEVELOPMENT — NOT FORMAL M2**\n\n"
        "This outcome-free 96-case corpus is the single MAXIMUM superset. COMPACT and STANDARD are immutable subsets. Budget tier selection remains deferred to M2-07.\n",
        encoding="utf-8",
    )
    return {"packets": 96, "pit_violations": 0, "future_label_violations": 0, "corpus_sha256": corpus_sha, "research_hashes": research_hashes(output)}


def file_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"manifests/phase_a_sha256.json", "manifests/final_sha256.json"}:
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": SCHEMA_VERSION, "files": files}


def research_hashes(root: Path) -> dict[str, str]:
    return {item["path"]: item["sha256"] for item in file_manifest(root)["files"] if item["path"] != "README.md"}


def verify_final_determinism(pre_qwen_root: Path, caption_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m2-06-determinism-") as temporary:
        first, second = Path(temporary) / "first", Path(temporary) / "second"
        left = build_final_archive(pre_qwen_root, caption_root, first)
        right = build_final_archive(pre_qwen_root, caption_root, second)
        if left["research_hashes"] != right["research_hashes"] or left["corpus_sha256"] != right["corpus_sha256"]:
            raise EvidenceBuildError("double-generation determinism failed")
        return {"status": "PASS", "corpus_sha256": left["corpus_sha256"], "research_hashes": left["research_hashes"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("pre-qwen")
    pre.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    pre.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    pre.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    pre.add_argument("--m1-archive", type=Path, default=DEFAULT_M1_ARCHIVE)
    pre.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("--pre-qwen-root", type=Path, required=True)
    final.add_argument("--caption-root", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args()
    if args.command == "pre-qwen":
        result = write_pre_qwen_archive(args.output, args.raw_root, args.plan, args.protocol, args.m1_archive)
    else:
        if args.verify_determinism:
            result = verify_final_determinism(args.pre_qwen_root, args.caption_root)
        else:
            result = build_final_archive(args.pre_qwen_root, args.caption_root, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
