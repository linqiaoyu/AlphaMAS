#!/usr/bin/env python3
"""Build the deterministic, pre-caption M1 FinMultiTime working subset.

The raw FinMultiTime tree is read through its existing ZIP members and image
files only.  No raw file is extracted, modified, renamed, or deleted.  The
selection rules are imported from ``design_m1_contract`` so preprocessing and
the frozen contract cannot silently drift apart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.audit_finmultitime import (  # noqa: E402
    NEWS_ARCHIVE,
    NEWS_MEMBERS,
    TABLE_ARCHIVE,
    TABLE_MEMBER_RE,
    TARGETS,
    TS_ARCHIVE,
    TS_MEMBERS,
)
from scripts.finmultitime.design_m1_contract import (  # noqa: E402
    CONTRACT_VERSION,
    MAX_ARTICLE_BODY_CHARS,
    MAX_ARTICLE_TITLE_CHARS,
    MIN_TIME_SERIES_ROWS,
    SELECTED_TABLE_CONCEPTS,
    canonical_json,
    case_simulation,
    compact_source_hashes,
    formal_context,
    image_selection,
    load_raw_data,
    selected_news,
    table_selection,
    time_series_selection,
    truncate,
    ts_summary,
)

DATASET_ID = "finmultitime_3stocks_2024h1_v1"
DEFAULT_RAW_ROOT = Path("/Volumes/Jackson/Dataset/FinMultiTime")
DEFAULT_OUTPUT = REPO_ROOT / "data/processed" / DATASET_ID
DEFAULT_CONTRACT_DIR = REPO_ROOT / "docs/m1"
DEFAULT_M0_BASE_SHA = "2535896c8b1070b19c06fa6a936663babb4356f7"
FORMAL_SCHEDULE_ID = "XNYS_2024H1_26W_DECISION_CLOSES_FINAL_2024-07-05"
CASE_SIMULATION_NAME = "m1_evidence_contract_case_simulation.csv"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def enforce_frozen_contract(contract_dir: Path = DEFAULT_CONTRACT_DIR) -> dict[str, Any]:
    """Return the frozen identity or fail closed before reading raw inputs."""
    contract_path = contract_dir / "m1_evidence_contract.json"
    freeze_path = contract_dir / "m1_evidence_contract_freeze.json"
    simulation_path = contract_dir / CASE_SIMULATION_NAME
    if not all(path.is_file() for path in (contract_path, freeze_path, simulation_path)):
        raise ValueError("frozen M1 contract artifacts are incomplete")

    contract = read_json(contract_path)
    freeze = read_json(freeze_path)
    actual_contract_sha = sha256_file(contract_path)
    actual_simulation_sha = sha256_file(simulation_path)
    expected_version = freeze.get("contract_version")
    expected_contract_sha = freeze.get("lineage", {}).get("new_contract_sha256")
    expected_simulation_sha = freeze.get("artifacts", {}).get(
        CASE_SIMULATION_NAME, {}
    ).get("sha256")
    if expected_version != CONTRACT_VERSION or expected_version != contract.get("packet_version"):
        raise ValueError(
            f"M1 contract version mismatch: expected {CONTRACT_VERSION!r}, "
            f"freeze={expected_version!r}, contract={contract.get('packet_version')!r}"
        )
    if actual_contract_sha != expected_contract_sha:
        raise ValueError("M1 contract SHA256 does not match frozen contract identity")
    if expected_simulation_sha and actual_simulation_sha != expected_simulation_sha:
        raise ValueError("M1 case simulation SHA256 does not match frozen identity")
    if freeze.get("status") != "FROZEN":
        raise ValueError("M1 contract freeze is not marked FROZEN")
    if freeze.get("frozen_m0_base_sha") != DEFAULT_M0_BASE_SHA:
        raise ValueError("frozen M0 base SHA does not match the required identity")
    return {
        "packet_version": CONTRACT_VERSION,
        "contract_sha256": actual_contract_sha,
        "simulation_sha256": actual_simulation_sha,
        "frozen_m0_base_sha": DEFAULT_M0_BASE_SHA,
        "freeze_sha256": sha256_file(freeze_path),
        "contract_path": str(contract_path.relative_to(REPO_ROOT)),
        "simulation_path": str(simulation_path.relative_to(REPO_ROOT)),
        "freeze_path": str(freeze_path.relative_to(REPO_ROOT)),
        "expected_images": {
            item["filename"]: {
                "symbol": item["symbol"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in contract.get("modalities", {}).get("IMAGE", {}).get(
                "coverage_summary", {}
            ).get("used_files", [])
        },
    }


def load_frozen_qwen_config(
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
) -> dict[str, Any]:
    """Load the Qwen adapter only after enforcing the frozen contract identity."""
    identity = enforce_frozen_contract(contract_dir)
    adapter = read_json(contract_dir / "m1_evidence_contract.json").get(
        "qwen_image_adapter"
    )
    if not isinstance(adapter, dict):
        raise ValueError("frozen M1 contract lacks qwen_image_adapter")

    required = {
        "model": str,
        "prompt": str,
        "caption_schema": list,
        "max_caption_chars": int,
        "prohibited_content": list,
        "additional_text_context": bool,
        "ticker_or_company_metadata": bool,
        "offline_preprocessing_only": bool,
        "exact_model_revision_frozen": bool,
        "runtime_and_generation_environment_frozen": bool,
    }
    for field, expected_type in required.items():
        if field not in adapter or not isinstance(adapter[field], expected_type):
            raise ValueError(f"invalid frozen qwen_image_adapter field: {field}")
    if not adapter["model"] or not adapter["prompt"]:
        raise ValueError("frozen Qwen model and prompt must be non-empty")
    schema = adapter["caption_schema"]
    prohibited = adapter["prohibited_content"]
    if (
        not schema
        or any(not isinstance(item, str) or not item for item in schema)
        or len(schema) != len(set(schema))
    ):
        raise ValueError("frozen Qwen caption schema must be an ordered unique string list")
    if not prohibited or any(not isinstance(item, str) or not item for item in prohibited):
        raise ValueError("frozen Qwen prohibited-content policy is invalid")
    if adapter["max_caption_chars"] <= 0:
        raise ValueError("frozen Qwen max_caption_chars must be positive")
    if adapter["additional_text_context"] or adapter["ticker_or_company_metadata"]:
        raise ValueError("frozen Qwen adapter must remain image-and-prompt only")
    if not adapter["offline_preprocessing_only"]:
        raise ValueError("frozen Qwen adapter must remain offline-only")
    if adapter["exact_model_revision_frozen"]:
        raise ValueError("Qwen model revision must remain unfrozen in this task")
    if adapter["runtime_and_generation_environment_frozen"]:
        raise ValueError("Qwen runtime must remain unfrozen in this task")

    prompt = adapter["prompt"]
    schema_serialization = canonical_json(schema)
    return {
        "evidence_contract_version": identity["packet_version"],
        "evidence_contract_sha256": identity["contract_sha256"],
        "model": adapter["model"],
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "caption_schema": list(schema),
        "caption_schema_canonical_json": schema_serialization,
        "caption_schema_sha256": sha256_bytes(schema_serialization.encode("utf-8")),
        "max_caption_chars": adapter["max_caption_chars"],
        "prohibited_content": list(prohibited),
        "additional_text_context": adapter["additional_text_context"],
        "ticker_or_company_metadata": adapter["ticker_or_company_metadata"],
        "offline_preprocessing_only": adapter["offline_preprocessing_only"],
        "exact_model_revision": "NOT_FROZEN",
        "runtime_environment": "NOT_FROZEN",
        "generation_settings": "NOT_FROZEN",
    }


def source_member_hash(
    raw_root: Path, archive_relative: str, member: str,
) -> dict[str, Any]:
    archive_path = raw_root / archive_relative
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        digest = hashlib.sha256()
        with archive.open(info) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "source_kind": "zip_member",
        "outer_archive": archive_relative,
        "outer_archive_path": str(archive_path),
        "member": member,
        "member_uncompressed_size": info.file_size,
        "member_compressed_size": info.compress_size,
        "member_crc32": f"{info.CRC:08x}",
        "sha256": digest.hexdigest(),
    }


def source_manifest(raw_root: Path, data: dict[str, Any], image_items: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for symbol in TARGETS:
        if data["news"][symbol] is None:
            records.append({
                "symbol": symbol, "modality": "TEXT",
                "member": NEWS_MEMBERS[symbol], "available": False,
            })
        else:
            records.append({
                "symbol": symbol,
                "modality": "TEXT",
                "member": source_member_hash(raw_root, NEWS_ARCHIVE, NEWS_MEMBERS[symbol]),
                "available": True,
            })
        records.append({
            "symbol": symbol,
            "modality": "TIME_SERIES",
            "member": source_member_hash(raw_root, TS_ARCHIVE, TS_MEMBERS[symbol]),
        })
        with zipfile.ZipFile(raw_root / TABLE_ARCHIVE) as archive:
            table_members = sorted(
                name for name in archive.namelist() if TABLE_MEMBER_RE[symbol].match(name)
            )
        records.append({
            "symbol": symbol,
            "modality": "TABLE",
            "members": [source_member_hash(raw_root, TABLE_ARCHIVE, member) for member in table_members],
        })
    for item in sorted(image_items, key=lambda value: (value["filename"], value["path"])):
        path = Path(item["path"])
        records.append({
            "symbol": next(symbol for symbol in TARGETS if symbol.lower() in path.parts),
            "modality": "IMAGE",
            "file": {
                "source_kind": "file",
                "path": str(path),
                "filename": item["filename"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            },
        })
    return {
        "schema_version": "1.0",
        "raw_source_root": str(raw_root),
        "records": records,
    }


def selected_text_record(record: dict[str, Any], removed_by_kept: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "record_hash": record["_record_hash"],
        "source_index": record["_source_index"],
        "date": record.get("Date"),
        "title": truncate(record.get("Article_title"), MAX_ARTICLE_TITLE_CHARS),
        "url": record.get("Url", ""),
        "body": truncate(record.get("Article"), MAX_ARTICLE_BODY_CHARS),
        "source_stock_symbol": record.get("Stock_symbol"),
        "duplicate_provenance": removed_by_kept.get(record["_record_hash"], []),
    }


def text_case_section(raw: dict[str, Any] | None, symbol: str, decision: date) -> dict[str, Any]:
    selected = selected_news(raw, decision)
    removed_by_kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected.get("dedup", {}).get("removed", []):
        removed_by_kept[item["kept_hash"]].append(json_safe(item))
    records = [selected_text_record(item, removed_by_kept) for item in selected["records"]]
    ambiguous = [
        {
            "record_hash": item.get("_record_hash"),
            "source_index": item.get("_source_index"),
            "date": item.get("Date"),
            "title": truncate(item.get("Article_title"), MAX_ARTICLE_TITLE_CHARS),
            "url": item.get("Url", ""),
            "reason": "same-day Date is not PIT-safe evidence",
        }
        for item in selected.get("ambiguous_records", [])
    ]
    return {
        "status": selected["status"],
        "reason": selected.get("reason", ""),
        "selected_records": records,
        "selected_record_hashes": [item["record_hash"] for item in records],
        "dedup_removed_count": selected.get("dedup_removed_count", 0),
        "same_day_ambiguous_rejected": ambiguous,
        "same_day_ambiguous_rejected_count": len(ambiguous),
        "latest_safe_date": selected.get("latest_safe_date"),
        "source_member": NEWS_MEMBERS[symbol] if raw is not None else None,
    }


def table_case_section(
    tables: dict[str, dict[str, Any] | None], symbol: str, decision: date,
) -> dict[str, Any]:
    selected = table_selection(tables, symbol, decision)
    facts = {
        concept: selected["facts"].get(concept)
        for concept in SELECTED_TABLE_CONCEPTS
    }
    return {
        "status": selected["status"],
        "facts": facts,
        "unavailable_concepts": selected["unavailable_concepts"],
        "selection_diagnostics": selected["diagnostics"],
        "same_day_ambiguous_rejected_count": selected["same_day_count"],
        "latest_safe_filed_date": selected["latest_safe"],
        "source_members": sorted({
            fact["source_member"] for fact in facts.values() if fact is not None
        }),
        "provenance": "filed_date < decision_session_date; selected facts retain source-reported duration",
    }


def ts_case_section(
    series: dict[str, dict[str, Any] | None], symbol: str, decision: date,
) -> dict[str, Any]:
    selected = time_series_selection(series, symbol, decision)
    return {
        "status": selected["status"],
        "through_session": selected.get("latest"),
        "selected_row_count": len(selected.get("rows", [])),
        "required_row_count": MIN_TIME_SERIES_ROWS,
        "selected_session_dates": [row["session_date"] for row in selected.get("rows", [])],
        "summary": ts_summary(selected),
        "source_row_count_through_decision": selected.get("source_row_count_through_decision", 0),
        "source_member": TS_MEMBERS[symbol],
        "provenance": "source-native descriptive data; no OHLC repair or normalisation",
    }


def image_case_section(item: dict[str, Any] | None, decision: date) -> dict[str, Any]:
    if item is None:
        return {
            "status": "UNAVAILABLE",
            "eligible_image": None,
            "evidence_age_calendar_days": None,
            "caption_status": "NOT_APPLICABLE",
            "reason": "no completed image with inferred period end strictly before decision",
        }
    return {
        "status": "AVAILABLE",
        "eligible_image": {
            "filename": item["filename"],
            "source_path": item["path"],
            "processed_path": f"image/{item['filename']}",
            "period_start": item["period_start"],
            "period_end": item["period_end"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        },
        "evidence_age_calendar_days": (decision - item["period_end"]).days,
        "caption_status": "PENDING",
        "reason": "window end inferred from YYYY_H1/H2 filename convention",
    }


def case_id(symbol: str, decision: date) -> str:
    decision_text = decision.isoformat() if isinstance(decision, date) else str(decision)
    return f"{symbol}:{decision_text}"


def build_case_records(
    data: dict[str, Any], context: dict[str, Any], contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    simulation = case_simulation(data, context)
    simulation_by_key = {
        (row["symbol"], row["decision_session"]): json_safe(row) for row in simulation
    }
    records: list[dict[str, Any]] = []
    ts_selections: dict[str, dict[str, Any]] = {}
    for event, decision in zip(context["events"], context["decisions"], strict=True):
        for symbol in TARGETS:
            text = text_case_section(data["news"][symbol], symbol, decision)
            table = table_case_section(data["tables"], symbol, decision)
            ts = ts_case_section(data["series"], symbol, decision)
            image = image_selection(data["images"], symbol, decision)
            image_section = image_case_section(image.get("item"), decision)
            key = (symbol, decision.isoformat())
            projection = simulation_by_key[key]
            raw_case = {
                "record_type": "preprocessed_case_record",
                "contract_version": contract["packet_version"],
                "dataset_id": DATASET_ID,
                "symbol": symbol,
                "decision_session": decision,
                "decision_time_utc": event["decision_close_utc"],
                "decision_close_ny": event["decision_close_ny"],
                "execution_session": event["execution_session"],
                "TEXT": text,
                "TABLE": table,
                "TIME_SERIES": ts,
                "IMAGE": image_section,
                "source_hash_reference": compact_source_hashes(
                    selected_news(data["news"][symbol], decision),
                    table_selection(data["tables"], symbol, decision),
                    time_series_selection(data["series"], symbol, decision),
                    image,
                ),
                "contract_projection": projection,
                "final_evidence_packet_status": "NOT_GENERATED",
            }
            records.append(raw_case)
            ts_selections.setdefault(symbol, {})[decision.isoformat()] = ts
    return records, simulation, ts_selections


def canonical_time_series_subsets(
    data: dict[str, Any], records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for symbol in TARGETS:
        dates: set[str] = set()
        for record in records:
            if record["symbol"] == symbol:
                dates.update(
                    value.isoformat() if isinstance(value, date) else str(value)
                    for value in record["TIME_SERIES"]["selected_session_dates"]
                )
        raw = data["series"][symbol]
        by_date = {
            row["session_date"].isoformat(): row
            for row in (raw["rows"] if raw else [])
            if row.get("session_date")
        }
        selected_rows = [json_safe(by_date[item]) for item in sorted(dates)]
        result[symbol] = {
            "source_member": TS_MEMBERS[symbol],
            "required_completed_rows_per_case": MIN_TIME_SERIES_ROWS,
            "first_session": selected_rows[0]["session_date"] if selected_rows else None,
            "last_session": selected_rows[-1]["session_date"] if selected_rows else None,
            "rows": selected_rows,
            "subset_sha256": sha256_bytes(canonical_json(selected_rows).encode("utf-8")),
        }
    return result


def write_text_catalog(records: list[dict[str, Any]], destination: Path) -> None:
    selected = []
    for record in records:
        for item in record["TEXT"]["selected_records"]:
            selected.append({
                "symbol": record["symbol"],
                "decision_session": record["decision_session"],
                **item,
            })
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in selected:
        unique[(item["symbol"], item["record_hash"])] = item
    write_json(destination, {
        "schema_version": "1.0",
        "representation": "selected bounded source records; no LLM summarisation",
        "records": [unique[key] for key in sorted(unique)],
    })


def write_table_catalog(records: list[dict[str, Any]], destination: Path) -> None:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        for concept, fact in record["TABLE"]["facts"].items():
            if fact is not None:
                unique[(record["symbol"], concept, fact["source_provenance_hash"])] = {
                    "symbol": record["symbol"],
                    **fact,
                }
    write_json(destination, {
        "schema_version": "1.0",
        "concepts": list(SELECTED_TABLE_CONCEPTS),
        "facts": [unique[key] for key in sorted(unique)],
    })


def stage_images(
    destination: Path, images: Iterable[dict[str, Any]], expected: dict[str, Any],
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for item in sorted(images, key=lambda value: value["filename"]):
        source = Path(item["path"])
        expected_item = expected.get(item["filename"])
        if expected_item is None:
            raise ValueError(f"image {item['filename']} is not in the frozen contract")
        actual_sha = sha256_file(source)
        if actual_sha != expected_item["sha256"] or source.stat().st_size != expected_item["bytes"]:
            raise ValueError(f"frozen image identity mismatch for {item['filename']}")
        target = destination / "image" / item["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(target) != actual_sha:
            raise ValueError(f"staged image checksum mismatch for {item['filename']}")
        inventory.append({
            "symbol": next(symbol for symbol in TARGETS if symbol.lower() in source.parts),
            "filename": item["filename"],
            "original_dataset_path": str(source),
            "processed_path": f"image/{item['filename']}",
            "period_start": item["period_start"],
            "period_end": item["period_end"],
            "bytes": item["bytes"],
            "sha256": actual_sha,
        })
    return inventory


def write_qwen_manifest(
    destination: Path,
    records: list[dict[str, Any]],
    images: list[dict[str, Any]],
    qwen: dict[str, Any],
) -> None:
    references = defaultdict(list)
    for record in records:
        image = record["IMAGE"].get("eligible_image")
        if image:
            references[image["filename"]].append(case_id(record["symbol"], record["decision_session"]))
    output = []
    for item in images:
        output.append({
            "symbol": item["symbol"],
            "original_dataset_path": item["original_dataset_path"],
            "copied_processed_path": item["processed_path"],
            "filename": item["filename"],
            "inferred_chart_period": f"{item['period_start'].year}-H{1 if item['period_start'].month == 1 else 2}",
            "inferred_period_end": item["period_end"],
            "byte_size": item["bytes"],
            "sha256": item["sha256"],
            "formal_case_references": sorted(references[item["filename"]]),
            "formal_case_reference_count": len(references[item["filename"]]),
            "caption_status": "PENDING",
        })
    write_json(destination, {
        "schema_version": "1.0",
        "caption_status": "NOT_GENERATED",
        "caption_status_counts": {
            status: sum(
                record["IMAGE"]["caption_status"] == status for record in records
            )
            for status in ("PENDING", "NOT_APPLICABLE", "GENERATED")
        },
        "qwen_runner_must_pass_only_image_and_frozen_visual_prompt": True,
        "qwen_runner_preflight": [
            "verify evidence_contract_sha256",
            "verify prompt_sha256",
            "verify caption_schema_sha256",
        ],
        **qwen,
        "images": output,
    })


def load_simulation_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["symbol"], row["decision_session"]): row for row in csv.DictReader(handle)}


def contract_equivalence(records: list[dict[str, Any]], simulation_path: Path) -> dict[str, Any]:
    expected = load_simulation_rows(simulation_path)
    differences: list[dict[str, Any]] = []
    comparable = (
        "packet_version", "TEXT_status", "selected_news_count", "text_dedup_removed_count",
        "text_same_day_ambiguous_rejected_count", "TABLE_status", "selected_filing_date",
        "selected_table_concept_count", "table_unavailable_concepts", "TIME_SERIES_status",
        "latest_included_session", "time_series_selected_session_count",
        "time_series_required_session_count", "time_series_summary_fields",
        "time_series_summary_values", "IMAGE_status", "selected_image",
        "selected_image_period_start", "selected_image_window_end",
        "image_evidence_age_calendar_days", "source_hash_reference",
        "PIT_rejection_flags", "PIT_violation_count", "ambiguous_rejected_count",
    )
    for record in records:
        decision = record["decision_session"]
        decision_text = decision.isoformat() if isinstance(decision, date) else str(decision)
        key = (record["symbol"], decision_text)
        actual = record["contract_projection"]
        expected_row = expected.get(key)
        if expected_row is None:
            differences.append({"case": case_id(*key), "field": "case_identity", "expected": "present", "actual": "missing"})
            continue
        for field in comparable:
            actual_raw = actual.get(field, "")
            actual_value = "" if actual_raw is None else str(actual_raw)
            expected_value = str(expected_row.get(field, ""))
            if actual_value != expected_value:
                differences.append({
                    "case": case_id(*key), "field": field,
                    "expected": expected_value, "actual": actual_value,
                })
    if len(records) != 78 or len(expected) != 78:
        differences.append({"field": "case_count", "expected": 78, "actual": len(records)})
    return {
        "contract_version": CONTRACT_VERSION,
        "cases_compared": len(records),
        "research_relevant_differences": len(differences),
        "differences": differences,
        "status": "PASS" if not differences else "FAIL",
    }


def inventory_files(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "manifests/processed_sha256.json"}:
            continue
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def research_hashes(root: Path) -> dict[str, str]:
    return {
        item["path"]: item["sha256"]
        for item in inventory_files(root)
        if item["path"] != "README.md"
    }


def validate_build(root: Path, records: list[dict[str, Any]], images: list[dict[str, Any]], equivalence: dict[str, Any]) -> None:
    if len(records) != 78 or len({case_id(r["symbol"], r["decision_session"]) for r in records}) != 78:
        raise ValueError("preprocessing did not produce exactly 78 unique case records")
    if equivalence["research_relevant_differences"] != 0:
        raise ValueError("preprocessing differs from the frozen 78-case simulation")
    if {item["filename"] for item in images} != {"amzn_2023_H2_candlestick.png", "jpm_2023_H2_candlestick.png"}:
        raise ValueError("staged image inventory does not match the frozen two-image subset")
    for record in records:
        if set(record) & {"future_label", "target", "forecast"}:
            raise ValueError(f"future semantic field found in {case_id(record['symbol'], record['decision_session'])}")
        if set(record) != {
            "record_type", "contract_version", "dataset_id", "symbol", "decision_session",
            "decision_time_utc", "decision_close_ny", "execution_session", "TEXT", "TABLE",
            "TIME_SERIES", "IMAGE", "source_hash_reference", "contract_projection",
            "final_evidence_packet_status",
        }:
            raise ValueError("unexpected case record topology")
        expected_caption_status = (
            "PENDING" if record["IMAGE"]["status"] == "AVAILABLE" else "NOT_APPLICABLE"
        )
        if record["IMAGE"]["caption_status"] != expected_caption_status:
            raise ValueError("image caption status does not match image availability")
        if record["final_evidence_packet_status"] != "NOT_GENERATED":
            raise ValueError("final evidence packets must not be generated in Stage 1")
    if not (root / "qwen/qwen_input_manifest.json").is_file():
        raise ValueError("Qwen input manifest is missing")


def _build_to_directory(
    destination: Path, raw_root: Path, contract_dir: Path, m0_snapshot_dir: Path,
) -> dict[str, Any]:
    contract = enforce_frozen_contract(contract_dir)
    qwen = load_frozen_qwen_config(contract_dir)
    context = formal_context()
    if len(context["decisions"]) != 26:
        raise ValueError("formal schedule is not the frozen 26-decision schedule")
    if not raw_root.is_dir():
        raise ValueError(f"raw FinMultiTime root is missing: {raw_root}")
    data = load_raw_data(raw_root)
    records, simulation, _ = build_case_records(data, context, contract)
    unique_image_items = {}
    for record in records:
        image = record["IMAGE"].get("eligible_image")
        if image:
            unique_image_items[image["filename"]] = next(
                item for item in data["images"][record["symbol"]] if item["filename"] == image["filename"]
            )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "cases").mkdir()
    (destination / "manifests").mkdir()
    (destination / "text").mkdir()
    (destination / "table").mkdir()
    (destination / "time_series").mkdir()
    (destination / "qwen").mkdir()
    source_manifest_value = source_manifest(raw_root, data, list(unique_image_items.values()))
    write_json(destination / "manifests/source_manifest.json", source_manifest_value)
    write_text_catalog(records, destination / "text/selected_records.json")
    write_table_catalog(records, destination / "table/selected_facts.json")
    time_series = canonical_time_series_subsets(data, records)
    for symbol, value in time_series.items():
        write_json(destination / f"time_series/{symbol}.json", value)
    staged_images = stage_images(destination, unique_image_items.values(), contract["expected_images"])
    write_qwen_manifest(
        destination / "qwen/qwen_input_manifest.json", records, staged_images, qwen
    )
    for record in sorted(records, key=lambda value: (value["symbol"], value["decision_session"])):
        write_json(
            destination / "cases" / record["symbol"] / f"{record['decision_session']}.json",
            record,
        )
    equivalence = contract_equivalence(records, contract_dir / CASE_SIMULATION_NAME)
    write_json(REPO_ROOT / "docs/m1/m1_preprocessing_contract_equivalence.json", equivalence)
    write_json(destination / "manifests/processed_sha256.json", {
        "schema_version": "1.0",
        "hash_scope": "all processed research files except this file and manifest.json",
        "files": inventory_files(destination),
    })
    source_manifest_sha = sha256_file(destination / "manifests/source_manifest.json")
    processed_sha = sha256_file(destination / "manifests/processed_sha256.json")
    manifest = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "frozen_evidence_contract_version": contract["packet_version"],
        "evidence_contract_sha256": contract["contract_sha256"],
        "evidence_contract_case_simulation_sha256": contract["simulation_sha256"],
        "m1_preprocessing_code_base_sha": git_sha(),
        "m1_preprocessing_script_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_m0_base_sha": DEFAULT_M0_BASE_SHA,
        "target_symbols": list(TARGETS),
        "formal_case_count": len(records),
        "formal_schedule_identity": FORMAL_SCHEDULE_ID,
        "raw_source_root": str(raw_root),
        "source_manifest": {
            "path": "manifests/source_manifest.json",
            "sha256": source_manifest_sha,
        },
        "processed_checksums": {
            "path": "manifests/processed_sha256.json",
            "sha256": processed_sha,
            "file_count": len(inventory_files(destination)),
        },
        "processed_file_inventory": inventory_files(destination),
        "image_inventory": staged_images,
        "modality_availability": {
            symbol: {
                modality: sorted({record[modality]["status"] for record in records if record["symbol"] == symbol})
                for modality in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")
            }
            for symbol in TARGETS
        },
        "caption_status": "NOT_GENERATED",
        "final_evidence_packets": "NOT_GENERATED",
        "input_bundle_frozen": False,
        "source_manifest_identity": sha256_bytes(canonical_json(source_manifest_value).encode("utf-8")),
        "deterministic_equivalence_report": "docs/m1/m1_preprocessing_contract_equivalence.json",
        "m0_snapshot_reference": str(m0_snapshot_dir),
    }
    write_json(destination / "manifest.json", manifest)
    (destination / "README.md").write_text(
        "# M1 preprocessed working subset\n\n"
        "This is an intermediate, local, pre-caption source subset. It is not a final Evidence Packet bundle.\n\n"
        f"Contract: `{CONTRACT_VERSION}`\n\n"
        "The raw FinMultiTime source is read-only. TEXT is original bounded source content; TABLE and TIME_SERIES retain source provenance; IMAGE bytes are copied exactly. Qwen captions and final Evidence Packets are intentionally not generated.\n",
        encoding="utf-8",
    )
    validate_build(destination, records, staged_images, equivalence)
    return {
        "records": len(records),
        "simulation": simulation,
        "images": staged_images,
        "equivalence": equivalence,
        "research_hashes": research_hashes(destination),
    }


def build(
    output_dir: Path = DEFAULT_OUTPUT, raw_root: Path = DEFAULT_RAW_ROOT,
    contract_dir: Path = DEFAULT_CONTRACT_DIR, m0_snapshot_dir: Path | None = None,
    verify_determinism: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    raw_root = raw_root.resolve()
    contract_dir = contract_dir.resolve()
    m0_snapshot_dir = (m0_snapshot_dir or (
        REPO_ROOT / "results/backtests/M0_original_prompt_2024H1/runs/20260812T082530978211Z_2535896c/inputs"
    )).resolve()
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{DATASET_ID}-", dir=parent))
    first = staging_root / "first"
    second = staging_root / "second"
    try:
        first_result = _build_to_directory(first, raw_root, contract_dir, m0_snapshot_dir)
        if verify_determinism:
            second_result = _build_to_directory(second, raw_root, contract_dir, m0_snapshot_dir)
            if first_result["research_hashes"] != second_result["research_hashes"]:
                raise ValueError("deterministic rerun produced different research-relevant hashes")
        backup = output_dir.with_name(output_dir.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            os.replace(output_dir, backup)
        os.replace(first, output_dir)
        if backup.exists():
            shutil.rmtree(backup)
        return {
            "output_dir": str(output_dir),
            "records": first_result["records"],
            "images": first_result["images"],
            "equivalence": first_result["equivalence"],
            "deterministic_rerun": "PASS" if verify_determinism else "NOT_REQUESTED",
        }
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--m0-snapshot-dir", type=Path)
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args()
    result = build(
        output_dir=args.output_dir,
        raw_root=args.raw_root,
        contract_dir=args.contract_dir,
        m0_snapshot_dir=args.m0_snapshot_dir,
        verify_determinism=args.verify_determinism,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
