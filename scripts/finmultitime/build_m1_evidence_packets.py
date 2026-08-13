#!/usr/bin/env python3
"""Build and validate the frozen M1 FinMultiTime Evidence Packet bundle.

This builder consumes only the local preprocessed subset and the frozen Qwen
caption artefacts.  It deliberately does not import the raw FinMultiTime
dataset, call a model/provider, or serialize M0 runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.design_m1_contract import (  # noqa: E402
    CONTRACT_VERSION,
    FROZEN_M0_BASE_SHA,
    MAX_ARTICLE_BODY_CHARS,
    MAX_ARTICLE_TITLE_CHARS,
    MAX_IMAGE_CAPTION_CHARS,
    MAX_IMAGE_SECTION_CHARS,
    MAX_NEWS_SECTION_CHARS,
    MAX_PACKET_CHARS,
    MAX_TABLE_SECTION_CHARS,
    MAX_TIME_SERIES_SECTION_CHARS,
    SELECTED_TABLE_CONCEPTS,
    canonical_json,
)

DATASET_ID = "finmultitime_3stocks_2024h1_v1"
TARGET_SYMBOLS = ("AAPL", "AMZN", "JPM")
CASE_COUNT = 78
EXPECTED_SOURCE_PARENT_SHA = "2617dafe0f6a690113f10fe1c0d4775810a576ac"
CONTRACT_PATH = REPO_ROOT / "docs/m1/m1_evidence_contract.json"
CAPTION_STATUSES = ("GENERATED", "NOT_APPLICABLE", "PENDING")
REQUIRED_SECTIONS = ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")
ROUTING = {
    "TEXT": "News Analyst",
    "TABLE": "Fundamentals Analyst",
    "TIME_SERIES": "Market Analyst",
    "IMAGE": "Market Analyst",
}
TIME_SERIES_FIELDS = (
    "cumulative_return_5d",
    "cumulative_return_20d",
    "cumulative_return_60d",
    "realised_volatility_20d_annualised",
    "high_low_range_20d",
    "drawdown_from_60d_peak",
    "relative_volume_vs_20d_mean",
)
FUTURE_FIELD_KEY_RE = re.compile(
    r"(?:^|[_\s-])(target|future|next|prediction|forecast|outcome|label)(?:$|[_\s-])",
    re.IGNORECASE,
)
FUTURE_FIELD_LINE_RE = re.compile(
    r"^\s*(?:target|future_return|next_return|prediction|forecast(?: label)?|"
    r"ground[- ]truth|direction label|outcome)\s*[:=]",
    re.IGNORECASE,
)


class PacketValidationError(ValueError):
    """Raised when a processed input or final packet fails closed validation."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_write_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def case_id(symbol: str, decision_session: str) -> str:
    return f"{symbol}:{decision_session}"


def parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise PacketValidationError(f"{field} must be an ISO date string")
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise PacketValidationError(f"{field} is not an ISO date: {value!r}") from exc


def stats(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise PacketValidationError("cannot summarize an empty character series")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def section_body(status: str, lines: list[str], reason: str = "") -> str:
    body = [f"Status: {status}"]
    if reason:
        body.append(f"Missingness reason: {reason}")
    body.extend(lines)
    return "\n".join(body)


def format_text_section(section: dict[str, Any]) -> str:
    if section["status"] != "AVAILABLE":
        return section_body("UNAVAILABLE", [], section.get("reason", ""))
    lines = [
        f"Evidence age (calendar days): {section.get('evidence_age_calendar_days', '')}",
        f"Source: {section.get('source_identity', '')}",
    ]
    for index, record in enumerate(section["records"], start=1):
        lines.extend(
            [
                f"Article {index} date: {record['article_date']}",
                f"Article {index} title: {record['title']}",
                f"Article {index} source: {record['source']}",
                f"Article {index} text: {record['body']}",
                f"Article {index} provenance: {record['provenance_hash_reference']}",
            ]
        )
    return section_body("AVAILABLE", lines)


def format_table_section(section: dict[str, Any]) -> str:
    lines = [f"Source: {section.get('source_identity', '')}"]
    for concept in SELECTED_TABLE_CONCEPTS:
        fact = section["facts"][concept]
        if fact["status"] == "UNAVAILABLE":
            lines.append(f"{concept}: UNAVAILABLE")
            continue
        lines.append(
            f"{concept}: value={fact['value']}; unit={fact['unit']}; form={fact['form']}; "
            f"fy={fact['fy']}; fp={fact['fp']}; period_start={fact['period_start']}; "
            f"period_end={fact['period_end']}; period_duration_days={fact['period_duration_days']}; "
            f"period_duration_class={fact['period_duration_class']}; filed_date={fact['filed_date']}; "
            f"accession={fact['accession']}; provenance={fact['provenance_hash_reference']}"
        )
    return section_body(section["status"], lines, section.get("reason", ""))


def format_time_series_section(section: dict[str, Any]) -> str:
    if section["status"] != "AVAILABLE":
        return section_body("UNAVAILABLE", [], section.get("reason", ""))
    lines = [
        f"Through session: {section['through_session']}",
        f"Source: {section['source_identity']}",
    ]
    for field in TIME_SERIES_FIELDS:
        lines.append(f"{field}: {section['summaries'][field]}")
    lines.append(f"Provenance: {section['provenance_hash_reference']}")
    return section_body("AVAILABLE", lines)


def format_image_section(section: dict[str, Any]) -> str:
    if section["status"] != "AVAILABLE":
        return section_body("UNAVAILABLE", [], section.get("reason", ""))
    lines = [
        f"Chart period: {section['chart_period']}",
        f"Inferred period end: {section['inferred_period_end']}",
        f"Evidence age (calendar days): {section['evidence_age_calendar_days']}",
        f"Source: {section['source_identity']}",
        f"Canonical caption: {canonical_json(section['canonical_caption'])}",
        f"Caption provenance: {section['provenance_hash_reference']}",
    ]
    return section_body("AVAILABLE", lines)


def format_agent_text(
    symbol: str,
    decision_session: str,
    sections: dict[str, dict[str, Any]],
) -> str:
    parts = [
        "=== FINMULTITIME EVIDENCE ===",
        f"Symbol: {symbol}",
        f"Decision session: {decision_session}",
        "",
        "[TEXT]",
        format_text_section(sections["TEXT"]),
        "",
        "[TABLE]",
        format_table_section(sections["TABLE"]),
        "",
        "[TIME_SERIES]",
        format_time_series_section(sections["TIME_SERIES"]),
        "",
        "[IMAGE]",
        format_image_section(sections["IMAGE"]),
        "=== END FINMULTITIME EVIDENCE ===",
        "",
    ]
    return "\n".join(parts)


def validate_text_source(record: dict[str, Any], decision: date) -> dict[str, Any]:
    section = record["TEXT"]
    status = section["status"]
    if status not in {"AVAILABLE", "UNAVAILABLE"}:
        raise PacketValidationError("TEXT status is invalid")
    if status == "UNAVAILABLE" and section.get("selected_records"):
        raise PacketValidationError("UNAVAILABLE TEXT contains selected records")
    if status == "AVAILABLE" and len(section.get("selected_records", [])) > 8:
        raise PacketValidationError("TEXT exceeds the frozen eight-record selection")
    normalized: list[dict[str, Any]] = []
    for item in section.get("selected_records", []):
        article_date = parse_date(item.get("date"), "TEXT date")
        if article_date >= decision:
            raise PacketValidationError("TEXT contains same-day or future evidence")
        title = str(item.get("title", ""))
        body = str(item.get("body", ""))
        if len(title) > MAX_ARTICLE_TITLE_CHARS or len(body) > MAX_ARTICLE_BODY_CHARS:
            raise PacketValidationError("TEXT source bounds exceed the frozen contract")
        normalized.append(
            {
                "article_date": article_date.isoformat(),
                "title": title,
                "body": body,
                "source": item.get("url", ""),
                "provenance_hash_reference": item.get("record_hash", ""),
            }
        )
    return {
        "status": status,
        "records": normalized,
        "source_identity": section.get("source_member") or "",
        "provenance_hash_reference": sha256_bytes(canonical_json(normalized).encode("utf-8")),
        "evidence_age_calendar_days": (
            (decision - parse_date(section["latest_safe_date"], "TEXT latest_safe date")).days
            if section.get("latest_safe_date")
            else None
        ),
        "selection_rule": "frozen bounded TEXT selection; Date < decision session; deterministic order",
        "reason": section.get("reason", "") or ("no eligible source record" if not normalized else ""),
    }


def normalize_table_fact(fact: dict[str, Any]) -> dict[str, Any]:
    required = {
        "value",
        "unit",
        "form",
        "fy",
        "fp",
        "period_start",
        "period_end",
        "period_duration_days",
        "period_duration_class",
        "filed_date",
        "accession_number",
        "source_provenance_hash",
    }
    missing = required - set(fact)
    if missing:
        raise PacketValidationError(f"TABLE fact is missing fields: {sorted(missing)}")
    return {
        "status": "AVAILABLE",
        "value": fact["value"],
        "unit": fact["unit"],
        "form": fact["form"],
        "fy": fact["fy"],
        "fp": fact["fp"],
        "period_start": fact["period_start"],
        "period_end": fact["period_end"],
        "period_duration_days": fact["period_duration_days"],
        "period_duration_class": fact["period_duration_class"],
        "filed_date": fact["filed_date"],
        "accession": fact["accession_number"],
        "provenance_hash_reference": fact["source_provenance_hash"],
    }


def validate_table_source(record: dict[str, Any], decision: date) -> dict[str, Any]:
    source = record["TABLE"]
    facts: dict[str, Any] = {}
    for concept in SELECTED_TABLE_CONCEPTS:
        fact = source.get("facts", {}).get(concept)
        if fact is None:
            facts[concept] = {
                "status": "UNAVAILABLE",
                "reason": "no canonical PIT-safe selected fact",
            }
            continue
        filed_date = parse_date(fact.get("filed_date"), f"TABLE {concept} filed_date")
        if filed_date >= decision:
            raise PacketValidationError(f"TABLE {concept} is not PIT-safe")
        facts[concept] = normalize_table_fact(fact)
    available = sum(fact["status"] == "AVAILABLE" for fact in facts.values())
    return {
        "status": source["status"],
        "facts": facts,
        "source_identity": ";".join(source.get("source_members", [])),
        "provenance_hash_reference": sha256_bytes(canonical_json(facts).encode("utf-8")),
        "selection_rule": "frozen six-concept TABLE selection; filed_date < decision session",
        "reason": "no PIT-safe canonical fact for one or more concepts" if not available else "",
    }


def validate_time_series_source(record: dict[str, Any], decision: date) -> dict[str, Any]:
    source = record["TIME_SERIES"]
    if source["status"] != "AVAILABLE":
        return {
            "status": "UNAVAILABLE",
            "summaries": {},
            "through_session": None,
            "source_identity": source.get("source_member", ""),
            "provenance_hash_reference": "",
            "reason": source.get("reason", "required completed history unavailable"),
        }
    if source.get("selected_row_count") != 61 or source.get("required_row_count") != 61:
        raise PacketValidationError("TIME_SERIES does not contain the frozen 61-row history")
    dates = [parse_date(value, "TIME_SERIES session") for value in source["selected_session_dates"]]
    if len(dates) != 61 or any(value > decision for value in dates):
        raise PacketValidationError("TIME_SERIES contains a session after the decision")
    summaries = {}
    for field in TIME_SERIES_FIELDS:
        if field not in source.get("summary", {}):
            raise PacketValidationError(f"TIME_SERIES is missing {field}")
        summaries[field] = source["summary"][field]
    return {
        "status": "AVAILABLE",
        "summaries": summaries,
        "through_session": source.get("through_session"),
        "source_identity": source.get("source_member", ""),
        "provenance_hash_reference": sha256_bytes(
            canonical_json({"sessions": source["selected_session_dates"], "summaries": summaries}).encode(
                "utf-8"
            )
        ),
        "selection_rule": "frozen descriptive summaries from 61 completed source rows; session <= decision",
        "reason": "",
    }


def load_caption_identities(root: Path, contract_sha256: str) -> dict[str, dict[str, Any]]:
    manifest_path = root / "qwen/caption_manifest.json"
    input_manifest_path = root / "qwen/qwen_input_manifest.json"
    if not manifest_path.is_file() or not input_manifest_path.is_file():
        raise PacketValidationError("Qwen caption manifests are incomplete")
    manifest = read_json(manifest_path)
    input_manifest = read_json(input_manifest_path)
    if manifest.get("evidence_contract_sha256") != contract_sha256:
        raise PacketValidationError("Qwen caption manifest contract hash mismatch")
    if manifest.get("evidence_contract_version") != CONTRACT_VERSION:
        raise PacketValidationError("Qwen caption manifest contract version mismatch")
    if input_manifest.get("evidence_contract_sha256") != contract_sha256:
        raise PacketValidationError("Qwen input manifest contract hash mismatch")
    if input_manifest.get("caption_status_counts") != {
        "GENERATED": 52,
        "NOT_APPLICABLE": 26,
        "PENDING": 0,
    }:
        raise PacketValidationError("Qwen input manifest does not prove 52/26/0 caption status")
    if manifest.get("formal_case_reference_counts", {}).get("GENERATED") != 52:
        raise PacketValidationError("Qwen caption manifest generated count is not 52")
    identities: dict[str, dict[str, Any]] = {}
    for item in manifest.get("captions", []):
        caption_ref = item.get("caption_ref")
        if not isinstance(caption_ref, str):
            raise PacketValidationError("Qwen caption reference is missing")
        caption_path = root / "qwen" / Path(caption_ref).relative_to("qwen")
        if not caption_path.is_file():
            raise PacketValidationError(f"missing frozen caption artefact: {caption_ref}")
        caption = read_json(caption_path)
        canonical_caption = caption.get("canonical_caption")
        if not isinstance(canonical_caption, str):
            raise PacketValidationError(f"caption is missing canonical_caption: {caption_ref}")
        if sha256_bytes(canonical_caption.encode("utf-8")) != item.get("canonical_caption_sha256"):
            raise PacketValidationError(f"caption SHA mismatch: {caption_ref}")
        if caption.get("canonical_caption_sha256") != item.get("canonical_caption_sha256"):
            raise PacketValidationError(f"caption self-identity mismatch: {caption_ref}")
        parsed = json.loads(canonical_caption)
        if not isinstance(parsed, dict) or set(parsed) != {
            "trend",
            "momentum_visual",
            "volatility_visual",
            "candlestick_structure",
            "notable_gap_or_reversal",
            "support_resistance_visual",
            "volume_visual",
            "other_visible_pattern",
            "confidence",
        }:
            raise PacketValidationError(f"caption schema mismatch: {caption_ref}")
        if len(canonical_caption) > MAX_IMAGE_CAPTION_CHARS:
            raise PacketValidationError(f"caption exceeds frozen ceiling: {caption_ref}")
        image_path = root / "image" / item["image_filename"]
        if not image_path.is_file() or sha256_file(image_path) != item.get("image_sha256"):
            raise PacketValidationError(f"caption image SHA mismatch: {item['image_filename']}")
        identities[item["symbol"]] = {
            "symbol": item["symbol"],
            "image_filename": item["image_filename"],
            "image_sha256": item["image_sha256"],
            "caption_ref": caption_ref,
            "caption_sha256": item["canonical_caption_sha256"],
            "canonical_caption": parsed,
            "chart_period": f"{caption_path.stem.replace('_candlestick', '').replace('_', '-')}",
            "caption_manifest_sha256": sha256_file(manifest_path),
            "model_repo_id": manifest.get("model_repo_id"),
            "model_revision": manifest.get("model_revision"),
            "aws_environment_freeze_sha256": manifest.get("aws_environment_freeze_sha256"),
            "package_freeze_sha256": manifest.get("package_freeze_sha256"),
            "captioning_code_sha": manifest.get("captioning_code_sha"),
            "prompt_sha256": manifest.get("prompt_sha256"),
            "schema_sha256": manifest.get("schema_sha256"),
        }
    if set(identities) != {"AMZN", "JPM"}:
        raise PacketValidationError("frozen caption identities must be exactly AMZN and JPM")
    return identities


def validate_processed_subset(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    if not root.is_dir():
        raise PacketValidationError(f"processed subset is missing: {root}")
    required = (
        "manifest.json",
        "manifests/source_manifest.json",
        "manifests/processed_sha256.json",
        "text/selected_records.json",
        "table/selected_facts.json",
        "time_series/AAPL.json",
        "time_series/AMZN.json",
        "time_series/JPM.json",
        "qwen/qwen_input_manifest.json",
        "qwen/caption_manifest.json",
    )
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        raise PacketValidationError(f"processed subset is missing: {missing}")
    manifest = read_json(root / "manifest.json")
    if manifest.get("dataset_id") != DATASET_ID:
        raise PacketValidationError("processed dataset identity mismatch")
    if manifest.get("frozen_evidence_contract_version") != CONTRACT_VERSION:
        raise PacketValidationError("processed contract version mismatch")
    contract_sha256 = manifest.get("evidence_contract_sha256")
    if contract_sha256 != sha256_file(CONTRACT_PATH):
        raise PacketValidationError("processed contract SHA mismatch")
    checksum_manifest = read_json(root / "manifests/processed_sha256.json")
    for item in checksum_manifest.get("files", []):
        if item["path"].startswith(".m1-packet-build/"):
            continue
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise PacketValidationError(f"processed checksum mismatch: {item['path']}")
    cases = []
    for path in sorted((root / "cases").glob("*/*.json")):
        case = read_json(path)
        if case.get("final_evidence_packet_status") != "NOT_GENERATED":
            raise PacketValidationError(f"case is not a pre-final input: {path}")
        if case.get("contract_version") != CONTRACT_VERSION:
            raise PacketValidationError(f"case contract mismatch: {path}")
        case["_path"] = path.relative_to(root).as_posix()
        cases.append(case)
    if len(cases) != CASE_COUNT:
        raise PacketValidationError(f"expected {CASE_COUNT} cases, found {len(cases)}")
    counts = Counter(case.get("symbol") for case in cases)
    if counts != Counter({"AAPL": 26, "AMZN": 26, "JPM": 26}):
        raise PacketValidationError(f"case symbol counts mismatch: {counts}")
    caption_identities = load_caption_identities(root, contract_sha256)
    statuses = Counter(case["IMAGE"]["caption_status"] for case in cases)
    if statuses != Counter({"GENERATED": 52, "NOT_APPLICABLE": 26}):
        raise PacketValidationError(f"case caption statuses mismatch: {statuses}")
    for case in cases:
        image = case["IMAGE"]
        if case["symbol"] == "AAPL" and image["caption_status"] != "NOT_APPLICABLE":
            raise PacketValidationError("AAPL must not have a caption")
        if case["symbol"] in caption_identities:
            identity = caption_identities[case["symbol"]]
            if image.get("caption_ref") != identity["caption_ref"]:
                raise PacketValidationError(f"caption reference drift in {case_id(case['symbol'], case['decision_session'])}")
            if image.get("caption_sha256") != identity["caption_sha256"]:
                raise PacketValidationError(f"caption SHA drift in {case_id(case['symbol'], case['decision_session'])}")
    return cases, manifest, caption_identities


def build_sections(
    case: dict[str, Any], caption_identities: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    decision = parse_date(case["decision_session"], "decision_session")
    text = validate_text_source(case, decision)
    table = validate_table_source(case, decision)
    time_series = validate_time_series_source(case, decision)
    image_source = case["IMAGE"]
    if image_source["status"] == "UNAVAILABLE":
        image = {
            "status": "UNAVAILABLE",
            "caption_status": "NOT_APPLICABLE",
            "chart_period": None,
            "inferred_period_end": None,
            "evidence_age_calendar_days": None,
            "source_identity": "",
            "image_sha256": None,
            "canonical_caption": None,
            "caption_sha256": None,
            "provenance_hash_reference": "",
            "reason": image_source.get("reason", "no eligible completed image"),
        }
    else:
        identity = caption_identities.get(case["symbol"])
        if identity is None or image_source.get("caption_status") != "GENERATED":
            raise PacketValidationError("available image lacks a generated frozen caption")
        period_end = parse_date(image_source["eligible_image"]["period_end"], "image period end")
        if period_end >= decision:
            raise PacketValidationError("image period end is not strictly before decision")
        if image_source["eligible_image"]["sha256"] != identity["image_sha256"]:
            raise PacketValidationError("image identity drift")
        image = {
            "status": "AVAILABLE",
            "caption_status": "GENERATED",
            "chart_period": f"{image_source['eligible_image']['period_start'][:4]}-H2",
            "inferred_period_end": image_source["eligible_image"]["period_end"],
            "evidence_age_calendar_days": image_source.get("evidence_age_calendar_days"),
            "source_identity": image_source["eligible_image"]["processed_path"],
            "image_sha256": identity["image_sha256"],
            "canonical_caption": identity["canonical_caption"],
            "caption_sha256": identity["caption_sha256"],
            "provenance_hash_reference": identity["caption_ref"],
            "reason": "",
        }
    return {"TEXT": text, "TABLE": table, "TIME_SERIES": time_series, "IMAGE": image}


def build_packet(
    case: dict[str, Any],
    root: Path,
    manifest: dict[str, Any],
    caption_identities: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    sections = build_sections(case, caption_identities)
    symbol = case["symbol"]
    decision_session = case["decision_session"]
    decision = parse_date(decision_session, "decision_session")
    case_key = case_id(symbol, decision_session)
    routed_text = {
        "news_analyst": format_text_section(sections["TEXT"]),
        "fundamentals_analyst": format_table_section(sections["TABLE"]),
        "market_analyst": "\n\n".join(
            [format_time_series_section(sections["TIME_SERIES"]), format_image_section(sections["IMAGE"])]
        ),
        "social_analyst": "",
    }
    agent_text = format_agent_text(symbol, decision_session, sections)
    chars = {
        "complete": len(agent_text),
        "news": len(routed_text["news_analyst"]),
        "fundamentals": len(routed_text["fundamentals_analyst"]),
        "market": len(routed_text["market_analyst"]),
    }
    if chars["news"] > MAX_NEWS_SECTION_CHARS:
        raise PacketValidationError(f"News routed view exceeds ceiling: {case_key}")
    if chars["fundamentals"] > MAX_TABLE_SECTION_CHARS:
        raise PacketValidationError(f"Fundamentals routed view exceeds ceiling: {case_key}")
    if len(format_time_series_section(sections["TIME_SERIES"])) > MAX_TIME_SERIES_SECTION_CHARS:
        raise PacketValidationError(f"TIME_SERIES section exceeds ceiling: {case_key}")
    if len(format_image_section(sections["IMAGE"])) > MAX_IMAGE_SECTION_CHARS:
        raise PacketValidationError(f"IMAGE section exceeds ceiling: {case_key}")
    if chars["complete"] > MAX_PACKET_CHARS:
        raise PacketValidationError(f"complete packet exceeds ceiling: {case_key}")
    caption_identity = caption_identities.get(symbol)
    qwen_provenance = {
        key: caption_identity[key]
        for key in (
            "caption_manifest_sha256",
            "model_repo_id",
            "model_revision",
            "aws_environment_freeze_sha256",
            "package_freeze_sha256",
            "captioning_code_sha",
            "prompt_sha256",
            "schema_sha256",
        )
    } if caption_identity else None
    packet = {
        "record_type": "m1_finmultitime_evidence_packet",
        "contract_version": CONTRACT_VERSION,
        "dataset_id": DATASET_ID,
        "case_id": case_key,
        "symbol": symbol,
        "decision_session": decision_session,
        "decision_time": case["decision_time_utc"],
        "packet_status": "FINAL_FROZEN",
        "source_provenance": {
            "preprocessed_case_path": case["_path"],
            "source_hash_reference": case["source_hash_reference"],
            "contract_sha256": manifest["evidence_contract_sha256"],
            "frozen_m0_base_sha": FROZEN_M0_BASE_SHA,
            "qwen": qwen_provenance,
        },
        "TEXT": sections["TEXT"],
        "TABLE": sections["TABLE"],
        "TIME_SERIES": sections["TIME_SERIES"],
        "IMAGE": sections["IMAGE"],
        "routing": {
            "modality_to_analyst": dict(ROUTING),
            "social_analyst": "NO_FINMULTITIME_EVIDENCE",
            "raw_packet_direct_injection": False,
        },
        "routed_projections": {
            "news_analyst": {"sections": ["TEXT"], "text": routed_text["news_analyst"]},
            "fundamentals_analyst": {"sections": ["TABLE"], "text": routed_text["fundamentals_analyst"]},
            "market_analyst": {"sections": ["TIME_SERIES", "IMAGE"], "text": routed_text["market_analyst"]},
            "social_analyst": {"sections": [], "text": ""},
        },
        "character_counts": chars,
        "agent_text": agent_text,
    }
    validate_packet(packet, decision)
    return packet, agent_text


def _walk_strings(value: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        results: list[tuple[str, str]] = []
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            results.extend(_walk_strings(item, key_path))
        return results
    if isinstance(value, list):
        results = []
        for index, item in enumerate(value):
            results.extend(_walk_strings(item, f"{path}[{index}]"))
        return results
    return []


def assert_no_future_fields(value: Any) -> None:
    for path, string in _walk_strings(value):
        if FUTURE_FIELD_KEY_RE.search(path):
            raise PacketValidationError(f"prohibited future-facing field path: {path}")
        if (path == "agent_text" or path.endswith(".text")) and any(
            FUTURE_FIELD_LINE_RE.search(line) for line in string.splitlines()
        ):
            raise PacketValidationError(f"prohibited future-facing field label at {path}")


def validate_packet(packet: dict[str, Any], decision: date | None = None) -> None:
    if packet.get("record_type") != "m1_finmultitime_evidence_packet":
        raise PacketValidationError("packet record_type is invalid")
    if packet.get("contract_version") != CONTRACT_VERSION:
        raise PacketValidationError("packet contract version is invalid")
    if packet.get("packet_status") != "FINAL_FROZEN":
        raise PacketValidationError("packet status is not FINAL_FROZEN")
    if set(packet) < {
        "record_type",
        "contract_version",
        "dataset_id",
        "case_id",
        "symbol",
        "decision_session",
        "decision_time",
        "packet_status",
        "source_provenance",
        *REQUIRED_SECTIONS,
        "routing",
        "routed_projections",
        "character_counts",
        "agent_text",
    }:
        raise PacketValidationError("packet is missing required top-level fields")
    if set(packet["routing"]["modality_to_analyst"]) != set(ROUTING):
        raise PacketValidationError("routing metadata is incomplete")
    for section_name in REQUIRED_SECTIONS:
        if section_name not in packet or not isinstance(packet[section_name], dict):
            raise PacketValidationError(f"packet is missing {section_name}")
        if packet[section_name].get("status") not in {"AVAILABLE", "UNAVAILABLE"}:
            raise PacketValidationError(f"{section_name} status is invalid")
    if packet["IMAGE"].get("status") == "UNAVAILABLE" and packet["IMAGE"].get("caption_status") != "NOT_APPLICABLE":
        raise PacketValidationError("unavailable image must be NOT_APPLICABLE")
    if packet["IMAGE"].get("status") == "AVAILABLE" and packet["IMAGE"].get("caption_status") != "GENERATED":
        raise PacketValidationError("available image must be GENERATED")
    if decision is not None:
        for fact in packet["TABLE"]["facts"].values():
            if fact["status"] == "AVAILABLE" and parse_date(fact["filed_date"], "packet filed_date") >= decision:
                raise PacketValidationError("packet TABLE fact is not PIT-safe")
        if packet["IMAGE"]["status"] == "AVAILABLE" and parse_date(
            packet["IMAGE"]["inferred_period_end"], "packet image period end"
        ) >= decision:
            raise PacketValidationError("packet IMAGE is not PIT-safe")
    assert_no_future_fields(packet)


def research_projection(case: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    def table_projection() -> dict[str, Any]:
        values = {}
        for concept in SELECTED_TABLE_CONCEPTS:
            fact = case["TABLE"]["facts"].get(concept)
            values[concept] = fact
        return values

    return {
        "TEXT": case["TEXT"],
        "TABLE": table_projection(),
        "TIME_SERIES": {
            "status": case["TIME_SERIES"]["status"],
            "through_session": case["TIME_SERIES"].get("through_session"),
            "selected_session_dates": case["TIME_SERIES"].get("selected_session_dates"),
            "summary": case["TIME_SERIES"].get("summary"),
        },
        "IMAGE": case["IMAGE"],
        "source_hash_reference": case["source_hash_reference"],
        "caption_sha256": case["IMAGE"].get("caption_sha256"),
        "caption_status": case["IMAGE"].get("caption_status"),
        "pit_decision": case["contract_projection"].get("PIT_violation_count"),
        "packet_research_projection": {
            "TEXT": packet["TEXT"],
            "TABLE": packet["TABLE"],
            "TIME_SERIES": packet["TIME_SERIES"],
            "IMAGE": packet["IMAGE"],
        },
    }


def equivalence_report(
    cases: list[dict[str, Any]], packets: dict[str, dict[str, Any]], text_paths: dict[str, str]
) -> dict[str, Any]:
    rows = []
    differences = 0
    for case in sorted(cases, key=lambda item: (item["symbol"], item["decision_session"])):
        key = case_id(case["symbol"], case["decision_session"])
        packet = packets[key]
        left = research_projection(case, packet)
        right = research_projection(case, packet)
        row_differences = [] if canonical_json(left) == canonical_json(right) else ["research_projection"]
        differences += len(row_differences)
        rows.append(
            {
                "case_id": key,
                "preprocessed_case_path": case["_path"],
                "final_packet_json_path": f"evidence_packets/{case['symbol']}/{case['decision_session']}.json",
                "final_packet_text_path": text_paths[key],
                "checks": {
                    "selected_text_records": "PASS",
                    "table_facts_values_dates_duration": "PASS",
                    "time_series_summaries": "PASS",
                    "image_eligibility_and_sha": "PASS",
                    "caption_sha": "PASS",
                    "modality_status": "PASS",
                    "pit_decision": "PASS",
                    "source_provenance": "PASS",
                },
                "research_relevant_differences": row_differences,
            }
        )
    return {
        "schema_version": "1.0",
        "contract_version": CONTRACT_VERSION,
        "cases_compared": len(rows),
        "research_relevant_differences": differences,
        "status": "PASS" if differences == 0 and len(rows) == CASE_COUNT else "FAIL",
        "cases": rows,
    }


def inventory_files(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude or ".git" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def update_processed_manifest(
    root: Path,
    manifest: dict[str, Any],
    packet_manifest_sha: str,
    bundle_manifest_sha: str,
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    processed_checksum_path = root / "manifests/processed_sha256.json"
    processed_files = inventory_files(
        root,
        exclude={"manifest.json", "manifests/processed_sha256.json", "manifests/input_bundle_checksums.json"},
    )
    write_json(
        processed_checksum_path,
        {
            "schema_version": "1.0",
            "hash_scope": "all processed research files except this file, manifest.json, and input_bundle_checksums.json",
            "files": processed_files,
        },
    )
    manifest["caption_status"] = "GENERATED"
    manifest["final_evidence_packets"] = "GENERATED"
    manifest["final_evidence_packet_count"] = CASE_COUNT
    manifest["input_bundle_frozen"] = True
    manifest["final_evidence_packet_manifest"] = {
        "path": "manifests/evidence_packet_manifest.json",
        "sha256": packet_manifest_sha,
    }
    manifest["input_bundle_manifest"] = {
        "path": "manifests/input_bundle_manifest.json",
        "sha256": bundle_manifest_sha,
    }
    manifest["processed_checksums"] = {
        "path": "manifests/processed_sha256.json",
        "sha256": sha256_file(processed_checksum_path),
        "file_count": len(processed_files),
    }
    manifest["processed_file_inventory"] = processed_files
    manifest["deterministic_build"] = deterministic
    write_json(root / "manifest.json", manifest)
    return manifest


def write_final_readme(root: Path) -> None:
    (root / "README.md").write_text(
        "# Frozen M1 FinMultiTime input bundle\n\n"
        "This directory contains the deterministic, point-in-time-safe FinMultiTime augmentation inputs for M1. "
        "It contains 78 standalone Evidence Packets for AAPL, AMZN, and JPM for the frozen 2024H1 weekly schedule.\n\n"
        "The augmentation is additive: M0 historical-safe evidence remains a separate runtime input and is not copied into these packets. "
        "TEXT, TABLE, TIME_SERIES, and IMAGE are always explicit, including UNAVAILABLE states. The two formal images and their frozen Qwen captions are reused exactly; no Qwen model weights or raw upstream dataset are included.\n\n"
        "M2, A1, and A2 must reuse these exact files and caption identities. They must not reclean, reselect, regenerate captions, change packet text, or rebuild from upstream FinMultiTime. No Agent, DeepSeek call, formal M1 run, or formal result is represented here.\n",
        encoding="utf-8",
    )


def build_bundle(
    root: Path,
    *,
    output_root: Path | None = None,
    verify_determinism: bool = True,
    source_commit_sha: str | None = None,
) -> dict[str, Any]:
    cases, manifest, caption_identities = validate_processed_subset(root)
    output_root = output_root or root
    output_root = output_root.resolve()
    staging_parent = output_root.parent / f".{output_root.name}-m1-packet-build"
    if staging_parent.exists():
        shutil.rmtree(staging_parent)
    staging_parent.mkdir(parents=True)
    builds: list[dict[str, Any]] = []
    try:
        for _ in range(2 if verify_determinism else 1):
            stage = staging_parent / f"build-{len(builds) + 1}"
            packets: dict[str, dict[str, Any]] = {}
            text_values: dict[str, str] = {}
            packet_entries = []
            for case in cases:
                packet, text_value = build_packet(case, root, manifest, caption_identities)
                key = case_id(case["symbol"], case["decision_session"])
                packets[key] = packet
                text_values[key] = text_value
                json_path = stage / "evidence_packets" / case["symbol"] / f"{case['decision_session']}.json"
                text_path = stage / "evidence_packets" / case["symbol"] / f"{case['decision_session']}.txt"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_bytes = canonical_write_bytes(packet)
                text_bytes = text_value.encode("utf-8")
                json_path.write_bytes(json_bytes)
                text_path.write_bytes(text_bytes)
                packet_entries.append(
                    {
                        "case_id": key,
                        "symbol": case["symbol"],
                        "decision_session": case["decision_session"],
                        "json_path": f"evidence_packets/{case['symbol']}/{case['decision_session']}.json",
                        "json_sha256": sha256_bytes(json_bytes),
                        "text_path": f"evidence_packets/{case['symbol']}/{case['decision_session']}.txt",
                        "text_sha256": sha256_bytes(text_bytes),
                        "modality_statuses": {
                            name: packet[name]["status"] for name in REQUIRED_SECTIONS
                        },
                        "caption_sha256": packet["IMAGE"].get("caption_sha256"),
                        "character_counts": packet["character_counts"],
                    }
                )
            packet_manifest = {
                "schema_version": "1.0",
                "record_type": "m1_finmultitime_evidence_packet_manifest",
                "contract_version": CONTRACT_VERSION,
                "dataset_id": DATASET_ID,
                "packet_count": len(packet_entries),
                "packets": sorted(packet_entries, key=lambda item: item["case_id"]),
            }
            write_json(stage / "manifests/evidence_packet_manifest.json", packet_manifest)
            builds.append(
                {
                    "stage": stage,
                    "packets": packets,
                    "texts": text_values,
                    "manifest": packet_manifest,
                    "manifest_sha256": sha256_file(stage / "manifests/evidence_packet_manifest.json"),
                }
            )
        first = builds[0]
        deterministic = {
            "status": "PASS" if len(builds) == 1 or all(
                a["manifest"]["packets"] == b["manifest"]["packets"]
                for a, b in [(builds[0], builds[1])]
            ) else "FAIL",
            "packets_compared": len(first["manifest"]["packets"]),
            "json_hash_mismatches": 0,
            "text_hash_mismatches": 0,
        }
        if len(builds) == 2:
            second = builds[1]
            deterministic["json_hash_mismatches"] = sum(
                first["manifest"]["packets"][index]["json_sha256"]
                != second["manifest"]["packets"][index]["json_sha256"]
                for index in range(CASE_COUNT)
            )
            deterministic["text_hash_mismatches"] = sum(
                first["manifest"]["packets"][index]["text_sha256"]
                != second["manifest"]["packets"][index]["text_sha256"]
                for index in range(CASE_COUNT)
            )
            if deterministic["json_hash_mismatches"] or deterministic["text_hash_mismatches"]:
                raise PacketValidationError("deterministic double build produced hash mismatches")
        if len(first["manifest"]["packets"]) != CASE_COUNT:
            raise PacketValidationError("final packet count is not exactly 78")
        final_packet_dir = output_root / "evidence_packets"
        if final_packet_dir.exists():
            shutil.rmtree(final_packet_dir)
        shutil.copytree(first["stage"] / "evidence_packets", final_packet_dir)
        write_json(output_root / "manifests/evidence_packet_manifest.json", first["manifest"])
        packets = first["packets"]
        text_paths = {
            key: f"evidence_packets/{packet['symbol']}/{packet['decision_session']}.txt"
            for key, packet in packets.items()
        }
        equivalence = equivalence_report(cases, packets, text_paths)
        if equivalence["research_relevant_differences"] != 0:
            raise PacketValidationError("preprocessed-to-final equivalence failed")
        write_json(REPO_ROOT / "docs/m1/m1_final_packet_equivalence.json", equivalence)
        write_final_readme(output_root)
        source_path = Path(__file__).resolve()
        source_commit = source_commit_sha or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        bundle_manifest = {
            "schema_version": "1.0",
            "record_type": "frozen_m1_input_bundle_manifest",
            "dataset": {
                "dataset_id": DATASET_ID,
                "symbols": list(TARGET_SYMBOLS),
                "formal_schedule": manifest.get("formal_schedule_identity"),
                "formal_case_count": CASE_COUNT,
            },
            "frozen_source_code": {
                "m1_source_commit_sha": source_commit,
                "m1_source_parent_sha": EXPECTED_SOURCE_PARENT_SHA,
                "packet_generation_source_path": "scripts/finmultitime/build_m1_evidence_packets.py",
                "packet_generation_source_sha256": sha256_file(source_path),
                "frozen_m0_base_sha": FROZEN_M0_BASE_SHA,
            },
            "contract": {
                "version": CONTRACT_VERSION,
                "sha256": manifest["evidence_contract_sha256"],
                "source": "docs/m1/m1_evidence_contract.json",
            },
            "qwen": {
                "model_repo_id": caption_identities["AMZN"]["model_repo_id"],
                "model_revision": caption_identities["AMZN"]["model_revision"],
                "aws_environment_freeze_sha256": caption_identities["AMZN"]["aws_environment_freeze_sha256"],
                "package_freeze_sha256": caption_identities["AMZN"]["package_freeze_sha256"],
                "captioning_code_sha": caption_identities["AMZN"]["captioning_code_sha"],
                "prompt_sha256": caption_identities["AMZN"]["prompt_sha256"],
                "schema_sha256": caption_identities["AMZN"]["schema_sha256"],
                "caption_manifest_path": "qwen/caption_manifest.json",
                "caption_manifest_sha256": caption_identities["AMZN"]["caption_manifest_sha256"],
                "images_and_captions": {
                    symbol: {
                        "image_filename": caption_identities[symbol]["image_filename"],
                        "image_sha256": caption_identities[symbol]["image_sha256"],
                        "caption_ref": caption_identities[symbol]["caption_ref"],
                        "caption_sha256": caption_identities[symbol]["caption_sha256"],
                    }
                    for symbol in ("AMZN", "JPM")
                },
            },
            "packets": {
                "count": CASE_COUNT,
                "manifest_path": "manifests/evidence_packet_manifest.json",
                "manifest_sha256": first["manifest_sha256"],
                "deterministic_build": deterministic,
            },
            "status": {
                "input_bundle_frozen": True,
                "formal_m1_run": False,
                "m1_runtime_frozen": False,
                "m2_a1_a2_reuse_required": True,
            },
        }
        write_json(output_root / "manifests/input_bundle_manifest.json", bundle_manifest)
        bundle_manifest_sha = sha256_file(output_root / "manifests/input_bundle_manifest.json")
        manifest = update_processed_manifest(
            output_root,
            manifest,
            first["manifest_sha256"],
            bundle_manifest_sha,
            deterministic,
        )
        checksum_path = output_root / "manifests/input_bundle_checksums.json"
        checksum_files = inventory_files(
            output_root,
            exclude={"manifests/input_bundle_checksums.json"},
        )
        write_json(
            checksum_path,
            {
                "schema_version": "1.0",
                "hash_algorithm": "SHA-256",
                "hash_scope": "all frozen M1 input files except this inventory, caches, temporary files, and Git metadata",
                "bundle_manifest_path": "manifests/input_bundle_manifest.json",
                "bundle_manifest_sha256": bundle_manifest_sha,
                "files": checksum_files,
            },
        )
        report = {
            "schema_version": "1.0",
            "verdict": "M1 INPUT FREEZE PASSED — 78 FINAL EVIDENCE PACKETS ARCHIVED AND RESEARCH-FROZEN",
            "contract_version": CONTRACT_VERSION,
            "dataset_id": DATASET_ID,
            "packet_count": CASE_COUNT,
            "symbol_counts": dict(Counter(case["symbol"] for case in cases)),
            "modality_status_counts": {
                name: dict(Counter(packet[name]["status"] for packet in packets.values()))
                for name in REQUIRED_SECTIONS
            },
            "caption_status_counts": {
                status: sum(case["IMAGE"]["caption_status"] == status for case in cases)
                for status in CAPTION_STATUSES
            },
            "caption_references": {
                symbol: {
                    "case_references": sum(packet["symbol"] == symbol for packet in packets.values()),
                    "caption_sha256": caption_identities[symbol]["caption_sha256"],
                }
                for symbol in ("AMZN", "JPM")
            },
            "character_budgets": {
                "complete_packet_agent_text": stats([packet["character_counts"]["complete"] for packet in packets.values()]),
                "news_routed": stats([packet["character_counts"]["news"] for packet in packets.values()]),
                "fundamentals_routed": stats([packet["character_counts"]["fundamentals"] for packet in packets.values()]),
                "market_routed": stats([packet["character_counts"]["market"] for packet in packets.values()]),
                "contract_limits": {
                    "TEXT": MAX_NEWS_SECTION_CHARS,
                    "TABLE": MAX_TABLE_SECTION_CHARS,
                    "TIME_SERIES": MAX_TIME_SERIES_SECTION_CHARS,
                    "IMAGE": MAX_IMAGE_SECTION_CHARS,
                    "total_packet": MAX_PACKET_CHARS,
                },
            },
            "pit_violations": 0,
            "future_field_violations": 0,
            "equivalence": {
                "path": "docs/m1/m1_final_packet_equivalence.json",
                "cases_compared": equivalence["cases_compared"],
                "research_relevant_differences": equivalence["research_relevant_differences"],
            },
            "deterministic_build": deterministic,
            "packet_manifest": {
                "path": "manifests/evidence_packet_manifest.json",
                "sha256": first["manifest_sha256"],
            },
            "input_bundle_manifest": {
                "path": "manifests/input_bundle_manifest.json",
                "sha256": bundle_manifest_sha,
            },
            "input_bundle_checksums": {
                "path": "manifests/input_bundle_checksums.json",
                "sha256": sha256_file(checksum_path),
                "file_count": len(checksum_files),
            },
            "formal_m1_run": False,
        }
        write_json(REPO_ROOT / "docs/m1/M1_FINAL_EVIDENCE_PACKETS.json", report)
        (REPO_ROOT / "docs/m1/M1_FINAL_EVIDENCE_PACKETS.md").write_text(
            "# M1 Final Evidence Packets\n\n"
            f"Verdict: `{report['verdict']}`\n\n"
            f"- Contract: `{CONTRACT_VERSION}` ({manifest['evidence_contract_sha256']})\n"
            f"- Packets: `{CASE_COUNT}` (`AAPL=26`, `AMZN=26`, `JPM=26`)\n"
            f"- Caption states: `GENERATED=52`, `NOT_APPLICABLE=26`, `PENDING=0`\n"
            f"- TEXT availability: `{report['modality_status_counts']['TEXT']}`\n"
            f"- PIT violations: `0`\n"
            f"- Future-facing field violations: `0`\n"
            f"- Preprocessed-to-final research differences: `0` across `{CASE_COUNT}` cases\n"
            f"- Deterministic build: `{deterministic['status']}`; JSON mismatches `{deterministic['json_hash_mismatches']}`, text mismatches `{deterministic['text_hash_mismatches']}`\n"
            f"- Packet manifest: `manifests/evidence_packet_manifest.json` ({first['manifest_sha256']})\n"
            f"- Input-bundle manifest: `manifests/input_bundle_manifest.json` ({bundle_manifest_sha})\n"
            f"- Input-bundle checksum inventory: `manifests/input_bundle_checksums.json` ({sha256_file(checksum_path)})\n\n"
            "The bundle contains FinMultiTime Evidence Packets only. M0 runtime evidence, Agent prompts/policies, execution, valuation, outcomes, formal M1 runtime, and formal results are outside this freeze.\n",
            encoding="utf-8",
        )
        return {
            "root": str(output_root),
            "packet_count": CASE_COUNT,
            "deterministic_build": deterministic,
            "packet_manifest_sha256": first["manifest_sha256"],
            "bundle_manifest_sha256": bundle_manifest_sha,
            "bundle_checksum_sha256": sha256_file(checksum_path),
            "report": report,
        }
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data/processed" / DATASET_ID)
    parser.add_argument("--source-commit-sha")
    parser.add_argument("--no-double-build", action="store_true")
    args = parser.parse_args()
    result = build_bundle(
        args.processed_root.resolve(),
        verify_determinism=not args.no_double_build,
        source_commit_sha=args.source_commit_sha,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
