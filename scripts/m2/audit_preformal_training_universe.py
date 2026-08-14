#!/usr/bin/env python3
"""Freeze the performance-blind M2 pre-Formal universe and temporal split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.audit_finmultitime import (  # noqa: E402
    IMAGE_RE,
    IMAGE_ROOT,
    NEWS_ARCHIVE,
    TABLE_ARCHIVE,
    TS_ARCHIVE,
)
from tradingagents.backtesting.calendar import ExchangeSchedule  # noqa: E402

SCHEMA_VERSION = "1.1"
TASK_ID = "M2-02B"
STARTING_SHA = "cf189c53a3600030911efcd1ceb5afdad1e06765"
PARENT_PROTOCOL_COMMIT = "cf189c53a3600030911efcd1ceb5afdad1e06765"
ARCHITECTURE_SHA = "ea9ca73c15ed94ba3fa37a86e3ee145960d94bc2"
ENVIRONMENT_RECOVERY_SHA = "c1bebcbd73d1d30bfd081e4529181f7163450c49"
FORMAL_SYMBOLS = ("AAPL", "AMZN", "JPM")
SELECTION_VERSION = "M2-PREFORMAL-UNIVERSE-v1"
TIE_BREAK_IDENTITY = "AlphaMAS-M2-preformal-universe-v1"
FORMAL_BOUNDARY = date(2024, 1, 1)
HOLDING_SESSIONS = 5
MIN_HISTORY_SESSIONS = 61
RAW_DESCRIPTION = "sp500stock_data_description.csv"
DEFAULT_RAW_ROOT = Path("/Volumes/Jackson/Dataset/FinMultiTime")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/m2"
PROTOCOL_JSON = "m2_preformal_data_and_split_protocol.json"
SYMBOL_AUDIT_CSV = "m2_preformal_symbol_audit.csv"
CASE_PLAN_CSV = "m2_preformal_semantic_case_plan.csv"
PROTOCOL_MD = "M2_PREFORMAL_DATA_AND_SPLIT_PROTOCOL.md"

ROLE_WEEKS = {
    "TRAIN": (
        "2023-05-01",
        "2023-05-08",
        "2023-05-15",
        "2023-05-22",
        "2023-05-29",
        "2023-06-05",
        "2023-06-12",
    ),
    "VALIDATION": ("2023-06-26", "2023-07-03"),
    "FINAL_HOLDOUT": ("2023-07-17", "2023-07-24"),
    "E2E_PILOT": ("2023-10-02",),
}
ROLE_ORDER = {role: index for index, role in enumerate(ROLE_WEEKS)}
TIER_TRAIN_SESSIONS = {
    "COMPACT": ("2023-05-26", "2023-06-02", "2023-06-09", "2023-06-16"),
    "STANDARD": (
        "2023-05-19",
        "2023-05-26",
        "2023-06-02",
        "2023-06-09",
        "2023-06-16",
    ),
    "MAXIMUM": (
        "2023-05-05",
        "2023-05-12",
        "2023-05-19",
        "2023-05-26",
        "2023-06-02",
        "2023-06-09",
        "2023-06-16",
    ),
}

SYMBOL_AUDIT_FIELDS = (
    "symbol",
    "sector",
    "formal_symbol",
    "text_present",
    "table_present",
    "timeseries_present",
    "image_present",
    "earliest_safe_date",
    "latest_safe_pre2024_decision",
    "eligible_weekly_cases",
    "pre2024_matured_reward_cases",
    "modality_profile",
    "data_integrity_status",
    "structural_invalid_rows",
    "duplicate_session_rows",
    "non_xnys_rows",
    "correctness_eligible",
    "selected_in_any_tier",
    "tie_break_rank",
    "known_source_integrity_limitation",
    "exclusion_reason",
)

CASE_PLAN_FIELDS = (
    "case_id",
    "symbol",
    "decision_session",
    "decision_time",
    "maturity_session",
    "split_role",
    "compact_included",
    "standard_included",
    "maximum_included",
    "formal_symbol",
    "modality_profile",
    "text_status",
    "table_status",
    "timeseries_status",
    "image_status",
    "eligibility_status",
    "source_identity",
    "inclusion_rationale",
)

PERFORMANCE_FIELD_TOKENS = (
    "return",
    "reward",
    "alpha",
    "sharpe",
    "sortino",
    "calmar",
    "volatility",
    "drawdown",
    "momentum",
    "profit",
    "direction",
    "label",
    "outcome",
)
STRUCTURAL_FIELD_EXEMPTIONS = frozenset({"pre2024_matured_reward_cases"})


@dataclass(frozen=True)
class PlannedEvent:
    split_role: str
    week_start: str
    decision_session: str
    decision_time: str
    maturity_session: str


@dataclass(frozen=True)
class WeeklyCandidate:
    decision_session: str
    maturity_session: str
    required_history: tuple[str, ...]
    required_outcome: tuple[str, ...]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def stable_bool(value: bool) -> str:
    return "true" if value else "false"


def csv_bytes(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: stable_bool(value) if isinstance((value := row.get(field, "")), bool) else value
                for field in fields
            }
        )
    return stream.getvalue().encode("utf-8")


def tie_break_digest(symbol: str) -> str:
    return sha256_bytes(f"{TIE_BREAK_IDENTITY}|{symbol.upper()}".encode())


def parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def structural_row_valid(row: dict[str, Any]) -> bool:
    """Validate only structural OHLC integrity; never compute performance."""
    try:
        opened = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        volume = float(row["Volume"])
    except (KeyError, TypeError, ValueError):
        return False
    values = (opened, high, low, close, volume)
    if not all(math.isfinite(value) for value in values):
        return False
    if min(opened, high, low, close) <= 0 or volume < 0:
        return False
    tolerance = max(abs(high), abs(low), abs(opened), abs(close), 1.0) * 1e-12
    return not (
        high + tolerance < max(opened, low, close)
        or low - tolerance > min(opened, high, close)
    )


def eligible_case_ids_from_price_rows(
    symbol: str,
    rows: Iterable[dict[str, Any]],
    events: Iterable[WeeklyCandidate],
) -> list[str]:
    """Synthetic-test surface proving selection sees validity, never trajectory values."""
    valid = {
        str(row["Date"])[:10]
        for row in rows
        if parse_iso_date(str(row.get("Date", ""))) is not None and structural_row_valid(row)
    }
    result = []
    for event in events:
        if all(session in valid for session in event.required_history + event.required_outcome):
            result.append(f"{symbol.upper()}:{event.decision_session}")
    return result


def planned_events(schedule: ExchangeSchedule) -> list[PlannedEvent]:
    events: list[PlannedEvent] = []
    for role, weeks in ROLE_WEEKS.items():
        for week in weeks:
            weekly = schedule.weekly_events(week, week)[0]
            current = pd.Timestamp(weekly.decision_session)
            for _ in range(HOLDING_SESSIONS):
                current = schedule.next_session(current)
            events.append(
                PlannedEvent(
                    split_role=role,
                    week_start=week,
                    decision_session=weekly.decision_session,
                    decision_time=weekly.decision_close_utc.isoformat(),
                    maturity_session=current.date().isoformat(),
                )
            )
    return sorted(events, key=lambda item: (item.decision_session, ROLE_ORDER[item.split_role]))


def weekly_candidates(schedule: ExchangeSchedule) -> list[WeeklyCandidate]:
    sessions = schedule.sessions("1990-01-02", "2024-01-10")
    labels = [session.date().isoformat() for session in sessions]
    location = {label: index for index, label in enumerate(labels)}
    candidates: list[WeeklyCandidate] = []
    for event in schedule.weekly_events("1990-01-08", "2023-12-25"):
        decision = event.decision_session
        index = location[decision]
        if index < MIN_HISTORY_SESSIONS - 1 or index + HOLDING_SESSIONS >= len(labels):
            continue
        candidates.append(
            WeeklyCandidate(
                decision_session=decision,
                maturity_session=labels[index + HOLDING_SESSIONS],
                required_history=tuple(labels[index - (MIN_HISTORY_SESSIONS - 1) : index + 1]),
                required_outcome=tuple(labels[index + 1 : index + HOLDING_SESSIONS + 1]),
            )
        )
    return candidates


def archive_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        for item in sorted(members, key=lambda value: value.filename):
            digest.update(
                f"{item.filename}|{item.CRC}|{item.file_size}|{item.compress_size}\n".encode()
            )
    return {
        "path": path.name,
        "member_count": len(members),
        "central_directory_identity_sha256": digest.hexdigest(),
        "compressed_bytes": sum(item.compress_size for item in members),
        "uncompressed_bytes": sum(item.file_size for item in members),
    }


def stat_guard(root: Path) -> dict[str, tuple[int, int]]:
    paths = [
        root / RAW_DESCRIPTION,
        root / NEWS_ARCHIVE,
        root / TABLE_ARCHIVE,
        root / TS_ARCHIVE,
        root / IMAGE_ROOT,
    ]
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def load_description(root: Path) -> list[dict[str, str]]:
    with (root / RAW_DESCRIPTION).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: row["stock_name"].upper())
    if len({row["stock_name"].upper() for row in rows}) != len(rows):
        raise ValueError("S&P500 description contains duplicate symbols")
    return rows


def member_maps(root: Path) -> dict[str, Any]:
    with zipfile.ZipFile(root / NEWS_ARCHIVE) as archive:
        text = {
            Path(item.filename).stem.upper(): item.filename
            for item in archive.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".jsonl")
        }
    table: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(root / TABLE_ARCHIVE) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            if not item.is_dir() and len(parts) == 3 and parts[0] == "financial_reports":
                table[parts[1].upper()].append(item.filename)
    with zipfile.ZipFile(root / TS_ARCHIVE) as archive:
        timeseries = {
            Path(item.filename).stem.upper(): item.filename
            for item in archive.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".csv")
        }
    return {
        "text": text,
        "table": {symbol: sorted(names) for symbol, names in table.items()},
        "timeseries": timeseries,
    }


def image_inventory(root: Path) -> tuple[dict[str, list[date]], dict[str, Any]]:
    images: dict[str, list[date]] = defaultdict(list)
    identity = hashlib.sha256()
    file_count = 0
    for path in sorted((root / IMAGE_ROOT).rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        match = IMAGE_RE.match(path.name)
        if not match:
            continue
        symbol = match.group("symbol").upper()
        year = int(match.group("year"))
        period_end = date(year, 6, 30) if match.group("half") == "1" else date(year, 12, 31)
        images[symbol].append(period_end)
        relative = path.relative_to(root).as_posix()
        identity.update(f"{relative}|{path.stat().st_size}\n".encode())
        file_count += 1
    for values in images.values():
        values.sort()
    return dict(images), {
        "file_count": file_count,
        "listing_identity_sha256": identity.hexdigest(),
    }


def _numeric_row(parts: list[bytes]) -> dict[str, bytes]:
    return {
        "Open": parts[1],
        "High": parts[2],
        "Low": parts[3],
        "Close": parts[4],
        "Volume": parts[5],
    }


def scan_time_series_member(
    archive: zipfile.ZipFile,
    member: str,
    calendar_sessions: set[str],
    candidates: list[WeeklyCandidate],
    required_plan: set[str],
) -> dict[str, Any]:
    raw = archive.read(member)
    lines = raw.splitlines()
    expected = b"Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"
    if not lines or lines[0].strip() != expected:
        return {
            "eligible": [],
            "matured": [],
            "invalid_rows": 0,
            "duplicate_rows": 0,
            "non_xnys_rows": 0,
            "planned_complete": False,
            "status": "INVALID_HEADER",
        }
    valid: set[str] = set()
    seen: set[str] = set()
    duplicates: set[str] = set()
    invalid_rows = 0
    non_xnys_rows = 0
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split(b",")
        if len(parts) != 8:
            invalid_rows += 1
            continue
        label = parts[0][:10].decode("ascii", errors="ignore")
        if parse_iso_date(label) is None:
            invalid_rows += 1
            continue
        if label in seen:
            duplicates.add(label)
            valid.discard(label)
            continue
        seen.add(label)
        if label not in calendar_sessions:
            non_xnys_rows += 1
            continue
        decoded = {key: value.decode("ascii", errors="ignore") for key, value in _numeric_row(parts).items()}
        if structural_row_valid(decoded):
            valid.add(label)
        else:
            invalid_rows += 1
    eligible = [
        event
        for event in candidates
        if all(session in valid for session in event.required_history + event.required_outcome)
    ]
    matured = [event for event in eligible if event.maturity_session < FORMAL_BOUNDARY.isoformat()]
    matured_decisions = {event.decision_session for event in matured}
    planned_complete = required_plan <= matured_decisions
    status = "PASS_REQUIRED_WINDOWS" if planned_complete else "INCOMPLETE_REQUIRED_WINDOWS"
    return {
        "eligible": eligible,
        "matured": matured,
        "invalid_rows": invalid_rows,
        "duplicate_rows": len(duplicates),
        "non_xnys_rows": non_xnys_rows,
        "planned_complete": planned_complete,
        "status": status,
    }


def scan_all_time_series(
    root: Path,
    symbols: list[str],
    members: dict[str, str],
    candidates: list[WeeklyCandidate],
    required_plan: set[str],
    calendar_sessions: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(root / TS_ARCHIVE) as archive:
        for symbol in symbols:
            member = members.get(symbol)
            if member is None:
                result[symbol] = {
                    "eligible": [],
                    "matured": [],
                    "invalid_rows": 0,
                    "duplicate_rows": 0,
                    "non_xnys_rows": 0,
                    "planned_complete": False,
                    "status": "NO_TIME_SERIES_MEMBER",
                }
            else:
                result[symbol] = scan_time_series_member(
                    archive, member, calendar_sessions, candidates, required_plan
                )
    return result


def source_limitation(symbol: str, maps: dict[str, Any]) -> str:
    if symbol == "AAPL":
        return (
            "TEXT_FAIL_CLOSED_M1_VERIFIED_SOURCE_CORRUPTION;"
            "IMAGE_FROZEN_UNAVAILABLE_BY_M1_CONTRACT"
        )
    if symbol == "AMZN":
        return "TEXT_M1_FORMAL_SOURCE_POLICY_VERIFIED_MATCH;IMAGE_WINDOW_INFERRED"
    if symbol == "JPM":
        return "TEXT_MEMBER_MISSING;IMAGE_WINDOW_INFERRED"
    limitations = ["TEXT_CONTENT_INTEGRITY_NOT_AUDITED_M2_02"]
    if symbol not in maps["text"]:
        limitations.append("TEXT_MEMBER_MISSING")
    if symbol not in maps["table"]:
        limitations.append("TABLE_MEMBER_MISSING")
    return ";".join(limitations)


def build_symbol_rows(
    descriptions: list[dict[str, str]],
    maps: dict[str, Any],
    images: dict[str, list[date]],
    series: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in descriptions:
        symbol = source["stock_name"].upper()
        item = series[symbol]
        matured = item["matured"]
        correctness = bool(item["planned_complete"] and len(matured) >= 12)
        text_present = symbol in maps["text"]
        table_present = symbol in maps["table"]
        ts_present = symbol in maps["timeseries"]
        image_present = bool(images.get(symbol))
        if correctness:
            exclusion = ""
        elif not ts_present:
            exclusion = "NO_TIME_SERIES_MEMBER"
        else:
            exclusion = "INCOMPLETE_STRUCTURALLY_VALID_REQUIRED_WINDOWS"
        rows.append(
            {
                "symbol": symbol,
                "sector": source.get("Sector", "") or "N/A",
                "formal_symbol": symbol in FORMAL_SYMBOLS,
                "text_present": text_present,
                "table_present": table_present,
                "timeseries_present": ts_present,
                "image_present": image_present,
                "earliest_safe_date": matured[0].decision_session if matured else "",
                "latest_safe_pre2024_decision": matured[-1].decision_session if matured else "",
                "eligible_weekly_cases": len(item["eligible"]),
                "pre2024_matured_reward_cases": len(matured),
                "modality_profile": (
                    f"T{int(text_present)}-B{int(table_present)}-"
                    f"S{int(ts_present)}-I{int(image_present)}"
                ),
                "data_integrity_status": item["status"],
                "structural_invalid_rows": item["invalid_rows"],
                "duplicate_session_rows": item["duplicate_rows"],
                "non_xnys_rows": item["non_xnys_rows"],
                "correctness_eligible": correctness,
                "selected_in_any_tier": False,
                "tie_break_rank": "",
                "known_source_integrity_limitation": source_limitation(symbol, maps),
                "exclusion_reason": exclusion,
            }
        )
    return rows


def select_symbols(symbol_rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    by_symbol = {row["symbol"]: row for row in symbol_rows}
    missing_formal = [symbol for symbol in FORMAL_SYMBOLS if not by_symbol[symbol]["correctness_eligible"]]
    if missing_formal:
        raise ValueError(f"required Formal symbols are not correctness-eligible: {missing_formal}")
    eligible = [
        row for row in symbol_rows if row["correctness_eligible"] and row["symbol"] not in FORMAL_SYMBOLS
    ]
    eligible.sort(key=lambda row: (tie_break_digest(row["symbol"]), row["symbol"]))
    for rank, row in enumerate(eligible, start=1):
        row["tie_break_rank"] = rank
    selected_nonformal: list[str] = []
    used_sectors = {by_symbol[symbol]["sector"] for symbol in FORMAL_SYMBOLS}
    for row in eligible:
        sector = row["sector"]
        if sector not in used_sectors and sector not in {"N/A", "Client Error"}:
            selected_nonformal.append(row["symbol"])
            used_sectors.add(sector)
        if len(selected_nonformal) == 5:
            break
    if len(selected_nonformal) < 5:
        for row in eligible:
            if row["symbol"] not in selected_nonformal:
                selected_nonformal.append(row["symbol"])
            if len(selected_nonformal) == 5:
                break
    if len(selected_nonformal) < 5:
        raise ValueError("fewer than five eligible non-Formal symbols")
    selected = [*FORMAL_SYMBOLS, *selected_nonformal]
    for row in symbol_rows:
        if row["symbol"] in selected:
            row["selected_in_any_tier"] = True
            row["exclusion_reason"] = ""
        elif row["correctness_eligible"]:
            row["exclusion_reason"] = "NOT_SELECTED_BY_DETERMINISTIC_DIVERSITY_TIE_BREAK"
    return selected, [row["symbol"] for row in eligible]


def load_news_dates(root: Path, maps: dict[str, Any], symbols: list[str]) -> dict[str, list[date]]:
    result: dict[str, list[date]] = {symbol: [] for symbol in symbols}
    pattern = re.compile(rb'"Date"\s*:\s*"([^"\\]+)"')
    with zipfile.ZipFile(root / NEWS_ARCHIVE) as archive:
        for symbol in symbols:
            member = maps["text"].get(symbol)
            if member is None:
                continue
            dates = []
            for match in pattern.finditer(archive.read(member)):
                parsed = parse_iso_date(match.group(1).decode("utf-8", errors="ignore"))
                if parsed is not None:
                    dates.append(parsed)
            result[symbol] = sorted(set(dates))
    return result


def load_filed_dates(root: Path, maps: dict[str, Any], symbols: list[str]) -> dict[str, list[date]]:
    result: dict[str, list[date]] = {symbol: [] for symbol in symbols}
    pattern = re.compile(rb'"filed"\s*:\s*"(\d{4}-\d{2}-\d{2})"')
    with zipfile.ZipFile(root / TABLE_ARCHIVE) as archive:
        for symbol in symbols:
            dates: set[date] = set()
            for member in maps["table"].get(symbol, []):
                for match in pattern.finditer(archive.read(member)):
                    parsed = parse_iso_date(match.group(1).decode("ascii"))
                    if parsed is not None:
                        dates.add(parsed)
            result[symbol] = sorted(dates)
    return result


def text_status(symbol: str, decision: date, dates: list[date]) -> str:
    if symbol == "AAPL":
        return "UNAVAILABLE_SOURCE_INTEGRITY"
    if not dates:
        return "UNAVAILABLE_NO_MEMBER_OR_VALID_DATE"
    start = decision - timedelta(days=30)
    return (
        "AVAILABLE_BY_PIT_DATE"
        if any(start <= available < decision for available in dates)
        else "UNAVAILABLE_NO_SAFE_LOOKBACK_RECORD"
    )


def build_cases(
    selected: list[str],
    events: list[PlannedEvent],
    news_dates: dict[str, list[date]],
    filed_dates: dict[str, list[date]],
    images: dict[str, list[date]],
    series: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for symbol in selected:
        valid_matured = {item.decision_session for item in series[symbol]["matured"]}
        for event in events:
            decision = date.fromisoformat(event.decision_session)
            text = text_status(symbol, decision, news_dates[symbol])
            table = (
                "AVAILABLE_BY_FILED_DATE"
                if any(filed < decision for filed in filed_dates[symbol])
                else "UNAVAILABLE_NO_SAFE_FILED_FACT"
            )
            ts = "AVAILABLE" if event.decision_session in valid_matured else "UNAVAILABLE"
            if symbol == "AAPL":
                image = "UNAVAILABLE_M1_FROZEN_POLICY"
            else:
                image = (
                    "AVAILABLE_COMPLETED_HALF_YEAR"
                    if any(period_end < decision for period_end in images.get(symbol, []))
                    else "UNAVAILABLE_NO_COMPLETED_IMAGE"
                )
            available = (
                int(text.startswith("AVAILABLE")),
                int(table.startswith("AVAILABLE")),
                int(ts == "AVAILABLE"),
                int(image.startswith("AVAILABLE")),
            )
            cases.append(
                {
                    "case_id": f"{symbol}:{event.decision_session}",
                    "symbol": symbol,
                    "decision_session": event.decision_session,
                    "decision_time": event.decision_time,
                    "maturity_session": event.maturity_session,
                    "split_role": event.split_role,
                    "compact_included": (
                        event.split_role != "TRAIN"
                        or event.decision_session in TIER_TRAIN_SESSIONS["COMPACT"]
                    ),
                    "standard_included": (
                        event.split_role != "TRAIN"
                        or event.decision_session in TIER_TRAIN_SESSIONS["STANDARD"]
                    ),
                    "maximum_included": True,
                    "formal_symbol": symbol in FORMAL_SYMBOLS,
                    "modality_profile": (
                        f"T{available[0]}-B{available[1]}-S{available[2]}-I{available[3]}"
                    ),
                    "text_status": text,
                    "table_status": table,
                    "timeseries_status": ts,
                    "image_status": image,
                    "eligibility_status": "ELIGIBLE" if ts == "AVAILABLE" else "INELIGIBLE",
                    "source_identity": f"FinMultiTime:S&P500:{symbol}",
                    "inclusion_rationale": (
                        f"{SELECTION_VERSION};CONTIGUOUS_{event.split_role}_BLOCK;"
                        "PERFORMANCE_BLIND"
                    ),
                }
            )
    cases.sort(key=lambda row: (selected.index(row["symbol"]), row["decision_session"]))
    return cases


def tier_cases(cases: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    key = f"{tier.lower()}_included"
    return [row for row in cases if row[key]]


def case_list_bytes(cases: list[dict[str, Any]]) -> bytes:
    lines = [
        f"{row['case_id']}|{row['maturity_session']}|{row['split_role']}"
        for row in sorted(cases, key=lambda item: item["case_id"])
    ]
    return ("\n".join(lines) + "\n").encode()


def split_summary(cases: list[dict[str, Any]], role: str) -> dict[str, Any]:
    rows = [row for row in cases if row["split_role"] == role]
    decisions = sorted({row["decision_session"] for row in rows})
    maturities = sorted({row["maturity_session"] for row in rows})
    return {
        "case_count": len(rows),
        "symbols": sorted({row["symbol"] for row in rows}),
        "decision_sessions": decisions,
        "first_decision_session": decisions[0],
        "last_decision_session": decisions[-1],
        "first_maturity_session": maturities[0],
        "last_maturity_session": maturities[-1],
        "case_ids": sorted(row["case_id"] for row in rows),
    }


def embargo_record(
    schedule: ExchangeSchedule,
    name: str,
    left_maturity: str,
    right_decision: str,
) -> dict[str, Any]:
    start = date.fromisoformat(left_maturity) + timedelta(days=1)
    end = date.fromisoformat(right_decision) - timedelta(days=1)
    sessions = [] if end < start else [item.date().isoformat() for item in schedule.sessions(start, end)]
    return {
        "boundary": name,
        "left_last_maturity_session": left_maturity,
        "right_first_decision_session": right_decision,
        "strict_order_passed": left_maturity < right_decision,
        "embargo_xnys_sessions": sessions,
    }


def no_performance_fields(rows: Iterable[dict[str, Any]]) -> bool:
    for row in rows:
        for field in row:
            lowered = field.lower()
            if lowered in STRUCTURAL_FIELD_EXEMPTIONS:
                continue
            if any(token in lowered for token in PERFORMANCE_FIELD_TOKENS):
                return False
    return True


def validate_plan(selected: list[str], cases: list[dict[str, Any]]) -> None:
    if len(selected) != 8 or not set(FORMAL_SYMBOLS) <= set(selected):
        raise ValueError("MAXIMUM must contain exactly eight symbols including all Formal symbols")
    if sum(symbol not in FORMAL_SYMBOLS for symbol in selected) < 3:
        raise ValueError("insufficient non-Formal symbol diversity")
    if len(cases) > 104 or len(cases) != 96:
        raise ValueError("MAXIMUM case count is outside the frozen design")
    ids = [row["case_id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs overlap")
    if any(row["decision_session"] >= "2024-01-01" for row in cases):
        raise ValueError("2024 decision leakage")
    if any(row["maturity_session"] >= "2024-01-01" for row in cases):
        raise ValueError("2024 maturity leakage")
    if any(row["eligibility_status"] != "ELIGIBLE" for row in cases):
        raise ValueError("ineligible case entered MAXIMUM")
    by_role = {
        role: [row for row in cases if row["split_role"] == role] for role in ROLE_WEEKS
    }
    adjacent_roles = (
        ("TRAIN", "VALIDATION"),
        ("VALIDATION", "FINAL_HOLDOUT"),
        ("FINAL_HOLDOUT", "E2E_PILOT"),
    )
    for left, right in adjacent_roles:
        if max(row["maturity_session"] for row in by_role[left]) >= min(
            row["decision_session"] for row in by_role[right]
        ):
            raise ValueError(f"{left} outcome horizon overlaps {right}")
    compact = {row["case_id"] for row in tier_cases(cases, "COMPACT")}
    standard = {row["case_id"] for row in tier_cases(cases, "STANDARD")}
    maximum = {row["case_id"] for row in tier_cases(cases, "MAXIMUM")}
    if not compact < standard < maximum:
        raise ValueError("budget tiers are not strictly nested")
    evaluation = {row["case_id"] for row in cases if row["split_role"] != "TRAIN"}
    for tier_name, tier_ids in (
        ("COMPACT", compact),
        ("STANDARD", standard),
        ("MAXIMUM", maximum),
    ):
        if evaluation - tier_ids:
            raise ValueError(f"{tier_name} omits fixed evaluation cases")
        if {row["symbol"] for row in tier_cases(cases, tier_name)} != set(selected):
            raise ValueError(f"{tier_name} does not contain all selected symbols")
    if any(
        row["split_role"] != "TRAIN"
        for row in cases
        if row["case_id"] in (maximum - compact)
    ):
        raise ValueError("budget-tier differences include evaluation cases")
    train = [row for row in cases if row["split_role"] == "TRAIN"]
    counts = Counter(row["symbol"] for row in train)
    if max(counts.values()) / len(train) > 0.25:
        raise ValueError("single-symbol TRAIN cap exceeded")
    formal_count = sum(row["formal_symbol"] for row in train)
    if formal_count / len(train) > 0.50:
        raise ValueError("Formal-symbol TRAIN cap exceeded")
    if not no_performance_fields(cases):
        raise ValueError("performance field found in case plan")


def build_protocol(
    root: Path,
    raw_identity: dict[str, Any],
    symbol_rows: list[dict[str, Any]],
    selected: list[str],
    candidate_order: list[str],
    cases: list[dict[str, Any]],
    schedule: ExchangeSchedule,
    raw_unchanged: bool,
) -> dict[str, Any]:
    splits = {role: split_summary(cases, role) for role in ROLE_WEEKS}
    tiers = {}
    for tier in TIER_TRAIN_SESSIONS:
        rows = tier_cases(cases, tier)
        tiers[tier] = {
            "case_count": len(rows),
            "symbols": sorted({row["symbol"] for row in rows}),
            "split_counts": dict(sorted(Counter(row["split_role"] for row in rows).items())),
            "temporal_blocks": {
                role: {
                    "decision_sessions": sorted(
                        {row["decision_session"] for row in rows if row["split_role"] == role}
                    ),
                    "maturity_sessions": sorted(
                        {row["maturity_session"] for row in rows if row["split_role"] == role}
                    ),
                }
                for role in ROLE_WEEKS
            },
            "modality_profiles": sorted({row["modality_profile"] for row in rows}),
            "case_ids": sorted(row["case_id"] for row in rows),
            "canonical_case_list_sha256": sha256_bytes(case_list_bytes(rows)),
        }
    holdout = [row for row in cases if row["split_role"] == "FINAL_HOLDOUT"]
    validation_rows = [row for row in cases if row["split_role"] == "VALIDATION"]
    pilot_rows = [row for row in cases if row["split_role"] == "E2E_PILOT"]
    fixed_evaluation = [row for row in cases if row["split_role"] != "TRAIN"]
    eligible_count = sum(bool(row["correctness_eligible"]) for row in symbol_rows)
    exclusion_categories = Counter(
        row["exclusion_reason"]
        for row in symbol_rows
        if not row["correctness_eligible"] and row["exclusion_reason"]
    )
    train = splits["TRAIN"]
    validation = splits["VALIDATION"]
    final_holdout = splits["FINAL_HOLDOUT"]
    pilot = splits["E2E_PILOT"]
    embargoes = [
        embargo_record(
            schedule,
            "TRAIN_TO_VALIDATION",
            train["last_maturity_session"],
            validation["first_decision_session"],
        ),
        embargo_record(
            schedule,
            "VALIDATION_TO_FINAL_HOLDOUT",
            validation["last_maturity_session"],
            final_holdout["first_decision_session"],
        ),
        embargo_record(
            schedule,
            "FINAL_HOLDOUT_TO_E2E_PILOT",
            final_holdout["last_maturity_session"],
            pilot["first_decision_session"],
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "erratum_task": TASK_ID,
        "parent_protocol_commit": PARENT_PROTOCOL_COMMIT,
        "starting_sha": STARTING_SHA,
        "inherited_architecture_sha": ARCHITECTURE_SHA,
        "environment_recovery_sha": ENVIRONMENT_RECOVERY_SHA,
        "environment": {
            "canonical_interpreter": ".venv-m2-locked/bin/python",
            "underlying_environment": "/Users/yulinqiao/.local/share/alphamas/venvs/m2-locked",
            "python": "3.12.10",
            "numpy": "2.5.2",
            "pandas": "2.3.3",
            "exchange_calendars": "4.13.2",
            "yfinance": "1.5.2",
            "tradingagents": "0.3.1",
        },
        "raw_dataset": {
            "logical_identifier": "FinMultiTime:S&P500:local-read-only",
            "path": str(root),
            "read_only": True,
            "identity": raw_identity,
            "mutation_guard_passed": raw_unchanged,
        },
        "candidate_universe": {
            "partition": "FinMultiTime S&P500",
            "decision_boundary": "decision_session < 2024-01-01",
            "maturity_boundary": "fifth subsequent XNYS session < 2024-01-01",
            "symbols_audited": len(symbol_rows),
            "correctness_eligible_symbols": eligible_count,
            "correctness_excluded_symbols": len(symbol_rows) - eligible_count,
            "exclusion_categories": dict(sorted(exclusion_categories.items())),
            "missing_modalities_allowed": True,
            "formal_performance_inspected": False,
        },
        "pit_rules": {
            "text": "article_date < decision_session; fixed 30-calendar-day lookback",
            "table": "filed_date < decision_session; period_end is not availability",
            "time_series": "61 structurally valid completed XNYS rows through decision",
            "image": "inferred half-year period_end < decision_session",
            "outcome": "decision plus five subsequent XNYS sessions; maturity must be pre-2024",
        },
        "architecture_boundary": {
            "semantic_source": "existing M1 Research Manager investment_plan and unchanged Prompt Trader proposal",
            "direct_raw_analyst_reports_forbidden": True,
            "future_outcomes_in_state_forbidden": True,
        },
        "selection_algorithm": {
            "version": SELECTION_VERSION,
            "performance_selection_forbidden": True,
            "tie_break_identity": TIE_BREAK_IDENTITY,
            "tie_break_formula": "SHA256(tie_break_identity + '|' + upper(symbol))",
            "formal_symbols_required_if_correctness_eligible": list(FORMAL_SYMBOLS),
            "nonformal_rule": "hash order, first pass adds previously unused reliable sectors, then hash-order fill",
            "candidate_order_artifact": SYMBOL_AUDIT_CSV,
            "eligible_nonformal_order_sha256": sha256_bytes(("\n".join(candidate_order) + "\n").encode()),
        },
        "selected_symbols": selected,
        "selected_symbol_metadata": [
            {
                "symbol": row["symbol"],
                "sector": row["sector"],
                "formal_symbol": row["formal_symbol"],
                "physical_modality_profile": row["modality_profile"],
            }
            for row in symbol_rows
            if row["symbol"] in selected
        ],
        "split_definitions": splits,
        "embargo_boundaries": embargoes,
        "budget_tier_policy": {
            "evaluation_membership_fixed_across_tiers": True,
            "all_tiers_use_all_selected_symbols": True,
            "tier_variable": "TRAIN_CASE_COUNT_ONLY",
            "right_aligned_train_windows": True,
        },
        "budget_tiers": tiers,
        "fixed_evaluation_identity": {
            "case_count": len(fixed_evaluation),
            "validation_case_list_sha256": sha256_bytes(case_list_bytes(validation_rows)),
            "final_holdout_case_list_sha256": sha256_bytes(case_list_bytes(holdout)),
            "e2e_pilot_case_list_sha256": sha256_bytes(case_list_bytes(pilot_rows)),
            "combined_case_list_sha256": sha256_bytes(case_list_bytes(fixed_evaluation)),
        },
        "final_holdout_identity": {
            "case_count": len(holdout),
            "symbols": sorted({row["symbol"] for row in holdout}),
            "decision_sessions": sorted({row["decision_session"] for row in holdout}),
            "maturity_sessions": sorted({row["maturity_session"] for row in holdout}),
            "modality_profiles": sorted({row["modality_profile"] for row in holdout}),
            "case_ids": sorted(row["case_id"] for row in holdout),
            "canonical_holdout_case_list_sha256": sha256_bytes(case_list_bytes(holdout)),
            "protected_until_task": "M2-15",
            "performance_inspected": False,
        },
        "pilot_reuse": {
            "aapl_existing_m1_pilot_session_reused": "2023-10-06",
            "existing_m1_pilot_bundle": "finmultitime_m1_pilot_aapl_2023q4_4w_v1",
            "complementary_symbols": [symbol for symbol in selected if symbol != "AAPL"],
            "performance_use_forbidden": True,
        },
        "future_api_tier_selection_rule": (
            "M2-07 may select only among these frozen tiers using measured API cost per case; "
            "performance-driven expansion is forbidden"
        ),
        "future_source_integrity_policy": {
            "modality_failure": "MARK_MODALITY_UNAVAILABLE_AND_RETAIN_CASE",
            "automatic_symbol_reselection": False,
            "performance_driven_symbol_replacement": False,
            "whole_case_correctness_failure": "BLOCK_FOR_EXPLICIT_RESEARCH_REVIEW",
        },
        "api_budget": {
            "target_yuan": "20-40",
            "normal_hard_stop_yuan": 40,
            "absolute_ceiling_yuan": 50,
        },
        "research_validity": {
            "decision_2024_leakage": False,
            "reward_label_2024_leakage": False,
            "performance_based_selection": False,
            "formal_result_selection": False,
            "split_outcome_overlap": False,
            "raw_dataset_mutation": not raw_unchanged,
            "performance_fields_in_manifests": not (
                no_performance_fields(symbol_rows) and no_performance_fields(cases)
            ),
        },
        "cost": {
            "deepseek_api_calls": 0,
            "deepseek_api_cost_yuan": 0,
            "qwen_inference_calls": 0,
            "aws_gpu_hours": 0,
            "aws_incremental_cost_usd": 0,
        },
        "final_freeze_verdict": (
            "PASS — canonical M2 resident checkout established and fixed-evaluation "
            "budget tiers frozen; ready for M2-03"
        ),
    }


def render_markdown(protocol: dict[str, Any]) -> bytes:
    tiers = protocol["budget_tiers"]
    splits = protocol["split_definitions"]
    selected = protocol["selected_symbols"]
    embargo_lines = "\n".join(
        f"- `{item['boundary']}`: `{item['left_last_maturity_session']}` < "
        f"`{item['right_first_decision_session']}`; embargo sessions: "
        f"`{', '.join(item['embargo_xnys_sessions'])}`"
        for item in protocol["embargo_boundaries"]
    )
    split_lines = "\n".join(
        f"- **{role}:** {item['case_count']} cases; decisions "
        f"`{item['first_decision_session']}`–`{item['last_decision_session']}`; "
        f"maturities `{item['first_maturity_session']}`–`{item['last_maturity_session']}`."
        for role in ROLE_WEEKS
        for item in (splits[role],)
    )
    tier_lines = "\n".join(
        f"- **{tier}:** {item['case_count']} cases; "
        f"SHA-256 `{item['canonical_case_list_sha256']}`; symbols "
        f"`{', '.join(item['symbols'])}`."
        for tier in TIER_TRAIN_SESSIONS
        for item in (tiers[tier],)
    )
    metadata = protocol["candidate_universe"]
    holdout = protocol["final_holdout_identity"]
    text = f"""# M2 Pre-Formal Data and Split Protocol

## 1. Objective

This protocol freezes the performance-blind FinMultiTime input universe and exact
TRAIN, VALIDATION, FINAL_HOLDOUT, and E2E_PILOT membership for Full M2, A1, and
A2. It selects inputs only: no semantic state, reward, return, model, or policy
was generated or evaluated.

## 2. Inherited architecture boundary

The inherited M2-01A boundary is unchanged. Future Actor semantics may use only
the existing M1 Research Manager `investment_plan`, unchanged Prompt Trader
proposal/action/reasoning, permitted instrument/time/PIT context, and permitted
endogenous portfolio state. Raw analyst reports, debates, Risk/PM output, and
future outcomes remain forbidden direct inputs.

## 3. Read-only dataset policy

The logical source is `{protocol['raw_dataset']['logical_identifier']}` at
`{protocol['raw_dataset']['path']}`. ZIP members were streamed or inspected via
their central directories; images were inventoried in place. No raw member was
extracted, rewritten, renamed, or deleted. The pre/post mutation guard passed:
**{str(protocol['raw_dataset']['mutation_guard_passed']).upper()}**.

## 4. Candidate universe

The complete 4,213-row FinMultiTime S&P500 description partition was audited
against physical text/table/time-series members and image inventory. Time-series
eligibility uses only XNYS continuity and structural validity. Missing modalities
remain legitimate. Correctness-eligible: {metadata['correctness_eligible_symbols']};
correctness-excluded: {metadata['correctness_excluded_symbols']}.

## 5. Frozen PIT eligibility rules

- TEXT: `article_date < decision_session`, with the frozen 30-calendar-day lookback.
- TABLE: `filed_date < decision_session`; reporting period end is never availability.
- TIME_SERIES: 61 structurally valid completed XNYS rows through decision.
- IMAGE: inferred half-year end must be strictly before decision.
- OUTCOME: the fifth subsequent XNYS session is maturity and must be pre-2024.

`UNAVAILABLE` is retained as a legitimate modality status. AAPL TEXT remains
fail-closed under the frozen M1 source-integrity erratum, and AAPL IMAGE remains
unavailable under the frozen M1 contract even if later physical inventory differs.

## 6. Performance-blind selection design

Prices are inspected only for missing/non-finite values, positivity, duplicate
sessions, volume sign, and impossible OHLC relations. No return, direction,
volatility, drawdown, reward, or profitability is calculated. Formal M0/M1
performance artifacts were not inspected.

The algorithm `{SELECTION_VERSION}` requires correctness-eligible AAPL, AMZN,
and JPM. Non-Formal candidates are ordered by
`SHA256("{TIE_BREAK_IDENTITY}" + "|" + upper(symbol))`; a first pass adds new
reliable sectors and a second pass fills strictly by that hash order. The exact
rank is recorded in `{SYMBOL_AUDIT_CSV}`.

## 7. Selected symbols and modality diversity

All budget tiers use the same eight symbols: `{', '.join(selected)}`. The three Formal symbols contribute
distinct legitimate availability profiles (including AAPL image/TEXT restrictions
and JPM missing TEXT), while deterministic non-Formal selection supplies
cross-asset and sector diversity. Formal symbols are 3/8; non-Formal symbols are
5/8.

## 8. Sequential split definitions

Each symbol contributes contiguous weekly blocks within each role:

{split_lines}

TRAIN may later fit approved components. VALIDATION is for pre-Formal selection
only. FINAL_HOLDOUT is protected until M2-15. E2E_PILOT is engineering-only and
includes the provenance-intact AAPL `2023-10-06` case from the frozen M1 pilot;
its performance cannot drive design.

## 9. Embargo logic

Every left-hand maturity precedes the next protected decision:

{embargo_lines}

These are XNYS session boundaries, not calendar-day approximations.

## 10. M2-02B fixed-evaluation budget-tier erratum

M2-02 varied case count by dropping complete symbols, which unintentionally made
VALIDATION, FINAL_HOLDOUT, and E2E_PILOT depend on API affordability. M2-02B
corrects that inconsistency: API budget changes TRAIN volume only, while the
40-case evaluation set and all eight symbols remain fixed across tiers.

TRAIN windows are right-aligned at decision `2023-06-16` (maturity
`2023-06-26`). COMPACT uses the final four TRAIN weeks, STANDARD the final five,
and MAXIMUM all seven:

{tier_lines}

The corrected totals are COMPACT 72, STANDARD 80, and MAXIMUM 96. COMPACT is a
strict subset of STANDARD, which is a strict subset of MAXIMUM; every set
difference contains TRAIN cases only.
M2-07 may choose among them using measured API cost per case only. Performance
cannot trigger hand-picked additions or tier expansion.

## 11. Final Holdout identity and protection

FINAL_HOLDOUT contains {holdout['case_count']} cases across
`{', '.join(holdout['symbols'])}`, with decisions
`{', '.join(holdout['decision_sessions'])}` and maturity sessions
`{', '.join(holdout['maturity_sessions'])}`. Canonical SHA-256:
`{holdout['canonical_holdout_case_list_sha256']}`. Its outcomes remain blocked
from development/model-selection code until M2-15. No Holdout performance was
calculated or inspected.

VALIDATION, FINAL_HOLDOUT, and E2E_PILOT have separate deterministic identities,
plus one combined fixed-evaluation identity, in `{PROTOCOL_JSON}`. The frozen
Holdout identity is unchanged by this erratum.

## 12. Environment and reproducibility

Audit execution used resident CPython 3.12.10 through
`.venv-m2-locked/bin/python`, backed by
`/Users/yulinqiao/.local/share/alphamas/venvs/m2-locked`, with NumPy 2.5.2,
pandas 2.3.3, exchange-calendars 4.13.2, yfinance 1.5.2, and tradingagents
0.3.1. JSON is sorted and CSV fields/order are fixed. Exact case identities and
canonical hashes are in `{PROTOCOL_JSON}` and `{CASE_PLAN_CSV}`.

## 13. Limitations

FinMultiTime's partition name is retained as supplied and includes 4,213 symbols;
the stock-description availability flags are advisory, so physical members are
authoritative. Non-Formal news members were date/PIT audited but not subjected to
the separate M1 article-content integrity investigation. Image availability is
inferred conservatively from half-year filenames. OHLC adjustment semantics are
not used for selection.

Future modality-integrity failure is fail-closed: retain the case, symbol, role,
and tier membership and mark only the affected modality unavailable. Automatic
symbol reselection, including performance-driven replacement, is forbidden. A
whole-case/core correctness failure blocks processing pending explicit research
review; it never triggers automatic replacement.

## 14. Research validity and cost

2024 decision leakage: **NO**. 2024 reward-label leakage: **NO**.
Performance-based selection: **NO**. Formal-result selection: **NO**.
Split outcome overlap: **NO**. Raw dataset mutation: **NO**.

DeepSeek API calls: **0**; DeepSeek API cost: **¥0**; Qwen inference calls:
**0**; AWS GPU hours: **0**; AWS incremental cost: **$0**.

## 15. Final freeze verdict

**{protocol['final_freeze_verdict']}**
"""
    return text.encode("utf-8")


def artifact_bytes(
    protocol: dict[str, Any],
    symbol_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, bytes]:
    return {
        PROTOCOL_JSON: canonical_json_bytes(protocol),
        SYMBOL_AUDIT_CSV: csv_bytes(symbol_rows, SYMBOL_AUDIT_FIELDS),
        CASE_PLAN_CSV: csv_bytes(cases, CASE_PLAN_FIELDS),
        PROTOCOL_MD: render_markdown(protocol),
    }


def run(raw_root: Path, output_dir: Path) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    pre_guard = stat_guard(raw_root)
    description = load_description(raw_root)
    maps = member_maps(raw_root)
    images, image_identity = image_inventory(raw_root)
    schedule = ExchangeSchedule(start="1990-01-01", end="2025-12-31")
    events = planned_events(schedule)
    candidates = weekly_candidates(schedule)
    required_plan = {event.decision_session for event in events}
    calendar_sessions = {
        session.date().isoformat() for session in schedule.sessions("1990-01-02", "2025-12-31")
    }
    symbols = [row["stock_name"].upper() for row in description]
    series = scan_all_time_series(
        raw_root,
        symbols,
        maps["timeseries"],
        candidates,
        required_plan,
        calendar_sessions,
    )
    symbol_rows = build_symbol_rows(description, maps, images, series)
    if not no_performance_fields(symbol_rows):
        raise ValueError("performance field found in symbol audit")
    selected, candidate_order = select_symbols(symbol_rows)
    news_dates = load_news_dates(raw_root, maps, selected)
    filed_dates = load_filed_dates(raw_root, maps, selected)
    cases = build_cases(selected, events, news_dates, filed_dates, images, series)
    validate_plan(selected, cases)
    post_images, post_image_identity = image_inventory(raw_root)
    del post_images
    raw_unchanged = pre_guard == stat_guard(raw_root) and image_identity == post_image_identity
    if not raw_unchanged:
        raise RuntimeError("raw FinMultiTime mutation guard failed")
    raw_identity = {
        "description_sha256": sha256_bytes((raw_root / RAW_DESCRIPTION).read_bytes()),
        "description_symbol_count": len(description),
        "archives": {
            "text": archive_identity(raw_root / NEWS_ARCHIVE),
            "table": archive_identity(raw_root / TABLE_ARCHIVE),
            "time_series": archive_identity(raw_root / TS_ARCHIVE),
        },
        "images": image_identity,
    }
    protocol = build_protocol(
        raw_root,
        raw_identity,
        symbol_rows,
        selected,
        candidate_order,
        cases,
        schedule,
        raw_unchanged,
    )
    outputs = artifact_bytes(protocol, symbol_rows, cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)
    return {
        "protocol": protocol,
        "symbol_rows": symbol_rows,
        "cases": cases,
        "artifact_sha256": {name: sha256_bytes(payload) for name, payload in outputs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.raw_root, args.output_dir)
    summary = {
        "selected_symbols": result["protocol"]["selected_symbols"],
        "tier_counts": {
            tier: item["case_count"] for tier, item in result["protocol"]["budget_tiers"].items()
        },
        "artifact_sha256": result["artifact_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
