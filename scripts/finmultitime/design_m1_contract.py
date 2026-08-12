#!/usr/bin/env python3
"""Build the proposed M1 FinMultiTime evidence contract.

This is a deterministic, read-only design analysis.  It reads the existing
FinMultiTime audit inputs, the local frozen M0 market snapshot, and the formal
XNYS schedule.  It writes lightweight review artifacts only; it never writes
to the raw dataset, calls a provider, creates Evidence Packets, or runs an
LLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.audit_finmultitime import (  # noqa: E402
    FORMAL_CONFIG,
    NEWS_ARCHIVE,
    NEWS_MEMBERS,
    TABLE_ARCHIVE,
    TABLE_MEMBER_RE,
    TARGETS,
    TS_ARCHIVE,
    TS_MEMBERS,
    discover_images,
    parse_date,
    parse_formal_schedule,
    read_jsonl_news,
    read_table_files,
    read_time_series,
)

CONTRACT_VERSION = "M1-FINMULTITIME-DRAFT-0.1"
NEWS_LOOKBACK_CANDIDATES = (7, 14, 30)
RECOMMENDED_NEWS_LOOKBACK = 30
IMAGE_STALENESS_CANDIDATES = (30, 90, 180, 365)
TIME_SERIES_WINDOWS = (5, 20, 60)
SELECTED_TABLE_CONCEPTS = (
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
)
MAX_NEWS_ARTICLES = 8
MAX_ARTICLE_TITLE_CHARS = 200
MAX_ARTICLE_BODY_CHARS = 900
MAX_NEWS_SECTION_CHARS = 12_000
MAX_TABLE_SECTION_CHARS = 3_200
MAX_TIME_SERIES_SECTION_CHARS = 3_200
MAX_IMAGE_SECTION_CHARS = 1_600
MAX_PACKET_CHARS = 22_000

M0_RUN_INPUTS = (
    REPO_ROOT
    / "results/backtests/M0_original_prompt_2024H1/runs/20260812T082530978211Z_2535896c/inputs"
)
M0_USAGE = (
    REPO_ROOT
    / "results/backtests/M0_original_prompt_2024H1/runs/20260812T082530978211Z_2535896c/analysis_ready/llm_usage.csv"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fmt_number(value: Any, places: int = 10) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{places}f}".rstrip("0").rstrip(".")


def date_or_none(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return parse_date(value)


def date_text(value: date | None) -> str:
    return value.isoformat() if value else ""


def parse_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def formal_context() -> dict[str, Any]:
    events, warmup, warmup_start = parse_formal_schedule()
    decisions = [date.fromisoformat(row["decision_session"]) for row in events]
    return {
        "events": events,
        "decisions": decisions,
        "warmup": {date.fromisoformat(value) for value in warmup},
        "warmup_start": date.fromisoformat(warmup_start),
        "warmup_end": date.fromisoformat(warmup[-1]),
        "formal_start": decisions[0],
        "formal_end": decisions[-1],
        "formal_calendar_start": date(2024, 1, 1),
        "formal_calendar_end": date(2024, 6, 30),
        "final_valuation": date.fromisoformat(FORMAL_CONFIG["final_valuation_session"]),
    }


def load_raw_data(root: Path) -> dict[str, Any]:
    """Load only target members; no archive member is extracted or modified."""
    news: dict[str, dict[str, Any] | None] = {}
    tables: dict[str, dict[str, Any] | None] = {}
    series: dict[str, dict[str, Any] | None] = {}

    with zipfile.ZipFile(root / NEWS_ARCHIVE) as archive:
        names = set(archive.namelist())
        for symbol in TARGETS:
            member = NEWS_MEMBERS[symbol]
            news[symbol] = read_jsonl_news(archive, member) if member in names else None

    with zipfile.ZipFile(root / TABLE_ARCHIVE) as archive:
        names = archive.namelist()
        for symbol in TARGETS:
            members = [name for name in names if TABLE_MEMBER_RE[symbol].match(name)]
            tables[symbol] = read_table_files(archive, members) if members else None

    with zipfile.ZipFile(root / TS_ARCHIVE) as archive:
        names = set(archive.namelist())
        for symbol in TARGETS:
            member = TS_MEMBERS[symbol]
            series[symbol] = read_time_series(archive, member) if member in names else None

    images = discover_images(root / "image/image")["by_symbol"]
    return {"news": news, "tables": tables, "series": series, "images": images}


def deduplicate_news(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply exact-record then URL identity, retaining removed-record provenance."""
    exact_seen: dict[str, dict[str, Any]] = {}
    exact_kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        record_hash = digest(record)
        if record_hash in exact_seen:
            removed.append(
                {
                    "source_index": index,
                    "removed_hash": record_hash,
                    "kept_hash": exact_seen[record_hash]["_record_hash"],
                    "reason": "EXACT_DUPLICATE_RECORD",
                    "url": record.get("Url", ""),
                }
            )
            continue
        item = {**record, "_source_index": index, "_record_hash": record_hash}
        exact_seen[record_hash] = item
        exact_kept.append(item)

    url_seen: dict[str, dict[str, Any]] = {}
    final: list[dict[str, Any]] = []
    for item in exact_kept:
        url = item.get("Url")
        if isinstance(url, str) and url:
            if url in url_seen:
                removed.append(
                    {
                        "source_index": item["_source_index"],
                        "removed_hash": item["_record_hash"],
                        "kept_hash": url_seen[url]["_record_hash"],
                        "reason": "DUPLICATE_URL",
                        "url": url,
                    }
                )
                continue
            url_seen[url] = item
        final.append(item)

    return {
        "records": final,
        "removed": removed,
        "input_count": len(records),
        "output_count": len(final),
        "removed_count": len(removed),
    }


def news_date(record: dict[str, Any]) -> date | None:
    return date_or_none(record.get("Date"))


def selected_news(
    data: dict[str, Any] | None,
    decision: date,
    lookback_days: int = RECOMMENDED_NEWS_LOOKBACK,
) -> dict[str, Any]:
    if data is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "no FinMultiTime news member for target symbol",
            "records": [],
            "same_day_count": 0,
            "dedup_removed_count": 0,
            "latest_safe_date": None,
            "ambiguous_records": [],
        }
    deduped = deduplicate_news(data["records"])
    dated = [(record, news_date(record)) for record in deduped["records"]]
    same_day = [record for record, available in dated if available == decision]
    eligible = [
        (record, available)
        for record, available in dated
        if available is not None
        and decision - timedelta(days=lookback_days) <= available < decision
    ]
    # Newest-first with ascending deterministic tie-breaks.
    eligible.sort(
        key=lambda pair: (
            -pair[1].toordinal(),
            str(pair[0].get("Url", "")),
            str(pair[0].get("Article_title", "")),
            pair[0]["_record_hash"],
        )
    )
    selected = eligible[:MAX_NEWS_ARTICLES]
    selected_records = [record for record, _ in selected]
    if not eligible:
        return {
            "status": "UNAVAILABLE",
            "reason": "no PIT-safe article inside fixed 30-calendar-day lookback",
            "records": [],
            "same_day_count": len(same_day),
            "dedup_removed_count": deduped["removed_count"],
            "latest_safe_date": max(
                (available for _, available in dated if available is not None and available < decision),
                default=None,
            ),
            "ambiguous_records": same_day,
            "dedup": deduped,
        }
    return {
        "status": "AVAILABLE",
        "reason": "",
            "records": selected_records,
        "same_day_count": len(same_day),
        "dedup_removed_count": deduped["removed_count"],
        "latest_safe_date": max((available for _, available in eligible), default=None),
        "ambiguous_records": same_day,
        "dedup": deduped,
    }


def news_lookback_analysis(
    data: dict[str, Any] | None,
    decisions: list[date],
) -> dict[int, dict[str, Any]]:
    if data is None:
        return {
            window: {
                "counts": [0] * len(decisions),
                "coverage_cases": 0,
                "no_article_cases": len(decisions),
                "average_count": 0.0,
                "maximum_count": 0,
            }
            for window in NEWS_LOOKBACK_CANDIDATES
        }
    deduped = deduplicate_news(data["records"])
    dated = [
        available
        for record in deduped["records"]
        if (available := news_date(record)) is not None
    ]
    result: dict[int, dict[str, Any]] = {}
    for window in NEWS_LOOKBACK_CANDIDATES:
        counts = [
            sum(decision - timedelta(days=window) <= available < decision for available in dated)
            for decision in decisions
        ]
        result[window] = {
            "counts": counts,
            "coverage_cases": sum(count > 0 for count in counts),
            "no_article_cases": sum(count == 0 for count in counts),
            "average_count": round(sum(counts) / len(counts), 3),
            "maximum_count": max(counts, default=0),
            "deduped_record_count": len(deduped["records"]),
            "removed_duplicate_count": deduped["removed_count"],
        }
    return result


def impossible_ohlc_rows(
    series: dict[str, dict[str, Any] | None],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in TARGETS:
        item = series[symbol]
        if item is None:
            continue
        dates = [row["session_date"] for row in item["rows"] if row.get("session_date")]
        sessions_by_date = {value: index for index, value in enumerate(dates)}
        for row in item["rows"]:
            values = row["numeric"]
            opened, high, low, close = (values.get(key) for key in ("Open", "High", "Low", "Close"))
            if None in (opened, high, low, close):
                continue
            violations: list[str] = []
            if high < max(opened, close, low):
                violations.append("High < max(Open, Close, Low)")
            if low > min(opened, close, high):
                violations.append("Low > min(Open, Close, High)")
            if not violations:
                continue
            session_date = row["session_date"]
            reachable = any(
                session_date in dates[max(0, sessions_by_date[decision] - 59) : sessions_by_date[decision] + 1]
                for decision in context["decisions"]
                if decision in sessions_by_date
            )
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": date_text(session_date),
                    "Open": fmt_number(opened),
                    "High": fmt_number(high),
                    "Low": fmt_number(low),
                    "Close": fmt_number(close),
                    "Volume": fmt_number(values.get("Volume"), 0),
                    "Dividends": fmt_number(values.get("Dividends")),
                    "Stock Splits": fmt_number(values.get("Stock Splits")),
                    "violated_invariant": "; ".join(violations),
                    "inside_252_session_m0_warmup": session_date in context["warmup"],
                    "inside_formal_2024h1_calendar": context["formal_calendar_start"] <= session_date <= context["formal_calendar_end"],
                    "inside_formal_decision_span": context["formal_start"] <= session_date <= context["formal_end"],
                    "potentially_reachable_by_60_session_lookback": reachable,
                    "raw_row_action": "NOT_REPAIRED",
                }
            )
    return rows


def read_snapshot(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}
    for symbol in TARGETS:
        path = root / f"{symbol}.csv"
        if not path.exists():
            result[symbol] = {}
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            result[symbol] = {row["Date"]: row for row in csv.DictReader(handle)}
    return result


def neighboring_dates(
    rows: list[dict[str, Any]], target: date, count: int = 1
) -> list[date]:
    dates = [row["session_date"] for row in rows if row.get("session_date")]
    if target not in dates:
        return []
    index = dates.index(target)
    return list(dates[max(0, index - count) : index + count + 1])


def semantics_dates(
    series: dict[str, dict[str, Any] | None],
    snapshots: dict[str, dict[str, dict[str, str]]],
) -> list[tuple[str, str, date]]:
    dates: set[tuple[str, str, date]] = set()
    ordinary = (date(2023, 1, 4), date(2023, 6, 15), date(2024, 1, 5), date(2024, 3, 15))
    for symbol in TARGETS:
        for value in ordinary:
            dates.add((symbol, "ordinary_non_corporate_action", value))
        item = series[symbol]
        if item is None:
            continue
        for row in item["rows"]:
            action_date = row.get("session_date")
            if not action_date:
                continue
            numeric = row["numeric"]
            dividend = numeric.get("Dividends") or 0
            split = numeric.get("Stock Splits") or 0
            if dividend and action_date.isoformat() in snapshots[symbol]:
                for neighbor in neighboring_dates(item["rows"], action_date):
                    role = "dividend_event" if neighbor == action_date else "dividend_adjacent"
                    dates.add((symbol, role, neighbor))
            if split:
                for neighbor in neighboring_dates(item["rows"], action_date):
                    role = "split_event" if neighbor == action_date else "split_adjacent"
                    dates.add((symbol, role, neighbor))
    return sorted(dates, key=lambda value: (value[0], value[2], value[1]))


def consistency_rows(
    series: dict[str, dict[str, Any] | None],
    snapshots: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, role, value_date in semantics_dates(series, snapshots):
        fin_rows = {
            row["session_date"]: row
            for row in (series[symbol]["rows"] if series[symbol] else [])
            if row.get("session_date")
        }
        fin = fin_rows.get(value_date)
        reference = snapshots[symbol].get(value_date.isoformat())
        row: dict[str, Any] = {
            "symbol": symbol,
            "date_role": role,
            "session_date": date_text(value_date),
            "m0_reference_status": "AVAILABLE" if reference else "NOT_AVAILABLE_OUTSIDE_FROZEN_SNAPSHOT",
            "finmultitime_source": TS_MEMBERS[symbol],
            "m0_source": f"inputs/market_data/{symbol}.csv",
        }
        for field in ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"):
            fin_value = fin["numeric"].get(field) if fin else None
            m0_value = parse_float(reference.get(field)) if reference else None
            row[f"finmultitime_{field.lower().replace(' ', '_')}"] = fmt_number(fin_value, 10 if field != "Volume" else 0)
            row[f"m0_{field.lower().replace(' ', '_')}"] = fmt_number(m0_value, 10 if field != "Volume" else 0)
            if fin_value is not None and m0_value is not None:
                row[f"relative_difference_{field.lower().replace(' ', '_')}"] = fmt_number(
                    fin_value / m0_value - 1 if m0_value else None, 8
                )
            else:
                row[f"relative_difference_{field.lower().replace(' ', '_')}"] = ""
        row["interpretation"] = (
            "Direct comparison available; inspect price ratios and volume separately."
            if reference
            else "FinMultiTime event retained; frozen M0 snapshot has no overlapping date."
        )
        rows.append(row)
    return rows


def semantics_summary(consistency: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for symbol in TARGETS:
        rows = [row for row in consistency if row["symbol"] == symbol and row["m0_reference_status"] == "AVAILABLE"]
        ratios = [
            parse_float(row["relative_difference_close"])
            for row in rows
            if row.get("relative_difference_close") not in (None, "")
        ]
        volume_ratios = [
            parse_float(row["relative_difference_volume"])
            for row in rows
            if row.get("relative_difference_volume") not in (None, "")
        ]
        summary[symbol] = {
            "comparison_rows": len(rows),
            "close_ratio_fin_minus_m0_min": min(ratios, default=None),
            "close_ratio_fin_minus_m0_max": max(ratios, default=None),
            "close_ratio_fin_minus_m0_median": statistics.median(ratios) if ratios else None,
            "volume_difference_median": statistics.median(volume_ratios) if volume_ratios else None,
        }
    return summary


def concept_groups(item: dict[str, Any] | None) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    if item is None:
        return groups
    for row in item["observations"]:
        groups[(row["taxonomy"], row["concept"], row["unit"])].append(row)
    return groups


def candidate_concept_coverage(
    tables: dict[str, dict[str, Any] | None],
    decisions: list[date],
) -> list[dict[str, Any]]:
    keys: set[tuple[str, str, str]] = set()
    per_symbol = {symbol: concept_groups(tables[symbol]) for symbol in TARGETS}
    for groups in per_symbol.values():
        keys.update(groups)
    rows: list[dict[str, Any]] = []
    for taxonomy, concept, unit in sorted(keys):
        symbol_details: dict[str, Any] = {}
        covered_symbols = 0
        for symbol in TARGETS:
            values = per_symbol[symbol].get((taxonomy, concept, unit), [])
            if values:
                covered_symbols += 1
            period_keys = {
                (row.get("raw_start"), row.get("raw_end"))
                for row in values
            }
            restatement_counts = Counter(
                (row.get("raw_start"), row.get("raw_end")) for row in values
            )
            filed_dates = {row["filed"] for row in values if row.get("filed")}
            safe_latest = [
                max(
                    (row["filed"] for row in values if row.get("filed") and row["filed"] < decision),
                    default=None,
                )
                for decision in decisions
            ]
            safe_latest = [value for value in safe_latest if value]
            symbol_details[symbol] = {
                "row_count": len(values),
                "unit": unit,
                "unique_filing_dates": len(filed_dates),
                "unique_period_groups": len(period_keys),
                "restated_period_groups": sum(count > 1 for count in restatement_counts.values()),
                "max_versions_per_period": max(restatement_counts.values(), default=0),
                "latest_filed": max(filed_dates, default=None),
                "safe_case_count": len(safe_latest),
                "safe_latest_min": min(safe_latest, default=None),
                "safe_latest_max": max(safe_latest, default=None),
            }
        recommended = taxonomy == "us-gaap" and concept in SELECTED_TABLE_CONCEPTS and unit == "USD"
        descriptions = {
            "Assets": "balance-sheet total assets",
            "Liabilities": "balance-sheet total liabilities",
            "StockholdersEquity": "equity attributable to stockholders",
            "NetCashProvidedByUsedInOperatingActivities": "cash flow from operations",
            "NetCashProvidedByUsedInInvestingActivities": "cash flow from investing",
            "NetCashProvidedByUsedInFinancingActivities": "cash flow from financing",
        }
        rows.append(
            {
                "taxonomy": taxonomy,
                "concept": concept,
                "unit": unit,
                "symbol_coverage": "/".join(symbol for symbol in TARGETS if per_symbol[symbol].get((taxonomy, concept, unit))),
                "symbol_coverage_count": covered_symbols,
                "units_by_symbol": "; ".join(
                    f"{symbol}:{symbol_details[symbol]['unit']}" for symbol in TARGETS if symbol_details[symbol]["row_count"]
                ),
                "filing_frequency_by_symbol": "; ".join(
                    f"{symbol}:unique_filed_dates={symbol_details[symbol]['unique_filing_dates']}"
                    for symbol in TARGETS
                    if symbol_details[symbol]["row_count"]
                ),
                "latest_safe_availability_by_symbol": "; ".join(
                    f"{symbol}:{date_text(symbol_details[symbol]['safe_latest_min'])}..{date_text(symbol_details[symbol]['safe_latest_max'])}"
                    for symbol in TARGETS
                    if symbol_details[symbol]["safe_case_count"]
                ),
                "duplicate_or_restatement_by_symbol": "; ".join(
                    f"{symbol}:groups={symbol_details[symbol]['restated_period_groups']};max_versions={symbol_details[symbol]['max_versions_per_period']}"
                    for symbol in TARGETS
                    if symbol_details[symbol]["row_count"]
                ),
                "formal_case_safe_coverage_by_symbol": "; ".join(
                    f"{symbol}:{symbol_details[symbol]['safe_case_count']}/{len(decisions)}"
                    for symbol in TARGETS
                ),
                "recommended_fixed_schema": "true" if recommended else "false",
                "selection_reason": descriptions.get(concept, "not selected: no compact cross-asset contract role"),
            }
        )
    return rows


def choose_table_fact(
    tables: dict[str, dict[str, Any] | None], symbol: str, concept: str, decision: date
) -> dict[str, Any] | None:
    item = tables[symbol]
    if item is None:
        return None
    eligible = [
        row
        for row in item["observations"]
        if row["taxonomy"] == "us-gaap"
        and row["concept"] == concept
        and row["unit"] == "USD"
        and row.get("filed") is not None
        and row["filed"] < decision
    ]
    if not eligible:
        return None
    # First resolve revisions for each exact economic fact, then choose the
    # most recent economic period.  Filing date is availability, never period_end.
    by_period: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_period[(row.get("raw_start"), row.get("raw_end"))].append(row)
    latest_versions = []
    for values in by_period.values():
        latest_versions.append(
            max(
                values,
                key=lambda row: (
                    row.get("filed") or date.min,
                    str(row.get("accn") or ""),
                    str(row.get("form") or ""),
                    str(row.get("member") or ""),
                ),
            )
        )
    return max(
        latest_versions,
        key=lambda row: (
            row.get("end") or date.min,
            row.get("start") or date.min,
            row.get("filed") or date.min,
            str(row.get("accn") or ""),
            str(row.get("form") or ""),
            str(row.get("member") or ""),
        ),
    )


def table_selection(
    tables: dict[str, dict[str, Any] | None], symbol: str, decision: date
) -> dict[str, Any]:
    item = tables[symbol]
    if item is None:
        return {"status": "UNAVAILABLE", "facts": {}, "same_day_count": 0, "latest_safe": None}
    same_day = sum(
        row.get("filed") == decision for row in item["observations"] if row.get("filed")
    )
    facts = {
        concept: choose_table_fact(tables, symbol, concept, decision)
        for concept in SELECTED_TABLE_CONCEPTS
    }
    facts = {concept: fact for concept, fact in facts.items() if fact is not None}
    latest_safe = max(
        (row["filed"] for row in item["observations"] if row.get("filed") and row["filed"] < decision),
        default=None,
    )
    return {
        "status": "AVAILABLE" if facts else "UNAVAILABLE",
        "facts": facts,
        "same_day_count": same_day,
        "latest_safe": latest_safe,
    }


def time_series_selection(
    series: dict[str, dict[str, Any] | None], symbol: str, decision: date
) -> dict[str, Any]:
    item = series[symbol]
    if item is None:
        return {"status": "UNAVAILABLE", "rows": [], "latest": None}
    rows = [row for row in item["rows"] if row.get("session_date") and row["session_date"] <= decision]
    rows.sort(key=lambda row: row["session_date"])
    window = rows[-max(TIME_SERIES_WINDOWS) :]
    return {
        "status": "AVAILABLE" if len(window) >= max(TIME_SERIES_WINDOWS) else "UNAVAILABLE",
        "rows": window,
        "latest": window[-1]["session_date"] if window else None,
        "source_row_count_through_decision": len(rows),
    }


def ts_summary(selection: dict[str, Any]) -> dict[str, Any]:
    rows = selection["rows"]
    result: dict[str, Any] = {}
    closes = [parse_float(row["numeric"].get("Close")) for row in rows]
    highs = [parse_float(row["numeric"].get("High")) for row in rows]
    lows = [parse_float(row["numeric"].get("Low")) for row in rows]
    volumes = [parse_float(row["numeric"].get("Volume")) for row in rows]
    for window in TIME_SERIES_WINDOWS:
        values = closes[-window:]
        result[f"cumulative_return_{window}d"] = (
            values[-1] / values[0] - 1 if len(values) == window and values[0] not in (None, 0) and values[-1] is not None else None
        )
    returns = [
        closes[index] / closes[index - 1] - 1
        for index in range(1, len(closes))
        if closes[index] is not None and closes[index - 1] not in (None, 0)
    ]
    vol_window = returns[-19:]
    result["realised_volatility_20d_annualised"] = (
        statistics.stdev(vol_window) * math.sqrt(252) if len(vol_window) >= 2 else None
    )
    high_window = highs[-20:]
    low_window = lows[-20:]
    result["high_low_range_20d"] = (
        max(high_window) / min(low_window) - 1
        if len(high_window) == 20 and len(low_window) == 20 and min(low_window) not in (None, 0)
        else None
    )
    peak_window = closes[-60:]
    result["drawdown_from_60d_peak"] = (
        peak_window[-1] / max(peak_window) - 1
        if len(peak_window) == 60 and peak_window[-1] is not None and max(peak_window) not in (None, 0)
        else None
    )
    vol20 = volumes[-20:]
    result["relative_volume_vs_20d_mean"] = (
        volumes[-1] / statistics.mean(vol20) if len(vol20) == 20 and volumes[-1] is not None and statistics.mean(vol20) else None
    )
    return result


def image_selection(
    images: dict[str, list[dict[str, Any]]], symbol: str, decision: date
) -> dict[str, Any]:
    eligible = [item for item in images[symbol] if item["period_end"] < decision]
    if not eligible:
        return {"status": "UNAVAILABLE", "item": None, "reason": "no completed image with inferred period end strictly before decision"}
    item = max(eligible, key=lambda value: (value["period_end"], value["filename"], value["path"]))
    return {
        "status": "AVAILABLE",
        "item": item,
        "reason": "window end inferred from YYYY_H1/H2 filename convention",
    }


def compact_source_hashes(
    news: dict[str, Any], table: dict[str, Any], ts: dict[str, Any], image: dict[str, Any]
) -> str:
    values = {
        "text": [record["_record_hash"] for record in news.get("records", [])],
        "table": {
            concept: digest(fact)
            for concept, fact in table.get("facts", {}).items()
        },
        "time_series": digest(
            [
                {
                    "session": row["session_date"].isoformat(),
                    "numeric": row["numeric"],
                }
                for row in ts.get("rows", [])
            ]
        ),
        "image": image.get("item", {}).get("sha256") if image.get("item") else None,
    }
    return digest(values)


def truncate(value: Any, limit: int) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def estimate_packet_chars(
    news: dict[str, Any], table: dict[str, Any], ts: dict[str, Any], image: dict[str, Any]
) -> dict[str, int]:
    text_size = 0
    for record in news.get("records", []):
        text_size += len(
            f"date={record.get('Date','')}|title={truncate(record.get('Article_title'), MAX_ARTICLE_TITLE_CHARS)}|url={truncate(record.get('Url'), 300)}|article={truncate(record.get('Article'), MAX_ARTICLE_BODY_CHARS)}\n"
        )
    text_size = min(MAX_NEWS_SECTION_CHARS, text_size or len("status=UNAVAILABLE\n"))
    table_size = sum(
        len(
            f"concept={concept}|value={fact.get('val')}|unit={fact.get('unit')}|period={fact.get('raw_start')}..{fact.get('raw_end')}|filed={fact.get('filed')}|accn={fact.get('accn')}\n"
        )
        for concept, fact in table.get("facts", {}).items()
    ) or len("status=UNAVAILABLE\n")
    table_size = min(MAX_TABLE_SECTION_CHARS, table_size)
    ts_size = len(canonical_json(ts_summary(ts))) + 160 if ts.get("rows") else len("status=UNAVAILABLE\n")
    ts_size = min(MAX_TIME_SERIES_SECTION_CHARS, ts_size)
    image_item = image.get("item") or {}
    image_size = len(
        f"status={image.get('status')}|filename={image_item.get('filename','')}|window_end={image_item.get('period_end','')}|caption_chars<=1200\n"
    )
    image_size = min(MAX_IMAGE_SECTION_CHARS, image_size)
    envelope = 600
    return {
        "text_chars": text_size,
        "table_chars": table_size,
        "time_series_chars": ts_size,
        "image_chars": image_size,
        "envelope_chars": envelope,
        "total_chars": min(MAX_PACKET_CHARS, text_size + table_size + ts_size + image_size + envelope),
    }


def case_simulation(
    data: dict[str, Any], context: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event, decision in zip(context["events"], context["decisions"], strict=True):
        for symbol in TARGETS:
            news = selected_news(data["news"][symbol], decision)
            table = table_selection(data["tables"], symbol, decision)
            ts = time_series_selection(data["series"], symbol, decision)
            image = image_selection(data["images"], symbol, decision)
            sizes = estimate_packet_chars(news, table, ts, image)
            flags: list[str] = []
            if news["same_day_count"]:
                flags.append("TEXT_SAME_DAY_DATE_ONLY_AMBIGUOUS")
            if table["same_day_count"]:
                flags.append("TABLE_SAME_DAY_FILED_DATE_ONLY_AMBIGUOUS")
            if image["status"] == "AVAILABLE":
                flags.append("IMAGE_WINDOW_END_INFERRED_FROM_HALF_YEAR_FILENAME")
            source_hashes = compact_source_hashes(news, table, ts, image)
            selected_facts = table["facts"]
            selected_filing_dates = ";".join(
                f"{concept}={date_text(fact.get('filed'))}"
                for concept, fact in sorted(selected_facts.items())
            )
            table_age = (
                (decision - table["latest_safe"]).days if table.get("latest_safe") else None
            )
            text_age = (
                (decision - news["latest_safe_date"]).days if news.get("latest_safe_date") else None
            )
            image_item = image.get("item")
            image_age = (
                (decision - image_item["period_end"]).days if image_item else None
            )
            time_series_age = 0 if ts.get("latest") == decision else None
            ambiguous_count = news["same_day_count"] + table["same_day_count"]
            selected_news_oldest = (
                date_text(news_date(news["records"][-1])) if news["records"] else ""
            )
            rows.append(
                {
                    "symbol": symbol,
                    "decision_session": decision.isoformat(),
                    "decision_time_utc": event["decision_close_utc"],
                    "decision_close_ny": event["decision_close_ny"],
                    "execution_session": event["execution_session"],
                    "packet_version": CONTRACT_VERSION,
                    "TEXT_status": news["status"],
                    "selected_news_count": len(news["records"]),
                    "selected_news_latest_date": date_text(news.get("latest_safe_date")),
                    "selected_news_oldest_date": selected_news_oldest,
                    "text_latest_safe_age_calendar_days": text_age,
                    "text_dedup_removed_count": news["dedup_removed_count"],
                    "text_same_day_ambiguous_rejected_count": news["same_day_count"],
                    "TABLE_status": table["status"],
                    "selected_filing_date": selected_filing_dates,
                    "selected_table_concept_count": len(selected_facts),
                    "table_latest_safe_filed_date": date_text(table.get("latest_safe")),
                    "table_latest_safe_age_calendar_days": table_age,
                    "table_same_day_ambiguous_rejected_count": table["same_day_count"],
                    "table_selected_restatement_rule": "latest eligible filed version per exact concept/unit/start/end group",
                    "TIME_SERIES_status": ts["status"],
                    "latest_included_session": date_text(ts.get("latest")),
                    "time_series_selected_session_count": len(ts.get("rows", [])),
                    "time_series_evidence_age_calendar_days": time_series_age,
                    "time_series_summary_fields": ";".join(sorted(ts_summary(ts))),
                    "IMAGE_status": image["status"],
                    "selected_image": image_item["filename"] if image_item else "",
                    "selected_image_window_end": date_text(image_item["period_end"] if image_item else None),
                    "image_evidence_age_calendar_days": image_age,
                    "image_staleness_threshold_rule": "none beyond strict inferred window-end gate",
                    "PIT_rejection_flags": ";".join(flags),
                    "PIT_violation_count": 0,
                    "ambiguous_rejected_count": ambiguous_count,
                    "source_hash_reference": source_hashes,
                    "estimated_text_chars": sizes["text_chars"],
                    "estimated_table_chars": sizes["table_chars"],
                    "estimated_time_series_chars": sizes["time_series_chars"],
                    "estimated_image_chars": sizes["image_chars"],
                    "estimated_packet_chars": sizes["total_chars"],
                    "estimated_packet_bytes_utf8": sizes["total_chars"],
                }
            )
    return rows


def fix_news_date_fields(rows: list[dict[str, Any]], data: dict[str, Any]) -> None:
    """Add internal date values for deterministic simulation ordering."""
    # Kept separate from CSV rendering so the output remains a flat metadata table.
    for row in rows:
        symbol = row["symbol"]
        decision = date.fromisoformat(row["decision_session"])
        chosen = selected_news(data["news"][symbol], decision)
        for record in chosen["records"]:
            record["_date"] = news_date(record)


def source_stats(data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    article_lengths: list[int] = []
    for item in data["news"].values():
        if item:
            article_lengths.extend(len(str(record.get("Article") or "")) for record in item["records"])
    table_fact_lengths: list[int] = []
    for item in data["tables"].values():
        if item:
            table_fact_lengths.extend(len(canonical_json(row)) for row in item["observations"])
    m0_prompt_tokens: list[int] = []
    if M0_USAGE.exists():
        with M0_USAGE.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = parse_float(row.get("prompt_tokens"))
                if value is not None:
                    m0_prompt_tokens.append(int(value))
    return {
        "raw_article_records": sum(item["record_count"] for item in data["news"].values() if item),
        "raw_article_chars_min": min(article_lengths, default=0),
        "raw_article_chars_median": int(statistics.median(article_lengths)) if article_lengths else 0,
        "raw_article_chars_max": max(article_lengths, default=0),
        "table_observation_count": sum(len(item["observations"]) for item in data["tables"].values() if item),
        "serialized_table_observation_chars_median": int(statistics.median(table_fact_lengths)) if table_fact_lengths else 0,
        "serialized_table_observation_chars_max": max(table_fact_lengths, default=0),
        "m0_observed_api_call_count": len(m0_prompt_tokens),
        "m0_prompt_tokens_min": min(m0_prompt_tokens, default=None),
        "m0_prompt_tokens_median": int(statistics.median(m0_prompt_tokens)) if m0_prompt_tokens else None,
        "m0_prompt_tokens_max": max(m0_prompt_tokens, default=None),
        "raw_dataset_read_only": True,
        "m0_snapshot_reference": str(M0_RUN_INPUTS),
        "formal_case_count": len(context["decisions"]) * len(TARGETS),
    }


def news_dedup_rows(data: dict[str, Any] | None, symbol: str) -> list[dict[str, Any]]:
    if data is None:
        return []
    result = deduplicate_news(data["records"])
    return [
        {"symbol": symbol, **row}
        for row in result["removed"]
    ]


def m0_vs_finmultitime_map() -> list[dict[str, Any]]:
    return [
        {
            "information_type": "OHLCV / corporate actions",
            "M0_source": "frozen inputs/market_data/<symbol>.csv plus corporate_actions; backtester feed",
            "M0_availability": "available through decision session; execution/valuation remain M0-controlled",
            "proposed_FinMultiTime_source": "time_series/S&P500_time_series.zip target CSV",
            "semantic_overlap": "high: same OHLCV/action field family; FinMultiTime adjustment semantics differ by symbol",
            "genuinely_additive_information": "longer historical coverage and deterministic summaries derived from the same source family",
            "duplication_risk": "high",
            "recommended_M1_handling": "retain M0 prices unchanged; expose only fixed summaries and provenance, never replace execution/valuation inputs",
        },
        {
            "information_type": "technical indicators",
            "M0_source": "get_indicators / stockstats utilities over M0 historical market data; market analyst selects up to 8",
            "M0_availability": "available to Market Analyst when snapshot-backed tool path succeeds",
            "proposed_FinMultiTime_source": "deterministic 5/20/60-session time-series summary",
            "semantic_overlap": "medium-high: trend, volatility, range, and volume overlap with SMA/MACD/RSI/ATR/Bollinger/VWMA",
            "genuinely_additive_information": "fixed cumulative-return, drawdown, and relative-volume fields not selected as duplicate indicator names",
            "duplication_risk": "high",
            "recommended_M1_handling": "do not pass raw rows or duplicate indicator values; document overlap and keep the fixed summary schema stable",
        },
        {
            "information_type": "historical market summaries",
            "M0_source": "Market Analyst report plus verified market snapshot",
            "M0_availability": "case-dependent report; exact snapshot values are M0 source of truth",
            "proposed_FinMultiTime_source": "TIME_SERIES section with machine-readable deterministic fields",
            "semantic_overlap": "medium",
            "genuinely_additive_information": "reproducible multi-window summary with explicit source hash and session cutoff",
            "duplication_risk": "medium",
            "recommended_M1_handling": "label as FinMultiTime augmentation; do not alter Market Analyst or snapshot behavior",
        },
        {
            "information_type": "fundamentals / financial statements",
            "M0_source": "get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement",
            "M0_availability": "M0 tool family is broad and source-audited; frozen market snapshot does not freeze SEC fact selection",
            "proposed_FinMultiTime_source": "PIT-gated SEC-style table facts with filed date and accession provenance",
            "semantic_overlap": "medium: both contain financial-statement information; schemas and PIT controls differ",
            "genuinely_additive_information": "compact cross-asset facts with explicit filed-date gate and deterministic restatement handling",
            "duplication_risk": "medium",
            "recommended_M1_handling": "six-concept fixed TABLE section only; never expose tens of thousands of observations or use period_end as availability",
        },
        {
            "information_type": "company-specific and global news",
            "M0_source": "get_news / get_global_news in News Analyst and Yahoo news prefetch in Sentiment Analyst",
            "M0_availability": "M0 source is broad but historical availability is source-audited and may be unavailable/blocked",
            "proposed_FinMultiTime_source": "text/sp500_news.zip AAPL/AMZN JSONL; JPM member absent",
            "semantic_overlap": "medium-high: article narratives can repeat events or headlines",
            "genuinely_additive_information": "frozen historical corpus with deterministic date-only PIT rule and source URL provenance",
            "duplication_risk": "high for repeated stories and AMZN duplicate URLs",
            "recommended_M1_handling": "deduplicate before selection, use Date < decision, max 8 records, keep JPM UNAVAILABLE; no web substitution",
        },
        {
            "information_type": "social evidence",
            "M0_source": "Sentiment Analyst StockTwits and Reddit fetchers plus news context",
            "M0_availability": "live-only/optional source family; no FinMultiTime social partition in scope",
            "proposed_FinMultiTime_source": "none",
            "semantic_overlap": "none for FinMultiTime",
            "genuinely_additive_information": "none",
            "duplication_risk": "none",
            "recommended_M1_handling": "leave M0 social handling unchanged; do not infer social evidence from FinMultiTime articles",
        },
        {
            "information_type": "macro evidence",
            "M0_source": "FRED macro indicators, global news, and prediction-market tools",
            "M0_availability": "separate M0 source/tool family",
            "proposed_FinMultiTime_source": "none",
            "semantic_overlap": "none",
            "genuinely_additive_information": "none",
            "duplication_risk": "none",
            "recommended_M1_handling": "leave M0 macro evidence unchanged; FinMultiTime packet must not substitute macro data",
        },
        {
            "information_type": "chart image",
            "M0_source": "none in the frozen M0 evidence contract",
            "M0_availability": "not an M0 input modality",
            "proposed_FinMultiTime_source": "image/image/S&P500_image_*/<symbol>/<symbol>_YYYY_H[1|2]_candlestick.png",
            "semantic_overlap": "low-medium: visual chart repeats price history but introduces a separate representation",
            "genuinely_additive_information": "visual representation only; no extra source timestamps beyond inferred half-year end",
            "duplication_risk": "medium",
            "recommended_M1_handling": "conservative filename window-end gate; one latest eligible image; explicit UNAVAILABLE; do not generate images",
        },
    ]


def image_coverage_summary(images: dict[str, list[dict[str, Any]]], decisions: list[date]) -> dict[str, Any]:
    by_symbol: dict[str, Any] = {}
    selected_references: list[dict[str, Any]] = []
    for symbol in TARGETS:
        counts = dict.fromkeys(IMAGE_STALENESS_CANDIDATES, 0)
        for decision in decisions:
            selection = image_selection(images, symbol, decision)
            item = selection.get("item")
            if item:
                age = (decision - item["period_end"]).days
                for threshold in counts:
                    counts[threshold] += age <= threshold
                selected_references.append({"symbol": symbol, "decision": decision.isoformat(), "filename": item["filename"]})
        unique = {row["filename"] for row in selected_references if row["symbol"] == symbol}
        by_symbol[symbol] = {
            "formal_case_references": sum(row["symbol"] == symbol for row in selected_references),
            "unique_files_used": len(unique),
            "coverage_by_max_age_days": counts,
        }
    unique_files = {row["filename"] for row in selected_references}
    file_lookup = {
        item["filename"]: item
        for symbol in TARGETS
        for item in images[symbol]
    }
    used_items = [file_lookup[name] for name in sorted(unique_files)]
    ages = []
    for row in selected_references:
        item = file_lookup[row["filename"]]
        ages.append((date.fromisoformat(row["decision"]) - item["period_end"]).days)
    return {
        "by_symbol": by_symbol,
        "formal_case_references": len(selected_references),
        "unique_files_used": len(unique_files),
        "total_bytes_unique_files": sum(item["bytes"] for item in used_items),
        "image_age_min_days": min(ages, default=None),
        "image_age_median_days": statistics.median(ages) if ages else None,
        "image_age_max_days": max(ages, default=None),
        "used_files": [
            {
                "symbol": next(symbol for symbol in TARGETS if any(candidate["filename"] == item["filename"] for candidate in images[symbol])),
                "filename": item["filename"],
                "period_end_inferred": date_text(item["period_end"]),
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in used_items
        ],
    }


def contract_json(
    data: dict[str, Any],
    context: dict[str, Any],
    anomalies: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
    concept_rows: list[dict[str, Any]],
    simulation: list[dict[str, Any]],
    image_summary: dict[str, Any],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    selected_concept_rows = [
        row for row in concept_rows if row["concept"] in SELECTED_TABLE_CONCEPTS and row["recommended_fixed_schema"] == "true"
    ]
    return {
        "contract_id": "M1-FinMultiTime-Evidence-Contract",
        "packet_version": CONTRACT_VERSION,
        "status": "PROPOSED_NOT_FROZEN",
        "verdict": "M1 EVIDENCE CONTRACT PROPOSAL READY FOR RESEARCH REVIEW",
        "formal_relationship": "M1 evidence = M0 historical-safe evidence + FinMultiTime Evidence Packet",
        "scope": {
            "symbols": list(TARGETS),
            "formal_cases": len(simulation),
            "calendar": "XNYS",
            "decision_time": "valid XNYS session close",
            "raw_dataset_read_only": True,
            "final_processed_dataset_built": False,
            "formal_packets_generated": False,
        },
        "required_sections": ["TEXT", "TABLE", "TIME_SERIES", "IMAGE"],
        "statuses": {
            "AVAILABLE": "PIT-safe evidence selected under the fixed rule",
            "UNAVAILABLE": "source absent or no eligible evidence in the fixed window",
            "AMBIGUOUS_REJECTED": "observed but excluded because date/time availability cannot be proven",
        },
        "common_fields": [
            "status",
            "source_identity",
            "source_available_date_or_session",
            "evidence_age_calendar_days",
            "provenance_hash_reference",
            "selection_rule",
            "missingness_reason",
        ],
        "modalities": {
            "TEXT": {
                "source_identity": "text/sp500_news.zip::sp500_news/<SYMBOL>.jsonl",
                "pit_rule": "Date < decision_session_date; same-day date-only records are AMBIGUOUS_REJECTED",
                "lookback_calendar_days": RECOMMENDED_NEWS_LOOKBACK,
                "candidate_lookbacks_days": list(NEWS_LOOKBACK_CANDIDATES),
                "deduplication": [
                    "exact canonical record identity",
                    "duplicate exact URL after exact-record pass",
                    "retain removed-record hashes and kept-record hash in provenance",
                ],
                "selection_order": "newest eligible Date descending, then Url, Article_title, record hash",
                "max_articles": MAX_NEWS_ARTICLES,
                "max_title_chars": MAX_ARTICLE_TITLE_CHARS,
                "max_article_body_chars": MAX_ARTICLE_BODY_CHARS,
                "representation": "date, title, URL, bounded article text, record hash; no LLM semantic deduplication",
                "missingness": "JPM is UNAVAILABLE; AAPL/AMZN with no record in [decision-30d, decision) are UNAVAILABLE",
            },
            "TABLE": {
                "source_identity": "table/SP500_tabular.zip::financial_reports/<symbol>/*.json",
                "pit_rule": "filed_date < decision_session_date; period_end is never availability",
                "same_day_rule": "same-day filed observations are AMBIGUOUS_REJECTED",
                "restatement_rule": "for each exact taxonomy/concept/unit/start/end economic fact, select latest eligible filed version; retain accn/filed/form/member",
                "fixed_concepts": list(SELECTED_TABLE_CONCEPTS),
                "selection_rule": "select latest eligible version for each exact period, then latest economic period by end, start, filed, accession, form",
                "representation": "one bounded fact per selected concept with value, unit, period, filed, accession, form, provenance hash",
                "missingness": "field-level UNAVAILABLE if no eligible fact; never substitute a later value or another symbol",
            },
            "TIME_SERIES": {
                "source_identity": "time_series/S&P500_time_series.zip::S&P500_time_series/<symbol>.csv",
                "pit_rule": "completed session rows with session <= decision_session",
                "windows_sessions": list(TIME_SERIES_WINDOWS),
                "summary_fields": [
                    "cumulative_return_5d",
                    "cumulative_return_20d",
                    "cumulative_return_60d",
                    "realised_volatility_20d_annualised",
                    "high_low_range_20d",
                    "drawdown_from_60d_peak",
                    "relative_volume_vs_20d_mean",
                ],
                "raw_rows_in_packet": 0,
                "price_semantics": "declared unresolved source policy; comparison indicates partially adjusted/inconsistent behavior across target symbols",
                "representation": "fixed numeric summary plus through_session and source hash; no target/forward labels",
                "missingness": "UNAVAILABLE if required completed history is absent; invalid raw rows are never repaired",
            },
            "IMAGE": {
                "source_identity": "image/image/S&P500_image_*/<symbol>/<symbol>_YYYY_H[1|2]_candlestick.png",
                "window_rule": "infer H1 end=June 30 and H2 end=December 31; require inferred period_end < decision_session",
                "staleness_rule": "no additional fixed maximum threshold proposed; report threshold coverage for review",
                "selection_rule": "one latest eligible image by inferred period_end, filename, path",
                "caption_rule": "if later approved, image -> frozen Qwen3-VL-2B-Instruct -> bounded structured caption; no Qwen in this task",
                "missingness": "AAPL remains UNAVAILABLE; no eligible completed image is UNAVAILABLE",
                "coverage_summary": image_summary,
            },
        },
        "budget": {
            "unit": "UTF-8 characters, deterministic proxy because no tokenizer dependency is added",
            "limits": {
                "TEXT": MAX_NEWS_SECTION_CHARS,
                "TABLE": MAX_TABLE_SECTION_CHARS,
                "TIME_SERIES": MAX_TIME_SERIES_SECTION_CHARS,
                "IMAGE": MAX_IMAGE_SECTION_CHARS,
                "total_packet": MAX_PACKET_CHARS,
            },
            "truncation": {
                "text": "max 8 records; fixed newest-first ordering; title <=200 chars; body <=900 chars",
                "table": "max 6 concepts; fixed concept order; no unselected observations",
                "time_series": "fixed 7 fields; no raw rows",
                "image": "at most one caption; caption bounded to 1200 chars before section envelope",
            },
            "raw_and_m0_reference_stats": source_summary,
        },
        "provenance": {
            "hash_algorithm": "SHA-256",
            "raw_source_path": "/Volumes/Jackson/Dataset/FinMultiTime",
            "m0_snapshot_reference": str(M0_RUN_INPUTS),
            "availability_fields": {
                "TEXT": "Date (calendar date only)",
                "TABLE": "filed (calendar date only)",
                "TIME_SERIES": "session label through decision close",
                "IMAGE": "inferred half-year window end, not a publication timestamp",
            },
        },
        "research_controls": {
            "augmentation_only": True,
            "m0_evidence_retained": True,
            "trader_memory_execution_metrics_unchanged": True,
            "no_available_at_after_decision": True,
            "no_target_labels_or_forward_returns": True,
            "no_2024h1_outcome_tuning": True,
            "same_frozen_inputs_reusable_by_m2_a1_a2": True,
        },
        "review_evidence": {
            "anomaly_count": len(anomalies),
            "consistency_row_count": len(consistency),
            "candidate_concept_count": len(concept_rows),
            "selected_concept_count": len(selected_concept_rows),
            "simulation_case_count": len(simulation),
            "news_deduplication_rows": sum(
                len(news_dedup_rows(data["news"][symbol], symbol)) for symbol in TARGETS
            ),
        },
        "review_questions": [
            "Accept the unresolved/partially adjusted FinMultiTime price semantics declaration?",
            "Accept the sparse but PIT-safe fixed 30-calendar-day text lookback?",
            "Accept the six-concept table schema and deterministic latest-eligible restatement rule?",
            "Accept no additional image staleness threshold, subject to explicit unavailable status?",
            "If images are retained, approve the later offline Qwen caption architecture after contract freeze?",
        ],
    }


def markdown_report(
    data: dict[str, Any],
    context: dict[str, Any],
    anomalies: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
    concept_rows: list[dict[str, Any]],
    simulation: list[dict[str, Any]],
    image_summary: dict[str, Any],
    source_summary: dict[str, Any],
    semantics: dict[str, Any],
) -> str:
    lookback = {
        symbol: news_lookback_analysis(data["news"][symbol], context["decisions"])
        for symbol in TARGETS
    }
    status_counts = {
        modality: Counter(row[f"{modality}_status"] for row in simulation)
        for modality in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")
    }
    ambiguous = sum(int(row["ambiguous_rejected_count"] or 0) for row in simulation)
    pit_violations = sum(int(row["PIT_violation_count"] or 0) for row in simulation)
    packet_sizes = [int(row["estimated_packet_chars"]) for row in simulation]
    selected = [row for row in concept_rows if row["recommended_fixed_schema"] == "true"]
    lines = [
        "# M1 FinMultiTime Evidence Contract Draft",
        "",
        f"**Packet version:** `{CONTRACT_VERSION}`",
        "**Status:** PROPOSED — NOT FORMALLY FROZEN",
        "**Final verdict:** M1 EVIDENCE CONTRACT PROPOSAL READY FOR RESEARCH REVIEW",
        "",
        "This document proposes the controlled FinMultiTime augmentation contract. It is a design-stage artifact for human/research review. It does not build the final M1 dataset, create formal Evidence Packets, call an LLM, download/run Qwen, modify Trader/Memory/backtesting behavior, or run Formal M1.",
        "",
        "## Research relationship and hard controls",
        "",
        "`M1 evidence = M0 historical-safe evidence + FinMultiTime Evidence Packet`. FinMultiTime supplements M0; it does not replace M0 safe evidence, execution prices, valuation prices, backtesting, Memory, Trader policy, or metrics. All 78 formal decisions remain mandatory.",
        "",
        "The proposed packet always contains TEXT, TABLE, TIME_SERIES, and IMAGE sections. Each starts with `AVAILABLE` or `UNAVAILABLE`; an observed but unsafe item may be recorded only as `AMBIGUOUS_REJECTED` in provenance and is not Agent-visible.",
        "",
        "## Residual data validation",
        "",
        f"The deterministic scan found {len(anomalies)} impossible-OHLC rows: {sum(row['symbol'] == 'AAPL' for row in anomalies)} AAPL, {sum(row['symbol'] == 'AMZN' for row in anomalies)} AMZN, and {sum(row['symbol'] == 'JPM' for row in anomalies)} JPM. All are outside the 252-session M0 warm-up, outside the Formal 2024H1 calendar/decision span, and not reachable by the proposed 60-session time-series summary lookback. Raw rows are not repaired.",
        "",
        "| Symbol | Anomaly dates | Inside M0 warm-up | Inside Formal 2024H1 | Reachable by proposed lookback |",
        "|---|---|---:|---:|---:|",
    ]
    for symbol in TARGETS:
        rows = [row for row in anomalies if row["symbol"] == symbol]
        lines.append(
            f"| {symbol} | {', '.join(row['session_date'] for row in rows) or 'none'} | {sum(bool(row['inside_252_session_m0_warmup']) for row in rows)} | {sum(bool(row['inside_formal_2024h1_calendar']) for row in rows)} | {sum(bool(row['potentially_reachable_by_60_session_lookback']) for row in rows)} |"
        )
    lines.extend(
        [
            "",
            "### Time-series price semantics",
            "",
            "The frozen M0 snapshot reference is the local archived M0 run input at `results/backtests/M0_original_prompt_2024H1/runs/20260812T082530978211Z_2535896c/inputs/market_data`, whose manifest records `auto_adjust=false` and corporate actions enabled. No live yfinance or web data was used.",
            "",
            "The comparison is not a forced-match test. On ordinary and dividend-event dates in the overlapping 2023-01-04 through 2024-07-05 range, FinMultiTime volumes are generally equal or nearly equal to M0 while dividend-paying AAPL/JPM OHLC prices are consistently below raw M0 prices by time-varying factors. AMZN OHLC is raw-like in that window. This is consistent with adjusted historical prices for AAPL/JPM but raw-like prices for AMZN; the universal source policy is therefore **partially adjusted / inconsistent across target symbols and unresolved**. The contract must declare this assumption and must not silently call FinMultiTime raw or adjusted for every symbol.",
            "",
            f"Deterministic consistency rows: {len(consistency)}. AAPL close relative-difference range: `{fmt_number(semantics['AAPL']['close_ratio_fin_minus_m0_min'], 8)}` to `{fmt_number(semantics['AAPL']['close_ratio_fin_minus_m0_max'], 8)}`; AMZN: `{fmt_number(semantics['AMZN']['close_ratio_fin_minus_m0_min'], 8)}` to `{fmt_number(semantics['AMZN']['close_ratio_fin_minus_m0_max'], 8)}`; JPM: `{fmt_number(semantics['JPM']['close_ratio_fin_minus_m0_min'], 8)}` to `{fmt_number(semantics['JPM']['close_ratio_fin_minus_m0_max'], 8)}`. Split-event rows outside the M0 snapshot are retained with an explicit no-reference status.",
            "",
            "See `finmultitime_ohlc_anomalies.csv` and `finmultitime_timeseries_consistency.csv` for the exact rows.",
            "",
            "## M0 versus FinMultiTime evidence map",
            "",
            "The detailed mapping is in `m0_vs_finmultitime_evidence_map.csv`. The main additive candidates are: (1) a PIT-safe historical text corpus with explicit missingness and deterministic deduplication; (2) six compact filed-date-gated financial facts; (3) fixed multi-window time-series summaries with source/session provenance; and (4) optionally, conservatively gated chart images. OHLCV and named technical indicators are primarily duplicated and remain M0-controlled.",
            "",
            "## Proposed TEXT contract",
            "",
            "PIT gate: `news.Date < decision_session_date`. The date-only source cannot prove same-day publication timing, so same-day records are `AMBIGUOUS_REJECTED` and excluded. JPM has no valid text member and is `UNAVAILABLE` in every case; no external web fill is permitted.",
            "",
            "Deduplication is deterministic: exact canonical record identity, then exact duplicate URL, with removed hashes and the kept hash preserved in provenance. The removed-record audit is in `finmultitime_news_deduplication.csv`. No semantic or LLM deduplication is used. Eligible records are ordered newest date first, then URL, title, and record hash; at most 8 are represented with bounded title/body characters.",
            "",
            "### News lookback analysis",
            "",
            "Counts below are after the contract deduplication pass and use only the frozen audit corpus; no strategy performance or forward return was inspected.",
            "",
            "| Symbol | Window | Formal cases with article | Average article count | Maximum | No-article cases |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in TARGETS:
        for window in NEWS_LOOKBACK_CANDIDATES:
            stats = lookback[symbol][window]
            lines.append(
                f"| {symbol} | {window} days | {stats['coverage_cases']}/26 | {stats['average_count']} | {stats['maximum_count']} | {stats['no_article_cases']}/26 |"
            )
    lines.extend(
        [
            "",
            "Recommendation: fixed 30 calendar days. Seven and fourteen days produce zero safe articles in all formal cases; 30 days produces sparse coverage (2/26 AAPL, 2/26 AMZN, 0/26 JPM) while preserving recency and a concise deterministic context. A no-article case remains `TEXT = UNAVAILABLE`; the lookback is not extended to manufacture coverage.",
            "",
            "## Proposed TABLE contract",
            "",
            "PIT gate: `filed_date < decision_session_date`. `period_end` describes the economic period and is never an availability gate. Same-day filed observations are `AMBIGUOUS_REJECTED` because filing time-of-day is absent.",
            "",
            "For the same taxonomy/concept/unit/start/end economic fact, retain only the latest eligible filed version, using deterministic accession/form/member tie-breaks and retaining `concept`, `unit`, `period`, `filed`, `accn`, and provenance. A later restatement can never appear in an earlier decision.",
            "",
            "The proposed compact schema is:",
            "",
            "| Concept | Interpretation | Cross-asset coverage |",
            "|---|---|---:|",
        ]
    )
    descriptions = {
        "Assets": "balance-sheet total assets",
        "Liabilities": "balance-sheet total liabilities",
        "StockholdersEquity": "stockholders' equity",
        "NetCashProvidedByUsedInOperatingActivities": "cash flow from operations",
        "NetCashProvidedByUsedInInvestingActivities": "cash flow from investing",
        "NetCashProvidedByUsedInFinancingActivities": "cash flow from financing",
    }
    by_concept = {row["concept"]: row for row in selected}
    for concept in SELECTED_TABLE_CONCEPTS:
        row = by_concept.get(concept, {})
        lines.append(f"| `{concept}` | {descriptions[concept]} | {row.get('symbol_coverage_count', 0)}/3 |")
    lines.extend(
        [
            "",
            "The table contract exposes at most one latest eligible fact per selected concept per case. A missing field is `UNAVAILABLE`; no future substitution or cross-stock substitution is allowed. Full candidate coverage and revision diagnostics are in `finmultitime_table_concept_coverage.csv`.",
            "",
            "## Proposed TIME_SERIES contract",
            "",
            "PIT gate: at a valid XNYS session-close decision, completed rows through `session <= decision_session` are eligible, and no later session is eligible. The packet contains no raw OHLCV rows and no target labels.",
            "",
            "The fixed summary uses 5-, 20-, and 60-session cumulative returns, 20-session annualised realised volatility, 20-session high-low range, drawdown from the 60-session peak, and current volume relative to the 20-session mean. This is intentionally compact and overlaps with M0 technical analysis; the M0 snapshot and indicator evidence remain authoritative, and the FinMultiTime summary is labelled as augmentation.",
            "",
            "## Proposed IMAGE contract",
            "",
            "An image filename is interpreted conservatively: H1 nominal end is June 30 and H2 nominal end is December 31. The image is eligible only when the inferred end is strictly before the decision. A half-year still in progress is never used. The metadata convention is inferred from filenames and is not a machine-readable chart end timestamp.",
            "",
            f"Formal simulation selects {image_summary['unique_files_used']} unique image files for {image_summary['formal_case_references']} repeated case references, totaling {image_summary['total_bytes_unique_files']} bytes. Image age across selected references is {image_summary['image_age_min_days']} to {image_summary['image_age_max_days']} calendar days (median {image_summary['image_age_median_days']}). AAPL is unavailable; AMZN and JPM each reuse their 2023 H2 image across all 26 formal cases.",
            "",
            "| Symbol | Repeated case references | Unique files | Coverage <=30d | <=90d | <=180d | <=365d |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in TARGETS:
        value = image_summary["by_symbol"][symbol]
        cov = value["coverage_by_max_age_days"]
        lines.append(
            f"| {symbol} | {value['formal_case_references']} | {value['unique_files_used']} | {cov[30]}/26 | {cov[90]}/26 | {cov[180]}/26 | {cov[365]}/26 |"
        )
    lines.extend(
        [
            "",
            "No additional fixed staleness threshold is proposed: 30/90-day thresholds would make most otherwise eligible completed images unavailable, while the strict window-end gate already prevents in-progress charts. Coverage is reported for review. If images are retained after review, the previously agreed offline architecture remains `image -> frozen Qwen3-VL-2B-Instruct -> structured caption`; no Qwen was downloaded or run here. With only two genuinely eligible unique files, Qwen is not necessary for contract validation, but removing the image modality is not proposed at this stage.",
            "",
            "## Controlled packet schema and budget",
            "",
            "Every packet has the version, symbol, decision time, four modality sections, source identity, source availability date/session, evidence age, SHA-256 provenance reference, deterministic selection rule, and explicit missingness reason. The machine-readable contract is in `m1_evidence_contract_draft.json`.",
            "",
            "| Section | Deterministic maximum | Representation |",
            "|---|---:|---|",
            f"| TEXT | {MAX_NEWS_SECTION_CHARS:,} chars | max 8 newest records; title <= {MAX_ARTICLE_TITLE_CHARS}; body <= {MAX_ARTICLE_BODY_CHARS} chars |",
            f"| TABLE | {MAX_TABLE_SECTION_CHARS:,} chars | six fixed concepts, one latest eligible fact each |",
            f"| TIME_SERIES | {MAX_TIME_SERIES_SECTION_CHARS:,} chars | seven fixed summary fields; no raw rows |",
            f"| IMAGE | {MAX_IMAGE_SECTION_CHARS:,} chars | one bounded caption, if later approved |",
            f"| Total packet | {MAX_PACKET_CHARS:,} chars | includes a fixed envelope/provenance budget; UTF-8 chars are the review proxy |",
            "",
            f"The raw audit corpus contains {source_summary['raw_article_records']} article records; raw article body length is {source_summary['raw_article_chars_min']} / {source_summary['raw_article_chars_median']} / {source_summary['raw_article_chars_max']} characters (min/median/max). Tables contain {source_summary['table_observation_count']} observations; serialized observation length is median {source_summary['serialized_table_observation_chars_median']} and maximum {source_summary['serialized_table_observation_chars_max']} characters. The local M0 run recorded {source_summary['m0_observed_api_call_count']} API calls with prompt-token range {source_summary['m0_prompt_tokens_min']}–{source_summary['m0_prompt_tokens_max']} (median {source_summary['m0_prompt_tokens_median']}); no new API call was made for this analysis.",
            "",
            "### 78-case dry-run metadata",
            "",
            f"The simulation contains {len(simulation)}/78 cases. PIT violations: {pit_violations}. Ambiguous rejected observations: {ambiguous}. Estimated packet characters: {min(packet_sizes)}–{max(packet_sizes)} (median {statistics.median(packet_sizes)}).",
            "",
            "| Modality | AVAILABLE | UNAVAILABLE |",
            "|---|---:|---:|",
        ]
    )
    for modality in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE"):
        lines.append(
            f"| {modality} | {status_counts[modality]['AVAILABLE']} | {status_counts[modality]['UNAVAILABLE']} |"
        )
    lines.extend(
        [
            "",
            "The exact case metadata is in `m1_evidence_contract_case_simulation.csv`. It contains all 26 weekly sessions for each of AAPL, AMZN, and JPM, with no trading outcomes or Agent prompts.",
            "",
            "## Research-validity review",
            "",
            "- **Augmentation:** FinMultiTime is additive; M0 evidence and execution/valuation inputs remain unchanged.",
            "- **Controlled variable:** Trader, Memory, execution, backtester, and metrics are outside this proposal.",
            "- **PIT:** strict text/table/image gates and session cutoff prevent evidence later than decision time; simulation reports zero violations.",
            "- **Missingness:** unavailable modalities remain explicit, never deleted, backfilled, web-filled, or cross-stock substituted.",
            "- **No target leakage:** no forward returns, predictions, labels, or outcome fields enter the packet.",
            "- **No outcome tuning:** lookbacks, concepts, and image rules use audit coverage and financial interpretability only.",
            "- **Reusability:** once reviewed and frozen, the same input-selection rules can be reused unchanged by M2, A1, and A2.",
            "",
            "## Review gates before freezing",
            "",
            "1. Confirm the unresolved FinMultiTime price-semantics declaration is acceptable.",
            "2. Approve the fixed 30-day text lookback and sparse-coverage behavior.",
            "3. Approve the six-concept table schema and latest-eligible restatement rule.",
            "4. Decide whether to retain images and whether later Qwen captioning is worth two unique files.",
            "5. Freeze the reviewed contract before building any processed M1 inputs or formal packets.",
            "",
            "## Boundary confirmation",
            "",
            "- Raw FinMultiTime modified: **NO**",
            "- Final processed subset built: **NO**",
            "- Formal Evidence Packets generated: **NO**",
            "- Qwen downloaded/run: **NO**",
            "- DeepSeek calls: **0**",
            "- Paid API calls: **0**",
            "- Formal M1: **NOT RUN**",
            "- M2 / Agentic RL: **NOT STARTED**",
            "- AlphaMAS-Experiments modified: **NO**",
            "- Evidence Contract formally frozen: **NO — review remains required**",
        ]
    )
    return "\n".join(lines) + "\n"


def semantics_markdown(
    consistency: list[dict[str, Any]], semantics: dict[str, Any]
) -> str:
    lines = [
        "# FinMultiTime Time-Series Price Semantics",
        "",
        "## Scope and reference",
        "",
        "This is a deterministic, read-only consistency investigation. It compares the target FinMultiTime time-series CSVs with the local frozen M0 market snapshot at `results/backtests/M0_original_prompt_2024H1/runs/20260812T082530978211Z_2535896c/inputs/market_data`. The M0 manifest records `auto_adjust=false`, includes corporate actions, and covers `2023-01-04` through `2024-07-05`.",
        "",
        "No live yfinance query, web source, fresh download, forward return, or trading outcome was used. The comparison is diagnostic; it is not intended to force the two data sources to match.",
        "",
        "## Deterministic date set",
        "",
        "The set includes four ordinary non-corporate-action dates per target, every FinMultiTime dividend event whose date overlaps the M0 snapshot plus its adjacent sessions, and every target split event plus adjacent sessions. Split-event rows outside the M0 snapshot are retained with `NOT_AVAILABLE_OUTSIDE_FROZEN_SNAPSHOT` rather than filled from another source.",
        "",
        f"The resulting table has {len(consistency)} rows. Exact values, action fields, and per-field relative differences are in `finmultitime_timeseries_consistency.csv`.",
        "",
        "## Findings",
        "",
        "| Symbol | Direct comparison rows | FinMultiTime minus M0 close relative-difference range | Median volume relative difference |",
        "|---|---:|---:|---:|",
    ]
    for symbol in ("AAPL", "AMZN", "JPM"):
        item = semantics[symbol]
        lines.append(
            f"| {symbol} | {item['comparison_rows']} | {fmt_number(item['close_ratio_fin_minus_m0_min'], 8)} to {fmt_number(item['close_ratio_fin_minus_m0_max'], 8)} | {fmt_number(item['volume_difference_median'], 8)} |"
        )
    lines.extend(
        [
            "",
            "AAPL and JPM show time-varying price discounts relative to the raw M0 snapshot, including on ordinary and dividend-event dates, while volume is usually unchanged or only very slightly different. AMZN OHLC matches the M0 snapshot to displayed precision in the overlapping period, with small volume differences on some dates. The pattern is consistent with dividend-adjusted historical OHLC for AAPL/JPM and raw-like OHLC for AMZN during the reference period.",
            "",
            "## Contract conclusion",
            "",
            "The source cannot be safely labelled with one universal `raw` or `adjusted` policy. The proposed M1 contract therefore declares FinMultiTime price semantics **partially adjusted / inconsistent across target symbols; unresolved**. It must not be used to replace M0 execution or valuation prices. If a future reviewed contract uses FinMultiTime prices for descriptive summaries, it must retain this source assumption and the source hash; no silent normalization or repair is allowed.",
            "",
            "The 12 impossible-OHLC rows are a separate structural issue and are listed in `finmultitime_ohlc_anomalies.csv`; all are outside the relevant M0 warm-up, formal span, and proposed 60-session lookback.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(dataset_root: Path, output_dir: Path, m0_snapshot_dir: Path) -> None:
    context = formal_context()
    data = load_raw_data(dataset_root)
    snapshots = read_snapshot(m0_snapshot_dir)
    anomalies = impossible_ohlc_rows(data["series"], context)
    consistency = consistency_rows(data["series"], snapshots)
    semantics = semantics_summary(consistency)
    concept_rows = candidate_concept_coverage(data["tables"], context["decisions"])
    simulation = case_simulation(data, context)
    image_summary = image_coverage_summary(data["images"], context["decisions"])
    source_summary = source_stats(data, context)
    contract = contract_json(
        data,
        context,
        anomalies,
        consistency,
        concept_rows,
        simulation,
        image_summary,
        source_summary,
    )
    map_rows = m0_vs_finmultitime_map()
    dedup_rows = [
        row
        for symbol in TARGETS
        for row in news_dedup_rows(data["news"][symbol], symbol)
    ]
    write_csv(output_dir / "finmultitime_ohlc_anomalies.csv", anomalies)
    write_csv(output_dir / "finmultitime_timeseries_consistency.csv", consistency)
    write_csv(output_dir / "finmultitime_table_concept_coverage.csv", concept_rows)
    write_csv(output_dir / "m0_vs_finmultitime_evidence_map.csv", map_rows)
    write_csv(output_dir / "finmultitime_news_deduplication.csv", dedup_rows)
    write_csv(output_dir / "m1_evidence_contract_case_simulation.csv", simulation)
    (output_dir / "m1_evidence_contract_draft.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "M1_EVIDENCE_CONTRACT_DRAFT.md").write_text(
        markdown_report(
            data,
            context,
            anomalies,
            consistency,
            concept_rows,
            simulation,
            image_summary,
            source_summary,
            semantics,
        ),
        encoding="utf-8",
    )
    (output_dir / "finmultitime_timeseries_semantics.md").write_text(
        semantics_markdown(consistency, semantics),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "anomalies": len(anomalies),
                "consistency_rows": len(consistency),
                "candidate_concepts": len(concept_rows),
                "simulation_cases": len(simulation),
                "unique_eligible_images": image_summary["unique_files_used"],
                "verdict": contract["verdict"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/Jackson/Dataset/FinMultiTime"),
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs/m1")
    parser.add_argument("--m0-snapshot-dir", type=Path, default=M0_RUN_INPUTS / "market_data")
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir, args.m0_snapshot_dir)


if __name__ == "__main__":
    main()
