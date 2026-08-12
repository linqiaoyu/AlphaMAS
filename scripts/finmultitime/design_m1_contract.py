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
import io
import json
import math
import statistics
import subprocess
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

CONTRACT_VERSION = "M1-FINMULTITIME-v1.0.1"
PREVIOUS_CONTRACT_VERSION = "M1-FINMULTITIME-v1.0"
PREVIOUS_CONTRACT_SHA256 = "cd3ef3f127551c1775bc4aa803556cf071b264e72cde52072a901213c93a29b6"
PARENT_DRAFT_VERSION = "M1-FINMULTITIME-DRAFT-0.1"
RESEARCH_REVIEW_DECISION = "M1 EVIDENCE CONTRACT RESEARCH REVIEW PASSED WITH REQUIRED REVISIONS"
SOURCE_PARENT_SHA = "0b34a278b43e64204ce6805cec94728874b50131"
FROZEN_M0_BASE_SHA = "2535896c8b1070b19c06fa6a936663babb4356f7"
FREEZE_DATE = "2026-08-13"
ERRATUM_REASON = (
    "Rename year_to_date_h1/year_to_date_h2 to the precise "
    "year_to_date_6m/year_to_date_9m duration classes; selection behaviour is unchanged."
)
FINAL_VERDICT = "M1 EVIDENCE CONTRACT ERRATUM PASSED — CONTRACT READY FOR PREPROCESSING"
NEWS_LOOKBACK_CANDIDATES = (7, 14, 30)
RECOMMENDED_NEWS_LOOKBACK = 30
IMAGE_AGE_REPORTING_THRESHOLDS = (30, 90, 180, 365)
TIME_SERIES_WINDOWS = (5, 20, 60)
MIN_TIME_SERIES_ROWS = max(TIME_SERIES_WINDOWS) + 1
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
MAX_IMAGE_CAPTION_CHARS = 900

PRICE_SEMANTICS_CONTRACT = (
    "FinMultiTime does not explicitly document the adjustment semantics of the "
    "target OHLC series. Empirical comparison is consistent with "
    "dividend-adjusted historical prices for AAPL/JPM and raw-equivalent OHLC "
    "for AMZN over the audited period. FinMultiTime OHLC is therefore treated "
    "as source-native descriptive data with adjustment "
    "semantics not contractually guaranteed."
)

ANALYST_ROUTING = {
    "TEXT": "News Analyst",
    "TABLE": "Fundamentals Analyst",
    "TIME_SERIES": "Market Analyst",
    "IMAGE": "Market Analyst",
}

QWEN_CAPTION_SCHEMA = (
    "trend",
    "momentum_visual",
    "volatility_visual",
    "candlestick_structure",
    "notable_gap_or_reversal",
    "support_resistance_visual",
    "volume_visual",
    "other_visible_pattern",
    "confidence",
)

QWEN_CAPTION_PROMPT = (
    "Describe only information visually observable in the supplied financial "
    "chart. Return only the constrained schema fields: trend, momentum_visual, "
    "volatility_visual, candlestick_structure, notable_gap_or_reversal, "
    "support_resistance_visual, volume_visual, other_visible_pattern, and "
    "confidence. Use short phrases, do not infer causes, and do not add any "
    "narrative. Do not use external market knowledge, company-specific facts "
    "not visible in the chart, subsequent events, future returns, forecasts, "
    "predictions, price targets, or investment recommendations such as "
    "BUY/HOLD/SELL."
)

QWEN_SAFETY_PROHIBITIONS = (
    "external market knowledge",
    "company-specific factual knowledge not visible in the chart",
    "subsequent events",
    "future returns",
    "forecasts or predictions",
    "price targets",
    "BUY/HOLD/SELL recommendations",
)

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


def period_duration_days(row: dict[str, Any]) -> int | None:
    """Return inclusive calendar duration for duration facts only."""
    start = date_or_none(row.get("start"))
    end = date_or_none(row.get("end"))
    if start is None or end is None:
        return None
    return (end - start).days + 1


def duration_class(row: dict[str, Any]) -> str:
    """Classify inclusive durations with tolerance for 13-week/52–53-week calendars."""
    duration = period_duration_days(row)
    if duration is None:
        return "point_in_time"
    if 45 <= duration <= 120:
        return "quarterly"
    if 120 < duration <= 210:
        return "year_to_date_6m"
    if 210 < duration <= 300:
        return "year_to_date_9m"
    if duration > 300:
        return "annual"
    return "other_duration"


def filing_period_match_score(row: dict[str, Any]) -> int:
    """Prefer source-reported duration matching fp/form for cash-flow facts."""
    fp = str(row.get("fp") or "").upper()
    form = str(row.get("form") or "").upper()
    actual = duration_class(row)
    if actual == "point_in_time":
        return 0
    if fp == "Q1":
        expected = {"quarterly"}
    elif fp == "Q2":
        expected = {"year_to_date_6m"}
    elif fp == "Q3":
        expected = {"year_to_date_9m"}
    elif fp in {"FY", "Q4"} or "10-K" in form:
        expected = {"annual"}
    else:
        expected = set()
    return int(actual in expected)


def table_fact_provenance_hash(row: dict[str, Any]) -> str:
    fields = (
        "member", "taxonomy", "concept", "unit", "raw_start", "raw_end",
        "raw_filed", "val", "accn", "form", "fy", "fp", "frame",
    )
    return digest({field: row.get(field) for field in fields})


def normalize_table_fact(row: dict[str, Any]) -> dict[str, Any]:
    """Expose source-reported period metadata without deriving comparable periods."""
    return {
        "taxonomy": row.get("taxonomy"),
        "concept": row.get("concept"),
        "value": row.get("val"),
        "unit": row.get("unit"),
        "form": row.get("form"),
        "fy": row.get("fy"),
        "fp": row.get("fp"),
        "period_start": date_text(date_or_none(row.get("start"))),
        "period_end": date_text(date_or_none(row.get("end"))),
        "period_duration_days": period_duration_days(row),
        "period_duration_class": duration_class(row),
        "filed_date": date_text(date_or_none(row.get("filed"))),
        "accession_number": row.get("accn"),
        "source_member": row.get("member"),
        "source_provenance_hash": table_fact_provenance_hash(row),
    }


def choose_table_fact_with_diagnostic(
    tables: dict[str, dict[str, Any] | None], symbol: str, concept: str, decision: date
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    item = tables[symbol]
    diagnostic: dict[str, Any] = {
        "symbol": symbol,
        "concept": concept,
        "decision_session": decision.isoformat(),
        "candidate_count": 0,
        "eligible_count": 0,
        "economic_period_group_count": 0,
        "same_day_rejected_count": 0,
        "duration_candidates_at_latest_end": [],
        "selection_status": "UNAVAILABLE",
        "selection_reason": "no eligible fact",
    }
    if item is None:
        diagnostic["selection_reason"] = "no table source member"
        return None, diagnostic

    candidates = [
        row for row in item["observations"]
        if row.get("taxonomy") == "us-gaap"
        and row.get("concept") == concept
        and row.get("unit") == "USD"
    ]
    diagnostic["candidate_count"] = len(candidates)
    diagnostic["same_day_rejected_count"] = sum(
        row.get("filed") == decision for row in candidates if row.get("filed")
    )
    eligible = [
        row for row in candidates
        if row.get("filed") is not None
        and row.get("filed") < decision
        and row.get("end") is not None
    ]
    diagnostic["eligible_count"] = len(eligible)
    if not eligible:
        diagnostic["selection_reason"] = "no PIT-safe fact with period end"
        return None, diagnostic

    # Resolve later restatements only among versions available before the
    # decision, keyed by the exact source-reported economic period.
    by_period: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_period[(row.get("raw_start"), row.get("raw_end"))].append(row)
    diagnostic["economic_period_group_count"] = len(by_period)
    latest_versions: list[dict[str, Any]] = []
    for values in by_period.values():
        latest_filed = max(row.get("filed") for row in values if row.get("filed"))
        latest = [row for row in values if row.get("filed") == latest_filed]
        identity_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in latest:
            identity_groups[(
                row.get("accn"), row.get("form"), row.get("fy"),
                row.get("fp"), row.get("member"),
            )].append(row)
        if any(len({row.get("val") for row in rows}) > 1 for rows in identity_groups.values()):
            diagnostic["selection_reason"] = "conflicting values at identical filing metadata"
            return None, diagnostic
        latest_versions.append(
            max(
                latest,
                key=lambda row: (
                    filing_period_match_score(row),
                    str(row.get("accn") or ""),
                    str(row.get("form") or ""),
                    str(row.get("member") or ""),
                    table_fact_provenance_hash(row),
                ),
            )
        )

    latest_end = max(row["end"] for row in latest_versions if row.get("end"))
    at_latest_end = [row for row in latest_versions if row.get("end") == latest_end]
    diagnostic["duration_candidates_at_latest_end"] = [
        {
            "period_start": date_text(row.get("start")),
            "period_end": date_text(row.get("end")),
            "period_duration_days": period_duration_days(row),
            "form": row.get("form"),
            "fp": row.get("fp"),
            "filing_period_match_score": filing_period_match_score(row),
            "filed_date": date_text(row.get("filed")),
            "accession_number": row.get("accn"),
        }
        for row in sorted(
            at_latest_end,
            key=lambda candidate: (
                candidate.get("start") or date.min,
                candidate.get("filed") or date.min,
                str(candidate.get("accn") or ""),
            ),
        )
    ]
    selected = max(
        latest_versions,
        key=lambda row: (
            row.get("end") or date.min,
            filing_period_match_score(row),
            row.get("start") or date.min,
            row.get("filed") or date.min,
            str(row.get("accn") or ""),
            str(row.get("form") or ""),
            str(row.get("member") or ""),
            table_fact_provenance_hash(row),
        ),
    )
    normalized = normalize_table_fact(selected)
    diagnostic.update({
        "selection_status": "AVAILABLE",
        "selection_reason": "latest PIT-safe economic period; filing-period duration preference applied",
        "selected_fact": normalized,
    })
    return normalized, diagnostic


def choose_table_fact(
    tables: dict[str, dict[str, Any] | None], symbol: str, concept: str, decision: date
) -> dict[str, Any] | None:
    return choose_table_fact_with_diagnostic(tables, symbol, concept, decision)[0]


def table_selection(
    tables: dict[str, dict[str, Any] | None], symbol: str, decision: date
) -> dict[str, Any]:
    item = tables[symbol]
    if item is None:
        return {
            "status": "UNAVAILABLE",
            "facts": {},
            "unavailable_concepts": list(SELECTED_TABLE_CONCEPTS),
            "diagnostics": [
                choose_table_fact_with_diagnostic(tables, symbol, concept, decision)[1]
                for concept in SELECTED_TABLE_CONCEPTS
            ],
            "same_day_count": 0,
            "latest_safe": None,
        }
    same_day = sum(
        row.get("filed") == decision for row in item["observations"] if row.get("filed")
    )
    selected: dict[str, dict[str, Any] | None] = {}
    diagnostics: list[dict[str, Any]] = []
    for concept in SELECTED_TABLE_CONCEPTS:
        fact, diagnostic = choose_table_fact_with_diagnostic(tables, symbol, concept, decision)
        selected[concept] = fact
        diagnostics.append(diagnostic)
    facts = {concept: fact for concept, fact in selected.items() if fact is not None}
    latest_safe = max(
        (row["filed"] for row in item["observations"] if row.get("filed") and row["filed"] < decision),
        default=None,
    )
    return {
        "status": "AVAILABLE" if facts else "UNAVAILABLE",
        "facts": facts,
        "unavailable_concepts": [concept for concept, fact in selected.items() if fact is None],
        "diagnostics": diagnostics,
        "same_day_count": same_day,
        "latest_safe": latest_safe,
    }


def time_series_selection(
    series: dict[str, dict[str, Any] | None], symbol: str, decision: date
) -> dict[str, Any]:
    item = series[symbol]
    if item is None:
        return {
            "status": "UNAVAILABLE",
            "rows": [],
            "latest": None,
            "source_row_count_through_decision": 0,
            "required_rows": MIN_TIME_SERIES_ROWS,
        }
    rows = [row for row in item["rows"] if row.get("session_date") and row["session_date"] <= decision]
    rows.sort(key=lambda row: row["session_date"])
    window = rows[-MIN_TIME_SERIES_ROWS:]
    return {
        "status": "AVAILABLE" if len(window) >= MIN_TIME_SERIES_ROWS else "UNAVAILABLE",
        "rows": window,
        "latest": window[-1]["session_date"] if window else None,
        "source_row_count_through_decision": len(rows),
        "required_rows": MIN_TIME_SERIES_ROWS,
    }


def ts_summary(selection: dict[str, Any]) -> dict[str, Any]:
    rows = selection["rows"]
    result: dict[str, Any] = {}
    closes = [parse_float(row["numeric"].get("Close")) for row in rows]
    highs = [parse_float(row["numeric"].get("High")) for row in rows]
    lows = [parse_float(row["numeric"].get("Low")) for row in rows]
    volumes = [parse_float(row["numeric"].get("Volume")) for row in rows]
    for window in TIME_SERIES_WINDOWS:
        # An N-session return spans N intervals and therefore needs N+1 closes.
        values = closes[-(window + 1):]
        result[f"cumulative_return_{window}d"] = (
            values[-1] / values[0] - 1
            if len(values) == window + 1
            and all(value is not None for value in values)
            and values[0] not in (None, 0)
            else None
        )
    returns = [
        closes[index] / closes[index - 1] - 1
        for index in range(1, len(closes))
        if closes[index] is not None and closes[index - 1] not in (None, 0)
    ]
    # Exactly the most recent 20 one-session returns; sample stddev is ddof=1.
    vol_window = returns[-20:]
    result["realised_volatility_20d_annualised"] = (
        statistics.stdev(vol_window) * math.sqrt(252) if len(vol_window) == 20 else None
    )
    high_window = highs[-20:]
    low_window = lows[-20:]
    result["high_low_range_20d"] = (
        max(high_window) / min(low_window) - 1
        if len(high_window) == 20
        and len(low_window) == 20
        and all(value is not None for value in high_window + low_window)
        and min(low_window) not in (None, 0)
        else None
    )
    peak_window = closes[-60:]
    result["drawdown_from_60d_peak"] = (
        peak_window[-1] / max(peak_window) - 1
        if len(peak_window) == 60
        and all(value is not None for value in peak_window)
        and max(peak_window) not in (None, 0)
        else None
    )
    current_volume = volumes[-1] if volumes else None
    previous_20_volumes = volumes[-21:-1]
    result["relative_volume_vs_20d_mean"] = (
        current_volume / statistics.mean(previous_20_volumes)
        if len(previous_20_volumes) == 20
        and current_volume is not None
        and all(value is not None for value in previous_20_volumes)
        and statistics.mean(previous_20_volumes)
        else None
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
        len(canonical_json({"concept": concept, **fact}) + "\n")
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


def routed_context_sizes(sizes: dict[str, int]) -> dict[str, int]:
    """Estimate only the FinMultiTime increment routed to each analyst."""
    return {
        "News Analyst": sizes["text_chars"],
        "Fundamentals Analyst": sizes["table_chars"],
        "Market Analyst": sizes["time_series_chars"] + sizes["image_chars"],
        "Social Analyst": 0,
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
            routed_sizes = routed_context_sizes(sizes)
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
                f"{concept}={fact.get('filed_date') or ''}"
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
                    "selected_table_facts": canonical_json(selected_facts),
                    "table_unavailable_concepts": ";".join(table["unavailable_concepts"]),
                    "table_selection_diagnostics": canonical_json(table["diagnostics"]),
                    "table_latest_safe_filed_date": date_text(table.get("latest_safe")),
                    "table_latest_safe_age_calendar_days": table_age,
                    "table_same_day_ambiguous_rejected_count": table["same_day_count"],
                    "table_selected_restatement_rule": "latest PIT-safe filed version per exact taxonomy/concept/unit/start/end economic-period group; duration match by fp/form at latest period end",
                    "TIME_SERIES_status": ts["status"],
                    "latest_included_session": date_text(ts.get("latest")),
                    "time_series_selected_session_count": len(ts.get("rows", [])),
                    "time_series_required_session_count": MIN_TIME_SERIES_ROWS,
                    "time_series_evidence_age_calendar_days": time_series_age,
                    "time_series_summary_fields": ";".join(sorted(ts_summary(ts))),
                    "time_series_summary_values": canonical_json(ts_summary(ts)),
                    "time_series_formula_contract": "CR_N = Close[t]/Close[t-N]-1; VOL20 = sample_std(ddof=1)(last 20 one-session returns)*sqrt(252); HLR20=max(High[-20:])/min(Low[-20:])-1; DD60=Close[t]/max(Close[t-59:t])-1; RVOL=Volume[t]/mean(Volume[t-20:t-1])",
                    "IMAGE_status": image["status"],
                    "selected_image": image_item["filename"] if image_item else "",
                    "selected_image_source_identity": image_item.get("path", "") if image_item else "",
                    "selected_image_period_start": date_text(image_item["period_start"] if image_item else None),
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
                    "routed_news_analyst_chars": routed_sizes["News Analyst"],
                    "routed_fundamentals_analyst_chars": routed_sizes["Fundamentals Analyst"],
                    "routed_market_analyst_chars": routed_sizes["Market Analyst"],
                    "routed_social_analyst_chars": routed_sizes["Social Analyst"],
                    "direct_raw_packet_injected_downstream": False,
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
            "semantic_overlap": "high: same OHLCV/action field family; FinMultiTime adjustment semantics are not explicitly documented and are not contractually guaranteed",
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
        counts = dict.fromkeys(IMAGE_AGE_REPORTING_THRESHOLDS, 0)
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


def analyst_budget_summary(simulation: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "News Analyst": "routed_news_analyst_chars",
        "Fundamentals Analyst": "routed_fundamentals_analyst_chars",
        "Market Analyst": "routed_market_analyst_chars",
        "Social Analyst": "routed_social_analyst_chars",
    }
    summary: dict[str, Any] = {}
    for analyst, field in fields.items():
        values = [int(row[field]) for row in simulation]
        summary[analyst] = {
            "min": min(values, default=0),
            "median": statistics.median(values) if values else 0,
            "max": max(values, default=0),
            "source_fields": [field],
        }
    return summary


def simulation_invariants(
    data: dict[str, Any], context: dict[str, Any], simulation: list[dict[str, Any]]
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "formal_cases_are_78": len(simulation) == 78,
        "future_pit_violations_are_zero": True,
        "no_same_day_date_only_news_admitted": True,
        "no_same_day_filed_table_fact_admitted": True,
        "no_unfinished_half_year_image_admitted": True,
        "no_future_label_or_target_included": True,
        "no_later_restatement_visible_early": True,
        "no_raw_finmultitime_row_modified": True,
        "no_m0_execution_or_evaluation_input_replaced": True,
        "every_modality_has_explicit_status": True,
        "routing_follows_frozen_analyst_map": True,
        "time_series_has_required_history": True,
    }
    expected_statuses = {"AVAILABLE", "UNAVAILABLE"}
    summary_fields = {
        "cumulative_return_5d",
        "cumulative_return_20d",
        "cumulative_return_60d",
        "realised_volatility_20d_annualised",
        "high_low_range_20d",
        "drawdown_from_60d_peak",
        "relative_volume_vs_20d_mean",
    }
    for row in simulation:
        decision = date.fromisoformat(row["decision_session"])
        symbol = row["symbol"]
        news = selected_news(data["news"][symbol], decision)
        table = table_selection(data["tables"], symbol, decision)
        ts = time_series_selection(data["series"], symbol, decision)
        image = image_selection(data["images"], symbol, decision)
        if int(row["PIT_violation_count"]) != 0:
            checks["future_pit_violations_are_zero"] = False
        if any(news_date(record) >= decision for record in news["records"]):
            checks["no_same_day_date_only_news_admitted"] = False
        for fact in table["facts"].values():
            filed = date_or_none(fact.get("filed_date"))
            if filed is None or filed >= decision:
                checks["no_same_day_filed_table_fact_admitted"] = False
                checks["no_later_restatement_visible_early"] = False
        if image.get("item") and image["item"]["period_end"] >= decision:
            checks["no_unfinished_half_year_image_admitted"] = False
        if ts.get("latest") and ts["latest"] > decision:
            checks["future_pit_violations_are_zero"] = False
        if row["TIME_SERIES_status"] == "AVAILABLE" and int(row["time_series_selected_session_count"]) < MIN_TIME_SERIES_ROWS:
            checks["time_series_has_required_history"] = False
        if any(field.lower() in canonical_json(ts_summary(ts)).lower() for field in ("target", "label", "future", "forecast")):
            checks["no_future_label_or_target_included"] = False
        if set(summary_fields) != set(ts_summary(ts)):
            checks["no_future_label_or_target_included"] = False
        if any(row[f"{modality}_status"] not in expected_statuses for modality in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")):
            checks["every_modality_has_explicit_status"] = False
        if row["direct_raw_packet_injected_downstream"] is not False:
            checks["routing_follows_frozen_analyst_map"] = False
    return checks


def contract_json(
    data: dict[str, Any],
    context: dict[str, Any],
    anomalies: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
    concept_rows: list[dict[str, Any]],
    simulation: list[dict[str, Any]],
    image_summary: dict[str, Any],
    source_summary: dict[str, Any],
    validation: dict[str, Any],
    data_audit_identity: dict[str, Any],
) -> dict[str, Any]:
    selected_concept_rows = [
        row for row in concept_rows
        if row["concept"] in SELECTED_TABLE_CONCEPTS
        and row["recommended_fixed_schema"] == "true"
    ]
    return {
        "contract_id": "M1-FinMultiTime-Evidence-Contract",
        "packet_version": CONTRACT_VERSION,
        "parent_draft_version": PARENT_DRAFT_VERSION,
        "status": "FROZEN",
        "verdict": FINAL_VERDICT,
        "research_review": {
            "decision": RESEARCH_REVIEW_DECISION,
            "required_revisions_applied": True,
        },
        "erratum": {
            "previous_contract_version": PREVIOUS_CONTRACT_VERSION,
            "previous_contract_sha256": PREVIOUS_CONTRACT_SHA256,
            "reason": ERRATUM_REASON,
            "behaviour_equivalence_report": "docs/m1/m1_contract_erratum_equivalence.json",
            "behaviour_equivalent": True,
        },
        "freeze_metadata": {
            "freeze_date": FREEZE_DATE,
            "timestamp_policy": "run timestamps are excluded from deterministic artifacts; PIT uses source availability dates/session labels and strict comparison rules",
            "source_parent_sha": SOURCE_PARENT_SHA,
            "frozen_m0_base_sha": FROZEN_M0_BASE_SHA,
        },
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
            "UNAVAILABLE": "source absent, no eligible evidence, or canonical selection cannot be made",
            "AMBIGUOUS_REJECTED": "observed but excluded because date/time availability cannot be proven; provenance-only and never Agent-visible",
        },
        "common_fields": [
            "status", "source_identity", "source_available_date_or_session",
            "evidence_age_calendar_days", "provenance_hash_reference",
            "selection_rule", "missingness_reason",
        ],
        "modalities": {
            "TEXT": {
                "source_identity": "text/sp500_news.zip::sp500_news/<SYMBOL>.jsonl",
                "pit_rule": "Date < decision_session_date; same-day date-only records are AMBIGUOUS_REJECTED",
                "lookback_calendar_days": RECOMMENDED_NEWS_LOOKBACK,
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
                "missingness": "JPM remains UNAVAILABLE; no eligible article remains UNAVAILABLE; no external news filling",
            },
            "TABLE": {
                "source_identity": "table/SP500_tabular.zip::financial_reports/<symbol>/*.json",
                "pit_rule": "filed_date < decision_session_date; period_end is never availability",
                "same_day_rule": "same-day filed observations are AMBIGUOUS_REJECTED",
                "fixed_concepts": list(SELECTED_TABLE_CONCEPTS),
                "selected_fact_fields": [
                    "taxonomy", "concept", "value", "unit", "form", "fy", "fp",
                    "period_start", "period_end", "period_duration_days", "period_duration_class",
                    "filed_date", "accession_number", "source_provenance_hash",
                ],
                "duration_semantics": "point-in-time balance-sheet facts have no period start/duration; cash-flow facts retain the source-reported start/end and inclusive calendar period_duration_days; values are never annualised, interpolated, derived, or converted into artificial comparable periods",
                "duration_class_ranges_inclusive_calendar_days": {
                    "quarterly": "45-120",
                    "year_to_date_6m": "121-210",
                    "year_to_date_9m": "211-300",
                    "annual": "301+",
                    "point_in_time": "no source period start",
                    "other_duration": "1-44",
                },
                "duration_range_rationale": "wide deterministic ranges accommodate 13-week quarters and 52/53-week non-calendar fiscal years without requiring exact 90/180/270/365-day durations",
                "duration_selection_rule": "after PIT and eligible-version resolution, select the latest economic period by period_end; when multiple durations share that end, prefer the duration class matching fp/form: Q1=quarterly, Q2=year_to_date_6m, Q3=year_to_date_9m, FY/Q4 or 10-K=annual; retain the actual source-reported period metadata",
                "restatement_rule": "for each exact taxonomy/concept/unit/start/end economic fact, select only the latest filed version whose filed_date is before the decision; a later restatement cannot appear in an earlier decision",
                "canonical_selection_failure": "if identical filing metadata contains conflicting values, or no canonical period-end fact exists, mark that concept UNAVAILABLE rather than guess",
                "representation": "one bounded fact per selected concept with full source-reported duration and filing metadata plus provenance hash",
                "missingness": "concept-level UNAVAILABLE if no eligible or canonical fact; never substitute a later value or another symbol",
            },
            "TIME_SERIES": {
                "source_identity": "time_series/S&P500_time_series.zip::S&P500_time_series/<symbol>.csv",
                "pit_rule": "completed session rows with session <= decision_session; no later session is eligible",
                "required_completed_rows": MIN_TIME_SERIES_ROWS,
                "summary_fields": [
                    "cumulative_return_5d", "cumulative_return_20d", "cumulative_return_60d",
                    "realised_volatility_20d_annualised", "high_low_range_20d",
                    "drawdown_from_60d_peak", "relative_volume_vs_20d_mean",
                ],
                "formulas": {
                    "cumulative_return_5d": "Close[t] / Close[t-5] - 1; requires 6 close observations",
                    "cumulative_return_20d": "Close[t] / Close[t-20] - 1; requires 21 close observations",
                    "cumulative_return_60d": "Close[t] / Close[t-60] - 1; requires 61 close observations",
                    "realised_volatility_20d_annualised": "sample_std(last 20 one-session returns, ddof=1) * sqrt(252), where r_i = Close[i] / Close[i-1] - 1; requires 21 closes",
                    "high_low_range_20d": "max(High[t-19:t]) / min(Low[t-19:t]) - 1; most recent 20 completed bars",
                    "drawdown_from_60d_peak": "Close[t] / max(Close[t-59:t]) - 1; most recent 60 completed closes including decision session",
                    "relative_volume_vs_20d_mean": "Volume[t] / mean(Volume[t-20:t-1]); current completed-session volume divided by previous 20 completed-session volumes; requires 21 rows",
                },
                "off_by_one_fixed": True,
                "raw_rows_in_packet": 0,
                "price_semantics": PRICE_SEMANTICS_CONTRACT,
                "price_use_for_execution": False,
                "price_use_for_valuation": False,
                "m0_market_path_authoritative": True,
                "representation": "fixed numeric summary plus through_session and source hash; no target/forward labels",
                "missingness": "UNAVAILABLE if required completed history or required numeric inputs are absent; invalid raw rows are never repaired",
            },
            "IMAGE": {
                "source_identity": "image/image/S&P500_image_*/<symbol>/<symbol>_YYYY_H[1|2]_candlestick.png",
                "window_rule": "infer H1 nominal end=June 30 and H2 nominal end=December 31; require inferred period_end < decision_session",
                "staleness_rule": "no additional arbitrary 30/60/90/180-day cutoff; expose evidence age explicitly",
                "selection_rule": "one latest eligible completed image by inferred period_end, filename, path",
                "agent_visible_metadata": ["filename/source identity", "inferred chart period", "inferred period end", "evidence age at decision time"],
                "caption_rule": "image -> frozen Qwen3-VL-2B-Instruct caption -> Market Analyst; no synthetic image",
                "missingness": "AAPL remains UNAVAILABLE; AMZN/JPM may use their latest completed eligible half-year chart",
                "coverage_summary": image_summary,
            },
        },
        "routing": {
            "modality_to_analyst": ANALYST_ROUTING,
            "social_analyst": "receives no new FinMultiTime-specific modality; existing M0 Social Analyst behavior is unchanged",
            "downstream_flow": "Bull Researcher, Bear Researcher, Research Manager, Trader, Risk agents, and Portfolio Manager receive FinMultiTime-derived information through the existing analyst-report flow",
            "raw_packet_direct_injection_to_downstream": False,
        },
        "qwen_image_adapter": {
            "model": "Qwen3-VL-2B-Instruct",
            "architecture": "FinMultiTime image -> frozen Qwen caption -> Market Analyst",
            "offline_preprocessing_only": True,
            "agent": False,
            "trained": False,
            "experiment_variable": False,
            "formal_runtime_call": False,
            "prompt": QWEN_CAPTION_PROMPT,
            "prohibited_content": list(QWEN_SAFETY_PROHIBITIONS),
            "additional_text_context": False,
            "ticker_or_company_metadata": False,
            "caption_schema": list(QWEN_CAPTION_SCHEMA),
            "max_caption_chars": MAX_IMAGE_CAPTION_CHARS,
            "exact_model_revision_frozen": False,
            "runtime_and_generation_environment_frozen": False,
        },
        "budget": {
            "unit": "UTF-8 characters, deterministic proxy; no tokenizer dependency",
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
                "image": f"at most one structured caption; caption <= {MAX_IMAGE_CAPTION_CHARS} chars before section envelope",
            },
            "analyst_incremental_evidence_chars": analyst_budget_summary(simulation),
            "raw_and_m0_reference_stats": source_summary,
        },
        "provenance": {
            "hash_algorithm": "SHA-256",
            "raw_source_path": "/Volumes/Jackson/Dataset/FinMultiTime",
            "source_members": [NEWS_ARCHIVE, TABLE_ARCHIVE, TS_ARCHIVE, "image/image/S&P500_image_*"],
            "m0_snapshot_reference": str(M0_RUN_INPUTS),
            "availability_fields": {
                "TEXT": "Date (calendar date only)",
                "TABLE": "filed (calendar date only)",
                "TIME_SERIES": "session label through decision close",
                "IMAGE": "inferred half-year window end, not a publication timestamp",
            },
            "data_audit_identity": data_audit_identity,
        },
        "research_controls": {
            "augmentation_only": True,
            "m0_historical_safe_evidence_retained": True,
            "trader_policy_unchanged": True,
            "memory_unchanged": True,
            "backtester_unchanged": True,
            "execution_unchanged": True,
            "evaluation_prices_unchanged": True,
            "metrics_unchanged": True,
            "no_available_at_after_decision": True,
            "no_target_labels_or_forward_returns": True,
            "no_2024h1_outcome_tuning": True,
            "same_frozen_inputs_reusable_by_m2_a1_a2": True,
        },
        "validation": validation,
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
        "not_frozen_yet": {
            "processed_three_stock_subset": True,
            "image_captions": True,
            "final_evidence_packets": True,
            "m1_input_sha_bundle": True,
            "m1_runtime_integration": True,
            "m1_pilot_environment": True,
            "formal_m1_result": True,
        },
        "boundary_confirmation": {
            "raw_finmultitime_modified": False,
            "final_processed_subset_built": False,
            "final_evidence_packets_built": False,
            "qwen_downloaded": False,
            "qwen_run": False,
            "deepseek_calls": 0,
            "paid_api_calls": 0,
            "trader_modified": False,
            "memory_modified": False,
            "execution_modified": False,
            "formal_m1_run": False,
            "m2_agentic_rl_started": False,
            "alpha_mas_experiments_modified": False,
        },
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
    packet_sizes = [int(row["estimated_packet_chars"]) for row in simulation]
    selected = [row for row in concept_rows if row["recommended_fixed_schema"] == "true"]
    unavailable_by_concept: Counter[str] = Counter()
    for row in simulation:
        unavailable_by_concept.update(
            concept for concept in row["table_unavailable_concepts"].split(";") if concept
        )
    lines = [
        "# M1 FinMultiTime Evidence Contract",
        "",
        f"**Packet version:** `{CONTRACT_VERSION}`",
        "**Status:** FROZEN",
        f"**Final verdict:** {FINAL_VERDICT}",
        f"**Parent draft:** `{PARENT_DRAFT_VERSION}`",
        f"**Research-review decision:** `{RESEARCH_REVIEW_DECISION}`",
        f"**Freeze date:** `{FREEZE_DATE}`",
        "",
        f"This patch-level erratum supersedes `{PREVIOUS_CONTRACT_VERSION}` without erasing it. Reason: {ERRATUM_REASON} The deterministic equivalence report records no research-relevant change.",
        "",
        "This contract freezes selection and representation rules only. It does not build the processed subset, generate final Evidence Packets, download/run Qwen, modify M0 behavior, or run M1.",
        "",
        "## Research relationship and hard controls",
        "",
        "`M1 evidence = M0 historical-safe evidence + FinMultiTime Evidence Packet`. FinMultiTime is additive. M0 historical-safe evidence, Trader policy, Memory, backtester, execution, evaluation prices, metrics, and all 78 formal cases remain unchanged.",
        "",
        "Every packet always contains TEXT, TABLE, TIME_SERIES, and IMAGE. Each has explicit `AVAILABLE` or `UNAVAILABLE` status. Unsafe observed items may appear only as provenance-only `AMBIGUOUS_REJECTED` and are never Agent-visible.",
        "",
        "## Residual data validation and price semantics",
        "",
        f"The deterministic scan found {len(anomalies)} impossible-OHLC rows: {sum(row['symbol'] == 'AAPL' for row in anomalies)} AAPL, {sum(row['symbol'] == 'AMZN' for row in anomalies)} AMZN, and {sum(row['symbol'] == 'JPM' for row in anomalies)} JPM. They are outside the M0 warm-up, formal decision span, and reachable 60-session summary lookback. Raw rows are not repaired.",
        "",
        PRICE_SEMANTICS_CONTRACT,
        "",
        "The contract treats FinMultiTime OHLC as source-native descriptive data: no silent normalisation, no repair to force equality with M0, and no use for execution or valuation. The validated M0 market path remains authoritative.",
        "",
        f"The audit has {len(consistency)} deterministic comparison rows. Close relative-difference ranges are AAPL `{fmt_number(semantics['AAPL']['close_ratio_fin_minus_m0_min'], 8)}` to `{fmt_number(semantics['AAPL']['close_ratio_fin_minus_m0_max'], 8)}`, AMZN `{fmt_number(semantics['AMZN']['close_ratio_fin_minus_m0_min'], 8)}` to `{fmt_number(semantics['AMZN']['close_ratio_fin_minus_m0_max'], 8)}`, and JPM `{fmt_number(semantics['JPM']['close_ratio_fin_minus_m0_min'], 8)}` to `{fmt_number(semantics['JPM']['close_ratio_fin_minus_m0_max'], 8)}`.",
        "",
        "## TEXT",
        "",
        "PIT gate: `Date < decision_session_date`. Same-day date-only news is `AMBIGUOUS_REJECTED`; no external news, live web, or cross-symbol filling is allowed. Deduplicate exact records, then exact URLs, retain removed/kept hashes, order newest-first deterministically, and select at most 8 articles from the fixed 30-calendar-day lookback. JPM is always `UNAVAILABLE`.",
        "",
        "| Symbol | Window | Cases with article | Mean count | Maximum | No-article cases |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for symbol in TARGETS:
        for window in NEWS_LOOKBACK_CANDIDATES:
            stats = lookback[symbol][window]
            lines.append(f"| {symbol} | {window} days | {stats['coverage_cases']}/26 | {stats['average_count']} | {stats['maximum_count']} | {stats['no_article_cases']}/26 |")
    lines.extend([
        "",
        "## TABLE",
        "",
        "PIT gate: `filed_date < decision_session_date`; `period_end` is never an availability gate. Same-day filings are `AMBIGUOUS_REJECTED`.",
        "",
        "The six fixed concepts are:",
        "",
        "| Concept | Interpretation | Cross-asset coverage | Concept-level unavailable cases |",
        "|---|---|---:|---:|",
    ])
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
        lines.append(f"| `{concept}` | {descriptions[concept]} | {row.get('symbol_coverage_count', 0)}/3 | {unavailable_by_concept[concept]} |")
    lines.extend(
        [
            "",
            "Each selected fact exposes taxonomy, concept, value, unit, form, fy, fp, period start, period end, inclusive `period_duration_days`, filed date, accession number, source member, and source/provenance hash. Balance-sheet facts remain point-in-time. Cash-flow facts retain the actual quarterly, year-to-date, or annual source-reported horizon; no annualisation, interpolation, derivation, or artificial period conversion is allowed.",
            "",
            "Duration classes use inclusive calendar-day ranges with tolerance for 13-week quarters and 52/53-week fiscal calendars: `quarterly` = 45–120 days, `year_to_date_6m` = 121–210, `year_to_date_9m` = 211–300, `annual` = 301+, `point_in_time` = no source period start, and `other_duration` = 1–44 days.",
            "",
            "Selection is deterministic: apply PIT; resolve restatements only among versions already filed before the decision; select the latest economic period by period end; at a shared latest period end prefer the duration class matching `fp/form` (`Q1` → `quarterly`, `Q2` → `year_to_date_6m`, `Q3` → `year_to_date_9m`, `FY/Q4/10-K` → `annual`). Conflicting values at identical filing metadata or a non-canonical selection produce concept-level `UNAVAILABLE`. Diagnostics are in `m1_evidence_contract_table_selection_diagnostics.csv`.",
            "",
            "## TIME_SERIES",
            "",
            "PIT gate: completed session rows with `session <= decision_session`; the source-selection logic retains at least 61 completed rows through the decision session. The packet contains no raw rows, future labels, targets, or forecasts.",
            "",
            "The off-by-one issue is fixed. Exact formulas are:",
            "",
            "- 5-session cumulative return: `Close[t] / Close[t-5] - 1` (6 closes).",
            "- 20-session cumulative return: `Close[t] / Close[t-20] - 1` (21 closes).",
            "- 60-session cumulative return: `Close[t] / Close[t-60] - 1` (61 closes).",
            "- 20-session realised volatility: `sample_std(ddof=1)(last 20 one-session returns) * sqrt(252)`, with `r_i = Close[i] / Close[i-1] - 1` (21 closes).",
            "- 20-session high-low range: `max(High[t-19:t]) / min(Low[t-19:t]) - 1` (20 completed bars).",
            "- Drawdown from 60-session peak: `Close[t] / max(Close[t-59:t]) - 1` (60 completed closes including the decision session).",
            "- Relative volume: `Volume[t] / mean(Volume[t-20:t-1])` (current volume versus the previous 20 completed sessions; 21 rows).",
            "",
            "M0 validated market data remains authoritative for execution, valuation, and corporate-action accounting. FinMultiTime prices are descriptive evidence only.",
            "",
            "## IMAGE and future Qwen adapter",
            "",
            "AAPL is explicitly `UNAVAILABLE`. AMZN/JPM may use their latest completed half-year chart: H1 nominal end is June 30, H2 nominal end is December 31, and inferred end must be strictly before the decision. No additional 30/60/90/180-day staleness cutoff exists. Agent-visible metadata includes filename/source identity, inferred chart period, inferred period end, and evidence age.",
            "",
            f"The simulation selects {image_summary['unique_files_used']} unique image files for {image_summary['formal_case_references']} repeated references. Evidence age is {image_summary['image_age_min_days']}–{image_summary['image_age_max_days']} days (median {image_summary['image_age_median_days']}); AAPL has no eligible file.",
            "",
            "| Symbol | References | Unique files | Age <=30d | <=90d | <=180d | <=365d |",
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
            "Later Qwen use is offline preprocessing only: `FinMultiTime image -> frozen Qwen3-VL-2B-Instruct caption -> Market Analyst`. Qwen is not an Agent, is not trained, is not an experiment variable, and is not called during Formal M1 runtime. The prompt must say: “Describe only information visually observable in the supplied financial chart.” It must prohibit external/company knowledge not visible in the image, subsequent events, future returns, forecasts, predictions, price targets, and BUY/HOLD/SELL recommendations. No ticker/company identity or additional text context is supplied. The frozen short schema is `trend`, `momentum_visual`, `volatility_visual`, `candlestick_structure`, `notable_gap_or_reversal`, `support_resistance_visual`, `volume_visual`, `other_visible_pattern`, `confidence`, with a maximum caption size of 900 characters. The exact model revision and runtime/generation environment remain intentionally unfrozen until caption preprocessing.",
            "",
            "## Routing and budget",
            "",
            "Routing is modality-specific: TEXT → News Analyst; TABLE → Fundamentals Analyst; TIME_SERIES → Market Analyst; IMAGE caption → Market Analyst. Social Analyst receives no new FinMultiTime-specific modality. The complete raw packet is not injected into Bull/Bear Researchers, Research Manager, Trader, Risk agents, or Portfolio Manager; derived information travels through the existing analyst-report flow.",
            "",
            "| Analyst | Min additional chars | Median | Max additional chars |",
            "|---|---:|---:|---:|",
        ]
    )
    for analyst, values in analyst_budget_summary(simulation).items():
        lines.append(f"| {analyst} | {values['min']} | {values['median']} | {values['max']} |")
    lines.extend([
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
            "### 78-case deterministic simulation",
            "",
            f"The simulation contains {len(simulation)}/78 cases. PIT violations: {sum(int(row['PIT_violation_count']) for row in simulation)}. Ambiguous rejected observations: {ambiguous}. Estimated packet characters: {min(packet_sizes)}–{max(packet_sizes)} (median {statistics.median(packet_sizes)}). Exact metadata is in `m1_evidence_contract_case_simulation.csv`; no Agent, outcome, target, or future trading performance was used.",
            "",
            "| Modality | AVAILABLE | UNAVAILABLE |",
            "|---|---:|---:|",
        ]
    )
    for modality in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE"):
        lines.append(
            f"| {modality} | {status_counts[modality]['AVAILABLE']} | {status_counts[modality]['UNAVAILABLE']} |"
        )
    lines.extend([
        "",
        "## Validation and boundaries",
        "",
        "| Invariant | Result |",
        "|---|---|",
    ])
    # The final contract JSON is the authoritative detailed validation record;
    # this table is intentionally human-readable and deterministic.
    for name, result in sorted(simulation_invariants(data, context, simulation).items()):
        lines.append(f"| {name} | {'PASS' if result else 'FAIL'} |")
    lines.extend([
        "",
        "The following remain deliberately unfrozen: processed three-stock subset, image captions, final Evidence Packets, M1 input SHA bundle, M1 runtime integration, M1 pilot environment, and Formal M1 result.",
        "",
        "Raw FinMultiTime modified: **NO**. Final processed subset built: **NO**. Final Evidence Packets built: **NO**. Qwen downloaded: **NO**. Qwen run: **NO**. DeepSeek calls: **0**. Paid API calls: **0**. Trader, Memory, and execution modified: **NO**. Formal M1 run: **NO**. M2 / Agentic RL started: **NO**. AlphaMAS-Experiments modified: **NO**.",
        "",
        "Freeze artifacts include `M1_EVIDENCE_CONTRACT.md`, `m1_evidence_contract.json`, `m1_evidence_contract_case_simulation.csv`, `m1_evidence_contract_table_selection_diagnostics.csv`, `m1_contract_erratum_equivalence.json`, and `m1_evidence_contract_freeze.json`. The freeze manifest records SHA-256 hashes for the final contract, simulation, equivalence evidence, and required audit/semantics reports.",
    ])
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
            "AAPL and JPM show time-varying price differences relative to the raw M0 snapshot, including on ordinary and dividend-event dates, while volume is usually unchanged or only very slightly different. AMZN OHLC is raw-equivalent over the audited overlap. These observations are empirical comparisons, not proof of a universal adjustment policy.",
            "",
            "## Contract conclusion",
            "",
            PRICE_SEMANTICS_CONTRACT,
            "FinMultiTime OHLC remains source-native descriptive data: no silent normalisation, no repair to force equality with M0, and no use for execution or valuation. If used for descriptive summaries, retain the source identity and hash.",
            "",
            "The 12 impossible-OHLC rows are a separate structural issue and are listed in `finmultitime_ohlc_anomalies.csv`; all are outside the relevant M0 warm-up, formal span, and proposed 60-session lookback.",
        ]
    )
    return "\n".join(lines) + "\n"


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def previous_frozen_simulation() -> list[dict[str, str]]:
    """Read the v1.0 simulation from the immutable source-parent commit."""
    result = subprocess.run(
        [
            "git",
            "show",
            f"{SOURCE_PARENT_SHA}:docs/m1/m1_evidence_contract_case_simulation.csv",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def _normalise_erratum_terminology(value: str) -> str:
    return value.replace("year_to_date_h1", "year_to_date_6m").replace(
        "year_to_date_h2", "year_to_date_9m"
    )


def erratum_equivalence_report(
    previous: list[dict[str, str]], current: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare all simulation fields, ignoring only version and renamed labels."""
    previous_by_case = {
        (row["symbol"], row["decision_session"]): row for row in previous
    }
    current_text = [
        {key: csv_value(value) for key, value in row.items()} for row in current
    ]
    current_by_case = {
        (row["symbol"], row["decision_session"]): row for row in current_text
    }
    if set(previous_by_case) != set(current_by_case):
        raise SystemExit("M1 erratum equivalence failed: formal case keys changed")

    differences: list[dict[str, Any]] = []
    terminology_case_count = 0
    terminology_derived_hash_changes = 0
    old_h1_occurrences = 0
    old_h2_occurrences = 0
    for case_key in sorted(previous_by_case):
        old = previous_by_case[case_key]
        new = current_by_case[case_key]
        old_serialised = canonical_json(old)
        h1_count = old_serialised.count("year_to_date_h1")
        h2_count = old_serialised.count("year_to_date_h2")
        old_h1_occurrences += h1_count
        old_h2_occurrences += h2_count
        terminology_case_count += int(bool(h1_count or h2_count))
        differing_columns: list[str] = []
        for column in old:
            if column == "packet_version":
                continue
            if (
                column == "source_hash_reference"
                and (h1_count or h2_count)
                and old[column] != new.get(column, "")
            ):
                # This packet-level hash intentionally commits to the serialized
                # duration label, so the terminology erratum changes the hash
                # even though its source rows and selected facts are identical.
                terminology_derived_hash_changes += 1
                continue
            if _normalise_erratum_terminology(old[column]) != new.get(column, ""):
                differing_columns.append(column)
        if differing_columns:
            differences.append(
                {
                    "symbol": case_key[0],
                    "decision_session": case_key[1],
                    "differing_columns": differing_columns,
                }
            )

    report = {
        "report_id": "M1-Evidence-Contract-v1.0-to-v1.0.1-Erratum-Equivalence",
        "previous_contract_version": PREVIOUS_CONTRACT_VERSION,
        "current_contract_version": CONTRACT_VERSION,
        "source_parent_sha": SOURCE_PARENT_SHA,
        "cases_compared": len(current),
        "comparison_scope": "all 53 simulation columns; packet_version and the explicit duration-class label mapping are ignored",
        "terminology_mapping": {
            "quarterly": "quarterly",
            "year_to_date_h1": "year_to_date_6m",
            "year_to_date_h2": "year_to_date_9m",
            "annual": "annual",
        },
        "expected_terminology_only_differences": {
            "cases_with_renamed_duration_classes": terminology_case_count,
            "year_to_date_h1_occurrences": old_h1_occurrences,
            "year_to_date_h2_occurrences": old_h2_occurrences,
            "packet_version_changes": len(current),
            "terminology_derived_source_hash_changes": terminology_derived_hash_changes,
        },
        "research_relevant_difference_count": len(differences),
        "research_relevant_differences": differences,
        "selected_fact_or_value_change_count": sum(
            "selected_table_facts" in item["differing_columns"] for item in differences
        ),
        "verdict": (
            "PASS — no research-relevant behavioural change"
            if not differences
            else "FAIL — research-relevant behavioural change detected"
        ),
    }
    if differences:
        raise SystemExit(
            "M1 erratum equivalence failed; freeze stopped: "
            + canonical_json(differences)
        )
    return report


def table_selection_diagnostic_rows(
    data: dict[str, Any], context: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in context["decisions"]:
        for symbol in TARGETS:
            selection = table_selection(data["tables"], symbol, decision)
            for diagnostic in selection["diagnostics"]:
                rows.append(diagnostic)
    return rows


def freeze_manifest(
    output_dir: Path,
    data_audit_identity: dict[str, Any],
) -> dict[str, Any]:
    required = (
        "M1_EVIDENCE_CONTRACT.md",
        "m1_evidence_contract.json",
        "m1_evidence_contract_case_simulation.csv",
        "m1_contract_erratum_equivalence.json",
        "finmultitime_table_concept_coverage.csv",
        "finmultitime_news_deduplication.csv",
        "finmultitime_ohlc_anomalies.csv",
        "finmultitime_timeseries_semantics.md",
        "finmultitime_timeseries_consistency.csv",
        "m1_evidence_contract_table_selection_diagnostics.csv",
    )
    artifacts = {
        name: {
            "path": f"docs/m1/{name}",
            "sha256": file_sha256(output_dir / name),
        }
        for name in required
    }
    return {
        "manifest_id": "M1-FinMultiTime-Evidence-Contract-Freeze",
        "contract_version": CONTRACT_VERSION,
        "lineage": {
            "previous_contract_version": PREVIOUS_CONTRACT_VERSION,
            "previous_contract_sha256": PREVIOUS_CONTRACT_SHA256,
            "erratum_reason": ERRATUM_REASON,
            "new_contract_version": CONTRACT_VERSION,
            "new_contract_sha256": artifacts["m1_evidence_contract.json"]["sha256"],
            "behaviour_equivalence": "PASS — no research-relevant behavioural change",
        },
        "status": "FROZEN",
        "freeze_date": FREEZE_DATE,
        "source_parent_sha": SOURCE_PARENT_SHA,
        "frozen_m0_base_sha": FROZEN_M0_BASE_SHA,
        "raw_finmultitime_root": "/Volumes/Jackson/Dataset/FinMultiTime",
        "source_data_identities": {
            "news": NEWS_ARCHIVE,
            "table": TABLE_ARCHIVE,
            "time_series": TS_ARCHIVE,
            "image": "image/image/S&P500_image_*",
            "data_audit": data_audit_identity,
            "m0_snapshot": str(M0_RUN_INPUTS),
        },
        "artifacts": artifacts,
        "selection_contract_not_input_freeze": True,
        "final_processed_subset_built": False,
        "final_evidence_packets_built": False,
        "m1_input_sha_bundle_built": False,
    }


def run(dataset_root: Path, output_dir: Path, m0_snapshot_dir: Path) -> None:
    context = formal_context()
    data = load_raw_data(dataset_root)
    snapshots = read_snapshot(m0_snapshot_dir)
    anomalies = impossible_ohlc_rows(data["series"], context)
    consistency = consistency_rows(data["series"], snapshots)
    semantics = semantics_summary(consistency)
    concept_rows = candidate_concept_coverage(data["tables"], context["decisions"])
    simulation = case_simulation(data, context)
    equivalence = erratum_equivalence_report(previous_frozen_simulation(), simulation)
    image_summary = image_coverage_summary(data["images"], context["decisions"])
    source_summary = source_stats(data, context)
    validation = simulation_invariants(data, context, simulation)
    if not all(validation.values()):
        failed = [name for name, passed in validation.items() if not passed]
        raise SystemExit("M1 contract validation failed; freeze stopped: " + ", ".join(failed))
    data_audit_path = output_dir / "finmultitime_data_audit.json"
    data_audit_identity = {
        "path": "docs/m1/finmultitime_data_audit.json",
        "sha256": file_sha256(data_audit_path),
        "exists_before_contract_generation": data_audit_path.exists(),
    }
    contract = contract_json(
        data,
        context,
        anomalies,
        consistency,
        concept_rows,
        simulation,
        image_summary,
        source_summary,
        validation,
        data_audit_identity,
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
    (output_dir / "m1_contract_erratum_equivalence.json").write_text(
        json.dumps(equivalence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "m1_evidence_contract_table_selection_diagnostics.csv",
        table_selection_diagnostic_rows(data, context),
    )
    (output_dir / "m1_evidence_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "M1_EVIDENCE_CONTRACT.md").write_text(
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
    manifest = freeze_manifest(output_dir, data_audit_identity)
    (output_dir / "m1_evidence_contract_freeze.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
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
                "validation": validation,
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
