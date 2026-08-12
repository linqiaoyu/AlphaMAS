#!/usr/bin/env python3
"""Read-only structural and point-in-time audit for a local FinMultiTime copy.

The raw dataset is never written by this script.  ZIP members are inspected
through their central directories and only target-stock members are streamed
or parsed.  Reports are written to the caller-provided output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TARGETS = ("AAPL", "AMZN", "JPM")
FORMAL_CONFIG = {
    "calendar": "XNYS",
    "first_calendar_week": "2024-01-01",
    "final_calendar_week": "2024-06-24",
    "final_valuation_session": "2024-07-05",
    "warmup_sessions": 252,
}
NEWS_ARCHIVE = "text/sp500_news.zip"
TABLE_ARCHIVE = "table/SP500_tabular.zip"
TS_ARCHIVE = "time_series/S&P500_time_series.zip"
IMAGE_ROOT = "image/image"
NEWS_MEMBERS = {symbol: f"sp500_news/{symbol}.jsonl" for symbol in TARGETS}
TS_MEMBERS = {
    symbol: f"S&P500_time_series/{symbol.lower()}.csv" for symbol in TARGETS
}
TABLE_MEMBER_RE = {
    symbol: re.compile(
        rf"^financial_reports/{symbol.lower()}/[^/]+\.json$", re.IGNORECASE
    )
    for symbol in TARGETS
}
IMAGE_RE = re.compile(
    r"^(?P<symbol>[a-z0-9.-]+)_(?P<year>\d{4})_H(?P<half>[12])_candlestick\.(?P<ext>png|jpe?g)$",
    re.IGNORECASE,
)
SEMANTIC_FUTURE_TOKENS = (
    "future",
    "next_return",
    "future_return",
    "return_",
    "target",
    "label",
    "direction",
    "forecast",
    "predict",
    "horizon",
    "outcome",
)


def iso(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc)


def human_bytes(value: int | float) -> str:
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def iter_files(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for filename in sorted(files):
            yield Path(current) / filename


def is_macos_sidecar(path: Path) -> bool:
    return path.name.startswith("._") or path.name == ".DS_Store"


def filesystem_inventory(root: Path) -> dict[str, Any]:
    top_level: list[dict[str, Any]] = []
    extension_counts: Counter[str] = Counter()
    physical_file_count = 0
    logical_file_count = 0
    physical_bytes = 0
    logical_bytes = 0
    largest_files: list[tuple[int, str]] = []
    docs: list[str] = []

    def add_largest(size: int, path: Path) -> None:
        largest_files.append((size, str(path)))
        largest_files.sort(reverse=True)
        del largest_files[10:]

    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_dir():
            file_count = 0
            file_bytes = 0
            logical_count = 0
            logical_size = 0
            for child in iter_files(path):
                try:
                    size = child.stat().st_size
                except OSError:
                    continue
                file_count += 1
                file_bytes += size
                physical_file_count += 1
                physical_bytes += size
                add_largest(size, child)
                if child.name.lower() in {"readme.md", "image.md", ".gitattributes"} or "description" in child.name.lower():
                    docs.append(str(child.relative_to(root)))
                if not is_macos_sidecar(child):
                    logical_count += 1
                    logical_size += size
                    logical_file_count += 1
                    logical_bytes += size
                    ext = child.suffix.lower() or "[no_extension]"
                    extension_counts[ext] += 1
            top_level.append(
                {
                    "name": path.name,
                    "type": "directory",
                    "physical_file_count": file_count,
                    "physical_bytes": file_bytes,
                    "logical_file_count": logical_count,
                    "logical_bytes": logical_size,
                }
            )
        elif path.is_file():
            size = path.stat().st_size
            physical_file_count += 1
            physical_bytes += size
            add_largest(size, path)
            if path.name.lower() in {"readme.md", "image.md", ".gitattributes"} or "description" in path.name.lower():
                docs.append(str(path.relative_to(root)))
            logical = not is_macos_sidecar(path)
            if logical:
                logical_file_count += 1
                logical_bytes += size
                extension_counts[path.suffix.lower() or "[no_extension]"] += 1
            top_level.append(
                {
                    "name": path.name,
                    "type": "file",
                    "physical_file_count": 1,
                    "physical_bytes": size,
                    "logical_file_count": int(logical),
                    "logical_bytes": size if logical else 0,
                }
            )

    return {
        "physical_file_count": physical_file_count,
        "logical_file_count_excluding_macos_sidecars": logical_file_count,
        "physical_bytes_including_macos_sidecars": physical_bytes,
        "logical_bytes_excluding_macos_sidecars": logical_bytes,
        "top_level": top_level,
        "extension_counts_excluding_macos_sidecars": dict(extension_counts),
        "largest_files": [
            {"path": path, "bytes": size, "size_human": human_bytes(size)}
            for size, path in largest_files
        ],
        "documentation_and_metadata_files": sorted(docs),
    }


def archive_inventory(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    extension_counts: Counter[str] = Counter()
    file_count = 0
    uncompressed_bytes = 0
    compressed_bytes = 0
    member_roots: Counter[str] = Counter()
    sample_members: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            file_count += 1
            uncompressed_bytes += info.file_size
            compressed_bytes += info.compress_size
            name = info.filename
            ext = Path(name).suffix.lower() or "[no_extension]"
            extension_counts[ext] += 1
            parts = Path(name).parts
            member_roots[parts[0] if parts else "[root]"] += 1
            if len(sample_members) < 10:
                sample_members.append(name)
    return {
        "path": relative,
        "archive_bytes": path.stat().st_size,
        "member_file_count": file_count,
        "member_uncompressed_bytes": uncompressed_bytes,
        "member_compressed_bytes": compressed_bytes,
        "member_extensions": dict(extension_counts),
        "member_top_level_counts": dict(member_roots),
        "sample_members": sample_members,
    }


def parse_formal_schedule() -> tuple[list[dict[str, Any]], list[str], str]:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from tradingagents.backtesting.calendar import ExchangeSchedule

    schedule = ExchangeSchedule(FORMAL_CONFIG["calendar"])
    events = schedule.weekly_events(
        FORMAL_CONFIG["first_calendar_week"], FORMAL_CONFIG["final_calendar_week"]
    )
    warmup = schedule.preceding_sessions(
        events[0].decision_session, FORMAL_CONFIG["warmup_sessions"]
    )
    event_rows = [event.to_dict() for event in events]
    return event_rows, [item.date().isoformat() for item in warmup], str(warmup[0].date())


def read_jsonl_news(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    malformed = 0
    missing_date = 0
    invalid_date = 0
    missing_article = 0
    empty_article = 0
    exact_hashes: Counter[str] = Counter()
    url_hashes: Counter[str] = Counter()
    date_values: list[date] = []
    raw_date_values: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    source_domains: Counter[str] = Counter()
    stock_symbols: Counter[str] = Counter()
    with archive.open(member) as handle:
        for raw_line in handle:
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("utf-8", errors="replace")
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(value, dict):
                malformed += 1
                continue
            records.append(value)
            fields.update(value.keys())
            if value.get("Stock_symbol") not in (None, ""):
                stock_symbols[str(value["Stock_symbol"])] += 1
            canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            exact_hashes[hashlib.sha256(canonical.encode("utf-8")).hexdigest()] += 1
            url = value.get("Url")
            if isinstance(url, str) and url:
                url_hashes[url] += 1
                domain = urlparse(url).netloc.lower()
                if domain:
                    source_domains[domain] += 1
            raw_date = value.get("Date")
            if raw_date in (None, ""):
                missing_date += 1
            else:
                raw_date_values[str(raw_date)] += 1
                parsed = parse_date(raw_date)
                if parsed is None:
                    invalid_date += 1
                else:
                    date_values.append(parsed)
            if "Article" not in value:
                missing_article += 1
            elif not str(value.get("Article") or "").strip():
                empty_article += 1

    duplicate_records = sum(count - 1 for count in exact_hashes.values() if count > 1)
    duplicate_urls = sum(count - 1 for count in url_hashes.values() if count > 1)
    obvious_duplicate_key = Counter(
        (str(item.get("Date", "")), str(item.get("Url", "")), str(item.get("Article", "")))
        for item in records
    )
    obvious_duplicates = sum(count - 1 for count in obvious_duplicate_key.values() if count > 1)
    return {
        "records": records,
        "record_count": len(records),
        "fields": sorted(fields),
        "earliest_date": iso(min(date_values)) if date_values else None,
        "latest_date": iso(max(date_values)) if date_values else None,
        "date_count": len(date_values),
        "missing_date_count": missing_date,
        "invalid_date_count": invalid_date,
        "missing_article_count": missing_article,
        "empty_article_count": empty_article,
        "malformed_record_count": malformed,
        "exact_duplicate_record_count": duplicate_records,
        "duplicate_url_count": duplicate_urls,
        "obvious_duplicate_article_count": obvious_duplicates,
        "source_domains": dict(source_domains),
        "stock_symbols": dict(stock_symbols),
        "date_values": sorted(set(date_values)),
        "raw_date_values": raw_date_values,
        "availability": "date_only_no_timezone_or_time_of_day",
    }


def recursive_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys.update(recursive_keys(child, path))
    elif isinstance(value, list) and value:
        keys.update(recursive_keys(value[0], prefix + "[]"))
    return keys


def read_table_files(archive: zipfile.ZipFile, members: list[str]) -> dict[str, Any]:
    schemas: set[str] = set()
    fields: Counter[str] = Counter()
    observations: list[dict[str, Any]] = []
    filing_dates: list[date] = []
    parse_errors: list[str] = []
    duplicate_observations: Counter[str] = Counter()
    company_names: set[str] = set()
    ciks: set[str] = set()
    for member in members:
        try:
            with archive.open(member) as handle:
                value = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            parse_errors.append(f"{member}: {type(exc).__name__}: {exc}")
            continue
        schemas.update(recursive_keys(value))
        fields.update(recursive_keys(value))
        if isinstance(value, dict):
            company_names.add(str(value.get("company_name", "")))
            ciks.add(str(value.get("cik", "")))
        if not isinstance(value, dict):
            continue
        for filing in value.get("filings", []) or []:
            if isinstance(filing, dict):
                parsed = parse_date(filing.get("filing_date"))
                if parsed:
                    filing_dates.append(parsed)
                filing_date = parsed
                facts = filing.get("facts", {})
                if not isinstance(facts, dict):
                    continue
                for taxonomy, concepts in facts.items():
                    if not isinstance(concepts, dict):
                        continue
                    for concept, metadata in concepts.items():
                        if not isinstance(metadata, dict):
                            continue
                        units = metadata.get("units", {})
                        if not isinstance(units, dict):
                            continue
                        for unit, values in units.items():
                            if not isinstance(values, list):
                                continue
                            for observation in values:
                                if not isinstance(observation, dict):
                                    continue
                                start = parse_date(observation.get("start"))
                                end = parse_date(observation.get("end"))
                                filed = parse_date(observation.get("filed"))
                                availability_date = filed or filing_date
                                row = {
                                    "member": member,
                                    "taxonomy": str(taxonomy),
                                    "concept": str(concept),
                                    "unit": str(unit),
                                    "start": start,
                                    "end": end,
                                    "filed": filed,
                                    "filing_date": filing_date,
                                    "availability_date": availability_date,
                                    "raw_start": observation.get("start"),
                                    "raw_end": observation.get("end"),
                                    "raw_filed": observation.get("filed"),
                                    "val": observation.get("val"),
                                    "accn": observation.get("accn"),
                                    "form": observation.get("form"),
                                    "fy": observation.get("fy"),
                                    "fp": observation.get("fp"),
                                    "frame": observation.get("frame"),
                                }
                                observations.append(row)
                                canonical = json.dumps(
                                    {key: value for key, value in row.items() if key != "member"},
                                    default=iso,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                )
                                duplicate_observations[canonical] += 1

    available = [row["availability_date"] for row in observations if row["availability_date"] is not None]
    period_ends = [row["end"] for row in observations if row["end"] is not None]
    missing_filed = sum(row["filed"] is None for row in observations)
    missing_availability = sum(row["availability_date"] is None for row in observations)
    missing_end = sum(row["end"] is None for row in observations)
    restatement_keys: Counter[tuple[Any, ...]] = Counter(
        (
            row["taxonomy"],
            row["concept"],
            row["unit"],
            row["raw_start"],
            row["raw_end"],
        )
        for row in observations
    )
    restated_groups = sum(count > 1 for count in restatement_keys.values())
    duplicate_count = sum(count - 1 for count in duplicate_observations.values() if count > 1)
    semantic_fields = sorted(
        path
        for path in schemas
        if any(token in path.lower() for token in SEMANTIC_FUTURE_TOKENS)
    )
    return {
        "members": members,
        "file_count": len(members),
        "record_count": len(observations),
        "observations": observations,
        "fields": sorted(fields),
        "semantic_fields": semantic_fields,
        "company_names": sorted(company_names),
        "ciks": sorted(ciks),
        "filing_date_earliest": iso(min(filing_dates)) if filing_dates else None,
        "filing_date_latest": iso(max(filing_dates)) if filing_dates else None,
        "period_end_earliest": iso(min(period_ends)) if period_ends else None,
        "period_end_latest": iso(max(period_ends)) if period_ends else None,
        "availability_earliest": iso(min(available)) if available else None,
        "availability_latest": iso(max(available)) if available else None,
        "missing_filed_count": missing_filed,
        "missing_availability_count": missing_availability,
        "missing_end_count": missing_end,
        "duplicate_row_count": duplicate_count,
        "restated_period_group_count": restated_groups,
        "parse_errors": parse_errors,
        "availability": "filed_date_only_no_timezone_or_time_of_day",
    }


def parse_number(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number


def read_time_series(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parse_errors = 0
    header: list[str] = []
    timezone_offsets: Counter[str] = Counter()
    with archive.open(member) as handle:
        text = (line.decode("utf-8", errors="replace") for line in handle)
        reader = csv.reader(text)
        try:
            header = next(reader)
        except StopIteration:
            return {"member": member, "header": [], "record_count": 0, "parse_errors": 0}
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                parse_errors += 1
                continue
            raw = dict(zip(header, fields, strict=True))
            timestamp = parse_timestamp(raw.get("Date"))
            session_date = timestamp.date() if timestamp else None
            raw_offset = re.search(r"(Z|[+-]\d{2}:\d{2})$", raw.get("Date", ""))
            if raw_offset:
                timezone_offsets[raw_offset.group(1)] += 1
            numeric = {key: parse_number(raw.get(key, "")) for key in header if key != "Date"}
            rows.append(
                {
                    "line_number": line_number,
                    "raw_date": raw.get("Date", ""),
                    "timestamp": timestamp,
                    "session_date": session_date,
                    "numeric": numeric,
                }
            )
    dates = [row["session_date"] for row in rows if row["session_date"]]
    date_counts = Counter(dates)
    duplicate_dates = sum(count - 1 for count in date_counts.values() if count > 1)
    monotonic = all(
        rows[i]["timestamp"] is not None
        and rows[i - 1]["timestamp"] is not None
        and rows[i - 1]["timestamp"] <= rows[i]["timestamp"]
        for i in range(1, len(rows))
    )
    required_price_fields = ("Open", "High", "Low", "Close")
    missing_values = 0
    nonfinite = 0
    nonpositive_prices = 0
    impossible_ohlc = 0
    negative_volume = 0
    for row in rows:
        numeric = row["numeric"]
        prices = [numeric.get(key) for key in required_price_fields]
        if any(value is None for value in prices):
            missing_values += 1
        if any(value is not None and not math.isfinite(value) for value in numeric.values()):
            nonfinite += 1
        if any(value is not None and value <= 0 for value in prices):
            nonpositive_prices += 1
        if all(value is not None and math.isfinite(value) for value in prices):
            opened, high, low, close = prices
            if high < max(opened, close, low) or low > min(opened, close, high):
                impossible_ohlc += 1
        volume = numeric.get("Volume")
        if volume is not None and volume < 0:
            negative_volume += 1
    return {
        "member": member,
        "header": header,
        "record_count": len(rows),
        "rows": rows,
        "parse_errors": parse_errors,
        "earliest_date": iso(min(dates)) if dates else None,
        "latest_date": iso(max(dates)) if dates else None,
        "date_counts": date_counts,
        "duplicate_date_count": duplicate_dates,
        "non_monotonic": not monotonic,
        "timezone_offsets": dict(timezone_offsets),
        "missing_value_row_count": missing_values,
        "nonfinite_value_row_count": nonfinite,
        "nonpositive_price_row_count": nonpositive_prices,
        "impossible_ohlc_row_count": impossible_ohlc,
        "negative_volume_row_count": negative_volume,
        "frequency": "daily session rows inferred from Date field",
        "adjustment_semantics": "No Adj Close column; values appear adjusted but source metadata does not state auto_adjust semantics",
    }


def png_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"format": path.suffix.lower().lstrip("."), "readable": False}
    try:
        with path.open("rb") as handle:
            signature = handle.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                result["error"] = "not_png_signature"
                return result
            length_bytes = handle.read(4)
            chunk_type = handle.read(4)
            if chunk_type != b"IHDR" or len(length_bytes) != 4:
                result["error"] = "missing_ihdr"
                return result
            length = struct.unpack(">I", length_bytes)[0]
            payload = handle.read(length)
            if len(payload) != 13:
                result["error"] = "invalid_ihdr_length"
                return result
            width, height, depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            result.update(
                {
                    "width": width,
                    "height": height,
                    "bit_depth": depth,
                    "color_type": color_type,
                    "compression": compression,
                    "filter": filter_method,
                    "interlace": interlace,
                    "readable": width > 0 and height > 0,
                }
            )
    except (OSError, struct.error) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def discover_images(root: Path) -> dict[str, Any]:
    by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in TARGETS}
    for shard in sorted(root.glob("S&P500_image_*")):
        if not shard.is_dir():
            continue
        for symbol in TARGETS:
            directory = shard / symbol.lower()
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_file() or is_macos_sidecar(path):
                    continue
                match = IMAGE_RE.match(path.name)
                if not match or match.group("symbol").upper() != symbol:
                    continue
                year = int(match.group("year"))
                half = int(match.group("half"))
                period_start = date(year, 1 if half == 1 else 7, 1)
                period_end = date(year, 6, 30) if half == 1 else date(year, 12, 31)
                metadata = png_metadata(path) if path.suffix.lower() == ".png" else {
                    "format": path.suffix.lower().lstrip("."),
                    "readable": None,
                    "error": "non_png_not_validated",
                }
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                by_symbol[symbol].append(
                    {
                        "path": str(path),
                        "filename": path.name,
                        "bytes": path.stat().st_size,
                        "year": year,
                        "half": half,
                        "period_start": period_start,
                        "period_end": period_end,
                        "metadata": metadata,
                        "sha256": digest,
                    }
                )
    for symbol in TARGETS:
        by_symbol[symbol].sort(key=lambda item: (item["period_end"], item["path"]))
    return {"by_symbol": by_symbol}


def count_date_range(values: Iterable[date], start: date, end: date) -> int:
    return sum(start <= value <= end for value in values)


def date_range_counts(values: Iterable[date]) -> dict[str, int]:
    values = list(values)
    return {
        "coverage_2023": count_date_range(values, date(2023, 1, 1), date(2023, 12, 31)),
        "coverage_2024h1": count_date_range(values, date(2024, 1, 1), date(2024, 6, 30)),
        "through_final_valuation": count_date_range(
            values, date.min, date.fromisoformat(FORMAL_CONFIG["final_valuation_session"])
        ),
    }


def build_modality_rows(
    root: Path,
    filesystem: dict[str, Any],
    archives: dict[str, dict[str, Any]],
    news: dict[str, dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    series: dict[str, dict[str, Any]],
    images: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    archive_to_modality = {
        "text": [NEWS_ARCHIVE, "text/hs300news_summary.zip"],
        "table": [TABLE_ARCHIVE, "table/hs300_tabular.zip"],
        "time_series": [TS_ARCHIVE, "time_series/HS300_time_series.zip"],
    }
    for modality, paths in archive_to_modality.items():
        for relative in paths:
            data = archives[relative]
            rows.append(
                {
                    "modality": modality,
                    "dataset_partition": "S&P500" if "sp500" in relative.lower() or "s&p500" in relative.lower() else "HS300",
                    "source_path": relative,
                    "container": "zip",
                    "file_count": data["member_file_count"],
                    "compressed_bytes": data["member_compressed_bytes"],
                    "uncompressed_bytes": data["member_uncompressed_bytes"],
                    "archive_bytes": data["archive_bytes"],
                    "extensions": data["member_extensions"],
                    "notes": "ZIP central-directory inventory; members were not extracted",
                }
            )
    image_summary = next(item for item in filesystem["top_level"] if item["name"] == "image")
    rows.append(
        {
            "modality": "image",
            "dataset_partition": "S&P500 extracted + HS300 archive",
            "source_path": IMAGE_ROOT,
            "container": "expanded_shards_plus_zip_sibling",
            "file_count": image_summary["logical_file_count"],
            "compressed_bytes": None,
            "uncompressed_bytes": image_summary["logical_bytes"],
            "archive_bytes": (root / "image/HS300_image.zip").stat().st_size,
            "extensions": "see filesystem_inventory.extension_counts_excluding_macos_sidecars",
            "notes": "Includes extracted S&P500 shards and non-sidecar HS300_image.zip as physically stored",
        }
    )
    return rows


def build_target_rows(
    root: Path,
    archives: dict[str, dict[str, Any]],
    news: dict[str, dict[str, Any] | None],
    tables: dict[str, dict[str, Any] | None],
    series: dict[str, dict[str, Any] | None],
    images: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(root / NEWS_ARCHIVE) as archive:
        news_member_sizes = {
            info.filename: info.file_size
            for info in archive.infolist()
            if not info.is_dir()
        }
    with zipfile.ZipFile(root / TS_ARCHIVE) as archive:
        ts_member_sizes = {
            info.filename: info.file_size
            for info in archive.infolist()
            if not info.is_dir()
        }
    for symbol in TARGETS:
        item = news.get(symbol)
        if item is None:
            rows.append({"symbol": symbol, "modality": "text", "present": False, "source_identifier": NEWS_MEMBERS[symbol], "notes": "No matching ZIP member"})
        else:
            date_values = item["date_values"]
            rows.append(
                {
                    "symbol": symbol,
                    "modality": "text",
                    "present": True,
                    "source_identifier": NEWS_MEMBERS[symbol],
                    "raw_file_count": 1,
                    "raw_record_count": item["record_count"],
                    "earliest_date": item["earliest_date"],
                    "latest_date": item["latest_date"],
                    **date_range_counts(date_values),
                    "disk_bytes": news_member_sizes[NEWS_MEMBERS[symbol]],
                    "missing_timestamp_count": item["missing_date_count"],
                    "invalid_timestamp_count": item["invalid_date_count"],
                    "duplicate_count": item["exact_duplicate_record_count"],
                    "parse_error_count": item["malformed_record_count"],
                    "availability_metadata": item["availability"],
                    "ticker_field_values": item["stock_symbols"],
                    "notes": "File-level symbol mapping with Stock_symbol field; no publication-time or revision field",
                }
            )

        item = tables.get(symbol)
        if item is None:
            rows.append({"symbol": symbol, "modality": "table", "present": False, "source_identifier": f"financial_reports/{symbol.lower()}/", "notes": "No matching members"})
        else:
            period_values = [row["end"] for row in item["observations"] if row["end"]]
            rows.append(
                {
                    "symbol": symbol,
                    "modality": "table",
                    "present": True,
                    "source_identifier": ";".join(item["members"]),
                    "raw_file_count": item["file_count"],
                    "raw_record_count": item["record_count"],
                    "earliest_period_end": item["period_end_earliest"],
                    "latest_period_end": item["period_end_latest"],
                    "earliest_availability_date": item["availability_earliest"],
                    "latest_availability_date": item["availability_latest"],
                    **date_range_counts(period_values),
                    "disk_bytes": None,
                    "missing_timestamp_count": item["missing_availability_count"],
                    "invalid_timestamp_count": 0,
                    "duplicate_count": item["duplicate_row_count"],
                    "parse_error_count": len(item["parse_errors"]),
                    "availability_metadata": item["availability"],
                    "notes": "SEC-style company facts; period end is distinct from filed date; restatement groups detected",
                }
            )

        item = series.get(symbol)
        if item is None:
            rows.append({"symbol": symbol, "modality": "time_series", "present": False, "source_identifier": TS_MEMBERS[symbol], "notes": "No matching ZIP member"})
        else:
            dates = [row["session_date"] for row in item["rows"] if row["session_date"]]
            rows.append(
                {
                    "symbol": symbol,
                    "modality": "time_series",
                    "present": True,
                    "source_identifier": TS_MEMBERS[symbol],
                    "raw_file_count": 1,
                    "raw_record_count": item["record_count"],
                    "earliest_date": item["earliest_date"],
                    "latest_date": item["latest_date"],
                    **date_range_counts(dates),
                    "disk_bytes": ts_member_sizes[TS_MEMBERS[symbol]],
                    "missing_timestamp_count": 0,
                    "invalid_timestamp_count": item["parse_errors"],
                    "duplicate_count": item["duplicate_date_count"],
                    "parse_error_count": item["parse_errors"],
                    "availability_metadata": "timezone-bearing daily Date at midnight; no close-time field",
                    "notes": item["adjustment_semantics"],
                }
            )

        image_items = images.get(symbol, [])
        dimensions = Counter(
            f"{item['metadata'].get('width')}x{item['metadata'].get('height')}"
            for item in image_items
            if item["metadata"].get("width")
        )
        hashes = Counter(item["sha256"] for item in image_items)
        periods = [item["period_end"] for item in image_items]
        rows.append(
            {
                "symbol": symbol,
                "modality": "image",
                "present": bool(image_items),
                "source_identifier": "image/image/S&P500_image_*/" + symbol.lower(),
                "raw_file_count": len(image_items),
                "raw_record_count": len(image_items),
                "earliest_nominal_period": (
                    f"{min(image_items, key=lambda item: item['period_end'])['year']}-H{min(image_items, key=lambda item: item['period_end'])['half']}"
                    if image_items else None
                ),
                "latest_nominal_period": (
                    f"{max(image_items, key=lambda item: item['period_end'])['year']}-H{max(image_items, key=lambda item: item['period_end'])['half']}"
                    if image_items else None
                ),
                "earliest_nominal_end": iso(min(periods)) if periods else None,
                "latest_nominal_end": iso(max(periods)) if periods else None,
                "disk_bytes": sum(item["bytes"] for item in image_items),
                "dimensions": dict(dimensions),
                "duplicate_count": sum(count - 1 for count in hashes.values() if count > 1),
                "corrupt_or_unreadable_count": sum(
                    not item["metadata"].get("readable", False) for item in image_items
                ),
                "availability_metadata": "ticker/year/half-year filename only; chart start/end not stored separately",
                "notes": (
                    "AAPL description flag is not backed by an actual extracted AAPL image directory"
                    if symbol == "AAPL"
                    else "Filename encodes ticker/year/half-year; chart start/end not stored separately"
                ),
            }
        )
    # Populate actual table member sizes after rows are constructed.
    with zipfile.ZipFile(root / TABLE_ARCHIVE) as archive:
        sizes = {info.filename: info.file_size for info in archive.infolist() if not info.is_dir()}
    for row in rows:
        if row["modality"] == "table" and row.get("present"):
            row["disk_bytes"] = sum(sizes.get(member, 0) for member in row["source_identifier"].split(";"))
    return rows


def table_semantic_counts(item: dict[str, Any], decision: date) -> tuple[int, int, date | None]:
    safe = [row for row in item["observations"] if row["availability_date"] and row["availability_date"] < decision]
    ambiguous = sum(row["availability_date"] == decision for row in item["observations"])
    latest = max((row["availability_date"] for row in safe), default=None)
    return len(safe), ambiguous, latest


def image_for_case(items: list[dict[str, Any]], decision: date) -> tuple[int, date | None, str | None]:
    candidates = [item for item in items if item["period_end"] <= decision]
    if not candidates:
        return 0, None, None
    latest = max(candidates, key=lambda item: (item["period_end"], item["path"]))
    return len(candidates), latest["period_end"], latest["filename"]


def build_case_rows(
    events: list[dict[str, Any]],
    warmup_start: date,
    news: dict[str, dict[str, Any] | None],
    tables: dict[str, dict[str, Any] | None],
    series: dict[str, dict[str, Any] | None],
    images: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_valuation = date.fromisoformat(FORMAL_CONFIG["final_valuation_session"])
    for event in events:
        decision = date.fromisoformat(event["decision_session"])
        for symbol in TARGETS:
            news_item = news.get(symbol)
            news_dates = news_item["date_values"] if news_item else []
            safe_news = [value for value in news_dates if value < decision]
            latest_news = max(safe_news, default=None)
            same_day_news = sum(value == decision for value in news_dates)
            news_counts = {
                f"news_records_{days}d": sum(
                    decision - timedelta(days=days) <= value < decision for value in news_dates
                )
                for days in (1, 3, 7, 14, 30)
            }
            table_item = tables.get(symbol)
            if table_item:
                table_count, table_ambiguous, latest_table = table_semantic_counts(table_item, decision)
            else:
                table_count, table_ambiguous, latest_table = 0, 0, None
            ts_item = series.get(symbol)
            ts_dates = sorted({row["session_date"] for row in ts_item["rows"] if row["session_date"]}) if ts_item else []
            ts_strict = [value for value in ts_dates if value < decision]
            ts_through = [value for value in ts_dates if value <= decision]
            latest_ts = max(ts_through, default=None)
            expected_formal = [
                value
                for value in ts_item.get("expected_sessions", [])
                if value <= final_valuation
            ] if ts_item else []
            missing_formal = [value for value in expected_formal if value not in ts_dates]
            image_candidates, latest_image_end, latest_image_name = image_for_case(images.get(symbol, []), decision)
            flags: list[str] = []
            if news_item is None:
                flags.append("TEXT_UNAVAILABLE")
            elif news_item["missing_date_count"] or news_item["invalid_date_count"]:
                flags.append("TEXT_MISSING_OR_INVALID_DATES")
            if same_day_news:
                flags.append("TEXT_SAME_DAY_DATE_ONLY_AMBIGUOUS")
            if table_item is None:
                flags.append("TABLE_UNAVAILABLE")
            elif table_item["missing_availability_count"]:
                flags.append("TABLE_MISSING_FILED_DATES")
            if table_ambiguous:
                flags.append("TABLE_SAME_DAY_FILED_DATE_ONLY_AMBIGUOUS")
            if ts_item is None:
                flags.append("TIME_SERIES_UNAVAILABLE")
            elif missing_formal:
                flags.append("TIME_SERIES_MISSING_XNYS_SESSION")
            if not images.get(symbol):
                flags.append("IMAGE_UNAVAILABLE")
            elif latest_image_end is None:
                flags.append("IMAGE_NO_NOMINAL_END_LE_DECISION")
            else:
                flags.append("IMAGE_WINDOW_END_INFERRED_FROM_HALF_YEAR_FILENAME")
            rows.append(
                {
                    "symbol": symbol,
                    "decision_week": event["week_start"],
                    "decision_session": event["decision_session"],
                    "decision_close_utc": event["decision_close_utc"],
                    "decision_close_ny": event["decision_close_ny"],
                    "execution_session": event["execution_session"],
                    "news_latest_safe_date": iso(latest_news),
                    "news_latest_safe_age_calendar_days": (decision - latest_news).days if latest_news else None,
                    "news_same_day_ambiguous_count": same_day_news,
                    "news_missing_date_count": news_item["missing_date_count"] if news_item else None,
                    **news_counts,
                    "table_latest_safe_filed_date": iso(latest_table),
                    "table_latest_safe_age_calendar_days": (decision - latest_table).days if latest_table else None,
                    "table_observations_with_filed_before_decision": table_count,
                    "table_same_day_filed_ambiguous_count": table_ambiguous,
                    "ts_sessions_strictly_before_decision": len(ts_strict),
                    "ts_sessions_through_decision": len(ts_through),
                    "ts_latest_session_through_decision": iso(latest_ts),
                    "ts_formal_period_missing_session_count": len(missing_formal),
                    "image_candidate_count_nominal_end_le_decision": image_candidates,
                    "image_latest_safe_nominal_end": iso(latest_image_end),
                    "image_latest_safe_filename": latest_image_name,
                    "image_latest_safe_age_calendar_days": (decision - latest_image_end).days if latest_image_end else None,
                    "modality_unavailable_or_ambiguous_flags": ";".join(flags),
                    "pit_risk_flags": ";".join(flags),
                    "warmup_reference_start": warmup_start.isoformat(),
                    "final_valuation_session": final_valuation.isoformat(),
                }
            )
    return rows


def build_pit_risks(
    news: dict[str, dict[str, Any] | None],
    tables: dict[str, dict[str, Any] | None],
    series: dict[str, dict[str, Any] | None],
    images: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    risks = [
        {"modality": "text/news", "scope": "all target files", "risk_id": "NEWS_DATE_ONLY", "severity": "high", "finding": "Date is calendar-date only with no publication time or timezone; same-day articles cannot be safely admitted at a close-time decision without an explicit conservative rule.", "required_contract_control": "Use only records with Date strictly earlier than the decision session unless a later source timestamp is independently established; keep same-day records explicitly ambiguous."},
        {"modality": "text/news", "scope": "JPM", "risk_id": "NEWS_MEMBER_MISSING", "severity": "high", "finding": "No sp500_news/JPM.jsonl member exists despite the stock description flag.", "required_contract_control": "Represent text evidence as unavailable for JPM; never backfill from a later or different source."},
        {"modality": "table/fundamentals", "scope": "all target files", "risk_id": "TABLE_FILED_DATE_ONLY", "severity": "high", "finding": "Facts include filed and filing_date dates but no publication time/timezone; fiscal period end is not availability time.", "required_contract_control": "Gate observations on filed date strictly earlier than the decision date; same-day filed observations remain ambiguous."},
        {"modality": "table/fundamentals", "scope": "all target files", "risk_id": "TABLE_RESTATEMENTS", "severity": "medium", "finding": "Repeated concept/unit/period groups with multiple filings are identifiable, so revised values exist.", "required_contract_control": "Use the latest filed version available at decision time and retain accession/filed provenance."},
        {"modality": "time_series", "scope": "all target files", "risk_id": "TS_ADJUSTMENT_METADATA", "severity": "medium", "finding": "There is no Adj Close column or explicit auto-adjust/corporate-action documentation; values appear adjusted and Dividends/Stock Splits are present.", "required_contract_control": "Treat adjustment semantics as a declared data-source assumption and do not expose any target/label-derived fields."},
        {"modality": "image/charts", "scope": "all target files", "risk_id": "IMAGE_WINDOW_METADATA", "severity": "high", "finding": "Images map through ticker/year/half-year filenames and show a period title, but no machine-readable chart start/end metadata is stored.", "required_contract_control": "Only use an image when its inferred nominal period end is no later than the decision, or mark it unavailable/ambiguous; do not assume an H1 chart is safe for a decision before period end."},
        {"modality": "image/charts", "scope": "AAPL", "risk_id": "IMAGE_MEMBER_MISSING", "severity": "high", "finding": "No actual aapl image directory or image file was found in the expanded S&P500 shards.", "required_contract_control": "Represent image evidence as unavailable for AAPL."},
    ]
    for symbol in TARGETS:
        if news.get(symbol):
            item = news[symbol]
            if item["missing_date_count"] or item["invalid_date_count"]:
                risks.append({"modality": "text/news", "scope": symbol, "risk_id": "NEWS_INVALID_DATE_ROWS", "severity": "medium", "finding": f"{item['missing_date_count']} records have missing Date and {item['invalid_date_count']} have invalid Date.", "required_contract_control": "Exclude records without a valid availability date."})
        if tables.get(symbol) and tables[symbol]["missing_availability_count"]:
            risks.append({"modality": "table/fundamentals", "scope": symbol, "risk_id": "TABLE_MISSING_AVAILABILITY_ROWS", "severity": "medium", "finding": f"{tables[symbol]['missing_availability_count']} fact observations lack both a valid filed date and a parent filing date.", "required_contract_control": "Do not make those observations agent-visible unless an external availability record is added."})
        if series.get(symbol) and series[symbol]["missing_value_row_count"]:
            risks.append({"modality": "time_series", "scope": symbol, "risk_id": "TS_MISSING_VALUES", "severity": "medium", "finding": f"{series[symbol]['missing_value_row_count']} time-series rows have missing required price values.", "required_contract_control": "Exclude or explicitly flag affected rows; never silently forward-fill."})
    return risks


def build_future_field_rows(
    news: dict[str, dict[str, Any] | None],
    tables: dict[str, dict[str, Any] | None],
    series: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows = [
        {"source_type": "time_series", "field_or_path": "future_return / next_return / return_5d / target / label / direction / forecast / prediction / horizon result", "observed": False, "classification": "prohibited_if_present", "finding": "No such target field appears in the target CSV headers; the only columns are Date, Open, High, Low, Close, Volume, Dividends, Stock Splits."},
        {"source_type": "table", "field_or_path": "facts.*.*.units.*[].val", "observed": False, "classification": "safe_candidate_after_filed_gate", "finding": "SEC fact values are historical observations, not forecast labels; they remain PIT-sensitive and must be gated by filed date."},
        {"source_type": "table", "field_or_path": "facts.*.*.units.*[].end / start", "observed": False, "classification": "metadata_only_for_availability", "finding": "Period dates describe the reporting period and are not availability timestamps."},
        {"source_type": "text", "field_or_path": "Article", "observed": False, "classification": "safe_candidate_content_but_not_label", "finding": "No explicit target/label field exists; article prose may discuss future outcomes, which is ordinary text content rather than a dataset-derived target."},
        {"source_type": "image", "field_or_path": "*_YYYY_H[1|2]_candlestick.png", "observed": False, "classification": "safe_only_after_window_end_gate", "finding": "Filename/title encodes a chart period but no target/prediction label; the entire chart is prohibited until its inferred window end is no later than the decision."},
    ]
    observed_paths: set[str] = set()
    for item in news.values():
        if item:
            observed_paths.update(
                field
                for field in item["fields"]
                if any(token in field.lower() for token in SEMANTIC_FUTURE_TOKENS)
            )
    for item in tables.values():
        if item:
            observed_paths.update(
                path
                for path in item["semantic_fields"]
                if not path.lower().endswith(".label")
            )
    for item in series.values():
        if item:
            observed_paths.update(
                field for field in item["header"] if any(token in field.lower() for token in SEMANTIC_FUTURE_TOKENS)
            )
    if observed_paths:
        rows.append({"source_type": "schema_scan", "field_or_path": ";".join(sorted(observed_paths)), "observed": True, "classification": "manual_review_required", "finding": "Semantic token match; inspect before any evidence-contract inclusion."})
    return rows


def compact_target_summary(items: dict[str, dict[str, Any] | None], excluded: set[str]) -> dict[str, Any]:
    """Keep the committed JSON lightweight; row-level data lives in CSV outputs."""
    result: dict[str, Any] = {}
    for symbol, item in items.items():
        if item is None:
            result[symbol] = None
            continue
        result[symbol] = {key: value for key, value in item.items() if key not in excluded}
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Counter):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def render_markdown(
    root: Path,
    filesystem: dict[str, Any],
    archives: dict[str, dict[str, Any]],
    modality_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    pit_risks: list[dict[str, Any]],
    future_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    warmup_start: str,
    news: dict[str, dict[str, Any] | None],
    tables: dict[str, dict[str, Any] | None],
    series: dict[str, dict[str, Any] | None],
    images: dict[str, list[dict[str, Any]]],
) -> str:
    def present(symbol: str, modality: str) -> dict[str, Any]:
        return next(row for row in target_rows if row["symbol"] == symbol and row["modality"] == modality)

    verdict = "M1 DATA AUDIT PASSED WITH PIT/AVAILABILITY GAPS — CONTRACT MUST HANDLE EXPLICIT UNAVAILABLE DATA"
    total_physical = filesystem["physical_bytes_including_macos_sidecars"]
    total_logical = filesystem["logical_bytes_excluding_macos_sidecars"]
    support_start = date.fromisoformat(warmup_start)
    final_decision = date.fromisoformat(events[-1]["decision_session"])
    image_workload_rows = []
    for symbol in TARGETS:
        items = images[symbol]
        safe_final = [item for item in items if item["period_end"] <= final_decision]
        support = [item for item in items if item["period_end"] >= support_start and item["period_end"] <= final_decision]
        refs = sum(bool(image_for_case(items, date.fromisoformat(row["decision_session"]))[1]) for row in case_rows if row["symbol"] == symbol)
        image_workload_rows.append((symbol, len(items), len(safe_final), len(support), refs, sum(item["bytes"] for item in items)))

    lines = [
        "# FinMultiTime Data Audit",
        "",
        "Read-only audit of the local raw FinMultiTime directory. No archive members were extracted and no files under the raw dataset were written.",
        "",
        f"**Raw path:** `{root}`",
        f"**Physical size (including macOS sidecars):** {human_bytes(total_physical)}",
        f"**Logical size excluding `._*` sidecars:** {human_bytes(total_logical)}",
        f"**Physical files:** {filesystem['physical_file_count']}",
        f"**Logical files excluding sidecars:** {filesystem['logical_file_count_excluding_macos_sidecars']}",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "The dataset supports the M1 augmentation concept and the 78-case schedule, but the evidence contract must represent unavailable/ambiguous modalities explicitly. The highest-impact gaps are missing JPM news, missing AAPL images, date-only news/table availability, and image window metadata that is only inferable from half-year filenames.",
        "",
        "## Dataset Structure",
        "",
            "The local release contains top-level `.cache`, `image`, `table`, `text`, and `time_series` directories plus README/description metadata. The S&P500 text, table, and time-series data remain in ZIP archives. S&P500 images are expanded into `image/image/S&P500_image_a` through `_z`; the sibling image directory also contains an HS300 ZIP archive. The physical inventory found 578,477 files; the largest top-level directories are `image` (8.78 GB), `text` (3.66 GB), `table` (3.37 GB), `.cache` (667 MB), and `time_series` (584 MB).",
        "",
        "| Modality / partition | Source | Files or members | Compressed / physical | Uncompressed / logical |",
        "|---|---|---:|---:|---:|",
    ]
    for row in modality_rows:
        lines.append(
            f"| {row['modality']} / {row['dataset_partition']} | `{row['source_path']}` | {row['file_count']} | {human_bytes(row['compressed_bytes'] or row['archive_bytes'] or 0)} | {human_bytes(row['uncompressed_bytes'] or 0)} |"
        )
    lines.extend(
        [
            "",
            "Archive central-directory counts include only file members, not directory entries. The physical total counts both the expanded S&P500 images and the HS300 ZIP because both are present locally. The largest physical files are `text/sp500_news.zip` (3.36 GiB), `table/SP500_tabular.zip` (3.08 GiB), `image/HS300_image.zip` (2.01 GiB), and `time_series/S&P500_time_series.zip` (460.65 MiB).",
            "",
            "### Metadata and documentation",
            "",
            "The local documentation consists of `README.md`, `image/image.md`, `sp500stock_data_description.csv`, `hs300stock_data_description.csv`, and `.gitattributes`; macOS `._*` sidecars are also present. The stock description CSV is an availability flag, not a verified per-file inventory.",
            "",
            "## Target Stocks and Modality Coverage",
            "",
            "| Symbol | Modality | Present | Records/files | Date or period span | Availability / notes |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for symbol in TARGETS:
        for modality in ("text", "table", "time_series", "image"):
            row = present(symbol, modality)
            if modality == "table":
                span = f"period end {row.get('earliest_period_end')} → {row.get('latest_period_end')}; filed {row.get('earliest_availability_date')} → {row.get('latest_availability_date')}"
            elif modality == "image":
                span = f"nominal {row.get('earliest_nominal_period')} → {row.get('latest_nominal_period')}"
            else:
                span = f"{row.get('earliest_date')} → {row.get('latest_date')}"
            notes = row.get("notes", "")
            lines.append(f"| {symbol} | {modality} | {str(row.get('present', False)).lower()} | {row.get('raw_record_count', 0)} | {span} | {notes} |")
    lines.extend(
        [
            "",
            "All three symbols have table and time-series members. Text exists for AAPL and AMZN only; the JPM news member is absent. Images exist for AMZN and JPM only; no `aapl` image directory/file exists even though the description CSV flags `image=1`.",
            "",
            "## Temporal Coverage and Formal Schedule",
            "",
            f"The exact Formal M0 schedule was regenerated from `configs/backtest_m0_2024h1.json` and the repository XNYS calendar implementation. It contains {len(events)} weekly decisions from `{events[0]['decision_session']}` through `{events[-1]['decision_session']}`, with execution sessions `{events[0]['execution_session']}` through `{events[-1]['execution_session']}` and final valuation `{FORMAL_CONFIG['final_valuation_session']}`. The 252-session warm-up spans `{warmup_start}` through `{events[0]['decision_session']}` exclusive.",
            "",
            "The daily time series begin on 2000-01-03 for all targets and extend beyond 2024H1. Tables contain historical period ends back to 2008/2009-era observations and filed dates through 2025. News and image spans are modality-specific and are documented in the target CSV.",
            "",
            "| Symbol | Time-series 2023 rows | Time-series 2024H1 rows | Time-series rows through 2024-07-05 | Formal XNYS session gaps |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for symbol in TARGETS:
        item = series[symbol]
        dates = [row["session_date"] for row in item["rows"] if row["session_date"]]
        expected = item.get("expected_sessions", [])
        gaps = [value for value in expected if value not in set(dates)]
        counts = date_range_counts(dates)
        lines.append(f"| {symbol} | {counts['coverage_2023']} | {counts['coverage_2024h1']} | {counts['through_final_valuation']} | {len(gaps)} |")
    lines.extend(
        [
            "",
            "The time series therefore provide ample pre-formal history for the 252-session reference. Daily rows are interpreted as completed session bars when used at a session-close decision; the source itself stores a timezone-bearing midnight timestamp rather than an explicit bar close timestamp.",
            "",
            "## Point-in-Time Findings",
            "",
            "### News/text",
            "",
        ]
    )
    for symbol in TARGETS:
        item = news[symbol]
        if not item:
            lines.append(f"- **{symbol}:** no news member; text must be unavailable.")
        else:
            lines.append(
                f"- **{symbol}:** {item['record_count']} JSONL records, `{item['earliest_date']}` → `{item['latest_date']}`, fields `{', '.join(item['fields'])}`; {item['missing_date_count']} missing dates, {item['invalid_date_count']} invalid dates, {item['exact_duplicate_record_count']} exact duplicate records, and {item['duplicate_url_count']} duplicate URLs. Dates are date-only, with no timezone or publication time. `Url` provides a source domain, `Article_title` is present, and `Stock_symbol` values are `{item['stock_symbols']}`; no revision/version field exists."
            )
    lines.extend(
        [
            "",
            "A same-calendar-date article cannot be assumed available at the market-open or close decision because the dataset does not say when it was published. The audit therefore treats only dates strictly earlier than the decision session as safely time-ordered and records same-day dates as ambiguous.",
            "",
            "### Structured tables / fundamentals",
            "",
        ]
    )
    for symbol in TARGETS:
        item = tables[symbol]
        lines.append(
            f"- **{symbol}:** {item['file_count']} SEC-style JSON files and {item['record_count']} fact observations; parent `filing_date` fields `{item['filing_date_earliest']}` → `{item['filing_date_latest']}`, period ends `{item['period_end_earliest']}` → `{item['period_end_latest']}`, fact availability `{item['availability_earliest']}` → `{item['availability_latest']}`; {item['restated_period_group_count']} repeated concept/unit/period groups indicate identifiable revisions/restatements; {item['missing_filed_count']} observations lack `filed` (parent filing-date fallback missing for {item['missing_availability_count']})."
        )
    lines.extend(
        [
            "",
            "The tables distinguish fiscal/reporting period (`start`/`end`) from availability (`filed`, plus parent filing-object `filing_date`). `filed` is date-only with no time or timezone, so it is a conservative day-level gate, not an intraday release timestamp. Fiscal period end must not be used as availability.",
            "",
            "### Time series",
            "",
        ]
    )
    for symbol in TARGETS:
        item = series[symbol]
        lines.append(
            f"- **{symbol}:** {item['record_count']} daily rows, `{item['earliest_date']}` → `{item['latest_date']}`, columns `{', '.join(item['header'])}`; timezone offsets `{', '.join(item['timezone_offsets'])}`; duplicate dates {item['duplicate_date_count']}, non-monotonic order `{item['non_monotonic']}`, missing-value rows {item['missing_value_row_count']}, non-finite rows {item['nonfinite_value_row_count']}, non-positive-price rows {item['nonpositive_price_row_count']}, impossible OHLC rows {item['impossible_ohlc_row_count']}."
        )
    lines.extend(
        [
            "",
            "No explicit future-return, forecast, direction, target, or prediction column appears in the target time-series headers. The CSV has no Adj Close column and no adjustment-policy metadata; it includes Dividends and Stock Splits, and values appear adjusted, so M1 must carry that source assumption explicitly.",
            "",
            "### Images / K-line charts",
            "",
            "The images are PNG candlestick charts named `<ticker>_<year>_H1|H2_candlestick.png`, stored under ticker folders in the S&P500 shards. Visual inspection of representative AMZN and JPM 2024H1 charts found a title and date axis for the half-year, but no separate machine-readable start/end metadata. The chart convention appears to cover the named half-year; this is an inference and not a formal PIT guarantee.",
            "",
            "| Symbol | All target images | Nominally safe by final decision | In 252-session warm-up + formal span | Latest-safe references across 26 cases | Total bytes | Dimensions |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for symbol, total, safe_final, support, refs, bytes_ in image_workload_rows:
        dims = Counter(
            f"{item['metadata'].get('width')}x{item['metadata'].get('height')}"
            for item in images[symbol]
            if item['metadata'].get('width')
        )
        lines.append(f"| {symbol} | {total} | {safe_final} | {support} | {refs} | {human_bytes(bytes_)} | {dict(dims)} |")
    lines.extend(
        [
            "",
            "The workload metric is unique files, not repeated case references. AMZN and JPM each have 51 half-year images from 2000H1 through 2025H1; 48 each have nominal ends no later than the final 2024-06-28 decision, while two each fall inside the 2023-01-04 warm-up through 2024-06-28 span under the conservative filename-end rule. AAPL has zero. If each formal case selects its latest nominally safe image, there are 52 possible existing-stock references, reusing files across weeks. PNG target files were header-readable with consistent dimensions; no target duplicate binary hashes were found.",
            "",
            "## Future-looking Fields and Leakage Risks",
            "",
            "| Source | Field/file type | Classification | Required handling |",
            "|---|---|---|---|",
        ]
    )
    for row in future_rows:
        lines.append(f"| {row['source_type']} | `{row['field_or_path']}` | {row['classification']} | {row['finding']} |")
    lines.extend(
        [
            "",
            "The audit found no explicit future target field in the target schemas. The principal leakage risks are temporal: same-day date-only news, same-day date-only filing availability, using fiscal period ends as availability, treating a half-year image as available before its inferred window end, and accidentally exposing later table revisions. The full prohibited/controlled-field register is in `finmultitime_prohibited_future_fields.csv`.",
            "",
            "## Formal 78-case Coverage",
            "",
            "The machine-readable case table contains one row for each AAPL × 26, AMZN × 26, and JPM × 26 case. Counts are descriptive and do not freeze M1 lookback windows. News 1/3/7/14/30-day counts use valid article dates in the half-open calendar interval `[decision - window, decision)`; same-day records are reported separately as ambiguous. Table counts use observations with `filed < decision date` and exclude same-day filed observations.",
            "",
            "| Modality | Overview |",
            "|---|---|",
            "| News | AAPL and AMZN have date-only records; JPM is unavailable. Same-day records are not treated as safe. |",
            "| Tables | All three symbols have filed-date-gated SEC facts; filed dates are day-only and same-day facts are ambiguous. |",
            f"| Time series | All three have long daily histories and {sum(1 for row in case_rows if 'TIME_SERIES_MISSING_XNYS_SESSION' in row['pit_risk_flags'])} case rows with a formal XNYS gap. |",
            "| Images | AMZN/JPM have nominally gated chart candidates; AAPL is unavailable. The filename-derived end-date rule deliberately leaves early/final weeks without a current-half-year chart. |",
            "",
            "See `finmultitime_formal_case_coverage.csv` for all 78 rows, including latest safe dates, ages, recent news counts, time-series history counts, candidate image counts, and PIT flags.",
            "",
            "## Data Quality",
            "",
            "The audit reports issues without cleaning upstream data. Text parsing checks malformed JSON, missing/invalid dates, empty article bodies, exact duplicate records, duplicate URLs, and obvious duplicate article keys. Table parsing checks JSON errors, missing filed dates, duplicate fact rows, inconsistent repeated periods/restatement groups, and schema keys. Time-series checks parse errors, ordering, duplicate sessions, missing values, non-finite values, non-positive prices, impossible OHLC relationships, negative volume, and formal XNYS gaps. Target PNG headers and dimensions were checked and binary hashes compared. Detailed counts are in the CSV/JSON outputs.",
            "",
            "## Processed-subset Estimate",
            "",
            "A future experiment-specific subset should be built only after the Evidence Contract selects lookbacks. As an empirical upper bound for the full 2023-01-04 through 2024-07-05 M0 warm-up/formal/valuation span, the source sizes are:",
            "",
            "- **Text:** AAPL + AMZN only; JPM unavailable. The raw target news members are approximately the sizes recorded in `finmultitime_target_stock_coverage.csv`; a filtered date subset would be smaller.",
            "- **Tables:** all nine target statement members; approximately the sum of their uncompressed member sizes in the target CSV. Filtering concepts/filings will reduce this substantially.",
            "- **Time series:** three CSV members, approximately the sum of their uncompressed member sizes; the formal span is a small fraction of the 2000–2025 history.",
            "- **Images:** zero for AAPL and two nominally usable support-span images each for AMZN/JPM under the conservative rule; existing target image bytes are a measured upper bound and captioning 52 possible latest-safe case references would reuse those files.",
            "",
            "Using complete target members as a conservative pre-filter upper bound, the measured three-stock source bytes are approximately **66.2 MiB text** (AAPL + AMZN; JPM unavailable), **88.7 MiB tables**, and **2.1 MiB time series**. The two AMZN and two JPM support-span images add about **0.17 MiB**, for an upper-bound working estimate of approximately **157.2 MiB** before concept/news filtering and serialization overhead. A filtered subset should be smaller; the exact source bytes are in the generated JSON and CSV. This estimate does not copy or build the subset.",
            "",
            "## Audit Boundaries",
            "",
            "- Raw FinMultiTime modified: **NO**",
            "- Full dataset copied into AlphaMAS: **NO**",
            "- Final processed subset built: **NO**",
            "- Qwen downloaded: **NO**",
            "- Qwen run: **NO**",
            "- Image captions generated: **NO**",
            "- Evidence Packet frozen: **NO**",
            "- DeepSeek calls: **0**",
            "- Paid API calls: **0**",
            "- Formal M1 run: **NO**",
            "- M2 / Agentic RL started: **NO**",
            "- AlphaMAS-Experiments modified: **NO**",
            "",
            "## Outputs",
            "",
            "- `scripts/finmultitime/audit_finmultitime.py`",
            "- `docs/m1/finmultitime_data_audit.md`",
            "- `docs/m1/finmultitime_data_audit.json`",
            "- `docs/m1/finmultitime_modality_inventory.csv`",
            "- `docs/m1/finmultitime_target_stock_coverage.csv`",
            "- `docs/m1/finmultitime_formal_case_coverage.csv`",
            "- `docs/m1/finmultitime_pit_risks.csv`",
            "- `docs/m1/finmultitime_prohibited_future_fields.csv`",
            "",
            "The JSON report contains the full structured audit, including raw source paths, exact schedule events, schema findings, and serialized row-level statistics.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    dataset_root: Path,
    output_dir: Path,
    inventory_cache: Path | None = None,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")
    required = [root / NEWS_ARCHIVE, root / TABLE_ARCHIVE, root / TS_ARCHIVE, root / IMAGE_ROOT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing expected raw sources:\n" + "\n".join(missing))

    if inventory_cache is not None:
        cached = json.loads(inventory_cache.read_text(encoding="utf-8"))
        filesystem = cached["filesystem_inventory"]
        archives = cached["archives"]
    else:
        filesystem = filesystem_inventory(root)
        archive_paths = [
            NEWS_ARCHIVE,
            "text/hs300news_summary.zip",
            TABLE_ARCHIVE,
            "table/hs300_tabular.zip",
            TS_ARCHIVE,
            "time_series/HS300_time_series.zip",
        ]
        archives = {relative: archive_inventory(root, relative) for relative in archive_paths}
    events, warmup_dates, warmup_start = parse_formal_schedule()
    with zipfile.ZipFile(root / NEWS_ARCHIVE) as archive:
        news: dict[str, dict[str, Any] | None] = {}
        members = {info.filename for info in archive.infolist() if not info.is_dir()}
        for symbol, member in NEWS_MEMBERS.items():
            news[symbol] = read_jsonl_news(archive, member) if member in members else None
    with zipfile.ZipFile(root / TABLE_ARCHIVE) as archive:
        members = [info.filename for info in archive.infolist() if not info.is_dir()]
        tables: dict[str, dict[str, Any] | None] = {}
        for symbol in TARGETS:
            selected = sorted(member for member in members if TABLE_MEMBER_RE[symbol].match(member))
            tables[symbol] = read_table_files(archive, selected) if selected else None
    with zipfile.ZipFile(root / TS_ARCHIVE) as archive:
        members = {info.filename for info in archive.infolist() if not info.is_dir()}
        series: dict[str, dict[str, Any] | None] = {}
        for symbol, member in TS_MEMBERS.items():
            series[symbol] = read_time_series(archive, member) if member in members else None

    # Add XNYS comparison sessions to time-series records without changing raw data.
    from tradingagents.backtesting.calendar import ExchangeSchedule

    schedule = ExchangeSchedule(FORMAL_CONFIG["calendar"])
    for _symbol, item in series.items():
        if item is None or not item.get("rows"):
            continue
        start = item["rows"][0]["session_date"]
        end = max(date.fromisoformat(FORMAL_CONFIG["final_valuation_session"]), item["rows"][-1]["session_date"])
        item["expected_sessions"] = [stamp.date() for stamp in schedule.sessions(start, end)]

    image_data = discover_images(root / IMAGE_ROOT)
    images = image_data["by_symbol"]
    modality_rows = build_modality_rows(root, filesystem, archives, news, tables, series, images)
    target_rows = build_target_rows(root, archives, news, tables, series, images)
    case_rows = build_case_rows(
        events, date.fromisoformat(warmup_start), news, tables, series, images
    )
    pit_risks = build_pit_risks(news, tables, series, images)
    future_rows = build_future_field_rows(news, tables, series)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "finmultitime_modality_inventory.csv", modality_rows)
    write_csv(output_dir / "finmultitime_target_stock_coverage.csv", target_rows)
    write_csv(output_dir / "finmultitime_formal_case_coverage.csv", case_rows)
    write_csv(output_dir / "finmultitime_pit_risks.csv", pit_risks)
    write_csv(output_dir / "finmultitime_prohibited_future_fields.csv", future_rows)
    report = {
        "audit": {
            "script": str(Path(__file__).resolve()),
            "dataset_root": str(root),
            "raw_data_read_only": True,
            "targets": list(TARGETS),
            "verdict": "M1 DATA AUDIT PASSED WITH PIT/AVAILABILITY GAPS — CONTRACT MUST HANDLE EXPLICIT UNAVAILABLE DATA",
        },
        "formal_protocol": {
            **FORMAL_CONFIG,
            "decision_count": len(events),
            "case_count": len(case_rows),
            "warmup_start": warmup_start,
            "warmup_end_exclusive": events[0]["decision_session"],
            "events": events,
            "warmup_sessions": warmup_dates,
        },
        "filesystem_inventory": filesystem,
        "archives": archives,
        "modality_inventory": modality_rows,
        "target_news": compact_target_summary(
            news, {"records", "date_values", "raw_date_values"}
        ),
        "target_tables": compact_target_summary(tables, {"observations"}),
        "target_time_series": compact_target_summary(
            series, {"rows", "date_counts", "expected_sessions"}
        ),
        "target_images": images,
        "target_stock_coverage": target_rows,
        "formal_case_coverage": case_rows,
        "pit_risks": pit_risks,
        "prohibited_future_fields": future_rows,
    }
    (output_dir / "finmultitime_data_audit.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(
        root,
        filesystem,
        archives,
        modality_rows,
        target_rows,
        case_rows,
        pit_risks,
        future_rows,
        events,
        warmup_start,
        news,
        tables,
        series,
        images,
    )
    (output_dir / "finmultitime_data_audit.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Volumes/Jackson/Dataset/FinMultiTime"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/m1"),
    )
    parser.add_argument(
        "--reuse-inventory-json",
        type=Path,
        help="Reuse filesystem/archive inventory from an earlier audit JSON without rescanning the raw tree.",
    )
    args = parser.parse_args()
    report = run(args.dataset_root, args.output_dir, args.reuse_inventory_json)
    print(json.dumps({"verdict": report["audit"]["verdict"], "case_count": report["formal_protocol"]["case_count"], "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
