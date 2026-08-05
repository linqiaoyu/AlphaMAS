"""Central data-source capability policy for point-in-time runs."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from .run_context import current_run_context


class DataCapability(str, Enum):
    POINT_IN_TIME = "POINT_IN_TIME"
    APPROXIMATE = "APPROXIMATE"
    LIVE_ONLY = "LIVE_ONLY"


# The key is (configured vendor, tool method).  Unknown sources are LIVE_ONLY:
# uncertainty is never permission to use a source during a historical run.
CAPABILITY_MATRIX: dict[tuple[str, str], DataCapability] = {
    ("yfinance", "get_stock_data"): DataCapability.POINT_IN_TIME,
    ("yfinance", "get_indicators"): DataCapability.POINT_IN_TIME,
    ("yfinance", "get_news"): DataCapability.APPROXIMATE,
    ("yfinance", "get_global_news"): DataCapability.APPROXIMATE,
    ("yfinance", "get_fundamentals"): DataCapability.LIVE_ONLY,
    ("yfinance", "get_balance_sheet"): DataCapability.LIVE_ONLY,
    ("yfinance", "get_cashflow"): DataCapability.LIVE_ONLY,
    ("yfinance", "get_income_statement"): DataCapability.LIVE_ONLY,
    ("yfinance", "get_insider_transactions"): DataCapability.LIVE_ONLY,
    ("alpha_vantage", "get_stock_data"): DataCapability.POINT_IN_TIME,
    ("alpha_vantage", "get_indicators"): DataCapability.POINT_IN_TIME,
    ("alpha_vantage", "get_news"): DataCapability.POINT_IN_TIME,
    ("alpha_vantage", "get_global_news"): DataCapability.POINT_IN_TIME,
    ("alpha_vantage", "get_fundamentals"): DataCapability.LIVE_ONLY,
    ("alpha_vantage", "get_balance_sheet"): DataCapability.POINT_IN_TIME,
    ("alpha_vantage", "get_cashflow"): DataCapability.POINT_IN_TIME,
    ("alpha_vantage", "get_income_statement"): DataCapability.POINT_IN_TIME,
    ("alpha_vantage", "get_insider_transactions"): DataCapability.POINT_IN_TIME,
    ("fred", "get_macro_indicators"): DataCapability.LIVE_ONLY,
    ("polymarket", "get_prediction_markets"): DataCapability.LIVE_ONLY,
}

APPROXIMATION_RULES = {
    ("yfinance", "get_news"): (
        "Yahoo Finance news archive completeness is not guaranteed; only articles "
        "with a parsed publication timestamp inside the requested window are used."
    ),
    ("yfinance", "get_global_news"): (
        "Yahoo Finance Search is not a complete historical archive; only articles "
        "with a parsed publication timestamp inside the requested window are used."
    ),
}

_DATE_POSITIONS: dict[str, tuple[int, ...]] = {
    "get_stock_data": (1, 2),
    "get_indicators": (2,),
    "get_fundamentals": (1,),
    "get_balance_sheet": (2,),
    "get_cashflow": (2,),
    "get_income_statement": (2,),
    "get_news": (1, 2),
    "get_global_news": (0,),
    "get_macro_indicators": (1,),
}


def capability_for(vendor: str, method: str) -> DataCapability:
    return CAPABILITY_MATRIX.get((vendor, method), DataCapability.LIVE_ONLY)


def source_name(vendor: str, method: str) -> str:
    return f"{vendor}.{method}"


def _parse_request_time(value: Any) -> datetime | None:
    tz = current_run_context().as_of.tzinfo
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=tz)
    if isinstance(value, date):
        return datetime.combine(value, datetime.max.time(), tzinfo=tz)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=current_run_context().as_of.tzinfo)
    return parsed.astimezone(current_run_context().as_of.tzinfo)


def requested_dates(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """Extract a best-effort requested window for policy and audit metadata."""
    positions = _DATE_POSITIONS.get(method, ())
    values: list[Any] = []
    for position in positions:
        if position < len(args):
            values.append(args[position])
    keyword_names = {
        "get_stock_data": ("start_date", "end_date"),
        "get_news": ("start_date", "end_date"),
        "get_global_news": ("curr_date",),
        "get_macro_indicators": ("curr_date",),
    }.get(method, ())
    for name in keyword_names:
        if name in kwargs:
            values.append(kwargs[name])
    if not values:
        return None, None
    return values[0], values[-1]


def request_exceeds_as_of(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    context = current_run_context()
    if context.mode != "historical":
        return None
    start, end = requested_dates(method, args, kwargs)
    for label, value in (("start", start), ("end", end)):
        parsed = _parse_request_time(value)
        if parsed is not None and parsed > context.as_of:
            return (
                f"requested {label}={value} is later than historical_as_of="
                f"{context.as_of.isoformat()}"
            )
    return None


def historical_block_reason(vendor: str, method: str) -> str | None:
    context = current_run_context()
    if context.mode != "historical":
        return None
    capability = capability_for(vendor, method)
    if capability == DataCapability.POINT_IN_TIME:
        return None
    if capability == DataCapability.APPROXIMATE:
        return APPROXIMATION_RULES.get((vendor, method))
    return f"{vendor} is a LIVE_ONLY source and cannot prove data availability at historical_as_of."
