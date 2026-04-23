from datetime import date, datetime
from typing import Any, Annotated

from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    }
}

# Canonical argument order for date-guarded methods.
_METHOD_ARG_NAMES = {
    "get_stock_data": ["symbol", "start_date", "end_date"],
    "get_news": ["ticker", "start_date", "end_date"],
    "get_global_news": ["curr_date", "look_back_days", "limit"],
    "get_indicators": ["symbol", "indicator", "curr_date", "look_back_days"],
    "get_fundamentals": ["ticker", "curr_date"],
    "get_balance_sheet": ["ticker", "freq", "curr_date"],
    "get_cashflow": ["ticker", "freq", "curr_date"],
    "get_income_statement": ["ticker", "freq", "curr_date"],
    "get_insider_transactions": ["ticker", "curr_date"],
}

VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "yfinance": get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "yfinance": get_yfinance_income_statement,
    },
    # news_data
    "get_news": {
        "yfinance": get_news_yfinance,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
    },
    "get_insider_transactions": {
        "yfinance": get_yfinance_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")


def _parse_yyyy_mm_dd(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _set_param(
    method: str,
    args: list,
    kwargs: dict,
    param: str,
    value: Any,
) -> None:
    if param in kwargs:
        kwargs[param] = value
        return
    names = _METHOD_ARG_NAMES.get(method, [])
    if param in names:
        idx = names.index(param)
        if idx < len(args):
            args[idx] = value
            return
    kwargs[param] = value


def _get_param(method: str, args: list, kwargs: dict, param: str) -> Any:
    if param in kwargs:
        return kwargs[param]
    names = _METHOD_ARG_NAMES.get(method, [])
    if param in names:
        idx = names.index(param)
        if idx < len(args):
            return args[idx]
    return None


def _enforce_backtest_date_guard(method: str, args: tuple, kwargs: dict) -> tuple[list, dict]:
    """Clamp all tool dates to configured backtest as-of date."""
    config = get_config()
    as_of = _parse_yyyy_mm_dd(config.get("backtest_as_of_date"))
    if as_of is None:
        return list(args), dict(kwargs)

    as_of_str = as_of.strftime("%Y-%m-%d")
    guarded_args = list(args)
    guarded_kwargs = dict(kwargs)

    def clamp(value: Any) -> Any:
        value_date = _parse_yyyy_mm_dd(value)
        if value_date is None:
            return value
        return min(value_date, as_of).strftime("%Y-%m-%d")

    if method in {"get_stock_data", "get_news"}:
        start_date = clamp(_get_param(method, guarded_args, guarded_kwargs, "start_date"))
        end_date = clamp(_get_param(method, guarded_args, guarded_kwargs, "end_date"))
        _set_param(method, guarded_args, guarded_kwargs, "start_date", start_date)
        _set_param(method, guarded_args, guarded_kwargs, "end_date", end_date)

        start_dt = _parse_yyyy_mm_dd(start_date)
        end_dt = _parse_yyyy_mm_dd(end_date)
        if start_dt and end_dt and start_dt > end_dt:
            _set_param(method, guarded_args, guarded_kwargs, "start_date", end_date)

    if method in {"get_global_news", "get_indicators"}:
        curr_date = _get_param(method, guarded_args, guarded_kwargs, "curr_date")
        if curr_date is None:
            curr_date = as_of_str
        _set_param(method, guarded_args, guarded_kwargs, "curr_date", clamp(curr_date))

    if method in {
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_insider_transactions",
    }:
        curr_date = _get_param(method, guarded_args, guarded_kwargs, "curr_date")
        if curr_date is None:
            curr_date = as_of_str
        _set_param(method, guarded_args, guarded_kwargs, "curr_date", clamp(curr_date))

    return guarded_args, guarded_kwargs


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        guarded_args, guarded_kwargs = _enforce_backtest_date_guard(method, args, kwargs)
        return impl_func(*guarded_args, **guarded_kwargs)

    raise RuntimeError(f"No available vendor for '{method}'")
