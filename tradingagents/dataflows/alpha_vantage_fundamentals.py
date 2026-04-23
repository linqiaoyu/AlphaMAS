import json

from .alpha_vantage_common import _make_api_request
from .config import get_config


def _effective_cutoff(curr_date: str) -> str:
    lag_days = int(get_config().get("fundamentals_release_lag_days", 45) or 0)
    if lag_days <= 0:
        return curr_date
    from datetime import datetime, timedelta
    cutoff_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lag_days)
    return cutoff_dt.strftime("%Y-%m-%d")


def _filter_reports_by_date(result, curr_date: str):
    """Filter annualReports/quarterlyReports to exclude entries after curr_date.

    Prevents look-ahead bias by removing fiscal periods that end after
    the simulation's current date.
    """
    if not curr_date:
        return result

    parsed = result
    as_json_text = False
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            as_json_text = True
        except json.JSONDecodeError:
            return result

    if not isinstance(parsed, dict):
        return result

    cutoff = _effective_cutoff(curr_date)

    for key in ("annualReports", "quarterlyReports"):
        if key in parsed:
            parsed[key] = [
                r for r in parsed[key]
                if r.get("fiscalDateEnding", "") <= cutoff
            ]
    return json.dumps(parsed) if as_json_text else parsed


def _sanitize_overview_for_backtest(result, curr_date: str):
    """Keep only profile/static fields and date-safe fields for as-of analysis."""
    parsed = result
    as_json_text = False
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            as_json_text = True
        except json.JSONDecodeError:
            return result

    if not isinstance(parsed, dict):
        return result

    allowed_fields = {
        "Symbol",
        "AssetType",
        "Name",
        "Description",
        "CIK",
        "Exchange",
        "Currency",
        "Country",
        "Sector",
        "Industry",
        "Address",
        "OfficialSite",
        "FiscalYearEnd",
    }

    sanitized = {k: v for k, v in parsed.items() if k in allowed_fields}

    cutoff = _effective_cutoff(curr_date) if curr_date else None
    latest_quarter = parsed.get("LatestQuarter")
    if latest_quarter and (not cutoff or latest_quarter <= cutoff):
        sanitized["LatestQuarter"] = latest_quarter

    return json.dumps(sanitized) if as_json_text else sanitized


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd

    Returns:
        str: Sanitized company overview safe for as-of backtesting
    """
    params = {
        "symbol": ticker,
    }

    result = _make_api_request("OVERVIEW", params)
    return _sanitize_overview_for_backtest(result, curr_date)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)
