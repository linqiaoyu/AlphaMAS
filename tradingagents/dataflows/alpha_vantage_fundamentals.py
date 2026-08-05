import json
from datetime import datetime

from tradingagents.runtime.run_context import audit_source, current_run_context

from .alpha_vantage_common import _make_api_request


def _filter_reports_by_date(result, curr_date: str):
    """Drop annual/quarterly reports dated after curr_date to prevent look-ahead.

    ``_make_api_request`` returns the fundamentals payload as a JSON string, so
    parse, filter, and re-serialize. A non-JSON body or an unset ``curr_date`` is
    returned unchanged.
    """
    if not curr_date:
        return result
    try:
        payload = json.loads(result) if isinstance(result, str) else result
    except json.JSONDecodeError:
        if current_run_context().mode == "historical":
            return "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: financial response had no verifiable publication timestamp."
        return result
    if not isinstance(payload, dict):
        if current_run_context().mode == "historical":
            return "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: financial response was not structured publication-dated data."
        return result
    context = current_run_context()
    if context.mode != "historical":
        for key in ("annualReports", "quarterlyReports"):
            if isinstance(payload.get(key), list):
                payload[key] = [
                    r for r in payload[key]
                    if r.get("fiscalDateEnding", "") <= curr_date
                ]
        return json.dumps(payload)

    # A fiscal period ending before the cutoff is not enough: the report may
    # have been filed after it. Alpha Vantage sometimes exposes one of these
    # publication-time fields; records without one are rejected.
    publication_keys = (
        "acceptedDate", "acceptedTime", "filingDate", "reportedDate", "publicationDate"
    )

    def parsed_publication(record: dict) -> datetime | None:
        for key in publication_keys:
            value = record.get(key)
            if not value:
                continue
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed = datetime.strptime(str(value)[:10], "%Y-%m-%d")
                except ValueError:
                    continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=context.as_of.tzinfo)
            return parsed.astimezone(context.as_of.tzinfo)
        return None

    saw_reports = False
    kept_reports = 0
    for key in ("annualReports", "quarterlyReports"):
        reports = payload.get(key)
        if not isinstance(reports, list):
            continue
        saw_reports = saw_reports or bool(reports)
        filtered = []
        for report in reports:
            fiscal = report.get("fiscalDateEnding", "")
            published = parsed_publication(report)
            if (
                fiscal <= curr_date
                and published is not None
                and published <= context.as_of
            ):
                filtered.append(report)
        payload[key] = filtered
        kept_reports += len(filtered)

    if not saw_reports or kept_reports == 0:
        reason = (
            "Alpha Vantage reports had no usable publication timestamp at or before "
            f"historical_as_of {context.as_of.isoformat()}."
        )
        audit_source(
            source_name="alpha_vantage.financial_statements",
            capability="POINT_IN_TIME",
            status="unavailable",
            requested_end=curr_date,
            reason=reason,
        )
        return "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: " + reason
    audit_source(
        source_name="alpha_vantage.financial_statements",
        capability="POINT_IN_TIME",
        status="used" if kept_reports else "unavailable",
        requested_end=curr_date,
        latest_available_time=max(
            (
                r.get("acceptedDate") or r.get("filingDate") or r.get("reportedDate")
                for key in ("annualReports", "quarterlyReports")
                for r in payload.get(key, [])
            ),
            default=None,
        ),
        reason="Filtered by fiscal period and publication timestamp.",
    )
    return json.dumps(payload)


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    if current_run_context().mode == "historical":
        reason = "Alpha Vantage OVERVIEW is a current snapshot without a point-in-time publication timestamp."
        audit_source(
            source_name="alpha_vantage.overview",
            capability="LIVE_ONLY",
            status="blocked",
            requested_end=curr_date,
            reason=reason,
        )
        return "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: " + reason

    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


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
