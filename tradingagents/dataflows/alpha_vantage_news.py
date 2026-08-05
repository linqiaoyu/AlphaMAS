import json
from datetime import datetime, time, timezone

from tradingagents.runtime.run_context import audit_source, current_run_context

from .alpha_vantage_common import _make_api_request, format_datetime_for_api


def _filter_historical_news(result, start_date: str, end_date: str):
    """Filter Alpha Vantage JSON news by publication timestamp in strict mode."""
    context = current_run_context()
    if context.mode != "historical":
        return result
    try:
        payload = json.loads(result) if isinstance(result, str) else result
    except json.JSONDecodeError:
        reason = "Alpha Vantage news response was not structured and had no verifiable publication timestamps."
        audit_source(
            source_name="alpha_vantage.news",
            capability="POINT_IN_TIME",
            status="unavailable",
            requested_start=start_date,
            requested_end=end_date,
            reason=reason,
        )
        return "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: " + reason
    if not isinstance(payload, dict) or not isinstance(payload.get("feed"), list):
        reason = "Alpha Vantage news response had no timestamped feed that could be verified historically."
        audit_source(
            source_name="alpha_vantage.news",
            capability="POINT_IN_TIME",
            status="unavailable",
            requested_start=start_date,
            requested_end=end_date,
            reason=reason,
        )
        return "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: " + reason

    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    requested_end = datetime.combine(
        datetime.strptime(end_date, "%Y-%m-%d").date(), time.max, tzinfo=timezone.utc
    )
    end = min(requested_end, context.as_of)
    kept = []
    for article in payload["feed"]:
        raw_time = article.get("time_published") or article.get("published")
        if not raw_time:
            continue
        try:
            if len(str(raw_time)) >= 15 and "T" in str(raw_time):
                published = datetime.strptime(str(raw_time)[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            else:
                published = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if published.tzinfo is None:
                    published = published.replace(tzinfo=timezone.utc)
                published = published.astimezone(timezone.utc)
        except ValueError:
            continue
        if start <= published <= end:
            kept.append(article)
    payload["feed"] = kept
    reason = "Filtered by timestamp_published/time_published and requested window."
    audit_source(
        source_name="alpha_vantage.news",
        capability="POINT_IN_TIME",
        status="used" if kept else "unavailable",
        requested_start=start_date,
        requested_end=end_date,
        latest_event_time=max((a.get("time_published") for a in kept), default=None),
        latest_available_time=max((a.get("time_published") for a in kept), default=None),
        reason=reason,
    )
    return json.dumps(payload)

def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    result = _make_api_request("NEWS_SENTIMENT", params)
    return _filter_historical_news(result, start_date, end_date)

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from datetime import datetime, timedelta

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    result = _make_api_request("NEWS_SENTIMENT", params)
    return _filter_historical_news(result, start_date, curr_date)


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    result = _make_api_request("INSIDER_TRANSACTIONS", {"symbol": symbol})
    context = current_run_context()
    if context.mode != "historical":
        return result

    try:
        payload = json.loads(result) if isinstance(result, str) else result
    except json.JSONDecodeError:
        payload = None

    if not isinstance(payload, dict):
        reason = "Alpha Vantage insider response was not structured data with public filing timestamps."
        audit_source(
            source_name="alpha_vantage.insider_transactions",
            capability="POINT_IN_TIME",
            status="unavailable",
            requested_end=context.as_of,
            reason=reason,
        )
        return "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: " + reason

    list_key = next(
        (key for key in ("data", "insiderTransactions", "transactions") if isinstance(payload.get(key), list)),
        None,
    )
    records = payload.get(list_key, []) if list_key else []
    publication_keys = ("acceptedDate", "filingDate", "reportedDate", "publicationDate")
    filtered = []
    for record in records:
        published = next((record.get(key) for key in publication_keys if record.get(key)), None)
        if not published:
            continue
        try:
            parsed = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(str(published)[:10], "%Y-%m-%d")
            except ValueError:
                continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=context.as_of.tzinfo)
        if parsed.astimezone(context.as_of.tzinfo) <= context.as_of:
            filtered.append(record)

    if list_key:
        payload[list_key] = filtered
    if records and not filtered:
        reason = "No insider transaction had a verifiable public filing timestamp at or before historical_as_of."
        status = "unavailable"
        output = "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: " + reason
    elif not records:
        reason = "Alpha Vantage insider response contained no timestamped transactions."
        status = "unavailable"
        output = "DATA_UNAVAILABLE_IN_HISTORICAL_MODE: " + reason
    else:
        reason = "Insider transactions filtered by public filing timestamp."
        status = "used"
        output = json.dumps(payload)
    audit_source(
        source_name="alpha_vantage.insider_transactions",
        capability="POINT_IN_TIME",
        status=status,
        requested_end=context.as_of,
        reason=reason,
    )
    return output
