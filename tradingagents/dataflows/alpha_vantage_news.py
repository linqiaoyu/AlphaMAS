import json
from datetime import datetime

from .alpha_vantage_common import _make_api_request, format_datetime_for_api

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

    return _make_api_request("NEWS_SENTIMENT", params)

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
    from datetime import timedelta

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

    return _make_api_request("NEWS_SENTIMENT", params)


def _parse_date(value: str):
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _filter_insider_payload_by_date(payload: dict, curr_date: str) -> dict:
    cutoff = _parse_date(curr_date)
    if cutoff is None or not isinstance(payload, dict):
        return payload

    entries = payload.get("data")
    if not isinstance(entries, list):
        return payload

    date_keys = [
        "transaction_date",
        "filing_date",
        "date",
    ]

    filtered_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_date = None
        for key in date_keys:
            if key in entry:
                entry_date = _parse_date(entry.get(key))
                if entry_date:
                    break

        # Strict guard: if timestamp is missing/unparseable, drop the row.
        if entry_date and entry_date <= cutoff:
            filtered_entries.append(entry)

    payload["data"] = filtered_entries
    return payload


def get_insider_transactions(symbol: str, curr_date: str = None) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    raw_payload = _make_api_request("INSIDER_TRANSACTIONS", params)

    if not curr_date:
        return raw_payload

    try:
        parsed_payload = json.loads(raw_payload)
    except (TypeError, json.JSONDecodeError):
        return raw_payload

    filtered_payload = _filter_insider_payload_by_date(parsed_payload, curr_date)
    return json.dumps(filtered_payload)
