import json
import logging
import os
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from tradingagents.runtime.run_context import audit_source, current_run_context

from .errors import VendorNotConfiguredError, VendorRateLimitError

API_BASE_URL = "https://www.alphavantage.co/query"

# Network timeout (seconds) so a stalled Alpha Vantage request can't hang the
# CLI/agents indefinitely (#990).
REQUEST_TIMEOUT = 30
HISTORICAL_UNAVAILABLE = "DATA_UNAVAILABLE_IN_HISTORICAL_MODE"
_DATE_COLUMN_NAMES = {"date", "datetime", "time", "timestamp"}
logger = logging.getLogger(__name__)


class AlphaVantageNotConfiguredError(VendorNotConfiguredError):
    """Raised when Alpha Vantage is selected but no API key is configured.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer's "vendor unavailable" handling and existing ValueError callers both
    keep working.
    """
    pass


def get_api_key() -> str:
    """Retrieve the API key for Alpha Vantage from environment variables."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise AlphaVantageNotConfiguredError(
            "ALPHA_VANTAGE_API_KEY environment variable is not set."
        )
    return api_key

def format_datetime_for_api(date_input) -> str:
    """Convert various date formats to YYYYMMDDTHHMM format required by Alpha Vantage API."""
    if isinstance(date_input, str):
        # If already in correct format, return as-is
        if len(date_input) == 13 and 'T' in date_input:
            return date_input
        # Try to parse common date formats
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return dt.strftime("%Y%m%dT0000")
        except ValueError:
            try:
                dt = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
                return dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                raise ValueError(f"Unsupported date format: {date_input}") from None
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")

class AlphaVantageRateLimitError(VendorRateLimitError):
    """Raised when the Alpha Vantage API rate limit is exceeded."""
    pass

def _make_api_request(function_name: str, params: dict) -> dict | str:
    """Helper function to make API requests and handle responses.

    Raises:
        AlphaVantageRateLimitError: When API rate limit is exceeded
    """
    # Create a copy of params to avoid modifying the original
    api_params = params.copy()
    api_params.update({
        "function": function_name,
        "apikey": get_api_key(),
        "source": "trading_agents",
    })

    # Handle entitlement parameter if present in params or global variable
    current_entitlement = globals().get('_current_entitlement')
    entitlement = api_params.get("entitlement") or current_entitlement

    if entitlement:
        api_params["entitlement"] = entitlement
    elif "entitlement" in api_params:
        # Remove entitlement if it's None or empty
        api_params.pop("entitlement", None)

    response = requests.get(API_BASE_URL, params=api_params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    response_text = response.text

    # Error responses are JSON; data responses are usually CSV (or data-keyed
    # JSON). A non-JSON body is normal data.
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    # Alpha Vantage reports problems via "Information" / "Note". Classify so a
    # genuine rate limit and an invalid/missing key aren't conflated (#991):
    # rate-limit phrasing is checked first because those notices also mention
    # "API key" ("your API key ... 25 requests per day").
    notice = response_json.get("Information") or response_json.get("Note")
    if notice:
        low = notice.lower()
        if any(m in low for m in ("rate limit", "requests per day", "call frequency", "premium")):
            raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit exceeded: {notice}")
        if "api key" in low or "apikey" in low:
            # Reuse the existing "not configured" error so a bad key surfaces as
            # a real, actionable failure rather than a mislabeled rate limit (#991).
            raise AlphaVantageNotConfiguredError(f"Alpha Vantage API key invalid or missing: {notice}")

    return response_text



def _filter_failure_result(
    csv_data: str,
    start_date: str,
    end_date: str,
    error: Exception | str,
) -> str:
    """Return a safe result for CSV filtering failures.

    Historical runs must not treat an unverifiable vendor response as usable
    evidence. Live runs retain the previous raw-response compatibility path.
    """
    reason = (
        "Alpha Vantage CSV date filtering failed; cannot prove the historical "
        f"time boundary: {error}"
    )
    if current_run_context().mode == "historical":
        audit_source(
            source_name="alpha_vantage.get_stock_data",
            capability="POINT_IN_TIME",
            status="unavailable",
            requested_start=start_date,
            requested_end=end_date,
            reason=reason,
        )
        return f"{HISTORICAL_UNAVAILABLE}: {reason}"

    logger.warning("%s", reason)
    return csv_data


def _parse_csv_bound(value: str, *, end: bool) -> pd.Timestamp:
    """Parse a date bound as an aware UTC timestamp."""
    bound = pd.Timestamp(pd.to_datetime(value, errors="raise", utc=True))
    if end and isinstance(value, str) and len(value.strip()) == 10:
        bound += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return bound


def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    """
    Filter CSV data to include only rows within the specified date range.

    Args:
        csv_data: CSV string from Alpha Vantage API
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Filtered CSV string
    """
    if not isinstance(csv_data, str) or not csv_data.strip():
        return _filter_failure_result(csv_data, start_date, end_date, "empty CSV response")

    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))

        date_columns = [
            column
            for column in df.columns
            if str(column).strip().lower() in _DATE_COLUMN_NAMES
        ]
        if len(date_columns) != 1:
            raise ValueError("CSV must contain exactly one recognized date column")

        date_col = date_columns[0]
        parsed_dates = pd.to_datetime(df[date_col], errors="raise", utc=True)

        # Filter by date range
        start_dt = _parse_csv_bound(start_date, end=False)
        end_dt = _parse_csv_bound(end_date, end=True)
        if start_dt > end_dt:
            raise ValueError(f"start date {start_date!r} is after end date {end_date!r}")

        context = current_run_context()
        if context.mode == "historical":
            end_dt = min(end_dt, pd.Timestamp(context.as_of))

        filtered_df = df[(parsed_dates >= start_dt) & (parsed_dates <= end_dt)]

        # Convert back to CSV string
        return filtered_df.to_csv(index=False)

    except Exception as exc:
        return _filter_failure_result(csv_data, start_date, end_date, exc)
