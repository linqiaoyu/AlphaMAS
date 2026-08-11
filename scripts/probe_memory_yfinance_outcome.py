#!/usr/bin/env python3
"""Run the bounded live-yfinance Memory outcome probe for Baseline M0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ASSET = "AAPL"
BENCHMARK = "SPY"
TRADE_DATE = "2024-03-28"
HOLDING_SESSIONS = 5
EXPECTED_SESSIONS = [
    "2024-03-28",
    "2024-04-01",
    "2024-04-02",
    "2024-04-03",
    "2024-04-04",
    "2024-04-05",
]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def _series_payload(series: pd.Series) -> dict[str, float]:
    return {idx.strftime("%Y-%m-%d"): float(value) for idx, value in series.items()}


def _history_payload(history: pd.DataFrame) -> dict[str, Any]:
    index = history.index if isinstance(history, pd.DataFrame) else pd.Index([])
    return {
        "rows": int(len(history)) if isinstance(history, pd.DataFrame) else 0,
        "index_timezone": str(getattr(index, "tz", None)),
        "first_index": str(index[0]) if len(index) else None,
        "last_index": str(index[-1]) if len(index) else None,
        "columns": list(map(str, history.columns)) if isinstance(history, pd.DataFrame) else [],
    }


def _finite_number(value: float | None) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def run_probe() -> dict[str, Any]:
    import exchange_calendars as xcals

    from tradingagents.graph import trading_graph as memory_graph
    from tradingagents.runtime.run_context import (
        AuditTrail,
        RunContext,
        activate_run_context,
    )

    calendar = xcals.get_calendar("XNYS")
    start_session = pd.Timestamp(TRADE_DATE)
    required_sessions = memory_graph._xnys_outcome_sessions(
        start_session,
        HOLDING_SESSIONS,
    )
    required_session_labels = [session.strftime("%Y-%m-%d") for session in required_sessions]
    endpoint_session = required_sessions[-1]
    endpoint_close = calendar.session_close(endpoint_session)
    context = RunContext.historical(endpoint_close.to_pydatetime())
    audit = AuditTrail(context)

    if required_session_labels != EXPECTED_SESSIONS:
        raise RuntimeError(
            "unexpected XNYS session window: "
            f"{required_session_labels!r} != {EXPECTED_SESSIONS!r}"
        )

    cutoff_session = memory_graph._completed_outcome_cutoff(context, BENCHMARK)
    cutoff_date = datetime.strptime(str(cutoff_session)[:10], "%Y-%m-%d").date()
    requested_end_exclusive = (cutoff_date + timedelta(days=1)).isoformat()
    history_options = dict(memory_graph._MEMORY_OUTCOME_HISTORY_OPTIONS)

    asset_history = yf.Ticker(ASSET).history(
        start=TRADE_DATE,
        end=requested_end_exclusive,
        **history_options,
    )
    benchmark_history = yf.Ticker(BENCHMARK).history(
        start=TRADE_DATE,
        end=requested_end_exclusive,
        **history_options,
    )

    asset_close = memory_graph._outcome_close_by_session(
        asset_history,
        start_session=start_session,
        cutoff_session=cutoff_session,
    ).rename("asset")
    benchmark_close = memory_graph._outcome_close_by_session(
        benchmark_history,
        start_session=start_session,
        cutoff_session=cutoff_session,
    ).rename("benchmark")
    asset_window = asset_close.reindex(required_sessions)
    benchmark_window = benchmark_close.reindex(required_sessions)

    benchmark_return = float(benchmark_window.iloc[-1] / benchmark_window.iloc[0] - 1)

    graph = object.__new__(memory_graph.TradingAgentsGraph)
    graph.config = {
        "memory_holding_horizon_sessions": HOLDING_SESSIONS,
        "memory_outcome_price_mode": memory_graph._MEMORY_OUTCOME_PRICE_MODE,
    }

    with activate_run_context(context, audit):
        raw_return, alpha_return, holding_days = memory_graph.TradingAgentsGraph._fetch_returns(
            graph,
            ASSET,
            TRADE_DATE,
            holding_days=HOLDING_SESSIONS,
            benchmark=BENCHMARK,
        )

    complete = (
        holding_days == HOLDING_SESSIONS
        and _finite_number(raw_return)
        and _finite_number(alpha_return)
        and not asset_window.isna().any()
        and not benchmark_window.isna().any()
        and required_sessions[-1] == cutoff_session
    )
    status = "used" if complete else "failed"

    probe_script = Path(__file__)
    return {
        "probe": "baseline_m0_memory_yfinance_outcome",
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "code_commit_sha": _git_output(["git", "rev-parse", "HEAD"]),
        "working_tree_status": _git_output(["git", "status", "--short"]),
        "probe_script_sha256": hashlib.sha256(probe_script.read_bytes()).hexdigest(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "packages": {
            "yfinance": _package_version("yfinance"),
            "pandas": _package_version("pandas"),
            "exchange-calendars": _package_version("exchange-calendars"),
        },
        "input": {
            "asset": ASSET,
            "benchmark": BENCHMARK,
            "decision_session": TRADE_DATE,
            "holding_horizon_sessions": HOLDING_SESSIONS,
            "price_mode": memory_graph._MEMORY_OUTCOME_PRICE_MODE,
            "history_options": history_options,
            "requested_start": TRADE_DATE,
            "requested_end_exclusive": requested_end_exclusive,
            "run_context_as_of": context.as_of.isoformat(),
            "endpoint_session": endpoint_session.strftime("%Y-%m-%d"),
            "endpoint_xnys_close": endpoint_close.isoformat(),
        },
        "provider_history": {
            "asset": _history_payload(asset_history),
            "benchmark": _history_payload(benchmark_history),
        },
        "sessions": {
            "expected": EXPECTED_SESSIONS,
            "resolved": required_session_labels,
            "cutoff_session": cutoff_session.strftime("%Y-%m-%d"),
            "max_used_session": required_sessions[-1].strftime("%Y-%m-%d"),
            "future_rows_used": bool(required_sessions[-1] > cutoff_session),
            "good_friday_2024_03_29_excluded": "2024-03-29" not in required_session_labels,
            "weekend_2024_03_30_31_excluded": (
                "2024-03-30" not in required_session_labels
                and "2024-03-31" not in required_session_labels
            ),
        },
        "prices": {
            "asset_adjusted_close": _series_payload(asset_window),
            "benchmark_adjusted_close": _series_payload(benchmark_window),
        },
        "returns": {
            "raw_return": raw_return,
            "benchmark_return": benchmark_return,
            "alpha_return": alpha_return,
            "holding_days": holding_days,
            "alpha_equals_raw_minus_benchmark": (
                abs((raw_return - benchmark_return) - alpha_return) < 1e-12
                if raw_return is not None and alpha_return is not None else False
            ),
        },
        "audit": audit.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m0_freeze/memory_yfinance_probe_20260811.json"),
    )
    args = parser.parse_args()

    payload = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "raw_return": payload["returns"]["raw_return"],
        "benchmark_return": payload["returns"]["benchmark_return"],
        "alpha_return": payload["returns"]["alpha_return"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
