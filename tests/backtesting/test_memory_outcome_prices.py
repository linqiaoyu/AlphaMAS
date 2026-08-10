"""Point-in-time Memory outcome price and session regressions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.run_context import RunContext, activate_run_context

TRADE_DATE = "2024-03-28"
SESSIONS = [
    "2024-03-28",
    "2024-04-01",  # Good Friday and the weekend do not appear.
    "2024-04-02",
    "2024-04-03",
    "2024-04-04",
    "2024-04-05",
    "2024-04-08",
]
HISTORY_OPTIONS = {
    "period": None,
    "interval": "1d",
    "prepost": False,
    "actions": False,
    "auto_adjust": True,
    "back_adjust": False,
    "repair": False,
    "keepna": False,
    "rounding": False,
}


def _prices(dates, closes, *, timezone_name=None) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if timezone_name is not None:
        index = index.tz_localize(timezone_name)
    return pd.DataFrame({"Close": closes}, index=index)


def _graph_config() -> dict:
    return {
        "benchmark_ticker": "SPY",
        "benchmark_map": {"": "SPY"},
        "memory_holding_horizon_sessions": 5,
        "memory_outcome_price_mode": "adjusted_close",
    }


def _ticker_factory(frames: dict[str, pd.DataFrame], requests: list[dict]):
    def ticker(symbol: str):
        frame = frames[symbol]

        class OfflineTicker:
            def history(self, **kwargs):
                requests.append({"symbol": symbol, **kwargs})
                return frame.copy()

        return OfflineTicker()

    return ticker


def _fetch(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[tuple[float | None, float | None, int | None], list[dict]]:
    graph = object.__new__(TradingAgentsGraph)
    graph.config = _graph_config()
    requests: list[dict] = []
    frames = {"AAPL": asset, "SPY": benchmark}
    context = RunContext.historical(as_of, experiment_id="memory-outcome-test")
    with (
        patch(
            "tradingagents.graph.trading_graph.yf.Ticker",
            side_effect=_ticker_factory(frames, requests),
        ),
        activate_run_context(context),
    ):
        result = graph._fetch_returns("AAPL", TRADE_DATE, benchmark="SPY")
    return result, requests


def _resolve_variant(
    tmp_path: Path,
    name: str,
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[dict, MagicMock]:
    graph = object.__new__(TradingAgentsGraph)
    graph.config = _graph_config()
    graph.reflector = MagicMock()
    graph.reflector.reflect_on_final_decision.return_value = "Mature reflection"
    graph.memory_log = TradingMemoryLog(
        {
            "memory_log_path": str(tmp_path / f"{name}.md"),
            "historical_as_of": "2024-04-05",
        }
    )
    graph.memory_log.store_decision("AAPL", TRADE_DATE, "Rating: Buy")
    frames = {"AAPL": asset, "SPY": benchmark}
    with (
        patch(
            "tradingagents.graph.trading_graph.yf.Ticker",
            side_effect=_ticker_factory(frames, []),
        ),
        activate_run_context(RunContext.historical("2024-04-05")),
    ):
        graph._resolve_pending_entries("AAPL")
    return graph.memory_log.load_entries()[0], graph.reflector


@pytest.mark.unit
def test_timezone_aware_outcome_stays_pending_then_resolves_exactly_once(tmp_path):
    timezone_name = "America/New_York"
    frames = {
        "AAPL": _prices(
            SESSIONS, [100, 101, 102, 103, 104, 110, 999],
            timezone_name=timezone_name,
        ),
        "SPY": _prices(
            SESSIONS, [200, 201, 202, 203, 204, 210, 1],
            timezone_name=timezone_name,
        ),
    }
    requests: list[dict] = []
    graph = object.__new__(TradingAgentsGraph)
    graph.config = _graph_config()
    graph.reflector = MagicMock()
    graph.reflector.reflect_on_final_decision.return_value = "Mature reflection"
    memory_path = tmp_path / "timezone-aware.md"
    graph.memory_log = TradingMemoryLog(
        {"memory_log_path": str(memory_path), "historical_as_of": "2024-04-04"}
    )
    graph.memory_log.store_decision("AAPL", TRADE_DATE, "Rating: Buy")

    with patch(
        "tradingagents.graph.trading_graph.yf.Ticker",
        side_effect=_ticker_factory(frames, requests),
    ):
        with activate_run_context(RunContext.historical("2024-04-04")):
            graph._resolve_pending_entries("AAPL")
        assert graph.memory_log.get_pending_entries()
        graph.reflector.reflect_on_final_decision.assert_not_called()

        graph.memory_log = TradingMemoryLog(
            {"memory_log_path": str(memory_path), "historical_as_of": "2024-04-05"}
        )
        with activate_run_context(RunContext.historical("2024-04-05")):
            graph._resolve_pending_entries("AAPL")
            graph._resolve_pending_entries("AAPL")

    assert graph.memory_log.get_pending_entries() == []
    graph.reflector.reflect_on_final_decision.assert_called_once()
    call = graph.reflector.reflect_on_final_decision.call_args.kwargs
    assert call["raw_return"] == pytest.approx(0.10)
    assert call["alpha_return"] == pytest.approx(0.05)
    assert [request["end"] for request in requests] == [
        "2024-04-05", "2024-04-05", "2024-04-06", "2024-04-06",
    ]


@pytest.mark.unit
def test_timezone_naive_sessions_align_asset_and_benchmark_endpoints():
    asset = _prices(SESSIONS, [100, 101, 999, 103, 104, 105, 110])
    benchmark = _prices(SESSIONS, [200, 202, 1, 206, 208, 210, 220])

    result, _ = _fetch(asset, benchmark, as_of="2024-04-08")

    raw, alpha, days = result
    assert raw == pytest.approx(0.05)
    assert raw - alpha == pytest.approx(0.05)
    assert days == 5


@pytest.mark.unit
def test_provider_with_missing_benchmark_session_fails_closed():
    asset = _prices(SESSIONS[:6], [100, 101, 102, 103, 104, 105])
    benchmark = _prices(
        ["2024-03-28", "2024-03-29", "2024-04-01", "2024-04-03", "2024-04-04", "2024-04-05"],
        [200, 201, 202, 203, 204, 205],
    )

    result, _ = _fetch(asset, benchmark, as_of="2024-04-05")

    assert result == (None, None, None)


@pytest.mark.unit
def test_common_provider_gap_cannot_shift_xnys_horizon_endpoint():
    dates_with_common_gap = [
        "2024-03-28", "2024-04-01", "2024-04-03",
        "2024-04-04", "2024-04-05", "2024-04-08",
    ]
    asset = _prices(dates_with_common_gap, [100, 101, 103, 104, 105, 108])
    benchmark = _prices(dates_with_common_gap, [200, 201, 203, 204, 205, 208])

    result, _ = _fetch(asset, benchmark, as_of="2024-04-08")

    assert result == (None, None, None)


@pytest.mark.unit
def test_timestamped_spy_cutoff_waits_for_actual_xnys_close():
    timezone_name = "America/New_York"
    asset = _prices(
        SESSIONS[:6], [100, 101, 102, 103, 104, 105],
        timezone_name=timezone_name,
    )
    benchmark = _prices(
        SESSIONS[:6], [200, 201, 202, 203, 204, 205],
        timezone_name=timezone_name,
    )

    before_close, before_requests = _fetch(
        asset, benchmark, as_of="2024-04-05T19:59:00+00:00"
    )
    at_close, at_close_requests = _fetch(
        asset, benchmark, as_of="2024-04-05T20:00:00+00:00"
    )

    assert before_close == (None, None, None)
    assert at_close[0] == pytest.approx(0.05)
    assert [request["end"] for request in before_requests] == [
        "2024-04-05", "2024-04-05",
    ]
    assert [request["end"] for request in at_close_requests] == [
        "2024-04-06", "2024-04-06",
    ]


@pytest.mark.unit
def test_live_mode_excludes_current_daily_bar_until_next_utc_date():
    asset = _prices(SESSIONS[:6], [100, 101, 102, 103, 104, 105])
    benchmark = _prices(SESSIONS[:6], [200, 201, 202, 203, 204, 205])
    frames = {"AAPL": asset, "SPY": benchmark}
    graph = object.__new__(TradingAgentsGraph)
    graph.config = _graph_config()

    def run(at: datetime):
        requests: list[dict] = []
        with (
            patch(
                "tradingagents.graph.trading_graph.yf.Ticker",
                side_effect=_ticker_factory(frames, requests),
            ),
            activate_run_context(RunContext.live(at)),
        ):
            result = graph._fetch_returns("AAPL", TRADE_DATE, benchmark="SPY")
        return result, requests

    same_day, same_day_requests = run(
        datetime(2024, 4, 5, 23, tzinfo=timezone.utc)
    )
    next_day, next_day_requests = run(
        datetime(2024, 4, 6, 1, tzinfo=timezone.utc)
    )

    assert same_day == (None, None, None)
    assert next_day[0] == pytest.approx(0.05)
    assert same_day_requests[0]["end"] == "2024-04-05"
    assert next_day_requests[0]["end"] == "2024-04-06"


@pytest.mark.unit
def test_same_day_replay_cannot_see_reflection_before_exact_visibility(tmp_path):
    timezone_name = "America/New_York"
    frames = {
        "AAPL": _prices(
            SESSIONS[:6], [100, 101, 102, 103, 104, 105],
            timezone_name=timezone_name,
        ),
        "SPY": _prices(
            SESSIONS[:6], [200, 201, 202, 203, 204, 205],
            timezone_name=timezone_name,
        ),
    }
    memory_path = tmp_path / "timestamp-visible.md"
    graph = object.__new__(TradingAgentsGraph)
    graph.config = _graph_config()
    graph.reflector = MagicMock()
    graph.reflector.reflect_on_final_decision.return_value = "Visible at close"
    graph.memory_log = TradingMemoryLog(
        {
            "memory_log_path": str(memory_path),
            "historical_as_of": "2024-04-05T20:00:00+00:00",
        }
    )
    graph.memory_log.store_decision("AAPL", TRADE_DATE, "Rating: Buy")

    with (
        patch(
            "tradingagents.graph.trading_graph.yf.Ticker",
            side_effect=_ticker_factory(frames, []),
        ),
        activate_run_context(RunContext.historical("2024-04-05T20:00:00+00:00")),
    ):
        graph._resolve_pending_entries("AAPL")

    before_close = TradingMemoryLog(
        {
            "memory_log_path": str(memory_path),
            "historical_as_of": "2024-04-05T19:59:00+00:00",
        }
    )
    at_close = TradingMemoryLog(
        {
            "memory_log_path": str(memory_path),
            "historical_as_of": "2024-04-05T20:00:00+00:00",
        }
    )

    assert before_close.get_past_context("AAPL") == ""
    assert "Visible at close" in at_close.get_past_context("AAPL")
    assert at_close.load_entries()[0]["outcome_visible_from"] == (
        "2024-04-05T20:00:00+00:00"
    )


@pytest.mark.unit
def test_rows_after_historical_cutoff_cannot_change_returns_or_reflection(tmp_path):
    asset = _prices(SESSIONS, [100, 101, 102, 103, 104, 110, 111])
    benchmark = _prices(SESSIONS, [200, 201, 202, 203, 204, 210, 211])
    mutated_asset = asset.copy()
    mutated_benchmark = benchmark.copy()
    mutated_asset.iloc[-1, 0] = 9_999_999
    mutated_benchmark.iloc[-1, 0] = 1

    first, _ = _fetch(asset, benchmark, as_of="2024-04-05")
    second, _ = _fetch(mutated_asset, mutated_benchmark, as_of="2024-04-05")
    first_entry, first_reflector = _resolve_variant(
        tmp_path, "first", asset, benchmark
    )
    second_entry, second_reflector = _resolve_variant(
        tmp_path, "second", mutated_asset, mutated_benchmark
    )

    assert first == second
    assert first[0] == pytest.approx(0.10)
    assert first[0] - first[1] == pytest.approx(0.05)
    assert first_entry["pending"] is second_entry["pending"] is False
    assert first_entry["reflection"] == second_entry["reflection"] == "Mature reflection"
    assert first_reflector.reflect_on_final_decision.call_args == (
        second_reflector.reflect_on_final_decision.call_args
    )


@pytest.mark.unit
def test_yfinance_outcome_price_options_are_fully_explicit():
    asset = _prices(SESSIONS[:6], [100, 101, 102, 103, 104, 105])
    benchmark = _prices(SESSIONS[:6], [200, 201, 202, 203, 204, 205])

    result, requests = _fetch(asset, benchmark, as_of="2024-04-05")

    assert result[0] is not None
    assert len(requests) == 2
    for request in requests:
        assert request["start"] == TRADE_DATE
        assert request["end"] == "2024-04-06"
        assert {key: request[key] for key in HISTORY_OPTIONS} == HISTORY_OPTIONS
