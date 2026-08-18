from __future__ import annotations

import inspect
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import tradingagents.backtesting.arma11 as arma
from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.data import InMemoryDataProvider
from tradingagents.backtesting.engine import WeeklyBacktestEngine
from tradingagents.backtesting.models import Action, PortfolioSnapshot


def frame(rows: int = 253, *, start: str = "2022-01-03") -> pd.DataFrame:
    sessions = ExchangeSchedule().sessions(start, "2023-12-29").tz_localize(None)[:rows]
    close = 100.0 * np.cumprod(1.0 + 0.001 * np.sin(np.arange(rows) / 7.0))
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=sessions,
    )


def snapshot(quantity: float = 0.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=datetime(2023, 1, 3, tzinfo=timezone.utc),
        session="2023-01-03",
        symbol="TEST",
        cash=100_000.0 if quantity == 0 else 0.0,
        quantity=quantity,
        close_price=100.0,
        market_value=quantity * 100.0,
        equity=100_000.0,
        current_weight=0.0 if quantity == 0 else 1.0,
        average_entry_price=0.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        cumulative_cost=0.0,
        current_drawdown=0.0,
        peak_equity=100_000.0,
    )


def decision(monkeypatch: pytest.MonkeyPatch, cumulative: float, quantity: float = 0.0):
    data = frame()
    fake = arma.FitForecast(
        params={"const": 0.0, "ar.L1": 0.1, "ma.L1": -0.1, "sigma2": 1e-6},
        forecasts=(cumulative, 0.0, 0.0, 0.0, 0.0),
        cumulative_forecast=cumulative,
        converged=True,
        warnings=(),
    )
    monkeypatch.setattr(arma, "fit_forecast_arma11", lambda _: fake)
    return arma.ARMA11Strategy().decide(
        symbol="TEST",
        decision_session=data.index[-1].date().isoformat(),
        decision_time=datetime(2023, 1, 3, tzinfo=timezone.utc),
        market_history=data,
        portfolio_snapshot=snapshot(quantity),
        context={
            "experiment_id": "TEST",
            "execution_session": "2023-01-04",
        },
    )


def test_exact_252_returns_use_253_consecutive_prices_and_no_future() -> None:
    data = frame(260)
    window = arma.pit_safe_total_returns(
        data, decision_session=data.index[-1].date().isoformat()
    )
    assert len(window.returns) == 252
    assert window.price_start_session == data.index[-253].date().isoformat()
    assert window.input_start_session == data.index[-252].date().isoformat()
    assert window.input_end_session == data.index[-1].date().isoformat()
    assert window.returns.index.max() <= data.index[-1]
    assert len(window.sha256) == 64


def test_session_local_dividend_and_split_are_pit_safe() -> None:
    data = frame()
    data.iloc[-1, data.columns.get_loc("Close")] = 50.0
    data.iloc[-1, data.columns.get_loc("Dividends")] = 1.0
    data.iloc[-1, data.columns.get_loc("Stock Splits")] = 2.0
    prior = float(data.iloc[-2]["Close"])
    expected = (50.0 * 2.0 + 1.0) / prior - 1.0
    window = arma.pit_safe_total_returns(
        data, decision_session=data.index[-1].date().isoformat()
    )
    assert window.returns.iloc[-1] == pytest.approx(expected)


def test_future_input_and_insufficient_history_fail_closed() -> None:
    data = frame()
    with pytest.raises(ValueError, match="after the decision"):
        arma.pit_safe_total_returns(
            data, decision_session=data.index[-2].date().isoformat()
        )
    with pytest.raises(ValueError, match="fewer than 252"):
        arma.pit_safe_total_returns(
            data.iloc[:-1], decision_session=data.index[-2].date().isoformat()
        )


@pytest.mark.parametrize(
    ("forecast", "quantity", "target", "action"),
    [
        (0.01, 0.0, 1.0, Action.BUY),
        (0.01, 10.0, 1.0, Action.HOLD),
        (0.0, 0.0, 0.0, Action.HOLD),
        (0.0, 10.0, 0.0, Action.SELL),
        (-0.01, 0.0, 0.0, Action.HOLD),
        (-0.01, 10.0, 0.0, Action.SELL),
    ],
)
def test_frozen_signal_and_action_label_mapping(
    monkeypatch: pytest.MonkeyPatch,
    forecast: float,
    quantity: float,
    target: float,
    action: Action,
) -> None:
    result = decision(monkeypatch, forecast, quantity)
    assert result.target_weight == target
    assert result.action is action
    assert result.metadata["action_label"] == action.value


def test_hard_failure_holds_and_preserves_current_position() -> None:
    data = frame(252)
    result = arma.ARMA11Strategy().decide(
        symbol="TEST",
        decision_session=data.index[-1].date().isoformat(),
        decision_time=datetime(2023, 1, 3, tzinfo=timezone.utc),
        market_history=data,
        portfolio_snapshot=snapshot(10.0),
        context={"experiment_id": "TEST", "execution_session": "2023-01-04"},
    )
    assert result.action is Action.HOLD
    assert result.target_weight == 1.0
    assert result.metadata["fit_status"] == arma.ARMA11_FAILURE_STATUS
    assert result.metadata["model_failure"] is True


def test_exact_five_step_forecast_compounding_and_warning_provenance(monkeypatch) -> None:
    expected = np.array([0.01, -0.02, 0.03, 0.0, 0.04])

    class Result:
        params = np.array([0.0, 0.1, -0.1, 1e-6])
        param_names = ["const", "ar.L1", "ma.L1", "sigma2"]
        mle_retvals = {"converged": False}

        def forecast(self, *, steps):
            assert steps == 5
            return expected

    class Model:
        def __init__(self, values, **kwargs):
            assert len(values) == 252
            assert kwargs == {
                "order": (1, 0, 1),
                "trend": "c",
                "enforce_stationarity": True,
                "enforce_invertibility": True,
            }

        def fit(self, *, method, method_kwargs):
            assert method == "statespace"
            assert method_kwargs == {"maxiter": 200, "disp": 0}
            warnings.warn("retained warning", ConvergenceWarning, stacklevel=2)
            return Result()

    monkeypatch.setattr(arma, "ARIMA", Model)
    fitted = arma.fit_forecast_arma11(pd.Series(np.linspace(-0.01, 0.01, 252)))
    assert fitted.forecasts == tuple(expected)
    assert fitted.cumulative_forecast == pytest.approx(float(np.prod(1 + expected) - 1))
    assert fitted.converged is False
    assert fitted.warnings == ("ConvergenceWarning: retained warning",)


def test_actual_statespace_fit_is_deterministic() -> None:
    values = pd.Series(0.001 * np.sin(np.arange(252) / 9.0))
    first = arma.fit_forecast_arma11(values)
    second = arma.fit_forecast_arma11(values)
    assert first.forecasts == pytest.approx(second.forecasts, rel=0.0, abs=1e-12)
    assert first.cumulative_forecast == pytest.approx(
        second.cumulative_forecast, rel=0.0, abs=1e-12
    )


def test_xnys_weekly_schedule_includes_good_friday_week() -> None:
    events = ExchangeSchedule().weekly_events("2024-01-01", "2024-06-24")
    assert len(events) == 26
    assert events[0].decision_session == "2024-01-05"
    assert events[12].decision_session == "2024-03-28"
    assert events[12].execution_session == "2024-04-01"
    assert events[-1].execution_session == "2024-07-01"


def test_common_engine_next_open_costs_long_only_and_final_mark(monkeypatch) -> None:
    schedule = ExchangeSchedule()
    sessions = schedule.sessions("2022-01-03", "2023-01-13").tz_localize(None)
    close = np.linspace(100.0, 120.0, len(sessions))
    data = pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=sessions,
    )
    fake = arma.FitForecast(
        params={"const": 0.0, "ar.L1": 0.0, "ma.L1": 0.0, "sigma2": 1e-6},
        forecasts=(0.01, 0.0, 0.0, 0.0, 0.0),
        cumulative_forecast=0.01,
        converged=True,
        warnings=(),
    )
    monkeypatch.setattr(arma, "fit_forecast_arma11", lambda _: fake)
    result = WeeklyBacktestEngine(
        data_provider=InMemoryDataProvider({"TEST": data}),
        schedule=schedule,
        commission_bps=5,
        slippage_bps=5,
        fractional_shares=True,
    ).run(
        symbol="TEST",
        first_week="2023-01-02",
        final_week="2023-01-02",
        final_valuation_session="2023-01-13",
        strategy=arma.ARMA11Strategy(),
        experiment_id="TEST",
        warmup_start=sessions[0].date().isoformat(),
    )
    assert result.decisions.iloc[0]["decision_session"] == "2023-01-06"
    assert result.orders.iloc[0]["intended_execution_session"] == "2023-01-09"
    assert str(result.fills.iloc[0]["execution_time"])[:10] == "2023-01-09"
    assert result.fills.iloc[0]["commission"] > 0
    assert result.fills.iloc[0]["slippage_cost"] > 0
    assert (result.daily_equity["cash"] >= -1e-8).all()
    assert (result.daily_equity["quantity"] >= -1e-8).all()
    assert result.daily_equity.iloc[-1]["session"] == "2023-01-13"
    assert result.daily_equity.iloc[-1]["quantity"] > 0


def test_arma_module_has_no_mas_finmultitime_memory_or_rl_dependency() -> None:
    source = inspect.getsource(arma)
    forbidden = (
        "TradingAgentsGraph",
        "TradingAgentsStrategy",
        "FinMultiTime",
        "DecisionCache",
        "tradingagents.m2",
        "DeepSeek",
        "Qwen",
    )
    assert all(name not in source for name in forbidden)
    strategy = arma.ARMA11Strategy()
    assert not hasattr(strategy, "memory")
    assert not hasattr(strategy, "rl_state")
