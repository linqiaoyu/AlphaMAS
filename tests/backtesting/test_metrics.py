import math

import pandas as pd
import pytest

from tradingagents.backtesting.metrics import compute_metrics


def test_metrics_on_hand_computable_curve():
    curve = pd.Series([100.0, 110.0, 99.0, 108.9])
    fills = pd.DataFrame({
        "raw_open_price": [9.99, 10.01],
        "fill_price": [10.0, 10.0],
        "quantity": [5.0, -2.5],
        "notional": [50.0, 25.0],
        "commission": [0.05, 0.025],
        "slippage_cost": [0.05, 0.025],
        "total_transaction_cost": [0.10, 0.05],
    })
    metrics = compute_metrics(
        curve, fills=fills, exposure=pd.Series([0, 1, 1, 0]),
        positions=pd.Series([0, 2, 2, 0]), decision_count=4, decision_failure_count=1,
    )
    assert metrics["cumulative_return"] == pytest.approx(0.089)
    assert metrics["maximum_drawdown"] == pytest.approx(0.10)
    assert metrics["annualized_volatility"] > 0
    assert math.isfinite(metrics["sharpe_ratio"])
    assert metrics["turnover"] == pytest.approx(75 / curve.mean())
    assert metrics["total_commission_cost"] == pytest.approx(0.075)
    assert metrics["total_slippage_cost"] == pytest.approx(0.075)
    assert metrics["total_transaction_cost"] == pytest.approx(0.15)
    assert metrics["commission_cost_rate"] == pytest.approx(0.075 / 100)
    assert metrics["slippage_cost_rate"] == pytest.approx(0.075 / 100)
    assert metrics["transaction_cost_rate"] == pytest.approx(0.15 / 100)
    assert metrics["average_exposure"] == 0.5
    assert metrics["time_in_market"] == 0.5
    assert metrics["decision_failure_rate"] == 0.25


def test_degenerate_metrics_return_null_not_infinity():
    metrics = compute_metrics(pd.Series([100.0, 100.0, 100.0]))
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["calmar_ratio"] is None


def test_sortino_uses_zero_filled_downside_components_and_configured_rf():
    curve = pd.Series([100.0, 102.0, 101.0, 104.0])
    risk_free_rate, annualization = 0.12, 12
    returns = curve.pct_change().dropna()
    excess = returns - risk_free_rate / annualization
    downside = (excess.clip(upper=0) ** 2).mean() ** 0.5
    expected = excess.mean() / downside * annualization**0.5
    metrics = compute_metrics(
        curve, risk_free_rate=risk_free_rate, annualization=annualization,
        initial_equity=200.0,
        fills=pd.DataFrame({
            "raw_open_price": [10.0], "fill_price": [10.0],
            "quantity": [5.0], "notional": [50.0], "commission": [2.0],
        }),
    )
    assert metrics["sortino_ratio"] == pytest.approx(expected)
    assert metrics["transaction_cost_rate"] == pytest.approx(0.01)
    assert metrics["risk_free_rate"] == risk_free_rate
    assert metrics["annualization_factor"] == annualization
