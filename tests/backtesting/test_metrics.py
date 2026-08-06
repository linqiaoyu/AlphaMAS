import math

import pandas as pd
import pytest

from tradingagents.backtesting.metrics import compute_metrics


def test_metrics_on_hand_computable_curve():
    curve = pd.Series([100.0, 110.0, 99.0, 108.9])
    fills = pd.DataFrame({"notional": [50.0, 25.0], "commission": [0.05, 0.025]})
    metrics = compute_metrics(
        curve, fills=fills, exposure=pd.Series([0, 1, 1, 0]),
        positions=pd.Series([0, 2, 2, 0]), decision_count=4, decision_failure_count=1,
    )
    assert metrics["cumulative_return"] == pytest.approx(0.089)
    assert metrics["maximum_drawdown"] == pytest.approx(0.10)
    assert metrics["annualized_volatility"] > 0
    assert math.isfinite(metrics["sharpe_ratio"])
    assert metrics["turnover"] == pytest.approx(75 / curve.mean())
    assert metrics["average_exposure"] == 0.5
    assert metrics["time_in_market"] == 0.5
    assert metrics["decision_failure_rate"] == 0.25


def test_degenerate_metrics_return_null_not_infinity():
    metrics = compute_metrics(pd.Series([100.0, 100.0, 100.0]))
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["calmar_ratio"] is None
