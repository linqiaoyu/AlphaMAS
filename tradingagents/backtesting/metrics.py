"""Paper-oriented metrics from daily equity and actual fills."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def compute_metrics(
    equity: pd.Series, *, fills: pd.DataFrame | None = None,
    exposure: pd.Series | None = None, positions: pd.Series | None = None,
    decision_count: int = 0, decision_failure_count: int = 0,
    risk_free_rate: float = 0.0, annualization: int = 252,
    initial_equity: float | None = None, decisions: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None, cumulative_dividends: float = 0.0,
) -> dict[str, Any]:
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    curve = pd.Series(equity, dtype=float).dropna()
    if curve.empty or (curve <= 0).any():
        raise ValueError("equity curve must be non-empty and positive")
    returns = curve.pct_change().dropna()
    cumulative = float(curve.iloc[-1] / curve.iloc[0] - 1)
    periods = max(len(curve) - 1, 1)
    annual_return = float((curve.iloc[-1] / curve.iloc[0]) ** (annualization / periods) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization)) if len(returns) > 1 else 0.0
    daily_rf = risk_free_rate / annualization
    excess = returns - daily_rf
    daily_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    sharpe = None if daily_std == 0 else float(excess.mean() / daily_std * np.sqrt(annualization))
    downside_component = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt(np.mean(downside_component**2)))
    sortino = None if downside_deviation == 0 else float(
        excess.mean() / downside_deviation * np.sqrt(annualization)
    )
    drawdowns = curve / curve.cummax() - 1
    maximum_drawdown = float(-drawdowns.min())
    calmar = None if maximum_drawdown == 0 else annual_return / maximum_drawdown
    fills = pd.DataFrame() if fills is None else fills
    total_cost = float(fills.get("commission", pd.Series(dtype=float)).sum())
    notional = float(fills.get("notional", pd.Series(dtype=float)).abs().sum())
    avg_equity = float(curve.mean())
    initial = float(curve.iloc[0] if initial_equity is None else initial_equity)
    decisions = pd.DataFrame() if decisions is None else decisions
    orders = pd.DataFrame() if orders is None else orders
    action_counts = decisions.get("action", pd.Series(dtype=str)).value_counts()
    order_status = orders.get("status", pd.Series(dtype=str)).value_counts()
    return {
        "cumulative_return": cumulative,
        "annualized_return": annual_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": _finite_or_none(sharpe) if sharpe is not None else None,
        "sortino_ratio": _finite_or_none(sortino) if sortino is not None else None,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": _finite_or_none(calmar) if calmar is not None else None,
        "turnover": notional / avg_equity,
        "total_transaction_cost": total_cost,
        "transaction_cost_rate": total_cost / initial,
        "trade_count": int(len(fills)),
        "average_exposure": float(pd.Series(exposure, dtype=float).mean()) if exposure is not None else None,
        "time_in_market": float((pd.Series(positions, dtype=float) > 0).mean()) if positions is not None else None,
        "decision_count": int(decision_count),
        "decision_failure_count": int(decision_failure_count),
        "decision_failure_rate": decision_failure_count / decision_count if decision_count else 0.0,
        "cumulative_dividends": float(cumulative_dividends),
        "buy_decision_count": int(action_counts.get("BUY", 0)),
        "hold_decision_count": int(action_counts.get("HOLD", 0)),
        "sell_decision_count": int(action_counts.get("SELL", 0)),
        "noop_rebalance_count": int(
            (decisions.get("rebalance_status", pd.Series(dtype=str)) == "noop").sum()
        ),
        "filled_order_count": int(order_status.get("filled", 0)),
        "rejected_order_count": int(order_status.get("rejected", 0)),
        "risk_free_rate": float(risk_free_rate),
        "annualization_factor": int(annualization),
    }
