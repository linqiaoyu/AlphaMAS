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
) -> dict[str, Any]:
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
    downside = excess[excess < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = None if downside_std == 0 else float(excess.mean() / downside_std * np.sqrt(annualization))
    drawdowns = curve / curve.cummax() - 1
    maximum_drawdown = float(-drawdowns.min())
    calmar = None if maximum_drawdown == 0 else annual_return / maximum_drawdown
    fills = pd.DataFrame() if fills is None else fills
    total_cost = float(fills.get("commission", pd.Series(dtype=float)).sum())
    notional = float(fills.get("notional", pd.Series(dtype=float)).abs().sum())
    avg_equity = float(curve.mean())
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
        "transaction_cost_rate": total_cost / avg_equity,
        "trade_count": int(len(fills)),
        "average_exposure": float(pd.Series(exposure, dtype=float).mean()) if exposure is not None else None,
        "time_in_market": float((pd.Series(positions, dtype=float) > 0).mean()) if positions is not None else None,
        "decision_count": int(decision_count),
        "decision_failure_count": int(decision_failure_count),
        "decision_failure_rate": decision_failure_count / decision_count if decision_count else 0.0,
    }
