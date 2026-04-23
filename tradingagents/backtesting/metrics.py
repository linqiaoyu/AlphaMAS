from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    cumulative_return: float
    annualized_return: float
    sharpe_ratio: float
    maximum_drawdown: float
    final_equity: float
    initial_equity: float
    trading_days: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "CR": float(self.cumulative_return),
            "AR": float(self.annualized_return),
            "SR": float(self.sharpe_ratio),
            "MDD": float(self.maximum_drawdown),
            "final_equity": float(self.final_equity),
            "initial_equity": float(self.initial_equity),
            "trading_days": int(self.trading_days),
        }


def compute_backtest_metrics(
    equity_curve: pd.Series,
    *,
    annual_risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Compute the paper-aligned summary metrics from a daily equity curve."""
    cleaned = pd.Series(equity_curve).dropna().astype(float)
    if cleaned.empty:
        raise ValueError("Cannot compute backtest metrics from an empty equity curve.")

    cleaned = cleaned.sort_index()
    initial_equity = float(cleaned.iloc[0])
    if initial_equity <= 0:
        raise ValueError(
            "Backtest metrics require a strictly positive initial equity."
        )

    final_equity = float(cleaned.iloc[-1])
    cumulative_return = (final_equity / initial_equity) - 1.0

    if len(cleaned.index) == 1:
        years = 1.0 / 252.0
    else:
        first_date = pd.Timestamp(cleaned.index[0])
        last_date = pd.Timestamp(cleaned.index[-1])
        years = max((last_date - first_date).days / 365.25, 1.0 / 252.0)

    # Once equity reaches zero or below, the strategy has effectively blown up.
    # Geometric annualization is not meaningful there, so we cap AR at -100%.
    if final_equity <= 0:
        annualized_return = -1.0
    else:
        annualized_return = (final_equity / initial_equity) ** (1.0 / years) - 1.0

    daily_returns = cleaned.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily_risk_free_rate = annual_risk_free_rate / 252.0
    excess_returns = daily_returns - daily_risk_free_rate
    volatility = float(excess_returns.std(ddof=0))
    if volatility <= 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(np.sqrt(252.0) * excess_returns.mean() / volatility)

    rolling_peak = cleaned.cummax()
    drawdowns = 1.0 - (cleaned / rolling_peak)
    maximum_drawdown = float(drawdowns.max())

    return BacktestMetrics(
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        maximum_drawdown=maximum_drawdown,
        final_equity=final_equity,
        initial_equity=initial_equity,
        trading_days=len(cleaned),
    )
