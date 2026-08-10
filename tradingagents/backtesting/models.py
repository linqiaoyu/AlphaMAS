"""Stable, serializable models for weekly walk-forward backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class DecisionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class StrategyDecision:
    symbol: str
    decision_session: str
    decision_time_utc: datetime
    action: Action | None
    target_weight: float | None
    status: DecisionStatus
    reason: str
    strategy_id: str
    experiment_id: str
    raw_signal: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision_time_utc"] = self.decision_time_utc.isoformat()
        data["action"] = self.action.value if self.action else None
        data["status"] = self.status.value
        return data


@dataclass
class Order:
    order_id: str
    symbol: str
    created_at: datetime
    intended_execution_session: str
    target_weight: float
    current_weight: float
    side: str
    status: str = "pending"
    reason: str = "target weight changed"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    execution_time: datetime
    raw_open_price: float
    slippage_bps: float
    fill_price: float
    quantity: float
    notional: float
    commission: float
    cash_after: float
    position_after: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["execution_time"] = self.execution_time.isoformat()
        return data


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    session: str
    symbol: str
    cash: float
    quantity: float
    close_price: float
    market_value: float
    equity: float
    current_weight: float
    average_entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    cumulative_cost: float
    current_drawdown: float
    peak_equity: float
    cumulative_dividends: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
