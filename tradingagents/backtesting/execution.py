"""Target-weight execution at the next XNYS open."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from tradingagents.backtesting.models import Fill, Order
from tradingagents.backtesting.portfolio import Portfolio


class Broker:
    def __init__(
        self, *, commission_bps: float = 5, slippage_bps: float = 5,
        fractional_shares: bool = True,
    ) -> None:
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("cost parameters must be non-negative")
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.fractional_shares = fractional_shares
        self.filled_order_ids: set[str] = set()

    def execute(
        self, order: Order, portfolio: Portfolio, raw_open: float | None,
        execution_time: datetime,
    ) -> Fill | None:
        if order.order_id in self.filled_order_ids:
            raise ValueError("order already filled")
        if raw_open is None or raw_open <= 0:
            order.status = "rejected"
            order.reason = "missing valid execution-session open"
            return None
        fee_rate = self.commission_bps / 10_000
        slip_rate = self.slippage_bps / 10_000
        if order.target_weight == 1.0:
            if portfolio.quantity > Portfolio.tolerance:
                order.status = "noop"
                return None
            fill_price = raw_open * (1 + slip_rate)
            quantity = portfolio.cash / (fill_price * (1 + fee_rate))
            if not self.fractional_shares:
                quantity = float(int(quantity))
            notional = quantity * fill_price
            commission = notional * fee_rate
            portfolio.cash -= notional + commission
            portfolio.quantity += quantity
            portfolio.average_entry_price = fill_price
        elif order.target_weight == 0.0:
            if portfolio.quantity <= Portfolio.tolerance:
                order.status = "noop"
                return None
            fill_price = raw_open * (1 - slip_rate)
            quantity = -portfolio.quantity
            notional = abs(quantity) * fill_price
            commission = notional * fee_rate
            portfolio.cash += notional - commission
            portfolio.realized_pnl += (
                (fill_price - portfolio.average_entry_price) * abs(quantity) - commission
            )
            portfolio.quantity = 0.0
            portfolio.average_entry_price = 0.0
        else:
            raise ValueError("only target weights 0 and 1 are supported")
        if portfolio.cash < -Portfolio.tolerance:
            raise AssertionError("execution produced negative cash")
        portfolio.cash = max(0.0, portfolio.cash)
        portfolio.cumulative_cost += commission
        order.status = "filled"
        self.filled_order_ids.add(order.order_id)
        return Fill(
            fill_id=uuid4().hex, order_id=order.order_id, symbol=order.symbol,
            execution_time=execution_time, raw_open_price=float(raw_open),
            slippage_bps=self.slippage_bps, fill_price=fill_price, quantity=quantity,
            notional=notional, commission=commission, cash_after=portfolio.cash,
            position_after=portfolio.quantity,
        )
