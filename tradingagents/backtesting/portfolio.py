"""Single-symbol, long-only portfolio accounting."""

from __future__ import annotations

from datetime import datetime

from tradingagents.backtesting.models import PortfolioSnapshot


class Portfolio:
    tolerance = 1e-8

    def __init__(self, symbol: str, initial_cash: float = 100_000.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.symbol = symbol
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.quantity = 0.0
        self.average_entry_price = 0.0
        self.open_position_commission = 0.0
        self.realized_pnl = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_dividends = 0.0
        self.peak_equity = float(initial_cash)

    def apply_dividend(self, dividend_per_share: float) -> float:
        cash_effect = self.quantity * float(dividend_per_share)
        self.cash += cash_effect
        self.cumulative_dividends += cash_effect
        return cash_effect

    def apply_split(self, ratio: float) -> None:
        if ratio <= 0:
            raise ValueError("split ratio must be positive")
        if self.quantity > self.tolerance:
            self.quantity *= ratio
            self.average_entry_price /= ratio

    def weight(self, mark_price: float) -> float:
        equity = self.cash + self.quantity * mark_price
        return 0.0 if equity <= 0 else self.quantity * mark_price / equity

    def snapshot(
        self, session: str, timestamp: datetime, close_price: float,
    ) -> PortfolioSnapshot:
        if close_price <= 0:
            raise ValueError("close price must be positive")
        market_value = self.quantity * close_price
        equity = self.cash + market_value
        if self.cash < -self.tolerance or self.quantity < -self.tolerance:
            raise AssertionError("long-only portfolio invariant violated")
        self.cash = max(0.0, self.cash)
        self.quantity = max(0.0, self.quantity)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = equity / self.peak_equity - 1.0
        unrealized = (
            self.quantity * (close_price - self.average_entry_price)
            - self.open_position_commission
            if self.quantity else 0.0
        )
        return PortfolioSnapshot(
            timestamp=timestamp, session=session, symbol=self.symbol,
            cash=self.cash, quantity=self.quantity, close_price=close_price,
            market_value=market_value, equity=equity,
            current_weight=0.0 if equity <= 0 else market_value / equity,
            average_entry_price=self.average_entry_price,
            unrealized_pnl=unrealized, realized_pnl=self.realized_pnl,
            cumulative_cost=self.cumulative_cost,
            cumulative_dividends=self.cumulative_dividends, current_drawdown=drawdown,
            peak_equity=self.peak_equity,
        )
