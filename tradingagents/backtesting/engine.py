"""Daily event loop with weekly close decisions and next-open execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradingagents.backtesting.calendar import ExchangeSchedule, WeeklyEvent
from tradingagents.backtesting.data import MarketDataProvider, normalize_ohlcv
from tradingagents.backtesting.execution import Broker
from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.models import DecisionStatus, Order
from tradingagents.backtesting.portfolio import Portfolio
from tradingagents.backtesting.strategies import Strategy


@dataclass
class BacktestResult:
    symbol: str
    decisions: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    daily_equity: pd.DataFrame
    corporate_action_events: pd.DataFrame
    metrics: dict[str, Any]


class WeeklyBacktestEngine:
    def __init__(
        self, *, data_provider: MarketDataProvider, schedule: ExchangeSchedule | None = None,
        initial_cash: float = 100_000, commission_bps: float = 5,
        slippage_bps: float = 5, fractional_shares: bool = True,
        risk_free_rate: float = 0.0, annualization: int = 252,
    ) -> None:
        self.data_provider = data_provider
        self.schedule = schedule or ExchangeSchedule()
        self.initial_cash = initial_cash
        self.risk_free_rate = risk_free_rate
        self.annualization = annualization
        self.broker_args = {
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "fractional_shares": fractional_shares,
        }

    def run(
        self, *, symbol: str, first_week: str, final_week: str,
        final_valuation_session: str, strategy: Strategy, experiment_id: str,
        warmup_start: str | None = None, max_decisions: int | None = None,
    ) -> BacktestResult:
        events = self.schedule.weekly_events(first_week, final_week)
        valuation_sessions = self.schedule.sessions(
            events[0].decision_session, final_valuation_session,
        )
        data_start = warmup_start or self.schedule.preceding_sessions(
            events[0].decision_session, 252,
        )[0].date().isoformat()
        data = normalize_ohlcv(self.data_provider.load(symbol, data_start, final_valuation_session))
        missing = [session.date().isoformat() for session in valuation_sessions if session.tz_localize(None) not in data.index]
        if missing:
            raise ValueError(f"missing market sessions for {symbol}: {missing[:5]}")
        event_by_decision = {event.decision_session: event for event in events}
        portfolio = Portfolio(symbol, self.initial_cash)
        broker = Broker(**self.broker_args)
        pending: Order | None = None
        decision_rows: list[dict[str, Any]] = []
        order_rows: list[dict[str, Any]] = []
        fill_rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []
        snapshot_rows: list[dict[str, Any]] = []

        for session_ts in valuation_sessions:
            session = session_ts.date().isoformat()
            bar = data.loc[session_ts.tz_localize(None)]
            # 1. Corporate actions at the open, before new orders. Eligibility is
            # determined by the position carried from the previous close.
            dividend = float(bar["Dividends"])
            if dividend:
                eligible = portfolio.quantity
                cash_effect = portfolio.apply_dividend(dividend)
                action_rows.append({
                    "session": session, "symbol": symbol, "type": "dividend",
                    "value": dividend, "quantity_eligible": eligible,
                    "cash_effect": cash_effect,
                })
            split = float(bar["Stock Splits"])
            if split:
                eligible = portfolio.quantity
                portfolio.apply_split(split)
                action_rows.append({
                    "session": session, "symbol": symbol, "type": "split",
                    "value": split, "quantity_eligible": eligible, "cash_effect": 0.0,
                })

            # 2. Market open: only the immediately intended pending order can execute.
            if pending is not None:
                if pending.intended_execution_session != session:
                    pending.status = "rejected"
                    pending.reason = "pending order missed its intended session"
                    order_rows[-1] = pending.to_dict()
                    pending = None
                else:
                    raw_open = None if pd.isna(bar["Open"]) else float(bar["Open"])
                    fill = broker.execute(pending, portfolio, raw_open, self.schedule.session_open(session))
                    order_rows[-1] = pending.to_dict()
                    if fill:
                        fill_rows.append(fill.to_dict())
                    pending = None

            # 3. Market close valuation.
            close_time = self.schedule.session_close(session)
            snapshot = portfolio.snapshot(session, close_time.to_pydatetime(), float(bar["Close"]))
            snapshot_rows.append(snapshot.to_dict())

            # 4. Weekly decision at the actual close; data is sliced point-in-time.
            event: WeeklyEvent | None = event_by_decision.get(session)
            if event:
                if max_decisions is not None and len(decision_rows) >= max_decisions:
                    continue
                visible = data.loc[:session_ts.tz_localize(None)].copy()
                decision = strategy.decide(
                    symbol=symbol, decision_session=session,
                    decision_time=event.decision_close_utc.to_pydatetime(),
                    market_history=visible, portfolio_snapshot=snapshot,
                    context={"experiment_id": experiment_id, "point_in_time": True},
                )
                decision_rows.append(decision.to_dict())
                if decision.status is DecisionStatus.FAILED:
                    continue
                assert decision.target_weight is not None
                current_target = 1.0 if portfolio.quantity > Portfolio.tolerance else 0.0
                if decision.target_weight == current_target:
                    decision_rows[-1]["rebalance_status"] = "noop"
                    continue
                pending = Order(
                    order_id=hashlib.sha256(
                        f"{experiment_id}:{symbol}:{session}:{decision.target_weight}".encode()
                    ).hexdigest()[:32], symbol=symbol,
                    created_at=event.decision_close_utc.to_pydatetime(),
                    intended_execution_session=event.execution_session,
                    target_weight=decision.target_weight,
                    current_weight=snapshot.current_weight,
                    side="BUY" if decision.target_weight == 1.0 else "SELL",
                )
                order_rows.append(pending.to_dict())
                decision_rows[-1]["rebalance_status"] = "ordered"

        decisions = pd.DataFrame(decision_rows)
        orders = pd.DataFrame(order_rows)
        fills = pd.DataFrame(fill_rows)
        daily = pd.DataFrame(snapshot_rows)
        actions = pd.DataFrame(action_rows, columns=(
            "session", "symbol", "type", "value", "quantity_eligible", "cash_effect",
        ))
        failure_count = int(
            (decisions["status"] == "failed").sum() if "status" in decisions else 0
        )
        metrics = compute_metrics(
            daily["equity"], fills=fills, exposure=daily["current_weight"],
            positions=daily["quantity"], decision_count=len(decisions),
            decision_failure_count=failure_count, decisions=decisions, orders=orders,
            initial_equity=self.initial_cash,
            cumulative_dividends=portfolio.cumulative_dividends,
            risk_free_rate=self.risk_free_rate, annualization=self.annualization,
        )
        return BacktestResult(symbol, decisions, orders, fills, daily, actions, metrics)
