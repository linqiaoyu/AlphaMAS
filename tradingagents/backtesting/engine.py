"""Daily event loop with weekly close decisions and next-open execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradingagents.backtesting.calendar import ExchangeSchedule, WeeklyEvent
from tradingagents.backtesting.data import MarketDataProvider, normalize_ohlcv
from tradingagents.backtesting.execution import Broker
from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.models import DecisionStatus, Order, StrategyDecision
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


class ChronologicalDecisionFailure(RuntimeError):
    """Abort an Agent attempt at its first unsuccessful chronological case."""

    def __init__(self, decision: StrategyDecision) -> None:
        self.decision = decision
        super().__init__(
            "TradingAgents decision failed at "
            f"{decision.symbol}:{decision.decision_session}: {decision.reason}"
        )


def _market_history_visibility(
    history: pd.DataFrame, schedule: ExchangeSchedule,
) -> dict[str, Any]:
    """Describe the exact daily market-history frame passed to a strategy."""
    if history.empty:
        return {
            "first_session": None,
            "last_session": None,
            "last_observation_time_utc": None,
            "row_count": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    first_session = pd.Timestamp(history.index.min()).date().isoformat()
    last_session = pd.Timestamp(history.index.max()).date().isoformat()
    payload = history.to_csv(lineterminator="\n", float_format="%.12g")
    return {
        "first_session": first_session,
        "last_session": last_session,
        "last_observation_time_utc": schedule.session_close(last_session).isoformat(),
        "row_count": len(history),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def market_history_visibility_errors(
    results: Mapping[str, BacktestResult], schedule: ExchangeSchedule,
) -> list[str]:
    """Return errors when a recorded strategy input extends past its decision time."""
    errors: list[str] = []
    for result_symbol, result in results.items():
        for row_number, decision in result.decisions.iterrows():
            symbol = str(decision.get("symbol") or result_symbol)
            session = str(decision.get("decision_session", ""))
            case = f"{symbol}:{session or row_number}"
            metadata = decision.get("metadata")
            visibility = (
                metadata.get("market_history_visibility")
                if isinstance(metadata, Mapping) else None
            )
            if not isinstance(visibility, Mapping):
                errors.append(f"{case}: missing market_history_visibility audit")
                continue
            try:
                first_value = pd.Timestamp(visibility["first_session"])
                last_value = pd.Timestamp(visibility["last_session"])
                decision_session_value = pd.Timestamp(session)
                last_observation = pd.Timestamp(
                    visibility["last_observation_time_utc"]
                )
                decision_time = pd.Timestamp(decision["decision_time_utc"])
                if any(pd.isna(value) for value in (
                    first_value,
                    last_value,
                    decision_session_value,
                    last_observation,
                    decision_time,
                )):
                    raise ValueError("market-history audit times cannot be null")
                first_session = pd.Timestamp(first_value.date())
                last_session = pd.Timestamp(last_value.date())
                decision_session = pd.Timestamp(decision_session_value.date())
                expected_last_observation = schedule.session_close(
                    last_session.date().isoformat()
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{case}: invalid market-history time audit ({exc})")
                continue
            if last_observation.tzinfo is None or decision_time.tzinfo is None:
                errors.append(f"{case}: market-history and decision times must be timezone-aware")
            elif last_observation != expected_last_observation:
                errors.append(
                    f"{case}: last observation time is not its XNYS session close"
                )
            elif last_observation > decision_time:
                errors.append(f"{case}: market history extends beyond decision time")
            if first_session > last_session:
                errors.append(f"{case}: first market-history session is after the last")
            if last_session > decision_session:
                errors.append(f"{case}: market history extends beyond decision session")
            row_count = visibility.get("row_count")
            if (
                not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count <= 0
            ):
                errors.append(f"{case}: market-history row_count must be positive")
            history_hash = visibility.get("sha256")
            if not isinstance(history_hash, str) or len(history_hash) != 64:
                errors.append(f"{case}: market-history sha256 is invalid")
            else:
                try:
                    int(history_hash, 16)
                except ValueError:
                    errors.append(f"{case}: market-history sha256 is invalid")
    return errors


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
        event_index = {event.decision_session: index for index, event in enumerate(events)}
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
                visibility = _market_history_visibility(visible, self.schedule)
                decision = strategy.decide(
                    symbol=symbol, decision_session=session,
                    decision_time=event.decision_close_utc.to_pydatetime(),
                    market_history=visible, portfolio_snapshot=snapshot,
                    context={
                        "experiment_id": experiment_id,
                        "point_in_time": True,
                        "execution_session": event.execution_session,
                        "market_history_visibility": dict(visibility),
                        "portfolio_reward_state": {
                            "cash": portfolio.cash,
                            "quantity": portfolio.quantity,
                            "average_entry_price": portfolio.average_entry_price,
                            "open_position_commission": portfolio.open_position_commission,
                        },
                        "next_decision_session": (
                            events[event_index[session] + 1].decision_session
                            if event_index[session] + 1 < len(events)
                            and (
                                max_decisions is None
                                or event_index[session] + 1 < max_decisions
                            )
                            else None
                        ),
                    },
                )
                # This is engine-owned provenance: overwrite any strategy-provided
                # value with the audit of the frame that was actually passed above.
                decision.metadata["market_history_visibility"] = visibility
                decision_rows.append(decision.to_dict())
                if decision.status is DecisionStatus.FAILED:
                    if getattr(strategy, "fail_fast_on_decision_failure", False):
                        raise ChronologicalDecisionFailure(decision)
                    continue
                record_success = getattr(
                    strategy, "record_chronological_success", None
                )
                if callable(record_success):
                    record_success(decision)
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

        finalize = getattr(strategy, "finalize_symbol", None)
        if callable(finalize):
            finalize(
                symbol=symbol,
                final_session=final_valuation_session,
                market_history=data.loc[:pd.Timestamp(final_valuation_session)].copy(),
            )

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
        model_failure_count = int(sum(
            bool(metadata.get("model_failure"))
            for metadata in decisions.get("metadata", pd.Series(dtype=object))
            if isinstance(metadata, Mapping)
        ))
        metrics.update({
            "model_failure_count": model_failure_count,
            "model_failure_rate": model_failure_count / len(decisions) if len(decisions) else 0.0,
        })
        return BacktestResult(symbol, decisions, orders, fills, daily, actions, metrics)
