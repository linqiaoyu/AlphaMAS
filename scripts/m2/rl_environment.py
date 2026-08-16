"""Sequential M2 TRAIN environment built on the frozen execution primitives."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.m2.reward_simulator import (
    FORMAL_COMMISSION_BPS,
    FORMAL_SLIPPAGE_BPS,
    REWARD_IDS,
    MarketBar,
    PortfolioState,
    RewardStatus,
    RewardWindow,
    candidate_rewards,
    simulate_counterfactuals,
    validate_action,
)
from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.execution import Broker
from tradingagents.backtesting.models import Action, Order, PortfolioSnapshot
from tradingagents.backtesting.portfolio import Portfolio

TREE_CONTRACT = "M2_TRAIN_COUNTERFACTUAL_TREE_v1"
TRAIN_SYMBOLS = ("AAPL", "AMZN", "JPM", "JBSS", "EML", "AGI", "ARR", "AEMD")
TRAIN_SESSIONS = (
    "2023-05-05",
    "2023-05-12",
    "2023-05-19",
    "2023-05-26",
    "2023-06-02",
    "2023-06-09",
    "2023-06-16",
)
ACTIONS = tuple(action.value for action in Action)
INITIAL_CASH = 100_000.0
ECONOMIC_RTOL = 1e-12
ECONOMIC_ATOL = 1e-10


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class SequentialPortfolioState:
    """All path-dependent Portfolio fields needed to continue an episode exactly."""

    cash: float = INITIAL_CASH
    quantity: float = 0.0
    average_entry_price: float = 0.0
    open_position_commission: float = 0.0
    peak_equity: float = INITIAL_CASH
    realized_pnl: float = 0.0
    cumulative_commission_cost: float = 0.0
    cumulative_slippage_cost: float = 0.0
    cumulative_transaction_cost: float = 0.0
    cumulative_dividends: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("sequential portfolio state must be finite")
        if self.cash < -Portfolio.tolerance or self.quantity < -Portfolio.tolerance:
            raise ValueError("sequential portfolio state must remain long-only")
        if self.average_entry_price < 0 or self.open_position_commission < 0:
            raise ValueError("entry basis must be non-negative")
        if self.quantity > Portfolio.tolerance and self.average_entry_price <= 0:
            raise ValueError("positive quantity requires a positive entry basis")
        if self.peak_equity <= 0:
            raise ValueError("peak equity must be positive")

    def reward_state(self) -> PortfolioState:
        return PortfolioState(
            cash=self.cash,
            quantity=self.quantity,
            average_entry_price=self.average_entry_price,
            open_position_commission=self.open_position_commission,
        )


@dataclass(frozen=True)
class TransitionResult:
    action: str
    reward_r3: float
    next_state: SequentialPortfolioState
    next_snapshot: PortfolioSnapshot
    terminal_equity: float
    terminal_cash: float
    terminal_quantity: float
    transaction_cost: float
    commission_cost: float
    slippage_cost: float
    action_was_noop: bool
    execution_session: str
    maturity_session: str


def _seed_portfolio(symbol: str, state: SequentialPortfolioState) -> Portfolio:
    portfolio = Portfolio(symbol, initial_cash=max(state.peak_equity, 1.0))
    portfolio.cash = state.cash
    portfolio.quantity = state.quantity
    portfolio.average_entry_price = state.average_entry_price
    portfolio.open_position_commission = state.open_position_commission
    portfolio.peak_equity = state.peak_equity
    portfolio.realized_pnl = state.realized_pnl
    portfolio.cumulative_commission_cost = state.cumulative_commission_cost
    portfolio.cumulative_slippage_cost = state.cumulative_slippage_cost
    portfolio.cumulative_transaction_cost = state.cumulative_transaction_cost
    portfolio.cumulative_dividends = state.cumulative_dividends
    return portfolio


def _state_from_portfolio(portfolio: Portfolio) -> SequentialPortfolioState:
    return SequentialPortfolioState(
        cash=portfolio.cash,
        quantity=portfolio.quantity,
        average_entry_price=portfolio.average_entry_price,
        open_position_commission=portfolio.open_position_commission,
        peak_equity=portfolio.peak_equity,
        realized_pnl=portfolio.realized_pnl,
        cumulative_commission_cost=portfolio.cumulative_commission_cost,
        cumulative_slippage_cost=portfolio.cumulative_slippage_cost,
        cumulative_transaction_cost=portfolio.cumulative_transaction_cost,
        cumulative_dividends=portfolio.cumulative_dividends,
    )


def _apply_corporate_actions(portfolio: Portfolio, bar: MarketBar) -> None:
    if bar.dividend_per_share:
        portfolio.apply_dividend(bar.dividend_per_share)
    if bar.split_ratio:
        portfolio.apply_split(bar.split_ratio)


def initial_snapshot(
    symbol: str,
    decision_session: str,
    decision_close: float,
    *,
    schedule: ExchangeSchedule | None = None,
) -> PortfolioSnapshot:
    calendar = schedule or ExchangeSchedule()
    portfolio = Portfolio(symbol, initial_cash=INITIAL_CASH)
    return portfolio.snapshot(
        decision_session,
        calendar.session_close(decision_session).to_pydatetime(),
        decision_close,
    )


def simulate_transition(
    window: RewardWindow,
    state: SequentialPortfolioState,
    action: Action | str,
    *,
    schedule: ExchangeSchedule | None = None,
) -> TransitionResult:
    """Advance one five-session window and prove equivalence to frozen R3 simulation."""

    selected = validate_action(action)
    calendar = schedule or ExchangeSchedule()
    reward_simulation = simulate_counterfactuals(window, state.reward_state(), schedule=calendar)
    if reward_simulation.status is not RewardStatus.MATURED or reward_simulation.outcomes is None:
        raise ValueError("transition requires a fully matured five-session reward window")
    outcome = reward_simulation.outcomes[selected.value]
    required = calendar.calendar.sessions_window(window.decision_session, 6)[1:].tz_localize(None)
    labels = tuple(item.date().isoformat() for item in required)
    bars_by_session = {bar.session: bar for bar in window.bars if bar.session in labels}
    if tuple(bars_by_session) != labels:
        bars_by_session = {label: bars_by_session[label] for label in labels}

    portfolio = _seed_portfolio(window.symbol, state)
    execution_bar = bars_by_session[labels[0]]
    _apply_corporate_actions(portfolio, execution_bar)
    fill = None
    action_was_noop = selected is Action.HOLD
    if selected is not Action.HOLD:
        broker = Broker(
            commission_bps=FORMAL_COMMISSION_BPS,
            slippage_bps=FORMAL_SLIPPAGE_BPS,
            fractional_shares=True,
        )
        order = Order(
            order_id=f"m2-rl:{window.symbol}:{window.decision_session}:{selected.value}",
            symbol=window.symbol,
            created_at=calendar.session_close(window.decision_session).to_pydatetime(),
            intended_execution_session=execution_bar.session,
            target_weight=1.0 if selected is Action.BUY else 0.0,
            current_weight=portfolio.weight(execution_bar.open_price),
            side=selected.value,
        )
        fill = broker.execute(
            order,
            portfolio,
            execution_bar.open_price,
            calendar.session_open(execution_bar.session).to_pydatetime(),
        )
        action_was_noop = fill is None and order.status == "noop"

    snapshot: PortfolioSnapshot | None = None
    for index, label in enumerate(labels):
        bar = bars_by_session[label]
        if index:
            _apply_corporate_actions(portfolio, bar)
        snapshot = portfolio.snapshot(
            label,
            calendar.session_close(label).to_pydatetime(),
            bar.close_price,
        )
    assert snapshot is not None
    transaction_cost = 0.0 if fill is None else fill.total_transaction_cost
    commission = 0.0 if fill is None else fill.commission
    slippage = 0.0 if fill is None else fill.slippage_cost
    comparisons = (
        (snapshot.equity, outcome.terminal_equity, "terminal equity"),
        (portfolio.cash, outcome.terminal_cash, "terminal cash"),
        (portfolio.quantity, outcome.terminal_shares, "terminal quantity"),
        (transaction_cost, outcome.total_cost, "transaction cost"),
    )
    for actual, expected, name in comparisons:
        if not math.isclose(actual, expected, rel_tol=ECONOMIC_RTOL, abs_tol=ECONOMIC_ATOL):
            raise AssertionError(f"transition/reward {name} mismatch: {actual} != {expected}")
    if action_was_noop != outcome.action_was_noop:
        raise AssertionError("transition/reward no-op status mismatch")
    reward = candidate_rewards(outcome)[REWARD_IDS[2]]
    if not math.isfinite(reward):
        raise AssertionError("R3 must be finite")
    return TransitionResult(
        action=selected.value,
        reward_r3=reward,
        next_state=_state_from_portfolio(portfolio),
        next_snapshot=snapshot,
        terminal_equity=snapshot.equity,
        terminal_cash=portfolio.cash,
        terminal_quantity=portfolio.quantity,
        transaction_cost=transaction_cost,
        commission_cost=commission,
        slippage_cost=slippage,
        action_was_noop=action_was_noop,
        execution_session=labels[0],
        maturity_session=labels[-1],
    )


def load_market_snapshot(path: Path, symbol: str) -> dict[str, MarketBar]:
    """Read a frozen per-symbol snapshot without consulting any external data."""

    rows: dict[str, MarketBar] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            session = row["Date"]
            if session in rows:
                raise ValueError(f"duplicate market session {session}")
            rows[session] = MarketBar(
                session=session,
                open_price=float(row["Open"]),
                close_price=float(row["Close"]),
                dividend_per_share=float(row["Dividends"]),
                split_ratio=float(row["Stock Splits"]),
            )
    if not rows:
        raise ValueError(f"empty market snapshot for {symbol}")
    return rows


def reward_window(
    symbol: str,
    decision_session: str,
    market: dict[str, MarketBar],
    *,
    schedule: ExchangeSchedule | None = None,
) -> RewardWindow:
    calendar = schedule or ExchangeSchedule()
    required = calendar.calendar.sessions_window(decision_session, 6).tz_localize(None)
    labels = tuple(item.date().isoformat() for item in required)
    missing = [label for label in labels if label not in market]
    if missing:
        raise ValueError(f"frozen market snapshot lacks required sessions: {missing}")
    return RewardWindow(
        symbol=symbol,
        decision_session=decision_session,
        decision_close_price=market[decision_session].close_price,
        bars=tuple(market[label] for label in labels[1:]),
    )
