"""PIT-safe weekly M2 state transitions and independent delayed R3 credit."""

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

TREE_CONTRACT = "M2_TRAIN_COUNTERFACTUAL_TREE-v2"
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@dataclass(frozen=True)
class SequentialPortfolioState:
    """All path-dependent fields required to continue a weekly episode exactly."""

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
        if not all(math.isfinite(float(value)) for value in asdict(self).values()):
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
class WeeklyTransitionWindow:
    """Bars permitted to construct one next-week decision state."""

    symbol: str
    decision_session: str
    child_decision_session: str
    bars: tuple[MarketBar, ...]


@dataclass(frozen=True)
class WeeklyTransitionResult:
    action: str
    next_state: SequentialPortfolioState
    next_snapshot: PortfolioSnapshot
    next_decision_session: str
    next_decision_equity: float
    transaction_cost: float
    commission_cost: float
    slippage_cost: float
    action_was_noop: bool
    execution_session: str
    state_transition_input_sha256: str
    state_transition_sha256: str


@dataclass(frozen=True)
class LocalCreditResult:
    decision_session: str
    execution_session: str
    reward_maturity_session: str
    rewards_r3: dict[str, float]
    input_sha256_by_action: dict[str, str]
    terminal_by_action: dict[str, dict[str, float | bool]]


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


def _bar_payload(bar: MarketBar) -> dict[str, float | str]:
    return {
        "session": bar.session,
        "open_price": bar.open_price,
        "close_price": bar.close_price,
        "dividend_per_share": bar.dividend_per_share,
        "split_ratio": bar.split_ratio,
    }


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


def simulate_weekly_transition(
    window: WeeklyTransitionWindow,
    state: SequentialPortfolioState,
    action: Action | str,
    *,
    schedule: ExchangeSchedule | None = None,
) -> WeeklyTransitionResult:
    """Execute next open and stop exactly at the next weekly decision close."""

    selected = validate_action(action)
    calendar = schedule or ExchangeSchedule()
    expected_sessions = calendar.sessions(
        calendar.next_session(window.decision_session), window.child_decision_session
    ).tz_localize(None)
    labels = tuple(item.date().isoformat() for item in expected_sessions)
    if not labels or labels[-1] != window.child_decision_session:
        raise ValueError("child decision must be a later valid XNYS session")
    if tuple(bar.session for bar in window.bars) != labels:
        raise ValueError("weekly transition window must contain only execution-through-child bars")
    input_payload = {
        "symbol": window.symbol,
        "decision_session": window.decision_session,
        "child_decision_session": window.child_decision_session,
        "action": selected.value,
        "state": asdict(state),
        "bars": [_bar_payload(bar) for bar in window.bars],
    }
    input_sha = sha256_bytes(canonical_json_bytes(input_payload))

    portfolio = _seed_portfolio(window.symbol, state)
    execution_bar = window.bars[0]
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
            order_id=f"m2-rl-v2:{window.symbol}:{window.decision_session}:{selected.value}",
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
    for index, bar in enumerate(window.bars):
        if index:
            _apply_corporate_actions(portfolio, bar)
        snapshot = portfolio.snapshot(
            bar.session,
            calendar.session_close(bar.session).to_pydatetime(),
            bar.close_price,
        )
    assert snapshot is not None
    if snapshot.session != window.child_decision_session:
        raise AssertionError("weekly transition consumed information after child close")
    next_state = _state_from_portfolio(portfolio)
    transaction_cost = 0.0 if fill is None else fill.total_transaction_cost
    commission = 0.0 if fill is None else fill.commission
    slippage = 0.0 if fill is None else fill.slippage_cost
    output_payload = {
        "input_sha256": input_sha,
        "next_state": asdict(next_state),
        "next_snapshot": snapshot.to_dict(),
        "transaction_cost": transaction_cost,
        "action_was_noop": action_was_noop,
    }
    return WeeklyTransitionResult(
        action=selected.value,
        next_state=next_state,
        next_snapshot=snapshot,
        next_decision_session=window.child_decision_session,
        next_decision_equity=snapshot.equity,
        transaction_cost=transaction_cost,
        commission_cost=commission,
        slippage_cost=slippage,
        action_was_noop=action_was_noop,
        execution_session=execution_bar.session,
        state_transition_input_sha256=input_sha,
        state_transition_sha256=sha256_bytes(canonical_json_bytes(output_payload)),
    )


def simulate_local_credit(
    window: RewardWindow,
    state: SequentialPortfolioState,
    *,
    schedule: ExchangeSchedule | None = None,
) -> LocalCreditResult:
    """Evaluate all three local R3 probes without advancing the weekly portfolio."""

    calendar = schedule or ExchangeSchedule()
    simulation = simulate_counterfactuals(window, state.reward_state(), schedule=calendar)
    if simulation.status is not RewardStatus.MATURED or simulation.outcomes is None:
        raise ValueError("local credit requires a fully matured five-session window")
    window_payload = {
        "symbol": window.symbol,
        "decision_session": window.decision_session,
        "decision_close_price": window.decision_close_price,
        "state": asdict(state.reward_state()),
        "bars": [_bar_payload(bar) for bar in window.bars],
        "reward_id": REWARD_IDS[2],
    }
    rewards: dict[str, float] = {}
    input_hashes: dict[str, str] = {}
    terminals: dict[str, dict[str, float | bool]] = {}
    for action in ACTIONS:
        outcome = simulation.outcomes[action]
        reward = candidate_rewards(outcome)[REWARD_IDS[2]]
        if not math.isfinite(reward):
            raise AssertionError("local R3 must be finite")
        rewards[action] = reward
        input_hashes[action] = sha256_bytes(
            canonical_json_bytes({**window_payload, "action": action})
        )
        terminals[action] = {
            "terminal_equity": outcome.terminal_equity,
            "terminal_cash": outcome.terminal_cash,
            "terminal_quantity": outcome.terminal_shares,
            "transaction_cost": outcome.total_cost,
            "action_was_noop": outcome.action_was_noop,
        }
    return LocalCreditResult(
        decision_session=window.decision_session,
        execution_session=simulation.execution_session,
        reward_maturity_session=simulation.maturity_session,
        rewards_r3=rewards,
        input_sha256_by_action=input_hashes,
        terminal_by_action=terminals,
    )


def load_market_snapshot(path: Path, symbol: str) -> dict[str, MarketBar]:
    """Read a frozen per-symbol snapshot without consulting external data."""

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


def weekly_transition_window(
    symbol: str,
    decision_session: str,
    child_decision_session: str,
    market: dict[str, MarketBar],
    *,
    schedule: ExchangeSchedule | None = None,
) -> WeeklyTransitionWindow:
    calendar = schedule or ExchangeSchedule()
    sessions = calendar.sessions(
        calendar.next_session(decision_session), child_decision_session
    ).tz_localize(None)
    labels = tuple(item.date().isoformat() for item in sessions)
    missing = [label for label in labels if label not in market]
    if missing:
        raise ValueError(f"frozen market snapshot lacks transition sessions: {missing}")
    return WeeklyTransitionWindow(
        symbol=symbol,
        decision_session=decision_session,
        child_decision_session=child_decision_session,
        bars=tuple(market[label] for label in labels),
    )


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
        raise ValueError(f"frozen market snapshot lacks credit sessions: {missing}")
    return RewardWindow(
        symbol=symbol,
        decision_session=decision_session,
        decision_close_price=market[decision_session].close_price,
        bars=tuple(market[label] for label in labels[1:]),
    )
