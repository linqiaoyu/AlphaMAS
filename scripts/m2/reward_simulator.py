"""Deterministic M2 counterfactual reward simulation on matured market windows.

This module is deliberately separate from the Formal Trader runtime.  Execution
reuses the frozen M1 ``Broker`` and ``Portfolio`` primitives so BUY/SELL costs,
fractional shares, and no-op behaviour cannot drift from the backtester.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.execution import Broker
from tradingagents.backtesting.models import Action, Order
from tradingagents.backtesting.portfolio import Portfolio

MATURITY_SESSIONS = 5
FORMAL_COMMISSION_BPS = 5.0
FORMAL_SLIPPAGE_BPS = 5.0
REWARD_IDS = (
    "R1_ABSOLUTE_NET_LOG_RETURN",
    "R2_HOLD_RELATIVE_LOG_ADVANTAGE",
    "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY",
)


class RewardStatus(str, Enum):
    """Reward lifecycle states implemented by this framework."""

    PENDING = "PENDING"
    MATURED = "MATURED"


@dataclass(frozen=True)
class PortfolioState:
    """Single-symbol state carried into the execution-session open."""

    cash: float
    quantity: float
    average_entry_price: float = 0.0
    open_position_commission: float = 0.0


@dataclass(frozen=True)
class MarketBar:
    """Synthetic or supplied market values for one XNYS session."""

    session: str
    open_price: float
    close_price: float
    dividend_per_share: float = 0.0
    split_ratio: float = 0.0


@dataclass(frozen=True)
class RewardWindow:
    """Market observations available when a reward is queried.

    ``decision_close_price`` is audit-only.  The simulator never uses it in an
    economic reward, which mechanically excludes the pre-execution overnight
    move from action attribution.
    """

    symbol: str
    decision_session: str
    decision_close_price: float
    bars: tuple[MarketBar, ...]


@dataclass(frozen=True)
class CounterfactualRewardOutcome:
    """One matured selected-action outcome against the same HOLD baseline."""

    symbol: str
    decision_session: str
    execution_session: str
    maturity_session: str
    action: str
    pre_execution_equity: float
    post_execution_equity: float
    terminal_equity: float
    hold_terminal_equity: float
    action_equity_path: tuple[float, ...]
    hold_equity_path: tuple[float, ...]
    turnover_notional: float
    commission_cost: float
    slippage_cost: float
    total_cost: float
    action_was_noop: bool
    terminal_cash: float
    terminal_shares: float


@dataclass(frozen=True)
class RewardSimulation:
    """Maturity-gated result for all three counterfactual actions."""

    status: RewardStatus
    decision_session: str
    execution_session: str
    maturity_session: str
    outcomes: dict[str, CounterfactualRewardOutcome] | None = None


@dataclass(frozen=True)
class _PathResult:
    pre_execution_equity: float
    post_execution_equity: float
    terminal_equity: float
    equity_path: tuple[float, ...]
    turnover_notional: float
    commission_cost: float
    slippage_cost: float
    total_cost: float
    action_was_noop: bool
    terminal_cash: float
    terminal_shares: float


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validate_state(state: PortfolioState) -> None:
    cash = _finite(state.cash, "cash")
    quantity = _finite(state.quantity, "quantity")
    entry = _finite(state.average_entry_price, "average_entry_price")
    commission = _finite(
        state.open_position_commission, "open_position_commission"
    )
    if cash < 0 or quantity < 0:
        raise ValueError("counterfactual state must be long-only with non-negative cash")
    if entry < 0 or commission < 0:
        raise ValueError("entry price and open commission must be non-negative")
    if quantity > Portfolio.tolerance and entry <= 0:
        raise ValueError("a positive position requires a positive average entry price")


def _validate_bar(bar: MarketBar) -> None:
    if _finite(bar.open_price, "open_price") <= 0:
        raise ValueError("open_price must be positive")
    if _finite(bar.close_price, "close_price") <= 0:
        raise ValueError("close_price must be positive")
    _finite(bar.dividend_per_share, "dividend_per_share")
    split = _finite(bar.split_ratio, "split_ratio")
    if split < 0:
        raise ValueError("split_ratio must be zero or positive")


def _copy_portfolio(symbol: str, state: PortfolioState) -> Portfolio:
    _validate_state(state)
    seed_cash = state.cash + state.quantity * max(state.average_entry_price, 1.0)
    portfolio = Portfolio(symbol, initial_cash=max(seed_cash, 1.0))
    portfolio.cash = float(state.cash)
    portfolio.quantity = float(state.quantity)
    portfolio.average_entry_price = float(state.average_entry_price)
    portfolio.open_position_commission = float(state.open_position_commission)
    portfolio.peak_equity = portfolio.cash + (
        portfolio.quantity * portfolio.average_entry_price
    )
    return portfolio


def _apply_corporate_actions(portfolio: Portfolio, bar: MarketBar) -> None:
    if bar.dividend_per_share:
        portfolio.apply_dividend(bar.dividend_per_share)
    if bar.split_ratio:
        portfolio.apply_split(bar.split_ratio)


def validate_action(action: Action | str) -> Action:
    """Return a frozen action enum or reject values outside BUY/HOLD/SELL."""

    if isinstance(action, Action):
        return action
    try:
        return Action(action)
    except (TypeError, ValueError) as exc:
        raise ValueError("action must be exactly BUY, HOLD, or SELL") from exc


def _simulate_path(
    *,
    symbol: str,
    state: PortfolioState,
    action: Action | str,
    bars: tuple[MarketBar, ...],
    decision_session: str,
    commission_bps: float,
    slippage_bps: float,
    schedule: ExchangeSchedule,
) -> _PathResult:
    selected = validate_action(action)
    portfolio = _copy_portfolio(symbol, state)
    execution_bar = bars[0]
    _apply_corporate_actions(portfolio, execution_bar)
    pre_equity = portfolio.cash + portfolio.quantity * execution_bar.open_price
    if not math.isfinite(pre_equity) or pre_equity <= 0:
        raise ValueError("pre-execution equity must be finite and positive")

    turnover = commission = slippage = total_cost = 0.0
    action_was_noop = selected is Action.HOLD
    if selected is not Action.HOLD:
        broker = Broker(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            fractional_shares=True,
        )
        target = 1.0 if selected is Action.BUY else 0.0
        order = Order(
            order_id=f"m2-reward:{symbol}:{decision_session}:{selected.value}",
            symbol=symbol,
            created_at=schedule.session_close(decision_session).to_pydatetime(),
            intended_execution_session=execution_bar.session,
            target_weight=target,
            current_weight=portfolio.weight(execution_bar.open_price),
            side=selected.value,
        )
        fill = broker.execute(
            order,
            portfolio,
            execution_bar.open_price,
            schedule.session_open(execution_bar.session).to_pydatetime(),
        )
        action_was_noop = fill is None and order.status == "noop"
        if fill is not None:
            turnover = fill.notional
            commission = fill.commission
            slippage = fill.slippage_cost
            total_cost = fill.total_transaction_cost

    post_equity = portfolio.cash + portfolio.quantity * execution_bar.open_price
    equity_path = [post_equity]
    for index, bar in enumerate(bars):
        if index:
            _apply_corporate_actions(portfolio, bar)
        snapshot = portfolio.snapshot(
            bar.session,
            schedule.session_close(bar.session).to_pydatetime(),
            bar.close_price,
        )
        equity_path.append(snapshot.equity)

    if any(not math.isfinite(value) or value <= 0 for value in equity_path):
        raise ValueError("valid reward paths require finite positive net equity")
    return _PathResult(
        pre_execution_equity=pre_equity,
        post_execution_equity=post_equity,
        terminal_equity=equity_path[-1],
        equity_path=tuple(equity_path),
        turnover_notional=turnover,
        commission_cost=commission,
        slippage_cost=slippage,
        total_cost=total_cost,
        action_was_noop=action_was_noop,
        terminal_cash=portfolio.cash,
        terminal_shares=portfolio.quantity,
    )


def simulate_counterfactuals(
    window: RewardWindow,
    state: PortfolioState,
    *,
    commission_bps: float = FORMAL_COMMISSION_BPS,
    slippage_bps: float = FORMAL_SLIPPAGE_BPS,
    schedule: ExchangeSchedule | None = None,
) -> RewardSimulation:
    """Return BUY/HOLD/SELL outcomes, or PENDING until maturity is observable."""

    commission_bps = _finite(commission_bps, "commission_bps")
    slippage_bps = _finite(slippage_bps, "slippage_bps")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("transaction-cost parameters must be non-negative")
    if not window.symbol:
        raise ValueError("symbol must be non-empty")
    if _finite(window.decision_close_price, "decision_close_price") <= 0:
        raise ValueError("decision_close_price must be positive")
    _validate_state(state)

    calendar = schedule or ExchangeSchedule()
    decision = window.decision_session
    if not calendar.calendar.is_session(decision):
        raise ValueError("decision_session must be a valid XNYS session")
    required = calendar.calendar.sessions_window(
        decision, MATURITY_SESSIONS + 1
    )[1:].tz_localize(None)
    execution = required[0].date().isoformat()
    maturity = required[-1].date().isoformat()
    required_labels = tuple(item.date().isoformat() for item in required)
    required_set = set(required_labels)
    by_session: dict[str, MarketBar] = {}
    for bar in window.bars:
        # Post-maturity observations are outside the reward window and are not
        # read, validated, or allowed to influence the result.
        if bar.session not in required_set:
            continue
        _validate_bar(bar)
        if bar.session in by_session:
            raise ValueError(f"duplicate market bar for {bar.session}")
        by_session[bar.session] = bar
    if any(label not in by_session for label in required_labels):
        return RewardSimulation(
            status=RewardStatus.PENDING,
            decision_session=decision,
            execution_session=execution,
            maturity_session=maturity,
        )

    # Exact indexing makes appended post-maturity observations irrelevant.
    bars = tuple(by_session[label] for label in required_labels)
    paths = {
        action.value: _simulate_path(
            symbol=window.symbol,
            state=state,
            action=action,
            bars=bars,
            decision_session=decision,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            schedule=calendar,
        )
        for action in Action
    }
    hold = paths[Action.HOLD.value]
    outcomes = {
        action: CounterfactualRewardOutcome(
            symbol=window.symbol,
            decision_session=decision,
            execution_session=execution,
            maturity_session=maturity,
            action=action,
            pre_execution_equity=path.pre_execution_equity,
            post_execution_equity=path.post_execution_equity,
            terminal_equity=path.terminal_equity,
            hold_terminal_equity=hold.terminal_equity,
            action_equity_path=path.equity_path,
            hold_equity_path=hold.equity_path,
            turnover_notional=path.turnover_notional,
            commission_cost=path.commission_cost,
            slippage_cost=path.slippage_cost,
            total_cost=path.total_cost,
            action_was_noop=path.action_was_noop,
            terminal_cash=path.terminal_cash,
            terminal_shares=path.terminal_shares,
        )
        for action, path in paths.items()
    }
    return RewardSimulation(
        status=RewardStatus.MATURED,
        decision_session=decision,
        execution_session=execution,
        maturity_session=maturity,
        outcomes=outcomes,
    )


def maximum_drawdown(equity_path: tuple[float, ...]) -> float:
    """Return maximum peak-to-trough drawdown magnitude for a positive path."""

    if not equity_path:
        raise ValueError("equity_path must not be empty")
    peak = -math.inf
    maximum = 0.0
    for raw_value in equity_path:
        value = _finite(raw_value, "equity_path value")
        if value <= 0:
            raise ValueError("equity_path values must be positive")
        peak = max(peak, value)
        maximum = max(maximum, 1.0 - value / peak)
    return maximum


def drawdown_utility(equity_path: tuple[float, ...]) -> float:
    """Return the dimensionless ``-log(1-MDD)`` downside utility."""

    return -math.log1p(-maximum_drawdown(equity_path))


def candidate_rewards(outcome: CounterfactualRewardOutcome) -> dict[str, float]:
    """Calculate the three frozen raw economic rewards for one outcome."""

    values = (
        outcome.pre_execution_equity,
        outcome.terminal_equity,
        outcome.hold_terminal_equity,
    )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("reward equities must be finite and positive")
    r1 = math.log(outcome.terminal_equity / outcome.pre_execution_equity)
    r2 = math.log(outcome.terminal_equity / outcome.hold_terminal_equity)
    excess_drawdown = max(
        0.0,
        drawdown_utility(outcome.action_equity_path)
        - drawdown_utility(outcome.hold_equity_path),
    )
    rewards = {
        REWARD_IDS[0]: r1,
        REWARD_IDS[1]: r2,
        REWARD_IDS[2]: r2 - excess_drawdown,
    }
    if not all(math.isfinite(value) for value in rewards.values()):
        raise ValueError("candidate reward must be finite")
    return rewards


def _window(
    closes: tuple[float, ...], *, decision_close: float = 100.0
) -> RewardWindow:
    sessions = ("2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13")
    return RewardWindow(
        symbol="SYNTH",
        decision_session="2023-10-06",
        decision_close_price=decision_close,
        bars=tuple(
            MarketBar(session, 100.0 if index == 0 else close, close)
            for index, (session, close) in enumerate(zip(sessions, closes, strict=True))
        ),
    )


def _outcome_records(
    scenario_id: str,
    state_id: str,
    window: RewardWindow,
    state: PortfolioState,
    *,
    commission_bps: float = FORMAL_COMMISSION_BPS,
    slippage_bps: float = FORMAL_SLIPPAGE_BPS,
) -> tuple[RewardSimulation, list[dict[str, Any]]]:
    result = simulate_counterfactuals(
        window,
        state,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    assert result.status is RewardStatus.MATURED and result.outcomes is not None
    records = []
    for action in ("BUY", "HOLD", "SELL"):
        outcome = result.outcomes[action]
        rewards = candidate_rewards(outcome)
        records.append(
            {
                "action": action,
                "assertions": [],
                "commission_cost": outcome.commission_cost,
                "initial_portfolio_state": state_id,
                "pass": True,
                "rewards": {
                    "R1": rewards[REWARD_IDS[0]],
                    "R2": rewards[REWARD_IDS[1]],
                    "R3": rewards[REWARD_IDS[2]],
                },
                "scenario_id": scenario_id,
                "slippage_cost": outcome.slippage_cost,
                "terminal_equity": outcome.terminal_equity,
                "total_cost": outcome.total_cost,
            }
        )
    return result, records


def synthetic_sanity_payload() -> dict[str, Any]:
    """Build the deterministic, synthetic-only M2-03 sanity artifact."""

    cash = PortfolioState(100_000.0, 0.0)
    long = PortfolioState(0.0, 1_000.0, 100.0)
    records: list[dict[str, Any]] = []

    def add_assertion(
        scenario_records: list[dict[str, Any]], text: str, passed: bool
    ) -> None:
        for record in scenario_records:
            record["assertions"].append(text)
            record["pass"] = record["pass"] and bool(passed)

    flat_cash, rows = _outcome_records("S1_FLAT", "CASH", _window((100,) * 5), cash)
    assert flat_cash.outcomes is not None
    add_assertion(rows, "traded BUY is cost-negative versus HOLD", candidate_rewards(flat_cash.outcomes["BUY"])[REWARD_IDS[1]] < 0)
    add_assertion(rows, "HOLD incurs no new cost", flat_cash.outcomes["HOLD"].total_cost == 0)
    records.extend(rows)

    flat_long, rows = _outcome_records("S1_FLAT", "LONG", _window((100,) * 5), long)
    assert flat_long.outcomes is not None
    add_assertion(rows, "traded SELL is cost-negative versus HOLD", candidate_rewards(flat_long.outcomes["SELL"])[REWARD_IDS[1]] < 0)
    records.extend(rows)

    rising_cash, rows = _outcome_records("S2_MONOTONIC_RISE", "CASH", _window((102, 104, 106, 108, 110)), cash)
    assert rising_cash.outcomes is not None
    add_assertion(rows, "CASH BUY beats CASH HOLD", candidate_rewards(rising_cash.outcomes["BUY"])[REWARD_IDS[1]] > 0)
    records.extend(rows)

    rising_long, rows = _outcome_records("S2_MONOTONIC_RISE", "LONG", _window((102, 104, 106, 108, 110)), long)
    assert rising_long.outcomes is not None
    add_assertion(rows, "LONG SELL is inferior to LONG HOLD", candidate_rewards(rising_long.outcomes["SELL"])[REWARD_IDS[1]] < 0)
    records.extend(rows)

    falling_cash, rows = _outcome_records("S3_MONOTONIC_FALL", "CASH", _window((98, 96, 94, 92, 90)), cash)
    assert falling_cash.outcomes is not None
    add_assertion(rows, "CASH BUY is inferior to CASH HOLD", candidate_rewards(falling_cash.outcomes["BUY"])[REWARD_IDS[1]] < 0)
    records.extend(rows)

    falling_long, rows = _outcome_records("S3_MONOTONIC_FALL", "LONG", _window((98, 96, 94, 92, 90)), long)
    assert falling_long.outcomes is not None
    add_assertion(rows, "LONG SELL beats LONG HOLD", candidate_rewards(falling_long.outcomes["SELL"])[REWARD_IDS[1]] > 0)
    records.extend(rows)

    v_shape, rows = _outcome_records("S4_V_SHAPE", "CASH", _window((100, 85, 80, 90, 100)), cash)
    assert v_shape.outcomes is not None
    v_rewards = candidate_rewards(v_shape.outcomes["BUY"])
    add_assertion(rows, "R3 penalises BUY excess drawdown beyond R2", v_rewards[REWARD_IDS[2]] < v_rewards[REWARD_IDS[1]])
    records.extend(rows)

    inverted, rows = _outcome_records("S5_INVERTED_V", "CASH", _window((100, 115, 120, 110, 100)), cash)
    assert inverted.outcomes is not None
    inv_rewards = candidate_rewards(inverted.outcomes["BUY"])
    add_assertion(rows, "R3 detects terminal-return/interim-drawdown distinction", inv_rewards[REWARD_IDS[2]] < inv_rewards[REWARD_IDS[1]])
    records.extend(rows)

    gap_low, rows_low = _outcome_records("S6_OVERNIGHT_GAP_LOW", "CASH", _window((101, 102, 103, 104, 105), decision_close=50), cash)
    gap_high, rows_high = _outcome_records("S6_OVERNIGHT_GAP_HIGH", "CASH", _window((101, 102, 103, 104, 105), decision_close=200), cash)
    assert gap_low.outcomes is not None and gap_high.outcomes is not None
    gap_equal = all(
        candidate_rewards(gap_low.outcomes[action]) == candidate_rewards(gap_high.outcomes[action])
        for action in ("BUY", "HOLD", "SELL")
    )
    add_assertion(rows_low, "decision-close overnight gap is excluded", gap_equal)
    add_assertion(rows_high, "decision-close overnight gap is excluded", gap_equal)
    records.extend(rows_low + rows_high)

    base_cost, rows_base = _outcome_records("S7_FORMAL_COST", "CASH", _window((102, 104, 106, 108, 110)), cash)
    high_cost, rows_high_cost = _outcome_records(
        "S7_HIGH_COST_STRESS", "CASH", _window((102, 104, 106, 108, 110)), cash,
        commission_bps=100, slippage_bps=100,
    )
    assert base_cost.outcomes is not None and high_cost.outcomes is not None
    cost_monotonic = candidate_rewards(high_cost.outcomes["BUY"])[REWARD_IDS[0]] < candidate_rewards(base_cost.outcomes["BUY"])[REWARD_IDS[0]]
    add_assertion(rows_base, "higher costs cannot improve traded BUY reward", cost_monotonic)
    add_assertion(rows_high_cost, "higher costs cannot improve traded BUY reward", cost_monotonic)
    records.extend(rows_base + rows_high_cost)

    scale_rewards = []
    scale_rows: list[dict[str, Any]] = []
    for scale in (10_000.0, 100_000.0, 1_000_000.0):
        scaled, rows = _outcome_records(
            f"S8_SCALE_{int(scale)}", "CASH", _window((102, 104, 106, 108, 110)),
            PortfolioState(scale, 0.0),
        )
        assert scaled.outcomes is not None
        scale_rewards.append(candidate_rewards(scaled.outcomes["BUY"]))
        scale_rows.extend(rows)
    scale_equal = all(
        math.isclose(scale_rewards[0][reward_id], item[reward_id], rel_tol=1e-14, abs_tol=1e-14)
        for item in scale_rewards[1:]
        for reward_id in REWARD_IDS
    )
    add_assertion(scale_rows, "raw rewards are equity-scale invariant", scale_equal)
    records.extend(scale_rows)

    return {
        "all_passed": all(record["pass"] for record in records),
        "artifact_scope": "DETERMINISTIC_SYNTHETIC_SCENARIOS_ONLY",
        "leaderboard": False,
        "records": records,
        "schema_version": "1.0",
        "task_id": "M2-03",
    }


def write_synthetic_sanity(path: Path) -> None:
    payload = synthetic_sanity_payload()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-sanity-artifact", type=Path, required=True)
    args = parser.parse_args()
    write_synthetic_sanity(args.write_sanity_artifact)


if __name__ == "__main__":
    main()
