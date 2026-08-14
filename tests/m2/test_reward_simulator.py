from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.m2.reward_simulator import (
    FORMAL_COMMISSION_BPS,
    FORMAL_SLIPPAGE_BPS,
    REWARD_IDS,
    CounterfactualRewardOutcome,
    MarketBar,
    PortfolioState,
    RewardStatus,
    RewardWindow,
    candidate_rewards,
    drawdown_utility,
    maximum_drawdown,
    simulate_counterfactuals,
    synthetic_sanity_payload,
    validate_action,
)
from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.execution import Broker
from tradingagents.backtesting.models import Action, Order
from tradingagents.backtesting.portfolio import Portfolio

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/m2"
SESSIONS = (
    "2023-10-09",
    "2023-10-10",
    "2023-10-11",
    "2023-10-12",
    "2023-10-13",
)


def _window(
    closes: tuple[float, ...],
    *,
    decision_close: float = 100.0,
    extra_bars: tuple[MarketBar, ...] = (),
) -> RewardWindow:
    return RewardWindow(
        symbol="SYNTH",
        decision_session="2023-10-06",
        decision_close_price=decision_close,
        bars=tuple(
            MarketBar(
                session=session,
                open_price=100.0 if index == 0 else close,
                close_price=close,
            )
            for index, (session, close) in enumerate(
                zip(SESSIONS, closes, strict=True)
            )
        )
        + extra_bars,
    )


def _matured(
    closes: tuple[float, ...],
    state: PortfolioState,
    **kwargs,
):
    result = simulate_counterfactuals(_window(closes), state, **kwargs)
    assert result.status is RewardStatus.MATURED
    assert result.outcomes is not None
    return result.outcomes


def _reward(outcome: CounterfactualRewardOutcome, reward_id: str) -> float:
    return candidate_rewards(outcome)[reward_id]


def test_r1_formula_exactness() -> None:
    outcome = _matured((102, 104, 106, 108, 110), PortfolioState(100_000, 0))["BUY"]
    assert _reward(outcome, REWARD_IDS[0]) == math.log(
        outcome.terminal_equity / outcome.pre_execution_equity
    )


def test_r2_formula_exactness() -> None:
    outcome = _matured((102, 104, 106, 108, 110), PortfolioState(100_000, 0))["BUY"]
    assert _reward(outcome, REWARD_IDS[1]) == math.log(
        outcome.terminal_equity / outcome.hold_terminal_equity
    )


def test_r3_formula_exactness() -> None:
    outcome = _matured((100, 85, 80, 90, 100), PortfolioState(100_000, 0))["BUY"]
    expected = _reward(outcome, REWARD_IDS[1]) - max(
        0.0,
        drawdown_utility(outcome.action_equity_path)
        - drawdown_utility(outcome.hold_equity_path),
    )
    assert _reward(outcome, REWARD_IDS[2]) == expected


def test_hold_relative_rewards_are_zero() -> None:
    for state in (
        PortfolioState(100_000, 0),
        PortfolioState(0, 1_000, 100),
        PortfolioState(50_000, 500, 100),
    ):
        hold = _matured((95, 105, 90, 110, 100), state)["HOLD"]
        assert _reward(hold, REWARD_IDS[1]) == 0.0
        assert _reward(hold, REWARD_IDS[2]) == 0.0


def test_flat_cash_market_penalises_unnecessary_buy() -> None:
    outcomes = _matured((100,) * 5, PortfolioState(100_000, 0))
    assert _reward(outcomes["BUY"], REWARD_IDS[1]) < 0
    assert outcomes["HOLD"].total_cost == 0
    assert _reward(outcomes["HOLD"], REWARD_IDS[0]) == 0


def test_flat_long_market_penalises_unnecessary_sell() -> None:
    outcomes = _matured((100,) * 5, PortfolioState(0, 1_000, 100))
    assert _reward(outcomes["SELL"], REWARD_IDS[1]) < 0
    assert outcomes["HOLD"].total_cost == 0
    assert _reward(outcomes["HOLD"], REWARD_IDS[0]) == 0


def test_rising_cash_market_action_ordering() -> None:
    outcomes = _matured((102, 104, 106, 108, 110), PortfolioState(100_000, 0))
    rewards = {action: _reward(value, REWARD_IDS[1]) for action, value in outcomes.items()}
    assert rewards["BUY"] > rewards["HOLD"] == rewards["SELL"]


def test_falling_cash_market_action_ordering() -> None:
    outcomes = _matured((98, 96, 94, 92, 90), PortfolioState(100_000, 0))
    rewards = {action: _reward(value, REWARD_IDS[1]) for action, value in outcomes.items()}
    assert rewards["BUY"] < rewards["HOLD"] == rewards["SELL"]


def test_rising_long_market_action_ordering() -> None:
    outcomes = _matured((102, 104, 106, 108, 110), PortfolioState(0, 1_000, 100))
    rewards = {action: _reward(value, REWARD_IDS[1]) for action, value in outcomes.items()}
    assert rewards["SELL"] < rewards["HOLD"] == rewards["BUY"]


def test_falling_long_market_action_ordering() -> None:
    outcomes = _matured((98, 96, 94, 92, 90), PortfolioState(0, 1_000, 100))
    rewards = {action: _reward(value, REWARD_IDS[1]) for action, value in outcomes.items()}
    assert rewards["SELL"] > rewards["HOLD"] == rewards["BUY"]


@pytest.mark.parametrize(
    "closes",
    [(100, 85, 80, 90, 100), (100, 115, 120, 110, 100)],
)
def test_interim_drawdown_can_distinguish_r2_from_r3(
    closes: tuple[float, ...],
) -> None:
    buy = _matured(closes, PortfolioState(100_000, 0))["BUY"]
    assert _reward(buy, REWARD_IDS[2]) < _reward(buy, REWARD_IDS[1])


def test_transaction_cost_monotonicity() -> None:
    state = PortfolioState(100_000, 0)
    normal = _matured((102, 104, 106, 108, 110), state)["BUY"]
    stressed = _matured(
        (102, 104, 106, 108, 110),
        state,
        commission_bps=100,
        slippage_bps=100,
    )["BUY"]
    assert stressed.total_cost > normal.total_cost
    assert _reward(stressed, REWARD_IDS[0]) < _reward(normal, REWARD_IDS[0])
    assert _reward(stressed, REWARD_IDS[1]) < _reward(normal, REWARD_IDS[1])


def test_equity_scale_invariance() -> None:
    rewards = []
    for equity in (10_000, 100_000, 1_000_000):
        outcome = _matured((102, 104, 106, 108, 110), PortfolioState(equity, 0))["BUY"]
        rewards.append(candidate_rewards(outcome))
    for candidate in REWARD_IDS:
        assert rewards[0][candidate] == pytest.approx(rewards[1][candidate], abs=1e-14)
        assert rewards[0][candidate] == pytest.approx(rewards[2][candidate], abs=1e-14)


def test_noop_semantics_match_discrete_frozen_position_state() -> None:
    cash_outcomes = _matured((101, 102, 103, 104, 105), PortfolioState(100_000, 0))
    long_outcomes = _matured((101, 102, 103, 104, 105), PortfolioState(0, 1_000, 100))
    fractional_outcomes = _matured(
        (101, 102, 103, 104, 105), PortfolioState(50_000, 500, 100)
    )
    for outcome in (
        cash_outcomes["SELL"],
        long_outcomes["BUY"],
        fractional_outcomes["BUY"],
    ):
        assert outcome.action_was_noop
        assert outcome.total_cost == 0
        assert _reward(outcome, REWARD_IDS[1]) == 0
        assert _reward(outcome, REWARD_IDS[2]) == 0


def test_pre_execution_overnight_gap_is_not_attributed() -> None:
    state = PortfolioState(100_000, 0)
    low = simulate_counterfactuals(
        _window((101, 102, 103, 104, 105), decision_close=50), state
    )
    high = simulate_counterfactuals(
        _window((101, 102, 103, 104, 105), decision_close=200), state
    )
    assert low.outcomes is not None and high.outcomes is not None
    for action in Action:
        assert candidate_rewards(low.outcomes[action.value]) == candidate_rewards(
            high.outcomes[action.value]
        )


def test_post_maturity_prices_cannot_change_outcome() -> None:
    state = PortfolioState(100_000, 0)
    base = simulate_counterfactuals(_window((101, 102, 103, 104, 105)), state)
    appended = simulate_counterfactuals(
        _window(
            (101, 102, 103, 104, 105),
            extra_bars=(MarketBar("2023-10-16", 1_000_000, 0.01),),
        ),
        state,
    )
    assert base == appended


def test_reward_is_pending_before_fifth_subsequent_session() -> None:
    window = _window((101, 102, 103, 104, 105))
    result = simulate_counterfactuals(replace(window, bars=window.bars[:-1]), PortfolioState(100_000, 0))
    assert result.status is RewardStatus.PENDING
    assert result.outcomes is None
    assert result.maturity_session == "2023-10-13"


def test_pending_query_still_rejects_invalid_portfolio_state() -> None:
    window = _window((101, 102, 103, 104, 105))
    with pytest.raises(ValueError, match="long-only"):
        simulate_counterfactuals(
            replace(window, bars=window.bars[:-1]), PortfolioState(-1, 0)
        )


def test_reward_matures_at_fifth_subsequent_session() -> None:
    result = simulate_counterfactuals(
        _window((101, 102, 103, 104, 105)), PortfolioState(100_000, 0)
    )
    assert result.status is RewardStatus.MATURED
    assert result.execution_session == "2023-10-09"
    assert result.maturity_session == "2023-10-13"


@pytest.mark.parametrize("action", ["buy", "SHORT", "", 1])
def test_action_space_enforcement(action) -> None:
    with pytest.raises(ValueError, match="BUY, HOLD, or SELL"):
        validate_action(action)


@pytest.mark.parametrize(
    "state",
    [PortfolioState(-1, 0), PortfolioState(0, -1), PortfolioState(0, 1, 0)],
)
def test_no_short_or_leverage_input_path(state: PortfolioState) -> None:
    with pytest.raises(ValueError):
        simulate_counterfactuals(_window((100,) * 5), state)


def test_candidate_rewards_are_finite_on_valid_scenarios() -> None:
    scenarios = ((100,) * 5, (110, 90, 120, 80, 105), (90, 92, 95, 98, 101))
    states = (PortfolioState(100_000, 0), PortfolioState(0, 1_000, 100))
    for closes in scenarios:
        for state in states:
            outcomes = _matured(closes, state)
            for outcome in outcomes.values():
                assert all(math.isfinite(value) for value in candidate_rewards(outcome).values())


def test_repeated_execution_is_deterministic() -> None:
    window = _window((102, 98, 104, 96, 110))
    state = PortfolioState(50_000, 500, 100)
    assert simulate_counterfactuals(window, state) == simulate_counterfactuals(window, state)


def _direct_frozen_path(
    action: Action,
    state: PortfolioState,
    closes: tuple[float, ...],
) -> dict[str, float]:
    schedule = ExchangeSchedule()
    portfolio = Portfolio("SYNTH", 100_000)
    portfolio.cash = state.cash
    portfolio.quantity = state.quantity
    portfolio.average_entry_price = state.average_entry_price
    portfolio.open_position_commission = state.open_position_commission
    broker = Broker(
        commission_bps=FORMAL_COMMISSION_BPS,
        slippage_bps=FORMAL_SLIPPAGE_BPS,
        fractional_shares=True,
    )
    commission = slippage = 0.0
    if action is not Action.HOLD:
        order = Order(
            order_id=f"direct:{action.value}",
            symbol="SYNTH",
            created_at=schedule.session_close("2023-10-06").to_pydatetime(),
            intended_execution_session=SESSIONS[0],
            target_weight=1.0 if action is Action.BUY else 0.0,
            current_weight=portfolio.weight(100.0),
            side=action.value,
        )
        fill = broker.execute(
            order,
            portfolio,
            100.0,
            schedule.session_open(SESSIONS[0]).to_pydatetime(),
        )
        if fill:
            commission = fill.commission
            slippage = fill.slippage_cost
    terminal = 0.0
    for session, close in zip(SESSIONS, closes, strict=True):
        terminal = portfolio.snapshot(
            session, schedule.session_close(session).to_pydatetime(), close
        ).equity
    return {
        "cash": portfolio.cash,
        "shares": portfolio.quantity,
        "commission": commission,
        "slippage": slippage,
        "terminal_equity": terminal,
    }


@pytest.mark.parametrize(
    ("action", "state"),
    [
        (Action.BUY, PortfolioState(100_000, 0)),
        (Action.SELL, PortfolioState(0, 1_000, 100, 50)),
        (Action.HOLD, PortfolioState(50_000, 500, 100, 25)),
    ],
)
def test_frozen_backtester_execution_equivalence(
    action: Action, state: PortfolioState
) -> None:
    closes = (102, 104, 106, 108, 110)
    expected = _direct_frozen_path(action, state, closes)
    outcome = _matured(closes, state)[action.value]
    assert outcome.terminal_cash == pytest.approx(expected["cash"], rel=1e-12, abs=1e-10)
    assert outcome.terminal_shares == pytest.approx(expected["shares"], rel=1e-12, abs=1e-10)
    assert outcome.commission_cost == pytest.approx(expected["commission"], rel=1e-12, abs=1e-10)
    assert outcome.slippage_cost == pytest.approx(expected["slippage"], rel=1e-12, abs=1e-10)
    assert outcome.terminal_equity == pytest.approx(expected["terminal_equity"], rel=1e-12, abs=1e-10)


def test_drawdown_uses_post_execution_path_without_pre_trade_peak() -> None:
    assert maximum_drawdown((90.0, 100.0, 80.0, 120.0)) == pytest.approx(0.2)


def test_synthetic_sanity_artifact_is_canonical_and_passes() -> None:
    payload = synthetic_sanity_payload()
    expected = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    assert (DOCS / "m2_reward_synthetic_sanity.json").read_bytes() == expected
    assert payload["all_passed"]
    assert payload["leaderboard"] is False


def test_machine_readable_contract_is_canonical_and_defers_selection() -> None:
    path = DOCS / "m2_reward_study_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()
    assert path.read_bytes() == expected
    assert payload["final_reward_selected"] is False
    assert len(payload["candidate_rewards"]) == 3
    assert payload["cost"]["deepseek_calls"] == 0
    assert payload["cost"]["aws_gpu_hours"] == 0
