from __future__ import annotations

import math

import pytest

from scripts.m2.reward_simulator import MarketBar, RewardWindow
from scripts.m2.rl_environment import (
    ACTIONS,
    INITIAL_CASH,
    SequentialPortfolioState,
    simulate_transition,
)

SESSIONS = ("2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13")


def _window(
    closes: tuple[float, ...],
    *,
    dividend: float = 0.0,
    split: float = 0.0,
) -> RewardWindow:
    return RewardWindow(
        symbol="SYNTH",
        decision_session="2023-10-06",
        decision_close_price=100.0,
        bars=tuple(
            MarketBar(
                session=session,
                open_price=100.0 if index == 0 else close,
                close_price=close,
                dividend_per_share=dividend if index == 1 else 0.0,
                split_ratio=split if index == 2 else 0.0,
            )
            for index, (session, close) in enumerate(zip(SESSIONS, closes, strict=True))
        ),
    )


def test_actions_are_exactly_buy_hold_sell() -> None:
    assert ACTIONS == ("BUY", "HOLD", "SELL")


@pytest.mark.parametrize("action", ACTIONS)
def test_transition_matches_frozen_reward_simulator_for_every_action(action: str) -> None:
    result = simulate_transition(
        _window((101.0, 102.0, 103.0, 104.0, 105.0)),
        SequentialPortfolioState(),
        action,
    )
    assert math.isfinite(result.reward_r3)
    assert result.terminal_cash >= 0
    assert result.terminal_quantity >= 0
    assert result.maturity_session == "2023-10-13"


def test_sequential_buy_preserves_true_entry_basis_and_hold_continuity() -> None:
    bought = simulate_transition(
        _window((101.0, 102.0, 103.0, 104.0, 105.0)),
        SequentialPortfolioState(),
        "BUY",
    )
    assert bought.next_state.quantity > 0
    assert bought.next_state.average_entry_price == pytest.approx(100.05)
    held = simulate_transition(
        RewardWindow(
            symbol="SYNTH",
            decision_session="2023-10-13",
            decision_close_price=105.0,
            bars=tuple(
                MarketBar(session, close, close)
                for session, close in zip(
                    ("2023-10-16", "2023-10-17", "2023-10-18", "2023-10-19", "2023-10-20"),
                    (106.0, 107.0, 108.0, 109.0, 110.0),
                    strict=True,
                )
            ),
        ),
        bought.next_state,
        "HOLD",
    )
    assert held.next_state.average_entry_price == bought.next_state.average_entry_price
    assert held.next_state.open_position_commission == bought.next_state.open_position_commission
    assert held.next_state.peak_equity >= bought.next_state.peak_equity


def test_split_and_dividend_continuity() -> None:
    state = SequentialPortfolioState(
        cash=0.0,
        quantity=1000.0,
        average_entry_price=100.0,
        open_position_commission=50.0,
        peak_equity=100_000.0,
    )
    result = simulate_transition(
        _window((100.0, 100.0, 50.0, 50.0, 50.0), dividend=1.0, split=2.0),
        state,
        "HOLD",
    )
    assert result.next_state.cash == pytest.approx(1000.0)
    assert result.next_state.quantity == pytest.approx(2000.0)
    assert result.next_state.average_entry_price == pytest.approx(50.0)
    assert result.next_state.cumulative_dividends == pytest.approx(1000.0)


def test_root_contract_and_long_only_validation() -> None:
    root = SequentialPortfolioState()
    assert root.cash == INITIAL_CASH
    assert root.quantity == 0
    assert root.peak_equity == INITIAL_CASH
    with pytest.raises(ValueError, match="long-only"):
        SequentialPortfolioState(cash=-1)


def test_post_maturity_bar_is_not_used() -> None:
    base = _window((101.0, 102.0, 103.0, 104.0, 105.0))
    appended = RewardWindow(
        symbol=base.symbol,
        decision_session=base.decision_session,
        decision_close_price=base.decision_close_price,
        bars=base.bars + (MarketBar("2023-10-16", 1.0, 1.0),),
    )
    assert simulate_transition(base, SequentialPortfolioState(), "BUY") == simulate_transition(
        appended, SequentialPortfolioState(), "BUY"
    )
