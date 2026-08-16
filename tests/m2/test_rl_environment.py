from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.m2.reward_simulator import MarketBar, RewardWindow
from scripts.m2.rl_environment import (
    ACTIONS,
    INITIAL_CASH,
    SequentialPortfolioState,
    WeeklyTransitionWindow,
    simulate_local_credit,
    simulate_weekly_transition,
)
from scripts.m2.semantic_state_representation import build_actor_observation


def _weekly_window(
    *,
    decision: str = "2023-10-06",
    child: str = "2023-10-13",
    sessions: tuple[str, ...] = (
        "2023-10-09",
        "2023-10-10",
        "2023-10-11",
        "2023-10-12",
        "2023-10-13",
    ),
    closes: tuple[float, ...] = (101.0, 102.0, 103.0, 104.0, 105.0),
    dividend: float = 0.0,
    split: float = 0.0,
) -> WeeklyTransitionWindow:
    return WeeklyTransitionWindow(
        symbol="SYNTH",
        decision_session=decision,
        child_decision_session=child,
        bars=tuple(
            MarketBar(
                session=session,
                open_price=100.0 if index == 0 else close,
                close_price=close,
                dividend_per_share=dividend if index == 1 else 0.0,
                split_ratio=split if index == 2 else 0.0,
            )
            for index, (session, close) in enumerate(zip(sessions, closes, strict=True))
        ),
    )


def _credit_window(
    *,
    decision: str = "2023-10-06",
    sessions: tuple[str, ...] = (
        "2023-10-09",
        "2023-10-10",
        "2023-10-11",
        "2023-10-12",
        "2023-10-13",
    ),
    closes: tuple[float, ...] = (101.0, 102.0, 103.0, 104.0, 105.0),
) -> RewardWindow:
    return RewardWindow(
        symbol="SYNTH",
        decision_session=decision,
        decision_close_price=100.0,
        bars=tuple(
            MarketBar(session, 100.0 if index == 0 else close, close)
            for index, (session, close) in enumerate(zip(sessions, closes, strict=True))
        ),
    )


def test_actions_and_root_are_frozen() -> None:
    assert ACTIONS == ("BUY", "HOLD", "SELL")
    root = SequentialPortfolioState()
    assert root.cash == root.peak_equity == INITIAL_CASH
    assert root.quantity == 0
    with pytest.raises(ValueError, match="long-only"):
        SequentialPortfolioState(cash=-1)


@pytest.mark.parametrize("action", ACTIONS)
def test_weekly_transition_stops_at_child_close(action: str) -> None:
    result = simulate_weekly_transition(
        _weekly_window(), SequentialPortfolioState(), action
    )
    assert result.next_snapshot.session == "2023-10-13"
    assert result.next_decision_session == "2023-10-13"
    assert result.next_state.cash >= 0
    assert result.next_state.quantity >= 0


def test_sequential_entry_basis_and_corporate_actions_are_preserved() -> None:
    bought = simulate_weekly_transition(
        _weekly_window(), SequentialPortfolioState(), "BUY"
    )
    assert bought.next_state.average_entry_price == pytest.approx(100.05)
    state = SequentialPortfolioState(
        cash=0.0,
        quantity=1000.0,
        average_entry_price=100.0,
        open_position_commission=50.0,
        peak_equity=100_000.0,
    )
    adjusted = simulate_weekly_transition(
        _weekly_window(
            closes=(100.0, 100.0, 50.0, 50.0, 50.0), dividend=1.0, split=2.0
        ),
        state,
        "HOLD",
    )
    assert adjusted.next_state.cash == pytest.approx(1000.0)
    assert adjusted.next_state.quantity == pytest.approx(2000.0)
    assert adjusted.next_state.average_entry_price == pytest.approx(50.0)
    assert adjusted.next_state.cumulative_dividends == pytest.approx(1000.0)


def test_local_credit_is_independent_and_matches_five_session_maturity() -> None:
    credit = simulate_local_credit(_credit_window(), SequentialPortfolioState())
    assert credit.reward_maturity_session == "2023-10-13"
    assert tuple(credit.rewards_r3) == ACTIONS
    assert all(np.isfinite(value) for value in credit.rewards_r3.values())
    assert credit.rewards_r3["HOLD"] == 0.0


def _memorial_windows(june_5_close: float):
    transition = _weekly_window(
        decision="2023-05-26",
        child="2023-06-02",
        sessions=("2023-05-30", "2023-05-31", "2023-06-01", "2023-06-02"),
        closes=(101.0, 102.0, 103.0, 104.0),
    )
    credit = _credit_window(
        decision="2023-05-26",
        sessions=("2023-05-30", "2023-05-31", "2023-06-01", "2023-06-02", "2023-06-05"),
        closes=(101.0, 102.0, 103.0, 104.0, june_5_close),
    )
    return transition, credit


def test_memorial_day_poison_changes_credit_not_child_state_or_observation() -> None:
    transition, base_credit_window = _memorial_windows(105.0)
    _, poisoned_credit_window = _memorial_windows(1.0)
    base_child = simulate_weekly_transition(
        transition, SequentialPortfolioState(), "BUY"
    )
    poisoned_child = simulate_weekly_transition(
        transition, SequentialPortfolioState(), "BUY"
    )
    assert base_child.next_state == poisoned_child.next_state
    assert base_child.next_snapshot == poisoned_child.next_snapshot
    semantic_base = np.zeros(3076, dtype=np.float32)
    semantic_base[3073] = 1.0
    base_observation = build_actor_observation(semantic_base, base_child.next_snapshot)
    poisoned_observation = build_actor_observation(
        semantic_base, poisoned_child.next_snapshot
    )
    assert base_observation.tobytes() == poisoned_observation.tobytes()
    base_credit = simulate_local_credit(
        base_credit_window, SequentialPortfolioState()
    )
    poisoned_credit = simulate_local_credit(
        poisoned_credit_window, SequentialPortfolioState()
    )
    assert base_credit.reward_maturity_session == "2023-06-05"
    assert base_credit.input_sha256_by_action != poisoned_credit.input_sha256_by_action


def test_parent_credit_cannot_depend_on_descendant_action() -> None:
    _, window = _memorial_windows(105.0)
    parent_credit_before_buy_child = simulate_local_credit(
        window, SequentialPortfolioState()
    )
    parent_credit_before_sell_child = simulate_local_credit(
        replace(window), SequentialPortfolioState()
    )
    assert parent_credit_before_buy_child == parent_credit_before_sell_child


def test_transition_rejects_any_post_child_bar() -> None:
    base = _weekly_window()
    contaminated = replace(
        base, bars=base.bars + (MarketBar("2023-10-16", 1.0, 1.0),)
    )
    with pytest.raises(ValueError, match="only execution-through-child"):
        simulate_weekly_transition(contaminated, SequentialPortfolioState(), "BUY")
