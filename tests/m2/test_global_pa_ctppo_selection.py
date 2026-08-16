from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.m2.pa_ctppo import deterministic_action
from scripts.m2.reward_simulator import MarketBar, RewardWindow
from scripts.m2.rl_environment import (
    SequentialPortfolioState,
    WeeklyTransitionWindow,
    simulate_local_credit,
    simulate_weekly_transition,
)
from scripts.m2.select_global_pa_ctppo import (
    CANDIDATE_GRID,
    SELECTION_TIE_TOLERANCE,
    assert_unprotected_path,
    choose_candidate,
)
from scripts.m2.semantic_state_representation import build_actor_observation


def _scores():
    return [
        {
            "candidate_id": candidate_id,
            "learning_rate": lr,
            "checkpoint_iteration": iteration,
            "primary_mean_sequential_local_r3": 0.0,
            "worst_symbol_cumulative_r3": 0.0,
            "override_rate": 0.0,
            "correctness_valid": True,
        }
        for candidate_id, (lr, iteration, _) in CANDIDATE_GRID.items()
    ]


def test_candidate_grid_is_exact_and_unique():
    assert list(CANDIDATE_GRID) == [f"C{i:02d}" for i in range(1, 10)]
    assert len({values[:2] for values in CANDIDATE_GRID.values()}) == 9
    assert len({values[2] for values in CANDIDATE_GRID.values()}) == 9


def test_deterministic_action_argmax_prompt_first_and_fallback():
    assert deterministic_action(torch.tensor([0.7, 0.2, 0.1]), 1) == "BUY"
    assert deterministic_action(torch.tensor([0.5, 0.5, 0.0]), 1) == "HOLD"
    assert deterministic_action(torch.tensor([0.5, 0.0, 0.5]), 1) == "BUY"


def test_selection_primary_then_each_ordered_tie_break():
    scores = _scores()
    scores[4]["primary_mean_sequential_local_r3"] = SELECTION_TIE_TOLERANCE * 2
    assert choose_candidate(scores)[0]["candidate_id"] == "C05"
    scores = _scores()
    scores[3]["worst_symbol_cumulative_r3"] = 1.0
    assert choose_candidate(scores)[0]["candidate_id"] == "C04"
    scores = _scores()
    scores[0]["override_rate"] = 0.5
    assert choose_candidate(scores)[0]["candidate_id"] == "C02"
    scores = _scores()
    assert choose_candidate(scores)[0]["candidate_id"] == "C01"
    scores = _scores()
    for item in scores:
        item["learning_rate"] = 1e-4
        if item["candidate_id"] not in {"C01", "C02", "C03"}:
            item["primary_mean_sequential_local_r3"] = -1.0
    assert choose_candidate(scores)[0]["candidate_id"] == "C01"


def test_primary_values_within_tolerance_are_tied():
    scores = _scores()
    scores[8]["primary_mean_sequential_local_r3"] = SELECTION_TIE_TOLERANCE / 2
    assert choose_candidate(scores)[0]["candidate_id"] == "C01"


@pytest.mark.parametrize(
    "name", ["FINAL_HOLDOUT/x", "E2E_PILOT/x", "Formal_2024/results", "2024H1.json"]
)
def test_protected_paths_fail_closed(name):
    with pytest.raises(RuntimeError, match="protected evaluation path"):
        assert_unprotected_path(Path("/tmp") / name)


def test_invalid_or_mutated_candidate_population_is_rejected():
    scores = _scores()
    scores[0]["correctness_valid"] = False
    with pytest.raises(ValueError, match="correctness-valid"):
        choose_candidate(scores)
    duplicated = deepcopy(_scores())
    duplicated[-1]["candidate_id"] = "C01"
    with pytest.raises(ValueError, match="exactly C01-C09"):
        choose_candidate(duplicated)


def _bars(last_close: float = 105.0):
    sessions = ("2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13")
    closes = (101.0, 102.0, 103.0, 104.0, last_close)
    return tuple(
        MarketBar(session, 100.0 if index == 0 else close, close)
        for index, (session, close) in enumerate(zip(sessions, closes, strict=True))
    )


def test_sequential_buy_hold_sell_state_and_frozen_costs():
    window = WeeklyTransitionWindow("SYNTH", "2023-10-06", "2023-10-13", _bars())
    bought = simulate_weekly_transition(window, SequentialPortfolioState(), "BUY")
    assert bought.next_state.quantity > 0
    assert bought.next_state.cash == pytest.approx(0.0)
    assert bought.commission_cost == pytest.approx(49.97501249375313)
    assert bought.slippage_cost == pytest.approx(49.95003747501278)
    held = simulate_weekly_transition(window, bought.next_state, "HOLD")
    assert held.next_state.quantity == bought.next_state.quantity
    sold = simulate_weekly_transition(window, bought.next_state, "SELL")
    assert sold.next_state.quantity == pytest.approx(0.0)
    assert sold.next_state.cash > 0


def test_child_observation_precedes_parent_reward_maturity():
    transition_bars = _bars()[:-1]
    transition = WeeklyTransitionWindow("SYNTH", "2023-10-06", "2023-10-12", transition_bars)
    child = simulate_weekly_transition(transition, SequentialPortfolioState(), "BUY")
    semantic = np.zeros(3076, dtype=np.float32)
    semantic[3073] = 1.0
    observation = build_actor_observation(semantic, child.next_snapshot)
    rewards = []
    for last_close in (105.0, 1.0):
        credit = simulate_local_credit(
            RewardWindow("SYNTH", "2023-10-06", 100.0, _bars(last_close)),
            SequentialPortfolioState(),
        )
        rewards.append(credit.rewards_r3["BUY"])
    assert build_actor_observation(semantic, child.next_snapshot).tobytes() == observation.tobytes()
    assert rewards[0] != rewards[1]
