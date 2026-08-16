from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from scripts.m2.online_delayed_adaptation import (
    C09_PARAMETER_SHA,
    CreditStatus,
    FrozenTrainTree,
    SymbolOnlineState,
    archive_terminal_credits,
    load_c09,
    process_decision,
)
from scripts.m2.pa_ctppo import fast_checkpoint_sha
from scripts.m2.rl_environment import TRAIN_SESSIONS, TRAIN_SYMBOLS
from scripts.m2.select_online_adaptation import GRID, choose_candidate
from scripts.m2.train_global_pa_ctppo import canonical_parameter_sha
from tradingagents.backtesting.calendar import ExchangeSchedule


@pytest.fixture(scope="module")
def experiments_root() -> Path:
    path = Path(__file__).resolve().parents[3] / "AlphaMAS-Experiments"
    if not path.exists():
        pytest.skip("canonical AlphaMAS-Experiments sibling is unavailable")
    return path


@pytest.fixture(scope="module")
def frozen(experiments_root: Path):
    checkpoint = (
        experiments_root
        / "experiments/M2/development/global_selection_v1/selected_global_checkpoint/model.pt"
    )
    return load_c09(checkpoint), FrozenTrainTree(experiments_root)


def test_pending_record_contains_no_future_outcome_and_same_close_update_is_delayed(frozen):
    model, tree = frozen
    state = SymbolOnlineState.fresh("AAPL", model, tree.roots["AAPL"], 1e-4)
    initial_fast = fast_checkpoint_sha(state.model)
    first = process_decision(state, tree, epochs=1)
    credit = state.credits[0]
    assert credit.status is CreditStatus.PENDING
    assert credit.counterfactual_r3 is None
    assert first["selected_action_local_r3"] is None
    second = process_decision(state, tree, epochs=1)
    assert second["fast_sha_at_action"] == initial_fast
    assert credit.status is CreditStatus.APPLIED
    assert credit.counterfactual_r3 is not None
    assert fast_checkpoint_sha(state.model) != initial_fast
    assert state.optimiser.state


def test_memorial_day_uses_five_xnys_sessions_not_calendar_days():
    sessions = tuple(
        item.date().isoformat()
        for item in ExchangeSchedule().calendar.sessions_window("2023-05-26", 6).tz_localize(None)
    )
    assert sessions == (
        "2023-05-26",
        "2023-05-30",
        "2023-05-31",
        "2023-06-01",
        "2023-06-02",
        "2023-06-05",
    )


def test_post_child_reward_poison_cannot_change_child_state_observation_or_action(frozen):
    model, tree = frozen
    poisoned_tree = copy.copy(tree)
    poisoned_tree.edges = dict(tree.edges)
    root = tree.roots["AAPL"]
    poisoned_tree.edges[root] = dict(tree.edges[root])
    original_edge = poisoned_tree.edges[root]["BUY"]
    poisoned_tree.edges[root]["BUY"] = replace(
        original_edge, local_r3=original_edge.local_r3 + 0.25
    )
    baseline = SymbolOnlineState.fresh("AAPL", model, root, 1e-4)
    poisoned = SymbolOnlineState.fresh("AAPL", model, root, 1e-4)
    first_base = process_decision(baseline, tree, epochs=1)
    first_poison = process_decision(poisoned, poisoned_tree, epochs=1)
    second_base = process_decision(baseline, tree, epochs=1)
    second_poison = process_decision(poisoned, poisoned_tree, epochs=1)
    assert (
        first_base["deterministic_selected_action"] == first_poison["deterministic_selected_action"]
    )
    assert second_base["node_id"] == second_poison["node_id"]
    assert (
        second_base["origin_observation_identity"] == second_poison["origin_observation_identity"]
    )
    assert (
        second_base["deterministic_selected_action"]
        == second_poison["deterministic_selected_action"]
    )
    assert second_base["fast_sha_at_action"] == second_poison["fast_sha_at_action"]
    assert fast_checkpoint_sha(baseline.model) != fast_checkpoint_sha(poisoned.model)


def test_per_symbol_optimizer_and_fast_parameters_are_isolated(frozen):
    model, tree = frozen
    states = {
        symbol: SymbolOnlineState.fresh(symbol, model, tree.roots[symbol], 1e-4)
        for symbol in TRAIN_SYMBOLS
    }
    before = {
        symbol: (fast_checkpoint_sha(state.model), state.optimiser_sha())
        for symbol, state in states.items()
    }
    process_decision(states["AAPL"], tree, epochs=1)
    process_decision(states["AAPL"], tree, epochs=1)
    assert all(
        (fast_checkpoint_sha(states[symbol].model), states[symbol].optimiser_sha())
        == before[symbol]
        for symbol in TRAIN_SYMBOLS
        if symbol != "AAPL"
    )
    assert canonical_parameter_sha(model) == C09_PARAMETER_SHA


@pytest.mark.parametrize("boundary", [1, 2])
def test_safe_resume_is_exact_on_both_sides_of_maturity(tmp_path: Path, frozen, boundary: int):
    model, tree = frozen
    continuous = SymbolOnlineState.fresh("AAPL", model, tree.roots["AAPL"], 1e-4)
    resumed = SymbolOnlineState.fresh("AAPL", model, tree.roots["AAPL"], 1e-4)
    for _ in range(7):
        process_decision(continuous, tree, epochs=1)
    archive_terminal_credits(continuous, tree)
    for _ in range(boundary):
        process_decision(resumed, tree, epochs=1)
    path = tmp_path / "state.json"
    resumed.save(path)
    resumed = SymbolOnlineState.load(path, model, 1e-4)
    while resumed.next_decision_index < len(TRAIN_SESSIONS):
        process_decision(resumed, tree, epochs=1)
    archive_terminal_credits(resumed, tree)
    assert continuous.issued_actions == resumed.issued_actions
    assert [asdict(item) for item in continuous.credits] == [
        asdict(item) for item in resumed.credits
    ]
    assert fast_checkpoint_sha(continuous.model) == fast_checkpoint_sha(resumed.model)
    assert continuous.optimiser_sha() == resumed.optimiser_sha()
    assert continuous.state_payload()["state_identity"] == resumed.state_payload()["state_identity"]


def test_selector_uses_only_preregistered_tie_break_order():
    scores = []
    for candidate, learning_rate, epochs in GRID:
        scores.append(
            {
                "candidate_id": candidate,
                "online_learning_rate": learning_rate,
                "update_epochs": epochs,
                "status": "VALID",
                "primary_mean_sequential_train_r3": 1.0,
                "worst_symbol_cumulative_r3": 0.0,
                "prompt_override_rate": 0.0,
            }
        )
    selected, tie = choose_candidate(scores)
    assert selected["candidate_id"] == "O01"
    assert tie["primary_metric_tie"] is True
    assert tie["path"][-1]["criterion"] == "online_learning_rate"
