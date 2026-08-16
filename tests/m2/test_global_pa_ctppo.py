from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.m2.pa_ctppo import (  # noqa: E402
    EXPECTED_PARAMETERS,
    PPO_CLIP,
    VALUE_LOSS_COEFFICIENT,
    PromptAnchoredActorCritic,
    configure_determinism,
    exact_ppo_loss,
)
from scripts.m2.train_global_pa_ctppo import (  # noqa: E402
    CANDIDATE_IDS,
    CHECKPOINT_ITERATIONS,
    FORBIDDEN_DESKTOP_ROOT,
    LEARNING_RATES,
    OPTIMISATION_EPOCHS,
    OUTER_ITERATIONS,
    OldPolicyChronology,
    TreeTensors,
    assert_not_desktop_checkout,
    canonical_parameter_sha,
    exact_tree_policy_evaluation,
    load_frozen_train_tree,
    normalise_policy_probabilities,
    prompt_prior,
)

EXPERIMENTS_ROOT = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-Experiments")


def _observation(prompt_index: int) -> torch.Tensor:
    observation = torch.zeros(3080)
    observation[0:1024] = torch.linspace(-1, 1, 1024)
    observation[1024:2048] = torch.linspace(1, -1, 1024)
    observation[2048:3072] = observation[1024:2048] - observation[0:1024]
    observation[3072] = 0.25
    observation[3073 + prompt_index] = 1.0
    observation[3076] = 1.0
    return observation


def _synthetic_tree() -> TreeTensors:
    observations = torch.stack([_observation(index % 3) for index in range(8)])
    rewards = torch.tensor(
        [[1.0 + index / 10, 0.1, -1.0] for index in range(8)], dtype=torch.float64
    )
    return TreeTensors(
        observations=observations,
        rewards=rewards,
        child_indices=torch.full((8, 3), -1, dtype=torch.int64),
        depths=torch.zeros(8, dtype=torch.int64),
        root_indices=torch.arange(8),
        prompt_action_indices=torch.tensor([index % 3 for index in range(8)]),
        node_ids=tuple(f"node-{index}" for index in range(8)),
        ordering_sha256="synthetic",
        roles=("TRAIN",) * 8,
    )


def _synthetic_outer_step(
    model: PromptAnchoredActorCritic,
    optimiser: torch.optim.Optimizer,
    tree: TreeTensors,
    chronology: OldPolicyChronology,
    iteration: int,
) -> tuple[str, list[str]]:
    old_sha = canonical_parameter_sha(model)
    chronology.begin_iteration(iteration, old_sha)
    with torch.no_grad():
        old = normalise_policy_probabilities(model(tree.observations)["probabilities"].clone())
    evaluation = exact_tree_policy_evaluation(tree, old)
    epoch_shas = []
    for _ in range(OPTIMISATION_EPOCHS):
        chronology.record_epoch(old_sha)
        epoch_shas.append(old_sha)
        output = model(tree.observations)
        losses = exact_ppo_loss(
            normalise_policy_probabilities(output["probabilities"]),
            old,
            evaluation.advantages,
            output["value"].double(),
            evaluation.values,
            evaluation.occupancy,
        )
        optimiser.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimiser.step()
    post_sha = canonical_parameter_sha(model)
    chronology.finish_iteration(post_sha)
    return post_sha, epoch_shas


def test_canonical_path_guard_rejects_legacy_desktop_checkout() -> None:
    with pytest.raises(RuntimeError, match="legacy Desktop"):
        assert_not_desktop_checkout(FORBIDDEN_DESKTOP_ROOT)


def test_initialization_parameter_sha_and_prompt_prior_are_deterministic() -> None:
    configure_determinism(20260816)
    first = PromptAnchoredActorCritic()
    configure_determinism(20260816)
    second = PromptAnchoredActorCritic()
    assert first.parameter_count() == second.parameter_count() == EXPECTED_PARAMETERS == 20_197
    assert canonical_parameter_sha(first) == canonical_parameter_sha(second)
    observations = _synthetic_tree().observations
    assert torch.allclose(
        first(observations)["probabilities"], prompt_prior(observations), atol=1e-7, rtol=0
    )
    state = copy.deepcopy(first.state_dict())
    starts = []
    for _ in LEARNING_RATES:
        model = PromptAnchoredActorCritic()
        model.load_state_dict(state)
        starts.append(canonical_parameter_sha(model))
    assert len(set(starts)) == 1


def test_frozen_train_tree_boundary_and_initial_prior_all_nodes() -> None:
    tree = load_frozen_train_tree(EXPERIMENTS_ROOT)
    assert tree.node_count == 8_744
    assert tree.rewards.shape == (8_744, 3)
    assert set(tree.roles) == {"TRAIN"}
    assert len(tree.root_indices) == 8
    configure_determinism(20260816)
    model = PromptAnchoredActorCritic()
    with torch.no_grad():
        difference = torch.max(
            torch.abs(model(tree.observations)["probabilities"] - prompt_prior(tree.observations))
        )
    assert float(difference) <= 1e-7


def test_exact_local_credit_objective_contract() -> None:
    tree = _synthetic_tree()
    old = torch.tensor([[0.5, 0.3, 0.2]] * 8, dtype=torch.float64)
    evaluation = exact_tree_policy_evaluation(tree, old)
    expected_values = torch.sum(old * tree.rewards, dim=1)
    assert torch.equal(evaluation.values, expected_values)
    assert torch.equal(evaluation.advantages, tree.rewards - expected_values[:, None])
    assert evaluation.max_weighted_advantage_residual <= 1e-10
    assert evaluation.occupancy_mass_by_depth == pytest.approx((1.0,), abs=1e-12)
    assert PPO_CLIP == 0.20
    assert VALUE_LOSS_COEFFICIENT == 0.5
    assert OUTER_ITERATIONS == 100
    assert OPTIMISATION_EPOCHS == 4


def test_old_policy_refresh_and_four_epoch_freeze() -> None:
    chronology = OldPolicyChronology()
    chronology.begin_iteration(1, "initial")
    for _ in range(4):
        chronology.record_epoch("initial")
    chronology.finish_iteration("post-1")
    chronology.begin_iteration(2, "post-1")
    for _ in range(4):
        chronology.record_epoch("post-1")
    chronology.finish_iteration("post-2")
    with pytest.raises(RuntimeError, match="stale iteration-0"):
        chronology.begin_iteration(3, "initial")


def test_old_policy_change_inside_epoch_block_is_rejected() -> None:
    chronology = OldPolicyChronology()
    chronology.begin_iteration(1, "old")
    with pytest.raises(RuntimeError, match="changed inside"):
        chronology.record_epoch("different")


def test_synthetic_multiple_iterations_resume_and_replay_are_exact(tmp_path: Path) -> None:
    tree = _synthetic_tree()
    configure_determinism(20260816)
    initial = PromptAnchoredActorCritic().state_dict()

    uninterrupted = PromptAnchoredActorCritic()
    uninterrupted.load_state_dict(initial)
    uninterrupted_optimiser = torch.optim.AdamW(
        uninterrupted.parameters(), lr=3e-4, weight_decay=1e-4
    )
    uninterrupted_chronology = OldPolicyChronology()
    checkpoint_shas = []
    for iteration in (1, 2):
        sha, epoch_shas = _synthetic_outer_step(
            uninterrupted, uninterrupted_optimiser, tree, uninterrupted_chronology, iteration
        )
        assert len(set(epoch_shas)) == 1
        checkpoint_shas.append(sha)

    configure_determinism(20260816)
    split = PromptAnchoredActorCritic()
    split.load_state_dict(initial)
    split_optimiser = torch.optim.AdamW(split.parameters(), lr=3e-4, weight_decay=1e-4)
    split_chronology = OldPolicyChronology()
    first_sha, _ = _synthetic_outer_step(split, split_optimiser, tree, split_chronology, 1)
    resume_path = tmp_path / "resume.pt"
    torch.save(
        {"model": split.state_dict(), "optimiser": split_optimiser.state_dict()}, resume_path
    )
    resumed = PromptAnchoredActorCritic()
    resumed_optimiser = torch.optim.AdamW(resumed.parameters(), lr=3e-4, weight_decay=1e-4)
    payload = torch.load(resume_path, weights_only=False)
    resumed.load_state_dict(payload["model"])
    resumed_optimiser.load_state_dict(payload["optimiser"])
    resumed_chronology = OldPolicyChronology(
        first_old_sha=split_chronology.first_old_sha,
        previous_post_sha=first_sha,
    )
    second_sha, _ = _synthetic_outer_step(resumed, resumed_optimiser, tree, resumed_chronology, 2)
    assert [first_sha, second_sha] == checkpoint_shas

    configure_determinism(20260816)
    replay = PromptAnchoredActorCritic()
    replay.load_state_dict(initial)
    replay_optimiser = torch.optim.AdamW(replay.parameters(), lr=3e-4, weight_decay=1e-4)
    replay_chronology = OldPolicyChronology()
    replay_shas = [
        _synthetic_outer_step(replay, replay_optimiser, tree, replay_chronology, iteration)[0]
        for iteration in (1, 2)
    ]
    assert replay_shas == checkpoint_shas


def test_candidate_grid_is_exactly_frozen_nine() -> None:
    assert LEARNING_RATES == (1e-4, 3e-4, 1e-3)
    assert CHECKPOINT_ITERATIONS == (25, 50, 100)
    assert len(CANDIDATE_IDS) == 9
    assert set(CANDIDATE_IDS.values()) == {f"C{number:02d}" for number in range(1, 10)}


def test_no_forbidden_training_mechanisms_are_exposed() -> None:
    source = Path("scripts/m2/train_global_pa_ctppo.py").read_text()
    contract = json.loads(Path("docs/m2/m2_global_training_contract.json").read_text())
    assert "torch.multinomial" not in source
    assert "DataLoader" not in source
    assert "scheduler" not in source.lower()
    assert "freeze_global_for_formal" not in source
    assert "gamma =" not in source.lower()
    assert "gae" not in source.lower()
    assert contract["bellman_bootstrap"] is False
    assert contract["gamma"] is None
    assert contract["gae"] is False
    assert contract["action_sampling"] is False
    assert contract["minibatch_sampling"] is False
    assert contract["reward_normalisation"] is False
    assert contract["advantage_normalisation"] is False
