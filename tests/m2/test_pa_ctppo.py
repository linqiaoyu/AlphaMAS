from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from scripts.m2.pa_ctppo import (  # noqa: E402
    EXPECTED_FAST_PARAMETERS,
    EXPECTED_PARAMETERS,
    METHOD_ID,
    RESIDUAL_BOUND,
    ExperienceStatus,
    OnlineExperience,
    PromptAnchoredActorCritic,
    build_adapted_proposal,
    configure_determinism,
    deterministic_action,
    exact_local_credit_policy_evaluation,
    exact_ppo_loss,
    fast_checkpoint_sha,
    online_fast_update,
    per_symbol_fast_adapters,
)


def _observation(prompt_index: int = 0) -> torch.Tensor:
    observation = torch.zeros(3080)
    observation[0:1024] = torch.linspace(-1, 1, 1024)
    observation[1024:2048] = torch.linspace(1, -1, 1024)
    observation[2048:3072] = observation[1024:2048] - observation[0:1024]
    observation[3072] = 0.25
    observation[3073 + prompt_index] = 1.0
    observation[3076] = 1.0
    return observation


def test_exact_architecture_slicing_sharing_and_parameter_counts() -> None:
    configure_determinism()
    model = PromptAnchoredActorCritic()
    slices = model.split_observation(_observation())
    assert tuple(value.shape[-1] for value in slices) == (1024, 1024, 1024, 1, 3, 4)
    assert model.compact_features(_observation()).shape == (56,)
    assert model.actor_trunk.in_features == model.critic_trunk.in_features == 56
    assert model.actor_trunk.out_features == model.critic_trunk.out_features == 32
    assert model.semantic_adapter is model.semantic_adapter
    assert model.parameter_count() == EXPECTED_PARAMETERS == 20_197
    assert model.fast_parameter_count() == EXPECTED_FAST_PARAMETERS == 165
    assert METHOD_ID == "M2-PA-CTPPO-v2"


@pytest.mark.parametrize("prompt_index", [0, 1, 2])
def test_initial_policy_is_exact_full_support_prompt_prior(prompt_index: int) -> None:
    model = PromptAnchoredActorCritic()
    output = model(_observation(prompt_index))
    expected = torch.full((3,), 1 / 6)
    expected[prompt_index] = 2 / 3
    assert torch.allclose(output["probabilities"], expected, atol=1e-7, rtol=0)
    assert torch.all(output["probabilities"] > 0)
    assert float(output["probabilities"].sum().detach()) == pytest.approx(1.0)
    assert torch.equal(output["delta_logits"], torch.zeros(3))


def test_residual_gate_and_policy_bounds_are_finite() -> None:
    model = PromptAnchoredActorCritic()
    with torch.no_grad():
        model.residual_head.bias.fill_(1000)
        model.gate_head.bias.fill_(1000)
    output = model(_observation())
    assert torch.all(output["delta_logits"] <= RESIDUAL_BOUND)
    assert torch.all(output["delta_logits"] >= -RESIDUAL_BOUND)
    assert 0 <= output["gate"] <= 1
    assert all(torch.isfinite(value).all() for value in output.values())


def test_deterministic_argmax_and_prompt_first_tie_break() -> None:
    assert deterministic_action(torch.tensor([0.2, 0.6, 0.2]), 0) == "HOLD"
    assert deterministic_action(torch.tensor([0.5, 0.5, 0.0]), 1) == "HOLD"
    assert deterministic_action(torch.tensor([0.5, 0.5, 0.0]), 2) == "BUY"


def _synthetic_tree():
    rewards = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 1.0, -2.0]],
        dtype=torch.float64,
    )
    children = torch.full((2, 3), -1)
    old = torch.tensor(
        [[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=torch.float64
    )
    depths = torch.tensor([0, 0])
    roots = torch.tensor([0, 1])
    return rewards, children, old, depths, roots


def test_exact_local_value_advantage_and_occupancy() -> None:
    evaluation = exact_local_credit_policy_evaluation(*_synthetic_tree())
    rewards, _, old, _, _ = _synthetic_tree()
    expected_values = torch.tensor([0.25, -0.3], dtype=torch.float64)
    assert torch.allclose(evaluation.values, expected_values)
    assert torch.allclose(evaluation.advantages, rewards - expected_values[:, None])
    assert torch.allclose(evaluation.occupancy, torch.tensor([0.5, 0.5], dtype=torch.float64))
    old = _synthetic_tree()[2]
    assert torch.allclose(
        torch.sum(old * evaluation.advantages, dim=1),
        torch.zeros(2, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


def test_hand_computable_clipped_objective_and_synthetic_backward() -> None:
    evaluation = exact_local_credit_policy_evaluation(*_synthetic_tree())
    model = PromptAnchoredActorCritic()
    observations = torch.stack([_observation(0), _observation(1)])
    output = model(observations)
    old = _synthetic_tree()[2].float()
    new = torch.tensor([[0.7, 0.15, 0.15], [0.1, 0.4, 0.5]])
    predicted = torch.tensor([0.5, -0.1], requires_grad=True)
    losses = exact_ppo_loss(
        new,
        old,
        evaluation.advantages.float(),
        predicted,
        evaluation.values.float(),
        evaluation.occupancy.float(),
    )
    ratio = new / old
    expected_surrogate = (
        evaluation.occupancy.float()
        * torch.sum(
            old
            * torch.minimum(
                ratio * evaluation.advantages.float(),
                torch.clamp(ratio, 0.8, 1.2) * evaluation.advantages.float(),
            ),
            dim=1,
        )
    ).sum() / evaluation.occupancy.sum()
    expected_value_loss = (
        evaluation.occupancy.float()
        * (predicted - evaluation.values.float()).square()
    ).sum() / evaluation.occupancy.sum()
    assert float(losses["policy_surrogate"].detach()) == pytest.approx(
        float(expected_surrogate.detach())
    )
    assert float(losses["value_loss"].detach()) == pytest.approx(
        float(expected_value_loss.detach())
    )
    assert float(losses["loss"].detach()) == pytest.approx(
        float((-expected_surrogate + 0.5 * expected_value_loss).detach())
    )

    backward_losses = exact_ppo_loss(
        output["probabilities"],
        old,
        evaluation.advantages.float(),
        output["value"],
        evaluation.values.float(),
        evaluation.occupancy.float(),
    )
    backward_losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert losses["ratio"].shape == (2, 3)
    assert model.residual_head.weight.grad is not None
    assert model.value_head.weight.grad is not None


def test_online_lifecycle_fast_only_and_symbol_isolation() -> None:
    base = PromptAnchoredActorCritic()
    adapters = per_symbol_fast_adapters(base, ("AAPL", "AMZN", "JPM"))
    amzn_before = fast_checkpoint_sha(adapters["AMZN"])
    jpm_before = fast_checkpoint_sha(adapters["JPM"])
    observation = _observation()
    probabilities = tuple(
        float(value) for value in adapters["AAPL"](observation)["probabilities"].detach()
    )
    experience = OnlineExperience("AAPL", "2024-01-05", "2024-01-12", probabilities, "v1")
    with pytest.raises(ValueError, match="cannot update"):
        online_fast_update(
            adapters["AAPL"], observation, experience, learning_rate=1e-3, epochs=1, event_identity="x"
        )
    experience.mature((0.1, 0.0, -0.1))
    global_before = {
        name: value.detach().clone()
        for name, value in adapters["AAPL"].named_parameters()
        if not value.requires_grad
    }
    audit = online_fast_update(
        adapters["AAPL"], observation, experience, learning_rate=1e-3, epochs=1, event_identity="x"
    )
    assert experience.status is ExperienceStatus.APPLIED
    assert experience.pre_update_checkpoint_sha != experience.post_update_checkpoint_sha
    assert audit["advantage_mean"] == pytest.approx(0.0, abs=1e-7)
    assert fast_checkpoint_sha(adapters["AMZN"]) == amzn_before
    assert fast_checkpoint_sha(adapters["JPM"]) == jpm_before
    for name, before in global_before.items():
        assert torch.equal(before, dict(adapters["AAPL"].named_parameters())[name])
    with pytest.raises(ValueError, match="cannot update"):
        online_fast_update(
            adapters["AAPL"], observation, experience, learning_rate=1e-3, epochs=1, event_identity="y"
        )


def test_adapted_proposal_preserves_original_and_exposes_no_hidden_internals() -> None:
    original = "Reasoning bytes: café\nFINAL TRANSACTION PROPOSAL: **BUY**"
    proposal = build_adapted_proposal(original, "BUY", "SELL")
    assert proposal.prompt_trader_proposal_original.encode() == original.encode()
    assert proposal.prompt_trader_proposal_sha256 == hashlib.sha256(original.encode()).hexdigest()
    assert proposal.calibration == "OVERRIDE"
    assert proposal.downstream_text.endswith("**SELL**")
    for forbidden in ("R3", "Critic", "probabilities", "gate", "checkpoint", "advantage"):
        assert forbidden not in proposal.downstream_text
    retained = build_adapted_proposal(original, "BUY", "BUY")
    assert retained.calibration == "RETAIN"
