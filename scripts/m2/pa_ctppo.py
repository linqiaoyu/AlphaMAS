"""Frozen Prompt-Anchored Counterfactual Tree PPO architecture and objectives."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn

from scripts.m2.semantic_state_representation import ACTION_ORDER

METHOD_ID = "M2-PA-CTPPO-v1"
OBSERVATION_DIMENSION = 3080
SEMANTIC_DIMENSION = 1024
ADAPTER_DIMENSION = 16
COMPACT_DIMENSION = 56
HIDDEN_DIMENSION = 32
RESIDUAL_BOUND = math.log(4.0)
PROMPT_SELECTED_PROBABILITY = 2.0 / 3.0
PROMPT_ALTERNATIVE_PROBABILITY = 1.0 / 6.0
GAMMA = 1.0
PPO_CLIP = 0.2
VALUE_LOSS_COEFFICIENT = 0.5
SEED = 20260816
EXPECTED_PARAMETERS = 20_197
EXPECTED_FAST_PARAMETERS = 165
TIE_TOLERANCE = 1e-12


def configure_determinism(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _orthogonal(linear: nn.Linear) -> None:
    nn.init.orthogonal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class PromptAnchoredActorCritic(nn.Module):
    """20,197-parameter Actor/Critic sharing only one semantic projection."""

    def __init__(self) -> None:
        super().__init__()
        self.semantic_adapter = nn.Linear(SEMANTIC_DIMENSION, ADAPTER_DIMENSION, bias=False)
        self.actor_trunk = nn.Linear(COMPACT_DIMENSION, HIDDEN_DIMENSION)
        self.critic_trunk = nn.Linear(COMPACT_DIMENSION, HIDDEN_DIMENSION)
        self.residual_head = nn.Linear(HIDDEN_DIMENSION, 3)
        self.gate_head = nn.Linear(HIDDEN_DIMENSION, 1)
        self.value_head = nn.Linear(HIDDEN_DIMENSION, 1)
        _orthogonal(self.semantic_adapter)
        _orthogonal(self.actor_trunk)
        _orthogonal(self.critic_trunk)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        _orthogonal(self.value_head)
        if self.parameter_count() != EXPECTED_PARAMETERS:
            raise AssertionError("frozen PA-CTPPO parameter count changed")

    @staticmethod
    def split_observation(observation: Tensor) -> tuple[Tensor, ...]:
        if observation.shape[-1] != OBSERVATION_DIMENSION:
            raise ValueError("Actor observation must have 3080 features")
        if not torch.is_floating_point(observation) or not torch.isfinite(observation).all():
            raise ValueError("Actor observation must contain finite floats")
        return (
            observation[..., 0:1024],
            observation[..., 1024:2048],
            observation[..., 2048:3072],
            observation[..., 3072:3073],
            observation[..., 3073:3076],
            observation[..., 3076:3080],
        )

    def compact_features(self, observation: Tensor) -> Tensor:
        z_rm, z_pt, delta_z, agreement, prompt_action, portfolio = self.split_observation(
            observation
        )
        features = torch.cat(
            (
                torch.tanh(self.semantic_adapter(z_rm)),
                torch.tanh(self.semantic_adapter(z_pt)),
                torch.tanh(self.semantic_adapter(delta_z)),
                agreement,
                prompt_action,
                portfolio,
            ),
            dim=-1,
        )
        if features.shape[-1] != COMPACT_DIMENSION:
            raise AssertionError("compact feature dimension changed")
        return features

    def forward(self, observation: Tensor) -> dict[str, Tensor]:
        features = self.compact_features(observation)
        actor_hidden = torch.tanh(self.actor_trunk(features))
        critic_hidden = torch.tanh(self.critic_trunk(features))
        delta_logits = RESIDUAL_BOUND * torch.tanh(self.residual_head(actor_hidden))
        gate = torch.sigmoid(self.gate_head(actor_hidden))
        prompt_onehot = observation[..., 3073:3076]
        if torch.any(torch.abs(prompt_onehot.sum(dim=-1) - 1.0) > 1e-6):
            raise ValueError("Prompt action slice must be one-hot")
        prompt_prior = prompt_onehot * PROMPT_SELECTED_PROBABILITY + (
            1.0 - prompt_onehot
        ) * PROMPT_ALTERNATIVE_PROBABILITY
        logits = torch.log(prompt_prior) + gate * delta_logits
        probabilities = torch.softmax(logits, dim=-1)
        value = self.value_head(critic_hidden).squeeze(-1)
        return {
            "policy_logits": logits,
            "probabilities": probabilities,
            "value": value,
            "gate": gate.squeeze(-1),
            "delta_logits": delta_logits,
            "compact_features": features,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def fast_named_parameters(self):
        for module_name in ("residual_head", "gate_head", "value_head"):
            module = getattr(self, module_name)
            for name, parameter in module.named_parameters():
                yield f"{module_name}.{name}", parameter

    def fast_parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.fast_named_parameters())

    def freeze_global_for_formal(self) -> None:
        fast_names = {name for name, _ in self.fast_named_parameters()}
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name in fast_names
        if self.fast_parameter_count() != EXPECTED_FAST_PARAMETERS:
            raise AssertionError("frozen Formal fast-parameter count changed")


def deterministic_action(probabilities: Tensor, prompt_action_index: int) -> str:
    """Argmax with Prompt-first then BUY/HOLD/SELL deterministic tie-breaking."""

    if probabilities.shape != (3,) or not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be one finite three-action vector")
    maximum = float(probabilities.max())
    tied = [
        index
        for index, probability in enumerate(probabilities.tolist())
        if abs(float(probability) - maximum) <= TIE_TOLERANCE
    ]
    if prompt_action_index in tied:
        return ACTION_ORDER[prompt_action_index]
    return ACTION_ORDER[tied[0]]


@dataclass(frozen=True)
class TreePolicyEvaluation:
    q_values: Tensor
    values: Tensor
    advantages: Tensor
    occupancy: Tensor


def exact_tree_policy_evaluation(
    rewards: Tensor,
    child_indices: Tensor,
    old_probabilities: Tensor,
    depths: Tensor,
    root_indices: Tensor,
) -> TreePolicyEvaluation:
    """Evaluate a finite forest exactly; -1 child indices are terminal leaves."""

    node_count = rewards.shape[0]
    if rewards.shape != (node_count, 3) or old_probabilities.shape != (node_count, 3):
        raise ValueError("tree rewards and old policy must have shape [nodes,3]")
    if child_indices.shape != (node_count, 3) or depths.shape != (node_count,):
        raise ValueError("tree topology has inconsistent shapes")
    if torch.any(old_probabilities <= 0) or torch.any(
        torch.abs(old_probabilities.sum(dim=1) - 1.0) > 1e-12
    ):
        raise ValueError("old policy must have full support and sum to one")
    values = torch.zeros(node_count, dtype=rewards.dtype, device=rewards.device)
    q_values = torch.empty_like(rewards)
    max_depth = int(depths.max())
    for depth in range(max_depth, -1, -1):
        for node in torch.nonzero(depths == depth, as_tuple=False).flatten().tolist():
            future = torch.zeros(3, dtype=rewards.dtype, device=rewards.device)
            for action in range(3):
                child = int(child_indices[node, action])
                if child >= 0:
                    future[action] = values[child]
            q_values[node] = rewards[node] + GAMMA * future
            values[node] = torch.sum(old_probabilities[node] * q_values[node])
    advantages = q_values - values.unsqueeze(1)
    occupancy = torch.zeros(node_count, dtype=rewards.dtype, device=rewards.device)
    root_weight = 1.0 / int(root_indices.numel())
    occupancy[root_indices] = root_weight
    for depth in range(max_depth + 1):
        for node in torch.nonzero(depths == depth, as_tuple=False).flatten().tolist():
            for action in range(3):
                child = int(child_indices[node, action])
                if child >= 0:
                    occupancy[child] = occupancy[child] + (
                        occupancy[node] * old_probabilities[node, action]
                    )
    return TreePolicyEvaluation(q_values, values, advantages, occupancy)


def exact_ppo_loss(
    new_probabilities: Tensor,
    old_probabilities: Tensor,
    advantages: Tensor,
    predicted_values: Tensor,
    exact_values: Tensor,
    occupancy: Tensor,
) -> dict[str, Tensor]:
    """Exact three-action clipped surrogate plus occupancy-weighted value MSE."""

    ratio = new_probabilities / old_probabilities
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
    per_node_policy = torch.sum(old_probabilities * torch.minimum(unclipped, clipped), dim=1)
    normaliser = occupancy.sum()
    if not torch.isfinite(normaliser) or normaliser <= 0:
        raise ValueError("occupancy must have positive finite mass")
    policy = torch.sum(occupancy * per_node_policy) / normaliser
    value = torch.sum(occupancy * (predicted_values - exact_values).square()) / normaliser
    loss = -policy + VALUE_LOSS_COEFFICIENT * value
    if not all(torch.isfinite(item) for item in (policy, value, loss)):
        raise ValueError("PA-CTPPO loss must be finite")
    return {"loss": loss, "policy_surrogate": policy, "value_loss": value, "ratio": ratio}


class ExperienceStatus(str, Enum):
    PENDING = "PENDING"
    MATURED = "MATURED"
    APPLIED = "APPLIED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class OnlineExperience:
    symbol: str
    decision_session: str
    maturity_session: str
    policy_probabilities: tuple[float, float, float]
    policy_version: str
    status: ExperienceStatus = ExperienceStatus.PENDING
    counterfactual_r3: tuple[float, float, float] | None = None
    pre_update_checkpoint_sha: str | None = None
    post_update_checkpoint_sha: str | None = None
    update_event_sha: str | None = None

    def mature(self, rewards: tuple[float, float, float]) -> None:
        if self.status is not ExperienceStatus.PENDING:
            raise ValueError("only PENDING experience can mature")
        if len(rewards) != 3 or not all(math.isfinite(value) for value in rewards):
            raise ValueError("BUY/HOLD/SELL rewards must mature simultaneously and finitely")
        self.counterfactual_r3 = rewards
        self.status = ExperienceStatus.MATURED

    def _mark_applied(
        self,
        model: PromptAnchoredActorCritic,
        event_identity: str,
        pre_update_checkpoint_sha: str,
    ) -> None:
        if self.status is not ExperienceStatus.MATURED:
            raise ValueError("only MATURED experience can be applied")
        self.pre_update_checkpoint_sha = pre_update_checkpoint_sha
        self.status = ExperienceStatus.APPLIED
        self.post_update_checkpoint_sha = fast_checkpoint_sha(model)
        payload = (
            f"{self.symbol}|{self.decision_session}|{self.pre_update_checkpoint_sha}|"
            f"{self.post_update_checkpoint_sha}|{event_identity}"
        )
        self.update_event_sha = hashlib.sha256(payload.encode()).hexdigest()


def fast_checkpoint_sha(model: PromptAnchoredActorCritic) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.fast_named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def per_symbol_fast_adapters(
    model: PromptAnchoredActorCritic, symbols: tuple[str, ...]
) -> dict[str, PromptAnchoredActorCritic]:
    result = {symbol: copy.deepcopy(model) for symbol in symbols}
    for adapter in result.values():
        adapter.freeze_global_for_formal()
    return result


def online_fast_update(
    model: PromptAnchoredActorCritic,
    observation: Tensor,
    experience: OnlineExperience,
    *,
    learning_rate: float,
    epochs: int,
    event_identity: str,
) -> dict[str, float]:
    """Apply one matured state's exact counterfactual update to only 165 fast parameters.

    The caller must supply later-approved hyperparameters; this architecture module
    intentionally has no online learning-rate or epoch defaults.
    """

    if experience.status is not ExperienceStatus.MATURED:
        raise ValueError("PENDING or APPLIED experience cannot update")
    if experience.counterfactual_r3 is None:
        raise ValueError("matured experience lacks counterfactual rewards")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("online learning rate must be a positive finite value")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("online epochs must be a positive integer")
    if observation.shape != (OBSERVATION_DIMENSION,):
        raise ValueError("online update requires one 3080-dimensional state")

    model.freeze_global_for_formal()
    global_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    pre_sha = fast_checkpoint_sha(model)
    old_probabilities = torch.tensor(
        experience.policy_probabilities,
        dtype=observation.dtype,
        device=observation.device,
    )
    if torch.any(old_probabilities <= 0) or not torch.isclose(
        old_probabilities.sum(),
        torch.tensor(1.0, dtype=observation.dtype, device=observation.device),
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError("stored old policy must have full support and sum to one")
    rewards = torch.tensor(
        experience.counterfactual_r3,
        dtype=observation.dtype,
        device=observation.device,
    )
    value_target = torch.sum(old_probabilities * rewards).detach()
    advantages = (rewards - value_target).detach()
    optimiser = torch.optim.AdamW(
        [parameter for _, parameter in model.fast_named_parameters()],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    final_loss = torch.tensor(float("nan"), device=observation.device)
    for _ in range(epochs):
        output = model(observation.unsqueeze(0))
        new_probabilities = output["probabilities"].squeeze(0)
        ratio = new_probabilities / old_probabilities
        surrogate = torch.sum(
            old_probabilities
            * torch.minimum(
                ratio * advantages,
                torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages,
            )
        )
        value_loss = (output["value"].squeeze(0) - value_target).square()
        final_loss = -surrogate + VALUE_LOSS_COEFFICIENT * value_loss
        optimiser.zero_grad(set_to_none=True)
        final_loss.backward()
        nn.utils.clip_grad_norm_(
            [parameter for _, parameter in model.fast_named_parameters()], 0.5
        )
        optimiser.step()
    for name, before in global_before.items():
        if not torch.equal(before, dict(model.named_parameters())[name].detach()):
            raise AssertionError(f"Formal global parameter changed: {name}")
    experience._mark_applied(model, event_identity, pre_sha)
    return {
        "loss": float(final_loss.detach()),
        "value_target": float(value_target),
        "advantage_mean": float(torch.sum(old_probabilities * advantages)),
    }


@dataclass(frozen=True)
class AdaptedProposal:
    prompt_trader_proposal_original: str
    prompt_trader_action: str
    prompt_trader_proposal_sha256: str
    rl_adapted_action: str
    calibration: str
    downstream_text: str


def build_adapted_proposal(
    original: str, prompt_action: str, rl_action: str
) -> AdaptedProposal:
    """Produce the only deterministic downstream representation exposed by M2 RL."""

    if not isinstance(original, str) or not original:
        raise ValueError("original Prompt Trader proposal must be non-empty")
    if prompt_action not in ACTION_ORDER or rl_action not in ACTION_ORDER:
        raise ValueError("actions must be BUY, HOLD, or SELL")
    calibration = "RETAIN" if prompt_action == rl_action else "OVERRIDE"
    rendered = (
        "### Original Prompt Trader Proposal\n"
        f"{original}\n\n"
        "### M2 RL Calibration\n"
        f"Prompt Trader Action: {prompt_action}\n"
        f"RL-Adapted Trader Action: {rl_action}\n"
        f"Calibration: {calibration}\n\n"
        "The RL layer modifies only the action recommendation.\n"
        "The original Prompt Trader reasoning is retained verbatim above.\n\n"
        f"M2 RL-ADAPTED FINAL TRANSACTION PROPOSAL: **{rl_action}**"
    )
    return AdaptedProposal(
        prompt_trader_proposal_original=original,
        prompt_trader_action=prompt_action,
        prompt_trader_proposal_sha256=hashlib.sha256(original.encode()).hexdigest(),
        rl_adapted_action=rl_action,
        calibration=calibration,
        downstream_text=rendered,
    )


def render_adapted_proposal(original: str, prompt_action: str, rl_action: str) -> str:
    return build_adapted_proposal(original, prompt_action, rl_action).downstream_text
