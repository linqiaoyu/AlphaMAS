"""TRAIN-only delayed per-symbol adaptation for frozen M2-PA-CTPPO-v2.

The runtime deliberately consumes the frozen TRAIN counterfactual tree rather
than mixed-period market files.  A credit maturing on a decision close is
applied only after that close's action has been irreversibly issued.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from scripts.m2.pa_ctppo import (
    EXPECTED_FAST_PARAMETERS,
    PPO_CLIP,
    SEED,
    VALUE_LOSS_COEFFICIENT,
    PromptAnchoredActorCritic,
    configure_determinism,
    deterministic_action,
    fast_checkpoint_sha,
)
from scripts.m2.rl_environment import TRAIN_SESSIONS, TRAIN_SYMBOLS
from scripts.m2.semantic_state_representation import ACTION_ORDER
from scripts.m2.train_global_pa_ctppo import canonical_parameter_sha, sha256_file

SCHEMA_VERSION = "M2-13-ONLINE-DELAYED-ADAPTATION-v1"
TREE_IDENTITY = "ff5e1bf21ef90b9eeec96d257c59e83656957f32ec4d2f94396a0a5d4bf1db13"
REPRESENTATION_IDENTITY = "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe"
REWARD_IDENTITY = "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"
C09_PARAMETER_SHA = "6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841"
C09_FILE_SHA = "56dc52128e1df9c9ddcf79fa6f7b293393bd61306ba4ef032f96cad6bf92126c"
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.5
FAST_PARAMETER_NAMES = (
    "residual_head.weight",
    "residual_head.bias",
    "gate_head.weight",
    "gate_head.bias",
    "value_head.weight",
    "value_head.bias",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def payload_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _tensor_payload(tensor: Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "hex": value.numpy().tobytes(order="C").hex(),
    }


_NUMPY_DTYPES = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int64": np.int64,
    "torch.int32": np.int32,
    "torch.bool": np.bool_,
}


def _tensor_from_payload(value: Mapping[str, Any]) -> Tensor:
    dtype = _NUMPY_DTYPES.get(str(value["dtype"]))
    if dtype is None:
        raise ValueError(f"unsupported serialised tensor dtype: {value['dtype']}")
    array = np.frombuffer(bytes.fromhex(str(value["hex"])), dtype=dtype).copy()
    return torch.from_numpy(array.reshape(tuple(value["shape"])))


def _encode_optimizer_state(optimiser: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimiser.state_dict()
    return {
        "state": {
            str(key): {
                name: _tensor_payload(item) if isinstance(item, Tensor) else item
                for name, item in sorted(value.items())
            }
            for key, value in sorted(state["state"].items())
        },
        "param_groups": state["param_groups"],
    }


def _decode_optimizer_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": {
            int(key): {
                name: _tensor_from_payload(item)
                if isinstance(item, Mapping) and {"dtype", "shape", "hex"} <= set(item)
                else item
                for name, item in state.items()
            }
            for key, state in value["state"].items()
        },
        "param_groups": value["param_groups"],
    }


class CreditStatus(str, Enum):
    PENDING = "PENDING"
    MATURED = "MATURED"
    APPLIED = "APPLIED"
    ARCHIVED = "ARCHIVED"


@dataclass
class DelayedCredit:
    event_id: str
    symbol: str
    origin_decision_session: str
    origin_node_id: str
    origin_observation_identity: str
    semantic_state_identity: str
    portfolio_state_identity: str
    prompt_action: str
    prompt_prior: tuple[float, float, float]
    pi_old: tuple[float, float, float]
    expected_maturity_session: str
    reward_identity: str = REWARD_IDENTITY
    status: CreditStatus = CreditStatus.PENDING
    counterfactual_r3: tuple[float, float, float] | None = None
    selected_action: str | None = None
    application_id: str | None = None
    pre_fast_sha: str | None = None
    post_fast_sha: str | None = None

    def mature(self, rewards: Iterable[float]) -> None:
        if self.status is not CreditStatus.PENDING:
            raise ValueError("only a PENDING credit may mature")
        values = tuple(float(item) for item in rewards)
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise ValueError("matured BUY/HOLD/SELL credits must be finite")
        self.counterfactual_r3 = values
        self.status = CreditStatus.MATURED


@dataclass(frozen=True)
class TrainNode:
    node_id: str
    symbol: str
    decision_session: str
    depth: int
    observation: Tensor
    observation_sha: str
    semantic_sha: str
    portfolio_sha: str


@dataclass(frozen=True)
class TrainEdge:
    action: str
    child_node_id: str | None
    local_r3: float
    reward_maturity_session: str
    credit_input_sha: str


class FrozenTrainTree:
    """Losslessly reconstruct only the 56-decision TRAIN tree."""

    def __init__(self, experiments_root: Path) -> None:
        base = experiments_root / "experiments/M2/development"
        tree_root = base / "rl_architecture_v2/train_environment"
        representation_root = base / "semantic_state_representation_v1"
        manifest = json.loads((tree_root / "tree_manifest.json").read_text())
        if manifest.get("train_counterfactual_tree_v2_identity_sha256") != TREE_IDENTITY:
            raise RuntimeError("frozen TRAIN tree identity mismatch")
        if manifest.get("representation_identity") != REPRESENTATION_IDENTITY:
            raise RuntimeError("frozen representation identity mismatch")
        if manifest.get("reward_id") != REWARD_IDENTITY:
            raise RuntimeError("frozen reward identity mismatch")
        identity = json.loads((representation_root / "representation_identity.json").read_text())
        if identity.get("semantic_state_representation_identity_sha256") != REPRESENTATION_IDENTITY:
            raise RuntimeError("representation freeze mismatch")
        rows = json.loads((representation_root / "manifests/row_index.json").read_text())
        train_rows = {row["row_index"]: row for row in rows if row.get("role") == "TRAIN"}
        if len(train_rows) != 56:
            raise RuntimeError("TRAIN membership must contain exactly 56 rows")
        semantic = np.load(
            representation_root / "embeddings/semantic_base.npy", mmap_mode="r", allow_pickle=False
        )
        with gzip.open(tree_root / "nodes.jsonl.gz", "rt", encoding="utf-8") as stream:
            raw_nodes = [json.loads(line) for line in stream]
        with gzip.open(tree_root / "edges.jsonl.gz", "rt", encoding="utf-8") as stream:
            raw_edges = [json.loads(line) for line in stream]
        self.nodes: dict[str, TrainNode] = {}
        for raw in raw_nodes:
            row = train_rows.get(raw["semantic_row"]["index"])
            if row is None or row["case_id"] != raw["semantic_row"]["identity"]:
                raise RuntimeError("tree references a non-TRAIN semantic row")
            portfolio = raw["portfolio_snapshot"]
            dynamic = np.array(
                [
                    float(portfolio["quantity"] <= 1e-12),
                    float(portfolio["quantity"] > 1e-12),
                    0.0
                    if portfolio["quantity"] <= 1e-12
                    else math.log(
                        ((portfolio["current_equity"] - portfolio["cash"]) / portfolio["quantity"])
                        / portfolio["average_entry_price"]
                    ),
                    portfolio["current_drawdown"],
                ],
                dtype=np.float32,
            )
            observation = np.concatenate(
                (np.asarray(semantic[row["row_index"]], dtype=np.float32), dynamic)
            ).astype(np.float32, copy=False)
            observation_sha = hashlib.sha256(observation.tobytes()).hexdigest()
            if observation_sha != raw["actor_observation"]["sha256"]:
                raise RuntimeError("TRAIN Actor observation identity mismatch")
            portfolio_sha = payload_sha(portfolio)
            self.nodes[raw["node_id"]] = TrainNode(
                node_id=raw["node_id"],
                symbol=raw["symbol"],
                decision_session=raw["decision_session"],
                depth=raw["depth"],
                observation=torch.from_numpy(observation.copy()),
                observation_sha=observation_sha,
                semantic_sha=raw["semantic_base_sha256"],
                portfolio_sha=portfolio_sha,
            )
        self.edges: dict[str, dict[str, TrainEdge]] = {}
        for raw in raw_edges:
            self.edges.setdefault(raw["parent_node_id"], {})[raw["action"]] = TrainEdge(
                action=raw["action"],
                child_node_id=raw["child_node_id"],
                local_r3=float(raw["local_r3"]),
                reward_maturity_session=raw["reward_maturity_session"],
                credit_input_sha=raw["credit_input_sha256"],
            )
        if len(self.nodes) != 8744 or sum(map(len, self.edges.values())) != 26232:
            raise RuntimeError("frozen TRAIN tree population mismatch")
        self.roots = {node.symbol: node.node_id for node in self.nodes.values() if node.depth == 0}
        if tuple(self.roots) != TRAIN_SYMBOLS:
            raise RuntimeError("TRAIN root symbol ordering changed")

    def action_edges(self, node_id: str) -> tuple[TrainEdge, TrainEdge, TrainEdge]:
        edges = self.edges[node_id]
        return tuple(edges[action] for action in ACTION_ORDER)  # type: ignore[return-value]


def load_c09(checkpoint: Path) -> PromptAnchoredActorCritic:
    if sha256_file(checkpoint) != C09_FILE_SHA:
        raise RuntimeError("C09 model.pt SHA mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PromptAnchoredActorCritic()
    model.load_state_dict(payload["model_state"], strict=True)
    if canonical_parameter_sha(model) != C09_PARAMETER_SHA:
        raise RuntimeError("C09 canonical parameter SHA mismatch")
    return model


def _global_state(model: PromptAnchoredActorCritic) -> dict[str, Tensor]:
    fast = set(FAST_PARAMETER_NAMES)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name not in fast
    }


@dataclass
class SymbolOnlineState:
    symbol: str
    model: PromptAnchoredActorCritic
    optimiser: torch.optim.AdamW
    next_node_id: str | None
    next_decision_index: int = 0
    credits: list[DelayedCredit] = field(default_factory=list)
    applied_event_ids: set[str] = field(default_factory=set)
    issued_actions: list[dict[str, Any]] = field(default_factory=list)
    update_records: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def fresh(
        cls, symbol: str, base_model: PromptAnchoredActorCritic, root: str, learning_rate: float
    ) -> SymbolOnlineState:
        model = copy.deepcopy(base_model)
        model.freeze_global_for_formal()
        parameters = [parameter for _, parameter in model.fast_named_parameters()]
        if tuple(name for name, _ in model.fast_named_parameters()) != FAST_PARAMETER_NAMES:
            raise RuntimeError("authorised fast-parameter set changed")
        if sum(item.numel() for item in parameters) != EXPECTED_FAST_PARAMETERS:
            raise RuntimeError("authorised fast-parameter count changed")
        optimiser = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)
        return cls(symbol=symbol, model=model, optimiser=optimiser, next_node_id=root)

    def optimiser_sha(self) -> str:
        return payload_sha(_encode_optimizer_state(self.optimiser))

    def state_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "symbol": self.symbol,
            "next_node_id": self.next_node_id,
            "next_decision_index": self.next_decision_index,
            "fast_parameters": {
                name: _tensor_payload(parameter)
                for name, parameter in self.model.fast_named_parameters()
            },
            "optimiser_state": _encode_optimizer_state(self.optimiser),
            "credits": [{**asdict(item), "status": item.status.value} for item in self.credits],
            "applied_event_ids": sorted(self.applied_event_ids),
            "issued_actions": self.issued_actions,
            "update_records": self.update_records,
        }
        payload["state_identity"] = payload_sha(payload)
        return payload

    def save(self, path: Path) -> str:
        payload = self.state_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(payload))
        return payload["state_identity"]

    @classmethod
    def load(
        cls, path: Path, base_model: PromptAnchoredActorCritic, learning_rate: float
    ) -> SymbolOnlineState:
        payload = json.loads(path.read_text())
        identity = payload.pop("state_identity")
        if payload_sha(payload) != identity:
            raise RuntimeError("online-state identity mismatch")
        result = cls.fresh(payload["symbol"], base_model, payload["next_node_id"], learning_rate)
        result.next_node_id = payload["next_node_id"]
        result.next_decision_index = payload["next_decision_index"]
        named = dict(result.model.fast_named_parameters())
        for name, tensor in payload["fast_parameters"].items():
            named[name].data.copy_(_tensor_from_payload(tensor))
        result.optimiser.load_state_dict(_decode_optimizer_state(payload["optimiser_state"]))
        result.credits = [
            DelayedCredit(
                **{
                    **item,
                    "status": CreditStatus(item["status"]),
                    "prompt_prior": tuple(item["prompt_prior"]),
                    "pi_old": tuple(item["pi_old"]),
                    "counterfactual_r3": None
                    if item["counterfactual_r3"] is None
                    else tuple(item["counterfactual_r3"]),
                }
            )
            for item in payload["credits"]
        ]
        result.applied_event_ids = set(payload["applied_event_ids"])
        result.issued_actions = payload["issued_actions"]
        result.update_records = payload["update_records"]
        return result


def _apply_credit(
    state: SymbolOnlineState, tree: FrozenTrainTree, credit: DelayedCredit, epochs: int
) -> dict[str, Any]:
    if credit.status is not CreditStatus.MATURED or credit.counterfactual_r3 is None:
        raise RuntimeError("only a matured credit may be applied")
    if credit.event_id in state.applied_event_ids:
        raise RuntimeError("duplicate credit application")
    observation = tree.nodes[credit.origin_node_id].observation
    old = torch.tensor(credit.pi_old, dtype=observation.dtype)
    rewards = torch.tensor(credit.counterfactual_r3, dtype=observation.dtype)
    value_target = torch.sum(old * rewards).detach()
    advantages = (rewards - value_target).detach()
    global_before = _global_state(state.model)
    pre_fast = fast_checkpoint_sha(state.model)
    losses: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(epochs):
        output = state.model(observation.unsqueeze(0))
        probabilities = output["probabilities"].squeeze(0)
        ratio = probabilities / old
        surrogate = torch.sum(
            old
            * torch.minimum(
                ratio * advantages,
                torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages,
            )
        )
        loss = (
            -surrogate
            + VALUE_LOSS_COEFFICIENT * (output["value"].squeeze(0) - value_target).square()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite online loss")
        state.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [parameter for _, parameter in state.model.fast_named_parameters()]
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        ):
            raise FloatingPointError("non-finite online gradient")
        norm = float(nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM))
        state.optimiser.step()
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError("non-finite online parameter")
        losses.append(float(loss.detach()))
        gradient_norms.append(norm)
    for name, before in global_before.items():
        if not torch.equal(before, dict(state.model.named_parameters())[name].detach().cpu()):
            raise RuntimeError(f"global C09 parameter changed: {name}")
    post_fast = fast_checkpoint_sha(state.model)
    application_id = payload_sha(
        {"event_id": credit.event_id, "pre_fast_sha": pre_fast, "post_fast_sha": post_fast}
    )
    credit.status = CreditStatus.APPLIED
    credit.application_id = application_id
    credit.pre_fast_sha = pre_fast
    credit.post_fast_sha = post_fast
    state.applied_event_ids.add(credit.event_id)
    record = {
        "event_id": credit.event_id,
        "application_id": application_id,
        "pre_fast_sha": pre_fast,
        "post_fast_sha": post_fast,
        "losses": losses,
        "gradient_norms_before_clipping": gradient_norms,
        "optimiser_sha": state.optimiser_sha(),
    }
    state.update_records.append(record)
    return record


def process_decision(
    state: SymbolOnlineState, tree: FrozenTrainTree, *, epochs: int
) -> dict[str, Any]:
    if state.next_node_id is None or state.next_decision_index >= len(TRAIN_SESSIONS):
        raise RuntimeError("no unissued TRAIN decision remains")
    node = tree.nodes[state.next_node_id]
    expected_session = TRAIN_SESSIONS[state.next_decision_index]
    if node.symbol != state.symbol or node.decision_session != expected_session:
        raise RuntimeError("online chronology drift")
    with torch.no_grad():
        output = state.model(node.observation)
        probabilities_tensor = output["probabilities"]
    if not torch.isfinite(probabilities_tensor).all():
        raise FloatingPointError("non-finite action probabilities")
    probabilities = tuple(float(item) for item in probabilities_tensor.tolist())
    prompt_index = int(torch.argmax(node.observation[3073:3076]))
    prompt_action = ACTION_ORDER[prompt_index]
    action = deterministic_action(probabilities_tensor, prompt_index)
    edges = tree.action_edges(node.node_id)
    selected_edge = edges[ACTION_ORDER.index(action)]
    event_id = payload_sha(
        {"symbol": state.symbol, "session": node.decision_session, "node_id": node.node_id}
    )
    prompt_prior = tuple(2 / 3 if i == prompt_index else 1 / 6 for i in range(3))
    credit = DelayedCredit(
        event_id=event_id,
        symbol=state.symbol,
        origin_decision_session=node.decision_session,
        origin_node_id=node.node_id,
        origin_observation_identity=node.observation_sha,
        semantic_state_identity=node.semantic_sha,
        portfolio_state_identity=node.portfolio_sha,
        prompt_action=prompt_action,
        prompt_prior=prompt_prior,
        pi_old=probabilities,
        expected_maturity_session=selected_edge.reward_maturity_session,
        selected_action=action,
    )
    state.credits.append(credit)
    record = {
        "symbol": state.symbol,
        "decision_session": node.decision_session,
        "decision_index": state.next_decision_index,
        "node_id": node.node_id,
        "origin_observation_identity": node.observation_sha,
        "semantic_state_identity": node.semantic_sha,
        "portfolio_state_identity": node.portfolio_sha,
        "prompt_action": prompt_action,
        "actor_probabilities": list(probabilities),
        "deterministic_selected_action": action,
        "prompt_override": action != prompt_action,
        "event_id": event_id,
        "reward_maturity_session": selected_edge.reward_maturity_session,
        "selected_action_local_r3": None,
        "fast_sha_at_action": fast_checkpoint_sha(state.model),
    }
    state.issued_actions.append(record)
    state.next_node_id = selected_edge.child_node_id
    state.next_decision_index += 1

    # Strict ordering: same-close rewards become visible after this action.
    for pending in state.credits:
        if (
            pending.status is CreditStatus.PENDING
            and pending.expected_maturity_session <= node.decision_session
        ):
            rewards = tuple(edge.local_r3 for edge in tree.action_edges(pending.origin_node_id))
            pending.mature(rewards)
            issued = next(
                item for item in state.issued_actions if item["event_id"] == pending.event_id
            )
            issued["selected_action_local_r3"] = rewards[
                ACTION_ORDER.index(pending.selected_action)
            ]
            if state.next_decision_index < len(TRAIN_SESSIONS):
                _apply_credit(state, tree, pending, epochs)
            else:
                pending.status = CreditStatus.ARCHIVED
    return record


def archive_terminal_credits(state: SymbolOnlineState, tree: FrozenTrainTree) -> None:
    """Mature score-only terminal credits without post-horizon optimisation."""

    for pending in state.credits:
        if pending.status is CreditStatus.PENDING:
            rewards = tuple(edge.local_r3 for edge in tree.action_edges(pending.origin_node_id))
            pending.mature(rewards)
            issued = next(
                item for item in state.issued_actions if item["event_id"] == pending.event_id
            )
            issued["selected_action_local_r3"] = rewards[
                ACTION_ORDER.index(pending.selected_action)
            ]
            pending.status = CreditStatus.ARCHIVED


def run_symbol(
    symbol: str,
    base_model: PromptAnchoredActorCritic,
    tree: FrozenTrainTree,
    *,
    learning_rate: float,
    epochs: int,
) -> SymbolOnlineState:
    configure_determinism(SEED)
    state = SymbolOnlineState.fresh(symbol, base_model, tree.roots[symbol], learning_rate)
    while state.next_decision_index < len(TRAIN_SESSIONS):
        process_decision(state, tree, epochs=epochs)
    archive_terminal_credits(state, tree)
    if len(state.issued_actions) != 7 or any(
        item["selected_action_local_r3"] is None for item in state.issued_actions
    ):
        raise RuntimeError("symbol run did not produce seven scored TRAIN decisions")
    return state
