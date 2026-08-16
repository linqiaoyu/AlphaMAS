#!/usr/bin/env python3
"""Deterministic TRAIN-only global M2-PA-CTPPO-v2 optimisation.

This module is deliberately an orchestration layer around the frozen M2-10A
architecture and objective.  It reconstructs observations from the frozen tree,
creates one canonical initial checkpoint, and trains one exact full-tree LR
lineage at a time.  Validation and protected evaluation data have no loader here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import Tensor, nn

from scripts.m2.pa_ctppo import (
    EXPECTED_PARAMETERS,
    METHOD_ID,
    PPO_CLIP,
    PROMPT_ALTERNATIVE_PROBABILITY,
    PROMPT_SELECTED_PROBABILITY,
    SEED,
    PromptAnchoredActorCritic,
    configure_determinism,
    exact_ppo_loss,
)

TASK_ID = "M2-11"
TREE_CONTRACT = "M2_TRAIN_COUNTERFACTUAL_TREE-v2"
TREE_IDENTITY = "ff5e1bf21ef90b9eeec96d257c59e83656957f32ec4d2f94396a0a5d4bf1db13"
REPRESENTATION_IDENTITY = "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe"
REWARD_ID = "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"
STARTING_SOURCE_SHA = "e37456f871b461ed913a13501805f99750e41b0b"
STARTING_EXPERIMENTS_SHA = "1569c97d3285b5f5e22c21fb1e97c00b1ed015c8"
EXPECTED_NODES = 8_744
EXPECTED_ACTION_CREDITS = 26_232
EXPECTED_DEPTHS = 7
EXPECTED_ROOTS = 8
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
CHECKPOINT_ITERATIONS = (25, 50, 100)
OUTER_ITERATIONS = 100
OPTIMISATION_EPOCHS = 4
WEIGHT_DECAY = 1e-4
GRADIENT_NORM_CEILING = 0.5
CANONICAL_LOCAL_ROOT = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-m2")
FORBIDDEN_DESKTOP_ROOT = Path("/Users/yulinqiao/Desktop/FTIPFinal/AlphaMAS")
ACTION_ORDER = ("BUY", "HOLD", "SELL")
CANDIDATE_IDS = {
    (1e-4, 25): "C01",
    (1e-4, 50): "C02",
    (1e-4, 100): "C03",
    (3e-4, 25): "C04",
    (3e-4, 50): "C05",
    (3e-4, 100): "C06",
    (1e-3, 25): "C07",
    (1e-3, 50): "C08",
    (1e-3, 100): "C09",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def canonical_parameter_sha(
    model_or_parameters: nn.Module | Iterable[tuple[str, Tensor]],
) -> str:
    """Hash sorted named parameters with name, shape, dtype, and raw bytes."""

    parameters = (
        model_or_parameters.named_parameters()
        if isinstance(model_or_parameters, nn.Module)
        else model_or_parameters
    )
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        tensor = parameter.detach().cpu().contiguous()
        _length_prefixed(digest, name.encode("utf-8"))
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", dimension))
        _length_prefixed(digest, str(tensor.dtype).encode("ascii"))
        _length_prefixed(digest, tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def actor_parameter_sha(model: PromptAnchoredActorCritic) -> str:
    prefixes = ("semantic_adapter.", "actor_trunk.", "residual_head.", "gate_head.")
    return canonical_parameter_sha(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def critic_parameter_sha(model: PromptAnchoredActorCritic) -> str:
    prefixes = ("semantic_adapter.", "critic_trunk.", "value_head.")
    return canonical_parameter_sha(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def assert_not_desktop_checkout(source_root: Path) -> Path:
    resolved = source_root.resolve()
    forbidden = FORBIDDEN_DESKTOP_ROOT.resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise RuntimeError("canonical path guard rejected the legacy Desktop checkout")
    return resolved


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_exact_source_lineage(source_root: Path, trainer_sha: str) -> None:
    assert_not_desktop_checkout(source_root)
    if _git(source_root, "rev-parse", "HEAD") != trainer_sha:
        raise RuntimeError("AlphaMAS actual HEAD does not match M2_11_TRAINER_SHA")
    if _git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("AlphaMAS worktree is not clean")


def verify_exact_plan_lineage(experiments_root: Path, plan_sha: str, plan_path: Path) -> None:
    if _git(experiments_root, "rev-parse", "HEAD") != plan_sha:
        raise RuntimeError("AlphaMAS-Experiments actual HEAD does not match M2_11_PLAN_SHA")
    if _git(experiments_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("AlphaMAS-Experiments worktree is not clean")
    plan = json.loads(plan_path.read_text())
    required = {
        "task": TASK_ID,
        "method": METHOD_ID,
        "tree_identity": TREE_IDENTITY,
        "representation_identity": REPRESENTATION_IDENTITY,
        "reward": REWARD_ID,
        "seed": SEED,
        "outer_iterations": OUTER_ITERATIONS,
        "epochs": OPTIMISATION_EPOCHS,
        "candidate_count": 9,
        "full_batch": True,
        "action_sampling": False,
        "validation_optimisation": False,
        "holdout_use": False,
        "deterministic_replay": "required",
    }
    for key, expected in required.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"training plan contract mismatch: {key}")
    if plan.get("lr_candidates") != [1e-4, 3e-4, 1e-3]:
        raise RuntimeError("training plan LR grid changed")
    if plan.get("checkpoint_iterations") != [25, 50, 100]:
        raise RuntimeError("training plan checkpoint grid changed")


@dataclass(frozen=True)
class TreeTensors:
    observations: Tensor
    rewards: Tensor
    child_indices: Tensor
    depths: Tensor
    root_indices: Tensor
    prompt_action_indices: Tensor
    node_ids: tuple[str, ...]
    ordering_sha256: str
    roles: tuple[str, ...]

    @property
    def node_count(self) -> int:
        return int(self.observations.shape[0])


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _load_bound_json(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"frozen artifact identity mismatch: {path.name}:{key}")
    return payload


def load_frozen_train_tree(experiments_root: Path) -> TreeTensors:
    """Load only TRAIN rows reachable from the frozen v2 tree."""

    base = experiments_root / "experiments/M2/development"
    tree_root = base / "rl_architecture_v2/train_environment"
    representation_root = base / "semantic_state_representation_v1"
    manifest = _load_bound_json(
        tree_root / "tree_manifest.json",
        {
            "contract": TREE_CONTRACT,
            "train_counterfactual_tree_v2_identity_sha256": TREE_IDENTITY,
            "representation_identity": REPRESENTATION_IDENTITY,
            "reward_id": REWARD_ID,
            "bellman_bootstrap": False,
            "gamma": None,
        },
    )
    summary = _load_bound_json(
        tree_root / "tree_summary.json",
        {
            "schema_version": TREE_CONTRACT,
            "train_counterfactual_tree_v2_identity_sha256": TREE_IDENTITY,
            "decision_nodes": EXPECTED_NODES,
            "action_edges": EXPECTED_ACTION_CREDITS,
            "local_r3_credits": EXPECTED_ACTION_CREDITS,
            "semantic_train_decisions": 56,
            "validation_transitions": 0,
            "final_holdout_accesses": 0,
        },
    )
    hashes = json.loads((tree_root / "tree_sha256.json").read_text())
    for filename in ("nodes.jsonl.gz", "edges.jsonl.gz"):
        if sha256_file(tree_root / filename) != hashes[filename]["compressed_sha256"]:
            raise RuntimeError(f"frozen tree compressed identity drift: {filename}")
    if manifest["nodes_logical_sha256"] != hashes["nodes.jsonl.gz"]["logical_sha256"]:
        raise RuntimeError("node logical identity drift")
    if manifest["edges_logical_sha256"] != hashes["edges.jsonl.gz"]["logical_sha256"]:
        raise RuntimeError("edge logical identity drift")

    row_index = json.loads((representation_root / "manifests/row_index.json").read_text())
    train_rows = {row["row_index"]: row for row in row_index if row.get("role") == "TRAIN"}
    if len(train_rows) != 56 or any(row.get("role") != "TRAIN" for row in train_rows.values()):
        raise RuntimeError("frozen TRAIN semantic membership changed")
    semantic_matrix = np.load(
        representation_root / "embeddings/semantic_base.npy", mmap_mode="r", allow_pickle=False
    )
    if semantic_matrix.shape != (72, 3076) or semantic_matrix.dtype != np.float32:
        raise RuntimeError("frozen semantic matrix contract changed")

    nodes = _read_jsonl_gz(tree_root / "nodes.jsonl.gz")
    edges = _read_jsonl_gz(tree_root / "edges.jsonl.gz")
    if len(nodes) != EXPECTED_NODES or len(edges) != EXPECTED_ACTION_CREDITS:
        raise RuntimeError("frozen TRAIN tree population changed")
    node_index = {node["node_id"]: index for index, node in enumerate(nodes)}
    if len(node_index) != EXPECTED_NODES:
        raise RuntimeError("tree node identity is not unique")

    observations = np.empty((EXPECTED_NODES, 3080), dtype=np.float32)
    depths = np.empty(EXPECTED_NODES, dtype=np.int64)
    roles: list[str] = []
    prompt_actions = np.empty(EXPECTED_NODES, dtype=np.int64)
    for index, node in enumerate(nodes):
        semantic_index = node["semantic_row"]["index"]
        row = train_rows.get(semantic_index)
        if row is None or row["case_id"] != node["semantic_row"]["identity"]:
            raise RuntimeError("tree referenced a non-TRAIN or drifted semantic row")
        semantic = np.asarray(semantic_matrix[semantic_index], dtype=np.float32)
        if sha256_bytes(semantic.tobytes()) != node["semantic_base_sha256"]:
            raise RuntimeError("semantic base identity mismatch during tree reconstruction")
        portfolio = node["portfolio_snapshot"]
        quantity = float(portfolio["quantity"])
        drawdown = float(portfolio["current_drawdown"])
        if quantity <= 1e-12:
            dynamic = np.array([1.0, 0.0, 0.0, drawdown], dtype=np.float32)
        else:
            close = (float(portfolio["current_equity"]) - float(portfolio["cash"])) / quantity
            entry = float(portfolio["average_entry_price"])
            if close <= 0 or entry <= 0:
                raise RuntimeError("invalid frozen LONG portfolio reconstruction")
            dynamic = np.array([0.0, 1.0, math.log(close / entry), drawdown], dtype=np.float32)
        observation = np.concatenate((semantic, dynamic)).astype(np.float32, copy=False)
        if sha256_bytes(observation.tobytes()) != node["actor_observation"]["sha256"]:
            raise RuntimeError("Actor observation reconstruction identity mismatch")
        observations[index] = observation
        depths[index] = node["depth"]
        roles.append("TRAIN")
        prompt_slice = observation[3073:3076]
        if not np.array_equal(np.sort(prompt_slice), np.array([0.0, 0.0, 1.0])):
            raise RuntimeError("frozen Prompt action is not one-hot")
        prompt_actions[index] = int(np.argmax(prompt_slice))

    rewards = np.empty((EXPECTED_NODES, 3), dtype=np.float64)
    children = np.full((EXPECTED_NODES, 3), -1, dtype=np.int64)
    seen_actions: set[tuple[int, int]] = set()
    action_index = {action: index for index, action in enumerate(ACTION_ORDER)}
    for edge in edges:
        parent = node_index.get(edge["parent_node_id"])
        if parent is None or edge["action"] not in action_index:
            raise RuntimeError("tree edge parent/action contract changed")
        action = action_index[edge["action"]]
        if (parent, action) in seen_actions:
            raise RuntimeError("duplicate tree action edge")
        seen_actions.add((parent, action))
        rewards[parent, action] = float(edge["local_r3"])
        child = edge["child_node_id"]
        children[parent, action] = node_index.get(child, -1)
    if len(seen_actions) != EXPECTED_ACTION_CREDITS or not np.isfinite(rewards).all():
        raise RuntimeError("tree local R3 action-credit contract changed")

    roots = np.flatnonzero(depths == 0)
    if len(roots) != EXPECTED_ROOTS or set(depths.tolist()) != set(range(EXPECTED_DEPTHS)):
        raise RuntimeError("tree root/depth contract changed")
    expected_depth_counts = {depth: 8 * (3**depth) for depth in range(EXPECTED_DEPTHS)}
    actual_depth_counts = {depth: int(np.sum(depths == depth)) for depth in range(EXPECTED_DEPTHS)}
    if actual_depth_counts != expected_depth_counts or summary["depth_counts"] != {
        str(key): value for key, value in expected_depth_counts.items()
    }:
        raise RuntimeError("tree depth population changed")
    ordering_sha = sha256_bytes(canonical_json_bytes([node["node_id"] for node in nodes]))
    if set(roles) != {"TRAIN"}:
        raise RuntimeError("optimisation tensor roles are not TRAIN-only")
    return TreeTensors(
        observations=torch.from_numpy(observations),
        rewards=torch.from_numpy(rewards),
        child_indices=torch.from_numpy(children),
        depths=torch.from_numpy(depths),
        root_indices=torch.from_numpy(roots.copy()),
        prompt_action_indices=torch.from_numpy(prompt_actions),
        node_ids=tuple(node["node_id"] for node in nodes),
        ordering_sha256=ordering_sha,
        roles=tuple(roles),
    )


@dataclass(frozen=True)
class PolicyEvaluation:
    values: Tensor
    advantages: Tensor
    occupancy: Tensor
    occupancy_mass_by_depth: tuple[float, ...]
    max_weighted_advantage_residual: float


@dataclass
class OldPolicyChronology:
    """Fail-closed state machine for the mandatory per-iteration refresh contract."""

    first_old_sha: str | None = None
    previous_post_sha: str | None = None
    active_old_sha: str | None = None
    epochs_seen: int = 0

    def begin_iteration(self, iteration: int, pre_update_sha: str) -> None:
        if self.active_old_sha is not None:
            raise RuntimeError("previous old-policy epoch block did not finish")
        if (
            iteration > 1
            and self.first_old_sha is not None
            and pre_update_sha == self.first_old_sha
        ):
            raise RuntimeError("stale iteration-0 old policy was reused")
        if self.previous_post_sha is not None and pre_update_sha != self.previous_post_sha:
            raise RuntimeError("old policy did not refresh from previous post-update model")
        if iteration == 1:
            self.first_old_sha = pre_update_sha
        self.active_old_sha = pre_update_sha
        self.epochs_seen = 0

    def record_epoch(self, old_policy_sha: str) -> None:
        if self.active_old_sha is None or old_policy_sha != self.active_old_sha:
            raise RuntimeError("old policy changed inside four-epoch block")
        self.epochs_seen += 1

    def finish_iteration(self, post_update_sha: str) -> None:
        if self.epochs_seen != OPTIMISATION_EPOCHS:
            raise RuntimeError("old policy was not frozen for exactly four epochs")
        self.previous_post_sha = post_update_sha
        self.active_old_sha = None


def exact_tree_policy_evaluation(tree: TreeTensors, old_probabilities: Tensor) -> PolicyEvaluation:
    """Vectorized exact frozen local credit and old-policy occupancy evaluation."""

    old = old_probabilities.to(dtype=torch.float64)
    rewards = tree.rewards.to(device=old.device, dtype=torch.float64)
    depths = tree.depths.to(device=old.device)
    children = tree.child_indices.to(device=old.device)
    roots = tree.root_indices.to(device=old.device)
    if old.shape != rewards.shape or torch.any(~torch.isfinite(old)) or torch.any(old <= 0):
        raise RuntimeError("old policy probabilities must be finite and positive")
    if float(torch.max(torch.abs(old.sum(dim=1) - 1.0))) > 1e-12:
        raise RuntimeError("old policy probability rows do not sum to one")
    values = torch.sum(old * rewards, dim=1)
    advantages = rewards - values[:, None]
    residual = torch.sum(old * advantages, dim=1)
    max_residual = float(torch.max(torch.abs(residual)))
    if max_residual > 1e-10:
        raise RuntimeError("weighted counterfactual advantage invariant failed")

    occupancy = torch.zeros(tree.node_count, dtype=torch.float64, device=old.device)
    occupancy[roots] = 1.0 / int(roots.numel())
    masses: list[float] = []
    max_depth = int(depths.max())
    for depth in range(max_depth + 1):
        parents = torch.nonzero(depths == depth, as_tuple=False).flatten()
        mass = float(occupancy[parents].sum())
        masses.append(mass)
        if not math.isclose(mass, 1.0, abs_tol=1e-10, rel_tol=0.0):
            raise RuntimeError(f"old-policy occupancy mass failed at depth {depth}")
        if depth < max_depth:
            for action in range(3):
                target = children[parents, action]
                if torch.any(target < 0):
                    raise RuntimeError("nonterminal tree node lacks a child")
                occupancy[target] = occupancy[parents] * old[parents, action]
    if torch.any(~torch.isfinite(occupancy)) or torch.any(occupancy < 0):
        raise RuntimeError("old-policy occupancy contains invalid values")
    return PolicyEvaluation(values, advantages, occupancy, tuple(masses), max_residual)


def normalise_policy_probabilities(probabilities: Tensor) -> Tensor:
    """Promote softmax output to float64 and remove float32 row-sum roundoff."""

    promoted = probabilities.double()
    if torch.any(~torch.isfinite(promoted)) or torch.any(promoted <= 0):
        raise RuntimeError("policy probabilities must be finite and positive")
    row_sums = promoted.sum(dim=1, keepdim=True)
    if float(torch.max(torch.abs(row_sums.detach() - 1.0))) > 1e-6:
        raise RuntimeError("policy probability rows violate softmax tolerance")
    return promoted / row_sums


def prompt_prior(observations: Tensor) -> Tensor:
    onehot = observations[:, 3073:3076]
    return onehot * PROMPT_SELECTED_PROBABILITY + (1.0 - onehot) * PROMPT_ALTERNATIVE_PROBABILITY


def configure_runtime_determinism(seed: int = SEED) -> None:
    configure_determinism(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def runtime_manifest(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_model": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _assert_finite_model(model: nn.Module) -> tuple[int, int]:
    nan_count = sum(int(torch.isnan(parameter).sum()) for parameter in model.parameters())
    inf_count = sum(int(torch.isinf(parameter).sum()) for parameter in model.parameters())
    if nan_count or inf_count:
        raise RuntimeError("model contains NaN or Inf parameters")
    return nan_count, inf_count


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    os.replace(temporary, path)


def create_initial_checkpoint(
    tree: TreeTensors, output_path: Path, *, source_sha: str
) -> dict[str, Any]:
    configure_runtime_determinism(SEED)
    model = PromptAnchoredActorCritic()
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise RuntimeError("global model parameter count changed")
    with torch.no_grad():
        probabilities = model(tree.observations)["probabilities"]
    maximum_difference = float(
        torch.max(torch.abs(probabilities - prompt_prior(tree.observations)))
    )
    if maximum_difference > 1e-7:
        raise RuntimeError("initial policy does not reproduce the frozen Prompt prior")
    nan_count, inf_count = _assert_finite_model(model)
    parameter_sha = canonical_parameter_sha(model)
    metadata = {
        "schema_version": "M2-11-INITIAL-CHECKPOINT-v1",
        "task": TASK_ID,
        "method": METHOD_ID,
        "tree_identity": TREE_IDENTITY,
        "representation_identity": REPRESENTATION_IDENTITY,
        "reward": REWARD_ID,
        "seed": SEED,
        "source_sha": source_sha,
        "initial_parameter_sha": parameter_sha,
        "parameter_count": model.parameter_count(),
        "actor_parameter_sha": actor_parameter_sha(model),
        "critic_parameter_sha": critic_parameter_sha(model),
        "tree_ordering_sha": tree.ordering_sha256,
        "train_nodes": tree.node_count,
        "roles_present": sorted(set(tree.roles)),
        "validation_optimisation_nodes": 0,
        "final_holdout_nodes": 0,
        "e2e_pilot_nodes": 0,
        "prompt_prior_max_abs_difference": maximum_difference,
        "nan_parameters": nan_count,
        "inf_parameters": inf_count,
    }
    _atomic_torch_save({"model_state": model.state_dict(), "metadata": metadata}, output_path)
    _write_json(output_path.with_suffix(".manifest.json"), metadata)
    return metadata


def load_initial_checkpoint(
    path: Path, device: torch.device
) -> tuple[PromptAnchoredActorCritic, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    for key, expected in {
        "method": METHOD_ID,
        "tree_identity": TREE_IDENTITY,
        "representation_identity": REPRESENTATION_IDENTITY,
        "reward": REWARD_ID,
        "seed": SEED,
        "parameter_count": EXPECTED_PARAMETERS,
        "roles_present": ["TRAIN"],
        "validation_optimisation_nodes": 0,
        "final_holdout_nodes": 0,
    }.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"initial checkpoint contract mismatch: {key}")
    model = PromptAnchoredActorCritic()
    model.load_state_dict(checkpoint["model_state"])
    if canonical_parameter_sha(model) != metadata["initial_parameter_sha"]:
        raise RuntimeError("initial checkpoint canonical parameter identity mismatch")
    model.to(device)
    return model, metadata


def _gradient_norm(parameters: Iterable[Tensor]) -> float:
    gradients = [
        parameter.grad.detach().double().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return float(torch.stack(gradients).norm(2))


def _rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    torch.set_rng_state(payload["cpu"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _trace_sha(records: list[dict[str, Any]]) -> str:
    return sha256_bytes(b"".join(canonical_json_bytes(record) + b"\n" for record in records))


def _candidate_metadata(
    *,
    model: PromptAnchoredActorCritic,
    candidate_id: str,
    learning_rate: float,
    iteration: int,
    initial_sha: str,
    plan_sha: str,
    trainer_sha: str,
    training_trace_sha: str,
    run_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": "M2-11-CANDIDATE-CHECKPOINT-v1",
        "candidate_id": candidate_id,
        "learning_rate": learning_rate,
        "outer_iteration": iteration,
        "seed": SEED,
        "method_identity": METHOD_ID,
        "tree_identity": TREE_IDENTITY,
        "representation_identity": REPRESENTATION_IDENTITY,
        "reward_identity": REWARD_ID,
        "initial_parameter_sha": initial_sha,
        "candidate_parameter_sha": canonical_parameter_sha(model),
        "parameter_count": model.parameter_count(),
        "actor_parameter_sha": actor_parameter_sha(model),
        "critic_parameter_sha": critic_parameter_sha(model),
        "training_plan_sha": plan_sha,
        "trainer_sha": trainer_sha,
        "training_trace_sha": training_trace_sha,
        "run_kind": run_kind,
        "selection_eligible": run_kind == "canonical",
        "replay_status": None if run_kind == "canonical" else "AUDIT_ONLY",
        "selection_status": "DEFERRED_TO_M2_12"
        if run_kind == "canonical"
        else "NOT_SELECTION_ELIGIBLE",
    }


def _validate_production_grid(learning_rate: float) -> None:
    if learning_rate not in LEARNING_RATES:
        raise RuntimeError("learning rate is outside the frozen candidate grid")
    if set(CANDIDATE_IDS) != {
        (learning_rate_item, iteration)
        for learning_rate_item in LEARNING_RATES
        for iteration in CHECKPOINT_ITERATIONS
    }:
        raise RuntimeError("nine-candidate grid changed")


def train_lineage(
    *,
    tree: TreeTensors,
    initial_checkpoint: Path,
    output_root: Path,
    learning_rate: float,
    device: torch.device,
    trainer_sha: str,
    plan_sha: str,
    run_kind: str,
    resume: bool,
) -> dict[str, Any]:
    """Train one frozen LR lineage; no candidate comparison is performed."""

    _validate_production_grid(learning_rate)
    if run_kind not in {"canonical", "replay"}:
        raise RuntimeError("run kind must be canonical or replay")
    configure_runtime_determinism(SEED)
    model, initial_metadata = load_initial_checkpoint(initial_checkpoint, device)
    initial_sha = initial_metadata["initial_parameter_sha"]
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    observations = tree.observations.to(device=device, dtype=torch.float32)
    rewards = tree.rewards.to(device=device, dtype=torch.float64)
    device_tree = TreeTensors(
        observations=observations,
        rewards=rewards,
        child_indices=tree.child_indices.to(device),
        depths=tree.depths.to(device),
        root_indices=tree.root_indices.to(device),
        prompt_action_indices=tree.prompt_action_indices.to(device),
        node_ids=tree.node_ids,
        ordering_sha256=tree.ordering_sha256,
        roles=tree.roles,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    resume_path = output_root / "resume.pt"
    trace_path = output_root / "training_trace.jsonl"
    records: list[dict[str, Any]] = []
    completed = 0
    safe_resumes = 0
    chronology = OldPolicyChronology()
    if resume:
        if not resume_path.exists() or not trace_path.exists():
            raise RuntimeError("safe resume requested without boundary checkpoint and trace")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        bindings = state["bindings"]
        expected_bindings = {
            "learning_rate": learning_rate,
            "trainer_sha": trainer_sha,
            "plan_sha": plan_sha,
            "initial_parameter_sha": initial_sha,
            "run_kind": run_kind,
        }
        if bindings != expected_bindings:
            raise RuntimeError("resume checkpoint lineage binding mismatch")
        records = state["trace_records"]
        if len(records) != state["completed_outer_iteration"]:
            raise RuntimeError("resume trace boundary mismatch")
        if _trace_sha(records) != state["training_trace_sha"]:
            raise RuntimeError("resume trace identity mismatch")
        archived_records = [
            json.loads(line) for line in trace_path.read_text().splitlines() if line
        ]
        if archived_records != records:
            trace_path.write_bytes(
                b"".join(canonical_json_bytes(record) + b"\n" for record in records)
            )
        model.load_state_dict(state["model_state"])
        optimiser.load_state_dict(state["optimiser_state"])
        _restore_rng_state(state["rng_state"])
        completed = state["completed_outer_iteration"]
        chronology.previous_post_sha = state["post_update_model_sha"]
        chronology.first_old_sha = state["first_old_policy_sha"]
        safe_resumes = state["safe_resumes"] + 1
        if canonical_parameter_sha(model) != chronology.previous_post_sha:
            raise RuntimeError("resume model identity mismatch")
    elif resume_path.exists() or trace_path.exists():
        raise RuntimeError("output lineage already exists; explicit safe resume is required")

    for iteration in range(completed + 1, OUTER_ITERATIONS + 1):
        started = time.perf_counter()
        pre_sha = canonical_parameter_sha(model)
        chronology.begin_iteration(iteration, pre_sha)
        with torch.no_grad():
            old_output = model(observations)
            old_probabilities = normalise_policy_probabilities(
                old_output["probabilities"].detach().clone()
            )
        old_probability_bytes = old_probabilities.detach().cpu().contiguous().numpy().tobytes()
        frozen_old_probability_sha = sha256_bytes(old_probability_bytes)
        evaluation = exact_tree_policy_evaluation(device_tree, old_probabilities)
        frozen_values = evaluation.values.detach().clone()
        frozen_advantages = evaluation.advantages.detach().clone()
        frozen_occupancy = evaluation.occupancy.detach().clone()
        epoch_old_policy_shas: list[str] = []
        final_losses: dict[str, Tensor] | None = None
        gradient_before = 0.0
        gradient_after = 0.0
        clipping_status = False
        for _epoch in range(1, OPTIMISATION_EPOCHS + 1):
            if (
                sha256_bytes(old_probabilities.detach().cpu().contiguous().numpy().tobytes())
                != frozen_old_probability_sha
            ):
                raise RuntimeError("old policy changed inside four-epoch block")
            epoch_old_policy_shas.append(pre_sha)
            chronology.record_epoch(pre_sha)
            output = model(observations)
            final_losses = exact_ppo_loss(
                normalise_policy_probabilities(output["probabilities"]),
                old_probabilities,
                frozen_advantages,
                output["value"].double(),
                frozen_values,
                frozen_occupancy,
            )
            optimiser.zero_grad(set_to_none=True)
            final_losses["loss"].backward()
            parameters = list(model.parameters())
            gradient_before = _gradient_norm(parameters)
            returned_norm = float(nn.utils.clip_grad_norm_(parameters, GRADIENT_NORM_CEILING))
            if not math.isclose(gradient_before, returned_norm, rel_tol=1e-5, abs_tol=1e-8):
                raise RuntimeError("gradient norm diagnostic mismatch")
            clipping_status = gradient_before > GRADIENT_NORM_CEILING
            gradient_after = _gradient_norm(parameters)
            if gradient_after > GRADIENT_NORM_CEILING + 1e-6:
                raise RuntimeError("gradient clipping ceiling failed")
            optimiser.step()
            _assert_finite_model(model)
        if epoch_old_policy_shas != [pre_sha] * OPTIMISATION_EPOCHS:
            raise RuntimeError("old policy SHA changed inside optimisation epoch block")
        if final_losses is None:
            raise AssertionError("four frozen optimisation epochs did not execute")
        post_sha = canonical_parameter_sha(model)
        chronology.finish_iteration(post_sha)
        with torch.no_grad():
            post_output = model(observations)
            new_probabilities = normalise_policy_probabilities(post_output["probabilities"])
            ratio = new_probabilities / old_probabilities
            row_sum_error = float(torch.max(torch.abs(new_probabilities.sum(dim=1) - 1.0)))
            prompt = device_tree.prompt_action_indices
            row_index = torch.arange(tree.node_count, device=device)
            prompt_probability = new_probabilities[row_index, prompt]
            retain = prompt_probability >= new_probabilities.max(dim=1).values - 1e-12
            nan_count = int(torch.isnan(new_probabilities).sum())
            inf_count = int(torch.isinf(new_probabilities).sum())
        if nan_count or inf_count or torch.any(new_probabilities <= 0) or row_sum_error > 1e-12:
            raise RuntimeError("post-update policy probability invariant failed")
        elapsed = time.perf_counter() - started
        record = {
            "learning_rate": learning_rate,
            "iteration": iteration,
            "pre_update_model_sha": pre_sha,
            "old_policy_sha": pre_sha,
            "post_update_model_sha": post_sha,
            "old_policy_probability_minimum": float(old_probabilities.min()),
            "old_policy_probability_maximum": float(old_probabilities.max()),
            "probability_row_sum_max_error": row_sum_error,
            "occupancy_mass_by_depth": list(evaluation.occupancy_mass_by_depth),
            "maximum_weighted_advantage_residual": evaluation.max_weighted_advantage_residual,
            "policy_surrogate": float(final_losses["policy_surrogate"].detach()),
            "value_loss": float(final_losses["value_loss"].detach()),
            "total_loss": float(final_losses["loss"].detach()),
            "gradient_norm_before_clipping": gradient_before,
            "gradient_clipping_status": clipping_status,
            "gradient_norm_after_clipping": gradient_after,
            "ppo_ratio_minimum": float(ratio.min()),
            "ppo_ratio_maximum": float(ratio.max()),
            "ppo_clip_fraction": float((torch.abs(ratio - 1.0) > PPO_CLIP).double().mean()),
            "mean_gate": float(post_output["gate"].mean()),
            "prompt_retain_rate": float(retain.double().mean()),
            "prompt_override_rate": float((~retain).double().mean()),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "elapsed_seconds": elapsed,
            "old_policy_frozen_epoch_shas": epoch_old_policy_shas,
        }
        records.append(record)
        trace_sha = _trace_sha(records)
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json_bytes(record).decode("utf-8") + "\n")
        if iteration in CHECKPOINT_ITERATIONS:
            candidate_id = CANDIDATE_IDS[(learning_rate, iteration)]
            metadata = _candidate_metadata(
                model=model,
                candidate_id=candidate_id,
                learning_rate=learning_rate,
                iteration=iteration,
                initial_sha=initial_sha,
                plan_sha=plan_sha,
                trainer_sha=trainer_sha,
                training_trace_sha=trace_sha,
                run_kind=run_kind,
            )
            checkpoint_path = output_root / "checkpoints" / f"{candidate_id}.pt"
            _atomic_torch_save(
                {"model_state": model.state_dict(), "metadata": metadata}, checkpoint_path
            )
            _write_json(checkpoint_path.with_suffix(".json"), metadata)
        bindings = {
            "learning_rate": learning_rate,
            "trainer_sha": trainer_sha,
            "plan_sha": plan_sha,
            "initial_parameter_sha": initial_sha,
            "run_kind": run_kind,
        }
        _atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "completed_outer_iteration": iteration,
                "rng_state": _rng_state(),
                "bindings": bindings,
                "training_trace_sha": trace_sha,
                "trace_records": records,
                "post_update_model_sha": post_sha,
                "first_old_policy_sha": chronology.first_old_sha,
                "safe_resumes": safe_resumes,
            },
            resume_path,
        )

    summary = {
        "schema_version": "M2-11-TRAINING-LINEAGE-v1",
        "task": TASK_ID,
        "run_kind": run_kind,
        "learning_rate": learning_rate,
        "iterations_completed": OUTER_ITERATIONS,
        "epochs_per_iteration": OPTIMISATION_EPOCHS,
        "full_batch": True,
        "node_shuffling": False,
        "minibatch_sampling": False,
        "action_sampling": False,
        "correctness_valid": True,
        "safe_resumes": safe_resumes,
        "initial_parameter_sha": initial_sha,
        "final_parameter_sha": chronology.previous_post_sha,
        "training_trace_sha": _trace_sha(records),
        "trainer_sha": trainer_sha,
        "training_plan_sha": plan_sha,
        "tree_ordering_sha": tree.ordering_sha256,
        "runtime": runtime_manifest(device),
        "nan_inf": 0,
    }
    _write_json(output_root / "training_summary.json", summary)
    return summary


def _default_tree_root(experiments_root: Path) -> Path:
    return experiments_root / "experiments/M2/development/rl_architecture_v2/train_environment"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--trainer-sha", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--output", type=Path, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--plan-sha", required=True)
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--initial-checkpoint", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--learning-rate", type=float, required=True)
    train.add_argument("--run-kind", choices=("canonical", "replay"), required=True)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = assert_not_desktop_checkout(args.source_root)
    verify_exact_source_lineage(source_root, args.trainer_sha)
    tree = load_frozen_train_tree(args.experiments_root)
    if args.command == "initialize":
        metadata = create_initial_checkpoint(tree, args.output, source_sha=args.trainer_sha)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    verify_exact_plan_lineage(args.experiments_root, args.plan_sha, args.plan)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("canonical CUDA training requested but CUDA is unavailable")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before Python starts")
    summary = train_lineage(
        tree=tree,
        initial_checkpoint=args.initial_checkpoint,
        output_root=args.output_root,
        learning_rate=args.learning_rate,
        device=device,
        trainer_sha=args.trainer_sha,
        plan_sha=args.plan_sha,
        run_kind=args.run_kind,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
