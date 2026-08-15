"""Frozen, outcome-independent M2 semantic state representation utilities."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.backtesting.portfolio import Portfolio

CONTRACT = "M2-SEMANTIC-STATE-v1"
SOURCE_CONTRACT = "M2-SEMANTIC-HANDOFF-v1"
MODEL_REPO = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
SOURCE_CORPUS_IDENTITY = "cc1fea692489a5b5791cae1fd96386bfbcf1f0a07f2ea3e22637dd3c43533a43"
REFERENCE_DIMENSION = 1024
CANDIDATE_DIMENSIONS = (256, 512, 1024)
ACTION_ORDER = ("BUY", "HOLD", "SELL")
ACTION_ONEHOT = {
    "BUY": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "HOLD": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "SELL": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}
THRESHOLDS = {"spearman": 0.98, "cosine_mae": 0.02, "top5_overlap": 0.80}
MAX_SEQUENCE_LENGTH = 32768


@dataclass(frozen=True)
class EconomicPayload:
    """The complete and exclusive learnable payload extracted from an actor state."""

    research_manager_text: str
    prompt_trader_text: str
    prompt_action: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("semantic text must be a string")
    return sha256_bytes(value.encode("utf-8"))


def validate_token_count(token_count: int) -> int:
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token count must be a non-negative integer")
    if token_count > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"semantic text exceeds {MAX_SEQUENCE_LENGTH} tokens; truncation is forbidden")
    return token_count


def extract_economic_payload(actor_state: Mapping[str, Any]) -> EconomicPayload:
    """Extract only the two verbatim texts and explicit action prior."""
    if actor_state.get("schema_version") != SOURCE_CONTRACT:
        raise ValueError("unexpected semantic hand-off contract")
    try:
        manager = actor_state["research_manager"]["investment_plan"]
        trader = actor_state["prompt_trader"]["rendered_proposal"]
        action = actor_state["prompt_trader"]["normalized_action"]
    except (KeyError, TypeError) as error:
        raise ValueError("approved semantic payload fields are missing") from error
    if not isinstance(manager, str) or not isinstance(trader, str):
        raise TypeError("approved semantic payload fields must be strings")
    if not manager or not trader:
        raise ValueError("approved semantic payload fields must not be empty")
    if action not in ACTION_ONEHOT:
        raise ValueError("normalized action must be BUY, HOLD, or SELL")
    return EconomicPayload(manager, trader, str(action))


def action_onehot(action: str) -> np.ndarray:
    try:
        return ACTION_ONEHOT[action].copy()
    except KeyError as error:
        raise ValueError("action must be BUY, HOLD, or SELL") from error


def _finite_matrix(value: Any, *, columns: int | None = None) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2:
        raise ValueError("embedding input must be a matrix")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(f"embedding matrix must have {columns} columns")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must contain only finite floats")
    return matrix.astype(np.float32, copy=False)


def mrl_prefix_normalize(full_embeddings: Any, dimension: int) -> np.ndarray:
    """Apply the Qwen3 MRL prefix construction then row-wise L2 normalization."""
    if dimension not in CANDIDATE_DIMENSIONS:
        raise ValueError("dimension is not preregistered")
    full = _finite_matrix(full_embeddings, columns=REFERENCE_DIMENSION)
    prefix = full[:, :dimension].astype(np.float64)
    norms = np.linalg.norm(prefix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero embedding")
    selected = (prefix / norms).astype(np.float32)
    if not np.all(np.abs(np.linalg.norm(selected, axis=1) - 1.0) <= 1e-5):
        raise ValueError("MRL output is not L2-normalized")
    return selected


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _geometry_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    if candidate.shape[0] != reference.shape[0] or candidate.shape[0] <= 5:
        raise ValueError("geometry views require matching populations greater than five")
    candidate_sim = candidate.astype(np.float64) @ candidate.astype(np.float64).T
    reference_sim = reference.astype(np.float64) @ reference.astype(np.float64).T
    triangle = np.triu_indices(candidate.shape[0], 1)
    candidate_pairs = candidate_sim[triangle]
    reference_pairs = reference_sim[triangle]
    overlaps = []
    for row in range(candidate.shape[0]):
        candidate_order = np.argsort(-candidate_sim[row], kind="mergesort")
        reference_order = np.argsort(-reference_sim[row], kind="mergesort")
        candidate_top5 = set(candidate_order[candidate_order != row][:5].tolist())
        reference_top5 = set(reference_order[reference_order != row][:5].tolist())
        overlaps.append(len(candidate_top5 & reference_top5) / 5.0)
    result: dict[str, float | bool] = {
        "spearman": _spearman(candidate_pairs, reference_pairs),
        "cosine_mae": float(np.mean(np.abs(candidate_pairs - reference_pairs))),
        "top5_overlap": float(np.mean(overlaps)),
    }
    result["pass"] = bool(
        result["spearman"] >= THRESHOLDS["spearman"]
        and result["cosine_mae"] <= THRESHOLDS["cosine_mae"]
        and result["top5_overlap"] >= THRESHOLDS["top5_overlap"]
    )
    return result


def select_embedding_dimension(
    train_manager_full: Any, train_trader_full: Any
) -> tuple[int, dict[str, dict[str, dict[str, float | bool]]]]:
    """Select the smallest passing MRL dimension from exactly 56 TRAIN cases."""
    manager_full = _finite_matrix(train_manager_full, columns=REFERENCE_DIMENSION)
    trader_full = _finite_matrix(train_trader_full, columns=REFERENCE_DIMENSION)
    if manager_full.shape[0] != 56 or trader_full.shape[0] != 56:
        raise ValueError("dimension selection requires exactly 56 TRAIN rows per channel")
    references = {
        "manager": mrl_prefix_normalize(manager_full, REFERENCE_DIMENSION),
        "trader": mrl_prefix_normalize(trader_full, REFERENCE_DIMENSION),
    }
    references["joint"] = _normalize_rows(
        np.concatenate((references["manager"], references["trader"]), axis=1)
    )
    diagnostics: dict[str, dict[str, dict[str, float | bool]]] = {}
    for dimension in CANDIDATE_DIMENSIONS:
        manager = mrl_prefix_normalize(manager_full, dimension)
        trader = mrl_prefix_normalize(trader_full, dimension)
        views = {
            "manager": manager,
            "trader": trader,
            "joint": _normalize_rows(np.concatenate((manager, trader), axis=1)),
        }
        metrics = {
            view: _geometry_metrics(values, references[view])
            for view, values in views.items()
        }
        diagnostics[str(dimension)] = metrics
    selected = next(
        (
            dimension
            for dimension in CANDIDATE_DIMENSIONS
            if all(bool(value["pass"]) for value in diagnostics[str(dimension)].values())
        ),
        REFERENCE_DIMENSION,
    )
    return selected, diagnostics


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    norms = np.linalg.norm(values64, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("cannot normalize joint embedding")
    return (values64 / norms).astype(np.float32)


def build_semantic_base(manager: Any, trader: Any, actions: Sequence[str]) -> np.ndarray:
    manager_matrix = _finite_matrix(manager)
    trader_matrix = _finite_matrix(trader)
    if manager_matrix.shape != trader_matrix.shape or len(actions) != manager_matrix.shape[0]:
        raise ValueError("semantic channels and actions must have matching rows")
    manager_norms = np.linalg.norm(manager_matrix, axis=1)
    trader_norms = np.linalg.norm(trader_matrix, axis=1)
    if not np.all(np.abs(manager_norms - 1.0) <= 1e-5):
        raise ValueError("manager embeddings must be L2-normalized")
    if not np.all(np.abs(trader_norms - 1.0) <= 1e-5):
        raise ValueError("trader embeddings must be L2-normalized")
    residual = trader_matrix - manager_matrix
    agreement = np.sum(manager_matrix * trader_matrix, axis=1, keepdims=True)
    priors = np.stack([action_onehot(action) for action in actions])
    result = np.concatenate((manager_matrix, trader_matrix, residual, agreement, priors), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("semantic base contains NaN or infinity")
    return result.astype(np.float32, copy=False)


def build_portfolio_state(snapshot: PortfolioSnapshot) -> np.ndarray:
    quantity = float(snapshot.quantity)
    close = float(snapshot.close_price)
    entry = float(snapshot.average_entry_price)
    drawdown = float(snapshot.current_drawdown)
    if not all(math.isfinite(value) for value in (quantity, close, entry, drawdown)):
        raise ValueError("portfolio inputs must be finite")
    if quantity < -Portfolio.tolerance:
        raise ValueError("negative quantity violates the long-only contract")
    if drawdown > 0:
        raise ValueError("current_drawdown must be non-positive")
    if quantity <= Portfolio.tolerance:
        result = np.array([1.0, 0.0, 0.0, drawdown], dtype=np.float32)
    else:
        if entry <= 0 or close <= 0:
            raise ValueError("LONG state requires positive entry and close prices")
        result = np.array([0.0, 1.0, math.log(close / entry), drawdown], dtype=np.float32)
    if not np.isfinite(result).all() or result[0] + result[1] != 1.0:
        raise ValueError("invalid portfolio-state output")
    return result


def build_actor_observation(semantic_base: Any, snapshot: PortfolioSnapshot) -> np.ndarray:
    semantic = np.asarray(semantic_base)
    if semantic.ndim != 1 or not np.issubdtype(semantic.dtype, np.floating):
        raise ValueError("semantic_base must be a floating-point vector")
    if not np.isfinite(semantic).all():
        raise ValueError("semantic_base contains NaN or infinity")
    result = np.concatenate((semantic.astype(np.float32), build_portfolio_state(snapshot)))
    return result.astype(np.float32, copy=False)


class SemanticBaseStore:
    """Fail-closed learnable loader: metadata is never accepted or returned."""

    def __init__(self, semantic_base_path: Path) -> None:
        matrix = np.load(semantic_base_path, allow_pickle=False)
        self._matrix = _finite_matrix(matrix)

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    def semantic_base(self, row: int) -> np.ndarray:
        return self._matrix[row].copy()

    def actor_observation(self, row: int, snapshot: PortfolioSnapshot) -> np.ndarray:
        return build_actor_observation(self.semantic_base(row), snapshot)
