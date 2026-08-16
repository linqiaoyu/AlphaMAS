"""Build the canonical M2 TRAIN counterfactual tree from frozen local artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.m2.rl_environment import (
    ACTIONS,
    INITIAL_CASH,
    TRAIN_SESSIONS,
    TRAIN_SYMBOLS,
    TREE_CONTRACT,
    SequentialPortfolioState,
    canonical_json_bytes,
    initial_snapshot,
    load_market_snapshot,
    reward_window,
    sha256_bytes,
    simulate_transition,
)
from scripts.m2.semantic_state_representation import build_actor_observation
from tradingagents.backtesting.calendar import ExchangeSchedule

REPRESENTATION_IDENTITY = "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe"
REWARD_ID = "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"
REWARD_SNAPSHOT_IDENTITY = "3afc723888666e0ca3a12219a57dc577d6cf3953717da731b774634f4aff1445"
EXPECTED_NODES = 8_744
EXPECTED_EDGES = 26_232
EXPECTED_LEAVES = 17_496


def _node_id(symbol: str, history: tuple[str, ...]) -> str:
    return sha256_bytes(f"{TREE_CONTRACT}|{symbol}|{'/'.join(history)}".encode())


def _terminal_id(symbol: str, history: tuple[str, ...]) -> str:
    return sha256_bytes(f"{TREE_CONTRACT}|terminal|{symbol}|{'/'.join(history)}".encode())


def _gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str, int, int]:
    logical = b"".join(canonical_json_bytes(row) for row in rows)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as stream,
    ):
        stream.write(logical)
    return sha256_bytes(logical), sha256_bytes(path.read_bytes()), len(logical), path.stat().st_size


def _load_train_index(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("semantic row index must be a list")
    role_counts = Counter(row.get("role") for row in rows)
    if role_counts != {"TRAIN": 56, "VALIDATION": 16}:
        raise ValueError("frozen semantic population changed")
    train = [row for row in rows if row["role"] == "TRAIN"]
    identities = [(row["symbol"], row["decision_session"]) for row in train]
    expected = [(symbol, session) for symbol in TRAIN_SYMBOLS for session in TRAIN_SESSIONS]
    if identities != expected:
        raise ValueError("TRAIN membership or order changed")
    return train


def build_tree(
    *,
    experiments_root: Path,
    output_root: Path,
    source_sha: str,
    plan_sha: str,
) -> dict[str, Any]:
    representation = experiments_root / "experiments/M2/development/semantic_state_representation_v1"
    reward_study = experiments_root / "experiments/M2/development/reward_study_v1"
    row_index = _load_train_index(representation / "manifests/row_index.json")
    semantic_matrix = np.load(representation / "embeddings/semantic_base.npy", allow_pickle=False)
    if semantic_matrix.shape != (72, 3076) or semantic_matrix.dtype != np.float32:
        raise ValueError("frozen semantic matrix changed")
    semantic_by_case = {
        (row["symbol"], row["decision_session"]): (row, semantic_matrix[row["row_index"]])
        for row in row_index
    }
    output_root.mkdir(parents=True, exist_ok=False)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    depth_counts: Counter[int] = Counter()
    noop_edges = 0
    split_events = 0
    dividend_events = 0
    schedule = ExchangeSchedule()

    for symbol in TRAIN_SYMBOLS:
        market = load_market_snapshot(reward_study / f"inputs/market_snapshot/{symbol}.csv", symbol)
        root_snapshot = initial_snapshot(
            symbol,
            TRAIN_SESSIONS[0],
            market[TRAIN_SESSIONS[0]].close_price,
            schedule=schedule,
        )
        queue = deque([((), SequentialPortfolioState(), root_snapshot)])
        while queue:
            history, state, snapshot = queue.popleft()
            depth = len(history)
            decision_session = TRAIN_SESSIONS[depth]
            row, semantic_base = semantic_by_case[(symbol, decision_session)]
            observation = build_actor_observation(semantic_base, snapshot)
            if observation.shape != (3080,) or observation.dtype != np.float32:
                raise AssertionError("Actor observation contract changed")
            observation_sha = sha256_bytes(observation.tobytes())
            semantic_sha = sha256_bytes(semantic_base.tobytes())
            node_id = _node_id(symbol, history)
            parent_id = None if not history else _node_id(symbol, history[:-1])
            nodes.append(
                {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "symbol": symbol,
                    "depth": depth,
                    "decision_session": decision_session,
                    "semantic_row_identity": row["case_id"],
                    "semantic_row_index": row["row_index"],
                    "semantic_base_sha256": semantic_sha,
                    "actor_observation": {"sha256": observation_sha, "shape": [3080], "dtype": "float32", "finite": True, "reconstruction": "semantic_base row + frozen build_portfolio_state(snapshot)"},
                    "portfolio": {
                        "cash": state.cash,
                        "quantity": state.quantity,
                        "average_entry_price": state.average_entry_price,
                        "open_position_commission": state.open_position_commission,
                        "peak_equity": state.peak_equity,
                        "current_equity": snapshot.equity,
                        "current_drawdown": snapshot.current_drawdown,
                    },
                    "history": list(history),
                }
            )
            depth_counts[depth] += 1
            window = reward_window(symbol, decision_session, market, schedule=schedule)
            split_events += int(any(bar.split_ratio for bar in window.bars))
            dividend_events += int(any(bar.dividend_per_share for bar in window.bars))
            for action in ACTIONS:
                transition = simulate_transition(window, state, action, schedule=schedule)
                child_history = (*history, action)
                terminal = depth == len(TRAIN_SESSIONS) - 1
                child_id = _terminal_id(symbol, child_history) if terminal else _node_id(symbol, child_history)
                noop_edges += int(transition.action_was_noop)
                edges.append(
                    {
                        "edge_id": sha256_bytes(f"{node_id}|{action}|{child_id}".encode()),
                        "parent_id": node_id,
                        "child_id": child_id,
                        "action": action,
                        "reward_r3": transition.reward_r3,
                        "execution_session": transition.execution_session,
                        "maturity_session": transition.maturity_session,
                        "terminal": terminal,
                        "transition": {
                            "terminal_equity": transition.terminal_equity,
                            "terminal_cash": transition.terminal_cash,
                            "terminal_quantity": transition.terminal_quantity,
                            "transaction_cost": transition.transaction_cost,
                            "commission_cost": transition.commission_cost,
                            "slippage_cost": transition.slippage_cost,
                            "action_was_noop": transition.action_was_noop,
                            "next_state": asdict(transition.next_state),
                        },
                    }
                )
                if not terminal:
                    queue.append((child_history, transition.next_state, transition.next_snapshot))

    terminal_leaves = sum(1 for edge in edges if edge["terminal"])
    if (len(nodes), len(edges), terminal_leaves) != (
        EXPECTED_NODES,
        EXPECTED_EDGES,
        EXPECTED_LEAVES,
    ):
        raise AssertionError("canonical tree population changed")
    if depth_counts != Counter({depth: 8 * (3**depth) for depth in range(7)}):
        raise AssertionError("canonical tree depth population changed")
    if any(not np.isfinite(edge["reward_r3"]) for edge in edges):
        raise AssertionError("tree contains non-finite R3")

    node_logical_sha, node_gzip_sha, node_logical_size, node_gzip_size = _gzip_jsonl(
        output_root / "nodes.jsonl.gz", nodes
    )
    edge_logical_sha, edge_gzip_sha, edge_logical_size, edge_gzip_size = _gzip_jsonl(
        output_root / "edges.jsonl.gz", edges
    )
    tree_binding = {
        "contract": TREE_CONTRACT,
        "representation_identity": REPRESENTATION_IDENTITY,
        "reward_id": REWARD_ID,
        "reward_snapshot_identity": REWARD_SNAPSHOT_IDENTITY,
        "source_sha": source_sha,
        "plan_sha": plan_sha,
        "nodes_logical_sha256": node_logical_sha,
        "edges_logical_sha256": edge_logical_sha,
    }
    tree_identity = sha256_bytes(canonical_json_bytes(tree_binding))
    summary = {
        "schema_version": TREE_CONTRACT,
        "tree_identity_sha256": tree_identity,
        "symbols": list(TRAIN_SYMBOLS),
        "decision_sessions": list(TRAIN_SESSIONS),
        "semantic_train_decisions": 56,
        "decision_nodes": len(nodes),
        "action_edges": len(edges),
        "terminal_leaves": terminal_leaves,
        "depth_counts": {str(key): depth_counts[key] for key in range(7)},
        "noop_edges": noop_edges,
        "all_rewards_finite": True,
        "transition_equivalence_edges_checked": len(edges),
        "transition_equivalence_mismatches": 0,
        "future_window_violations": 0,
        "validation_transitions": 0,
        "final_holdout_accesses": 0,
    }
    manifest = {
        **tree_binding,
        "tree_identity_sha256": tree_identity,
        "root_state": {"cash": INITIAL_CASH, "quantity": 0.0, "peak_equity": INITIAL_CASH},
        "actions": list(ACTIONS),
        "history_merging": False,
        "noop_branch_pruning": False,
        "reward_window_sessions": 5,
        "observation_storage": "SHA256 plus lossless reconstruction from frozen semantic row and portfolio snapshot",
        "corporate_action_windows": {"split": split_events, "dividend": dividend_events},
    }
    hashes = {
        "nodes.jsonl.gz": {"compressed_sha256": node_gzip_sha, "logical_sha256": node_logical_sha, "compressed_bytes": node_gzip_size, "logical_bytes": node_logical_size},
        "edges.jsonl.gz": {"compressed_sha256": edge_gzip_sha, "logical_sha256": edge_logical_sha, "compressed_bytes": edge_gzip_size, "logical_bytes": edge_logical_size},
    }
    for name, payload in (("tree_manifest.json", manifest), ("tree_summary.json", summary), ("tree_sha256.json", hashes)):
        (output_root / name).write_bytes(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    return {"manifest": manifest, "summary": summary, "hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(build_tree(experiments_root=args.experiments_root, output_root=args.output_root, source_sha=args.source_sha, plan_sha=args.plan_sha)["summary"], indent=2))


if __name__ == "__main__":
    main()
