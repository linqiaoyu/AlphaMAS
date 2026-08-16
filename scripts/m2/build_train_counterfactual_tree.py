#!/usr/bin/env python3
"""Build the canonical delayed-credit M2 TRAIN weekly-state tree."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    simulate_local_credit,
    simulate_weekly_transition,
    weekly_transition_window,
)
from scripts.m2.semantic_state_representation import build_actor_observation
from tradingagents.backtesting.calendar import ExchangeSchedule

REPRESENTATION_IDENTITY = "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe"
REWARD_ID = "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"
REWARD_SNAPSHOT_IDENTITY = "3afc723888666e0ca3a12219a57dc577d6cf3953717da731b774634f4aff1445"
EXPECTED_NODES = 8_744
EXPECTED_EDGES = 26_232
EXPECTED_LEAVES = 17_496
EXPECTED_NONTERMINAL_TRANSITIONS = 8_736
EXPECTED_AFTER_CHILD_CREDITS = 648


def _node_id(symbol: str, history: tuple[str, ...]) -> str:
    return sha256_bytes(f"{TREE_CONTRACT}|{symbol}|{'/'.join(history)}".encode())


def _terminal_id(symbol: str, history: tuple[str, ...]) -> str:
    return sha256_bytes(f"{TREE_CONTRACT}|terminal|{symbol}|{'/'.join(history)}".encode())


def _edge_id(parent_id: str, action: str, child_id: str) -> str:
    return sha256_bytes(f"{TREE_CONTRACT}|{parent_id}|{action}|{child_id}".encode())


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


def _overlap(maturity: str, child: str | None) -> str:
    if child is None:
        return "terminal_no_child"
    if maturity < child:
        return "credit_matures_before_child"
    if maturity == child:
        return "credit_matures_on_child"
    return "credit_matures_after_child"


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
    overlap_counts: Counter[str] = Counter()
    nonterminal_transitions = 0
    noop_credit_edges = 0
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
            node_id = _node_id(symbol, history)
            if history:
                parent_node_id = _node_id(symbol, history[:-1])
                parent_edge_id = _edge_id(parent_node_id, history[-1], node_id)
            else:
                parent_edge_id = None
            nodes.append(
                {
                    "node_id": node_id,
                    "parent_edge_id": parent_edge_id,
                    "symbol": symbol,
                    "depth": depth,
                    "decision_session": decision_session,
                    "semantic_row": {"identity": row["case_id"], "index": row["row_index"]},
                    "semantic_base_sha256": sha256_bytes(semantic_base.tobytes()),
                    "portfolio_snapshot": {
                        "cash": state.cash,
                        "quantity": state.quantity,
                        "average_entry_price": state.average_entry_price,
                        "open_position_commission": state.open_position_commission,
                        "peak_equity": state.peak_equity,
                        "current_equity": snapshot.equity,
                        "current_drawdown": snapshot.current_drawdown,
                    },
                    "actor_observation": {
                        "sha256": sha256_bytes(observation.tobytes()),
                        "shape": [3080],
                        "dtype": "float32",
                        "finite": True,
                        "reconstruction": "frozen semantic row + frozen portfolio-state helper",
                    },
                    "history": list(history),
                }
            )
            depth_counts[depth] += 1
            credit = simulate_local_credit(
                reward_window(symbol, decision_session, market, schedule=schedule),
                state,
                schedule=schedule,
            )
            terminal = depth == len(TRAIN_SESSIONS) - 1
            child_decision_session = None if terminal else TRAIN_SESSIONS[depth + 1]
            transition_window = (
                None
                if terminal
                else weekly_transition_window(
                    symbol,
                    decision_session,
                    child_decision_session,
                    market,
                    schedule=schedule,
                )
            )
            for action in ACTIONS:
                child_history = (*history, action)
                child_id = (
                    _terminal_id(symbol, child_history)
                    if terminal
                    else _node_id(symbol, child_history)
                )
                transition = (
                    None
                    if transition_window is None
                    else simulate_weekly_transition(
                        transition_window, state, action, schedule=schedule
                    )
                )
                if transition is not None:
                    nonterminal_transitions += 1
                    queue.append(
                        (child_history, transition.next_state, transition.next_snapshot)
                    )
                overlap = _overlap(
                    credit.reward_maturity_session, child_decision_session
                )
                overlap_counts[overlap] += 1
                credit_terminal = credit.terminal_by_action[action]
                noop_credit_edges += int(credit_terminal["action_was_noop"])
                edges.append(
                    {
                        "edge_id": _edge_id(node_id, action, child_id),
                        "parent_node_id": node_id,
                        "action": action,
                        "execution_session": credit.execution_session,
                        "child_node_id": child_id,
                        "child_decision_session": child_decision_session,
                        "reward_maturity_session": credit.reward_maturity_session,
                        "overlap_classification": overlap,
                        "local_r3": credit.rewards_r3[action],
                        "credit_input_sha256": credit.input_sha256_by_action[action],
                        "credit_terminal": credit_terminal,
                        "state_transition_sha256": (
                            None if transition is None else transition.state_transition_sha256
                        ),
                        "state_transition": (
                            None
                            if transition is None
                            else {
                                "input_sha256": transition.state_transition_input_sha256,
                                "next_decision_equity": transition.next_decision_equity,
                                "transaction_cost": transition.transaction_cost,
                                "commission_cost": transition.commission_cost,
                                "slippage_cost": transition.slippage_cost,
                                "action_was_noop": transition.action_was_noop,
                                "next_state": asdict(transition.next_state),
                            }
                        ),
                        "terminal": terminal,
                    }
                )

    terminal_leaves = sum(edge["terminal"] for edge in edges)
    population = (len(nodes), len(edges), terminal_leaves, nonterminal_transitions)
    expected_population = (
        EXPECTED_NODES,
        EXPECTED_EDGES,
        EXPECTED_LEAVES,
        EXPECTED_NONTERMINAL_TRANSITIONS,
    )
    if population != expected_population:
        raise AssertionError(f"canonical v2 tree population changed: {population}")
    if depth_counts != Counter({depth: 8 * (3**depth) for depth in range(7)}):
        raise AssertionError("canonical tree depth population changed")
    if overlap_counts["credit_matures_after_child"] != EXPECTED_AFTER_CHILD_CREDITS:
        raise AssertionError("Memorial Day delayed-credit overlap population changed")
    if any(not np.isfinite(edge["local_r3"]) for edge in edges):
        raise AssertionError("tree contains non-finite local R3")

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
        "weekly_transition_semantics": "execution open through next frozen decision close",
        "reward_maturity_semantics": "independent fifth subsequent XNYS close",
        "source_sha": source_sha,
        "plan_sha": plan_sha,
        "nodes_logical_sha256": node_logical_sha,
        "edges_logical_sha256": edge_logical_sha,
    }
    tree_identity = sha256_bytes(canonical_json_bytes(tree_binding))
    summary = {
        "schema_version": TREE_CONTRACT,
        "train_counterfactual_tree_v2_identity_sha256": tree_identity,
        "symbols": list(TRAIN_SYMBOLS),
        "decision_sessions": list(TRAIN_SESSIONS),
        "semantic_train_decisions": 56,
        "decision_nodes": len(nodes),
        "action_edges": len(edges),
        "terminal_leaves": terminal_leaves,
        "nonterminal_transitions": nonterminal_transitions,
        "local_r3_credits": len(edges),
        "depth_counts": {str(key): depth_counts[key] for key in range(7)},
        "overlap_counts": dict(sorted(overlap_counts.items())),
        "noop_credit_edges": noop_credit_edges,
        "all_local_r3_finite": True,
        "child_states_using_post_child_information": 0,
        "descendant_action_contamination": 0,
        "validation_transitions": 0,
        "final_holdout_accesses": 0,
    }
    manifest = {
        **tree_binding,
        "train_counterfactual_tree_v2_identity_sha256": tree_identity,
        "root_state": {"cash": INITIAL_CASH, "quantity": 0.0, "peak_equity": INITIAL_CASH},
        "actions": list(ACTIONS),
        "history_merging": False,
        "noop_branch_pruning": False,
        "state_reward_time_decoupled": True,
        "bellman_bootstrap": False,
        "gamma": None,
        "reward_window_sessions": 5,
        "observation_storage": "SHA256 plus lossless reconstruction from frozen semantic row and weekly portfolio snapshot",
    }
    hashes = {
        "nodes.jsonl.gz": {
            "compressed_sha256": node_gzip_sha,
            "logical_sha256": node_logical_sha,
            "compressed_bytes": node_gzip_size,
            "logical_bytes": node_logical_size,
        },
        "edges.jsonl.gz": {
            "compressed_sha256": edge_gzip_sha,
            "logical_sha256": edge_logical_sha,
            "compressed_bytes": edge_gzip_size,
            "logical_bytes": edge_logical_size,
        },
    }
    for name, payload in (
        ("tree_manifest.json", manifest),
        ("tree_summary.json", summary),
        ("tree_sha256.json", hashes),
    ):
        (output_root / name).write_bytes(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        )
    return {"manifest": manifest, "summary": summary, "hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    result = build_tree(
        experiments_root=args.experiments_root,
        output_root=args.output_root,
        source_sha=args.source_sha,
        plan_sha=args.plan_sha,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
