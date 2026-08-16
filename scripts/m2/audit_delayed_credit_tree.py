#!/usr/bin/env python3
"""Audit and finalise the corrected M2 delayed-credit tree archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from scripts.m2.pa_ctppo import (
    EXPECTED_FAST_PARAMETERS,
    EXPECTED_PARAMETERS,
    METHOD_ID,
    PromptAnchoredActorCritic,
    exact_local_credit_policy_evaluation,
    exact_ppo_loss,
)
from scripts.m2.reward_simulator import REWARD_IDS, candidate_rewards, simulate_counterfactuals
from scripts.m2.rl_environment import (
    ACTIONS,
    TRAIN_SESSIONS,
    TRAIN_SYMBOLS,
    SequentialPortfolioState,
    load_market_snapshot,
    reward_window,
    simulate_local_credit,
    simulate_weekly_transition,
    weekly_transition_window,
)
from scripts.m2.semantic_state_representation import build_actor_observation
from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.execution import Broker
from tradingagents.backtesting.models import Order
from tradingagents.backtesting.portfolio import Portfolio

STATE_FIELDS = (
    "cash",
    "quantity",
    "average_entry_price",
    "open_position_commission",
    "peak_equity",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_identity(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    logical = gzip.decompress(path.read_bytes())
    return [json.loads(line) for line in logical.splitlines()], logical


def _seed_portfolio(symbol: str, state: SequentialPortfolioState) -> Portfolio:
    portfolio = Portfolio(symbol, initial_cash=max(state.peak_equity, 1.0))
    for field, value in asdict(state).items():
        setattr(portfolio, field, value)
    return portfolio


def _independent_transition(
    symbol: str,
    state: SequentialPortfolioState,
    action: str,
    decision_session: str,
    child_session: str,
    market,
    schedule: ExchangeSchedule,
) -> dict[str, Any]:
    window = weekly_transition_window(
        symbol, decision_session, child_session, market, schedule=schedule
    )
    portfolio = _seed_portfolio(symbol, state)
    first = window.bars[0]
    if first.dividend_per_share:
        portfolio.apply_dividend(first.dividend_per_share)
    if first.split_ratio:
        portfolio.apply_split(first.split_ratio)
    fill = None
    noop = action == "HOLD"
    if action != "HOLD":
        order = Order(
            order_id=f"m2-v2-audit:{symbol}:{decision_session}:{action}",
            symbol=symbol,
            created_at=schedule.session_close(decision_session).to_pydatetime(),
            intended_execution_session=first.session,
            target_weight=1.0 if action == "BUY" else 0.0,
            current_weight=portfolio.weight(first.open_price),
            side=action,
        )
        fill = Broker(
            commission_bps=5.0, slippage_bps=5.0, fractional_shares=True
        ).execute(
            order,
            portfolio,
            first.open_price,
            schedule.session_open(first.session).to_pydatetime(),
        )
        noop = fill is None and order.status == "noop"
    snapshot = None
    for index, bar in enumerate(window.bars):
        if index:
            if bar.dividend_per_share:
                portfolio.apply_dividend(bar.dividend_per_share)
            if bar.split_ratio:
                portfolio.apply_split(bar.split_ratio)
        snapshot = portfolio.snapshot(
            bar.session,
            schedule.session_close(bar.session).to_pydatetime(),
            bar.close_price,
        )
    assert snapshot is not None
    return {
        "state": {field: getattr(portfolio, field) for field in asdict(state)},
        "equity": snapshot.equity,
        "drawdown": snapshot.current_drawdown,
        "cost": 0.0 if fill is None else fill.total_transaction_cost,
        "noop": noop,
    }


def _assert_close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10):
        raise AssertionError(f"{label} mismatch: {left} != {right}")


def audit(*, experiments_root: Path, archive_root: Path) -> dict[str, Any]:
    tree_root = archive_root / "train_environment"
    nodes, nodes_logical = _read_jsonl(tree_root / "nodes.jsonl.gz")
    edges, edges_logical = _read_jsonl(tree_root / "edges.jsonl.gz")
    stored_hashes = _read_json(tree_root / "tree_sha256.json")
    if hashlib.sha256(nodes_logical).hexdigest() != stored_hashes["nodes.jsonl.gz"]["logical_sha256"]:
        raise AssertionError("nodes logical SHA mismatch")
    if hashlib.sha256(edges_logical).hexdigest() != stored_hashes["edges.jsonl.gz"]["logical_sha256"]:
        raise AssertionError("edges logical SHA mismatch")
    if (len(nodes), len(edges)) != (8_744, 26_232):
        raise AssertionError("corrected tree population changed")

    edge_by_id = {edge["edge_id"]: edge for edge in edges}
    edge_groups: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        edge_groups.setdefault(edge["parent_node_id"], []).append(edge)
    for group in edge_groups.values():
        if tuple(edge["action"] for edge in group) != ACTIONS:
            raise AssertionError("every node must have ordered BUY/HOLD/SELL edges")

    schedule = ExchangeSchedule()
    reward_root = experiments_root / "experiments/M2/development/reward_study_v1"
    markets = {
        symbol: load_market_snapshot(
            reward_root / f"inputs/market_snapshot/{symbol}.csv", symbol
        )
        for symbol in TRAIN_SYMBOLS
    }

    state_checks = 0
    for node in nodes:
        if node["depth"] == 6:
            continue
        state = (
            SequentialPortfolioState()
            if node["parent_edge_id"] is None
            else SequentialPortfolioState(
                **edge_by_id[node["parent_edge_id"]]["state_transition"]["next_state"]
            )
        )
        child_session = TRAIN_SESSIONS[node["depth"] + 1]
        for edge in edge_groups[node["node_id"]]:
            expected = _independent_transition(
                node["symbol"],
                state,
                edge["action"],
                node["decision_session"],
                child_session,
                markets[node["symbol"]],
                schedule,
            )
            stored = edge["state_transition"]
            assert stored is not None
            for field in STATE_FIELDS:
                _assert_close(expected["state"][field], stored["next_state"][field], field)
            _assert_close(expected["equity"], stored["next_decision_equity"], "equity")
            stored_drawdown = (
                stored["next_decision_equity"] / stored["next_state"]["peak_equity"] - 1.0
            )
            _assert_close(expected["drawdown"], stored_drawdown, "drawdown")
            _assert_close(expected["cost"], stored["transaction_cost"], "cost")
            if expected["noop"] != stored["action_was_noop"]:
                raise AssertionError("weekly transition no-op mismatch")
            state_checks += 1
    if state_checks != 8_736:
        raise AssertionError("nonterminal transition audit count changed")

    local_credit_checks = 0
    for node in nodes:
        state = (
            SequentialPortfolioState()
            if node["parent_edge_id"] is None
            else SequentialPortfolioState(
                **edge_by_id[node["parent_edge_id"]]["state_transition"]["next_state"]
            )
        )
        window = reward_window(
            node["symbol"],
            node["decision_session"],
            markets[node["symbol"]],
            schedule=schedule,
        )
        frozen = simulate_counterfactuals(window, state.reward_state(), schedule=schedule)
        assert frozen.outcomes is not None
        for edge in edge_groups[node["node_id"]]:
            expected = candidate_rewards(frozen.outcomes[edge["action"]])[REWARD_IDS[2]]
            _assert_close(expected, edge["local_r3"], "local R3")
            if frozen.maturity_session != edge["reward_maturity_session"]:
                raise AssertionError("reward maturity mismatch")
            local_credit_checks += 1
    if local_credit_checks != 26_232:
        raise AssertionError("local-credit audit count changed")

    overlap_counts: dict[str, int] = {}
    for edge in edges:
        classification = edge["overlap_classification"]
        overlap_counts[classification] = overlap_counts.get(classification, 0) + 1
        if edge["terminal"] and edge["state_transition"] is not None:
            raise AssertionError("terminal action must not create a weekly child state")
        if not edge["terminal"] and edge["state_transition"] is None:
            raise AssertionError("nonterminal action lacks weekly child state")
    if overlap_counts.get("credit_matures_after_child") != 648:
        raise AssertionError("Memorial Day overlap count changed")

    poison_node = next(
        node
        for node in nodes
        if node["symbol"] == "AAPL"
        and node["decision_session"] == "2023-05-26"
        and node["history"] == ["BUY", "HOLD", "HOLD"]
    )
    poison_state = SequentialPortfolioState(
        **edge_by_id[poison_node["parent_edge_id"]]["state_transition"]["next_state"]
    )
    base_market = markets["AAPL"]
    poisoned_market = dict(base_market)
    poisoned_market["2023-06-05"] = replace(
        base_market["2023-06-05"],
        open_price=base_market["2023-06-05"].open_price * 7.0,
        close_price=base_market["2023-06-05"].close_price / 7.0,
    )
    base_transition = simulate_weekly_transition(
        weekly_transition_window(
            "AAPL", "2023-05-26", "2023-06-02", base_market, schedule=schedule
        ),
        poison_state,
        "BUY",
        schedule=schedule,
    )
    poisoned_transition = simulate_weekly_transition(
        weekly_transition_window(
            "AAPL", "2023-05-26", "2023-06-02", poisoned_market, schedule=schedule
        ),
        poison_state,
        "BUY",
        schedule=schedule,
    )
    if base_transition.next_state != poisoned_transition.next_state:
        raise AssertionError("June-5 poison changed June-2 child state")
    representation_root = (
        experiments_root / "experiments/M2/development/semantic_state_representation_v1"
    )
    semantic_matrix = np.load(
        representation_root / "embeddings/semantic_base.npy", allow_pickle=False
    )
    june_2_semantic = semantic_matrix[4]
    base_observation = build_actor_observation(
        june_2_semantic, base_transition.next_snapshot
    )
    poisoned_observation = build_actor_observation(
        june_2_semantic, poisoned_transition.next_snapshot
    )
    if base_observation.tobytes() != poisoned_observation.tobytes():
        raise AssertionError("June-5 poison changed June-2 Actor observation")
    base_credit = simulate_local_credit(
        reward_window("AAPL", "2023-05-26", base_market, schedule=schedule),
        poison_state,
        schedule=schedule,
    )
    poisoned_credit = simulate_local_credit(
        reward_window("AAPL", "2023-05-26", poisoned_market, schedule=schedule),
        poison_state,
        schedule=schedule,
    )
    if base_credit.input_sha256_by_action == poisoned_credit.input_sha256_by_action:
        raise AssertionError("June-5 poison did not change parent credit identity")

    descendant_first = simulate_local_credit(
        reward_window("AAPL", "2023-05-26", base_market, schedule=schedule),
        poison_state,
        schedule=schedule,
    )
    simulate_weekly_transition(
        weekly_transition_window(
            "AAPL", "2023-06-02", "2023-06-09", base_market, schedule=schedule
        ),
        base_transition.next_state,
        "BUY",
        schedule=schedule,
    )
    simulate_weekly_transition(
        weekly_transition_window(
            "AAPL", "2023-06-02", "2023-06-09", base_market, schedule=schedule
        ),
        base_transition.next_state,
        "SELL",
        schedule=schedule,
    )
    descendant_second = simulate_local_credit(
        reward_window("AAPL", "2023-05-26", base_market, schedule=schedule),
        poison_state,
        schedule=schedule,
    )
    if descendant_first != descendant_second:
        raise AssertionError("descendant action contaminated parent local credit")

    index_by_node = {node["node_id"]: index for index, node in enumerate(nodes)}
    rewards = torch.empty((len(nodes), 3), dtype=torch.float64)
    children = torch.full((len(nodes), 3), -1, dtype=torch.int64)
    depths = torch.tensor([node["depth"] for node in nodes], dtype=torch.int64)
    probabilities = torch.empty((len(nodes), 3), dtype=torch.float64)
    for index, node in enumerate(nodes):
        group = edge_groups[node["node_id"]]
        rewards[index] = torch.tensor([edge["local_r3"] for edge in group])
        for action_index, edge in enumerate(group):
            if not edge["terminal"]:
                children[index, action_index] = index_by_node[edge["child_node_id"]]
        prompt_index = int(np.argmax(semantic_matrix[node["semantic_row"]["index"], 3073:3076]))
        probabilities[index] = 1.0 / 6.0
        probabilities[index, prompt_index] = 2.0 / 3.0
    roots = torch.tensor(
        [index for index, node in enumerate(nodes) if node["depth"] == 0]
    )
    evaluation = exact_local_credit_policy_evaluation(
        rewards, children, probabilities, depths, roots
    )
    occupancy_by_depth = {
        str(depth): float(evaluation.occupancy[depths == depth].sum())
        for depth in range(7)
    }
    if any(abs(value - 1.0) > 1e-12 for value in occupancy_by_depth.values()):
        raise AssertionError("weekly state occupancy invariant failed")
    advantage_residual = float(
        torch.abs(torch.sum(probabilities * evaluation.advantages, dim=1)).max()
    )
    if advantage_residual > 1e-10:
        raise AssertionError("local counterfactual advantage identity failed")
    identity_loss = exact_ppo_loss(
        probabilities,
        probabilities,
        evaluation.advantages,
        evaluation.values,
        evaluation.values,
        evaluation.occupancy,
    )

    model = PromptAnchoredActorCritic()
    parameter_breakdown = {
        name: sum(parameter.numel() for parameter in getattr(model, name).parameters())
        for name in (
            "semantic_adapter",
            "actor_trunk",
            "critic_trunk",
            "residual_head",
            "gate_head",
            "value_head",
        )
    }
    if model.parameter_count() != EXPECTED_PARAMETERS:
        raise AssertionError("total parameter count changed")
    if model.fast_parameter_count() != EXPECTED_FAST_PARAMETERS:
        raise AssertionError("Formal fast parameter count changed")

    audits = archive_root / "audits"
    audit_payloads = {
        "chronology_audit.json": {
            "schema_version": "M2-10A-CHRONOLOGY-AUDIT-v1",
            "nonterminal_edges": state_checks,
            "local_credit_edges": local_credit_checks,
            "overlap_counts": overlap_counts,
            "after_child_maturity_edges": 648,
            "child_states_using_post_child_market_bars": 0,
            "descendant_action_contamination_of_parent_credit": 0,
            "validation_transitions": 0,
            "pass": True,
        },
        "state_transition_equivalence.json": {
            "schema_version": "M2-10A-STATE-TRANSITION-EQUIVALENCE-v1",
            "transitions_checked": state_checks,
            "cash_mismatches": 0,
            "quantity_mismatches": 0,
            "average_entry_mismatches": 0,
            "equity_mismatches": 0,
            "peak_equity_mismatches": 0,
            "drawdown_mismatches": 0,
            "cost_mismatches": 0,
            "pass": True,
        },
        "local_credit_equivalence.json": {
            "schema_version": "M2-10A-LOCAL-CREDIT-EQUIVALENCE-v1",
            "action_edges_checked": local_credit_checks,
            "r3_mismatches": 0,
            "reward_maturity_mismatches": 0,
            "reward_simulator_modified": False,
            "pass": True,
        },
        "holiday_poison_test.json": {
            "schema_version": "M2-10A-HOLIDAY-POISON-v1",
            "decision_session": "2023-05-26",
            "child_decision_session": "2023-06-02",
            "reward_maturity_session": "2023-06-05",
            "june_5_only_data_mutated": True,
            "june_2_child_state_changed": False,
            "june_2_actor_observation_changed": False,
            "parent_credit_input_identity_changed": True,
            "pass": True,
        },
        "descendant_action_independence.json": {
            "schema_version": "M2-10A-DESCENDANT-INDEPENDENCE-v1",
            "descendant_actions_compared": ["BUY", "SELL"],
            "parent_r3_identical": True,
            "parent_reward_maturity_identical": True,
            "parent_credit_input_identical": True,
            "pass": True,
        },
        "objective_sanity.json": {
            "schema_version": "M2-10A-OBJECTIVE-SANITY-v1",
            "nodes_checked": len(nodes),
            "occupancy_by_depth": occupancy_by_depth,
            "maximum_local_advantage_residual": advantage_residual,
            "bellman_bootstrap": False,
            "gamma": None,
            "gae": False,
            "exact_action_expectation": True,
            "action_sampling": False,
            "identity_policy_surrogate": float(identity_loss["policy_surrogate"]),
            "identity_value_loss": float(identity_loss["value_loss"]),
            "identity_total_loss": float(identity_loss["loss"]),
            "pass": True,
        },
        "architecture_parameter_count.json": {
            "schema_version": "M2-10A-ARCHITECTURE-PARAMETERS-v1",
            "method_id": METHOD_ID,
            "breakdown": parameter_breakdown,
            "trainable_parameters": model.parameter_count(),
            "formal_fast_parameters_per_symbol": model.fast_parameter_count(),
            "pass": True,
        },
    }
    for name, payload in audit_payloads.items():
        _write_json(audits / name, payload)

    tree_manifest = _read_json(tree_root / "tree_manifest.json")
    method_contract = {
        "schema_version": "M2-AGENTIC-RL-METHOD-FREEZE-v2",
        "task_id": "M2-10A",
        "method_id": METHOD_ID,
        "superseded_method": "M2-PA-CTPPO-v1",
        "original_plan_sha": "89c84945e19a5a532b7afc042300e2ea17377a79",
        "blocked_method_runner_sha": "f779069ab4b133b5421ff4d9b6a972e44f48712e",
        "correctness_erratum": True,
        "representation_identity": "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe",
        "representation_dimension": 3080,
        "tree_contract": "M2_TRAIN_COUNTERFACTUAL_TREE-v2",
        "train_counterfactual_tree_v2_identity_sha256": tree_manifest[
            "train_counterfactual_tree_v2_identity_sha256"
        ],
        "decision_nodes": 8_744,
        "action_edges": 26_232,
        "terminal_leaves": 17_496,
        "nonterminal_transitions": 8_736,
        "after_child_maturity_edges": 648,
        "weekly_state_transition": "next frozen weekly decision close",
        "reward_maturity": "fifth subsequent XNYS close",
        "state_reward_time_decoupled": True,
        "local_credit": "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY",
        "counterfactual_value_equation": "V_cf(s) = sum_a pi_old(a|s) R3(s,a)",
        "counterfactual_advantage_equation": "A_cf(s,a) = R3(s,a) - V_cf(s)",
        "bellman_bootstrap": False,
        "gamma": None,
        "gae": False,
        "exact_action_expectation": True,
        "action_sampling": False,
        "ppo_clip": 0.2,
        "value_loss_coefficient": 0.5,
        "trainable_parameters": 20_197,
        "fast_parameters_per_symbol": 165,
        "validation_performance_used": False,
        "final_holdout_used": False,
        "formal_result_used": False,
        "deepseek_calls": 0,
        "qwen_calls": 0,
        "aws_gpu_hours": 0,
    }
    identity_payload = {
        key: method_contract[key]
        for key in (
            "method_id",
            "representation_identity",
            "train_counterfactual_tree_v2_identity_sha256",
            "counterfactual_value_equation",
            "counterfactual_advantage_equation",
            "ppo_clip",
            "trainable_parameters",
            "fast_parameters_per_symbol",
        )
    }
    method_contract["method_identity_sha256"] = _canonical_identity(identity_payload)
    _write_json(archive_root / "method_contract.json", method_contract)

    files = {
        str(path.relative_to(archive_root)): _sha256(path)
        for path in sorted(archive_root.rglob("*"))
        if path.is_file() and path.name != "final_sha256.json"
    }
    _write_json(
        archive_root / "final_sha256.json",
        {"schema_version": "M2-10A-FINAL-SHA256-v1", "files": files},
    )
    return {
        "tree_identity": tree_manifest[
            "train_counterfactual_tree_v2_identity_sha256"
        ],
        "method_identity": method_contract["method_identity_sha256"],
        "state_transition_checks": state_checks,
        "local_credit_checks": local_credit_checks,
        "after_child_maturity_edges": 648,
        "maximum_local_advantage_residual": advantage_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                experiments_root=args.experiments_root,
                archive_root=args.archive_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
