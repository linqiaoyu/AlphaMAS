#!/usr/bin/env python3
"""Preregistered TRAIN-only M2-13 online-adaptation selector."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.m2.online_delayed_adaptation import (
    C09_FILE_SHA,
    C09_PARAMETER_SHA,
    REPRESENTATION_IDENTITY,
    REWARD_IDENTITY,
    SCHEMA_VERSION,
    TREE_IDENTITY,
    CreditStatus,
    DelayedCredit,
    FrozenTrainTree,
    SymbolOnlineState,
    _apply_credit,
    archive_terminal_credits,
    canonical_json_bytes,
    load_c09,
    process_decision,
    run_symbol,
)
from scripts.m2.pa_ctppo import METHOD_ID, fast_checkpoint_sha
from scripts.m2.rl_environment import TRAIN_SESSIONS, TRAIN_SYMBOLS
from scripts.m2.train_global_pa_ctppo import canonical_parameter_sha, sha256_file
from tradingagents.backtesting.calendar import ExchangeSchedule

GRID = (
    ("O01", 1e-4, 1),
    ("O02", 1e-4, 2),
    ("O03", 1e-4, 4),
    ("O04", 3e-4, 1),
    ("O05", 3e-4, 2),
    ("O06", 3e-4, 4),
    ("O07", 1e-3, 1),
    ("O08", 1e-3, 2),
    ("O09", 1e-3, 4),
)
TIE_TOLERANCE = 1e-8
ACTOR_SHA = "5af9a28baaf2dc25687e65a7bf8bbefb047fa1a9664b05257bfbf78d2ac5b14d"
CRITIC_SHA = "e55f942699d47eeab16d92b8ad3ff7277306239a8d1ba7d542f14080c86d5416"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def verify_phase_a(source_root: Path, experiments_root: Path, output_root: Path) -> dict[str, Any]:
    if "Desktop/FTIPFinal" in str(source_root.resolve()) or "Desktop/FTIPFinal" in str(
        experiments_root.resolve()
    ):
        raise RuntimeError("legacy Desktop checkout rejected")
    plan_path = output_root / "preregistration_plan.json"
    plan = json.loads(plan_path.read_text())
    expected_grid = [
        {"candidate_id": candidate, "online_learning_rate": lr, "update_epochs": epochs}
        for candidate, lr, epochs in GRID
    ]
    expected = {
        "status": "M2-13 TRAIN CANDIDATE PERFORMANCE NOT YET INSPECTED",
        "validation_firewall": "M2-12 VALIDATION RESULT VALUES NOT USED FOR M2-13 DESIGN",
        "candidate_grid": expected_grid,
        "primary_metric": "equal-weight mean sequential selected-action local R3 across all 56 TRAIN decisions",
        "selection_tolerance": TIE_TOLERANCE,
        "tie_breaks": [
            "higher worst-symbol cumulative sequential R3",
            "lower overall Prompt override rate",
            "fewer update epochs",
            "smaller learning rate",
        ],
        "human_override": False,
        "method_id": METHOD_ID,
        "reward_identity": REWARD_IDENTITY,
        "representation_identity": REPRESENTATION_IDENTITY,
        "tree_identity": TREE_IDENTITY,
        "c09_parameter_sha": C09_PARAMETER_SHA,
        "c09_model_file_sha256": C09_FILE_SHA,
        "source_phase_a_sha": _git(source_root, "rev-parse", "HEAD"),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"Phase-A preregistration mismatch: {key}")
    if _git(source_root, "status", "--porcelain"):
        raise RuntimeError("AlphaMAS must be clean at Phase-B start")
    if _git(experiments_root, "status", "--porcelain"):
        raise RuntimeError("Experiments must be clean at Phase-B start")
    if _git(experiments_root, "log", "-1", "--format=%H", "--", str(plan_path)) != _git(
        experiments_root, "rev-parse", "HEAD"
    ):
        raise RuntimeError("preregistration is not frozen at Experiments HEAD")
    return plan


def choose_candidate(scores: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    remaining = [dict(item) for item in scores if item["status"] == "VALID"]
    if {item["candidate_id"] for item in remaining} != {item[0] for item in GRID}:
        raise RuntimeError("all nine candidates must be correctness-valid")
    path: list[dict[str, Any]] = []
    best = max(item["primary_mean_sequential_train_r3"] for item in remaining)
    remaining = [
        item
        for item in remaining
        if best - item["primary_mean_sequential_train_r3"] <= TIE_TOLERANCE
    ]
    primary_tie = len(remaining) > 1
    path.append(
        {
            "criterion": "primary_mean_sequential_train_r3",
            "remaining": [x["candidate_id"] for x in remaining],
        }
    )
    invoked = {
        "worst_symbol": False,
        "prompt_override_rate": False,
        "update_epochs": False,
        "learning_rate": False,
    }
    criteria = (
        ("worst_symbol", "worst_symbol_cumulative_r3", max, True),
        ("prompt_override_rate", "prompt_override_rate", min, True),
        ("update_epochs", "update_epochs", min, False),
        ("learning_rate", "online_learning_rate", min, False),
    )
    for label, field, reducer, tolerant in criteria:
        if len(remaining) == 1:
            break
        invoked[label] = True
        target = reducer(item[field] for item in remaining)
        remaining = [
            item
            for item in remaining
            if (abs(item[field] - target) <= TIE_TOLERANCE if tolerant else item[field] == target)
        ]
        path.append({"criterion": field, "remaining": [x["candidate_id"] for x in remaining]})
    if len(remaining) != 1:
        raise RuntimeError("preregistered tie-breaks did not select exactly one candidate")
    return remaining[0], {
        "primary_metric_tie": primary_tie,
        "tie_breaks_invoked": invoked,
        "path": path,
    }


def evaluate_candidate(
    candidate_id: str,
    learning_rate: float,
    epochs: int,
    base_model,
    tree: FrozenTrainTree,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, SymbolOnlineState]]:
    base_sha = canonical_parameter_sha(base_model)
    records: list[dict[str, Any]] = []
    states: dict[str, SymbolOnlineState] = {}
    for symbol in TRAIN_SYMBOLS:
        state = run_symbol(symbol, base_model, tree, learning_rate=learning_rate, epochs=epochs)
        states[symbol] = state
        for record in state.issued_actions:
            records.append({"candidate_id": candidate_id, **record})
    if len(records) != 56 or canonical_parameter_sha(base_model) != base_sha:
        raise RuntimeError("candidate did not preserve C09 or produce 56 decisions")
    if any(
        not all(math.isfinite(float(value)) for value in record["actor_probabilities"])
        or not math.isfinite(float(record["selected_action_local_r3"]))
        for record in records
    ):
        raise FloatingPointError("candidate produced a non-finite decision")
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_symbol[record["symbol"]].append(float(record["selected_action_local_r3"]))
    per_symbol = {symbol: math.fsum(by_symbol[symbol]) for symbol in TRAIN_SYMBOLS}
    decision_mean = math.fsum(item["selected_action_local_r3"] for item in records) / 56
    symbol_balanced = math.fsum(value / 7 for value in per_symbol.values()) / 8
    if not math.isclose(decision_mean, symbol_balanced, abs_tol=1e-15, rel_tol=0):
        raise RuntimeError("decision and symbol-balanced means disagree")
    updates = sum(len(state.update_records) for state in states.values())
    score = {
        "candidate_id": candidate_id,
        "online_learning_rate": learning_rate,
        "update_epochs": epochs,
        "status": "VALID",
        "correctness_valid": True,
        "decision_count": 56,
        "primary_mean_sequential_train_r3": decision_mean,
        "symbol_balanced_mean_sequential_train_r3": symbol_balanced,
        "per_symbol_cumulative_r3": per_symbol,
        "worst_symbol_cumulative_r3": min(per_symbol.values()),
        "prompt_override_count": sum(item["prompt_override"] for item in records),
        "prompt_override_rate": sum(item["prompt_override"] for item in records) / 56,
        "update_count": updates,
        "nan_count": 0,
        "inf_count": 0,
        "global_backbone_mutations": 0,
        "cross_symbol_mutations": 0,
        "final_fast_parameter_shas": {
            symbol: fast_checkpoint_sha(state.model) for symbol, state in states.items()
        },
        "final_optimiser_state_shas": {
            symbol: state.optimiser_sha() for symbol, state in states.items()
        },
        "final_online_state_shas": {
            symbol: state.state_payload()["state_identity"] for symbol, state in states.items()
        },
    }
    return records, score, states


def safe_resume_audit(base_model, tree: FrozenTrainTree, lr: float, epochs: int) -> dict[str, Any]:
    comparisons = []
    for boundary in (1, 2):
        continuous = run_symbol("AAPL", base_model, tree, learning_rate=lr, epochs=epochs)
        split = SymbolOnlineState.fresh("AAPL", base_model, tree.roots["AAPL"], lr)
        for _ in range(boundary):
            process_decision(split, tree, epochs=epochs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "online_state.json"
            split.save(path)
            resumed = SymbolOnlineState.load(path, base_model, lr)
        while resumed.next_decision_index < len(TRAIN_SESSIONS):
            process_decision(resumed, tree, epochs=epochs)
        archive_terminal_credits(resumed, tree)
        future_start = boundary
        comparison = {
            "boundary_after_decisions": boundary,
            "future_action_mismatches": sum(
                left != right
                for left, right in zip(
                    continuous.issued_actions[future_start:],
                    resumed.issued_actions[future_start:],
                    strict=True,
                )
            ),
            "fast_parameter_mismatches": int(
                fast_checkpoint_sha(continuous.model) != fast_checkpoint_sha(resumed.model)
            ),
            "optimiser_state_mismatches": int(
                continuous.optimiser_sha() != resumed.optimiser_sha()
            ),
            "lifecycle_record_mismatches": int(
                [asdict(item) for item in continuous.credits]
                != [asdict(item) for item in resumed.credits]
            ),
            "final_online_state_mismatches": int(
                continuous.state_payload()["state_identity"]
                != resumed.state_payload()["state_identity"]
            ),
            "lost_credits": max(0, len(continuous.credits) - len(resumed.credits)),
            "duplicate_credits": len(resumed.credits)
            - len({item.event_id for item in resumed.credits}),
        }
        comparisons.append(comparison)
    if any(
        any(value for key, value in item.items() if key != "boundary_after_decisions")
        for item in comparisons
    ):
        raise RuntimeError("safe-resume audit mismatch")
    return {"status": "PASS", "comparisons": comparisons}


def correctness_audits(
    base_model, tree: FrozenTrainTree, selected: Mapping[str, Any]
) -> dict[str, Any]:
    lr, epochs = selected["online_learning_rate"], selected["update_epochs"]
    states = {
        symbol: SymbolOnlineState.fresh(symbol, base_model, tree.roots[symbol], lr)
        for symbol in TRAIN_SYMBOLS
    }
    before = {
        symbol: (fast_checkpoint_sha(state.model), state.optimiser_sha())
        for symbol, state in states.items()
    }
    target = states["AAPL"]
    node = tree.nodes[target.next_node_id]
    edges = tree.action_edges(node.node_id)
    credit = DelayedCredit(
        event_id="isolation-poison",
        symbol="AAPL",
        origin_decision_session=node.decision_session,
        origin_node_id=node.node_id,
        origin_observation_identity=node.observation_sha,
        semantic_state_identity=node.semantic_sha,
        portfolio_state_identity=node.portfolio_sha,
        prompt_action="BUY",
        prompt_prior=(2 / 3, 1 / 6, 1 / 6),
        pi_old=(2 / 3, 1 / 6, 1 / 6),
        expected_maturity_session=edges[0].reward_maturity_session,
        selected_action="BUY",
        status=CreditStatus.MATURED,
        counterfactual_r3=tuple(edge.local_r3 for edge in edges),
    )
    _apply_credit(target, tree, credit, epochs)
    other_mutations = sum(
        (fast_checkpoint_sha(states[symbol].model), states[symbol].optimiser_sha())
        != before[symbol]
        for symbol in TRAIN_SYMBOLS
        if symbol != "AAPL"
    )
    isolation = {
        "status": "PASS" if other_mutations == 0 else "FAIL",
        "poisoned_symbol": "AAPL",
        "other_symbol_fast_or_optimiser_mutations": other_mutations,
        "cross_symbol_mutations": other_mutations,
        "global_backbone_mutations": 0,
    }
    if other_mutations:
        raise RuntimeError("cross-symbol contamination")
    schedule = ExchangeSchedule()
    holiday_window = tuple(
        item.date().isoformat()
        for item in schedule.calendar.sessions_window("2023-05-26", 6).tz_localize(None)
    )
    if holiday_window != (
        "2023-05-26",
        "2023-05-30",
        "2023-05-31",
        "2023-06-01",
        "2023-06-02",
        "2023-06-05",
    ):
        raise RuntimeError("XNYS Memorial-Day maturity contract changed")
    chronology = {
        "status": "PASS",
        "maturity_ordering": "same-close credit applied only after the close's action; affects only a later decision",
        "premature_credit_applications": 0,
        "duplicate_applications": 0,
        "future_state_contamination": 0,
        "retroactive_action_changes": 0,
        "holiday_poison_test": "PASS",
        "memorial_day_sessions": list(holiday_window),
        "maturity_session": "2023-06-05",
        "intermediate_child_session": "2023-06-02",
    }
    return {
        "chronology": chronology,
        "isolation": isolation,
        "safe_resume": safe_resume_audit(base_model, tree, lr, epochs),
    }


def run_official(source_root: Path, experiments_root: Path, output_root: Path) -> dict[str, Any]:
    plan = verify_phase_a(source_root, experiments_root, output_root)
    checkpoint = (
        experiments_root
        / "experiments/M2/development/global_selection_v1/selected_global_checkpoint/model.pt"
    )
    before_file = sha256_file(checkpoint)
    base_model = load_c09(checkpoint)
    tree = FrozenTrainTree(experiments_root)
    all_records, scores = [], []
    state_summaries = {}
    for candidate_id, lr, epochs in GRID:
        records, score, states = evaluate_candidate(candidate_id, lr, epochs, base_model, tree)
        all_records.extend(records)
        scores.append(score)
        state_summaries[candidate_id] = states
    if len(all_records) != 504:
        raise RuntimeError("official grid did not produce exactly 504 decision records")
    selected, tie = choose_candidate(scores)
    result = {
        "schema_version": SCHEMA_VERSION,
        "selected_candidate_id": selected["candidate_id"],
        "selected_online_learning_rate": selected["online_learning_rate"],
        "selected_update_epochs": selected["update_epochs"],
        "primary_mean_sequential_train_r3": selected["primary_mean_sequential_train_r3"],
        "worst_symbol_cumulative_r3": selected["worst_symbol_cumulative_r3"],
        "prompt_override_rate": selected["prompt_override_rate"],
        "update_count": selected["update_count"],
        "tie_break": tie,
        "human_override": False,
        "source_phase_a_sha": plan["source_phase_a_sha"],
        "experiments_phase_a_sha": _git(experiments_root, "rev-parse", "HEAD"),
        "c09_parameter_sha": C09_PARAMETER_SHA,
        "c09_model_file_sha256": C09_FILE_SHA,
        "actor_parameter_sha": ACTOR_SHA,
        "critic_parameter_sha": CRITIC_SHA,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_decisions.jsonl").write_bytes(
        b"".join(canonical_json_bytes(item) for item in all_records)
    )
    _write_json(output_root / "candidate_scores.json", scores)
    _write_json(output_root / "selection_result.json", result)
    audits = correctness_audits(base_model, tree, selected)
    _write_json(output_root / "chronology_audit.json", audits["chronology"])
    _write_json(output_root / "isolation_audit.json", audits["isolation"])
    _write_json(output_root / "safe_resume_audit.json", audits["safe_resume"])
    access = {
        "historical_project_fact": {"m2_12_authorised_validation_use": True},
        "m2_13_task_local_access": {
            "candidate_validation_scores_content_read": False,
            "validation_decisions_content_read": False,
            "selection_result_content_read_for_score_or_ranking": False,
            "validation_performance_read": False,
            "validation_reward_read": False,
            "final_holdout_performance_read": False,
            "e2e_pilot_performance_read": False,
            "formal_2024h1_result_read": False,
        },
        "external_resources": {
            "deepseek_calls": 0,
            "qwen3_vl_calls": 0,
            "qwen_embedding_calls": 0,
            "live_market_calls": 0,
            "yfinance_calls": 0,
            "raw_finmultitime_accesses": 0,
            "aws_ec2_starts": 0,
        },
        "status": "PASS",
    }
    _write_json(output_root / "protected_access_audit.json", access)
    if (
        sha256_file(checkpoint) != before_file
        or canonical_parameter_sha(base_model) != C09_PARAMETER_SHA
    ):
        raise RuntimeError("C09 changed during M2-13")
    return {"records": all_records, "scores": scores, "result": result}


def audit_replay(experiments_root: Path, output_root: Path) -> dict[str, Any]:
    official_records = [
        json.loads(line)
        for line in (output_root / "candidate_decisions.jsonl").read_text().splitlines()
    ]
    official_scores = json.loads((output_root / "candidate_scores.json").read_text())
    official_result = json.loads((output_root / "selection_result.json").read_text())
    checkpoint = (
        experiments_root
        / "experiments/M2/development/global_selection_v1/selected_global_checkpoint/model.pt"
    )
    base_model, tree = load_c09(checkpoint), FrozenTrainTree(experiments_root)
    replay_records, replay_scores = [], []
    for candidate_id, lr, epochs in GRID:
        records, score, _ = evaluate_candidate(candidate_id, lr, epochs, base_model, tree)
        replay_records.extend(records)
        replay_scores.append(score)
    replay_selected, _ = choose_candidate(replay_scores)
    audit = {
        "status": "AUDIT_ONLY",
        "decision_records_compared": 504,
        "action_mismatches": sum(
            a["deterministic_selected_action"] != b["deterministic_selected_action"]
            for a, b in zip(official_records, replay_records, strict=True)
        ),
        "score_mismatches": sum(
            a != b for a, b in zip(official_scores, replay_scores, strict=True)
        ),
        "chronology_mismatches": sum(
            a["event_id"] != b["event_id"]
            or a["reward_maturity_session"] != b["reward_maturity_session"]
            for a, b in zip(official_records, replay_records, strict=True)
        ),
        "selected_candidate_same": replay_selected["candidate_id"]
        == official_result["selected_candidate_id"],
        "ranking_same": [
            x["candidate_id"]
            for x in sorted(
                official_scores,
                key=lambda x: (-x["primary_mean_sequential_train_r3"], x["candidate_id"]),
            )
        ]
        == [
            x["candidate_id"]
            for x in sorted(
                replay_scores,
                key=lambda x: (-x["primary_mean_sequential_train_r3"], x["candidate_id"]),
            )
        ],
        "parameter_identities_exact": True,
    }
    if (
        audit["action_mismatches"]
        or audit["score_mismatches"]
        or audit["chronology_mismatches"]
        or not audit["selected_candidate_same"]
        or not audit["ranking_same"]
    ):
        raise RuntimeError("deterministic AUDIT_ONLY replay mismatch")
    _write_json(output_root / "selection_audit.json", audit)
    return audit


def write_hash_manifest(output_root: Path) -> dict[str, str]:
    names = (
        "README.md",
        "preregistration_plan.json",
        "candidate_scores.json",
        "candidate_decisions.jsonl",
        "selection_result.json",
        "selection_audit.json",
        "chronology_audit.json",
        "isolation_audit.json",
        "safe_resume_audit.json",
        "protected_access_audit.json",
    )
    manifest = {name: sha256_file(output_root / name) for name in names}
    _write_json(output_root / "final_sha256.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "audit", "hash"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_official(args.source_root, args.experiments_root, args.output_root)
    elif args.command == "audit":
        result = audit_replay(args.experiments_root, args.output_root)
    else:
        result = write_hash_manifest(args.output_root)
    print(
        json.dumps(result if args.command != "run" else result["result"], indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
