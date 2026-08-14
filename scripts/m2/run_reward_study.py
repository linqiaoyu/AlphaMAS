"""Run the deterministic, offline M2-04 empirical reward study.

Phase B reads only the committed market snapshot.  This module has no network
or market-provider dependency and never reads protected case outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.m2.reward_simulator import (
    REWARD_IDS,
    MarketBar,
    PortfolioState,
    RewardStatus,
    RewardWindow,
    candidate_rewards,
    maximum_drawdown,
    simulate_counterfactuals,
)
from tradingagents.backtesting.calendar import ExchangeSchedule

ACTION_SPREAD_EPSILON = 1e-6
BEST_ACTION_TIE_TOLERANCE = 1e-12
NOOP_NEUTRALITY_TOLERANCE = 1e-12
NONDEGENERACY_OVERALL_MIN = 0.25
NONDEGENERACY_VALIDATION_MIN = 0.10
R3_MIN_BEST_SET_CHANGES = 8
R3_MIN_VALIDATION_CHANGES = 1
R3_MAX_SINGLE_SYMBOL_CHANGE_SHARE = 0.50
R3_MAX_P99_SCALE_MULTIPLIER_VS_R2 = 2.0
SELECTION_RULE_VERSION = "M2_REWARD_SELECTION_HIERARCHY_V1"
EXPECTED_SYMBOLS = ("AAPL", "AEMD", "AGI", "AMZN", "ARR", "EML", "JBSS", "JPM")
ALLOWED_ROLES = frozenset({"TRAIN", "VALIDATION"})
ACTION_ORDER = ("BUY", "HOLD", "SELL")
REWARD_COLUMNS = dict(zip(REWARD_IDS, ("R1", "R2", "R3"), strict=True))


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return sha256(path)


def analysis_plan_payload() -> dict[str, Any]:
    return {
        "record_type": "M2_REWARD_ANALYSIS_PLAN_V1",
        "task_id": "M2-04",
        "constants": {
            "ACTION_SPREAD_EPSILON": ACTION_SPREAD_EPSILON,
            "BEST_ACTION_TIE_TOLERANCE": BEST_ACTION_TIE_TOLERANCE,
            "NOOP_NEUTRALITY_TOLERANCE": NOOP_NEUTRALITY_TOLERANCE,
            "NONDEGENERACY_OVERALL_MIN": NONDEGENERACY_OVERALL_MIN,
            "NONDEGENERACY_VALIDATION_MIN": NONDEGENERACY_VALIDATION_MIN,
            "R3_MIN_BEST_SET_CHANGES": R3_MIN_BEST_SET_CHANGES,
            "R3_MIN_VALIDATION_CHANGES": R3_MIN_VALIDATION_CHANGES,
            "R3_MAX_SINGLE_SYMBOL_CHANGE_SHARE": R3_MAX_SINGLE_SYMBOL_CHANGE_SHARE,
            "R3_MAX_P99_SCALE_MULTIPLIER_VS_R2": R3_MAX_P99_SCALE_MULTIPLIER_VS_R2,
        },
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_hierarchy": [
            "Require finite/correct rewards and R2/R3 no-op neutrality.",
            "Require overall spread fraction >= 0.25 and VALIDATION >= 0.10.",
            "R3 outranks R2 only when every pre-registered R3 added-value criterion passes.",
            "Otherwise select correctness-valid, non-degenerate R2 by simplicity/local-credit tie-break.",
            "R1 is fallback only when both R2 and R3 fail correctness or non-degeneracy.",
            "Block if no unique candidate is resolved.",
        ],
        "statistics": {"quantile_interpolation": "linear", "standard_deviation_ddof": 0},
        "protected_data": {"FINAL_HOLDOUT": False, "E2E_PILOT": False, "FORMAL_2024H1": False},
    }


def freeze_analysis_plan(path: Path) -> str:
    expected = canonical_json_bytes(analysis_plan_payload())
    if path.exists() and path.read_bytes() != expected:
        raise ValueError("analysis plan exists but differs from frozen constants")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)
    return sha256(path)


def _load_cases(case_plan: Path) -> list[dict[str, str]]:
    with case_plan.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split_role"] in ALLOWED_ROLES and row["maximum_included"].lower() == "true"]
    rows.sort(key=lambda row: (row["symbol"], row["decision_session"]))
    counts = Counter(row["split_role"] for row in rows)
    if len(rows) != 72 or counts != Counter({"TRAIN": 56, "VALIDATION": 16}):
        raise ValueError(f"invalid empirical case population: {counts}")
    if tuple(sorted({row["symbol"] for row in rows})) != EXPECTED_SYMBOLS:
        raise ValueError("invalid empirical symbol population")
    return rows


def _verify_snapshot(study_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = study_root / "manifests/market_snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["case_counts"] != {"TRAIN": 56, "VALIDATION": 16, "total": 72}:
        raise ValueError("snapshot case population differs from frozen contract")
    if manifest["latest_retained_session"] != "2023-07-14":
        raise ValueError("snapshot crosses frozen maturity boundary")
    if any(manifest[key] for key in ("holdout_rows_included", "pilot_rows_included", "formal_2024_rows_included")):
        raise ValueError("snapshot contains protected data")
    for record in manifest["files"]:
        path = study_root / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"snapshot mutation detected: {record['symbol']}")
    identity = manifest["reward_market_snapshot_identity_sha256"]
    return manifest, identity


def _market_frames(study_root: Path, manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for record in manifest["files"]:
        frame = pd.read_csv(study_root / record["path"])
        if list(frame.columns) != ["Date", "Open", "Close", "Dividends", "Stock Splits"]:
            raise ValueError(f"unexpected snapshot columns for {record['symbol']}")
        if frame["Date"].duplicated().any() or frame["Date"].max() > "2023-07-14":
            raise ValueError(f"invalid snapshot dates for {record['symbol']}")
        if not frame.iloc[:, 1:].map(math.isfinite).all().all():
            raise ValueError(f"non-finite snapshot value for {record['symbol']}")
        frames[record["symbol"]] = frame.set_index("Date")
    return frames


def _state(state_template: str, decision_close: float) -> PortfolioState:
    if state_template == "CASH":
        return PortfolioState(100000.0, 0.0, 0.0, 0.0)
    if state_template == "LONG":
        return PortfolioState(0.0, 100000.0 / decision_close, decision_close, 0.0)
    raise ValueError(f"unknown state template: {state_template}")


def _format_best_set(rewards: dict[str, float]) -> str:
    maximum = max(rewards.values())
    selected = [action for action in ACTION_ORDER if abs(rewards[action] - maximum) <= BEST_ACTION_TIE_TOLERANCE]
    return "{" + ",".join(selected) + "}"


def build_outcomes(cases: list[dict[str, str]], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    schedule = ExchangeSchedule()
    records: list[dict[str, Any]] = []
    for case in cases:
        frame = frames[case["symbol"]]
        decision_close = float(frame.loc[case["decision_session"], "Close"])
        required = schedule.calendar.sessions_window(case["decision_session"], 6)[1:].tz_localize(None)
        labels = tuple(item.date().isoformat() for item in required)
        bars = tuple(MarketBar(label, float(frame.loc[label, "Open"]), float(frame.loc[label, "Close"]), float(frame.loc[label, "Dividends"]), float(frame.loc[label, "Stock Splits"])) for label in labels)
        window = RewardWindow(case["symbol"], case["decision_session"], decision_close, bars)
        for template in ("CASH", "LONG"):
            result = simulate_counterfactuals(window, _state(template, decision_close), schedule=schedule)
            if result.status is not RewardStatus.MATURED or result.outcomes is None:
                raise ValueError(f"unexpected pending empirical outcome: {case['case_id']}")
            for action in ACTION_ORDER:
                outcome = result.outcomes[action]
                rewards = candidate_rewards(outcome)
                action_mdd = maximum_drawdown(outcome.action_equity_path)
                hold_mdd = maximum_drawdown(outcome.hold_equity_path)
                records.append({
                    "case_id": case["case_id"], "symbol": case["symbol"], "split_role": case["split_role"],
                    "decision_session": case["decision_session"], "execution_session": outcome.execution_session,
                    "maturity_session": outcome.maturity_session, "state_template": template, "action": action,
                    "decision_close_price": decision_close, "pre_execution_equity": outcome.pre_execution_equity,
                    "terminal_equity": outcome.terminal_equity, "hold_terminal_equity": outcome.hold_terminal_equity,
                    "turnover_notional": outcome.turnover_notional, "commission_cost": outcome.commission_cost,
                    "slippage_cost": outcome.slippage_cost, "total_cost": outcome.total_cost,
                    "action_was_noop": outcome.action_was_noop, "action_mdd": action_mdd, "hold_mdd": hold_mdd,
                    "r3_drawdown_penalty": rewards[REWARD_IDS[1]] - rewards[REWARD_IDS[2]],
                    "R1": rewards[REWARD_IDS[0]], "R2": rewards[REWARD_IDS[1]], "R3": rewards[REWARD_IDS[2]],
                })
    result = pd.DataFrame(records)
    if len(result) != 432 or result.groupby(["case_id", "state_template"]).size().ne(3).any():
        raise ValueError("empirical outcome cardinality is not 72 x 2 x 3")
    if not result[["R1", "R2", "R3"]].map(math.isfinite).all().all():
        raise ValueError("non-finite candidate reward")
    return result


def build_state_analysis(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["case_id", "symbol", "split_role", "decision_session", "state_template"]
    for values, group in outcomes.groupby(keys, sort=True):
        row = dict(zip(keys, values, strict=True))
        best_sets: dict[str, str] = {}
        for candidate in ("R1", "R2", "R3"):
            rewards = dict(zip(group["action"], group[candidate], strict=True))
            row[f"{candidate}_action_spread"] = max(rewards.values()) - min(rewards.values())
            best_sets[candidate] = _format_best_set(rewards)
            row[f"{candidate}_best_action_set"] = best_sets[candidate]
        row["R2_R3_best_set_changed"] = best_sets["R2"] != best_sets["R3"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 144:
        raise ValueError("empirical state cardinality is not 144")
    return frame


def _distribution(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    absolute = np.abs(array)
    median_abs = float(np.median(absolute))
    return {
        "count": len(array), "finite_rate": float(np.isfinite(array).mean()), "min": float(np.min(array)),
        "max": float(np.max(array)), "mean": float(np.mean(array)), "median": float(np.median(array)),
        "std": float(np.std(array, ddof=0)), "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)), "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)), "p99": float(np.quantile(array, 0.99)),
        "median_absolute_reward": median_abs,
        "extreme_to_median_ratio": None if median_abs <= 1e-18 else float(np.max(absolute) / median_abs),
        "p95_absolute_reward": float(np.quantile(absolute, 0.95)), "p99_absolute_reward": float(np.quantile(absolute, 0.99)),
    }


def candidate_summary(outcomes: pd.DataFrame, states: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for reward_id, candidate in REWARD_COLUMNS.items():
        distributions = {"OVERALL": _distribution(outcomes[candidate])}
        spread: dict[str, float] = {}
        for role in ("TRAIN", "VALIDATION"):
            distributions[role] = _distribution(outcomes.loc[outcomes["split_role"] == role, candidate])
        for role in ("OVERALL", "TRAIN", "VALIDATION"):
            subset = states if role == "OVERALL" else states[states["split_role"] == role]
            spread[role] = float((subset[f"{candidate}_action_spread"] > ACTION_SPREAD_EPSILON).mean())
        noop = outcomes[outcomes["action_was_noop"]]
        noop_abs = noop[candidate].abs()
        market_drift = {template: {"count": int(len(group)), "median_absolute_reward": float(group[candidate].abs().median()), "max_absolute_reward": float(group[candidate].abs().max())} for template, group in noop.groupby("state_template")}
        correctness = distributions["OVERALL"]["finite_rate"] == 1.0 and (candidate == "R1" or float(noop_abs.max()) <= NOOP_NEUTRALITY_TOLERANCE)
        nondegenerate = spread["OVERALL"] >= NONDEGENERACY_OVERALL_MIN and spread["VALIDATION"] >= NONDEGENERACY_VALIDATION_MIN
        summary[reward_id] = {
            "distributions": distributions,
            "non_trivial_spread_fraction": spread,
            "per_symbol_non_trivial_spread_fraction": {symbol: float((states.loc[states["symbol"] == symbol, f"{candidate}_action_spread"] > ACTION_SPREAD_EPSILON).mean()) for symbol in EXPECTED_SYMBOLS},
            "no_op": {"count": int(len(noop)), "fraction": float(len(noop) / len(outcomes)), "max_absolute_reward": float(noop_abs.max()), "median_absolute_reward": float(noop_abs.median())},
            "market_drift_on_noop": market_drift,
            "gate_1_correctness": correctness,
            "gate_2_semantic_suitability": candidate != "R1",
            "gate_3_non_degeneracy": nondegenerate,
            "gate_4_stability": distributions["OVERALL"]["finite_rate"] == 1.0,
        }
    changed = states[states["R2_R3_best_set_changed"]]
    symbol_counts = changed["symbol"].value_counts()
    largest_share = 0.0 if changed.empty else float(symbol_counts.max() / len(changed))
    r2_p99 = summary[REWARD_IDS[1]]["distributions"]["OVERALL"]["p99_absolute_reward"]
    r3_p99 = summary[REWARD_IDS[2]]["distributions"]["OVERALL"]["p99_absolute_reward"]
    scale_ratio = None if r2_p99 <= 1e-12 else r3_p99 / r2_p99
    penalties = outcomes.loc[outcomes["r3_drawdown_penalty"] > BEST_ACTION_TIE_TOLERANCE, "r3_drawdown_penalty"]
    r3_criteria = {
        "correctness": summary[REWARD_IDS[2]]["gate_1_correctness"],
        "non_degeneracy": summary[REWARD_IDS[2]]["gate_3_non_degeneracy"],
        "best_set_change_count": int(len(changed)),
        "best_set_change_minimum_met": len(changed) >= R3_MIN_BEST_SET_CHANGES,
        "validation_best_set_change_count": int((changed["split_role"] == "VALIDATION").sum()),
        "validation_minimum_met": int((changed["split_role"] == "VALIDATION").sum()) >= R3_MIN_VALIDATION_CHANGES,
        "largest_single_symbol_change_share": largest_share,
        "cross_asset_distribution_met": largest_share <= R3_MAX_SINGLE_SYMBOL_CHANGE_SHARE,
        "p99_absolute_scale_ratio_vs_R2": scale_ratio,
        "scale_stability_met": True if scale_ratio is None else scale_ratio <= R3_MAX_P99_SCALE_MULTIPLIER_VS_R2,
        "penalty_activation_count": int(len(penalties)),
        "penalty_activation_fraction": float(len(penalties) / len(outcomes)),
        "penalty_nonzero_median": None if penalties.empty else float(penalties.median()),
        "penalty_nonzero_mean": None if penalties.empty else float(penalties.mean()),
        "penalty_nonzero_p95": None if penalties.empty else float(penalties.quantile(0.95)),
        "penalty_activation_by_role": {role: float((outcomes.loc[outcomes["split_role"] == role, "r3_drawdown_penalty"] > BEST_ACTION_TIE_TOLERANCE).mean()) for role in ("TRAIN", "VALIDATION")},
        "penalty_activation_by_symbol": {symbol: float((outcomes.loc[outcomes["symbol"] == symbol, "r3_drawdown_penalty"] > BEST_ACTION_TIE_TOLERANCE).mean()) for symbol in EXPECTED_SYMBOLS},
        "best_set_changes_by_symbol": {symbol: int((changed["symbol"] == symbol).sum()) for symbol in EXPECTED_SYMBOLS},
    }
    r3_criteria["all_added_value_conditions_met"] = all((r3_criteria["correctness"], r3_criteria["non_degeneracy"], r3_criteria["best_set_change_minimum_met"], r3_criteria["validation_minimum_met"], r3_criteria["cross_asset_distribution_met"], r3_criteria["scale_stability_met"]))
    summary["R3_added_value"] = r3_criteria
    summary["transaction_costs"] = {
        "traded_actions": int((~outcomes["action_was_noop"]).sum()), "no_op_actions": int(outcomes["action_was_noop"].sum()),
        "commission": _distribution(outcomes["commission_cost"]), "slippage": _distribution(outcomes["slippage_cost"]),
        "total_cost": _distribution(outcomes["total_cost"]),
        "reward_by_switching": {status: {candidate: float(group[candidate].mean()) for candidate in ("R2", "R3")} for status, group in outcomes.groupby(outcomes["action_was_noop"].map({True: "NO_OP", False: "TRADED"}))},
    }
    return summary


def cross_symbol_summary(outcomes: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reward_id, candidate in REWARD_COLUMNS.items():
        for symbol in EXPECTED_SYMBOLS:
            values = outcomes.loc[outcomes["symbol"] == symbol, candidate]
            spread = states.loc[states["symbol"] == symbol, f"{candidate}_action_spread"]
            rows.append({"candidate_id": reward_id, "symbol": symbol, "outcome_count": len(values), "state_count": len(spread), "mean": values.mean(), "median": values.median(), "std": values.std(ddof=0), "p05": values.quantile(0.05), "p95": values.quantile(0.95), "median_absolute_reward": values.abs().median(), "non_trivial_spread_fraction": (spread > ACTION_SPREAD_EPSILON).mean()})
    return pd.DataFrame(rows)


def select_reward(summary: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    gates: dict[str, Any] = {}
    for reward_id in REWARD_IDS:
        item = summary[reward_id]
        gates[reward_id] = {
            "GATE_1_CORRECTNESS": "PASS" if item["gate_1_correctness"] else "FAIL",
            "GATE_2_SEMANTIC_SUITABILITY": "PASS" if item["gate_2_semantic_suitability"] else "BASELINE_FALLBACK_ONLY",
            "GATE_3_NON_DEGENERACY": "PASS" if item["gate_3_non_degeneracy"] else "FAIL",
            "GATE_4_STABILITY": "PASS" if item["gate_4_stability"] else "FAIL",
            "GATE_5_RISK_INCENTIVE_INTERPRETATION": "R3_ADDED_VALUE_PASS" if reward_id == REWARD_IDS[2] and summary["R3_added_value"]["all_added_value_conditions_met"] else ("LOCAL_CREDIT_PASS" if reward_id == REWARD_IDS[1] else "BASELINE_OR_NO_ADDED_VALUE"),
        }
    r2 = summary[REWARD_IDS[1]]
    r3 = summary[REWARD_IDS[2]]
    r1 = summary[REWARD_IDS[0]]
    if r2["gate_1_correctness"] and r2["gate_3_non_degeneracy"]:
        if summary["R3_added_value"]["all_added_value_conditions_met"]:
            return REWARD_IDS[2], gates, "R3 passed every pre-registered added-value criterion over correctness-valid, non-degenerate R2."
        return REWARD_IDS[1], gates, "R2 passed correctness and non-degeneracy; R3 did not pass every added-value criterion, so the frozen simplicity/local-credit tie-break selects R2."
    if (not r2["gate_1_correctness"] or not r2["gate_3_non_degeneracy"]) and (not r3["gate_1_correctness"] or not r3["gate_3_non_degeneracy"]) and r1["gate_1_correctness"] and r1["gate_3_non_degeneracy"]:
        return REWARD_IDS[0], gates, "Both HOLD-relative candidates failed correctness or non-degeneracy; R1 passed as the frozen fallback."
    raise ValueError("pre-registered reward-selection rule did not resolve a unique reward")


def run(case_plan: Path, study_root: Path, phase_a_commit: str) -> dict[str, Any]:
    plan_path = study_root / "manifests/analysis_plan.json"
    expected_plan = canonical_json_bytes(analysis_plan_payload())
    if not plan_path.exists() or plan_path.read_bytes() != expected_plan:
        raise ValueError("analysis plan must be separately frozen before candidate analysis")
    plan_sha = sha256(plan_path)
    manifest, identity = _verify_snapshot(study_root)
    snapshot_hashes_before = {record["symbol"]: record["sha256"] for record in manifest["files"]}
    cases = _load_cases(case_plan)
    outcomes = build_outcomes(cases, _market_frames(study_root, manifest))
    states = build_state_analysis(outcomes)
    summary = candidate_summary(outcomes, states)
    selected, gates, rationale = select_reward(summary)
    results = study_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(results / "reward_outcomes.csv", index=False, lineterminator="\n", float_format="%.17g")
    states.to_csv(results / "reward_state_analysis.csv", index=False, lineterminator="\n", float_format="%.17g")
    write_json(results / "reward_candidate_summary.json", summary)
    cross_symbol_summary(outcomes, states).to_csv(results / "reward_cross_symbol_summary.csv", index=False, lineterminator="\n", float_format="%.17g")
    selection = {
        "record_type": "M2_REWARD_SELECTION_V1", "selected_reward_id": selected, "selection_status": "SELECTED",
        "selection_rule_version": SELECTION_RULE_VERSION, "candidate_gate_results": gates, "selection_rationale": rationale,
        "market_snapshot_archive_commit": phase_a_commit, "market_snapshot_identity_sha256": identity,
        "analysis_plan_sha256": plan_sha, "reward_framework_sha": "06b30f05749c53c8122d3997595aad556cbf0136",
        "split_contract_sha": "8574067d039f6288042ec85cebc4517f3c3da49c",
        "Formal 2024H1 used": False, "Final Holdout used": False, "E2E Pilot used": False, "DeepSeek used": False,
        "B&H used for selection": False, "M1 performance used for selection": False,
    }
    write_json(results / "reward_selection.json", selection)
    after = {record["symbol"]: sha256(study_root / record["path"]) for record in manifest["files"]}
    if after != snapshot_hashes_before:
        raise ValueError("market snapshot mutated after Phase A")
    inventory_files = [plan_path, *sorted(results.iterdir())]
    write_json(study_root / "manifests/final_sha256.json", {"record_type": "M2_REWARD_STUDY_FINAL_SHA256_V1", "market_snapshot_mutated_after_phase_a": False, "files": {str(path.relative_to(study_root)): sha256(path) for path in inventory_files}})
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--phase-a-commit", required=True)
    parser.add_argument("--freeze-plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_path = args.study_root / "manifests/analysis_plan.json"
    if args.freeze_plan_only:
        print(freeze_analysis_plan(plan_path))
        return
    selection = run(args.case_plan, args.study_root, args.phase_a_commit)
    print(selection["selected_reward_id"])


if __name__ == "__main__":
    main()
