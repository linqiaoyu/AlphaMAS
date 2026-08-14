from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from scripts.m2.reward_simulator import REWARD_IDS
from scripts.m2.run_reward_study import (
    ACTION_SPREAD_EPSILON,
    BEST_ACTION_TIE_TOLERANCE,
    NOOP_NEUTRALITY_TOLERANCE,
    _format_best_set,
    _load_cases,
    analysis_plan_payload,
    freeze_analysis_plan,
    run,
    select_reward,
)

ROOT = Path(__file__).resolve().parents[2]
CASE_PLAN = ROOT / "docs/m2/m2_preformal_semantic_case_plan.csv"
STUDY = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-Experiments/experiments/M2/development/reward_study_v1")
PHASE_A_COMMIT = "5c71719e8cde6a78088a47024a263f58378b83e2"
RESULT_FILES = (
    "reward_outcomes.csv",
    "reward_state_analysis.csv",
    "reward_candidate_summary.json",
    "reward_cross_symbol_summary.csv",
    "reward_selection.json",
)


def test_analysis_plan_constants_are_frozen(tmp_path: Path) -> None:
    plan = analysis_plan_payload()
    assert plan["constants"] == {
        "ACTION_SPREAD_EPSILON": ACTION_SPREAD_EPSILON,
        "BEST_ACTION_TIE_TOLERANCE": BEST_ACTION_TIE_TOLERANCE,
        "NOOP_NEUTRALITY_TOLERANCE": NOOP_NEUTRALITY_TOLERANCE,
        "NONDEGENERACY_OVERALL_MIN": 0.25,
        "NONDEGENERACY_VALIDATION_MIN": 0.10,
        "R3_MIN_BEST_SET_CHANGES": 8,
        "R3_MIN_VALIDATION_CHANGES": 1,
        "R3_MAX_SINGLE_SYMBOL_CHANGE_SHARE": 0.50,
        "R3_MAX_P99_SCALE_MULTIPLIER_VS_R2": 2.0,
    }
    path = tmp_path / "analysis_plan.json"
    first = freeze_analysis_plan(path)
    assert freeze_analysis_plan(path) == first
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="differs"):
        freeze_analysis_plan(path)


def test_best_action_sets_preserve_economic_ties() -> None:
    assert _format_best_set({"BUY": 1.0, "HOLD": 1.0, "SELL": 0.0}) == "{BUY,HOLD}"
    assert _format_best_set({"BUY": 0.0, "HOLD": 1.0, "SELL": 1.0}) == "{HOLD,SELL}"
    assert _format_best_set({"BUY": 1.0, "HOLD": 1.0 - 0.5e-12, "SELL": 0.0}) == "{BUY,HOLD}"


def test_archived_population_and_outcomes_are_exact() -> None:
    cases = _load_cases(CASE_PLAN)
    outcomes = pd.read_csv(STUDY / "results/reward_outcomes.csv")
    states = pd.read_csv(STUDY / "results/reward_state_analysis.csv")
    assert len(cases) == 72 and len(states) == 144 and len(outcomes) == 432
    assert set(outcomes["case_id"]) == {row["case_id"] for row in cases}
    assert set(outcomes["split_role"]) == {"TRAIN", "VALIDATION"}
    assert not outcomes["decision_session"].str.startswith("2024").any()
    assert outcomes[["R1", "R2", "R3"]].notna().all().all()
    assert outcomes[["R1", "R2", "R3"]].map(lambda value: abs(value) < float("inf")).all().all()
    noops = outcomes[outcomes["action_was_noop"]]
    assert noops["R2"].abs().max() <= NOOP_NEUTRALITY_TOLERANCE
    assert noops["R3"].abs().max() <= NOOP_NEUTRALITY_TOLERANCE
    assert (outcomes["r3_drawdown_penalty"] >= -BEST_ACTION_TIE_TOLERANCE).all()
    assert ((states[["R1_action_spread", "R2_action_spread", "R3_action_spread"]] > ACTION_SPREAD_EPSILON).mean() == 1.0).all()


def test_selection_is_reproduced_from_archived_summary() -> None:
    summary = json.loads((STUDY / "results/reward_candidate_summary.json").read_text())
    selected, gates, rationale = select_reward(summary)
    archived = json.loads((STUDY / "results/reward_selection.json").read_text())
    assert selected == REWARD_IDS[2] == archived["selected_reward_id"]
    assert gates == archived["candidate_gate_results"]
    assert rationale == archived["selection_rationale"]
    assert summary["R3_added_value"]["all_added_value_conditions_met"]


def test_r1_cannot_outrank_valid_relative_candidate() -> None:
    summary = json.loads((STUDY / "results/reward_candidate_summary.json").read_text())
    summary["R3_added_value"]["all_added_value_conditions_met"] = False
    selected, _, _ = select_reward(summary)
    assert selected == REWARD_IDS[1]


def test_offline_reproduction_is_byte_identical(tmp_path: Path) -> None:
    copied = tmp_path / "study"
    (copied / "inputs").mkdir(parents=True)
    shutil.copytree(STUDY / "inputs/market_snapshot", copied / "inputs/market_snapshot")
    (copied / "manifests").mkdir()
    for name in ("market_snapshot_manifest.json", "analysis_plan.json"):
        shutil.copy2(STUDY / "manifests" / name, copied / "manifests" / name)
    run(CASE_PLAN, copied, PHASE_A_COMMIT)
    for name in RESULT_FILES:
        expected = hashlib.sha256((STUDY / "results" / name).read_bytes()).hexdigest()
        actual = hashlib.sha256((copied / "results" / name).read_bytes()).hexdigest()
        assert actual == expected, name


def test_runner_has_no_market_provider_or_llm_path() -> None:
    source = (ROOT / "scripts/m2/run_reward_study.py").read_text()
    assert "yfinance" not in source
    assert "YFinanceDataProvider" not in source
    assert "requests." not in source
    assert "boto3" not in source
    selection = json.loads((STUDY / "results/reward_selection.json").read_text())
    assert selection["Formal 2024H1 used"] is False
    assert selection["Final Holdout used"] is False
    assert selection["E2E Pilot used"] is False
