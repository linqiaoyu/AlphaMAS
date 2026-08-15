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
from tests.m2 import archive_paths
from tests.m2.archive_paths import (
    RewardStudyArchiveUnavailable,
    resolve_reward_study_root,
)

ROOT = Path(__file__).resolve().parents[2]
CASE_PLAN = ROOT / "docs/m2/m2_preformal_semantic_case_plan.csv"
PHASE_A_COMMIT = "5c71719e8cde6a78088a47024a263f58378b83e2"
RESULT_FILES = (
    "reward_outcomes.csv",
    "reward_state_analysis.csv",
    "reward_candidate_summary.json",
    "reward_cross_symbol_summary.csv",
    "reward_selection.json",
)


@pytest.fixture
def reward_study_root() -> Path:
    try:
        return resolve_reward_study_root(ROOT)
    except RewardStudyArchiveUnavailable as exc:
        pytest.skip(str(exc))


def _build_minimal_valid_archive(
    experiments_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    study_root = experiments_root / archive_paths.REWARD_STUDY_RELATIVE_PATH
    (study_root / "inputs/market_snapshot").mkdir(parents=True)
    (study_root / "manifests").mkdir()
    (study_root / "results").mkdir()
    (study_root / "manifests/market_snapshot_manifest.json").write_text(
        json.dumps(
            {
                "reward_market_snapshot_identity_sha256": (
                    archive_paths.EXPECTED_MARKET_SNAPSHOT_IDENTITY
                )
            }
        )
    )
    (study_root / "results/reward_selection.json").write_text(
        json.dumps({"selected_reward_id": archive_paths.EXPECTED_SELECTED_REWARD})
    )
    for relative_path in (
        "manifests/analysis_plan.json",
        "results/reward_outcomes.csv",
        "results/reward_state_analysis.csv",
        "results/reward_candidate_summary.json",
        "results/reward_cross_symbol_summary.csv",
    ):
        (study_root / relative_path).write_text(f"fixture:{relative_path}\n")
    fixture_hashes = {
        relative_path: hashlib.sha256((study_root / relative_path).read_bytes()).hexdigest()
        for relative_path in archive_paths.EXPECTED_FINAL_HASHES
    }
    monkeypatch.setattr(archive_paths, "EXPECTED_FINAL_HASHES", fixture_hashes)
    (study_root / "manifests/final_sha256.json").write_text(
        json.dumps({"files": fixture_hashes})
    )
    return study_root


def test_explicit_experiments_root_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_experiments_root = tmp_path / "configured-archive"
    expected = _build_minimal_valid_archive(fake_experiments_root, monkeypatch)
    monkeypatch.setenv("ALPHAMAS_EXPERIMENTS_ROOT", str(fake_experiments_root))
    unrelated_repository = tmp_path / "elsewhere/AlphaMAS"
    unrelated_repository.mkdir(parents=True)

    assert resolve_reward_study_root(unrelated_repository) == expected


def test_invalid_explicit_experiments_root_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "AlphaMAS"
    repository_root.mkdir()
    _build_minimal_valid_archive(tmp_path / "AlphaMAS-Experiments", monkeypatch)
    invalid_root = tmp_path / "does-not-exist"
    monkeypatch.setenv("ALPHAMAS_EXPERIMENTS_ROOT", str(invalid_root))

    with pytest.raises(ValueError, match="explicitly set.*invalid"):
        resolve_reward_study_root(repository_root)


def test_sibling_experiments_root_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "AlphaMAS"
    repository_root.mkdir()
    expected = _build_minimal_valid_archive(tmp_path / "AlphaMAS-Experiments", monkeypatch)
    monkeypatch.delenv("ALPHAMAS_EXPERIMENTS_ROOT", raising=False)

    assert resolve_reward_study_root(repository_root) == expected


def test_missing_archive_is_reported_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "AlphaMAS"
    repository_root.mkdir()
    monkeypatch.delenv("ALPHAMAS_EXPERIMENTS_ROOT", raising=False)

    with pytest.raises(RewardStudyArchiveUnavailable, match="archive unavailable"):
        resolve_reward_study_root(repository_root)


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


def test_archived_population_and_outcomes_are_exact(reward_study_root: Path) -> None:
    cases = _load_cases(CASE_PLAN)
    outcomes = pd.read_csv(reward_study_root / "results/reward_outcomes.csv")
    states = pd.read_csv(reward_study_root / "results/reward_state_analysis.csv")
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


def test_selection_is_reproduced_from_archived_summary(reward_study_root: Path) -> None:
    summary = json.loads(
        (reward_study_root / "results/reward_candidate_summary.json").read_text()
    )
    selected, gates, rationale = select_reward(summary)
    archived = json.loads((reward_study_root / "results/reward_selection.json").read_text())
    assert selected == REWARD_IDS[2] == archived["selected_reward_id"]
    assert gates == archived["candidate_gate_results"]
    assert rationale == archived["selection_rationale"]
    assert summary["R3_added_value"]["all_added_value_conditions_met"]


def test_r1_cannot_outrank_valid_relative_candidate(reward_study_root: Path) -> None:
    summary = json.loads(
        (reward_study_root / "results/reward_candidate_summary.json").read_text()
    )
    summary["R3_added_value"]["all_added_value_conditions_met"] = False
    selected, _, _ = select_reward(summary)
    assert selected == REWARD_IDS[1]


def test_offline_reproduction_is_byte_identical(
    tmp_path: Path, reward_study_root: Path
) -> None:
    copied = tmp_path / "study"
    (copied / "inputs").mkdir(parents=True)
    shutil.copytree(
        reward_study_root / "inputs/market_snapshot", copied / "inputs/market_snapshot"
    )
    (copied / "manifests").mkdir()
    for name in ("market_snapshot_manifest.json", "analysis_plan.json"):
        shutil.copy2(reward_study_root / "manifests" / name, copied / "manifests" / name)
    run(CASE_PLAN, copied, PHASE_A_COMMIT)
    for name in RESULT_FILES:
        expected = hashlib.sha256(
            (reward_study_root / "results" / name).read_bytes()
        ).hexdigest()
        actual = hashlib.sha256((copied / "results" / name).read_bytes()).hexdigest()
        assert actual == expected, name


def test_runner_has_no_market_provider_or_llm_path(reward_study_root: Path) -> None:
    source = (ROOT / "scripts/m2/run_reward_study.py").read_text()
    assert "yfinance" not in source
    assert "YFinanceDataProvider" not in source
    assert "requests." not in source
    assert "boto3" not in source
    selection = json.loads(
        (reward_study_root / "results/reward_selection.json").read_text()
    )
    assert selection["Formal 2024H1 used"] is False
    assert selection["Final Holdout used"] is False
    assert selection["E2E Pilot used"] is False
