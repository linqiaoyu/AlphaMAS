from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.m2.run_semantic_handoff_corpus import (
    FROZEN_RUNNER_FILE_SHA256,
    FROZEN_RUNNER_SHA,
    REUSED_CASE_IDS,
    SELECTED_TIER,
    SELECTED_TIER_IDENTITY,
    _atomic_case_write,
    build_development_actor_state,
    canonical_membership,
    case_directory,
    enforce_budget,
    initialise_staging,
    no_cache_preflight,
    prepare_plan,
    resume_case_status,
    validate_development_actor,
)
from scripts.m2.run_semantic_handoff_probe import (
    _actor_state,
    canonical_json,
    sha256_bytes,
    sha256_file,
)

SOURCE = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-m2")
EXPERIMENTS = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-Experiments")
CORPUS = EXPERIMENTS / "experiments/M2/development/preformal_evidence_v1"
CALIBRATION = EXPERIMENTS / "experiments/M2/development/semantic_cost_calibration_v1"
ARCHIVE = EXPERIMENTS / "experiments/M2/development/semantic_handoff_trainval_v1"
FREEZE = SOURCE / "docs/m2/m2_semantic_handoff_trainval_freeze.json"


def synthetic_inputs(role: str = "TRAIN"):
    case = {
        "case_id": "AAPL:2023-05-26",
        "symbol": "AAPL",
        "decision_session": "2023-05-26",
        "role": role,
        "evidence_packet_sha256": "a" * 64,
    }
    packet = {"decision_time": "2023-05-26T16:00:00-04:00"}
    state = {
        "investment_plan": "**Recommendation**: Hold",
        "trader_investment_plan": (
            "**Action**: Hold\n\n**Reasoning**: Balanced.\n\nFINAL TRANSACTION PROPOSAL: **HOLD**"
        ),
        "company_of_interest": "AAPL",
        "instrument_context": "The instrument to analyze is `AAPL`.",
        "run_mode": "historical",
        "historical_as_of": "2023-05-26T16:00:00-04:00",
    }
    return case, packet, state


@pytest.fixture
def prepared_plan(tmp_path: Path) -> Path:
    root = tmp_path / "plan"
    prepare_plan(
        CORPUS,
        CALIBRATION,
        root,
        retrieved_at="2026-08-15T00:00:00Z",
        cache_hit="0.02",
        cache_miss="1.00",
        output_price="2.00",
    )
    return root


def test_exact_maximum_development_and_deferred_membership():
    development, deferred = canonical_membership(CORPUS)
    assert len(development) == 72
    assert sum(row["role"] == "TRAIN" for row in development) == 56
    assert sum(row["role"] == "VALIDATION" for row in development) == 16
    assert {
        row["case_id"] for row in development if row["generation_mode"] == "REUSE_M2_07"
    } == set(REUSED_CASE_IDS)
    assert (
        sum(
            row["role"] == "TRAIN" and row["generation_mode"] == "GENERATE_M2_08"
            for row in development
        )
        == 50
    )
    assert (
        sum(
            row["role"] == "VALIDATION" and row["generation_mode"] == "GENERATE_M2_08"
            for row in development
        )
        == 16
    )
    assert sum(row["role"] == "FINAL_HOLDOUT" for row in deferred) == 16
    assert sum(row["role"] == "E2E_PILOT" for row in deferred) == 8
    assert SELECTED_TIER == "MAXIMUM"
    assert SELECTED_TIER_IDENTITY == (
        "68fbc780cbb454f78d0ae719f91665737ed6e04928eba91f4c9c0530745d601f"
    )


def test_train_generalisation_is_byte_identical_to_m2_07_builder():
    case, packet, state = synthetic_inputs("TRAIN")
    expected = _actor_state(
        {**case, "packet_sha256": case["evidence_packet_sha256"]},
        packet,
        state,
        runner_sha="b" * 40,
        probe_inputs_sha="c" * 40,
    )
    actual = build_development_actor_state(
        case,
        packet,
        state,
        runner_sha="b" * 40,
        plan_sha="c" * 40,
    )
    assert canonical_json(actual) == canonical_json(expected)


def test_validation_differs_only_in_role_metadata():
    train_case, packet, state = synthetic_inputs("TRAIN")
    validation_case = {**train_case, "role": "VALIDATION"}
    train = build_development_actor_state(
        train_case,
        packet,
        state,
        runner_sha="b" * 40,
        plan_sha="c" * 40,
    )
    validation = build_development_actor_state(
        validation_case,
        packet,
        state,
        runner_sha="b" * 40,
        plan_sha="c" * 40,
    )
    assert train["role"] == "TRAIN"
    assert validation["role"] == "VALIDATION"
    train_without_role = {key: value for key, value in train.items() if key != "role"}
    validation_without_role = {key: value for key, value in validation.items() if key != "role"}
    assert canonical_json(train_without_role) == canonical_json(validation_without_role)


@pytest.mark.parametrize(
    "field",
    [
        "reward",
        "future_return",
        "future_price",
        "target",
        "label",
        "ground_truth",
        "best_action",
        "portfolio_outcome",
    ],
)
def test_actor_visible_outcome_metadata_rejected(field: str):
    case, packet, state = synthetic_inputs()
    actor = build_development_actor_state(
        case,
        packet,
        state,
        runner_sha="b" * 40,
        plan_sha="c" * 40,
    )
    actor[field] = "leak"
    with pytest.raises(ValueError):
        validate_development_actor(actor, case)


def test_plan_freezes_only_three_prepaid_files(prepared_plan: Path):
    assert {path.name for path in prepared_plan.iterdir()} == {
        "README.md",
        "generation_plan.json",
        "pricing_snapshot.json",
    }


def test_reused_states_copy_byte_identically_and_no_protected_states(
    tmp_path: Path, prepared_plan: Path
):
    staging = tmp_path / "staging"
    status = initialise_staging(staging, prepared_plan, CALIBRATION)
    assert status["reused_valid"] == 6
    assert status["new_completed"] == 0
    assert status["new_remaining"] == 66
    development, deferred = canonical_membership(CORPUS)
    cases = {row["case_id"]: row for row in development}
    for case_id in REUSED_CASE_IDS:
        source = CALIBRATION / "cases" / case_id.replace(":", "_")
        target = case_directory(staging, cases[case_id])
        assert {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()} == {
            path.name: sha256_file(path) for path in target.iterdir() if path.is_file()
        }
    assert not any(case_directory(staging, row).exists() for row in deferred)


def test_safe_resume_skips_complete_and_blocks_complete_corruption(
    tmp_path: Path, prepared_plan: Path
):
    staging = tmp_path / "staging"
    initialise_staging(staging, prepared_plan, CALIBRATION)
    development, _ = canonical_membership(CORPUS)
    case = next(row for row in development if row["case_id"] == REUSED_CASE_IDS[0])
    target = case_directory(staging, case)
    assert (
        resume_case_status(
            target,
            case,
            reused=True,
            runner_sha=FROZEN_RUNNER_SHA,
            plan_sha="unused",
        )
        == "SKIP"
    )
    actor_path = target / "actor_visible_state.json"
    actor_path.write_bytes(actor_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA manifest differs"):
        resume_case_status(
            target,
            case,
            reused=True,
            runner_sha=FROZEN_RUNNER_SHA,
            plan_sha="unused",
        )


def test_incomplete_case_is_retryable(tmp_path: Path):
    case, _packet, _state = synthetic_inputs()
    target = case_directory(tmp_path, case)
    target.mkdir(parents=True)
    (target / "usage.json").write_text("{}\n", encoding="utf-8")
    assert (
        resume_case_status(
            target,
            case,
            reused=False,
            runner_sha="b" * 40,
            plan_sha="c" * 40,
        )
        == "RETRY"
    )


def test_atomic_case_write_creates_all_four_files(tmp_path: Path):
    case, packet, state = synthetic_inputs()
    actor = build_development_actor_state(
        case,
        packet,
        state,
        runner_sha="b" * 40,
        plan_sha="c" * 40,
    )
    trace = {
        "actor_visible": False,
        "case_id": case["case_id"],
        "capture_boundary": {
            "stop_after": "Prompt Trader",
            "risk_debate_executed": False,
            "portfolio_manager_executed": False,
            "trade_executed": False,
        },
    }
    usage = {
        "case_cost_cny": "0",
        "total_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    _atomic_case_write(tmp_path, case, actor, trace, usage)
    assert {path.name for path in case_directory(tmp_path, case).iterdir()} == {
        "actor_visible_state.json",
        "upstream_trace.json",
        "usage.json",
        "sha256.json",
    }


def test_no_cache_preflight_and_cost_ceiling():
    prices = {
        "input_cache_hit_price": "0.02",
        "input_cache_miss_price": "1.00",
        "output_price": "2.00",
    }
    preflight = no_cache_preflight(CALIBRATION, prices)
    assert Decimal(preflight["no_cache_guard_cny"]) > 0
    assert Decimal(preflight["m2_08_no_cache_projection_cny"]) < Decimal("18")
    with pytest.raises(RuntimeError, match="reached CNY 18"):
        enforce_budget(Decimal("18"), [], 0, Decimal("0.1"))
    with pytest.raises(RuntimeError, match="forecast exceeds CNY 18"):
        enforce_budget(Decimal("0"), [], 0, Decimal("0.3"))


def test_m2_07_runner_file_is_unchanged():
    assert (
        sha256_file(SOURCE / "scripts/m2/run_semantic_handoff_probe.py")
        == FROZEN_RUNNER_FILE_SHA256
    )


def test_canonical_serialisation_is_stable():
    assert sha256_bytes(canonical_json({"b": 1, "a": 2})) == sha256_bytes(
        canonical_json({"a": 2, "b": 1})
    )


def test_committed_corpus_archive_is_exact_and_protected_roles_are_absent():
    manifest = json.loads(
        (ARCHIVE / "manifests/corpus_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["materialised_case_count"] == 72
    assert manifest["train"] == 56
    assert manifest["validation"] == 16
    assert manifest["m2_07_reused"] == 6
    assert manifest["m2_08_generated"] == 66
    assert manifest["final_holdout"] == {"deferred": 16, "materialised": 0}
    assert manifest["e2e_pilot"] == {"deferred": 8, "materialised": 0}
    assert not (ARCHIVE / "cases/final_holdout").exists()
    assert not (ARCHIVE / "cases/e2e_pilot").exists()


def test_freeze_document_binds_committed_corpus_and_costs():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    corpus = json.loads(
        (ARCHIVE / "manifests/corpus_manifest.json").read_text(encoding="utf-8")
    )
    costs = json.loads(
        (ARCHIVE / "manifests/cost_manifest.json").read_text(encoding="utf-8")
    )
    assert freeze["semantic_handoff_trainval_corpus_identity_sha256"] == corpus[
        "semantic_handoff_trainval_corpus_identity_sha256"
    ]
    assert freeze["m2_08_actual_cost_cny"] == costs["m2_08_actual_cost_cny"]
    assert freeze["deepseek_requests"] == costs["deepseek_requests"]
    assert freeze["input_tokens"] == costs["input_tokens"]
    assert freeze["output_tokens"] == costs["output_tokens"]
    assert freeze["reward_used"] is False
    assert freeze["performance_used"] is False
