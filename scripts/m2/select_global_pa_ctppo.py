"""Frozen M2-12 VALIDATION-only global checkpoint selector.

Phase A may run ``--verify-only``.  Real VALIDATION rewards are computed only by
``--score`` after the source selector and Experiments selection plan are pushed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.m2.pa_ctppo import (
    METHOD_ID,
    TIE_TOLERANCE as ACTION_TIE_TOLERANCE,
    PromptAnchoredActorCritic,
    deterministic_action,
)
from scripts.m2.rl_environment import (
    INITIAL_CASH,
    SequentialPortfolioState,
    initial_snapshot,
    load_market_snapshot,
    reward_window,
    simulate_local_credit,
    simulate_weekly_transition,
    weekly_transition_window,
)
from scripts.m2.semantic_state_representation import ACTION_ORDER, SemanticBaseStore
from scripts.m2.train_global_pa_ctppo import (
    actor_parameter_sha,
    canonical_parameter_sha,
    critic_parameter_sha,
    sha256_file,
)

TASK_ID = "M2-12"
SCHEMA_VERSION = "M2-12-GLOBAL-VALIDATION-SELECTION-v1"
CANONICAL_SOURCE_ROOT = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-m2")
CANONICAL_EXPERIMENTS_ROOT = Path(
    "/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-Experiments"
)
STARTING_SOURCE_SHA = "178746e23ff8cf0195529fc92adf5f7b533d24d3"
STARTING_EXPERIMENTS_SHA = "f9b5accc874dca89b72b405a9d8278d1facb38d6"
TRAINER_SHA = STARTING_SOURCE_SHA
TRAINING_PLAN_SHA = "00184248dbb0d51c06c27514bfb2abf4e913c345"
REPRESENTATION_IDENTITY = "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe"
TREE_IDENTITY = "ff5e1bf21ef90b9eeec96d257c59e83656957f32ec4d2f94396a0a5d4bf1db13"
REWARD_IDENTITY = "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"
MARKET_IDENTITY = "3afc723888666e0ca3a12219a57dc577d6cf3953717da731b774634f4aff1445"
INITIAL_PARAMETER_SHA = "60a0fec7b69ef2d0576a9c0894be09c377d573585db827c162279fb27483303e"
SELECTION_TIE_TOLERANCE = 1e-8
VALIDATION_SYMBOLS = ("AAPL", "AMZN", "JPM", "JBSS", "EML", "AGI", "ARR", "AEMD")
VALIDATION_SESSIONS = ("2023-06-30", "2023-07-07")
CANDIDATE_GRID = {
    "C01": (1e-4, 25, "9f2864a2ddc00937c55e777537762bee0c01b1a753cb2d81fe07311ba858ed89"),
    "C02": (1e-4, 50, "abb0813d118de86cb69202d32b74dc4e63e19b5397db48357f63c38288463a06"),
    "C03": (1e-4, 100, "57074b8b41ca79acc14f2bfb92c55da556c761cba868f54d946bce038e9eb16f"),
    "C04": (3e-4, 25, "9d093f23bf1baa90288e2d2e1526a946d0da1af45db5771bb2801afdabf32906"),
    "C05": (3e-4, 50, "3d05c96b2a25be08b44f01ee54d0098c2b6ffa2525df4440ce619a3cec044432"),
    "C06": (3e-4, 100, "3fc64d23a1633a599144e11219594bb347f708c866219464a0800e2d3547493e"),
    "C07": (1e-3, 25, "6daf39f0864306691377c971871b6ca560f0fdef21c1adf63fdc3a9055a71444"),
    "C08": (1e-3, 50, "27ab73e5abee5810553ac84b33d7aea4fdd56e8ac5963e026eef966f2c5c28c1"),
    "C09": (1e-3, 100, "6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841"),
}
PROTECTED_TOKENS = ("final_holdout", "e2e_pilot", "formal_2024", "2024h1")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def payload_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *args), check=True, text=True, capture_output=True
    ).stdout.strip()


def assert_unprotected_path(path: Path) -> Path:
    lowered = str(path.resolve()).lower()
    if any(token in lowered for token in PROTECTED_TOKENS):
        raise RuntimeError(f"protected evaluation path rejected: {path}")
    return path


def verify_repository(root: Path, branch: str, required_ancestor: str) -> str:
    root = root.resolve()
    if root not in (CANONICAL_SOURCE_ROOT, CANONICAL_EXPERIMENTS_ROOT):
        raise RuntimeError(f"non-canonical repository rejected: {root}")
    if _git(root, "branch", "--show-current") != branch:
        raise RuntimeError(f"wrong branch for {root}")
    if _git(root, "status", "--porcelain=v1"):
        raise RuntimeError(f"dirty worktree rejected: {root}")
    head = _git(root, "rev-parse", "HEAD")
    if subprocess.run(
        ("git", "-C", str(root), "merge-base", "--is-ancestor", required_ancestor, head)
    ).returncode:
        raise RuntimeError(f"required frozen lineage missing from {root}")
    return head


def load_checkpoint(path: Path) -> tuple[PromptAnchoredActorCritic, dict[str, Any]]:
    payload = torch.load(assert_unprotected_path(path), map_location="cpu", weights_only=True)
    if set(payload) != {"model_state", "metadata"}:
        raise RuntimeError("unexpected checkpoint payload")
    model = PromptAnchoredActorCritic()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(payload["metadata"])


def verify_candidates(experiments_root: Path) -> list[dict[str, Any]]:
    candidate_root = experiments_root / "experiments/M2/development/global_training_v1/candidates"
    directories = sorted(path.name for path in candidate_root.iterdir() if path.is_dir())
    if directories != list(CANDIDATE_GRID):
        raise RuntimeError(f"candidate population mismatch: {directories}")
    verified = []
    required = {
        "method_identity": METHOD_ID,
        "tree_identity": TREE_IDENTITY,
        "representation_identity": REPRESENTATION_IDENTITY,
        "reward_identity": REWARD_IDENTITY,
        "initial_parameter_sha": INITIAL_PARAMETER_SHA,
        "trainer_sha": TRAINER_SHA,
        "training_plan_sha": TRAINING_PLAN_SHA,
        "run_kind": "canonical",
        "selection_eligible": True,
        "replay_status": None,
    }
    for candidate_id, (learning_rate, iteration, expected_sha) in CANDIDATE_GRID.items():
        directory = candidate_root / candidate_id
        manifest = json.loads((directory / "manifest.json").read_text())
        model, metadata = load_checkpoint(directory / "model.pt")
        if manifest != metadata:
            raise RuntimeError(f"{candidate_id} manifest/checkpoint metadata mismatch")
        expected = {
            **required,
            "candidate_id": candidate_id,
            "learning_rate": learning_rate,
            "outer_iteration": iteration,
            "candidate_parameter_sha": expected_sha,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"{candidate_id} invalid {key}: {manifest.get(key)!r}")
        actual_sha = canonical_parameter_sha(model)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{candidate_id} parameter SHA mismatch")
        if actor_parameter_sha(model) != manifest["actor_parameter_sha"]:
            raise RuntimeError(f"{candidate_id} Actor SHA mismatch")
        if critic_parameter_sha(model) != manifest["critic_parameter_sha"]:
            raise RuntimeError(f"{candidate_id} Critic SHA mismatch")
        verified.append({**manifest, "model_file_sha256": sha256_file(directory / "model.pt")})
    return verified


def load_validation_cases(experiments_root: Path) -> tuple[list[dict[str, Any]], str]:
    base = experiments_root / "experiments/M2/development/semantic_state_representation_v1"
    identity = json.loads((base / "representation_identity.json").read_text())
    if identity.get("semantic_state_representation_identity_sha256") != REPRESENTATION_IDENTITY:
        raise RuntimeError("semantic representation identity mismatch")
    rows = json.loads((base / "manifests/row_index.json").read_text())
    semantic_manifest = json.loads((base / "manifests/semantic_state_manifest.json").read_text())
    semantic_by_case = {row["case_id"]: row for row in semantic_manifest}
    cases = []
    for row in rows:
        if row["role"] != "VALIDATION":
            continue
        semantic = semantic_by_case[row["case_id"]]
        cases.append(
            {
                **row,
                "prompt_action": semantic["prompt_action"],
                "semantic_base_row_sha256": semantic["semantic_base_row_sha256"],
            }
        )
    expected = [
        (symbol, session) for symbol in VALIDATION_SYMBOLS for session in VALIDATION_SESSIONS
    ]
    actual = [(case["symbol"], case["decision_session"]) for case in cases]
    if actual != expected or len(cases) != 16:
        raise RuntimeError(f"VALIDATION population mismatch: {actual}")
    return cases, payload_sha(
        [{"symbol": symbol, "decision_session": session} for symbol, session in expected]
    )


def verify_market(experiments_root: Path) -> dict[str, Any]:
    path = (
        experiments_root
        / "experiments/M2/development/reward_study_v1/manifests/market_snapshot_manifest.json"
    )
    manifest = json.loads(assert_unprotected_path(path).read_text())
    if manifest.get("reward_market_snapshot_identity_sha256") != MARKET_IDENTITY:
        raise RuntimeError("market snapshot identity mismatch")
    if manifest.get("case_counts", {}).get("VALIDATION") != 16:
        raise RuntimeError("market VALIDATION population mismatch")
    for key in ("holdout_rows_included", "pilot_rows_included", "formal_2024_rows_included"):
        if manifest.get(key) is not False:
            raise RuntimeError(f"protected market rows present: {key}")
    cash = manifest.get("state_templates", {}).get("CASH", {})
    if cash != {
        "average_entry_price": 0.0,
        "cash": INITIAL_CASH,
        "open_position_commission": 0.0,
        "quantity": 0.0,
    }:
        raise RuntimeError("initial validation portfolio contract mismatch")
    return manifest


def choose_candidate(scores: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    remaining = [dict(score) for score in scores]
    if {score["candidate_id"] for score in remaining} != set(CANDIDATE_GRID):
        raise ValueError("selection requires exactly C01-C09")
    if not all(score.get("correctness_valid") is True for score in remaining):
        raise ValueError("all candidates must be correctness-valid")
    path: list[dict[str, Any]] = []

    best_primary = max(score["primary_mean_sequential_local_r3"] for score in remaining)
    remaining = [
        score
        for score in remaining
        if best_primary - score["primary_mean_sequential_local_r3"] <= SELECTION_TIE_TOLERANCE
    ]
    primary_tie = len(remaining) > 1
    path.append(
        {
            "criterion": "primary_mean_sequential_local_r3",
            "remaining": [x["candidate_id"] for x in remaining],
        }
    )
    invoked = {
        "worst_symbol": False,
        "override_rate": False,
        "learning_rate": False,
        "checkpoint_iteration": False,
    }
    criteria = (
        ("worst_symbol", "worst_symbol_cumulative_r3", max),
        ("override_rate", "override_rate", min),
        ("learning_rate", "learning_rate", min),
        ("checkpoint_iteration", "checkpoint_iteration", min),
    )
    for label, field, reducer in criteria:
        if len(remaining) == 1:
            break
        invoked[label] = True
        best = reducer(score[field] for score in remaining)
        if field == "worst_symbol_cumulative_r3":
            remaining = [
                score for score in remaining if abs(score[field] - best) <= SELECTION_TIE_TOLERANCE
            ]
        else:
            remaining = [score for score in remaining if score[field] == best]
        path.append({"criterion": field, "remaining": [x["candidate_id"] for x in remaining]})
    if len(remaining) != 1:
        raise RuntimeError("frozen tie-breaks did not select exactly one candidate")
    return remaining[0], {
        "primary_metric_tie": primary_tie,
        "tie_breaks_invoked": invoked,
        "path": path,
    }


def _state_hash(state: SequentialPortfolioState) -> str:
    return payload_sha(asdict(state))


def evaluate_candidate(
    candidate: Mapping[str, Any], cases: list[dict[str, Any]], experiments_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_dir = (
        experiments_root
        / f"experiments/M2/development/global_training_v1/candidates/{candidate['candidate_id']}"
    )
    before_file = sha256_file(candidate_dir / "model.pt")
    model, _ = load_checkpoint(candidate_dir / "model.pt")
    before_parameters = canonical_parameter_sha(model)
    representation_root = (
        experiments_root / "experiments/M2/development/semantic_state_representation_v1"
    )
    store = SemanticBaseStore(representation_root / "embeddings/semantic_base.npy")
    market_root = (
        experiments_root / "experiments/M2/development/reward_study_v1/inputs/market_snapshot"
    )
    records: list[dict[str, Any]] = []
    rewards_by_symbol: dict[str, list[float]] = defaultdict(list)
    overrides = 0
    by_symbol = {
        symbol: [case for case in cases if case["symbol"] == symbol]
        for symbol in VALIDATION_SYMBOLS
    }
    with torch.inference_mode():
        for symbol in VALIDATION_SYMBOLS:
            market = load_market_snapshot(
                assert_unprotected_path(market_root / f"{symbol}.csv"), symbol
            )
            state = SequentialPortfolioState()
            snapshot = initial_snapshot(
                symbol, VALIDATION_SESSIONS[0], market[VALIDATION_SESSIONS[0]].close_price
            )
            for index, case in enumerate(by_symbol[symbol]):
                semantic = store.semantic_base(case["row_index"])
                if (
                    hashlib.sha256(semantic.tobytes(order="C")).hexdigest()
                    != case["semantic_base_row_sha256"]
                ):
                    raise RuntimeError(f"semantic row SHA mismatch: {case['case_id']}")
                observation = store.actor_observation(case["row_index"], snapshot)
                if observation.shape != (3080,) or not np.isfinite(observation).all():
                    raise RuntimeError("invalid Actor observation")
                probabilities_tensor = model(torch.from_numpy(observation))["probabilities"]
                prompt_index = ACTION_ORDER.index(case["prompt_action"])
                action = deterministic_action(probabilities_tensor, prompt_index)
                probabilities = [float(value) for value in probabilities_tensor.tolist()]
                override = action != case["prompt_action"]
                overrides += int(override)
                pre_hash = _state_hash(state)
                credit = simulate_local_credit(
                    reward_window(symbol, case["decision_session"], market), state
                )
                selected_r3 = credit.rewards_r3[action]
                if not math.isfinite(selected_r3):
                    raise RuntimeError("non-finite selected-action R3")
                rewards_by_symbol[symbol].append(selected_r3)
                child = (
                    VALIDATION_SESSIONS[index + 1]
                    if index + 1 < len(VALIDATION_SESSIONS)
                    else credit.reward_maturity_session
                )
                transition = simulate_weekly_transition(
                    weekly_transition_window(symbol, case["decision_session"], child, market),
                    state,
                    action,
                )
                state = transition.next_state
                snapshot = transition.next_snapshot
                records.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_parameter_sha": candidate["candidate_parameter_sha"],
                        "symbol": symbol,
                        "decision_session": case["decision_session"],
                        "semantic_row_identity": case["semantic_base_row_sha256"],
                        "pre_action_portfolio_state_hash": pre_hash,
                        "prompt_action": case["prompt_action"],
                        "actor_probabilities": probabilities,
                        "deterministic_selected_action": action,
                        "prompt_override": override,
                        "execution_identity": credit.input_sha256_by_action[action],
                        "selected_action_local_r3": selected_r3,
                        "post_transition_portfolio_state_hash": _state_hash(state),
                        "reward_maturity_session": credit.reward_maturity_session,
                    }
                )
    after_parameters = canonical_parameter_sha(model)
    after_file = sha256_file(candidate_dir / "model.pt")
    if (before_parameters, before_file) != (after_parameters, after_file):
        raise RuntimeError(f"candidate mutation detected: {candidate['candidate_id']}")
    per_symbol = {symbol: math.fsum(rewards_by_symbol[symbol]) for symbol in VALIDATION_SYMBOLS}
    all_rewards = [record["selected_action_local_r3"] for record in records]
    score = {
        "candidate_id": candidate["candidate_id"],
        "learning_rate": candidate["learning_rate"],
        "checkpoint_iteration": candidate["outer_iteration"],
        "candidate_parameter_sha": candidate["candidate_parameter_sha"],
        "primary_mean_sequential_local_r3": math.fsum(all_rewards) / 16,
        "per_symbol_cumulative_r3": per_symbol,
        "worst_symbol_cumulative_r3": min(per_symbol.values()),
        "override_count": overrides,
        "override_rate": overrides / 16,
        "validation_decisions": 16,
        "correctness_valid": True,
    }
    return records, score


def verify_plan(
    plan: Mapping[str, Any], source_sha: str, validation_population_identity: str
) -> None:
    expected_candidates = [
        {
            "candidate_id": candidate_id,
            "learning_rate": lr,
            "checkpoint_iteration": iteration,
            "candidate_parameter_sha": sha,
        }
        for candidate_id, (lr, iteration, sha) in CANDIDATE_GRID.items()
    ]
    checks = {
        "status": "PREREGISTERED — VALIDATION PERFORMANCE NOT YET INSPECTED",
        "selector_source_sha": source_sha,
        "candidates": expected_candidates,
        "validation_population_identity": validation_population_identity,
        "selection_tie_tolerance": SELECTION_TIE_TOLERANCE,
        "actor_action_tie_tolerance": ACTION_TIE_TOLERANCE,
        "reward_identity": REWARD_IDENTITY,
        "market_snapshot_identity": MARKET_IDENTITY,
        "representation_identity": REPRESENTATION_IDENTITY,
    }
    for key, expected in checks.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"selection plan mismatch for {key}")


def score_all(source_root: Path, experiments_root: Path, output_root: Path) -> dict[str, Any]:
    source_head = verify_repository(source_root, "baseline-m2", STARTING_SOURCE_SHA)
    experiments_head = verify_repository(experiments_root, "main", STARTING_EXPERIMENTS_SHA)
    candidates = verify_candidates(experiments_root)
    cases, population_identity = load_validation_cases(experiments_root)
    verify_market(experiments_root)
    plan_path = output_root / "selection_plan.json"
    plan = json.loads(plan_path.read_text())
    verify_plan(plan, source_head, population_identity)
    if _git(experiments_root, "log", "-1", "--format=%H", "--", str(plan_path)) != experiments_head:
        raise RuntimeError("selection plan is not committed at current Experiments HEAD")
    if (
        _git(source_root, "log", "-1", "--format=%H", "--", "scripts/m2/select_global_pa_ctppo.py")
        != source_head
    ):
        raise RuntimeError("selector is not committed at current source HEAD")
    decisions, scores = [], []
    for candidate in candidates:
        records, score = evaluate_candidate(candidate, cases, experiments_root)
        decisions.extend(records)
        scores.append(score)
    if len(decisions) != 144:
        raise RuntimeError("official selection did not produce exactly 144 decisions")
    selected, tie = choose_candidate(scores)
    result = {
        "schema_version": SCHEMA_VERSION,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_parameter_sha": selected["candidate_parameter_sha"],
        "actor_parameter_sha": next(
            x["actor_parameter_sha"]
            for x in candidates
            if x["candidate_id"] == selected["candidate_id"]
        ),
        "critic_parameter_sha": next(
            x["critic_parameter_sha"]
            for x in candidates
            if x["candidate_id"] == selected["candidate_id"]
        ),
        "learning_rate": selected["learning_rate"],
        "checkpoint_iteration": selected["checkpoint_iteration"],
        "primary_mean_sequential_local_r3": selected["primary_mean_sequential_local_r3"],
        "worst_symbol_cumulative_r3": selected["worst_symbol_cumulative_r3"],
        "override_rate": selected["override_rate"],
        "tie_break": tie,
        "selection_rule_identity": payload_sha(plan["selection_rule"]),
        "selector_source_sha": source_head,
        "selection_plan_sha": experiments_head,
        "m2_11_training_archive_sha": STARTING_EXPERIMENTS_SHA,
        "representation_identity": REPRESENTATION_IDENTITY,
        "reward_identity": REWARD_IDENTITY,
        "market_snapshot_identity": MARKET_IDENTITY,
        "validation_population_identity": population_identity,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "HUMAN_OVERRIDE": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "validation_decisions.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True, allow_nan=False) + "\n" for x in decisions)
    )
    _write_json(output_root / "candidate_validation_scores.json", scores)
    _write_json(output_root / "selection_result.json", result)
    selected_source = (
        experiments_root
        / f"experiments/M2/development/global_training_v1/candidates/{selected['candidate_id']}"
    )
    selected_root = output_root / "selected_global_checkpoint"
    selected_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_source / "model.pt", selected_root / "model.pt")
    shutil.copyfile(selected_source / "manifest.json", selected_root / "candidate_manifest.json")
    binding = {
        **result,
        "model_file_sha256": sha256_file(selected_root / "model.pt"),
        "byte_identical_to_canonical": True,
    }
    _write_json(selected_root / "manifest.json", binding)
    if sha256_file(selected_source / "model.pt") != sha256_file(selected_root / "model.pt"):
        raise RuntimeError("selected checkpoint copy is not byte-identical")
    return {"decisions": decisions, "scores": scores, "result": result}


def audit_replay(
    official: Mapping[str, Any], source_root: Path, experiments_root: Path, output_root: Path
) -> dict[str, Any]:
    candidates = verify_candidates(experiments_root)
    cases, _ = load_validation_cases(experiments_root)
    replay_decisions, replay_scores = [], []
    for candidate in candidates:
        records, score = evaluate_candidate(candidate, cases, experiments_root)
        replay_decisions.extend(records)
        replay_scores.append(score)
    selected, tie = choose_candidate(replay_scores)
    official_decisions = official["decisions"]
    action_mismatches = sum(
        a["deterministic_selected_action"] != b["deterministic_selected_action"]
        for a, b in zip(official_decisions, replay_decisions, strict=True)
    )
    reward_mismatches = sum(
        a["selected_action_local_r3"] != b["selected_action_local_r3"]
        for a, b in zip(official_decisions, replay_decisions, strict=True)
    )
    score_mismatches = sum(
        a["primary_mean_sequential_local_r3"] != b["primary_mean_sequential_local_r3"]
        for a, b in zip(official["scores"], replay_scores, strict=True)
    )
    audit = {
        "status": "AUDIT_ONLY",
        "same_selected_candidate": selected["candidate_id"]
        == official["result"]["selected_candidate_id"],
        "selected_candidate_id": selected["candidate_id"],
        "action_mismatches": action_mismatches,
        "reward_mismatches": reward_mismatches,
        "primary_score_mismatches": score_mismatches,
        "same_tie_break_path": tie == official["result"]["tie_break"],
        "decision_count": len(replay_decisions),
    }
    if not (
        audit["same_selected_candidate"]
        and audit["same_tie_break_path"]
        and action_mismatches == reward_mismatches == score_mismatches == 0
    ):
        raise RuntimeError("selection audit replay mismatch")
    _write_json(output_root / "selection_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=CANONICAL_SOURCE_ROOT)
    parser.add_argument("--experiments-root", type=Path, default=CANONICAL_EXPERIMENTS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CANONICAL_EXPERIMENTS_ROOT / "experiments/M2/development/global_selection_v1",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        candidates = verify_candidates(args.experiments_root)
        cases, population_identity = load_validation_cases(args.experiments_root)
        verify_market(args.experiments_root)
        print(
            json.dumps(
                {
                    "candidate_count": len(candidates),
                    "validation_case_count": len(cases),
                    "validation_population_identity": population_identity,
                    "validation_performance_inspected": False,
                },
                sort_keys=True,
            )
        )
        return
    official = score_all(args.source_root, args.experiments_root, args.output_root)
    audit_replay(official, args.source_root, args.experiments_root, args.output_root)
    print(json.dumps(official["result"], sort_keys=True))


if __name__ == "__main__":
    main()
