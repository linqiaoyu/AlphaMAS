#!/usr/bin/env python3
"""Materialise the frozen M2 TRAIN+VALIDATION semantic hand-off corpus.

This M2-08 orchestrator deliberately delegates semantic generation to the
unchanged M2-07 primitive.  It adds only membership planning, byte-for-byte
reuse, durable resume, validation, budget gates, and deterministic manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.m2.run_semantic_handoff_probe import (
    CORPUS_IDENTITY,
    MODEL,
    SCHEMA_VERSION,
    TIER_IDENTITIES,
    M2FrozenEvidenceStore,
    SemanticUsageRecorder,
    _actor_state,
    _upstream_trace,
    build_probe_graph,
    canonical_json,
    case_usage,
    read_json,
    sha256_bytes,
    sha256_file,
    validate_actor_visible_state,
    write_case_artifacts,
    write_json,
)
from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.runtime.run_context import AuditTrail, RunContext, activate_run_context

TASK_ID = "M2-08"
STARTING_SOURCE_SHA = "0a93008da45e0b2c066827ba423f226eae3cc793"
STARTING_EXPERIMENTS_SHA = "123a0b412c91379a1c2279ca9e31a5c19743c40f"
FROZEN_RUNNER_SHA = "6d3c46b96b8935944e0f232b73020262d870d953"
FROZEN_RUNNER_FILE_SHA256 = "f4aacbfaf6accc34632012bcacc330578518c9d2e6149dd4eee1d71d6c40d5ef"
FROZEN_EVIDENCE_SHA = "6b2406f1e12e1988c27b44880a1e153a9b750c2e"
SELECTED_TIER = "MAXIMUM"
SELECTED_TIER_IDENTITY = TIER_IDENTITIES[SELECTED_TIER]
CALIBRATION_ARCHIVE_SHA = STARTING_EXPERIMENTS_SHA
M2_07_ACTUAL_COST_CNY = Decimal("0.95720644")
M2_08_TARGET_CNY = Decimal("15.00")
M2_08_HARD_CEILING_CNY = Decimal("18.00")
CUMULATIVE_TASK_CEILING_CNY = Decimal("19.00")
PREFORMAL_TARGET_CNY = Decimal("40.00")
PREFORMAL_HARD_CEILING_CNY = Decimal("50.00")
RESERVE_CNY = Decimal("4.00")
NEW_PAID_CASES = 66

REUSED_CASE_IDS = (
    "AAPL:2023-06-09",
    "JBSS:2023-06-02",
    "AEMD:2023-06-02",
    "EML:2023-06-09",
    "AGI:2023-06-02",
    "AMZN:2023-06-02",
)
EXPECTED_ROLE_COUNTS = {
    "TRAIN": 56,
    "VALIDATION": 16,
    "FINAL_HOLDOUT": 16,
    "E2E_PILOT": 8,
}
DEVELOPMENT_ROLES = ("TRAIN", "VALIDATION")
DEFERRED_ROLES = ("FINAL_HOLDOUT", "E2E_PILOT")
CASE_FILES = (
    "actor_visible_state.json",
    "upstream_trace.json",
    "usage.json",
    "sha256.json",
)
OUTCOME_METADATA_KEYS = frozenset(
    {
        "reward",
        "future_return",
        "future_price",
        "target",
        "label",
        "ground_truth",
        "best_action",
        "portfolio_outcome",
    }
)
SECRET_PATTERNS = {
    "deepseek_key_name": re.compile(rb"DEEPSEEK_API_KEY", re.I),
    "authorization": re.compile(rb"Authorization\s*[:=]", re.I),
    "bearer": re.compile(rb"Bearer\s+[A-Za-z0-9._~+/=-]+", re.I),
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "aws_session_token": re.compile(rb"AWS_SESSION_TOKEN", re.I),
    "private_key": re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
}


def git(repository: Path, *args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args], cwd=repository, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def assert_clean_at(repository: Path, expected_sha: str) -> None:
    if git(repository, "rev-parse", "HEAD") != expected_sha:
        raise RuntimeError(f"repository HEAD differs from paid-run lineage: {repository}")
    if git(repository, "status", "--short"):
        raise RuntimeError(f"repository is dirty at paid-run gate: {repository}")


def assert_frozen_runner(source_repo: Path) -> None:
    runner = source_repo / "scripts/m2/run_semantic_handoff_probe.py"
    if sha256_file(runner) != FROZEN_RUNNER_FILE_SHA256:
        raise RuntimeError("frozen M2-07 runner byte identity differs")
    if git(
        source_repo,
        "diff",
        "--name-only",
        FROZEN_RUNNER_SHA,
        STARTING_SOURCE_SHA,
        "--",
        "scripts/m2/run_semantic_handoff_probe.py",
    ):
        raise RuntimeError("frozen M2-07 runner has Git drift")


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def role_directory(role: str) -> str:
    return role.lower()


def case_directory(root: Path, case: Mapping[str, Any]) -> Path:
    return (
        root / "cases" / role_directory(str(case["role"])) / str(case["case_id"]).replace(":", "_")
    )


def validate_development_actor(value: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    validate_actor_visible_state(value)
    for key in ("case_id", "symbol", "decision_session", "role"):
        if value.get(key) != case.get(key):
            raise ValueError(f"Actor state {key} differs for {case['case_id']}")
    if value.get("role") not in DEVELOPMENT_ROLES:
        raise ValueError("Actor state role is not TRAIN or VALIDATION")
    if value.get("evidence_packet_sha256") != case.get("evidence_packet_sha256"):
        raise ValueError("Actor state Evidence Packet SHA differs")
    leaked = _walk_keys(value) & OUTCOME_METADATA_KEYS
    if leaked:
        raise ValueError(f"outcome metadata keys are forbidden: {sorted(leaked)}")


def build_development_actor_state(
    case: Mapping[str, Any],
    packet: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    runner_sha: str,
    plan_sha: str,
) -> dict[str, Any]:
    """Generalise M2-07 only by assigning the frozen case role."""
    probe_case = {**case, "packet_sha256": case["evidence_packet_sha256"]}
    value = _actor_state(
        probe_case,
        packet,
        state,
        runner_sha=runner_sha,
        probe_inputs_sha=plan_sha,
    )
    value["role"] = case["role"]
    validate_development_actor(value, case)
    return value


def canonical_membership(corpus_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    store = M2FrozenEvidenceStore(corpus_root)
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in EXPECTED_ROLE_COUNTS}
    for case_id, membership in store.membership.items():
        if membership.get("in_maximum") is not True:
            raise ValueError(f"MAXIMUM membership excludes frozen case {case_id}")
        symbol, decision_session = case_id.split(":", 1)
        packet = store.get_packet(symbol, decision_session)
        role = str(packet.get("role"))
        if role not in by_role:
            raise ValueError(f"unexpected frozen role for {case_id}: {role}")
        by_role[role].append(
            {
                "case_id": case_id,
                "symbol": symbol,
                "decision_session": decision_session,
                "role": role,
                "evidence_packet_sha256": packet["packet_sha256"],
            }
        )
    actual = {role: len(rows) for role, rows in by_role.items()}
    if actual != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"frozen role membership differs: {actual}")
    development: list[dict[str, Any]] = []
    generation_order = 0
    for role in DEVELOPMENT_ROLES:
        for row in by_role[role]:
            generation_order += 1
            development.append(
                {
                    **row,
                    "selected_tier": SELECTED_TIER,
                    "generation_mode": (
                        "REUSE_M2_07" if row["case_id"] in REUSED_CASE_IDS else "GENERATE_M2_08"
                    ),
                    "generation_order": generation_order,
                    "protected": False,
                }
            )
    deferred = [
        {**row, "selected_tier": SELECTED_TIER, "status": "DEFERRED"}
        for role in DEFERRED_ROLES
        for row in by_role[role]
    ]
    validate_plan_cases(development, deferred)
    return development, deferred


def validate_plan_cases(
    development: list[Mapping[str, Any]], deferred: list[Mapping[str, Any]]
) -> None:
    counts = Counter(row["role"] for row in development)
    if counts != Counter({"TRAIN": 56, "VALIDATION": 16}):
        raise ValueError(f"development membership differs: {dict(counts)}")
    reuse = [row["case_id"] for row in development if row["generation_mode"] == "REUSE_M2_07"]
    if len(reuse) != 6 or set(reuse) != set(REUSED_CASE_IDS):
        raise ValueError("exactly the six frozen probes must be reused")
    generated = [row for row in development if row["generation_mode"] == "GENERATE_M2_08"]
    if Counter(row["role"] for row in generated) != Counter({"TRAIN": 50, "VALIDATION": 16}):
        raise ValueError("new paid TRAIN/VALIDATION membership differs")
    if [row["generation_order"] for row in development] != list(range(1, 73)):
        raise ValueError("generation order must be the canonical 1..72 sequence")
    deferred_counts = Counter(row["role"] for row in deferred)
    if deferred_counts != Counter({"FINAL_HOLDOUT": 16, "E2E_PILOT": 8}):
        raise ValueError("deferred membership differs")
    if _walk_keys({"development": development, "deferred": deferred}) & OUTCOME_METADATA_KEYS:
        raise ValueError("generation plan contains outcome metadata")


def pricing_document(
    *, retrieved_at: str, cache_hit: str, cache_miss: str, output: str
) -> dict[str, Any]:
    drift = (cache_hit, cache_miss, output) != ("0.02", "1.00", "2.00")
    return {
        "schema_version": "M2-DEEPSEEK-PRICING-v1",
        "task_id": TASK_ID,
        "model": MODEL,
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing",
        "source_authority": "official DeepSeek API documentation",
        "retrieval_timestamp": retrieved_at,
        "billing_currency": "CNY",
        "unit": "per_1m_tokens",
        "input_cache_hit_price": cache_hit,
        "input_cache_miss_price": cache_miss,
        "output_price": output,
        "m2_07_prices": {
            "input_cache_hit_price": "0.02",
            "input_cache_miss_price": "1.00",
            "output_price": "2.00",
        },
        "pricing_drift": drift,
        "selected_tier_changed": False,
        "conservative_fallback": "all input tokens are priced as cache miss when the provider split is incomplete",
    }


def no_cache_case_cost(usage: Mapping[str, Any], prices: Mapping[str, Any]) -> Decimal:
    return (
        Decimal(int(usage["input_tokens"])) * Decimal(str(prices["input_cache_miss_price"]))
        + Decimal(int(usage["output_tokens"])) * Decimal(str(prices["output_price"]))
    ) / Decimal(1_000_000)


def no_cache_preflight(calibration_root: Path, prices: Mapping[str, Any]) -> dict[str, Any]:
    per_case = []
    for case_id in REUSED_CASE_IDS:
        usage = read_json(calibration_root / "cases" / case_id.replace(":", "_") / "usage.json")
        cost = no_cache_case_cost(usage, prices)
        per_case.append({"case_id": case_id, "no_cache_case_cost_cny": str(cost)})
    guard = max(Decimal(row["no_cache_case_cost_cny"]) for row in per_case)
    projection = guard * Decimal(NEW_PAID_CASES)
    cumulative = M2_07_ACTUAL_COST_CNY + projection
    with_reserve = cumulative + RESERVE_CNY
    if with_reserve > PREFORMAL_TARGET_CNY:
        raise RuntimeError("no-cache cumulative projection plus reserve exceeds CNY 40")
    if projection > M2_08_HARD_CEILING_CNY:
        raise RuntimeError("projected M2-08 incremental spend exceeds CNY 18")
    if cumulative > CUMULATIVE_TASK_CEILING_CNY:
        raise RuntimeError("projected M2-07 plus M2-08 spend exceeds CNY 19")
    return {
        "formula": "66 * maximum(six M2-07 all-input-cache-miss case costs)",
        "per_case": per_case,
        "no_cache_guard_cny": str(guard),
        "new_paid_case_count": NEW_PAID_CASES,
        "m2_08_no_cache_projection_cny": str(projection),
        "m2_07_actual_cost_cny": str(M2_07_ACTUAL_COST_CNY),
        "cumulative_after_m2_08_projection_cny": str(cumulative),
        "reserve_cny": str(RESERVE_CNY),
        "projection_plus_reserve_cny": str(with_reserve),
        "within_preformal_target_40": True,
        "within_m2_08_hard_ceiling_18": True,
        "within_cumulative_task_ceiling_19": True,
    }


def prepare_plan(
    corpus_root: Path,
    calibration_root: Path,
    output_root: Path,
    *,
    retrieved_at: str,
    cache_hit: str,
    cache_miss: str,
    output_price: str,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"plan output root is not empty: {output_root}")
    development, deferred = canonical_membership(corpus_root)
    prices = pricing_document(
        retrieved_at=retrieved_at,
        cache_hit=cache_hit,
        cache_miss=cache_miss,
        output=output_price,
    )
    preflight = no_cache_preflight(calibration_root, prices)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "M2-SEMANTIC-TRAINVAL-GENERATION-PLAN-v1",
        "task_id": TASK_ID,
        "semantic_contract": SCHEMA_VERSION,
        "selected_tier": SELECTED_TIER,
        "selected_tier_case_count": 96,
        "selected_tier_identity": SELECTED_TIER_IDENTITY,
        "selection_uses_performance": False,
        "generation_ordering": "TRAIN then VALIDATION; canonical MAXIMUM manifest order within role",
        "cases": development,
        "deferred_cases": deferred,
        "deferred_roles": {
            "FINAL_HOLDOUT": {"case_count": 16, "status": "DEFERRED"},
            "E2E_PILOT": {"case_count": 8, "status": "DEFERRED"},
        },
        "no_cache_preflight": preflight,
    }
    write_json(output_root / "generation_plan.json", plan)
    write_json(output_root / "pricing_snapshot.json", prices)
    (output_root / "README.md").write_text(
        "# M2-08 semantic TRAIN+VALIDATION corpus\n\n"
        "This pre-Formal plan freezes the MAXIMUM-tier 56 TRAIN and 16 VALIDATION "
        "semantic hand-offs. Six TRAIN cases are reused byte-for-byte from M2-07; "
        "50 TRAIN and 16 VALIDATION cases are generated once. FINAL_HOLDOUT and "
        "E2E_PILOT remain deferred. No reward or outcome data is used.\n",
        encoding="utf-8",
    )
    return plan


def verify_sha_manifest(case_dir: Path) -> dict[str, str]:
    manifest = read_json(case_dir / "sha256.json")
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError(f"case SHA manifest is malformed: {case_dir}")
    if set(expected) != set(CASE_FILES[:-1]):
        raise ValueError(f"case SHA manifest membership differs: {case_dir}")
    actual = {name: sha256_file(case_dir / name) for name in CASE_FILES[:-1]}
    if actual != expected:
        raise ValueError(f"case SHA manifest differs: {case_dir}")
    return actual


def validate_case_artifacts(
    case_dir: Path,
    case: Mapping[str, Any],
    *,
    reused: bool,
    runner_sha: str,
    plan_sha: str,
) -> dict[str, str]:
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)
    missing = [name for name in CASE_FILES if not (case_dir / name).is_file()]
    if missing:
        raise ValueError(f"incomplete case artifacts at {case_dir}: {missing}")
    hashes = verify_sha_manifest(case_dir)
    actor = read_json(case_dir / "actor_visible_state.json")
    validate_development_actor(actor, case)
    trace = read_json(case_dir / "upstream_trace.json")
    if trace.get("actor_visible") is not False or trace.get("case_id") != case["case_id"]:
        raise ValueError(f"upstream trace contract differs: {case_dir}")
    capture = trace.get("capture_boundary", {})
    if capture != {
        "portfolio_manager_executed": False,
        "risk_debate_executed": False,
        "stop_after": "Prompt Trader",
        "trade_executed": False,
    }:
        raise ValueError(f"capture boundary differs: {case_dir}")
    usage = read_json(case_dir / "usage.json")
    if Decimal(str(usage.get("case_cost_cny"))) < 0:
        raise ValueError(f"negative case cost: {case_dir}")
    provenance = actor.get("provenance", {})
    if reused:
        if actor.get("reusable_for_M2_08") is not True:
            raise ValueError(f"reused state is not reusable: {case_dir}")
    elif (
        provenance.get("source_sha") != runner_sha or provenance.get("experiments_sha") != plan_sha
    ):
        raise ValueError(f"generated state lineage differs: {case_dir}")
    return hashes


def resume_case_status(
    case_dir: Path,
    case: Mapping[str, Any],
    *,
    reused: bool,
    runner_sha: str,
    plan_sha: str,
) -> str:
    """Classify an existing case without ever accepting corrupt complete bytes."""
    if not case_dir.exists():
        return "RETRY"
    present = {name for name in CASE_FILES if (case_dir / name).is_file()}
    if present != set(CASE_FILES):
        return "RETRY"
    validate_case_artifacts(
        case_dir,
        case,
        reused=reused,
        runner_sha=runner_sha,
        plan_sha=plan_sha,
    )
    return "SKIP"


def quarantine_incomplete_case(staging_root: Path, case: Mapping[str, Any]) -> None:
    target = case_directory(staging_root, case)
    if not target.exists():
        return
    present = {name for name in CASE_FILES if (target / name).is_file()}
    if present == set(CASE_FILES):
        raise RuntimeError("complete authoritative case cannot be quarantined")
    quarantine = staging_root / "incomplete_attempts"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{target.name}_{time.time_ns()}"
    os.replace(target, destination)


def verify_reuse_source(
    calibration_root: Path, plan: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    import subprocess

    archive_sha = read_json(calibration_root / "sha256.json")
    archive_files = archive_sha.get("files", {})
    experiments_repo = Path(git(calibration_root, "rev-parse", "--show-toplevel"))
    archive_relative_root = calibration_root.relative_to(experiments_repo)
    cases = {row["case_id"]: row for row in plan["cases"]}
    result = {}
    for case_id in REUSED_CASE_IDS:
        source = calibration_root / "cases" / case_id.replace(":", "_")
        case = cases[case_id]
        hashes = validate_case_artifacts(
            source,
            case,
            reused=True,
            runner_sha=FROZEN_RUNNER_SHA,
            plan_sha=STARTING_EXPERIMENTS_SHA,
        )
        for name in CASE_FILES:
            rel = f"cases/{case_id.replace(':', '_')}/{name}"
            file_sha = sha256_file(source / name)
            if name != "sha256.json" and archive_files.get(rel) != file_sha:
                raise ValueError(f"M2-07 archive hash differs for {rel}")
            committed = subprocess.run(
                [
                    "git",
                    "show",
                    f"{CALIBRATION_ARCHIVE_SHA}:{archive_relative_root / rel}",
                ],
                cwd=experiments_repo,
                check=True,
                capture_output=True,
            ).stdout
            if sha256_bytes(committed) != file_sha:
                raise ValueError(f"M2-07 committed bytes differ for {rel}")
        result[case_id] = {name: sha256_file(source / name) for name in CASE_FILES}
        result[case_id].update(hashes)
    return result


def write_staging_status(staging_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    complete_new = 0
    actual = Decimal(0)
    failed_spend = Decimal(0)
    for case in plan["cases"]:
        if case["generation_mode"] != "GENERATE_M2_08":
            continue
        target = case_directory(staging_root, case)
        if target.is_dir():
            complete_new += 1
            actual += Decimal(str(read_json(target / "usage.json")["case_cost_cny"]))
    for path in sorted((staging_root / "failed_attempts").glob("*.json")):
        failed_spend += Decimal(str(read_json(path).get("case_cost_cny", "0")))
    actual += failed_spend
    status = {
        "schema_version": "M2-SEMANTIC-TRAINVAL-STAGING-STATUS-v1",
        "reused_valid": sum(
            case_directory(staging_root, case).is_dir()
            for case in plan["cases"]
            if case["generation_mode"] == "REUSE_M2_07"
        ),
        "new_completed": complete_new,
        "new_remaining": NEW_PAID_CASES - complete_new,
        "m2_08_actual_spend_cny": str(actual),
        "m2_08_failed_billed_spend_cny": str(failed_spend),
    }
    write_json(staging_root / "staging_status.json", status)
    return status


def initialise_staging(
    staging_root: Path,
    plan_root: Path,
    calibration_root: Path,
) -> dict[str, Any]:
    plan = read_json(plan_root / "generation_plan.json")
    validate_plan_cases(plan["cases"], plan["deferred_cases"])
    reuse_hashes = verify_reuse_source(calibration_root, plan)
    staging_root.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "generation_plan.json", "pricing_snapshot.json"):
        source = plan_root / name
        target = staging_root / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise RuntimeError(f"staging metadata differs: {name}")
        if not target.exists():
            shutil.copy2(source, target)
    cases = {row["case_id"]: row for row in plan["cases"]}
    for case_id in REUSED_CASE_IDS:
        case = cases[case_id]
        source = calibration_root / "cases" / case_id.replace(":", "_")
        target = case_directory(staging_root, case)
        if target.exists():
            validate_case_artifacts(
                target,
                case,
                reused=True,
                runner_sha=FROZEN_RUNNER_SHA,
                plan_sha=STARTING_EXPERIMENTS_SHA,
            )
            if any(sha256_file(target / name) != sha256_file(source / name) for name in CASE_FILES):
                raise RuntimeError(f"staged reuse bytes differ: {case_id}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, copy_function=shutil.copy2)
        if any(sha256_file(target / name) != sha256_file(source / name) for name in CASE_FILES):
            raise RuntimeError(f"reuse copy changed bytes: {case_id}")
    write_json(
        staging_root / "reuse_source_validation.json",
        {
            "source_archive": "semantic_cost_calibration_v1",
            "source_commit": CALIBRATION_ARCHIVE_SHA,
            "reused_case_count": 6,
            "byte_identical": True,
            "cases": reuse_hashes,
        },
    )
    return write_staging_status(staging_root, plan)


def completed_costs(
    staging_root: Path, plan: Mapping[str, Any]
) -> tuple[Decimal, Decimal, list[Decimal], int]:
    actual = Decimal(0)
    failed = Decimal(0)
    successful: list[Decimal] = []
    count = 0
    for case in plan["cases"]:
        if case["generation_mode"] != "GENERATE_M2_08":
            continue
        target = case_directory(staging_root, case)
        if target.is_dir():
            cost = Decimal(str(read_json(target / "usage.json")["case_cost_cny"]))
            actual += cost
            successful.append(cost)
            count += 1
    for path in sorted((staging_root / "failed_attempts").glob("*.json")):
        cost = Decimal(str(read_json(path).get("case_cost_cny", "0")))
        failed += cost
        actual += cost
    return actual, failed, successful, count


def enforce_budget(
    actual: Decimal,
    successful_costs: list[Decimal],
    completed: int,
    no_cache_guard: Decimal,
) -> None:
    if actual >= M2_08_HARD_CEILING_CNY:
        raise RuntimeError("actual M2-08 spend reached CNY 18")
    if M2_07_ACTUAL_COST_CNY + actual >= CUMULATIVE_TASK_CEILING_CNY:
        raise RuntimeError("M2-07 plus M2-08 actual spend reached CNY 19")
    guard = max([no_cache_guard, *successful_costs])
    forecast = actual + guard * Decimal(NEW_PAID_CASES - completed)
    if forecast > M2_08_HARD_CEILING_CNY:
        raise RuntimeError(f"conservative M2-08 completion forecast exceeds CNY 18: {forecast}")


def _atomic_case_write(
    staging_root: Path,
    case: Mapping[str, Any],
    actor: Mapping[str, Any],
    trace: Mapping[str, Any],
    usage: Mapping[str, Any],
) -> None:
    target = case_directory(staging_root, case)
    if target.exists():
        raise RuntimeError(f"authoritative case already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".m2-08-case-", dir=target.parent))
    try:
        write_case_artifacts(temporary_root, case, actor, trace, usage)
        generated = temporary_root / "cases" / case["case_id"].replace(":", "_")
        os.replace(generated, target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def run_generation(
    source_repo: Path,
    experiments_repo: Path,
    corpus_root: Path,
    staging_root: Path,
    *,
    orchestrator_sha: str,
    plan_sha: str,
) -> dict[str, Any]:
    assert_clean_at(source_repo, orchestrator_sha)
    assert_clean_at(experiments_repo, plan_sha)
    assert_frozen_runner(source_repo)
    plan = read_json(staging_root / "generation_plan.json")
    prices = read_json(staging_root / "pricing_snapshot.json")
    validate_plan_cases(plan["cases"], plan["deferred_cases"])
    if (staging_root / "blocked_cases").exists():
        blocked = sorted((staging_root / "blocked_cases").glob("*.json"))
        if blocked:
            raise RuntimeError(f"complete invalid output requires investigation: {blocked[0]}")
    no_cache_guard = Decimal(plan["no_cache_preflight"]["no_cache_guard_cny"])
    recorder = SemanticUsageRecorder()
    store = M2FrozenEvidenceStore(corpus_root)
    graph, runtime_root = build_probe_graph(store, recorder)
    try:
        for case in plan["cases"]:
            target = case_directory(staging_root, case)
            if case["generation_mode"] == "REUSE_M2_07":
                validate_case_artifacts(
                    target,
                    case,
                    reused=True,
                    runner_sha=FROZEN_RUNNER_SHA,
                    plan_sha=STARTING_EXPERIMENTS_SHA,
                )
                continue
            if target.exists():
                status = resume_case_status(
                    target,
                    case,
                    reused=False,
                    runner_sha=FROZEN_RUNNER_SHA,
                    plan_sha=plan_sha,
                )
                if status == "SKIP":
                    continue
                quarantine_incomplete_case(staging_root, case)
            # TRAIN must be complete before the first VALIDATION paid call.
            if case["role"] == "VALIDATION":
                incomplete_train = [
                    row
                    for row in plan["cases"]
                    if row["role"] == "TRAIN" and not case_directory(staging_root, row).is_dir()
                ]
                if incomplete_train:
                    raise RuntimeError("VALIDATION cannot run before all TRAIN states complete")
            assert_frozen_runner(source_repo)
            actual, _failed, successful, completed = completed_costs(staging_root, plan)
            enforce_budget(actual, successful, completed, no_cache_guard)
            packet = store.get_packet(case["symbol"], case["decision_session"])
            context = RunContext.historical(
                packet["decision_time"],
                experiment_id="M2_semantic_handoff_trainval_v1",
                memory_lineage_id=f"M2-08:{case['case_id']}",
            )
            audit = AuditTrail(context)
            start = recorder.start_case(case)
            try:
                with activate_run_context(context, audit):
                    instrument_context = build_instrument_context(case["symbol"])
                    initial_state = graph.propagator.create_initial_state(
                        case["symbol"],
                        case["decision_session"],
                        asset_type="stock",
                        past_context="",
                        instrument_context=instrument_context,
                        run_context=context,
                    )
                    state = graph.graph.invoke(initial_state, **graph.propagator.get_graph_args())
            except Exception:
                rows = recorder.finish_case(start)
                failed_root = staging_root / "failed_attempts"
                failed_root.mkdir(parents=True, exist_ok=True)
                write_json(
                    failed_root / f"{case['case_id'].replace(':', '_')}_{time.time_ns()}.json",
                    {"case_id": case["case_id"], **case_usage(rows, prices)},
                )
                write_staging_status(staging_root, plan)
                raise
            rows = recorder.finish_case(start)
            usage = case_usage(rows, prices)
            try:
                actor = build_development_actor_state(
                    case,
                    packet,
                    state,
                    runner_sha=FROZEN_RUNNER_SHA,
                    plan_sha=plan_sha,
                )
                trace = _upstream_trace(case, state, audit)
                if trace.get("actor_visible") is not False:
                    raise ValueError("upstream trace became Actor-visible")
            except Exception as exc:
                blocked_root = staging_root / "blocked_cases"
                blocked_root.mkdir(parents=True, exist_ok=True)
                write_json(
                    blocked_root / f"{case['case_id'].replace(':', '_')}.json",
                    {
                        "case_id": case["case_id"],
                        "status": "COMPLETE_BUT_INVALID",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "usage": usage,
                        "state_sha256": sha256_bytes(canonical_json(state)),
                    },
                )
                write_staging_status(staging_root, plan)
                raise RuntimeError("complete output violates semantic contract") from exc
            _atomic_case_write(staging_root, case, actor, trace, usage)
            validate_case_artifacts(
                target,
                case,
                reused=False,
                runner_sha=FROZEN_RUNNER_SHA,
                plan_sha=plan_sha,
            )
            status = write_staging_status(staging_root, plan)
            actual, _failed, successful, completed = completed_costs(staging_root, plan)
            if completed < NEW_PAID_CASES:
                enforce_budget(actual, successful, completed, no_cache_guard)
            print(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "case_id": case["case_id"],
                        "new_completed": status["new_completed"],
                        "new_remaining": status["new_remaining"],
                        "actual_spend_cny": status["m2_08_actual_spend_cny"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)
    status = write_staging_status(staging_root, plan)
    if status["new_completed"] != NEW_PAID_CASES:
        raise RuntimeError("66 correctness-valid new cases are required")
    return status


def audit_corpus(
    staging_root: Path,
    *,
    orchestrator_sha: str,
    plan_sha: str,
    write_manifests: bool,
) -> dict[str, Any]:
    plan = read_json(staging_root / "generation_plan.json")
    validate_plan_cases(plan["cases"], plan["deferred_cases"])
    rows = []
    reuse_rows = []
    total_calls = total_input = total_output = 0
    actual_cost = Decimal(0)
    failed_cost = Decimal(0)
    maximum_cost = Decimal(0)
    for case in plan["cases"]:
        reused = case["generation_mode"] == "REUSE_M2_07"
        target = case_directory(staging_root, case)
        hashes = validate_case_artifacts(
            target,
            case,
            reused=reused,
            runner_sha=FROZEN_RUNNER_SHA,
            plan_sha=plan_sha,
        )
        actor_sha = hashes["actor_visible_state.json"]
        usage = read_json(target / "usage.json")
        cost = Decimal(str(usage["case_cost_cny"]))
        if not reused:
            total_calls += int(usage["total_calls"])
            total_input += int(usage["input_tokens"])
            total_output += int(usage["output_tokens"])
            actual_cost += cost
            maximum_cost = max(maximum_cost, cost)
        row = {
            "case_id": case["case_id"],
            "symbol": case["symbol"],
            "decision_session": case["decision_session"],
            "role": case["role"],
            "evidence_packet_sha256": case["evidence_packet_sha256"],
            "generation_mode": case["generation_mode"],
            "generation_order": case["generation_order"],
            "actor_visible_state_sha256": actor_sha,
            "case_sha256_manifest_sha256": sha256_file(target / "sha256.json"),
            "case_files": hashes,
        }
        rows.append(row)
        if reused:
            reuse_rows.append(row)
    for path in sorted((staging_root / "failed_attempts").glob("*.json")):
        failed = read_json(path)
        cost = Decimal(str(failed.get("case_cost_cny", "0")))
        failed_cost += cost
        actual_cost += cost
        total_calls += int(failed.get("total_calls", 0))
        total_input += int(failed.get("input_tokens", 0))
        total_output += int(failed.get("output_tokens", 0))
    identity_binding = {
        "semantic_contract": SCHEMA_VERSION,
        "selected_tier_identity": SELECTED_TIER_IDENTITY,
        "frozen_evidence_corpus_identity": CORPUS_IDENTITY,
        "frozen_generation_primitive_sha": FROZEN_RUNNER_SHA,
        "m2_08_orchestrator_sha": orchestrator_sha,
        "m2_08_plan_sha": plan_sha,
        "cases": rows,
    }
    identity = sha256_bytes(canonical_json(identity_binding))
    result = {
        "case_count": len(rows),
        "role_counts": dict(Counter(row["role"] for row in rows)),
        "reuse_count": len(reuse_rows),
        "generated_count": len(rows) - len(reuse_rows),
        "actor_hashes": {row["case_id"]: row["actor_visible_state_sha256"] for row in rows},
        "semantic_handoff_trainval_corpus_identity_sha256": identity,
        "total_llm_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "m2_08_actual_cost_cny": str(actual_cost),
        "failed_billed_spend_cny": str(failed_cost),
        "maximum_case_cost_cny": str(maximum_cost),
    }
    if len(rows) != 72 or result["role_counts"] != {"TRAIN": 56, "VALIDATION": 16}:
        raise ValueError("audited development corpus population differs")
    if result["reuse_count"] != 6 or result["generated_count"] != 66:
        raise ValueError("audited reuse/generated counts differ")
    if actual_cost > M2_08_HARD_CEILING_CNY:
        raise RuntimeError("audited M2-08 cost exceeds CNY 18")
    if M2_07_ACTUAL_COST_CNY + actual_cost >= CUMULATIVE_TASK_CEILING_CNY:
        raise RuntimeError("audited cumulative task cost reaches CNY 19")
    if write_manifests:
        manifests = staging_root / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        corpus_manifest = {
            "schema_version": "M2-SEMANTIC-TRAINVAL-CORPUS-MANIFEST-v1",
            "task_id": TASK_ID,
            "contract": SCHEMA_VERSION,
            "selected_tier": SELECTED_TIER,
            "selected_tier_case_count": 96,
            "selected_tier_identity": SELECTED_TIER_IDENTITY,
            "materialised_case_count": 72,
            "train": 56,
            "validation": 16,
            "final_holdout": {"materialised": 0, "deferred": 16},
            "e2e_pilot": {"materialised": 0, "deferred": 8},
            "m2_07_reused": 6,
            "m2_08_generated": 66,
            "frozen_m2_07_generation_primitive_sha": FROZEN_RUNNER_SHA,
            "m2_08_orchestrator_sha": orchestrator_sha,
            "m2_08_plan_sha": plan_sha,
            "semantic_handoff_trainval_corpus_identity_sha256": identity,
            "cases": rows,
        }
        reuse_manifest = {
            "schema_version": "M2-SEMANTIC-TRAINVAL-REUSE-MANIFEST-v1",
            "reused_case_count": 6,
            "source_archive": "semantic_cost_calibration_v1",
            "source_commit": CALIBRATION_ARCHIVE_SHA,
            "byte_identical": True,
            "cases": reuse_rows,
        }
        cost_manifest = {
            "schema_version": "M2-SEMANTIC-TRAINVAL-COST-MANIFEST-v1",
            "new_correctness_valid_cases": 66,
            "deepseek_requests": total_calls,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "m2_08_actual_cost_cny": str(actual_cost),
            "m2_08_failed_billed_spend_cny": str(failed_cost),
            "m2_07_actual_cost_cny": str(M2_07_ACTUAL_COST_CNY),
            "m2_07_plus_m2_08_cumulative_cost_cny": str(M2_07_ACTUAL_COST_CNY + actual_cost),
            "maximum_case_cost_cny": str(maximum_cost),
            "m2_08_target_cny": str(M2_08_TARGET_CNY),
            "m2_08_hard_ceiling_cny": str(M2_08_HARD_CEILING_CNY),
            "cumulative_task_ceiling_cny": str(CUMULATIVE_TASK_CEILING_CNY),
            "preformal_target_cny": str(PREFORMAL_TARGET_CNY),
            "preformal_hard_ceiling_cny": str(PREFORMAL_HARD_CEILING_CNY),
        }
        write_json(manifests / "corpus_manifest.json", corpus_manifest)
        write_json(manifests / "reuse_manifest.json", reuse_manifest)
        write_json(manifests / "cost_manifest.json", cost_manifest)
        write_archive_sha256(staging_root)
    return result


def scan_secrets(root: Path) -> list[dict[str, str]]:
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                findings.append({"file": path.relative_to(root).as_posix(), "pattern": name})
    return findings


def write_archive_sha256(root: Path) -> None:
    target = root / "manifests" / "sha256.json"
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path != target
        and path.name not in {"staging_status.json", "reuse_source_validation.json"}
        and "failed_attempts" not in path.parts
        and "blocked_cases" not in path.parts
    }
    write_json(target, {"algorithm": "SHA-256", "files": files})


def finalise_staging(staging_root: Path, *, orchestrator_sha: str, plan_sha: str) -> dict[str, Any]:
    first = audit_corpus(
        staging_root,
        orchestrator_sha=orchestrator_sha,
        plan_sha=plan_sha,
        write_manifests=True,
    )
    second = audit_corpus(
        staging_root,
        orchestrator_sha=orchestrator_sha,
        plan_sha=plan_sha,
        write_manifests=False,
    )
    for key in (
        "case_count",
        "role_counts",
        "actor_hashes",
        "semantic_handoff_trainval_corpus_identity_sha256",
    ):
        if first[key] != second[key]:
            raise RuntimeError(f"double corpus audit differs: {key}")
    findings = scan_secrets(staging_root)
    if findings:
        raise RuntimeError(f"secret scan failed: {findings}")
    plan = read_json(staging_root / "generation_plan.json")
    if any(case_directory(staging_root, row).exists() for row in plan["deferred_cases"]):
        raise RuntimeError("deferred role state files are present")
    if (staging_root / "blocked_cases").exists() and any(
        (staging_root / "blocked_cases").iterdir()
    ):
        raise RuntimeError("complete invalid output remains blocked")
    return {**first, "secret_scan": "PASS", "double_audit": "PASS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--corpus-root", type=Path, required=True)
    prepare.add_argument("--calibration-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--pricing-retrieved-at", required=True)
    prepare.add_argument("--cache-hit-price", required=True)
    prepare.add_argument("--cache-miss-price", required=True)
    prepare.add_argument("--output-price", required=True)
    initialise = sub.add_parser("initialise")
    initialise.add_argument("--staging-root", type=Path, required=True)
    initialise.add_argument("--plan-root", type=Path, required=True)
    initialise.add_argument("--calibration-root", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-repo", type=Path, required=True)
    run.add_argument("--experiments-repo", type=Path, required=True)
    run.add_argument("--corpus-root", type=Path, required=True)
    run.add_argument("--staging-root", type=Path, required=True)
    run.add_argument("--orchestrator-sha", required=True)
    run.add_argument("--plan-sha", required=True)
    finalise = sub.add_parser("finalise")
    finalise.add_argument("--staging-root", type=Path, required=True)
    finalise.add_argument("--orchestrator-sha", required=True)
    finalise.add_argument("--plan-sha", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_plan(
            args.corpus_root,
            args.calibration_root,
            args.output_root,
            retrieved_at=args.pricing_retrieved_at,
            cache_hit=args.cache_hit_price,
            cache_miss=args.cache_miss_price,
            output_price=args.output_price,
        )
        summary = {"status": "PASS", "development_cases": len(result["cases"])}
    elif args.command == "initialise":
        summary = {
            "status": "PASS",
            **initialise_staging(args.staging_root, args.plan_root, args.calibration_root),
        }
    elif args.command == "run":
        summary = {
            "status": "PASS",
            **run_generation(
                args.source_repo,
                args.experiments_repo,
                args.corpus_root,
                args.staging_root,
                orchestrator_sha=args.orchestrator_sha,
                plan_sha=args.plan_sha,
            ),
        }
    else:
        summary = {
            "status": "PASS",
            **finalise_staging(
                args.staging_root,
                orchestrator_sha=args.orchestrator_sha,
                plan_sha=args.plan_sha,
            ),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
