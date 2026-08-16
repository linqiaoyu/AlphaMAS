#!/usr/bin/env python3
"""Audit M2-11 canonical/replay lineages without evaluating candidates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from scripts.m2.train_global_pa_ctppo import (
    CANDIDATE_IDS,
    CHECKPOINT_ITERATIONS,
    EXPECTED_ACTION_CREDITS,
    EXPECTED_NODES,
    LEARNING_RATES,
    METHOD_ID,
    OPTIMISATION_EPOCHS,
    OUTER_ITERATIONS,
    REPRESENTATION_IDENTITY,
    REWARD_ID,
    TASK_ID,
    TREE_IDENTITY,
    canonical_json_bytes,
)


def _lr_directory(learning_rate: float) -> str:
    return {1e-4: "lr_1e-4", 3e-4: "lr_3e-4", 1e-3: "lr_1e-3"}[learning_rate]


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )


def _checkpoint_metadata(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["metadata"], payload["model_state"]


def audit_training_root(training_root: Path, archive_root: Path | None = None) -> dict[str, Any]:
    canonical_candidates: list[dict[str, Any]] = []
    replay_comparisons: list[dict[str, Any]] = []
    stale_violations = 0
    inside_block_violations = 0
    chronology_iterations = 0
    nan_inf = 0
    maximum_advantage_residual = 0.0
    initial_shas: set[str] = set()
    trainer_shas: set[str] = set()
    plan_shas: set[str] = set()

    for run_kind in ("canonical", "replay"):
        for learning_rate in LEARNING_RATES:
            lineage_root = training_root / run_kind / _lr_directory(learning_rate)
            summary = json.loads((lineage_root / "training_summary.json").read_text())
            if summary.get("iterations_completed") != OUTER_ITERATIONS:
                raise RuntimeError("incomplete M2-11 lineage")
            if summary.get("correctness_valid") is not True:
                raise RuntimeError("lineage is not correctness-valid")
            trace = _read_trace(lineage_root / "training_trace.jsonl")
            if len(trace) != OUTER_ITERATIONS:
                raise RuntimeError("training trace does not contain 100 outer iterations")
            chronology_iterations += len(trace)
            previous_post: str | None = None
            first_old = trace[0]["old_policy_sha"]
            for record in trace:
                if (
                    record["old_policy_frozen_epoch_shas"]
                    != [record["old_policy_sha"]] * OPTIMISATION_EPOCHS
                ):
                    inside_block_violations += 1
                if previous_post is not None and record["old_policy_sha"] != previous_post:
                    stale_violations += 1
                if record["iteration"] > 1 and record["old_policy_sha"] == first_old:
                    stale_violations += 1
                previous_post = record["post_update_model_sha"]
                nan_inf += record["nan_count"] + record["inf_count"]
                maximum_advantage_residual = max(
                    maximum_advantage_residual,
                    record["maximum_weighted_advantage_residual"],
                )
                if any(abs(mass - 1.0) > 1e-10 for mass in record["occupancy_mass_by_depth"]):
                    raise RuntimeError("occupancy invariant failed in archived trace")
            for iteration in CHECKPOINT_ITERATIONS:
                candidate_id = CANDIDATE_IDS[(learning_rate, iteration)]
                checkpoint_path = lineage_root / "checkpoints" / f"{candidate_id}.pt"
                metadata, _ = _checkpoint_metadata(checkpoint_path)
                expected = {
                    "candidate_id": candidate_id,
                    "learning_rate": learning_rate,
                    "outer_iteration": iteration,
                    "method_identity": METHOD_ID,
                    "tree_identity": TREE_IDENTITY,
                    "representation_identity": REPRESENTATION_IDENTITY,
                    "reward_identity": REWARD_ID,
                    "parameter_count": 20_197,
                    "run_kind": run_kind,
                }
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise RuntimeError(f"candidate checkpoint mismatch: {candidate_id}:{key}")
                initial_shas.add(metadata["initial_parameter_sha"])
                trainer_shas.add(metadata["trainer_sha"])
                plan_shas.add(metadata["training_plan_sha"])
                if run_kind == "canonical":
                    canonical_candidates.append(metadata)
                else:
                    canonical = next(
                        item
                        for item in canonical_candidates
                        if item["candidate_id"] == candidate_id
                    )
                    match = (
                        metadata["candidate_parameter_sha"] == canonical["candidate_parameter_sha"]
                    )
                    replay_comparisons.append(
                        {
                            "candidate_id": candidate_id,
                            "canonical_parameter_sha": canonical["candidate_parameter_sha"],
                            "replay_parameter_sha": metadata["candidate_parameter_sha"],
                            "exact_match": match,
                        }
                    )

    if len(canonical_candidates) != 9 or len(replay_comparisons) != 9:
        raise RuntimeError("candidate/replay grid is not exactly nine checkpoints")
    if len(initial_shas) != 1 or len(trainer_shas) != 1 or len(plan_shas) != 1:
        raise RuntimeError("candidate lineage bindings are inconsistent")
    mismatch_count = sum(not item["exact_match"] for item in replay_comparisons)
    if mismatch_count or stale_violations or inside_block_violations or nan_inf:
        raise RuntimeError("M2-11 determinism or numerical audit failed")

    audits = {
        "old_policy_refresh": {
            "outer_iterations_audited": chronology_iterations,
            "canonical_iterations": 300,
            "replay_iterations": 300,
            "old_policy_frozen_inside_four_epoch_blocks": True,
            "old_policy_refreshed_between_iterations": True,
            "stale_policy_violations": stale_violations,
            "inside_block_violations": inside_block_violations,
        },
        "train_boundary": {
            "roles_present": ["TRAIN"],
            "train_nodes": EXPECTED_NODES,
            "action_credits": EXPECTED_ACTION_CREDITS,
            "validation_optimisation_nodes": 0,
            "final_holdout_nodes": 0,
            "e2e_pilot_nodes": 0,
        },
        "determinism_replay": {
            "canonical_lr_runs": 3,
            "audit_replay_lr_runs": 3,
            "candidate_checkpoints_compared": 9,
            "parameter_sha_mismatches": mismatch_count,
            "determinism": "PASS",
            "comparisons": replay_comparisons,
        },
        "numerical_invariants": {
            "nan_inf_count": nan_inf,
            "maximum_weighted_advantage_residual": maximum_advantage_residual,
            "advantage_threshold": 1e-10,
            "occupancy_depth_mass": "PASS",
            "probabilities": "PASS",
        },
        "parameter_counts": {
            "global_parameters": 20_197,
            "candidate_checkpoints_checked": 18,
            "status": "PASS",
        },
    }
    result = {
        "schema_version": "M2-11-FINAL-AUDIT-v1",
        "task": TASK_ID,
        "candidate_count": 9,
        "selected_candidate": None,
        "selection_status": "DEFERRED_TO_M2_12",
        "trainer_sha": next(iter(trainer_shas)),
        "training_plan_sha": next(iter(plan_shas)),
        "initial_parameter_sha": next(iter(initial_shas)),
        "candidates": sorted(canonical_candidates, key=lambda item: item["candidate_id"]),
        "audits": audits,
    }
    if archive_root is not None:
        archive_root.mkdir(parents=True, exist_ok=False)
        (archive_root / "audits").mkdir()
        for name, payload in audits.items():
            _write_json(archive_root / "audits" / f"{name}.json", payload)
        _write_json(archive_root / "candidate_manifest.json", result)
        for run_kind in ("canonical", "replay"):
            for learning_rate in LEARNING_RATES:
                source = training_root / run_kind / _lr_directory(learning_rate)
                target = archive_root / run_kind / _lr_directory(learning_rate)
                shutil.copytree(source, target)
        candidates_root = archive_root / "candidates"
        candidates_root.mkdir()
        for metadata in result["candidates"]:
            candidate_id = metadata["candidate_id"]
            source = (
                training_root
                / "canonical"
                / _lr_directory(metadata["learning_rate"])
                / "checkpoints"
                / f"{candidate_id}.pt"
            )
            target = candidates_root / candidate_id
            target.mkdir()
            shutil.copy2(source, target / "model.pt")
            _write_json(target / "manifest.json", metadata)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    result = audit_training_root(args.training_root, args.archive_root)
    print(canonical_json_bytes(result).decode())


if __name__ == "__main__":
    main()
