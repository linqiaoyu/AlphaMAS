from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.archive_roots import (
    ExperimentsArchiveUnavailable,
    resolve_experiments_root as resolve_generic_experiments_root,
)

EXPERIMENTS_ROOT_ENV = "ALPHAMAS_EXPERIMENTS_ROOT"
REWARD_STUDY_RELATIVE_PATH = Path("experiments/M2/development/reward_study_v1")
EXPECTED_SELECTED_REWARD = "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"
EXPECTED_MARKET_SNAPSHOT_IDENTITY = (
    "3afc723888666e0ca3a12219a57dc577d6cf3953717da731b774634f4aff1445"
)
EXPECTED_FINAL_HASHES = {
    "manifests/analysis_plan.json": "1e0d1f999ed259f5facb3b0dd8002b6a6e0a7d3eafdca708338b8ba07462e030",
    "results/reward_candidate_summary.json": "cc7412dcc6db0c5ce1947cd870f47a2c1950b542183aca278ff8cb2ae69b0a94",
    "results/reward_cross_symbol_summary.csv": "351f8ee2978623990889b742eb04431dd77411bd4e8ba9c3516efbbcc527acec",
    "results/reward_outcomes.csv": "86723a9acc4c512cde2bf570107ece94b037bf634b24d1a7c1559a9fa18d7711",
    "results/reward_selection.json": "13f421e43957a9a35d8d3673e55f7e4ebf8ef22f7919317d78d0a251c9cccb01",
    "results/reward_state_analysis.csv": "ea4cac13bd30cbfac9d47ec2da1d95ea74246d5b2ab951d4d67da38def28172c",
}


class RewardStudyArchiveUnavailable(RuntimeError):
    """Raised when no archive location was explicitly configured or discovered."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid M2 reward-study archive JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid M2 reward-study archive JSON object: {path}")
    return payload


def _validate_reward_study_root(study_root: Path) -> Path:
    required_paths = (
        Path("inputs/market_snapshot"),
        Path("manifests/market_snapshot_manifest.json"),
        Path("manifests/analysis_plan.json"),
        Path("manifests/final_sha256.json"),
        Path("results/reward_outcomes.csv"),
        Path("results/reward_state_analysis.csv"),
        Path("results/reward_candidate_summary.json"),
        Path("results/reward_cross_symbol_summary.csv"),
        Path("results/reward_selection.json"),
    )
    missing = [str(path) for path in required_paths if not (study_root / path).exists()]
    if missing:
        raise ValueError(
            f"Invalid M2 reward-study archive at {study_root}; missing: {', '.join(missing)}"
        )
    if not (study_root / "inputs/market_snapshot").is_dir():
        raise ValueError(
            f"Invalid M2 reward-study archive at {study_root}; "
            "inputs/market_snapshot is not a directory"
        )

    selection_path = study_root / "results/reward_selection.json"
    selection = _read_json(selection_path)
    if selection.get("selected_reward_id") != EXPECTED_SELECTED_REWARD:
        raise ValueError(
            f"Invalid M2 reward-study archive at {study_root}; selected_reward_id is not "
            f"{EXPECTED_SELECTED_REWARD}"
        )

    snapshot_manifest = _read_json(study_root / "manifests/market_snapshot_manifest.json")
    if (
        snapshot_manifest.get("reward_market_snapshot_identity_sha256")
        != EXPECTED_MARKET_SNAPSHOT_IDENTITY
    ):
        raise ValueError(
            f"Invalid M2 reward-study archive at {study_root}; market snapshot identity differs"
        )

    final_manifest = _read_json(study_root / "manifests/final_sha256.json")
    if final_manifest.get("files") != EXPECTED_FINAL_HASHES:
        raise ValueError(
            f"Invalid M2 reward-study archive at {study_root}; final SHA256 manifest differs"
        )
    mismatches = [
        relative_path
        for relative_path, expected in EXPECTED_FINAL_HASHES.items()
        if _sha256(study_root / relative_path) != expected
    ]
    if mismatches:
        raise ValueError(
            f"Invalid M2 reward-study archive at {study_root}; hash mismatch: "
            f"{', '.join(mismatches)}"
        )
    return study_root


def resolve_experiments_root(repository_root: Path) -> Path:
    """Resolve the AlphaMAS-Experiments root without network or regeneration."""
    try:
        experiments_root = resolve_generic_experiments_root(repository_root)
    except ExperimentsArchiveUnavailable as exc:
        raise RewardStudyArchiveUnavailable(
            "M2 reward-study archive unavailable. Set ALPHAMAS_EXPERIMENTS_ROOT or "
            "provide a sibling AlphaMAS-Experiments checkout."
        ) from exc
    try:
        _validate_reward_study_root(experiments_root / REWARD_STUDY_RELATIVE_PATH)
    except ValueError as exc:
        raise ValueError(
            f"{EXPERIMENTS_ROOT_ENV} is explicitly set to an invalid "
            f"AlphaMAS-Experiments root: {experiments_root}"
        ) from exc
    return experiments_root


def resolve_reward_study_root(repository_root: Path) -> Path:
    return resolve_experiments_root(repository_root) / REWARD_STUDY_RELATIVE_PATH
