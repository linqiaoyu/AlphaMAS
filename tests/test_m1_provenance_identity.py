from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tradingagents.backtesting.config import (
    compute_graph_config_sha256,
    resolve_graph_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]
PILOT_ARCHIVE_COMMIT = "376a214a9cbd0a650b7e5ac96d6275ae7cb5974a"


def _m1_config() -> dict:
    config = json.loads(
        (REPOSITORY / "configs" / "backtest_m0_2024h1.json").read_text(
            encoding="utf-8"
        )
    )
    config.update(
        {
            "finmultitime_evidence_enabled": True,
            "finmultitime_bundle_scope": "PILOT",
            "finmultitime_archive_commit": PILOT_ARCHIVE_COMMIT,
        }
    )
    return config


def _graph_identity(config: dict, tmp_path: Path) -> str:
    graph = resolve_graph_config(
        config,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )
    return compute_graph_config_sha256(graph)


def test_bundle_scope_is_part_of_graph_research_identity(tmp_path: Path) -> None:
    pilot = _m1_config()
    formal = {**pilot, "finmultitime_bundle_scope": "FORMAL"}

    assert _graph_identity(pilot, tmp_path / "pilot") != _graph_identity(
        formal, tmp_path / "formal"
    )


def test_archive_commit_is_part_of_graph_research_identity(tmp_path: Path) -> None:
    pilot = _m1_config()
    changed = {
        **pilot,
        "finmultitime_archive_commit": "0" * 40,
    }

    assert _graph_identity(pilot, tmp_path / "pinned") != _graph_identity(
        changed, tmp_path / "changed"
    )


def test_invalid_archive_pin_is_absent_from_active_source_tree() -> None:
    invalid = "376a214c267ba8e731cd6b595dbd250ba1" + "a4d0a9"
    matches = []

    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for relative_bytes in tracked:
        if not relative_bytes:
            continue
        path = REPOSITORY / relative_bytes.decode()
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if invalid in content:
            matches.append(str(path.relative_to(REPOSITORY)))

    assert matches == []
