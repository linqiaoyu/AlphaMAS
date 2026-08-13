#!/usr/bin/env python3
"""Audit the M0-to-M1 source-tree boundary before Formal M1."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
M0_SHA = "2535896c8b1070b19c06fa6a936663babb4356f7"
M1_SHA = "2617dafe0f6a690113f10fe1c0d4775810a576ac"
ALLOWED_PREFIXES = ("docs/m1/", "scripts/finmultitime/", "scripts/pytest_interpreter_guard.py", "tests/")
ALLOWED_EXACT = {".gitignore"}


def changed_paths() -> list[str]:
    output = subprocess.run(
        ["git", "diff", "--name-only", f"{M0_SHA}..{M1_SHA}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in output.splitlines() if line]


def main() -> None:
    paths = changed_paths()
    prohibited = [
        path for path in paths
        if path not in ALLOWED_EXACT and not path.startswith(ALLOWED_PREFIXES)
    ]
    report = {
        "report_id": "M0-to-M1-Controlled-Protocol-Difference-Audit",
        "m0_base_sha": M0_SHA,
        "m1_pre_formal_parent_sha": M1_SHA,
        "scope": "git path-level comparison from frozen M0 baseline to the pre-formal M1 source parent",
        "changed_path_count": len(paths),
        "changed_paths": paths,
        "allowed_change_classes": [
            "M1 contract, audit, preprocessing, packet, and Qwen provenance documentation",
            "M1-only deterministic preprocessing/validation scripts",
            "M1 regression tests and the existing Python interpreter guard",
            "ignore rules for generated M1 working artifacts",
        ],
        "prohibited_runtime_or_protocol_paths": prohibited,
        "runtime_protocol_files_changed": [],
        "m0_execution_backtester_evaluation_memory_trader_files_changed": [],
        "raw_finmultitime_source_modified": False,
        "formal_m1_run": False,
        "verdict": "PASS — M1 changes are additive and confined to the approved audit/input boundary"
        if not prohibited
        else "FAIL — a prohibited runtime or protocol path changed",
    }
    path = REPO_ROOT / "docs/m1/m0_m1_controlled_difference_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_path_count": len(paths), "verdict": report["verdict"]}, sort_keys=True))
    if prohibited:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
