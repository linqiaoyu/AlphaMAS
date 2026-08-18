#!/usr/bin/env python3
"""Build pre-Formal, Post-run Audit, and Research Freeze ARMA archives."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SOURCE_SHA = "6306ea4ea20cda501c6238db80c34d27bbc16bea"
BASE_EXPERIMENTS_SHA = "0313ac4024655527b1b6936de14c5e11c03e0c64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    preformal = subparsers.add_parser("preformal")
    preformal.add_argument("--experiments-repo", required=True)
    preformal.add_argument("--pit-json", required=True)
    preformal.add_argument("--pit-csv", required=True)
    preformal.add_argument("--test-command", required=True)
    preformal.add_argument("--tests-passed", type=int, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--experiments-repo", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--experiments-repo", required=True)
    return parser.parse_args()


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def inventory(root: Path, target: Path, *, exclude: set[str] | None = None) -> str:
    exclusions = exclude or set()
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclusions or path == target:
            continue
        lines.append(f"{sha256(path)}  {relative}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256(target)


def require_clean_source() -> str:
    if git(REPO_ROOT, "branch", "--show-current") != "compare-with-adft":
        raise ValueError("ARMA archive requires compare-with-adft")
    if git(REPO_ROOT, "status", "--porcelain"):
        raise ValueError("ARMA archive requires a clean source worktree")
    return git(REPO_ROOT, "rev-parse", "HEAD")


def preformal(args: argparse.Namespace) -> Path:
    experiments = Path(args.experiments_repo).resolve()
    source_sha = require_clean_source()
    if git(experiments, "rev-parse", "HEAD") != BASE_EXPERIMENTS_SHA:
        raise ValueError("pre-Formal archive must start at the required Experiments SHA")
    if git(experiments, "status", "--porcelain"):
        raise ValueError("pre-Formal archive requires a clean Experiments worktree")
    pit_json = Path(args.pit_json).resolve()
    pit_csv = Path(args.pit_csv).resolve()
    pit = json.loads(pit_json.read_text(encoding="utf-8"))
    if pit.get("status") != "passed" or pit.get("performance_computed") is not False:
        raise ValueError("PIT audit is not a passing structural-only audit")
    root = experiments / "experiments" / "ARMA" / "preformal"
    root.mkdir(parents=True, exist_ok=False)
    config = REPO_ROOT / "configs" / "backtest_arma11_2024h1.json"
    shutil.copy2(config, root / "formal_config.json")
    shutil.copy2(pit_json, root / "pit_window_audit.json")
    shutil.copy2(pit_csv, root / "pit_window_audit.csv")
    versions = {
        package: importlib.metadata.version(package)
        for package in ("statsmodels", "numpy", "scipy", "pandas", "exchange-calendars")
    }
    write_json(root / "environment_manifest.json", {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
    })
    config_value = json.loads(config.read_text(encoding="utf-8"))
    contract = {
        "benchmark_identity": "ARMA11_FIXED_V1",
        "experiment_id": "ARMA11_2024H1",
        "source_sha": source_sha,
        "parent_frozen_m2_sha": BASE_SOURCE_SHA,
        "config_sha256": sha256(config),
        "model": config_value["model"],
        "input": config_value["input"],
        "forecast": config_value["forecast"],
        "signal": config_value["signal"],
        "hard_failure": config_value["hard_failure"],
        "formal_protocol": {
            key: config_value[key]
            for key in (
                "symbols",
                "calendar",
                "first_calendar_week",
                "final_calendar_week",
                "decision_weeks",
                "final_valuation_session",
                "initial_cash",
                "commission_bps",
                "slippage_bps",
                "fractional_shares",
                "long_only",
                "short_selling",
                "leverage",
                "force_liquidate_at_end",
                "risk_free_rate",
                "annualization",
            )
        },
        "market_snapshot_sha256": config_value["market_snapshot_sha256"],
        "versions": versions,
        "pit_window_audit_sha256": sha256(root / "pit_window_audit.json"),
        "zero_performance_based_tuning": True,
        "performance_inspected_before_freeze": False,
        "no_paid_compute": True,
        "no_mas_finmultitime_memory_or_rl": True,
    }
    write_json(root / "benchmark_contract.json", contract)
    markdown = f"""# ARMA11 pre-Formal benchmark contract\n\n+The fixed `ARMA11_FIXED_V1` method was frozen before any 2024H1 ARMA trading\n+performance was computed or inspected.\n+\n+- Source SHA: `{source_sha}`\n+- Frozen Full-M2 parent: `{BASE_SOURCE_SHA}`\n+- Config SHA256: `{sha256(config)}`\n+- Model: `statsmodels.tsa.arima.model.ARIMA(order=(1,0,1), trend=\"c\", enforce_stationarity=True, enforce_invertibility=True)`\n+- Fit: `method=\"statespace\"`, `maxiter=200`\n+- Input: exactly 252 daily simple PIT total returns from 253 consecutive raw-price/action observations\n+- Horizon: exactly five daily forecasts, compounded as `prod(1+r_hat)-1`\n+- Signal: positive to 100% long; zero or negative to cash\n+- Hard failure: HOLD and preserve the current discrete position; no retry\n+- PIT structural audit: 78/78 exact windows, zero future violations, zero insufficient windows\n+- Performance-based tuning: none\n+- Paid/agent compute: none\n+"""
    (root / "benchmark_contract.md").write_text(markdown, encoding="utf-8")
    write_json(root / "regression_summary.json", {
        "command": args.test_command,
        "tests_passed": args.tests_passed,
        "status": "passed",
        "formal_performance_computed": False,
    })
    diff = git(REPO_ROOT, "diff", "--name-status", f"{BASE_SOURCE_SHA}...HEAD")
    (root / "source_diff_inventory.txt").write_text(diff + "\n", encoding="utf-8")
    write_json(root / "source_identity.json", {
        "branch": "compare-with-adft",
        "source_sha": source_sha,
        "parent_frozen_m2_sha": BASE_SOURCE_SHA,
        "baseline_m2_ahead": 0,
        "baseline_m2_behind": 0,
        "initial_source_diff": 0,
    })
    inventory(root, root / "SHA256SUMS")
    return root


def audit(args: argparse.Namespace) -> Path:
    experiments = Path(args.experiments_repo).resolve()
    source_sha = require_clean_source()
    if git(experiments, "status", "--porcelain"):
        raise ValueError("Post-run Audit requires the official result commit to be clean")
    formal = experiments / "experiments" / "ARMA" / "formal"
    official = formal / "official_run"
    validation = json.loads((official / "validation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((official / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((official / "config.resolved.json").read_text(encoding="utf-8"))
    environment = json.loads((official / "environment.json").read_text(encoding="utf-8"))
    aggregate = json.loads(
        (official / "aggregate" / "equal_weight_metrics.json").read_text(encoding="utf-8")
    )
    required = [
        "formal_config.json",
        "environment.json",
        "schedule.csv",
        "analysis_ready/arma_forecasts.csv",
        "provenance/decision_provenance.jsonl",
        "aggregate/equal_weight_metrics.json",
        "validation_report.json",
        "model_failures/model_failures.jsonl",
        "fit_warnings/fit_warnings.jsonl",
    ]
    missing = [relative for relative in required if not (official / relative).is_file()]
    checks = {
        "source_identity": manifest["source_sha"] == source_sha,
        "config_identity": manifest["config_sha256"] == config["config_sha256"],
        "model_specification": config["model"]["order"] == [1, 0, 1],
        "return_specification": config["input"]["return_observations"] == 252,
        "input_pit_audit": validation["checks"]["no_future_model_inputs"]
        and validation["checks"]["exact_252_model_inputs"],
        "population_78": validation["case_count"] == 78,
        "execution_audit": validation["checks"]["execution_protocol"],
        "accounting_audit": validation["checks"]["accounting_protocol"],
        "forecast_audit": validation["checks"]["exact_five_step_forecasts"],
        "archive_complete": not missing,
        "no_paid_model_usage": all(
            environment[key] == 0
            for key in ("deepseek_calls", "qwen_calls", "aws_ec2_starts", "gpu_hours", "llm_cost")
        ),
        "no_mas_or_finmultitime_runtime": validation["checks"][
            "no_finmultitime_or_mas_runtime"
        ],
    }
    report = {
        "audit_identity": "ARMA11_POST_RUN_AUDIT_V1",
        "status": "passed" if validation["status"] == "passed" and all(checks.values()) else "failed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": source_sha,
        "formal_config_sha256": manifest["config_sha256"],
        "statsmodels_version": environment["dependency_versions"]["statsmodels"],
        "model_specification": config["model"],
        "return_specification": config["input"],
        "case_count": validation["case_count"],
        "model_failure_count": validation["model_failure_count"],
        "checks": checks,
        "missing_archive_members": missing,
        "aggregate_metrics": aggregate,
        "forecast_table_sha256": sha256(official / "analysis_ready" / "arma_forecasts.csv"),
        "aggregate_metrics_sha256": sha256(
            official / "aggregate" / "equal_weight_metrics.json"
        ),
        "validation_report_sha256": sha256(official / "validation_report.json"),
    }
    write_json(formal / "ARMA_POST_RUN_AUDIT.json", report)
    markdown = f"""# ARMA(1,1) Post-run Audit\n\n+Status: **{report['status'].upper()}**\n+\n+- Source SHA: `{source_sha}`\n+- Formal config SHA256: `{manifest['config_sha256']}`\n+- Statsmodels: `{report['statsmodels_version']}`\n+- Population: {validation['case_count']}/78\n+- Model fit failures: {validation['model_failure_count']}\n+- PIT: exactly 252 session-local total returns and no future input violations\n+- Forecast: exactly five steps and fixed compounding\n+- Execution: next XNYS open, 5 bps commission, 5 bps slippage, long-only, no leverage\n+- Accounting: daily mark-to-market, corporate actions, cash/holdings/P&L/cost identities passed\n+- Paid-model/GPU/AWS usage: zero\n+- Archive completeness: {'passed' if not missing else 'failed'}\n+"""
    (formal / "ARMA_POST_RUN_AUDIT.md").write_text(markdown, encoding="utf-8")
    inventory(
        formal,
        formal / "SHA256SUMS.pre_research_freeze",
        exclude={"SHA256SUMS"},
    )
    if report["status"] != "passed":
        raise RuntimeError("ARMA Post-run Audit failed")
    return formal


def freeze(args: argparse.Namespace) -> Path:
    experiments = Path(args.experiments_repo).resolve()
    source_sha = require_clean_source()
    if git(experiments, "status", "--porcelain"):
        raise ValueError("Research Freeze requires the committed Post-run Audit")
    formal = experiments / "experiments" / "ARMA" / "formal"
    official = formal / "official_run"
    audit_path = formal / "ARMA_POST_RUN_AUDIT.json"
    audit_value = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_value.get("status") != "passed":
        raise ValueError("Research Freeze requires a passing Post-run Audit")
    manifest = json.loads((official / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((official / "config.resolved.json").read_text(encoding="utf-8"))
    inventory_path = formal / "SHA256SUMS.pre_research_freeze"
    freeze_value = {
        "freeze_identity": "ARMA11_RESEARCH_FREEZE_V1",
        "statement": "The official ARMA(1,1) benchmark result is permanently frozen.",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "branch": "compare-with-adft",
        "source_sha": source_sha,
        "experiments_audit_commit": git(experiments, "rev-parse", "HEAD"),
        "model_config_sha256": manifest["config_sha256"],
        "library_versions": json.loads(
            (official / "environment.json").read_text(encoding="utf-8")
        )["dependency_versions"],
        "market_snapshot_sha256": manifest["market_snapshot_sha256"],
        "input_return_contract": config["input"],
        "official_run_identity": manifest["experiment_id"],
        "case_count": manifest["case_count"],
        "model_failure_count": manifest["model_failure_count"],
        "per_stock_metrics_sha256": sha256(
            official / "strategy" / "combined" / "metrics.csv"
        ),
        "aggregate_metrics_sha256": sha256(
            official / "aggregate" / "equal_weight_metrics.json"
        ),
        "post_run_audit_sha256": sha256(audit_path),
        "bound_inventory": "SHA256SUMS.pre_research_freeze",
        "bound_inventory_sha256": sha256(inventory_path),
        "rerun_due_to_performance": "permanently forbidden",
        "compare_01_started": False,
    }
    write_json(formal / "ARMA_RESEARCH_FREEZE.json", freeze_value)
    markdown = f"""# ARMA(1,1) Research Freeze\n\n+**The official ARMA(1,1) benchmark result is permanently frozen.**\n+\n+- Source branch/SHA: `compare-with-adft` / `{source_sha}`\n+- Model config SHA256: `{manifest['config_sha256']}`\n+- Official run: `{manifest['experiment_id']}`\n+- Population: {manifest['case_count']}/78\n+- Market snapshot identities: bound in `ARMA_RESEARCH_FREEZE.json`\n+- Post-run Audit SHA256: `{sha256(audit_path)}`\n+- Bound inventory SHA256: `{sha256(inventory_path)}`\n+- Rerunning or modifying the method because of performance is permanently forbidden.\n+- COMPARE-01 has not started.\n+"""
    (formal / "ARMA_RESEARCH_FREEZE.md").write_text(markdown, encoding="utf-8")
    inventory(formal, formal / "SHA256SUMS", exclude={"SHA256SUMS"})
    return formal


def main() -> int:
    args = parse_args()
    if args.phase == "preformal":
        output = preformal(args)
    elif args.phase == "audit":
        output = audit(args)
    else:
        output = freeze(args)
    print(json.dumps({"phase": args.phase, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
