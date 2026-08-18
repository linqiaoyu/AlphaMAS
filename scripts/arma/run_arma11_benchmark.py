#!/usr/bin/env python3
"""Execute the one allowed official fixed ARMA11_2024H1 run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.backtesting.arma11 import ARMA11Strategy  # noqa: E402
from tradingagents.backtesting.artifacts import (  # noqa: E402
    aggregate_outputs,
    analysis_ready,
    artifact_schema,
    save_strategy,
)
from tradingagents.backtesting.calendar import ExchangeSchedule  # noqa: E402
from tradingagents.backtesting.data import CSVSnapshotDataProvider  # noqa: E402
from tradingagents.backtesting.engine import BacktestResult, WeeklyBacktestEngine  # noqa: E402
from tradingagents.backtesting.recorder import write_json  # noqa: E402

STARTING_EXPERIMENTS_SHA = "0313ac4024655527b1b6936de14c5e11c03e0c64"
REQUIRED_BRANCH = "compare-with-adft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "backtest_arma11_2024h1.json"),
    )
    parser.add_argument("--experiments-repo", required=True)
    parser.add_argument("--preformal-audit", required=True)
    return parser.parse_args()


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(root: Path, symbol: str) -> BacktestResult:
    def table(name: str) -> pd.DataFrame:
        path = root / name
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return BacktestResult(
        symbol=symbol,
        decisions=table("decisions.csv"),
        orders=table("orders.csv"),
        fills=table("fills.csv"),
        daily_equity=table("daily_equity.csv"),
        corporate_action_events=table("corporate_action_events.csv"),
        metrics=json.loads((root / "metrics.json").read_text(encoding="utf-8")),
    )


def environment(source_sha: str) -> dict[str, Any]:
    versions = {}
    for package in ("statsmodels", "numpy", "scipy", "pandas", "exchange-calendars"):
        versions[package] = importlib.metadata.version(package)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_sha": source_sha,
        "dependency_versions": versions,
        "deepseek_calls": 0,
        "qwen_calls": 0,
        "aws_ec2_starts": 0,
        "gpu_hours": 0,
        "llm_cost": 0,
    }


def forecast_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, result in results.items():
        for _, decision in result.decisions.iterrows():
            metadata = decision["metadata"]
            params = metadata.get("fitted_parameters", {})
            forecasts = metadata.get("forecast_daily_returns", [None] * 5)
            forecasts = [*forecasts, *([None] * (5 - len(forecasts)))]
            rows.append({
                "symbol": symbol,
                "decision_session": decision["decision_session"],
                "input_start_session": metadata.get("input_start_session"),
                "input_end_session": metadata.get("input_end_session"),
                "n_returns": metadata.get("input_observation_count"),
                "input_sha256": metadata.get("return_series_sha256"),
                "fit_status": metadata.get("fit_status"),
                "converged": metadata.get("converged"),
                "ar1": params.get("ar.L1"),
                "ma1": params.get("ma.L1"),
                "constant": params.get("const"),
                "sigma2": params.get("sigma2"),
                **{f"forecast_r{index + 1}": forecasts[index] for index in range(5)},
                "forecast_5d_cumulative": metadata.get("forecast_5d_cumulative"),
                "target_weight": decision.get("target_weight"),
                "action": decision.get("action"),
                "model_failure": metadata.get("model_failure", False),
            })
    return pd.DataFrame(rows)


def decision_provenance(
    results: dict[str, BacktestResult], schedule: pd.DataFrame
) -> list[dict[str, Any]]:
    schedule_by_decision = schedule.set_index("decision_session")
    rows: list[dict[str, Any]] = []
    for symbol, result in results.items():
        for _, decision in result.decisions.iterrows():
            session = decision["decision_session"]
            orders = result.orders
            order = None
            if not orders.empty:
                matches = orders.loc[orders["created_at"].astype(str).str[:10] == session]
                if not matches.empty:
                    order = matches.iloc[0].to_dict()
            fill = None
            if order is not None and not result.fills.empty:
                matches = result.fills.loc[result.fills["order_id"] == order["order_id"]]
                if not matches.empty:
                    fill = matches.iloc[0].to_dict()
            rows.append({
                "symbol": symbol,
                "decision_session": session,
                "decision_time_utc": decision["decision_time_utc"],
                **decision["metadata"],
                "action": decision["action"],
                "target_weight": decision["target_weight"],
                "rebalance_status": decision.get("rebalance_status", ""),
                "next_execution_session": schedule_by_decision.loc[session][
                    "execution_session"
                ],
                "execution_provenance": {
                    "order": order,
                    "fill": fill,
                    "no_order_reason": "unchanged target" if order is None else None,
                },
            })
    return rows


def validate(
    *,
    config: dict[str, Any],
    results: dict[str, BacktestResult],
    schedule: pd.DataFrame,
    snapshots: dict[str, str],
    expected_snapshots: dict[str, str],
    forecasts: pd.DataFrame,
) -> dict[str, Any]:
    expected_sessions = set(schedule["decision_session"].astype(str))
    all_decisions = pd.concat(
        [result.decisions.assign(account_symbol=symbol) for symbol, result in results.items()],
        ignore_index=True,
    )
    future_violations = 0
    wrong_windows = 0
    malformed_forecasts = 0
    accounting_errors: list[str] = []
    execution_errors: list[str] = []
    for symbol, result in results.items():
        for _, row in result.decisions.iterrows():
            metadata = row["metadata"]
            if metadata.get("input_end_session") is not None:
                future_violations += int(
                    metadata["input_end_session"] > row["decision_session"]
                )
                wrong_windows += int(metadata.get("input_observation_count") != 252)
            if not metadata.get("model_failure"):
                malformed_forecasts += int(
                    len(metadata.get("forecast_daily_returns", [])) != 5
                    or not np.isfinite(metadata.get("forecast_5d_cumulative"))
                )
        schedule_map = schedule.set_index("decision_session")["execution_session"]
        for _, order in result.orders.iterrows():
            created = str(order["created_at"])[:10]
            if str(order["intended_execution_session"]) != str(schedule_map.loc[created]):
                execution_errors.append(f"{symbol}:{created}: wrong execution session")
        for _, fill in result.fills.iterrows():
            if float(fill["slippage_bps"]) != config["slippage_bps"]:
                execution_errors.append(f"{symbol}:{fill['order_id']}: wrong slippage")
            order = result.orders.set_index("order_id").loc[fill["order_id"]]
            if str(fill["execution_time"])[:10] != str(order["intended_execution_session"]):
                execution_errors.append(f"{symbol}:{fill['order_id']}: not next-open session")
        daily = result.daily_equity
        identity = (
            config["initial_cash"]
            + daily["realized_pnl"].astype(float)
            + daily["unrealized_pnl"].astype(float)
            + daily["cumulative_dividends"].astype(float)
        )
        if not np.allclose(daily["equity"].astype(float), identity, rtol=0, atol=1e-8):
            accounting_errors.append(f"{symbol}: equity/P&L identity failed")
        if (daily["cash"].astype(float) < -1e-8).any():
            accounting_errors.append(f"{symbol}: negative cash")
        if (daily["quantity"].astype(float) < -1e-8).any():
            accounting_errors.append(f"{symbol}: short position")
        if daily.iloc[-1]["session"] != config["final_valuation_session"]:
            accounting_errors.append(f"{symbol}: incorrect final valuation session")
    per_symbol = {symbol: len(result.decisions) for symbol, result in results.items()}
    duplicates = int(
        all_decisions.duplicated(["account_symbol", "decision_session"]).sum()
    )
    checks = {
        "snapshot_identities_exact": snapshots == expected_snapshots,
        "population_78": len(all_decisions) == 78,
        "population_per_symbol": per_symbol == dict.fromkeys(config["symbols"], 26),
        "schedule_population_exact": all(
            set(result.decisions["decision_session"].astype(str)) == expected_sessions
            for result in results.values()
        ),
        "no_duplicate_cases": duplicates == 0,
        "exact_252_model_inputs": wrong_windows == 0,
        "no_future_model_inputs": future_violations == 0,
        "exact_five_step_forecasts": malformed_forecasts == 0,
        "analysis_ready_forecast_rows": len(forecasts) == 78,
        "execution_protocol": not execution_errors,
        "accounting_protocol": not accounting_errors,
        "no_paid_model_usage": True,
        "no_finmultitime_or_mas_runtime": True,
        "no_forced_final_close": True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "case_count": len(all_decisions),
        "decisions_per_symbol": per_symbol,
        "duplicate_cases": duplicates,
        "model_failure_count": int(forecasts["model_failure"].astype(bool).sum()),
        "future_input_violations": future_violations,
        "wrong_input_window_count": wrong_windows,
        "malformed_forecasts": malformed_forecasts,
        "execution_errors": execution_errors,
        "accounting_errors": accounting_errors,
    }


def execute(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiments = Path(args.experiments_repo).resolve()
    preformal_audit_path = Path(args.preformal_audit).resolve()
    preformal_audit = json.loads(preformal_audit_path.read_text(encoding="utf-8"))
    if preformal_audit.get("status") != "passed" or any(
        preformal_audit.get(key) != expected
        for key, expected in (
            ("case_count", 78),
            ("exact_252_windows", 78),
            ("future_input_violations", 0),
            ("insufficient_windows", 0),
        )
    ):
        raise ValueError("the frozen pre-Formal structural audit is not acceptable")
    source_sha = git(REPO_ROOT, "rev-parse", "HEAD")
    if git(REPO_ROOT, "branch", "--show-current") != REQUIRED_BRANCH:
        raise ValueError(f"official ARMA run requires branch {REQUIRED_BRANCH}")
    if git(REPO_ROOT, "status", "--porcelain"):
        raise ValueError("official ARMA run requires a clean source worktree")
    if git(experiments, "branch", "--show-current") != "main":
        raise ValueError("official ARMA archive requires Experiments main")
    if git(experiments, "status", "--porcelain"):
        raise ValueError("official ARMA run requires a clean Experiments worktree")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_EXPERIMENTS_SHA, "HEAD"],
        cwd=experiments,
        check=True,
    )
    for frozen in ("M0", "M1", "M2", "A1", "A2"):
        changed = git(
            experiments,
            "diff",
            "--name-only",
            f"{STARTING_EXPERIMENTS_SHA}...HEAD",
            "--",
            f"experiments/{frozen}",
        )
        if changed:
            raise ValueError(f"prior frozen {frozen} archive changed: {changed}")

    formal_root = experiments / "experiments" / "ARMA" / "formal"
    official = formal_root / "official_run"
    sentinel = formal_root / "OFFICIAL_RUN_STARTED.json"
    if formal_root.exists() or sentinel.exists() or official.exists():
        raise FileExistsError(
            "an ARMA Formal archive or start sentinel already exists; automatic rerun is forbidden"
        )
    formal_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    write_json(sentinel, {
        "experiment_id": config["experiment_id"],
        "status": "started",
        "started_at": started.isoformat(),
        "source_sha": source_sha,
        "rerun_policy": "forbidden unless a proven correctness failure invalidates this run",
    })
    official.mkdir(parents=True, exist_ok=False)
    snapshot_dir = experiments / config["snapshot_path_relative_to_experiments"]
    benchmark_root = experiments / config["benchmark_path_relative_to_experiments"]
    snapshot_hashes = {
        symbol: sha256(snapshot_dir / f"{symbol}.csv")
        for symbol in [*config["symbols"], "SPY"]
    }
    if snapshot_hashes != config["market_snapshot_sha256"]:
        raise ValueError("Formal frozen snapshot identities do not match")
    schedule_utility = ExchangeSchedule(config["calendar"])
    events = schedule_utility.weekly_events(
        config["first_calendar_week"], config["final_calendar_week"]
    )
    schedule = pd.DataFrame([event.to_dict() for event in events])
    provider = CSVSnapshotDataProvider(snapshot_dir)
    engine = WeeklyBacktestEngine(
        data_provider=provider,
        schedule=schedule_utility,
        initial_cash=config["initial_cash"],
        commission_bps=config["commission_bps"],
        slippage_bps=config["slippage_bps"],
        fractional_shares=config["fractional_shares"],
        risk_free_rate=config["risk_free_rate"],
        annualization=config["annualization"],
    )
    results: dict[str, BacktestResult] = {}
    clock_start = time.perf_counter()
    for symbol in config["symbols"]:
        results[symbol] = engine.run(
            symbol=symbol,
            first_week=config["first_calendar_week"],
            final_week=config["final_calendar_week"],
            final_valuation_session=config["final_valuation_session"],
            strategy=ARMA11Strategy(),
            experiment_id=config["experiment_id"],
            warmup_start="2023-01-04",
        )
    wall_clock_seconds = time.perf_counter() - clock_start

    frozen_benchmarks = {
        symbol: load_result(benchmark_root / f"{symbol}_buy_and_hold", symbol)
        for symbol in config["symbols"]
    }
    spy = load_result(benchmark_root / "SPY_buy_and_hold", "SPY")
    equal_weight_bh_metrics = json.loads(
        (benchmark_root / "equal_weight_buy_and_hold_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    for symbol, result in results.items():
        benchmark_metrics = frozen_benchmarks[symbol].metrics
        result.metrics.update({
            "same_stock_buy_and_hold_cumulative_return": benchmark_metrics[
                "cumulative_return"
            ],
            "excess_return_vs_same_stock_buy_and_hold": (
                result.metrics["cumulative_return"]
                - benchmark_metrics["cumulative_return"]
            ),
            "spy_cumulative_return": spy.metrics["cumulative_return"],
            "excess_return_vs_spy": (
                result.metrics["cumulative_return"] - spy.metrics["cumulative_return"]
            ),
        })
    aggregate, aggregate_metrics = aggregate_outputs(results)
    aggregate_metrics.update({
        "equal_weight_buy_and_hold_cumulative_return": equal_weight_bh_metrics[
            "cumulative_return"
        ],
        "excess_return_vs_equal_weight_buy_and_hold": (
            aggregate_metrics["cumulative_return"]
            - equal_weight_bh_metrics["cumulative_return"]
        ),
    })

    schedule.to_csv(official / "schedule.csv", index=False, lineterminator="\n")
    shutil.copy2(config_path, official / "formal_config.json")
    resolved_config = {
        **config,
        "config_sha256": sha256(config_path),
        "source_sha": source_sha,
        "experiments_parent_sha": git(experiments, "rev-parse", "HEAD"),
        "official_run_started_at": started.isoformat(),
        "actual_cpu_wall_clock_seconds": wall_clock_seconds,
    }
    write_json(official / "config.resolved.json", resolved_config)
    env = environment(source_sha)
    env["cpu_wall_clock_seconds"] = wall_clock_seconds
    write_json(official / "environment.json", env)
    schema = artifact_schema()
    schema["arma11"] = {
        "forecast_table": "analysis_ready/arma_forecasts.csv",
        "decision_provenance": "provenance/decision_provenance.jsonl",
        "input_return": config["input"],
        "model": config["model"],
    }
    write_json(official / "artifact_schema.json", schema)
    input_market = official / "inputs" / "market_data"
    action_root = official / "inputs" / "corporate_actions"
    input_market.mkdir(parents=True)
    action_root.mkdir(parents=True)
    data_entries = []
    market_data: dict[str, pd.DataFrame] = {}
    for symbol, digest in snapshot_hashes.items():
        shutil.copy2(snapshot_dir / f"{symbol}.csv", input_market / f"{symbol}.csv")
        frame = provider.load(symbol, "2023-01-04", config["final_valuation_session"])
        market_data[symbol] = frame
        actions = frame[["Dividends", "Stock Splits"]].reset_index(names="Date")
        actions["Date"] = pd.to_datetime(actions["Date"]).dt.strftime("%Y-%m-%d")
        actions.to_csv(action_root / f"{symbol}.csv", index=False, lineterminator="\n")
        data_entries.append({
            "symbol": symbol,
            "sha256": digest,
            "source": "frozen M0 formal raw-price snapshot",
            "row_count": len(frame),
            "first_session": frame.index[0].date().isoformat(),
            "last_session": frame.index[-1].date().isoformat(),
        })
    write_json(official / "inputs" / "data_manifest.json", {"symbols": data_entries})

    save_strategy(official / "strategy", results)
    archived_benchmarks = official / "benchmarks"
    archived_benchmarks.mkdir()
    for symbol in config["symbols"]:
        shutil.copytree(
            benchmark_root / f"{symbol}_buy_and_hold",
            archived_benchmarks / f"{symbol}_buy_and_hold",
        )
    shutil.copytree(
        benchmark_root / "SPY_buy_and_hold", archived_benchmarks / "SPY_buy_and_hold"
    )
    for filename in (
        "equal_weight_buy_and_hold_equity.csv",
        "equal_weight_buy_and_hold_metrics.json",
    ):
        shutil.copy2(benchmark_root / filename, archived_benchmarks / filename)
    aggregate_root = official / "aggregate"
    aggregate_root.mkdir()
    aggregate.to_csv(
        aggregate_root / "equal_weight_equity.csv", index=False, lineterminator="\n"
    )
    write_json(aggregate_root / "equal_weight_metrics.json", aggregate_metrics)
    analysis_ready(
        official / "analysis_ready",
        experiment_id=config["experiment_id"],
        run_id=config["experiment_id"],
        results=results,
        stock_benchmarks=frozen_benchmarks,
        spy=spy,
        aggregate=aggregate,
        aggregate_metrics=aggregate_metrics,
        market_data=market_data,
        case_index=[],
        data_availability=[],
        llm_usage=[],
    )
    forecasts = forecast_table(results)
    forecasts.to_csv(
        official / "analysis_ready" / "arma_forecasts.csv",
        index=False,
        lineterminator="\n",
    )
    provenance = decision_provenance(results, schedule)
    provenance_root = official / "provenance"
    provenance_root.mkdir()
    with (provenance_root / "decision_provenance.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in provenance:
            handle.write(json.dumps(row, default=str, allow_nan=False) + "\n")
    warnings_root = official / "fit_warnings"
    warnings_root.mkdir()
    with (warnings_root / "fit_warnings.jsonl").open("w", encoding="utf-8") as handle:
        for row in provenance:
            for warning in row.get("fit_warnings", []):
                handle.write(json.dumps({
                    "symbol": row["symbol"],
                    "decision_session": row["decision_session"],
                    "warning": warning,
                }) + "\n")
    failures_root = official / "model_failures"
    failures_root.mkdir()
    with (failures_root / "model_failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in provenance:
            if row.get("model_failure"):
                handle.write(json.dumps(row, default=str, allow_nan=False) + "\n")

    validation = validate(
        config=config,
        results=results,
        schedule=schedule,
        snapshots=snapshot_hashes,
        expected_snapshots=config["market_snapshot_sha256"],
        forecasts=forecasts,
    )
    validation.update({
        "preformal_audit_sha256": sha256(preformal_audit_path),
        "source_sha": source_sha,
        "config_sha256": sha256(config_path),
    })
    write_json(official / "validation_report.json", validation)
    completed = datetime.now(timezone.utc)
    manifest = {
        "experiment_id": config["experiment_id"],
        "benchmark_identity": config["benchmark_identity"],
        "official_run": True,
        "first_complete_correctness_valid_run": validation["status"] == "passed",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "source_sha": source_sha,
        "config_sha256": sha256(config_path),
        "market_snapshot_sha256": snapshot_hashes,
        "case_count": validation["case_count"],
        "model_failure_count": validation["model_failure_count"],
        "cpu_wall_clock_seconds": wall_clock_seconds,
        "paid_compute": env,
    }
    write_json(official / "manifest.json", manifest)
    write_json(official / "run_status.json", {
        "experiment_id": config["experiment_id"],
        "status": "success" if validation["status"] == "passed" else "invalid",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
    })
    write_json(sentinel, {
        **json.loads(sentinel.read_text(encoding="utf-8")),
        "status": "completed" if validation["status"] == "passed" else "invalid",
        "completed_at": completed.isoformat(),
        "official_run_path": "official_run",
    })
    readme = f"""# Fixed ARMA(1,1) Formal benchmark\n\n+This directory contains the single official `{config['experiment_id']}` run.\n+The method was frozen before this performance was computed. No LLM, GPU, AWS,\n+FinMultiTime, Memory, or RL runtime was used.\n+\n+- Cases: {validation['case_count']}\n+- Model failures: {validation['model_failure_count']}\n+- CPU wall-clock seconds: {wall_clock_seconds:.6f}\n+- Source: `{source_sha}`\n+- Config SHA256: `{sha256(config_path)}`\n+- Validation: `{validation['status']}`\n+"""
    (formal_root / "README.md").write_text(readme, encoding="utf-8")
    if validation["status"] != "passed":
        raise RuntimeError("official ARMA run failed correctness validation and is invalid")
    return official


def main() -> int:
    output = execute(parse_args())
    print(json.dumps({"status": "success", "official_run": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
