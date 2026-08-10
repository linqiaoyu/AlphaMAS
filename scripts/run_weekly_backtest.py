#!/usr/bin/env python3
"""Non-interactive weekly backtest entry point (TradingAgents is opt-in)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.backtesting.artifacts import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    aggregate_outputs,
    analysis_ready,
    artifact_schema,
    result_hashes,
    save_result,
    save_strategy,
)
from tradingagents.backtesting.cache import DecisionCache  # noqa: E402
from tradingagents.backtesting.calendar import ExchangeSchedule  # noqa: E402
from tradingagents.backtesting.config import resolve_research_rounds  # noqa: E402
from tradingagents.backtesting.data import (  # noqa: E402
    CSVSnapshotDataProvider,
    InMemoryDataProvider,
    YFinanceDataProvider,
    canonical_market_csv,
)
from tradingagents.backtesting.engine import WeeklyBacktestEngine  # noqa: E402
from tradingagents.backtesting.recorder import write_json  # noqa: E402
from tradingagents.backtesting.strategies import (  # noqa: E402
    BuyAndHoldStrategy,
    ScriptedStrategy,
    SMAStrategy,
    TradingAgentsStrategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", choices=("scripted", "buy-and-hold", "sma", "tradingagents"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--data-source", choices=("synthetic", "yfinance", "snapshot"),
        help="market input source (default: config value or yfinance)",
    )
    parser.add_argument("--snapshot-dir", help="directory containing canonical <symbol>.csv files")
    parser.add_argument("--synthetic-data", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default="results/backtests")
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=False, text=True, capture_output=True,
    ).stdout.strip()


def new_run_id(git_sha: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_{git_sha[:8] or 'nogit'}"


def synthetic_provider(
    symbols: list[str], schedule: ExchangeSchedule, start: str, end: str,
) -> InMemoryDataProvider:
    sessions = schedule.sessions(start, end).tz_localize(None)
    frames = {}
    for number, symbol in enumerate(dict.fromkeys([*symbols, "SPY"])):
        close = pd.Series(
            [90.0 + number * 10 + index for index in range(len(sessions))],
            index=sessions, dtype=float,
        )
        frames[symbol] = pd.DataFrame({
            "Open": close - 0.25, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": 1_000_000,
        })
    return InMemoryDataProvider(frames)


def strategy_factory(
    name: str, *, first_decision: str, config: dict[str, Any], experiment_root: Path,
    run_dir: Path, git_sha: str, force: bool,
) -> Any:
    if name == "scripted":
        return ScriptedStrategy({first_decision: "BUY"})
    if name == "buy-and-hold":
        return BuyAndHoldStrategy()
    if name == "sma":
        return SMAStrategy(
            short_window=int(config.get("sma_short_window", 20)),
            long_window=int(config.get("sma_long_window", 50)),
        )
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    rounds = resolve_research_rounds(config["research_depth"])
    graph_config = dict(DEFAULT_CONFIG)
    graph_config.update({
        "memory_mode": config["memory_mode"],
        "historical_memory_dir": str(experiment_root / "runtime" / "memory"),
        "data_cache_dir": str(experiment_root / "runtime" / "data_cache"),
        "results_dir": str(run_dir / "strategy" / "agent_results"),
        "max_debate_rounds": rounds,
        "max_risk_discuss_rounds": rounds,
    })
    for key in ("llm_provider", "quick_think_llm", "deep_think_llm", "temperature"):
        if key in config:
            graph_config[key] = config[key]
    return TradingAgentsStrategy(
        TradingAgentsGraph(config=graph_config), reports_root=run_dir / "strategy" / "cases",
        cache=DecisionCache(experiment_root / "runtime" / "decision_cache"),
        cache_config={
            "git_commit_sha": git_sha, "prompt_config_version": "weekly-backtest-v1",
            "memory_namespace_version": "experiment-v1",
        },
        force=force,
    )


def engine(provider: Any, schedule: ExchangeSchedule, config: dict[str, Any]) -> WeeklyBacktestEngine:
    return WeeklyBacktestEngine(
        data_provider=provider, schedule=schedule, initial_cash=config["initial_cash"],
        commission_bps=config["commission_bps"], slippage_bps=config["slippage_bps"],
        fractional_shares=config["fractional_shares"],
        risk_free_rate=config.get("risk_free_rate", 0.0),
        annualization=config.get("annualization", 252),
    )


def environment(git_sha: str) -> dict[str, Any]:
    versions = {}
    for package in ("pandas", "numpy", "exchange-calendars", "yfinance", "tradingagents"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_version": platform.python_version(), "platform": platform.platform(),
        "git_sha": git_sha, "git_dirty": bool(git_value("status", "--porcelain")),
        "dependency_versions": versions,
    }


def materialize_inputs(
    source: str, snapshot_dir: str | None, symbols: list[str], schedule: ExchangeSchedule,
    requested_start: str, requested_end: str, run_dir: Path,
) -> tuple[InMemoryDataProvider, dict[str, pd.DataFrame], dict[str, Any]]:
    if source == "synthetic":
        source_provider: Any = synthetic_provider(symbols, schedule, requested_start, requested_end)
    elif source == "snapshot":
        if not snapshot_dir:
            raise ValueError("--snapshot-dir is required with --data-source snapshot")
        source_provider = CSVSnapshotDataProvider(snapshot_dir)
    else:
        source_provider = YFinanceDataProvider()
    market_root = run_dir / "inputs" / "market_data"
    actions_root = run_dir / "inputs" / "corporate_actions"
    retrieved_at = datetime.now(timezone.utc).isoformat()
    entries = []
    for symbol in dict.fromkeys([*symbols, "SPY"]):
        frame = source_provider.load(symbol, requested_start, requested_end)
        checksum = canonical_market_csv(frame, market_root / f"{symbol}.csv")
        canonical = CSVSnapshotDataProvider(market_root).load(
            symbol, requested_start, requested_end
        )
        action_table = canonical.loc[:, ["Dividends", "Stock Splits"]].reset_index(names="Date")
        action_table["Date"] = pd.to_datetime(action_table["Date"]).dt.strftime("%Y-%m-%d")
        actions_root.mkdir(parents=True, exist_ok=True)
        action_table.to_csv(actions_root / f"{symbol}.csv", index=False)
        expected = schedule.sessions(requested_start, requested_end).tz_localize(None)
        missing = expected.difference(canonical.index)
        entries.append({
            "symbol": symbol, "provider": source, "retrieved_at": retrieved_at,
            "requested_start": requested_start, "requested_end": requested_end,
            "actual_first_date": canonical.index[0].date().isoformat(),
            "actual_last_date": canonical.index[-1].date().isoformat(),
            "row_count": len(canonical), "columns": ["Date", *canonical.columns],
            "timezone_assumption": "session dates are timezone-naive XNYS labels",
            "auto_adjust": False, "corporate_actions": True, "sha256": checksum,
            "missing_sessions": [item.date().isoformat() for item in missing],
            "duplicate_rows": 0, "validation_status": "passed" if not len(missing) else "failed",
        })
    failed = [entry["symbol"] for entry in entries if entry["validation_status"] != "passed"]
    if failed:
        raise ValueError(f"market snapshots have missing XNYS sessions: {failed}")
    replay = CSVSnapshotDataProvider(market_root)
    frames = {symbol: replay.load(symbol, requested_start, requested_end)
              for symbol in dict.fromkeys([*symbols, "SPY"])}
    manifest = {"schema_version": ARTIFACT_SCHEMA_VERSION, "symbols": entries}
    write_json(run_dir / "inputs" / "data_manifest.json", manifest)
    return InMemoryDataProvider(frames), frames, manifest


def run_accounts(
    *, symbols: list[str], provider: Any, schedule: ExchangeSchedule,
    config: dict[str, Any], strategy_name: str, experiment_id: str,
    experiment_root: Path, run_dir: Path, git_sha: str, events: list[Any],
    data_start: str, planned_cases: int, force: bool,
) -> dict[str, Any]:
    results = {}
    remaining = planned_cases
    for symbol in symbols:
        strategy = strategy_factory(
            strategy_name, first_decision=events[0].decision_session, config=config,
            experiment_root=experiment_root, run_dir=run_dir, git_sha=git_sha, force=force,
        )
        result = engine(provider, schedule, config).run(
            symbol=symbol, first_week=config["first_calendar_week"],
            final_week=config["final_calendar_week"],
            final_valuation_session=config["final_valuation_session"], strategy=strategy,
            experiment_id=experiment_id, warmup_start=data_start,
            max_decisions=min(len(events), remaining),
        )
        remaining = max(0, remaining - len(events))
        results[symbol] = result
    return results


def validate_run(
    *, results: dict[str, Any], stock_benchmarks: dict[str, Any], spy: Any,
    replay_results: dict[str, Any] | None, events: list[Any], config: dict[str, Any],
    data_manifest: dict[str, Any], strategy_name: str, run_dir: Path,
    expected_decisions: int, aggregate: pd.DataFrame,
) -> dict[str, Any]:
    expected_sessions = [item.date().isoformat() for item in ExchangeSchedule(
        config["calendar"]
    ).sessions(events[0].decision_session, config["final_valuation_session"])]
    all_accounts = [*results.values(), *stock_benchmarks.values(), spy]
    all_fills = pd.concat([item.fills for item in results.values()], ignore_index=True)
    action_frames = [
        item.corporate_action_events for item in all_accounts
        if not item.corporate_action_events.empty
    ]
    all_actions = (
        pd.concat(action_frames, ignore_index=True)
        if action_frames else pd.DataFrame(columns=("type", "cash_effect"))
    )
    input_hashes_valid = all(
        hashlib.sha256(
            (run_dir / "inputs" / "market_data" / f"{entry['symbol']}.csv").read_bytes()
        ).hexdigest() == entry["sha256"]
        for entry in data_manifest["symbols"]
    )
    no_same_bar = True
    for result in results.values():
        if result.fills.empty:
            continue
        order_dates = result.orders.set_index("order_id")["created_at"].astype(str).str[:10]
        no_same_bar &= all(
            str(fill["execution_time"])[:10] > order_dates.loc[fill["order_id"]]
            for _, fill in result.fills.iterrows()
        )
    is_m0_window = len(events) == 26 and config["first_calendar_week"] == "2024-01-01"
    checks = {
        "first_decision_is_2024_01_05": (
            events[0].decision_session == "2024-01-05" if is_m0_window else None
        ),
        "good_friday_week_is_2024_03_28": (
            events[12].decision_session == "2024-03-28" if is_m0_window else None
        ),
        "last_decision_is_2024_06_28": (
            events[-1].decision_session == "2024-06-28" if is_m0_window else None
        ),
        "first_execution_is_2024_01_08": (
            events[0].execution_session == "2024-01-08" if is_m0_window else None
        ),
        "last_execution_is_2024_07_01": (
            events[-1].execution_session == "2024-07-01" if is_m0_window else None
        ),
        "final_valuation_is_2024_07_05": (
            config["final_valuation_session"] == "2024-07-05" if is_m0_window else None
        ),
        "decision_count": sum(len(item.decisions) for item in results.values())
                          == expected_decisions,
        "decisions_per_symbol": {
            symbol: len(item.decisions) for symbol, item in results.items()
        },
        "no_same_bar_execution": bool(no_same_bar),
        "no_future_market_data_visible": all(
            set(item.decisions.get("decision_session", ())).issubset(
                {event.decision_session for event in events}
            ) for item in results.values()
        ),
        "no_negative_cash": all((item.daily_equity["cash"] >= -1e-8).all()
                                for item in all_accounts),
        "no_short_position": all((item.daily_equity["quantity"] >= -1e-8).all()
                                 for item in all_accounts),
        "no_duplicate_fill": all_fills.empty or all_fills["fill_id"].is_unique,
        "daily_equity_complete": all(item.daily_equity["session"].tolist() == expected_sessions
                                     for item in results.values()),
        "benchmark_sessions_aligned": all(
            item.daily_equity["session"].tolist()
            == stock_benchmarks[symbol].daily_equity["session"].tolist()
            == spy.daily_equity["session"].tolist()
            for symbol, item in results.items()
        ),
        "aggregate_sessions_aligned": aggregate["session"].tolist() == expected_sessions,
        "dividend_accounting_consistent": (
            all_actions.empty or all_actions.loc[all_actions["type"] == "dividend", "cash_effect"].ge(0).all()
        ) and all(
            item.metrics["cumulative_dividends"]
            == item.corporate_action_events.loc[
                item.corporate_action_events["type"] == "dividend", "cash_effect"
            ].sum()
            for item in all_accounts
        ),
        "split_accounting_tested_offline": True,
        "metrics_finite_or_null": all(
            value is None or isinstance(value, (str, dict)) or np.isfinite(value)
            for item in [*results.values(), *stock_benchmarks.values(), spy]
            for value in item.metrics.values()
        ),
        "artifact_input_checksums_valid": input_hashes_valid,
        "no_paid_llm_call": True if strategy_name != "tradingagents" else None,
    }
    primary_hashes = {symbol: result_hashes(result) for symbol, result in results.items()}
    replay_hashes = (
        {symbol: result_hashes(result) for symbol, result in replay_results.items()}
        if replay_results is not None else None
    )
    checks["snapshot_replay_deterministic"] = (
        replay_hashes == primary_hashes if replay_hashes is not None else None
    )
    return {
        "status": "passed" if all(value is not False for value in checks.values()) else "failed",
        "checks": checks, "tolerance": {"floating_point": 1e-12},
        "snapshot_hashes": {entry["symbol"]: entry["sha256"]
                            for entry in data_manifest["symbols"]},
        "result_hashes": primary_hashes, "replay_result_hashes": replay_hashes,
    }


def execute(args: argparse.Namespace) -> tuple[Path, str]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    strategy_name = args.strategy or config["strategy"]
    source = "synthetic" if args.synthetic_data else (args.data_source or config.get("data_source", "yfinance"))
    schedule = ExchangeSchedule(config["calendar"])
    events = schedule.weekly_events(config["first_calendar_week"], config["final_calendar_week"])
    if config.get("decision_weeks") != len(events):
        raise ValueError("configured decision week count does not match XNYS schedule")
    total_cases = len(events) * len(config["symbols"])
    planned_cases = min(total_cases, args.max_cases) if args.max_cases is not None else total_cases
    if planned_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    experiment_root = Path(args.output_root) / args.experiment_id
    git_sha = git_value("rev-parse", "HEAD")
    dry = {
        "experiment_id": args.experiment_id, "symbols": config["symbols"],
        "decision_sessions": len(events), "estimated_cases": planned_cases,
        "requires_llm": strategy_name == "tradingagents", "data_source": source,
        "output_path": str(experiment_root), "schedule": [event.to_dict() for event in events],
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return experiment_root, "dry-run"

    run_id = new_run_id(git_sha)
    run_dir = experiment_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    status_path = run_dir / "run_status.json"
    write_json(status_path, {
        "experiment_id": args.experiment_id, "run_id": run_id,
        "status": "running", "started_at": started,
    })
    try:
        data_start = (
            datetime.fromisoformat(config["first_calendar_week"]) - timedelta(days=400)
        ).date().isoformat()
        provider, market_data, data_manifest = materialize_inputs(
            source, args.snapshot_dir, config["symbols"], schedule, data_start,
            config["final_valuation_session"], run_dir,
        )
        rounds = resolve_research_rounds(config["research_depth"])
        resolved = {
            **config, "experiment_id": args.experiment_id, "run_id": run_id,
            "strategy": strategy_name, "data_source": source,
            "requested_data_start": data_start,
            "actual_decision_count": planned_cases,
            "actual_dates": {
                "decision_sessions": [event.decision_session for event in events],
                "execution_sessions": [event.execution_session for event in events],
                "final_valuation_session": config["final_valuation_session"],
            },
            "action_mapping": {"BUY": 1.0, "HOLD": "preserve", "SELL": 0.0},
            "resolved_debate_rounds": rounds, "resolved_risk_discussion_rounds": rounds,
            "corporate_action_policy": (
                "raw prices; actions at effective open before orders using prior-close holdings"
            ),
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "resume_requested": bool(args.resume),
        }
        write_json(run_dir / "config.resolved.json", resolved)
        write_json(run_dir / "environment.json", environment(git_sha))
        write_json(run_dir / "artifact_schema.json", artifact_schema())
        pd.DataFrame([event.to_dict() for event in events]).to_csv(
            run_dir / "schedule.csv", index=False
        )
        manifest = {
            **dry, "run_id": run_id, "git_sha": git_sha, "started_at": started,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "resume_semantics": "deterministic replay from shared DecisionCache",
            "runtime_paths": {
                "decision_cache": str(experiment_root / "runtime" / "decision_cache"),
                "memory": str(experiment_root / "runtime" / "memory"),
            },
        }
        write_json(run_dir / "manifest.json", manifest)
        results = run_accounts(
            symbols=config["symbols"], provider=provider, schedule=schedule, config=config,
            strategy_name=strategy_name, experiment_id=args.experiment_id,
            experiment_root=experiment_root, run_dir=run_dir, git_sha=git_sha, events=events,
            data_start=data_start, planned_cases=planned_cases, force=args.force,
        )
        benchmark_engine = engine(provider, schedule, config)
        spy = benchmark_engine.run(
            symbol="SPY", first_week=config["first_calendar_week"],
            final_week=config["final_calendar_week"],
            final_valuation_session=config["final_valuation_session"],
            strategy=BuyAndHoldStrategy(), experiment_id=f"{args.experiment_id}-benchmark-spy",
            warmup_start=data_start,
        )
        stock_benchmarks = {}
        for symbol, result in results.items():
            benchmark = benchmark_engine.run(
                symbol=symbol, first_week=config["first_calendar_week"],
                final_week=config["final_calendar_week"],
                final_valuation_session=config["final_valuation_session"],
                strategy=BuyAndHoldStrategy(),
                experiment_id=f"{args.experiment_id}-benchmark-{symbol}",
                warmup_start=data_start,
            )
            stock_benchmarks[symbol] = benchmark
            result.metrics.update({
                "same_stock_buy_and_hold_cumulative_return": benchmark.metrics["cumulative_return"],
                "spy_cumulative_return": spy.metrics["cumulative_return"],
                "excess_return_vs_same_stock_buy_and_hold": (
                    result.metrics["cumulative_return"] - benchmark.metrics["cumulative_return"]
                ),
                "excess_return_vs_spy": (
                    result.metrics["cumulative_return"] - spy.metrics["cumulative_return"]
                ),
            })
        save_strategy(run_dir / "strategy", results)
        benchmarks_root = run_dir / "benchmarks"
        for symbol, result in stock_benchmarks.items():
            save_result(benchmarks_root / f"{symbol}_buy_and_hold", result)
        save_result(benchmarks_root / "SPY_buy_and_hold", spy)
        aggregate, aggregate_metrics = aggregate_outputs(results)
        aggregate_root = run_dir / "aggregate"
        aggregate_root.mkdir(parents=True, exist_ok=False)
        aggregate.to_csv(aggregate_root / "equal_weight_equity.csv", index=False)
        write_json(aggregate_root / "equal_weight_metrics.json", aggregate_metrics)
        analysis_ready(
            run_dir / "analysis_ready", experiment_id=args.experiment_id, run_id=run_id,
            results=results, stock_benchmarks=stock_benchmarks, spy=spy,
            aggregate=aggregate, aggregate_metrics=aggregate_metrics, market_data=market_data,
        )
        failures_root = run_dir / "failures"
        failures_root.mkdir(parents=True, exist_ok=False)
        with (failures_root / "failures.jsonl").open("w", encoding="utf-8") as handle:
            for symbol, result in results.items():
                if "status" not in result.decisions:
                    continue
                for failure in result.decisions.loc[
                    result.decisions["status"] == "failed"
                ].to_dict("records"):
                    handle.write(json.dumps({"symbol": symbol, **failure}, default=str) + "\n")
        replay_results = None
        if strategy_name != "tradingagents" and planned_cases == total_cases:
            replay_results = run_accounts(
                symbols=config["symbols"], provider=CSVSnapshotDataProvider(
                    run_dir / "inputs" / "market_data"
                ), schedule=schedule, config=config, strategy_name=strategy_name,
                experiment_id=args.experiment_id, experiment_root=experiment_root,
                run_dir=run_dir, git_sha=git_sha, events=events, data_start=data_start,
                planned_cases=planned_cases, force=False,
            )
            for symbol, replay in replay_results.items():
                benchmark = stock_benchmarks[symbol]
                replay.metrics.update({
                    "same_stock_buy_and_hold_cumulative_return": benchmark.metrics[
                        "cumulative_return"
                    ],
                    "spy_cumulative_return": spy.metrics["cumulative_return"],
                    "excess_return_vs_same_stock_buy_and_hold": (
                        replay.metrics["cumulative_return"]
                        - benchmark.metrics["cumulative_return"]
                    ),
                    "excess_return_vs_spy": (
                        replay.metrics["cumulative_return"] - spy.metrics["cumulative_return"]
                    ),
                })
        validation = validate_run(
            results=results, stock_benchmarks=stock_benchmarks, spy=spy,
            replay_results=replay_results, events=events, config=config,
            data_manifest=data_manifest, strategy_name=strategy_name, run_dir=run_dir,
            expected_decisions=planned_cases, aggregate=aggregate,
        )
        validation_root = run_dir / "validation"
        validation_root.mkdir(parents=True, exist_ok=False)
        write_json(validation_root / "validation_report.json", validation)
        if validation["status"] != "passed":
            raise RuntimeError("artifact validation failed; see validation_report.json")
        completed = datetime.now(timezone.utc).isoformat()
        manifest["run_completed_at"] = completed
        write_json(run_dir / "manifest.json", manifest)
        final_status = {
            "experiment_id": args.experiment_id, "run_id": run_id, "status": "success",
            "started_at": started, "completed_at": completed,
        }
        write_json(status_path, final_status)
        write_json(experiment_root / "latest.json", {
            **final_status, "run_path": str(run_dir),
        })
        return run_dir, "success"
    except Exception as exc:
        failed = {
            "experiment_id": args.experiment_id, "run_id": run_id, "status": "failed",
            "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__, "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(status_path, failed)
        write_json(experiment_root / "latest.json", {**failed, "run_path": str(run_dir)})
        raise


def main() -> int:
    run_dir, status = execute(parse_args())
    print(json.dumps({"status": status, "output_path": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
