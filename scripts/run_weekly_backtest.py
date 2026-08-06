#!/usr/bin/env python3
"""Non-interactive weekly backtest entry point (TradingAgents is opt-in)."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.backtesting.cache import DecisionCache  # noqa: E402
from tradingagents.backtesting.calendar import ExchangeSchedule  # noqa: E402
from tradingagents.backtesting.data import InMemoryDataProvider, YFinanceDataProvider  # noqa: E402
from tradingagents.backtesting.engine import WeeklyBacktestEngine  # noqa: E402
from tradingagents.backtesting.metrics import compute_metrics  # noqa: E402
from tradingagents.backtesting.recorder import (  # noqa: E402
    BacktestRecorder,
    equal_weight_aggregate,
    write_json,
)
from tradingagents.backtesting.strategies import (  # noqa: E402
    BuyAndHoldStrategy,
    ScriptedStrategy,
    SMAStrategy,
    TradingAgentsStrategy,
)
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", choices=("scripted", "buy-and-hold", "sma", "tradingagents"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true", help="offline deterministic data")
    parser.add_argument("--output-root", default="results/backtests")
    return parser.parse_args()


def synthetic_provider(symbols: list[str], schedule: ExchangeSchedule, start: str, end: str) -> InMemoryDataProvider:
    sessions = schedule.sessions(start, end).tz_localize(None)
    frames = {}
    for number, symbol in enumerate([*symbols, "SPY"]):
        close = pd.Series(
            [90.0 + number * 10 + index for index in range(len(sessions))],
            index=sessions, dtype=float,
        )
        frames[symbol] = pd.DataFrame({
            "Open": close - 0.25, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": 1_000_000,
        })
    return InMemoryDataProvider(frames)


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    strategy_name = args.strategy or config["strategy"]
    schedule = ExchangeSchedule(config["calendar"])
    events = schedule.weekly_events(config["first_calendar_week"], config["final_calendar_week"])
    if config.get("decision_weeks") != len(events):
        raise ValueError("configured decision week count does not match XNYS schedule")
    output = Path(args.output_root) / args.experiment_id
    total_cases = len(events) * len(config["symbols"])
    planned_cases = min(total_cases, args.max_cases) if args.max_cases is not None else total_cases
    if planned_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    dry = {
        "experiment_id": args.experiment_id,
        "symbols": config["symbols"],
        "decision_sessions": len(events),
        "estimated_cases": planned_cases,
        "requires_llm": strategy_name == "tradingagents",
        "output_path": str(output),
        "schedule": [event.to_dict() for event in events],
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return 0
    if output.exists() and not args.force and not args.resume:
        raise FileExistsError(f"output already exists: {output}; use --resume or --force")
    data_start = (
        datetime.fromisoformat(config["first_calendar_week"]) - timedelta(days=400)
    ).date().isoformat()
    provider = (
        synthetic_provider(config["symbols"], schedule, data_start, config["final_valuation_session"])
        if args.synthetic_data else YFinanceDataProvider()
    )
    results = {}
    remaining_cases = planned_cases
    for symbol in config["symbols"]:
        if strategy_name == "scripted":
            strategy = ScriptedStrategy({events[0].decision_session: "BUY"})
        elif strategy_name == "buy-and-hold":
            strategy = BuyAndHoldStrategy()
        elif strategy_name == "sma":
            strategy = SMAStrategy()
        else:
            graph_config = dict(DEFAULT_CONFIG)
            graph_config.update({
                "memory_mode": config["memory_mode"],
                "historical_memory_dir": str(output / "memory"),
                "data_cache_dir": str(output / "runtime_cache"),
                "results_dir": str(output / "agent_results"),
                "max_debate_rounds": 2 if config["research_depth"] == "medium" else 1,
                "max_risk_discuss_rounds": 2 if config["research_depth"] == "medium" else 1,
            })
            for key in (
                "llm_provider", "quick_think_llm", "deep_think_llm", "temperature",
            ):
                if key in config:
                    graph_config[key] = config[key]
            graph = TradingAgentsGraph(config=graph_config)
            strategy = TradingAgentsStrategy(
                graph, reports_root=output / "cases",
                cache=DecisionCache(output / "decision_cache"),
                cache_config={
                    "git_commit_sha": subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False,
                        text=True, capture_output=True,
                    ).stdout.strip(),
                    "prompt_config_version": "weekly-backtest-v1",
                    "memory_namespace_version": "experiment-v1",
                },
                force=args.force,
            )
        engine = WeeklyBacktestEngine(
            data_provider=provider, schedule=schedule, initial_cash=config["initial_cash"],
            commission_bps=config["commission_bps"], slippage_bps=config["slippage_bps"],
            fractional_shares=config["fractional_shares"],
        )
        result = engine.run(
            symbol=symbol, first_week=config["first_calendar_week"],
            final_week=config["final_calendar_week"],
            final_valuation_session=config["final_valuation_session"], strategy=strategy,
            experiment_id=args.experiment_id, warmup_start=data_start,
            max_decisions=min(len(events), remaining_cases),
        )
        remaining_cases = max(0, remaining_cases - len(events))
        results[symbol] = result
    benchmark_engine = WeeklyBacktestEngine(
        data_provider=provider, schedule=schedule, initial_cash=config["initial_cash"],
        commission_bps=config["commission_bps"], slippage_bps=config["slippage_bps"],
        fractional_shares=config["fractional_shares"],
    )
    spy_benchmark = benchmark_engine.run(
        symbol="SPY", first_week=config["first_calendar_week"],
        final_week=config["final_calendar_week"],
        final_valuation_session=config["final_valuation_session"],
        strategy=BuyAndHoldStrategy(), experiment_id=f"{args.experiment_id}-benchmark-spy",
        warmup_start=data_start,
    )
    for symbol, result in results.items():
        stock_benchmark = benchmark_engine.run(
            symbol=symbol, first_week=config["first_calendar_week"],
            final_week=config["final_calendar_week"],
            final_valuation_session=config["final_valuation_session"],
            strategy=BuyAndHoldStrategy(),
            experiment_id=f"{args.experiment_id}-benchmark-{symbol}", warmup_start=data_start,
        )
        result.metrics.update({
            "same_stock_buy_and_hold_cumulative_return": stock_benchmark.metrics["cumulative_return"],
            "spy_cumulative_return": spy_benchmark.metrics["cumulative_return"],
            "excess_return_vs_same_stock_buy_and_hold": (
                result.metrics["cumulative_return"] - stock_benchmark.metrics["cumulative_return"]
            ),
            "excess_return_vs_spy": (
                result.metrics["cumulative_return"] - spy_benchmark.metrics["cumulative_return"]
            ),
            "buy_and_hold_first_fill_time": stock_benchmark.fills.iloc[0]["execution_time"],
        })
        BacktestRecorder(output / symbol).save(
            result, {}, [event.to_dict() for event in events]
        )
        write_json(output / symbol / "benchmarks.json", {
            "same_stock_buy_and_hold": stock_benchmark.metrics,
            "SPY": spy_benchmark.metrics,
        })
    aggregate = equal_weight_aggregate(results)
    aggregate_dir = output / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_dir / "equal_weight_equity.csv", index=False)
    aggregate_metrics = compute_metrics(aggregate["normalized_equity"])
    write_json(aggregate_dir / "equal_weight_metrics.json", aggregate_metrics)
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False,
        text=True, capture_output=True,
    ).stdout.strip()
    manifest = {
        **dry, "git_sha": git_sha, "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "calendar": config["calendar"], "date_contract": {
            "first_week": config["first_calendar_week"],
            "final_week": config["final_calendar_week"],
            "final_valuation_session": config["final_valuation_session"],
        },
        "decision_rule": "last XNYS session close of each calendar week",
        "execution_rule": "next XNYS session open",
        "initial_cash_per_symbol": config["initial_cash"],
        "commission_bps": config["commission_bps"], "slippage_bps": config["slippage_bps"],
        "fractional_shares": config["fractional_shares"], "memory_mode": config["memory_mode"],
        "seed": config["seed"], "python": platform.python_version(),
    }
    write_json(output / "manifest.json", manifest)
    pd.DataFrame([event.to_dict() for event in events]).to_csv(output / "schedule.csv", index=False)
    for attribute, filename in (
        ("decisions", "decisions.csv"), ("orders", "orders.csv"),
        ("fills", "fills.csv"), ("daily_equity", "daily_equity.csv"),
    ):
        combined = []
        for symbol, result in results.items():
            frame = getattr(result, attribute).copy()
            frame.insert(0, "account_symbol", symbol)
            combined.append(frame)
        pd.concat(combined, ignore_index=True).to_csv(output / filename, index=False)
    pd.DataFrame([
        {"symbol": symbol, **result.metrics} for symbol, result in results.items()
    ]).to_csv(output / "summary.csv", index=False)
    with (output / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for symbol, result in results.items():
            failures = result.decisions.loc[result.decisions["status"] == "failed"]
            for failure in failures.to_dict("records"):
                handle.write(json.dumps({"symbol": symbol, **failure}, default=str) + "\n")
    print(json.dumps({"status": "success", "output_path": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
