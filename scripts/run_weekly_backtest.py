#!/usr/bin/env python3
"""Non-interactive weekly backtest entry point (TradingAgents is opt-in)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
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
    collect_agent_analysis_records,
    result_hashes,
    save_result,
    save_strategy,
    validate_artifact_bundle,
)
from tradingagents.backtesting.cache import DecisionCache  # noqa: E402
from tradingagents.backtesting.calendar import ExchangeSchedule  # noqa: E402
from tradingagents.backtesting.config import (  # noqa: E402
    compute_graph_config_sha256,
    resolve_graph_config,
    resolve_research_rounds,
    sanitize_graph_config,
    validate_fixed_backtest_contract,
    validate_formal_m0_config,
)
from tradingagents.backtesting.data import (  # noqa: E402
    CSVSnapshotDataProvider,
    InMemoryDataProvider,
    YFinanceDataProvider,
    canonical_market_csv,
)
from tradingagents.backtesting.engine import (  # noqa: E402
    WeeklyBacktestEngine,
    market_history_visibility_errors,
)
from tradingagents.backtesting.llm_usage import LLMUsageCallback  # noqa: E402
from tradingagents.backtesting.memory_archive import (  # noqa: E402
    archive_final_experiment_memory,
)
from tradingagents.backtesting.memory_lineage import (  # noqa: E402
    backtest_protocol_sha256,
    select_memory_lineage,
    snapshot_input_identity,
    validate_resume_data_source,
)
from tradingagents.backtesting.metrics import transaction_cost_components  # noqa: E402
from tradingagents.backtesting.recorder import write_json  # noqa: E402
from tradingagents.backtesting.strategies import (  # noqa: E402
    BuyAndHoldStrategy,
    DecisionChronology,
    ScriptedStrategy,
    SMAStrategy,
    TradingAgentsStrategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", choices=("scripted", "buy-and-hold", "sma", "tradingagents"))
    parser.add_argument("--experiment-id", required=True)
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--resume", action="store_true")
    lifecycle.add_argument("--force", action="store_true")
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


def implementation_identity(git_sha: str) -> str:
    """Identify HEAD plus dirty backtester/Graph code that can affect memory."""
    tracked_diff = git_value(
        "diff", "--binary", "HEAD", "--", "scripts", "tradingagents"
    )
    untracked = git_value(
        "ls-files", "--others", "--exclude-standard", "--",
        "scripts", "tradingagents",
    ).splitlines()
    if not tracked_diff and not untracked:
        return git_sha or "nogit"
    digest = hashlib.sha256(tracked_diff.encode("utf-8"))
    for relative in sorted(untracked):
        path = REPO_ROOT / relative
        if path.is_file():
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
    return f"{git_sha or 'nogit'}-dirty-{digest.hexdigest()}"


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
    run_dir: Path, git_sha: str, force: bool, graph_config: dict[str, Any],
    chronology: DecisionChronology | None = None,
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
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    usage_callback = LLMUsageCallback(
        provider=graph_config["llm_provider"],
        thinking_mode=graph_config.get("deepseek_thinking"),
        run_id=run_dir.name,
    )
    graph = TradingAgentsGraph(
        selected_analysts=tuple(config["selected_analysts"]),
        config=copy.deepcopy(graph_config), callbacks=[usage_callback],
    )
    return TradingAgentsStrategy(
        graph, reports_root=run_dir / "strategy" / "cases",
        cache=DecisionCache(experiment_root / "runtime" / "decision_cache"),
        cache_config={
            "git_commit_sha": git_sha, "prompt_config_version": "weekly-backtest-v1",
            "memory_namespace_version": "experiment-v3-run-lineage",
        },
        usage_callback=usage_callback, run_root=run_dir, force=force,
        chronology=chronology,
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
    warmup_metadata: dict[str, Any],
) -> tuple[InMemoryDataProvider, dict[str, pd.DataFrame], dict[str, Any]]:
    if source == "synthetic":
        source_provider: Any = synthetic_provider(symbols, schedule, requested_start, requested_end)
    elif source == "snapshot":
        if not snapshot_dir:
            raise ValueError("--snapshot-dir is required with --data-source snapshot")
        source_provider = CSVSnapshotDataProvider(snapshot_dir)
    elif source == "yfinance":
        source_provider = YFinanceDataProvider()
    else:
        raise ValueError(f"unsupported market data source: {source!r}")
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
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        **warmup_metadata,
        "symbols": entries,
    }
    write_json(run_dir / "inputs" / "data_manifest.json", manifest)
    return InMemoryDataProvider(frames), frames, manifest


def run_accounts(
    *, symbols: list[str], provider: Any, schedule: ExchangeSchedule,
    config: dict[str, Any], strategy_name: str, experiment_id: str,
    experiment_root: Path, run_dir: Path, git_sha: str, events: list[Any],
    data_start: str, planned_cases: int, force: bool,
    graph_config: dict[str, Any],
) -> dict[str, Any]:
    results = {}
    remaining = planned_cases
    # One rolling prefix spans every symbol in this attempt. This binds an
    # AMZN/JPM cache entry to all preceding AAPL Agent decisions, so a repaired
    # earlier failure invalidates stale downstream decisions across accounts.
    chronology = DecisionChronology() if strategy_name == "tradingagents" else None
    for symbol in symbols:
        strategy = strategy_factory(
            strategy_name, first_decision=events[0].decision_session, config=config,
            experiment_root=experiment_root, run_dir=run_dir, git_sha=git_sha, force=force,
            graph_config=graph_config, chronology=chronology,
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


PORTFOLIO_PNL_IDENTITY_ATOL = 1e-8


def portfolio_pnl_identity_errors(
    *, results: dict[str, Any], stock_benchmarks: dict[str, Any], spy: Any,
    initial_cash: float, atol: float = PORTFOLIO_PNL_IDENTITY_ATOL,
) -> list[str]:
    """Validate net P&L and recomputable cost reporting for every account."""
    accounts = [
        *((f"strategy/{symbol}", result) for symbol, result in results.items()),
        *((f"benchmark/{symbol}", result) for symbol, result in stock_benchmarks.items()),
        ("benchmark/SPY", spy),
    ]
    required_columns = {
        "session",
        "equity",
        "realized_pnl",
        "unrealized_pnl",
        "cumulative_dividends",
        "cumulative_cost",
        "cumulative_commission_cost",
        "cumulative_slippage_cost",
        "cumulative_transaction_cost",
    }
    errors: list[str] = []
    for account_name, result in accounts:
        daily = result.daily_equity
        missing = required_columns - set(daily.columns)
        if missing:
            errors.append(
                f"{account_name}: daily_equity lacks {sorted(missing)}"
            )
            continue
        if daily.empty:
            errors.append(f"{account_name}: daily_equity is empty")
            continue
        for _, row in daily.iterrows():
            values = {
                column: row[column]
                for column in required_columns - {"session"}
            }
            if not all(
                isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value)
                for value in values.values()
            ):
                errors.append(
                    f"{account_name}:{row['session']}: P&L identity contains non-finite values"
                )
                continue
            expected_equity = (
                float(initial_cash)
                + float(row["realized_pnl"])
                + float(row["unrealized_pnl"])
                + float(row["cumulative_dividends"])
            )
            if not np.isclose(
                float(row["equity"]), expected_equity, rtol=0.0, atol=atol
            ):
                errors.append(
                    f"{account_name}:{row['session']}: equity {row['equity']} != "
                    f"initial_cash + realized_pnl + unrealized_pnl + "
                    f"cumulative_dividends ({expected_equity}); "
                    f"difference={float(row['equity']) - expected_equity}"
                )
        cost_columns = (
            "cumulative_commission_cost",
            "cumulative_slippage_cost",
            "cumulative_transaction_cost",
        )
        for column in cost_columns:
            costs = daily[column].astype(float).to_numpy()
            if (np.diff(costs) < -atol).any():
                errors.append(f"{account_name}: {column} decreases")
        reported_total = daily["cumulative_transaction_cost"].astype(float).to_numpy()
        component_total = (
            daily["cumulative_commission_cost"].astype(float).to_numpy()
            + daily["cumulative_slippage_cost"].astype(float).to_numpy()
        )
        if not np.allclose(reported_total, component_total, rtol=0.0, atol=atol):
            errors.append(
                f"{account_name}: cumulative transaction cost != commission + slippage"
            )
        compatibility_total = daily["cumulative_cost"].astype(float).to_numpy()
        if not np.allclose(
            compatibility_total, reported_total, rtol=0.0, atol=atol
        ):
            errors.append(
                f"{account_name}: cumulative_cost alias != cumulative_transaction_cost"
            )
        try:
            fill_costs = transaction_cost_components(result.fills)
        except ValueError as exc:
            errors.append(f"{account_name}: fills transaction costs invalid ({exc})")
            continue
        fill_totals = {
            "cumulative_commission_cost": float(fill_costs["commission_cost"].sum()),
            "cumulative_slippage_cost": float(fill_costs["slippage_cost"].sum()),
            "cumulative_transaction_cost": float(
                fill_costs["total_transaction_cost"].sum()
            ),
        }
        for column, fill_total in fill_totals.items():
            final_cost = float(daily.iloc[-1][column])
            if not np.isclose(final_cost, fill_total, rtol=0.0, atol=atol):
                errors.append(
                    f"{account_name}: final {column} {final_cost} != recomputed fills "
                    f"{fill_total}; difference={final_cost - fill_total}"
                )
        metric_costs = {
            "total_commission_cost": fill_totals["cumulative_commission_cost"],
            "total_slippage_cost": fill_totals["cumulative_slippage_cost"],
            "total_transaction_cost": fill_totals["cumulative_transaction_cost"],
        }
        for metric, fill_total in metric_costs.items():
            value = result.metrics.get(metric)
            if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isclose(
                float(value), fill_total, rtol=0.0, atol=atol
            ):
                errors.append(
                    f"{account_name}: metric {metric} {value!r} != recomputed fills "
                    f"{fill_total}"
                )
        expected_rates = {
            "commission_cost_rate": metric_costs["total_commission_cost"] / initial_cash,
            "slippage_cost_rate": metric_costs["total_slippage_cost"] / initial_cash,
            "transaction_cost_rate": metric_costs["total_transaction_cost"] / initial_cash,
        }
        for metric, expected_rate in expected_rates.items():
            value = result.metrics.get(metric)
            if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isclose(
                float(value), expected_rate, rtol=0.0, atol=atol
            ):
                errors.append(
                    f"{account_name}: metric {metric} {value!r} != {expected_rate}"
                )
    return errors


def validate_run(
    *, results: dict[str, Any], stock_benchmarks: dict[str, Any], spy: Any,
    replay_results: dict[str, Any] | None, events: list[Any], config: dict[str, Any],
    data_manifest: dict[str, Any], strategy_name: str, run_dir: Path,
    expected_decisions: int, aggregate: pd.DataFrame,
) -> dict[str, Any]:
    validation_schedule = ExchangeSchedule(config["calendar"])
    expected_sessions = [
        item.date().isoformat() for item in validation_schedule.sessions(
            events[0].decision_session, config["final_valuation_session"]
        )
    ]
    market_history_errors = market_history_visibility_errors(
        results, validation_schedule
    )
    portfolio_pnl_errors = portfolio_pnl_identity_errors(
        results=results,
        stock_benchmarks=stock_benchmarks,
        spy=spy,
        initial_cash=float(config["initial_cash"]),
    )
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
        "no_future_market_data_visible": not market_history_errors,
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
        "portfolio_pnl_identity": not portfolio_pnl_errors,
        "split_accounting_tested_offline": True,
        "metrics_finite_or_null": all(
            value is None or isinstance(value, (str, dict)) or np.isfinite(value)
            for item in [*results.values(), *stock_benchmarks.values(), spy]
            for value in item.metrics.values()
        ),
        "artifact_input_checksums_valid": input_hashes_valid,
        "configured_warmup_sessions_honored": (
            data_manifest.get("configured_warmup_sessions") == config["warmup_sessions"]
            and data_manifest.get("actual_warmup_sessions") == config["warmup_sessions"]
            and data_manifest.get("first_decision_session") == events[0].decision_session
        ),
        "agent_decisions_successful": (
            all(
                "status" in item.decisions
                and not (item.decisions["status"] == "failed").any()
                for item in results.values()
                if not item.decisions.empty
            )
            if strategy_name == "tradingagents" else None
        ),
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
        "checks": checks,
        "market_history_visibility_errors": market_history_errors,
        "portfolio_pnl_identity_errors": portfolio_pnl_errors,
        "tolerance": {
            "floating_point": 1e-12,
            "portfolio_pnl_identity_absolute": PORTFOLIO_PNL_IDENTITY_ATOL,
        },
        "snapshot_hashes": {entry["symbol"]: entry["sha256"]
                            for entry in data_manifest["symbols"]},
        "result_hashes": primary_hashes, "replay_result_hashes": replay_hashes,
    }


def execute(args: argparse.Namespace) -> tuple[Path, str]:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    formal_config_path = (REPO_ROOT / "configs" / "backtest_m0_2024h1.json").resolve()
    if config_path == formal_config_path:
        validate_formal_m0_config(config)
    validate_fixed_backtest_contract(config)
    strategy_name = args.strategy or config["strategy"]
    source = "synthetic" if args.synthetic_data else (
        args.data_source or config.get("data_source", "yfinance")
    )
    if config_path == formal_config_path and strategy_name == "tradingagents" and source != "snapshot":
        raise ValueError("formal M0 TradingAgents execution requires data_source=snapshot")
    schedule = ExchangeSchedule(config["calendar"])
    events = schedule.weekly_events(config["first_calendar_week"], config["final_calendar_week"])
    if config.get("decision_weeks") != len(events):
        raise ValueError("configured decision week count does not match XNYS schedule")
    rounds = resolve_research_rounds(config["research_depth"])
    if (
        int(config["max_debate_rounds"]) != rounds
        or int(config["max_risk_discuss_rounds"]) != rounds
    ):
        raise ValueError("research depth and configured debate/risk rounds disagree")
    warmup = schedule.preceding_sessions(
        events[0].decision_session, config["warmup_sessions"]
    )
    data_start = warmup[0].date().isoformat()
    warmup_metadata = {
        "configured_warmup_sessions": int(config["warmup_sessions"]),
        "actual_warmup_sessions": len(warmup),
        "warmup_first_session": data_start,
        "warmup_last_session": warmup[-1].date().isoformat(),
        "first_decision_session": events[0].decision_session,
    }
    total_cases = len(events) * len(config["symbols"])
    planned_cases = min(total_cases, args.max_cases) if args.max_cases is not None else total_cases
    if planned_cases < 0:
        raise ValueError("--max-cases must be non-negative")
    experiment_root = Path(args.output_root) / args.experiment_id
    git_sha = git_value("rev-parse", "HEAD")
    protocol_mode = (
        "formal_m0"
        if (
            config_path == formal_config_path
            and strategy_name == "tradingagents"
            and source == "snapshot"
            and args.max_cases is None
        )
        else "engineering_validation" if config_path == formal_config_path else "custom"
    )
    decision_sessions = [event.decision_session for event in events]
    dry = {
        "experiment_id": args.experiment_id, "symbols": config["symbols"],
        "protocol_mode": protocol_mode,
        "decision_sessions": len(events), "decision_weeks": len(events),
        "expected_paid_agent_cases": total_cases, "estimated_cases": planned_cases,
        "planned_cases": planned_cases,
        "requires_llm": strategy_name == "tradingagents", "data_source": source,
        "snapshot_required": source == "snapshot",
        "snapshot_dir_supplied": bool(args.snapshot_dir),
        "snapshot_requirement": (
            "snapshot required for execution" if source == "snapshot" else None
        ),
        "research_depth": config["research_depth"],
        "debate_rounds": rounds,
        "risk_rounds": rounds,
        "llm_provider": config["llm_provider"],
        "quick_model": config["quick_think_llm"],
        "deep_model": config["deep_think_llm"],
        "thinking_mode": config["deepseek_thinking"],
        "temperature": float(config["temperature"]),
        "selected_analysts": config["selected_analysts"],
        "key_dates": {
            "first_decision": events[0].decision_session,
            "good_friday_week_decision": (
                "2024-03-28" if "2024-03-28" in decision_sessions else None
            ),
            "last_decision": events[-1].decision_session,
            "first_execution": events[0].execution_session,
            "last_execution": events[-1].execution_session,
            "final_valuation": config["final_valuation_session"],
        },
        "warmup": warmup_metadata,
        "will_execute": False if args.dry_run else None,
        "do_not_execute_paid_agent_cases": bool(args.dry_run and strategy_name == "tradingagents"),
        "output_path": str(experiment_root), "schedule": [event.to_dict() for event in events],
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return experiment_root, "dry-run"

    if source == "snapshot" and not args.snapshot_dir:
        raise ValueError("--snapshot-dir is required with --data-source snapshot")
    validate_resume_data_source(source, resume=bool(args.resume))
    if protocol_mode == "formal_m0" and git_value("status", "--porcelain"):
        raise ValueError("formal M0 execution requires a clean git worktree")

    market_input_identity = (
        snapshot_input_identity(args.snapshot_dir, config["symbols"])
        if source == "snapshot" else None
    )
    run_id = new_run_id(git_sha)
    run_dir = experiment_root / "runs" / run_id
    effective_config = {**config, "strategy": strategy_name, "data_source": source}
    graph_config = resolve_graph_config(
        effective_config,
        results_dir=run_dir / "strategy" / "agent_results",
        data_cache_dir=experiment_root / "runtime" / "data_cache",
        historical_memory_dir=experiment_root / "runtime" / "memory",
    )
    graph_config_sha256 = compute_graph_config_sha256(graph_config)
    code_identity = implementation_identity(git_sha)
    protocol_sha256 = backtest_protocol_sha256(
        effective_config,
        planned_cases=planned_cases,
        market_input_identity=market_input_identity,
        implementation_identity=code_identity,
    )
    memory_lineage = select_memory_lineage(
        experiment_root=experiment_root,
        experiment_id=args.experiment_id,
        run_id=run_id,
        graph_config_sha256=graph_config_sha256,
        backtest_protocol_sha256=protocol_sha256,
        resume=bool(args.resume),
        force=bool(args.force),
    )
    graph_config["historical_memory_lineage_id"] = memory_lineage.lineage_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    status_path = run_dir / "run_status.json"
    running_status = {
        "experiment_id": args.experiment_id, "run_id": run_id,
        "status": "running", "started_at": started,
        **memory_lineage.to_dict(),
    }
    write_json(status_path, running_status)
    # Publish the running attempt immediately so an interrupted process can be
    # selected explicitly by the next --resume. The run_path is informational;
    # resume reconstructs the path from the validated run_id.
    write_json(experiment_root / "latest.json", {
        **running_status, "run_path": str(run_dir),
    })
    try:
        provider, market_data, data_manifest = materialize_inputs(
            source, args.snapshot_dir, config["symbols"], schedule, data_start,
            config["final_valuation_session"], run_dir, warmup_metadata,
        )
        resolved = {
            **config, "experiment_id": args.experiment_id, "run_id": run_id,
            "strategy": strategy_name, "data_source": source,
            "protocol_mode": protocol_mode,
            "requested_data_start": data_start,
            **warmup_metadata,
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
            "graph_config_sha256": graph_config_sha256,
            "backtest_protocol_sha256": protocol_sha256,
            "implementation_identity": code_identity,
            "market_input_identity": market_input_identity,
            "resume_requested": bool(args.resume),
            **memory_lineage.to_dict(),
        }
        write_json(run_dir / "config.resolved.json", resolved)
        write_json(run_dir / "graph_config.resolved.json", sanitize_graph_config(graph_config))
        write_json(run_dir / "environment.json", environment(git_sha))
        write_json(run_dir / "artifact_schema.json", artifact_schema())
        pd.DataFrame([event.to_dict() for event in events]).to_csv(
            run_dir / "schedule.csv", index=False
        )
        manifest = {
            **dry, "run_id": run_id, "git_sha": git_sha, "started_at": started,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "graph_config_sha256": graph_config_sha256,
            "backtest_protocol_sha256": protocol_sha256,
            "implementation_identity": code_identity,
            "market_input_identity": market_input_identity,
            **memory_lineage.to_dict(),
            "resume_semantics": (
                "replay from the latest incomplete run using its exact memory "
                "lineage and lineage-scoped DecisionCache identity"
            ),
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
            graph_config=graph_config,
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
        if strategy_name == "tradingagents":
            case_index, data_availability, llm_usage = collect_agent_analysis_records(
                args.experiment_id, run_id, results,
                run_dir / "strategy" / "cases", graph_config,
            )
        else:
            case_index, data_availability, llm_usage = [], [], []
        analysis_ready(
            run_dir / "analysis_ready", experiment_id=args.experiment_id, run_id=run_id,
            results=results, stock_benchmarks=stock_benchmarks, spy=spy,
            aggregate=aggregate, aggregate_metrics=aggregate_metrics, market_data=market_data,
            case_index=case_index, data_availability=data_availability,
            llm_usage=llm_usage,
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
                planned_cases=planned_cases, force=False, graph_config=graph_config,
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
        if strategy_name == "tradingagents":
            if graph_config.get("memory_mode") != "experiment":
                raise ValueError(
                    "successful TradingAgents artifact publication requires "
                    "experiment Memory"
                )
            archived_symbols = [
                symbol for symbol, result in results.items()
                if not result.decisions.empty
            ]
            if not archived_symbols:
                raise ValueError(
                    "successful TradingAgents artifact publication requires "
                    "at least one completed Agent decision"
                )
            manifest["memory_archive"] = archive_final_experiment_memory(
                run_dir=run_dir,
                runtime_memory_dir=graph_config["historical_memory_dir"],
                experiment_id=args.experiment_id,
                run_id=run_id,
                memory_lineage_id=memory_lineage.lineage_id,
                memory_lifecycle=memory_lineage.lifecycle,
                memory_resumed_from_run_id=memory_lineage.resumed_from_run_id,
                graph_config_sha256=graph_config_sha256,
                symbols=archived_symbols,
                finmultitime_evidence_enabled=bool(
                    graph_config.get("finmultitime_evidence_enabled", False)
                ),
                finmultitime_bundle_scope=graph_config.get("finmultitime_bundle_scope"),
                finmultitime_bundle_identity=graph_config.get(
                    "finmultitime_expected_input_bundle_identity"
                ),
            )
        completed = datetime.now(timezone.utc).isoformat()
        manifest["run_completed_at"] = completed
        write_json(run_dir / "manifest.json", manifest)
        final_status = {
            "experiment_id": args.experiment_id, "run_id": run_id, "status": "success",
            "started_at": started, "completed_at": completed,
            **memory_lineage.to_dict(),
        }
        write_json(status_path, final_status)
        bundle_validation = validate_artifact_bundle(
            run_dir, symbols=config["symbols"], strategy_name=strategy_name,
            require_success=True,
        )
        validation["artifact_bundle"] = bundle_validation
        if bundle_validation["status"] != "passed":
            validation["status"] = "failed"
            write_json(validation_root / "validation_report.json", validation)
            raise RuntimeError(
                "artifact completeness validation failed; see validation_report.json"
            )
        write_json(validation_root / "validation_report.json", validation)
        write_json(experiment_root / "latest.json", {
            **final_status, "run_path": str(run_dir),
        })
        return run_dir, "success"
    except Exception as exc:
        failed = {
            "experiment_id": args.experiment_id, "run_id": run_id, "status": "failed",
            "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
            **memory_lineage.to_dict(),
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
