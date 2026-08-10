import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.backtesting.artifacts import aggregate_outputs
from tradingagents.backtesting.config import RESEARCH_DEPTH_ROUNDS, resolve_research_rounds
from tradingagents.backtesting.engine import WeeklyBacktestEngine
from tradingagents.backtesting.strategies import ScriptedStrategy


def test_backtest_medium_matches_cli_medium_depth():
    assert RESEARCH_DEPTH_ROUNDS["medium"] == 3
    assert resolve_research_rounds("medium") == 3


def test_aggregate_transaction_metrics_use_constituent_accounts(synthetic_provider, xnys):
    results = {}
    for symbol in ("AAPL", "AMZN", "JPM"):
        results[symbol] = WeeklyBacktestEngine(
            data_provider=synthetic_provider, schedule=xnys
        ).run(
            symbol=symbol, first_week="2024-01-01", final_week="2024-01-01",
            final_valuation_session="2024-01-12",
            strategy=ScriptedStrategy({"2024-01-05": "BUY"}), experiment_id="aggregate",
        )
    _, metrics = aggregate_outputs(results)
    expected_commission = sum(
        item.metrics["total_commission_cost"] for item in results.values()
    )
    expected_slippage = sum(
        item.metrics["total_slippage_cost"] for item in results.values()
    )
    expected_cost = expected_commission + expected_slippage
    assert metrics["trade_count"] == 3
    assert metrics["total_commission_cost"] == pytest.approx(expected_commission)
    assert metrics["total_slippage_cost"] == pytest.approx(expected_slippage)
    assert metrics["total_transaction_cost"] == pytest.approx(expected_cost)
    assert metrics["commission_cost_rate"] == pytest.approx(expected_commission / 300_000)
    assert metrics["slippage_cost_rate"] == pytest.approx(expected_slippage / 300_000)
    assert metrics["transaction_cost_rate"] == pytest.approx(expected_cost / 300_000)
    assert metrics["turnover"] > 0
    assert metrics["metric_scope"]["return_and_risk"].startswith("equal_weight")


def test_cli_creates_immutable_complete_run_bundles(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    command = [
        sys.executable, str(repository / "scripts" / "run_weekly_backtest.py"),
        "--config", str(repository / "configs" / "backtest_m0_2024h1.json"),
        "--strategy", "sma", "--experiment-id", "artifact-test",
        "--data-source", "synthetic", "--output-root", str(tmp_path), "--force",
    ]
    subprocess.run(command, cwd=repository, check=True, capture_output=True, text=True)
    subprocess.run(command, cwd=repository, check=True, capture_output=True, text=True)
    experiment = tmp_path / "artifact-test"
    runs = sorted((experiment / "runs").iterdir())
    assert len(runs) == 2
    assert runs[0].name != runs[1].name
    latest = json.loads((experiment / "latest.json").read_text())
    assert latest["run_id"] == runs[-1].name
    assert latest["status"] == "success"
    for run in runs:
        status = json.loads((run / "run_status.json").read_text())
        report = json.loads((run / "validation" / "validation_report.json").read_text())
        assert status["status"] == "success"
        assert report["checks"]["snapshot_replay_deterministic"] is True
        for path in (
            "manifest.json", "config.resolved.json", "environment.json",
            "artifact_schema.json", "inputs/data_manifest.json",
            "strategy/combined/daily_equity.csv",
            "benchmarks/SPY_buy_and_hold/daily_equity.csv",
            "analysis_ready/weekly_performance.csv",
        ):
            assert (run / path).is_file()

        benchmark_fills = pd.read_csv(
            run / "benchmarks/AAPL_buy_and_hold/fills.csv"
        )
        assert {
            "raw_open_price", "fill_price", "quantity", "commission",
            "slippage_cost", "total_transaction_cost",
        } <= set(benchmark_fills.columns)
        recomputed_slippage = benchmark_fills["quantity"] * (
            benchmark_fills["fill_price"] - benchmark_fills["raw_open_price"]
        )
        assert benchmark_fills["slippage_cost"].tolist() == pytest.approx(
            recomputed_slippage.tolist()
        )
        assert benchmark_fills["total_transaction_cost"].tolist() == pytest.approx(
            (benchmark_fills["commission"] + recomputed_slippage).tolist()
        )
        benchmark_metrics = json.loads(
            (run / "benchmarks/AAPL_buy_and_hold/metrics.json").read_text()
        )
        assert benchmark_metrics["total_commission_cost"] == pytest.approx(
            benchmark_fills["commission"].sum()
        )
        assert benchmark_metrics["total_slippage_cost"] == pytest.approx(
            recomputed_slippage.sum()
        )
        benchmark_daily = pd.read_csv(
            run / "benchmarks/AAPL_buy_and_hold/daily_equity.csv"
        )
        assert benchmark_daily.iloc[-1]["cumulative_transaction_cost"] == pytest.approx(
            benchmark_metrics["total_transaction_cost"]
        )
        timeline = pd.read_csv(run / "analysis_ready/decision_timeline.csv")
        weekly = pd.read_csv(run / "analysis_ready/weekly_performance.csv")
        assert {"commission", "slippage_cost", "total_transaction_cost"} <= set(
            timeline.columns
        )
        assert {
            "commission_cost_in_period", "slippage_cost_in_period",
            "transaction_cost_in_period",
        } <= set(weekly.columns)
