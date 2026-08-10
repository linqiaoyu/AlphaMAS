import copy
import hashlib

import pytest

from scripts.run_weekly_backtest import portfolio_pnl_identity_errors, validate_run
from tradingagents.backtesting.artifacts import aggregate_outputs
from tradingagents.backtesting.engine import (
    WeeklyBacktestEngine,
    market_history_visibility_errors,
)
from tradingagents.backtesting.strategies import ScriptedStrategy


def test_close_decision_executes_next_open_and_repeated_buy_is_noop(synthetic_provider, xnys):
    actions = {"2024-01-05": "BUY", "2024-01-12": "BUY", "2024-01-19": "HOLD"}
    result = WeeklyBacktestEngine(data_provider=synthetic_provider, schedule=xnys).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-15",
        final_valuation_session="2024-01-26", strategy=ScriptedStrategy(actions),
        experiment_id="test",
    )
    assert len(result.decisions) == 3
    assert len(result.orders) == 1
    assert len(result.fills) == 1
    fill = result.fills.iloc[0]
    assert fill["execution_time"].startswith("2024-01-08")
    open_price = synthetic_provider.frames["TEST"].loc["2024-01-08", "Open"]
    assert fill["raw_open_price"] == open_price
    friday_close = synthetic_provider.frames["TEST"].loc["2024-01-05", "Close"]
    assert fill["raw_open_price"] != friday_close
    assert result.daily_equity.iloc[-1]["quantity"] > 0


def test_decision_failure_is_not_hold(synthetic_provider, xnys):
    class Failed:
        strategy_id = "failed"

        def decide(self, **kwargs):
            from tradingagents.backtesting.models import DecisionStatus, StrategyDecision

            return StrategyDecision(
                kwargs["symbol"], kwargs["decision_session"], kwargs["decision_time"],
                None, None, DecisionStatus.FAILED, "boom", self.strategy_id,
                kwargs["context"]["experiment_id"],
            )

    result = WeeklyBacktestEngine(data_provider=synthetic_provider, schedule=xnys).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-01",
        final_valuation_session="2024-01-12", strategy=Failed(), experiment_id="test",
    )
    assert result.decisions.iloc[0]["status"] == "failed"
    assert result.fills.empty
    assert result.metrics["decision_failure_count"] == 1


def test_final_valuation_does_not_force_liquidation(synthetic_provider, xnys):
    result = WeeklyBacktestEngine(data_provider=synthetic_provider, schedule=xnys).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-01",
        final_valuation_session="2024-01-12",
        strategy=ScriptedStrategy({"2024-01-05": "BUY"}), experiment_id="test",
    )
    assert result.daily_equity.iloc[-1]["session"] == "2024-01-12"
    assert result.daily_equity.iloc[-1]["quantity"] > 0
    assert len(result.fills) == 1
    assert result.daily_equity.iloc[-1]["equity"] == pytest.approx(
        result.daily_equity.iloc[-1]["cash"]
        + result.daily_equity.iloc[-1]["quantity"] * result.daily_equity.iloc[-1]["close_price"]
    )


def test_max_decisions_caps_strategy_cases_but_keeps_daily_valuation(synthetic_provider, xnys):
    result = WeeklyBacktestEngine(data_provider=synthetic_provider, schedule=xnys).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-15",
        final_valuation_session="2024-01-26", strategy=ScriptedStrategy(),
        experiment_id="pilot", max_decisions=1,
    )
    assert len(result.decisions) == 1
    assert result.daily_equity.iloc[-1]["session"] == "2024-01-26"


def test_market_history_visibility_audits_the_actual_point_in_time_input(
    synthetic_provider, xnys,
):
    class CapturingStrategy(ScriptedStrategy):
        visible_history = None

        def decide(self, **kwargs):
            self.visible_history = kwargs["market_history"].copy()
            return super().decide(**kwargs)

    strategy = CapturingStrategy()
    result = WeeklyBacktestEngine(
        data_provider=synthetic_provider, schedule=xnys
    ).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-01",
        final_valuation_session="2024-01-12", strategy=strategy,
        experiment_id="visibility-valid",
    )
    audit = result.decisions.iloc[0]["metadata"]["market_history_visibility"]

    assert strategy.visible_history.index.max().date().isoformat() == "2024-01-05"
    assert audit["last_session"] == "2024-01-05"
    assert audit["last_observation_time_utc"] == xnys.session_close(
        "2024-01-05"
    ).isoformat()
    assert audit["row_count"] == len(strategy.visible_history)
    assert market_history_visibility_errors({"TEST": result}, xnys) == []


def test_market_history_visibility_validation_detects_future_strategy_input(
    synthetic_provider, xnys,
):
    result = WeeklyBacktestEngine(
        data_provider=synthetic_provider, schedule=xnys
    ).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-01",
        final_valuation_session="2024-01-12", strategy=ScriptedStrategy(),
        experiment_id="visibility-future",
    )
    audit = result.decisions.iloc[0]["metadata"]["market_history_visibility"]
    audit["last_session"] = "2024-01-08"
    audit["last_observation_time_utc"] = xnys.session_close("2024-01-08").isoformat()

    errors = market_history_visibility_errors({"TEST": result}, xnys)

    assert any("extends beyond decision session" in error for error in errors)
    assert any("extends beyond decision time" in error for error in errors)


def test_validation_report_check_uses_actual_market_history_visibility(
    tmp_path, synthetic_provider, xnys,
):
    result = WeeklyBacktestEngine(
        data_provider=synthetic_provider, schedule=xnys
    ).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-01",
        final_valuation_session="2024-01-12", strategy=ScriptedStrategy(),
        experiment_id="visibility-report",
    )
    results = {"TEST": result}
    aggregate, _ = aggregate_outputs(results)
    events = xnys.weekly_events("2024-01-01", "2024-01-01")
    market_path = tmp_path / "inputs/market_data/TEST.csv"
    market_path.parent.mkdir(parents=True)
    market_path.write_text("Date,Close\n2024-01-05,100\n", encoding="utf-8")
    manifest = {
        "configured_warmup_sessions": 252,
        "actual_warmup_sessions": 252,
        "first_decision_session": "2024-01-05",
        "symbols": [{
            "symbol": "TEST",
            "sha256": hashlib.sha256(market_path.read_bytes()).hexdigest(),
        }],
    }
    arguments = {
        "results": results,
        "stock_benchmarks": {"TEST": result},
        "spy": result,
        "replay_results": None,
        "events": events,
        "config": {
            "calendar": "XNYS",
            "first_calendar_week": "2024-01-01",
            "final_valuation_session": "2024-01-12",
            "warmup_sessions": 252,
            "initial_cash": 100_000,
        },
        "data_manifest": manifest,
        "strategy_name": "scripted",
        "run_dir": tmp_path,
        "expected_decisions": 1,
        "aggregate": aggregate,
    }

    valid_report = validate_run(**arguments)
    audit = result.decisions.iloc[0]["metadata"]["market_history_visibility"]
    audit["last_session"] = "2024-01-08"
    audit["last_observation_time_utc"] = xnys.session_close("2024-01-08").isoformat()
    invalid_report = validate_run(**arguments)
    audit["last_session"] = "2024-01-05"
    audit["last_observation_time_utc"] = xnys.session_close("2024-01-05").isoformat()
    result.daily_equity.loc[0, "realized_pnl"] += 1.0
    invalid_accounting_report = validate_run(**arguments)

    assert valid_report["checks"]["no_future_market_data_visible"] is True
    assert valid_report["market_history_visibility_errors"] == []
    assert invalid_report["status"] == "failed"
    assert invalid_report["checks"]["no_future_market_data_visible"] is False
    assert invalid_report["market_history_visibility_errors"]
    assert invalid_accounting_report["checks"]["no_future_market_data_visible"] is True
    assert invalid_accounting_report["checks"]["portfolio_pnl_identity"] is False
    assert invalid_accounting_report["portfolio_pnl_identity_errors"]


def test_portfolio_pnl_identity_validates_all_rows_costs_and_benchmarks(
    synthetic_provider, xnys,
):
    valid = WeeklyBacktestEngine(
        data_provider=synthetic_provider, schedule=xnys
    ).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-01",
        final_valuation_session="2024-01-12",
        strategy=ScriptedStrategy({"2024-01-05": "BUY"}),
        experiment_id="pnl-validation",
    )

    assert portfolio_pnl_identity_errors(
        results={"TEST": valid},
        stock_benchmarks={"TEST": valid},
        spy=valid,
        initial_cash=100_000,
    ) == []

    broken_identity = copy.deepcopy(valid)
    broken_identity.daily_equity.loc[0, "unrealized_pnl"] += 1.0
    identity_errors = portfolio_pnl_identity_errors(
        results={"TEST": broken_identity},
        stock_benchmarks={"TEST": valid},
        spy=valid,
        initial_cash=100_000,
    )
    assert any(
        "strategy/TEST" in error and "equity" in error
        for error in identity_errors
    )

    broken_benchmark_cost = copy.deepcopy(valid)
    broken_benchmark_cost.daily_equity.loc[
        broken_benchmark_cost.daily_equity.index[-1], "cumulative_cost"
    ] += 1.0
    cost_errors = portfolio_pnl_identity_errors(
        results={"TEST": valid},
        stock_benchmarks={"TEST": broken_benchmark_cost},
        spy=valid,
        initial_cash=100_000,
    )
    assert any(
        "benchmark/TEST" in error and "fills commissions" in error
        for error in cost_errors
    )

    decreasing_cost = copy.deepcopy(valid)
    decreasing_cost.daily_equity.loc[
        decreasing_cost.daily_equity.index[-1], "cumulative_cost"
    ] = 0.0
    monotonic_errors = portfolio_pnl_identity_errors(
        results={"TEST": valid},
        stock_benchmarks={"TEST": valid},
        spy=decreasing_cost,
        initial_cash=100_000,
    )
    assert any(
        "benchmark/SPY" in error and "cumulative_cost decreases" in error
        for error in monotonic_errors
    )
