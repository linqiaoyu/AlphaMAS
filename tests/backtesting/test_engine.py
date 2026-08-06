import pytest

from tradingagents.backtesting.engine import WeeklyBacktestEngine
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
