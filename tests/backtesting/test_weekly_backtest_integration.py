from tradingagents.backtesting.engine import WeeklyBacktestEngine
from tradingagents.backtesting.recorder import BacktestRecorder, equal_weight_aggregate
from tradingagents.backtesting.strategies import ScriptedStrategy


def test_26_week_three_account_offline_integration(tmp_path, synthetic_provider, xnys):
    results = {}
    for symbol in ("AAPL", "AMZN", "JPM"):
        result = WeeklyBacktestEngine(data_provider=synthetic_provider, schedule=xnys).run(
            symbol=symbol, first_week="2024-01-01", final_week="2024-06-24",
            final_valuation_session="2024-07-05",
            strategy=ScriptedStrategy({"2024-01-05": "BUY", "2024-06-28": "SELL"}),
            experiment_id="offline-integration",
        )
        results[symbol] = result
        assert len(result.decisions) == 26
        assert len(result.fills) == 2
        path = BacktestRecorder(tmp_path / symbol).save(result, {}, [])
        for filename in (
            "manifest.json", "schedule.csv", "decisions.csv", "orders.csv",
            "fills.csv", "daily_equity.csv", "metrics.json", "summary.csv", "failures.jsonl",
        ):
            assert (path / filename).exists()
    aggregate = equal_weight_aggregate(results)
    assert len(aggregate) == len(results["AAPL"].daily_equity)
    assert aggregate["normalized_equity"].iloc[0] == 1
