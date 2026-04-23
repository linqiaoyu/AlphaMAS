import unittest

import pandas as pd

from tradingagents.backtesting.backtester import SingleAssetBacktester, run_strategy_suite
from tradingagents.backtesting.strategies import BaseSignalStrategy, BuyAndHoldStrategy, StrategyDecision


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12"]),
            "Open": [10.0, 11.0, 12.0, 13.0],
            "High": [10.5, 12.5, 12.5, 13.5],
            "Low": [9.5, 10.5, 10.5, 12.5],
            "Close": [10.0, 12.0, 11.0, 13.0],
            "Volume": [1000, 1000, 1000, 1000],
        }
    )


class ScriptedStrategy(BaseSignalStrategy):
    def __init__(self, decisions):
        super().__init__("Scripted")
        self.decisions = decisions

    def generate_signal(self, *, symbol: str, trade_date: str, history: pd.DataFrame, has_position: bool) -> StrategyDecision:
        return StrategyDecision(self.decisions.get(trade_date, "HOLD"), {"has_position": has_position})


class BacktesterPipelineTests(unittest.TestCase):
    def setUp(self):
        self.engine = SingleAssetBacktester(
            initial_cash=100.0,
            market_data_loader=lambda symbol: _history(),
        )

    def test_single_asset_backtest_updates_equity_curve_and_trade_log(self):
        strategy = ScriptedStrategy(
            {
                "2024-03-07": "BUY",
                "2024-03-08": "HOLD",
                "2024-03-11": "SELL",
                "2024-03-12": "HOLD",
            }
        )

        result = self.engine.run(
            symbol="AAPL",
            start_date="2024-03-07",
            end_date="2024-03-12",
            strategy=strategy,
        )

        self.assertEqual(len(result.daily_records), 4)
        self.assertEqual(len(result.trades), 2)
        self.assertAlmostEqual(result.daily_records.iloc[-1]["equity"], 118.0)
        self.assertAlmostEqual(result.metrics.cumulative_return, 0.18)
        self.assertAlmostEqual(result.metrics.final_equity, 118.0)
        self.assertEqual(result.daily_records.iloc[-1]["position_quantity"], -7)

    def test_buy_and_hold_enters_on_first_trading_day_open(self):
        result = self.engine.run(
            symbol="AAPL",
            start_date="2024-03-07",
            end_date="2024-03-12",
            strategy=BuyAndHoldStrategy(),
        )

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["execution_date"], "2024-03-07")
        self.assertEqual(result.daily_records.iloc[0]["signal"], "BUY")
        self.assertEqual(result.daily_records.iloc[0]["execution_date"], "2024-03-07")
        self.assertEqual(result.daily_records.iloc[0]["position_quantity"], 10)
        self.assertAlmostEqual(result.metrics.final_equity, 130.0)
        self.assertAlmostEqual(result.metrics.cumulative_return, 0.30)

    def test_strategy_suite_returns_summary_table(self):
        strategies = {
            "Scripted": ScriptedStrategy({"2024-03-07": "BUY"}),
        }
        summary = run_strategy_suite(
            symbols=["AAPL"],
            start_date="2024-03-07",
            end_date="2024-03-12",
            strategies=strategies,
            backtester=self.engine,
        )

        self.assertEqual(list(summary["strategy"]), ["Scripted"])
        self.assertEqual(list(summary["symbol"]), ["AAPL"])
        self.assertIn("CR", summary.columns)
        self.assertIn("MDD", summary.columns)


if __name__ == "__main__":
    unittest.main()
