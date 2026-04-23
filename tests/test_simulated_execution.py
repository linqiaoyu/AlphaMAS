import unittest

import pandas as pd

from tradingagents.backtesting.execution import SimulatedExchange


def _sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-03-07", "2024-03-08", "2024-03-11"]),
            "Open": [100.0, 102.0, 104.0],
            "High": [101.0, 103.0, 105.0],
            "Low": [99.0, 101.0, 103.0],
            "Close": [101.0, 103.0, 105.0],
            "Volume": [1000, 1200, 1100],
        }
    )


class SimulatedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.exchange = SimulatedExchange(
            starting_cash=1000.0,
            market_data_loader=lambda symbol: _sample_history(),
        )

    def test_buy_fills_on_next_open_and_updates_portfolio(self):
        report, snapshot = self.exchange.execute_signal("AAPL", "BUY", "2024-03-07")

        self.assertEqual(report.status, "filled")
        self.assertEqual(report.execution_date, "2024-03-08")
        self.assertEqual(report.execution_price, 102.0)
        self.assertEqual(report.quantity, 9)
        self.assertAlmostEqual(snapshot["cash"] if isinstance(snapshot, dict) else snapshot.cash, 82.0)
        portfolio = snapshot.to_dict()
        self.assertEqual(portfolio["trade_count"], 1)
        self.assertEqual(portfolio["positions"]["AAPL"]["quantity"], 9)
        self.assertEqual(portfolio["positions"]["AAPL"]["side"], "long")
        self.assertAlmostEqual(portfolio["total_equity"], 1000.0)

    def test_hold_is_noop_and_keeps_cash_unchanged(self):
        report, snapshot = self.exchange.execute_signal("AAPL", "HOLD", "2024-03-07")

        self.assertEqual(report.status, "noop")
        self.assertEqual(report.reason, "Signal was HOLD; portfolio left unchanged.")
        self.assertEqual(snapshot.to_dict()["cash"], 1000.0)
        self.assertEqual(snapshot.to_dict()["trade_count"], 0)

    def test_sell_opens_short_when_flat(self):
        report, snapshot = self.exchange.execute_signal("AAPL", "SELL", "2024-03-07")

        portfolio = snapshot.to_dict()
        self.assertEqual(report.status, "filled")
        self.assertEqual(report.execution_date, "2024-03-08")
        self.assertEqual(report.quantity, 9)
        self.assertEqual(report.signed_quantity, -9)
        self.assertEqual(portfolio["positions"]["AAPL"]["quantity"], -9)
        self.assertEqual(portfolio["positions"]["AAPL"]["side"], "short")
        self.assertAlmostEqual(portfolio["cash"], 1918.0)
        self.assertAlmostEqual(portfolio["total_equity"], 1000.0)
        self.assertEqual(portfolio["trade_count"], 1)

    def test_sell_flips_long_to_short_and_realizes_pnl(self):
        self.exchange.execute_signal("AAPL", "BUY", "2024-03-07")
        report, snapshot = self.exchange.execute_signal("AAPL", "SELL", "2024-03-08")

        portfolio = snapshot.to_dict()
        self.assertEqual(report.status, "filled")
        self.assertEqual(report.execution_date, "2024-03-11")
        self.assertEqual(report.quantity, 18)
        self.assertEqual(report.signed_quantity, -18)
        self.assertEqual(portfolio["positions"]["AAPL"]["quantity"], -9)
        self.assertEqual(portfolio["positions"]["AAPL"]["side"], "short")
        self.assertAlmostEqual(portfolio["cash"], 1954.0)
        self.assertAlmostEqual(portfolio["total_realized_pnl"], 18.0)
        self.assertEqual(portfolio["trade_count"], 2)

    def test_buy_flips_short_to_long(self):
        self.exchange.execute_signal("AAPL", "SELL", "2024-03-07")
        report, snapshot = self.exchange.execute_signal("AAPL", "BUY", "2024-03-08")

        portfolio = snapshot.to_dict()
        self.assertEqual(report.status, "filled")
        self.assertEqual(report.execution_date, "2024-03-11")
        self.assertEqual(report.quantity, 18)
        self.assertEqual(report.signed_quantity, 18)
        self.assertEqual(portfolio["positions"]["AAPL"]["quantity"], 9)
        self.assertEqual(portfolio["positions"]["AAPL"]["side"], "long")
        self.assertAlmostEqual(portfolio["cash"], 46.0)
        self.assertAlmostEqual(portfolio["total_realized_pnl"], -18.0)
        self.assertEqual(portfolio["trade_count"], 2)

    def test_snapshot_restore_preserves_cash_and_trade_count(self):
        self.exchange.execute_signal("AAPL", "BUY", "2024-03-07")
        snapshot = self.exchange.snapshot("2024-03-08", use_open_if_exact=True).to_dict()

        restored = SimulatedExchange(
            starting_cash=1000.0,
            market_data_loader=lambda symbol: _sample_history(),
        )
        restored.load_snapshot(snapshot)
        restored_snapshot = restored.snapshot("2024-03-08", use_open_if_exact=True).to_dict()

        self.assertEqual(restored_snapshot["cash"], snapshot["cash"])
        self.assertEqual(restored_snapshot["trade_count"], snapshot["trade_count"])
        self.assertEqual(restored_snapshot["positions"]["AAPL"]["quantity"], 9)


if __name__ == "__main__":
    unittest.main()
