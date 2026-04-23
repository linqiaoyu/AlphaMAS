import unittest

import pandas as pd

from tradingagents.backtesting.metrics import compute_backtest_metrics


class BacktestMetricsTests(unittest.TestCase):
    def test_compute_metrics_uses_positive_drawdown_magnitude(self):
        equity_curve = pd.Series(
            [100.0, 110.0, 105.0],
            index=pd.to_datetime(["2024-03-07", "2024-03-08", "2024-03-11"]),
        )

        metrics = compute_backtest_metrics(equity_curve)

        self.assertAlmostEqual(metrics.cumulative_return, 0.05)
        self.assertAlmostEqual(metrics.maximum_drawdown, 5.0 / 110.0)
        self.assertEqual(metrics.initial_equity, 100.0)
        self.assertEqual(metrics.final_equity, 105.0)

    def test_compute_metrics_caps_annualized_return_after_equity_blows_up(self):
        equity_curve = pd.Series(
            [100.0, -50.0],
            index=pd.to_datetime(["2024-03-07", "2024-03-08"]),
        )

        metrics = compute_backtest_metrics(equity_curve)

        self.assertAlmostEqual(metrics.cumulative_return, -1.5)
        self.assertEqual(metrics.annualized_return, -1.0)
        self.assertEqual(metrics.final_equity, -50.0)

    def test_compute_metrics_requires_positive_initial_equity(self):
        equity_curve = pd.Series(
            [0.0, 10.0],
            index=pd.to_datetime(["2024-03-07", "2024-03-08"]),
        )

        with self.assertRaisesRegex(ValueError, "strictly positive initial equity"):
            compute_backtest_metrics(equity_curve)


if __name__ == "__main__":
    unittest.main()
