import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows.y_finance import get_YFin_data_online


class AdjustedPriceConventionTests(unittest.TestCase):
    def test_get_yfin_data_online_uses_adjusted_ohlcv_download(self):
        captured = {}
        sample = pd.DataFrame(
            {
                "Open": [10.0],
                "High": [11.0],
                "Low": [9.5],
                "Close": [10.5],
                "Volume": [1000],
            },
            index=pd.to_datetime(["2024-03-07"]),
        )

        def fake_download(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return sample

        with patch("tradingagents.dataflows.y_finance.yf.download", side_effect=fake_download):
            result = get_YFin_data_online("AAPL", "2024-03-07", "2024-03-08")

        self.assertTrue(captured["kwargs"]["auto_adjust"])
        self.assertEqual(captured["kwargs"]["multi_level_index"], False)
        self.assertIn("Open,High,Low,Close,Volume", result)
        self.assertIn("2024-03-07", result)


if __name__ == "__main__":
    unittest.main()
