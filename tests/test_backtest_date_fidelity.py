import unittest
from datetime import date
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import pandas as pd

from tradingagents.dataflows import interface
from tradingagents.dataflows.stockstats_utils import _clean_dataframe
from tradingagents.dataflows import yfinance_news


class BacktestDateFidelityTests(unittest.TestCase):
    def test_route_to_vendor_clamps_future_end_date(self):
        captured = {}

        def fake_stock_vendor(symbol, start_date, end_date):
            captured["args"] = (symbol, start_date, end_date)
            return "ok"

        config = {
            "data_vendors": {"core_stock_apis": "yfinance"},
            "tool_vendors": {},
            "backtest_as_of_date": "2024-06-30",
        }

        with patch("tradingagents.dataflows.interface.get_config", return_value=config):
            with patch.dict(
                interface.VENDOR_METHODS,
                {"get_stock_data": {"yfinance": fake_stock_vendor}},
                clear=False,
            ):
                result = interface.route_to_vendor(
                    "get_stock_data",
                    "AAPL",
                    "2024-05-01",
                    "2026-01-01",
                )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["args"], ("AAPL", "2024-05-01", "2024-06-30"))

    def test_route_to_vendor_injects_as_of_for_insider_transactions(self):
        captured = {}

        def fake_insider_vendor(ticker, curr_date=None):
            captured["args"] = (ticker, curr_date)
            return "ok"

        config = {
            "data_vendors": {"news_data": "yfinance"},
            "tool_vendors": {},
            "backtest_as_of_date": "2024-06-30",
        }

        with patch("tradingagents.dataflows.interface.get_config", return_value=config):
            with patch.dict(
                interface.VENDOR_METHODS,
                {"get_insider_transactions": {"yfinance": fake_insider_vendor}},
                clear=False,
            ):
                result = interface.route_to_vendor("get_insider_transactions", "AAPL")

        self.assertEqual(result, "ok")
        self.assertEqual(captured["args"], ("AAPL", "2024-06-30"))

    def test_clean_dataframe_avoids_backfill_lookahead(self):
        raw = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Open": [None, 10.0],
                "High": [10.5, 11.0],
                "Low": [9.5, 10.0],
                "Close": [10.0, 11.0],
                "Volume": [1000, 1100],
            }
        )

        cleaned = _clean_dataframe(raw)
        kept_dates = cleaned["Date"].dt.strftime("%Y-%m-%d").tolist()

        self.assertEqual(kept_dates, ["2024-01-02"])

    def test_news_yfinance_drops_undated_and_future_articles(self):
        articles = [
            {
                "content": {
                    "title": "In Range Article",
                    "summary": "valid",
                    "provider": {"displayName": "NewsWire"},
                    "canonicalUrl": {"url": "https://example.com/in-range"},
                    "pubDate": "2024-01-03T12:00:00Z",
                }
            },
            {
                "content": {
                    "title": "Undated Article",
                    "summary": "unknown timestamp",
                    "provider": {"displayName": "Unknown"},
                    "canonicalUrl": {"url": "https://example.com/undated"},
                }
            },
            {
                "content": {
                    "title": "Future Article",
                    "summary": "from the future",
                    "provider": {"displayName": "FutureWire"},
                    "canonicalUrl": {"url": "https://example.com/future"},
                    "pubDate": "2024-02-01T00:00:00Z",
                }
            },
        ]

        with TemporaryDirectory() as temp_dir:
            config = {"data_cache_dir": temp_dir}
            with patch("tradingagents.dataflows.yfinance_news.get_config", return_value=config):
                with patch("tradingagents.dataflows.yfinance_news._today_utc_date", return_value=date(2024, 1, 7)):
                    with patch("tradingagents.dataflows.yfinance_news.yf.Ticker"):
                        with patch("tradingagents.dataflows.yfinance_news.yf_retry", return_value=articles):
                            result = yfinance_news.get_news_yfinance("AAPL", "2024-01-01", "2024-01-07")

        self.assertIn("In Range Article", result)
        self.assertNotIn("Undated Article", result)
        self.assertNotIn("Future Article", result)

    def test_global_news_filters_flat_records_by_publish_time(self):
        in_range_flat = {
            "title": "In Range Flat",
            "publisher": "FlatWire",
            "link": "https://example.com/in-range-flat",
            # 2024-01-09 00:00:00 UTC
            "providerPublishTime": 1704758400,
        }
        future_flat = {
            "title": "Future Flat",
            "publisher": "FlatWire",
            "link": "https://example.com/future-flat",
            # 2024-01-15 00:00:00 UTC
            "providerPublishTime": 1705276800,
        }
        undated_flat = {
            "title": "Undated Flat",
            "publisher": "FlatWire",
            "link": "https://example.com/undated-flat",
        }

        def fake_search(*args, **kwargs):
            return SimpleNamespace(news=[in_range_flat, future_flat, undated_flat])

        with TemporaryDirectory() as temp_dir:
            config = {"data_cache_dir": temp_dir}
            with patch("tradingagents.dataflows.yfinance_news.get_config", return_value=config):
                with patch("tradingagents.dataflows.yfinance_news._today_utc_date", return_value=date(2024, 1, 10)):
                    with patch("tradingagents.dataflows.yfinance_news.yf.Search", side_effect=fake_search):
                        with patch("tradingagents.dataflows.yfinance_news.yf_retry", side_effect=lambda fn: fn()):
                            result = yfinance_news.get_global_news_yfinance("2024-01-10", look_back_days=2, limit=10)

        self.assertIn("In Range Flat", result)
        self.assertNotIn("Future Flat", result)
        self.assertNotIn("Undated Flat", result)

    def test_historical_news_is_disabled_for_strict_baseline(self):
        with TemporaryDirectory() as temp_dir:
            config = {"data_cache_dir": temp_dir}
            with patch("tradingagents.dataflows.yfinance_news.get_config", return_value=config):
                with patch("tradingagents.dataflows.yfinance_news._today_utc_date", return_value=date(2026, 4, 23)):
                    result = yfinance_news.get_news_yfinance("AAPL", "2024-01-01", "2024-01-07")

        self.assertIn("Historical yfinance news unavailable", result)
        self.assertIn("simplified baseline disables historical news replay", result)
        self.assertIn("[NEWS_UNAVAILABLE]", result)

    def test_historical_news_ignores_recent_cache(self):
        with TemporaryDirectory() as temp_dir:
            config = {"data_cache_dir": temp_dir}
            cache_key = "AAPL|2024-01-01|2024-01-07"
            cache_path = Path(temp_dir) / "yfinance_news" / "ticker" / (
                hashlib.sha256(cache_key.encode("utf-8")).hexdigest() + ".json"
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "fetched_at": "2026-04-23T00:00:00+00:00",
                        "articles": [
                            {
                                "title": "Cached But Invalid",
                                "summary": "should not be used for historical replay",
                                "publisher": "CacheWire",
                                "link": "https://example.com/cache",
                                "pub_date": "2024-01-03T12:00:00+00:00",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tradingagents.dataflows.yfinance_news.get_config", return_value=config):
                with patch("tradingagents.dataflows.yfinance_news._today_utc_date", return_value=date(2026, 4, 23)):
                    result = yfinance_news.get_news_yfinance("AAPL", "2024-01-01", "2024-01-07")

        self.assertIn("Historical yfinance news unavailable", result)
        self.assertNotIn("Cached But Invalid", result)
        self.assertIn("[NEWS_UNAVAILABLE]", result)

if __name__ == "__main__":
    unittest.main()
