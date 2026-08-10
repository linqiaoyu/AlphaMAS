import hashlib

import pandas as pd
import pytest

from tradingagents.backtesting.data import (
    CSVSnapshotDataProvider,
    YFinanceDataProvider,
    canonical_market_csv,
    normalize_ohlcv,
)


def frame():
    return pd.DataFrame({
        "Date": ["2024-01-02", "2024-01-03"],
        "Open": [100, 101], "High": [102, 103], "Low": [99, 100],
        "Close": [101, 102], "Volume": [1000, 1100],
        "Dividends": [0, 0.25], "Stock Splits": [0, 0],
    })


def test_snapshot_round_trip_and_checksum(tmp_path):
    path = tmp_path / "AAPL.csv"
    digest = canonical_market_csv(frame(), path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    replay = CSVSnapshotDataProvider(tmp_path).load("AAPL", "2024-01-02", "2024-01-03")
    pd.testing.assert_frame_equal(replay, normalize_ohlcv(frame()))


def test_snapshot_rejects_missing_values_and_never_downloads(tmp_path):
    broken = frame()
    broken.loc[1, "Open"] = None
    broken.to_csv(tmp_path / "AAPL.csv", index=False)
    with pytest.raises(ValueError, match="missing values"):
        CSVSnapshotDataProvider(tmp_path).load("AAPL", "2024-01-02", "2024-01-03")
    with pytest.raises(ValueError, match="missing SPY"):
        CSVSnapshotDataProvider(tmp_path).load("SPY", "2024-01-02", "2024-01-03")


def test_yfinance_requests_raw_prices_and_actions(monkeypatch):
    import yfinance as yf

    captured = {}

    def download(symbol, **kwargs):
        captured.update({"symbol": symbol, **kwargs})
        return frame().set_index("Date")

    monkeypatch.setattr(yf, "download", download)
    loaded = YFinanceDataProvider().load("AAPL", "2024-01-02", "2024-01-03")
    assert captured["auto_adjust"] is False
    assert captured["actions"] is True
    assert captured["end"] == "2024-01-04"
    assert loaded.loc["2024-01-03", "Dividends"] == 0.25
