"""Normalized, injectable daily OHLCV providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "Date" in data.columns:
        data = data.set_index("Date")
    data.index = pd.to_datetime(data.index, errors="raise").tz_localize(None).normalize()
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    data = data.loc[:, REQUIRED_COLUMNS].astype(float).sort_index()
    if not data.index.is_unique:
        raise ValueError("OHLCV index must be unique")
    if (data[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Open and Close must be positive")
    if (data["Volume"] < 0).any():
        raise ValueError("Volume must be non-negative")
    if (data["High"] < data[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("High violates OHLC bounds")
    if (data["Low"] > data[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Low violates OHLC bounds")
    return data


class MarketDataProvider(Protocol):
    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame: ...


class InMemoryDataProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = {key: normalize_ohlcv(value) for key, value in frames.items()}

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if symbol not in self.frames:
            raise ValueError(f"no market data for {symbol}")
        return self.frames[symbol].loc[pd.Timestamp(start):pd.Timestamp(end)].copy()


class CallableDataProvider:
    def __init__(self, loader: Callable[[str, str, str], pd.DataFrame]) -> None:
        self.loader = loader

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return normalize_ohlcv(self.loader(symbol, start, end))


class YFinanceDataProvider:
    """Network provider kept behind the same interface; never used by offline tests."""

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        data = yf.download(
            symbol, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False, progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return normalize_ohlcv(data)
