"""Normalized, injectable daily OHLCV providers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
ACTION_COLUMNS = ("Dividends", "Stock Splits")
CANONICAL_COLUMNS = (*REQUIRED_COLUMNS, *ACTION_COLUMNS)


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "Date" in data.columns:
        data = data.set_index("Date")
    data.index = pd.to_datetime(data.index, errors="raise").tz_localize(None).normalize()
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    for column in ACTION_COLUMNS:
        if column not in data:
            data[column] = 0.0
    data = data.loc[:, CANONICAL_COLUMNS].astype(float).sort_index()
    if not data.index.is_unique:
        raise ValueError("OHLCV index must be unique")
    if data.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("OHLCV required columns must not contain missing values")
    if (data[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Open and Close must be positive")
    if (data["Volume"] < 0).any():
        raise ValueError("Volume must be non-negative")
    if (data.loc[:, ACTION_COLUMNS] < 0).any().any():
        raise ValueError("corporate actions must be non-negative")
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
            auto_adjust=False, actions=True, progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return normalize_ohlcv(data)


class CSVSnapshotDataProvider:
    """Strict offline replay of canonical ``<symbol>.csv`` market snapshots."""

    provider_name = "snapshot"

    def __init__(self, snapshot_dir: str | Path) -> None:
        self.snapshot_dir = Path(snapshot_dir)

    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self.snapshot_dir / f"{symbol}.csv"
        if not path.is_file():
            raise ValueError(f"snapshot is missing {symbol}: {path}")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"invalid snapshot for {symbol}: {exc}") from exc
        data = normalize_ohlcv(frame)
        sliced = data.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if sliced.empty:
            raise ValueError(f"snapshot has no {symbol} rows in requested range")
        return sliced.copy()


def canonical_market_csv(frame: pd.DataFrame, path: str | Path) -> str:
    """Write normalized market data and return the exact-file SHA256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = normalize_ohlcv(frame).reset_index(names="Date")
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
    data.to_csv(target, index=False, lineterminator="\n", float_format="%.10g")
    return hashlib.sha256(target.read_bytes()).hexdigest()
