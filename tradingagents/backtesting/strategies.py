from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _normalize_signal(signal: str) -> str:
    normalized = str(signal or "").strip().upper()
    if normalized not in {"BUY", "HOLD", "SELL"}:
        return "HOLD"
    return normalized


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    lowest_low = low.rolling(window=period, min_periods=period).min()
    highest_high = high.rolling(window=period, min_periods=period).max()
    denominator = (highest_high - lowest_low).replace(0.0, np.nan)
    rsv = ((close - lowest_low) / denominator) * 100.0
    rsv = rsv.fillna(50.0)

    k_values = []
    d_values = []
    k_prev = 50.0
    d_prev = 50.0
    for value in rsv:
        k_prev = (2.0 / 3.0) * k_prev + (1.0 / 3.0) * float(value)
        d_prev = (2.0 / 3.0) * d_prev + (1.0 / 3.0) * k_prev
        k_values.append(k_prev)
        d_values.append(d_prev)

    k = pd.Series(k_values, index=close.index)
    d = pd.Series(d_values, index=close.index)
    j = (3.0 * k) - (2.0 * d)
    return k, d, j


@dataclass
class StrategyDecision:
    signal: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal,
            "metadata": self.metadata,
        }


class BaseSignalStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    def reset(self) -> None:
        """Reset any internal state before a new backtest run."""

    def initial_decision(
        self,
        *,
        symbol: str,
        start_date: str,
        history: pd.DataFrame,
    ) -> Optional[StrategyDecision]:
        """Optional immediate decision applied at the start of a backtest window."""
        return None

    @abstractmethod
    def generate_signal(
        self,
        *,
        symbol: str,
        trade_date: str,
        history: pd.DataFrame,
        has_position: bool,
    ) -> StrategyDecision:
        raise NotImplementedError


class BuyAndHoldStrategy(BaseSignalStrategy):
    def __init__(self) -> None:
        super().__init__("B&H")
        self._has_issued_entry = False

    def reset(self) -> None:
        self._has_issued_entry = False

    def initial_decision(
        self,
        *,
        symbol: str,
        start_date: str,
        history: pd.DataFrame,
    ) -> Optional[StrategyDecision]:
        self._has_issued_entry = True
        return StrategyDecision("BUY", {"rule": "enter_at_window_start"})

    def generate_signal(self, *, symbol: str, trade_date: str, history: pd.DataFrame, has_position: bool) -> StrategyDecision:
        if not self._has_issued_entry and not has_position:
            self._has_issued_entry = True
            return StrategyDecision("BUY", {"rule": "enter_once"})
        return StrategyDecision("HOLD", {"rule": "hold_forever"})


class MacdStrategy(BaseSignalStrategy):
    def __init__(self, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9) -> None:
        super().__init__("MACD")
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.signal_span = signal_span

    def generate_signal(self, *, symbol: str, trade_date: str, history: pd.DataFrame, has_position: bool) -> StrategyDecision:
        if len(history) < self.slow_span + self.signal_span:
            return StrategyDecision("HOLD", {"rule": "insufficient_history"})

        close = history["Close"].astype(float)
        macd_line = _ema(close, self.fast_span) - _ema(close, self.slow_span)
        signal_line = _ema(macd_line, self.signal_span)

        macd_prev, macd_now = macd_line.iloc[-2], macd_line.iloc[-1]
        signal_prev, signal_now = signal_line.iloc[-2], signal_line.iloc[-1]

        if macd_prev <= signal_prev and macd_now > signal_now:
            return StrategyDecision("BUY", {"rule": "bullish_macd_crossover"})
        if macd_prev >= signal_prev and macd_now < signal_now:
            return StrategyDecision("SELL", {"rule": "bearish_macd_crossover"})
        return StrategyDecision("HOLD", {"rule": "no_crossover"})


class KdjRsiStrategy(BaseSignalStrategy):
    def __init__(
        self,
        *,
        kdj_period: int = 9,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        super().__init__("KDJRSI")
        self.kdj_period = kdj_period
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signal(self, *, symbol: str, trade_date: str, history: pd.DataFrame, has_position: bool) -> StrategyDecision:
        if len(history) < max(self.kdj_period + 2, self.rsi_period + 2):
            return StrategyDecision("HOLD", {"rule": "insufficient_history"})

        close = history["Close"].astype(float)
        high = history["High"].astype(float)
        low = history["Low"].astype(float)
        k, d, _ = _kdj(high, low, close, period=self.kdj_period)
        rsi = _rsi(close, period=self.rsi_period)

        k_prev, k_now = float(k.iloc[-2]), float(k.iloc[-1])
        d_prev, d_now = float(d.iloc[-2]), float(d.iloc[-1])
        rsi_now = float(rsi.iloc[-1])

        if k_prev <= d_prev and k_now > d_now and rsi_now <= self.oversold:
            return StrategyDecision("BUY", {"rule": "kdj_up_rsi_oversold", "rsi": rsi_now})
        if k_prev >= d_prev and k_now < d_now and rsi_now >= self.overbought:
            return StrategyDecision("SELL", {"rule": "kdj_down_rsi_overbought", "rsi": rsi_now})
        return StrategyDecision("HOLD", {"rule": "no_kdj_rsi_trigger", "rsi": rsi_now})


class ZmrStrategy(BaseSignalStrategy):
    def __init__(self, window: int = 20, entry_threshold: float = 1.0) -> None:
        super().__init__("ZMR")
        self.window = window
        self.entry_threshold = entry_threshold
        self._waiting_for_long_reversion = False
        self._waiting_for_short_reversion = False

    def reset(self) -> None:
        self._waiting_for_long_reversion = False
        self._waiting_for_short_reversion = False

    def generate_signal(self, *, symbol: str, trade_date: str, history: pd.DataFrame, has_position: bool) -> StrategyDecision:
        if len(history) < self.window + 2:
            return StrategyDecision("HOLD", {"rule": "insufficient_history"})

        close = history["Close"].astype(float)
        rolling_mean = close.rolling(window=self.window, min_periods=self.window).mean()
        rolling_std = close.rolling(window=self.window, min_periods=self.window).std(ddof=0)
        zscore = ((close - rolling_mean) / rolling_std.replace(0.0, np.nan)).fillna(0.0)
        z_prev = float(zscore.iloc[-2])
        z_now = float(zscore.iloc[-1])

        if z_now <= -self.entry_threshold:
            self._waiting_for_long_reversion = True
        if z_now >= self.entry_threshold:
            self._waiting_for_short_reversion = True

        if self._waiting_for_long_reversion and z_prev < 0.0 <= z_now:
            self._waiting_for_long_reversion = False
            return StrategyDecision("BUY", {"rule": "zscore_reverted_to_zero_from_oversold", "zscore": z_now})

        if self._waiting_for_short_reversion and z_prev > 0.0 >= z_now:
            self._waiting_for_short_reversion = False
            return StrategyDecision("SELL", {"rule": "zscore_reverted_to_zero_from_overbought", "zscore": z_now})

        return StrategyDecision("HOLD", {"rule": "no_zmr_trigger", "zscore": z_now})


class SmaCrossStrategy(BaseSignalStrategy):
    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        super().__init__("SMA")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, *, symbol: str, trade_date: str, history: pd.DataFrame, has_position: bool) -> StrategyDecision:
        if len(history) < self.long_window + 2:
            return StrategyDecision("HOLD", {"rule": "insufficient_history"})

        close = history["Close"].astype(float)
        short_sma = close.rolling(window=self.short_window, min_periods=self.short_window).mean()
        long_sma = close.rolling(window=self.long_window, min_periods=self.long_window).mean()
        short_prev, short_now = float(short_sma.iloc[-2]), float(short_sma.iloc[-1])
        long_prev, long_now = float(long_sma.iloc[-2]), float(long_sma.iloc[-1])

        if short_prev <= long_prev and short_now > long_now:
            return StrategyDecision("BUY", {"rule": "bullish_sma_crossover"})
        if short_prev >= long_prev and short_now < long_now:
            return StrategyDecision("SELL", {"rule": "bearish_sma_crossover"})
        return StrategyDecision("HOLD", {"rule": "no_crossover"})


class TradingAgentsStrategy(BaseSignalStrategy):
    def __init__(self, graph, name: str = "TradingAgents") -> None:
        super().__init__(name)
        self.graph = graph

    def reset(self) -> None:
        """TradingAgents decisions do not maintain strategy-local backtest state."""

    def generate_signal(self, *, symbol: str, trade_date: str, history: pd.DataFrame, has_position: bool) -> StrategyDecision:
        final_state, signal = self.graph.propagate(symbol, trade_date)
        return StrategyDecision(
            _normalize_signal(signal),
            {
                "rating": _normalize_signal(signal),
                "final_trade_decision": final_state.get("final_trade_decision", ""),
            },
        )


def build_rule_based_baselines() -> Dict[str, BaseSignalStrategy]:
    return {
        "B&H": BuyAndHoldStrategy(),
        "MACD": MacdStrategy(),
        "KDJRSI": KdjRsiStrategy(),
        "ZMR": ZmrStrategy(),
        "SMA": SmaCrossStrategy(),
    }
