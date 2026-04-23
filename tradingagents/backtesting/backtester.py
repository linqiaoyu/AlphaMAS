from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from tradingagents.backtesting.execution import SimulatedExchange
from tradingagents.backtesting.metrics import BacktestMetrics, compute_backtest_metrics
from tradingagents.backtesting.strategies import BaseSignalStrategy, build_rule_based_baselines
from tradingagents.dataflows.stockstats_utils import load_full_ohlcv


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    daily_records: pd.DataFrame
    trades: pd.DataFrame
    metrics: BacktestMetrics

    def summary_row(self) -> Dict[str, Any]:
        row = {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "trade_count": int(len(self.trades)),
        }
        row.update(self.metrics.to_dict())
        return row

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.daily_records.to_csv(output_path / "daily_records.csv", index=False)
        self.trades.to_csv(output_path / "trades.csv", index=False)
        with open(output_path / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(self.summary_row(), handle, indent=2)

        return output_path


class SingleAssetBacktester:
    """Run one strategy on one symbol using a shared execution model."""

    def __init__(
        self,
        *,
        initial_cash: float = 100000.0,
        annual_risk_free_rate: float = 0.0,
        fill_policy: str = "next_open",
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        buy_fraction: float = 1.0,
        market_data_loader=load_full_ohlcv,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.annual_risk_free_rate = float(annual_risk_free_rate)
        self.fill_policy = fill_policy
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.buy_fraction = float(buy_fraction)
        self.market_data_loader = market_data_loader

    def run(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        strategy: BaseSignalStrategy,
    ) -> BacktestResult:
        full_history = self.market_data_loader(symbol).copy()
        if full_history.empty:
            raise ValueError(f"No market data available for symbol '{symbol}'.")

        full_history["Date"] = pd.to_datetime(full_history["Date"], errors="coerce").dt.normalize()
        full_history = full_history.dropna(subset=["Date"]).sort_values("Date")

        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        trading_window = full_history.loc[
            (full_history["Date"] >= start_ts) & (full_history["Date"] <= end_ts)
        ].copy()
        if trading_window.empty:
            raise ValueError(
                f"No trading sessions found for {symbol} between {start_date} and {end_date}."
            )

        strategy.reset()
        exchange = SimulatedExchange(
            starting_cash=self.initial_cash,
            fill_policy=self.fill_policy,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            buy_fraction=self.buy_fraction,
            market_data_loader=self.market_data_loader,
        )

        pending_order: Optional[Dict[str, Any]] = None
        records: List[Dict[str, Any]] = []
        first_trade_date_str = trading_window["Date"].iloc[0].strftime("%Y-%m-%d")
        initial_history = full_history.loc[full_history["Date"] < trading_window["Date"].iloc[0]].copy()
        initial_decision = strategy.initial_decision(
            symbol=symbol,
            start_date=first_trade_date_str,
            history=initial_history,
        )
        initial_execution: Optional[Dict[str, Any]] = None
        if initial_decision and str(initial_decision.signal).upper() != "HOLD":
            execution_report, _ = exchange.execute_signal(
                symbol,
                initial_decision.signal,
                first_trade_date_str,
                fill_policy_override="same_open",
            )
            initial_execution = execution_report.to_dict()

        for trade_date in trading_window["Date"]:
            trade_date_str = trade_date.strftime("%Y-%m-%d")
            executed_order: Optional[Dict[str, Any]] = None

            if trade_date_str == first_trade_date_str and initial_execution is not None:
                executed_order = initial_execution
            if pending_order and pending_order["signal"] != "HOLD":
                execution_report, _ = exchange.execute_signal(
                    symbol,
                    pending_order["signal"],
                    pending_order["signal_date"],
                )
                executed_order = execution_report.to_dict()

            history_until_date = full_history.loc[full_history["Date"] <= trade_date].copy()
            pre_decision_snapshot = exchange.snapshot(trade_date_str).to_dict()
            current_position = pre_decision_snapshot["positions"].get(symbol, {})
            has_position = current_position.get("quantity", 0) != 0

            if trade_date_str == first_trade_date_str and initial_decision is not None:
                decision = initial_decision
            else:
                decision = strategy.generate_signal(
                    symbol=symbol,
                    trade_date=trade_date_str,
                    history=history_until_date,
                    has_position=has_position,
                )
            decision_signal = str(decision.signal).upper()

            end_of_day_snapshot = exchange.snapshot(trade_date_str).to_dict()
            records.append(
                {
                    "date": trade_date_str,
                    "strategy": strategy.name,
                    "symbol": symbol,
                    "signal": decision_signal,
                    "decision_metadata": json.dumps(decision.metadata, ensure_ascii=True, sort_keys=True),
                    "executed_signal_date": None if not executed_order else executed_order.get("requested_trade_date"),
                    "execution_status": None if not executed_order else executed_order.get("status"),
                    "execution_date": None if not executed_order else executed_order.get("execution_date"),
                    "execution_price": None if not executed_order else executed_order.get("execution_price"),
                    "position_quantity": end_of_day_snapshot["positions"].get(symbol, {}).get("quantity", 0),
                    "cash": end_of_day_snapshot["cash"],
                    "market_value": end_of_day_snapshot["total_market_value"],
                    "equity": end_of_day_snapshot["total_equity"],
                    "realized_pnl": end_of_day_snapshot["total_realized_pnl"],
                    "unrealized_pnl": end_of_day_snapshot["total_unrealized_pnl"],
                }
            )

            pending_order = None
            if not (trade_date_str == first_trade_date_str and initial_decision is not None):
                pending_order = {
                    "signal": decision_signal,
                    "signal_date": trade_date_str,
                    "metadata": decision.metadata,
                }

        daily_records = pd.DataFrame(records)
        equity_curve = pd.Series(
            daily_records["equity"].astype(float).values,
            index=pd.to_datetime(daily_records["date"]),
        )
        metrics = compute_backtest_metrics(
            equity_curve,
            annual_risk_free_rate=self.annual_risk_free_rate,
        )
        trades = pd.DataFrame(exchange.get_trade_log())

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            daily_records=daily_records,
            trades=trades,
            metrics=metrics,
        )


def run_strategy_suite(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    strategies: Optional[Dict[str, BaseSignalStrategy]] = None,
    backtester: Optional[SingleAssetBacktester] = None,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run a set of strategies across one or more symbols and return a summary table."""
    strategy_map = strategies or build_rule_based_baselines()
    engine = backtester or SingleAssetBacktester()
    results: List[BacktestResult] = []

    for symbol in symbols:
        for name, strategy in strategy_map.items():
            if strategy.name != name:
                strategy.name = name
            result = engine.run(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
            )
            results.append(result)
            if output_dir is not None:
                result.save(Path(output_dir) / name / symbol)

    summary = pd.DataFrame([result.summary_row() for result in results])
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path / "summary.csv", index=False)
    return summary
