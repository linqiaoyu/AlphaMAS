from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Any, Callable, Dict, Optional

import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_full_ohlcv


def _parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _safe_round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _normalize_signal(signal: str) -> str:
    normalized = str(signal or "").strip().upper()
    if normalized not in {"BUY", "HOLD", "SELL"}:
        return "HOLD"
    return normalized


@dataclass
class PositionState:
    quantity: int = 0
    average_cost: float = 0.0
    realized_pnl: float = 0.0

    def to_dict(
        self,
        mark_price: Optional[float] = None,
        mark_price_source: Optional[str] = None,
        mark_price_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        if mark_price is None:
            market_value = None
            unrealized_pnl = None
        else:
            market_value = self.quantity * mark_price
            unrealized_pnl = self.quantity * (mark_price - self.average_cost)

        return {
            "quantity": int(self.quantity),
            "side": "long" if self.quantity > 0 else "short" if self.quantity < 0 else "flat",
            "average_cost": _safe_round(self.average_cost),
            "realized_pnl": _safe_round(self.realized_pnl),
            "mark_price": _safe_round(mark_price),
            "mark_price_source": mark_price_source,
            "mark_price_date": mark_price_date,
            "market_value": _safe_round(market_value),
            "unrealized_pnl": _safe_round(unrealized_pnl),
        }


@dataclass
class PortfolioSnapshot:
    as_of_date: str
    cash: float
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_market_value: float = 0.0
    total_equity: float = 0.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "cash": _safe_round(self.cash),
            "positions": self.positions,
            "total_market_value": _safe_round(self.total_market_value),
            "total_equity": _safe_round(self.total_equity),
            "total_realized_pnl": _safe_round(self.total_realized_pnl),
            "total_unrealized_pnl": _safe_round(self.total_unrealized_pnl),
            "trade_count": int(self.trade_count),
        }


@dataclass
class ExecutionReport:
    symbol: str
    signal: str
    requested_trade_date: str
    status: str
    reason: str
    execution_date: Optional[str] = None
    execution_price: Optional[float] = None
    execution_price_source: Optional[str] = None
    quantity: int = 0
    signed_quantity: int = 0
    gross_notional: float = 0.0
    fees: float = 0.0
    cash_before: float = 0.0
    cash_after: float = 0.0
    position_quantity_before: int = 0
    position_quantity_after: int = 0
    total_equity_before: float = 0.0
    total_equity_after: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "requested_trade_date": self.requested_trade_date,
            "status": self.status,
            "reason": self.reason,
            "execution_date": self.execution_date,
            "execution_price": _safe_round(self.execution_price),
            "execution_price_source": self.execution_price_source,
            "quantity": int(self.quantity),
            "signed_quantity": int(self.signed_quantity),
            "gross_notional": _safe_round(self.gross_notional),
            "fees": _safe_round(self.fees),
            "cash_before": _safe_round(self.cash_before),
            "cash_after": _safe_round(self.cash_after),
            "position_quantity_before": int(self.position_quantity_before),
            "position_quantity_after": int(self.position_quantity_after),
            "total_equity_before": _safe_round(self.total_equity_before),
            "total_equity_after": _safe_round(self.total_equity_after),
        }


class SimulatedExchange:
    """A simple long/short simulated broker with portfolio bookkeeping.

    The execution model is intentionally conservative:
    - Signals are generated with data visible on `requested_trade_date`.
    - Orders fill on the next trading session's open by default.
    - Portfolio snapshots mark positions to the latest available close,
      except immediately after a fill, where the exact-date open is used
      to avoid leaking the rest of the day.
    - BUY and SELL are interpreted as target directions, so they can flip
      an existing position from short to long or from long to short.
    """

    def __init__(
        self,
        *,
        starting_cash: float = 100000.0,
        fill_policy: str = "next_open",
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        buy_fraction: float = 1.0,
        allow_fractional_shares: bool = False,
        market_data_loader: Optional[Callable[[str], pd.DataFrame]] = None,
    ) -> None:
        self.starting_cash = float(starting_cash)
        self.fill_policy = fill_policy
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.buy_fraction = float(buy_fraction)
        self.allow_fractional_shares = bool(allow_fractional_shares)
        if self.allow_fractional_shares:
            raise ValueError(
                "SimulatedExchange currently supports whole-share execution only."
            )
        self.market_data_loader = market_data_loader or load_full_ohlcv

        self.reset()

    def reset(self, starting_cash: Optional[float] = None) -> None:
        self.cash = float(self.starting_cash if starting_cash is None else starting_cash)
        self.positions: Dict[str, PositionState] = {}
        self.trade_log: list[Dict[str, Any]] = []
        self.executed_trade_count = 0
        self.realized_pnl_total = 0.0

    def load_snapshot(self, snapshot: Optional[Dict[str, Any]]) -> None:
        if not snapshot:
            self.reset()
            return

        self.cash = float(snapshot.get("cash", self.starting_cash))
        self.positions = {}
        self.trade_log = []
        self.executed_trade_count = int(snapshot.get("trade_count", 0))
        self.realized_pnl_total = float(snapshot.get("total_realized_pnl", 0.0))
        for symbol, data in snapshot.get("positions", {}).items():
            quantity = int(data.get("quantity", 0))
            if quantity == 0:
                continue
            self.positions[symbol] = PositionState(
                quantity=quantity,
                average_cost=float(data.get("average_cost", 0.0)),
                realized_pnl=float(data.get("realized_pnl", 0.0)),
            )

    def snapshot(self, as_of_date: str, use_open_if_exact: bool = False) -> PortfolioSnapshot:
        mark_date = str(as_of_date)
        positions: Dict[str, Dict[str, Any]] = {}
        total_market_value = 0.0
        total_realized_pnl = self.realized_pnl_total
        total_unrealized_pnl = 0.0

        for symbol, position in self.positions.items():
            mark = self._lookup_mark(symbol, mark_date, use_open_if_exact=use_open_if_exact)
            positions[symbol] = position.to_dict(
                mark_price=mark["price"],
                mark_price_source=mark["source"],
                mark_price_date=mark["date"],
            )
            if positions[symbol]["market_value"] is not None:
                total_market_value += positions[symbol]["market_value"]
            if positions[symbol]["unrealized_pnl"] is not None:
                total_unrealized_pnl += positions[symbol]["unrealized_pnl"]

        return PortfolioSnapshot(
            as_of_date=mark_date,
            cash=self.cash,
            positions=positions,
            total_market_value=total_market_value,
            total_equity=self.cash + total_market_value,
            total_realized_pnl=total_realized_pnl,
            total_unrealized_pnl=total_unrealized_pnl,
            trade_count=self.executed_trade_count,
        )

    def execute_signal(
        self,
        symbol: str,
        signal: str,
        signal_date: str,
        *,
        fill_policy_override: Optional[str] = None,
    ) -> tuple[ExecutionReport, PortfolioSnapshot]:
        normalized_signal = _normalize_signal(signal)
        before_snapshot = self.snapshot(signal_date)
        existing_position = self.positions.get(symbol, PositionState())

        report = ExecutionReport(
            symbol=symbol,
            signal=normalized_signal,
            requested_trade_date=str(signal_date),
            status="noop",
            reason="No action taken.",
            cash_before=before_snapshot.cash,
            cash_after=before_snapshot.cash,
            position_quantity_before=existing_position.quantity,
            position_quantity_after=existing_position.quantity,
            total_equity_before=before_snapshot.total_equity,
            total_equity_after=before_snapshot.total_equity,
        )

        if normalized_signal == "HOLD":
            report.reason = "Signal was HOLD; portfolio left unchanged."
            return report, before_snapshot

        fill = self._get_fill(symbol, signal_date, fill_policy_override=fill_policy_override)
        if fill is None:
            report.status = "rejected"
            report.reason = (
                f"No future trading session available to execute a {normalized_signal} order "
                f"after {signal_date}."
            )
            return report, before_snapshot

        execution_date = fill["date"]
        raw_price = fill["price"]
        report.execution_date = execution_date
        report.execution_price_source = fill["source"]
        report = self._rebalance_to_signal(symbol, normalized_signal, raw_price, report)

        if report.status != "filled":
            return report, before_snapshot

        self.executed_trade_count += 1
        after_snapshot = self.snapshot(execution_date, use_open_if_exact=True)
        report.cash_after = after_snapshot.cash
        report.total_equity_after = after_snapshot.total_equity
        report.position_quantity_after = after_snapshot.positions.get(symbol, {}).get("quantity", 0)
        self.trade_log.append(report.to_dict())
        return report, after_snapshot

    def get_trade_log(self) -> list[Dict[str, Any]]:
        return list(self.trade_log)

    def _rebalance_to_signal(
        self,
        symbol: str,
        signal: str,
        raw_price: float,
        report: ExecutionReport,
    ) -> ExecutionReport:
        position = self.positions.get(symbol, PositionState())
        current_quantity = int(position.quantity)
        equity_before = max(0.0, report.total_equity_before)
        target_quantity = self._target_quantity(signal, raw_price, equity_before)
        delta_quantity = target_quantity - current_quantity

        if delta_quantity == 0:
            report.status = "noop"
            report.reason = (
                f"{signal} signal already matches the target {signal.lower()} exposure."
            )
            return report

        fill_price = self._apply_slippage(raw_price, delta_quantity)
        gross_notional = abs(delta_quantity) * fill_price
        fees = gross_notional * self.commission_bps / 10000.0

        realized_pnl = self._apply_trade(
            symbol=symbol,
            delta_quantity=delta_quantity,
            fill_price=fill_price,
            fees=fees,
        )

        report.status = "filled"
        report.reason = self._build_fill_reason(signal, report.execution_price_source)
        report.execution_price = fill_price
        report.quantity = abs(delta_quantity)
        report.signed_quantity = int(delta_quantity)
        report.gross_notional = gross_notional
        report.fees = fees
        self.realized_pnl_total += realized_pnl
        return report

    def _target_quantity(self, signal: str, raw_price: float, equity_before: float) -> int:
        if raw_price <= 0:
            return 0
        target_notional = max(0.0, equity_before * self.buy_fraction)
        target_abs_quantity = floor(target_notional / raw_price) if target_notional > 0 else 0
        if target_abs_quantity <= 0:
            return 0
        if signal == "BUY":
            return int(target_abs_quantity)
        if signal == "SELL":
            return -int(target_abs_quantity)
        return 0

    def _apply_slippage(self, raw_price: float, delta_quantity: int) -> float:
        if delta_quantity > 0:
            return raw_price * (1 + self.slippage_bps / 10000.0)
        return raw_price * (1 - self.slippage_bps / 10000.0)

    def _apply_trade(
        self,
        *,
        symbol: str,
        delta_quantity: int,
        fill_price: float,
        fees: float,
    ) -> float:
        position = self.positions.get(symbol, PositionState())
        current_quantity = int(position.quantity)
        current_average_cost = float(position.average_cost)
        total_shares = abs(delta_quantity)
        fee_per_share = fees / total_shares if total_shares else 0.0
        realized_pnl = 0.0

        self.cash -= delta_quantity * fill_price + fees

        if current_quantity == 0 or current_quantity * delta_quantity > 0:
            new_quantity = current_quantity + delta_quantity
            if new_quantity > 0:
                existing_cost = max(current_quantity, 0) * current_average_cost
                opening_cost = delta_quantity * fill_price + fees
                position.average_cost = (existing_cost + opening_cost) / new_quantity
            else:
                existing_proceeds = abs(min(current_quantity, 0)) * current_average_cost
                opening_proceeds = abs(delta_quantity) * fill_price - fees
                position.average_cost = (existing_proceeds + opening_proceeds) / abs(new_quantity)
            position.quantity = new_quantity
            self.positions[symbol] = position
            return realized_pnl

        closing_quantity = min(abs(current_quantity), abs(delta_quantity))
        opening_quantity = abs(delta_quantity) - closing_quantity
        closing_fees = fee_per_share * closing_quantity
        opening_fees = fees - closing_fees

        if current_quantity > 0:
            realized_pnl += closing_quantity * (fill_price - current_average_cost) - closing_fees
        else:
            realized_pnl += closing_quantity * (current_average_cost - fill_price) - closing_fees
        position.realized_pnl += realized_pnl

        new_quantity = current_quantity + delta_quantity
        position.quantity = new_quantity

        if new_quantity == 0:
            position.average_cost = 0.0
            self.positions.pop(symbol, None)
            return realized_pnl

        if opening_quantity > 0:
            if new_quantity > 0:
                position.average_cost = (opening_quantity * fill_price + opening_fees) / opening_quantity
            else:
                position.average_cost = (opening_quantity * fill_price - opening_fees) / opening_quantity

        self.positions[symbol] = position
        return realized_pnl

    def _build_fill_reason(self, signal: str, execution_price_source: Optional[str]) -> str:
        source = execution_price_source or self.fill_policy
        if signal == "BUY":
            return f"BUY order filled using the {source} execution rule."
        return f"SELL order filled using the {source} execution rule."

    def _get_fill(
        self,
        symbol: str,
        signal_date: str,
        *,
        fill_policy_override: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        history = self._load_history(symbol)
        if history.empty:
            return None

        signal_ts = _parse_date(signal_date)
        fill_policy = fill_policy_override or self.fill_policy

        if fill_policy == "same_open":
            row = history.loc[history["Date"] == signal_ts].head(1)
            if row.empty:
                return None
            record = row.iloc[0]
            return {
                "date": record["Date"].strftime("%Y-%m-%d"),
                "price": float(record["Open"]),
                "source": "same_open",
            }

        if fill_policy == "same_close":
            row = history.loc[history["Date"] <= signal_ts].tail(1)
            if row.empty:
                return None
            record = row.iloc[0]
            return {
                "date": record["Date"].strftime("%Y-%m-%d"),
                "price": float(record["Close"]),
                "source": "same_close",
            }

        row = history.loc[history["Date"] > signal_ts].head(1)
        if row.empty:
            return None
        record = row.iloc[0]
        return {
            "date": record["Date"].strftime("%Y-%m-%d"),
            "price": float(record["Open"]),
            "source": "next_open",
        }

    def _lookup_mark(self, symbol: str, as_of_date: str, use_open_if_exact: bool = False) -> Dict[str, Any]:
        history = self._load_history(symbol)
        if history.empty:
            return {"price": None, "source": None, "date": None}

        as_of_ts = _parse_date(as_of_date)
        if use_open_if_exact:
            exact_row = history.loc[history["Date"] == as_of_ts].head(1)
            if not exact_row.empty:
                record = exact_row.iloc[0]
                return {
                    "price": float(record["Open"]),
                    "source": "open",
                    "date": record["Date"].strftime("%Y-%m-%d"),
                }

        row = history.loc[history["Date"] <= as_of_ts].tail(1)
        if row.empty:
            return {"price": None, "source": None, "date": None}

        record = row.iloc[0]
        return {
            "price": float(record["Close"]),
            "source": "close",
            "date": record["Date"].strftime("%Y-%m-%d"),
        }

    def _load_history(self, symbol: str) -> pd.DataFrame:
        history = self.market_data_loader(symbol)
        if history.empty:
            return history

        normalized = history.copy()
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.normalize()
        normalized = normalized.dropna(subset=["Date"]).sort_values("Date")
        return normalized.reset_index(drop=True)


def format_execution_summary(
    execution: Optional[Dict[str, Any]],
    portfolio: Optional[Dict[str, Any]],
) -> str:
    """Render a markdown summary of the latest simulated execution and account state."""
    if not execution:
        return "### Simulated Exchange\nNo simulated execution information was recorded."

    lines = [
        "### Simulated Exchange",
        f"- Signal: `{execution.get('signal', 'HOLD')}`",
        f"- Status: `{execution.get('status', 'unknown')}`",
        f"- Requested trade date: `{execution.get('requested_trade_date', 'N/A')}`",
    ]

    if execution.get("execution_date"):
        lines.append(f"- Execution date: `{execution['execution_date']}`")
    if execution.get("execution_price") is not None:
        lines.append(
            f"- Execution price: `{execution['execution_price']}` ({execution.get('execution_price_source', 'unknown')})"
        )
    if execution.get("quantity"):
        lines.append(
            f"- Filled quantity: `{execution['quantity']}` shares "
            f"(signed delta `{execution.get('signed_quantity', 0)}`)"
        )
    if execution.get("gross_notional"):
        lines.append(f"- Gross notional: `{execution['gross_notional']}`")
    if execution.get("fees"):
        lines.append(f"- Fees: `{execution['fees']}`")
    lines.append(f"- Reason: {execution.get('reason', 'N/A')}")

    if portfolio:
        lines.extend(
            [
                "",
                "### Portfolio Snapshot",
                f"- As of: `{portfolio.get('as_of_date', 'N/A')}`",
                f"- Cash: `{portfolio.get('cash', 0.0)}`",
                f"- Total market value: `{portfolio.get('total_market_value', 0.0)}`",
                f"- Total equity: `{portfolio.get('total_equity', 0.0)}`",
                f"- Realized PnL: `{portfolio.get('total_realized_pnl', 0.0)}`",
                f"- Unrealized PnL: `{portfolio.get('total_unrealized_pnl', 0.0)}`",
                f"- Trade count: `{portfolio.get('trade_count', 0)}`",
            ]
        )
        positions = portfolio.get("positions", {})
        if positions:
            lines.append("- Open positions:")
            for symbol, position in positions.items():
                lines.append(
                    "  - "
                    f"`{symbol}`: qty `{position.get('quantity', 0)}`, "
                    f"side `{position.get('side', 'flat')}`, "
                    f"avg cost `{position.get('average_cost', 0.0)}`, "
                    f"mark `{position.get('mark_price', 'N/A')}` "
                    f"({position.get('mark_price_source', 'N/A')} {position.get('mark_price_date', 'N/A')})"
                )
        else:
            lines.append("- Open positions: none")

    return "\n".join(lines)
