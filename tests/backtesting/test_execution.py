from datetime import datetime, timezone

import pytest

from tradingagents.backtesting.execution import Broker
from tradingagents.backtesting.models import Order
from tradingagents.backtesting.portfolio import Portfolio


def order(target):
    return Order("o1", "TEST", datetime.now(timezone.utc), "2024-01-08", target, 0, "BUY")


def test_fractional_buy_reserves_fee_and_uses_slipped_open():
    portfolio = Portfolio("TEST", 100_000)
    broker = Broker(commission_bps=5, slippage_bps=5, fractional_shares=True)
    fill = broker.execute(order(1.0), portfolio, 110, datetime.now(timezone.utc))
    assert fill is not None
    assert fill.fill_price == pytest.approx(110 * 1.0005)
    expected = 100_000 / (fill.fill_price * 1.0005)
    assert fill.quantity == pytest.approx(expected)
    assert portfolio.cash == pytest.approx(0, abs=1e-8)
    assert portfolio.cumulative_cost == pytest.approx(fill.notional * 0.0005)


def test_sell_slippage_commission_and_noop():
    portfolio = Portfolio("TEST")
    broker = Broker(commission_bps=5, slippage_bps=5)
    first = order(1.0)
    broker.execute(first, portfolio, 100, datetime.now(timezone.utc))
    noop = Order("o2", "TEST", datetime.now(timezone.utc), "2024-01-09", 1, 1, "BUY")
    assert broker.execute(noop, portfolio, 120, datetime.now(timezone.utc)) is None
    old_cost = portfolio.cumulative_cost
    sell = Order("o3", "TEST", datetime.now(timezone.utc), "2024-01-10", 0, 1, "SELL")
    fill = broker.execute(sell, portfolio, 120, datetime.now(timezone.utc))
    assert fill is not None
    assert fill.fill_price == pytest.approx(120 * 0.9995)
    assert portfolio.quantity == 0
    assert portfolio.cumulative_cost > old_cost


def test_round_trip_realized_pnl_includes_both_commissions_once():
    portfolio = Portfolio("TEST", initial_cash=101.0)
    broker = Broker(commission_bps=100, slippage_bps=0, fractional_shares=True)
    execution_time = datetime(2024, 1, 8, tzinfo=timezone.utc)

    buy_fill = broker.execute(order(1.0), portfolio, 100.0, execution_time)
    assert buy_fill is not None
    assert buy_fill.quantity == pytest.approx(1.0)
    assert buy_fill.commission == pytest.approx(1.0)

    open_snapshot = portfolio.snapshot("2024-01-08", execution_time, 110.0)
    assert open_snapshot.realized_pnl == pytest.approx(0.0)
    assert open_snapshot.unrealized_pnl == pytest.approx(9.0)
    assert open_snapshot.equity == pytest.approx(
        portfolio.initial_cash
        + open_snapshot.realized_pnl
        + open_snapshot.unrealized_pnl
        + open_snapshot.cumulative_dividends
    )

    sell = Order(
        "o2", "TEST", execution_time, "2024-01-09", 0.0, 1.0, "SELL",
    )
    sell_fill = broker.execute(sell, portfolio, 120.0, execution_time)
    assert sell_fill is not None
    assert sell_fill.commission == pytest.approx(1.2)

    # Gross price P&L is $20; the $1.00 entry and $1.20 exit commissions
    # are each attributed once when the round trip is completed.
    assert portfolio.realized_pnl == pytest.approx(17.8)
    assert portfolio.cumulative_cost == pytest.approx(2.2)
    assert portfolio.cumulative_cost == pytest.approx(
        buy_fill.commission + sell_fill.commission
    )

    closed_snapshot = portfolio.snapshot("2024-01-09", execution_time, 120.0)
    assert closed_snapshot.unrealized_pnl == pytest.approx(0.0)
    assert closed_snapshot.equity == pytest.approx(118.8)
    assert closed_snapshot.equity == pytest.approx(
        portfolio.initial_cash + portfolio.realized_pnl
    )


def test_round_trip_commission_basis_survives_dividend_and_split():
    portfolio = Portfolio("TEST", initial_cash=101.0)
    broker = Broker(commission_bps=100, slippage_bps=0, fractional_shares=True)
    execution_time = datetime(2024, 1, 8, tzinfo=timezone.utc)

    buy_fill = broker.execute(order(1.0), portfolio, 100.0, execution_time)
    assert buy_fill is not None
    assert portfolio.apply_dividend(2.0) == pytest.approx(2.0)
    portfolio.apply_split(2.0)

    open_snapshot = portfolio.snapshot("2024-01-09", execution_time, 55.0)
    assert open_snapshot.quantity == pytest.approx(2.0)
    assert open_snapshot.average_entry_price == pytest.approx(50.0)
    assert open_snapshot.unrealized_pnl == pytest.approx(9.0)
    assert open_snapshot.cumulative_dividends == pytest.approx(2.0)
    assert open_snapshot.equity == pytest.approx(
        portfolio.initial_cash
        + open_snapshot.realized_pnl
        + open_snapshot.unrealized_pnl
        + open_snapshot.cumulative_dividends
    )

    sell = Order(
        "o2", "TEST", execution_time, "2024-01-10", 0.0, 1.0, "SELL",
    )
    sell_fill = broker.execute(sell, portfolio, 60.0, execution_time)
    assert sell_fill is not None
    assert sell_fill.quantity == pytest.approx(-2.0)
    assert sell_fill.commission == pytest.approx(1.2)
    assert portfolio.realized_pnl == pytest.approx(17.8)
    assert portfolio.cumulative_cost == pytest.approx(2.2)

    closed_snapshot = portfolio.snapshot("2024-01-10", execution_time, 60.0)
    assert closed_snapshot.equity == pytest.approx(120.8)
    assert closed_snapshot.equity == pytest.approx(
        portfolio.initial_cash
        + closed_snapshot.realized_pnl
        + closed_snapshot.unrealized_pnl
        + closed_snapshot.cumulative_dividends
    )


def test_missing_open_rejects_without_close_substitution():
    portfolio = Portfolio("TEST")
    broker = Broker()
    pending = order(1.0)
    assert broker.execute(pending, portfolio, None, datetime.now(timezone.utc)) is None
    assert pending.status == "rejected"
    assert portfolio.cash == 100_000
    assert portfolio.quantity == 0
