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


def test_missing_open_rejects_without_close_substitution():
    portfolio = Portfolio("TEST")
    broker = Broker()
    pending = order(1.0)
    assert broker.execute(pending, portfolio, None, datetime.now(timezone.utc)) is None
    assert pending.status == "rejected"
    assert portfolio.cash == 100_000
    assert portfolio.quantity == 0
