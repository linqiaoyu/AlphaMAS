import pandas as pd
import pytest

from tradingagents.backtesting.data import InMemoryDataProvider
from tradingagents.backtesting.engine import WeeklyBacktestEngine
from tradingagents.backtesting.strategies import ScriptedStrategy


def market(xnys, *, dividend=0.0, split=0.0):
    sessions = xnys.sessions("2023-12-01", "2024-01-12").tz_localize(None)
    frame = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
        "Volume": 1000.0, "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=sessions)
    if dividend:
        frame.loc["2024-01-09", "Dividends"] = dividend
    if split:
        frame.loc["2024-01-09", "Stock Splits"] = split
        frame.loc["2024-01-09":, ["Open", "High", "Low", "Close"]] /= split
    return InMemoryDataProvider({"TEST": frame})


def run(provider, xnys):
    return WeeklyBacktestEngine(data_provider=provider, schedule=xnys).run(
        symbol="TEST", first_week="2024-01-01", final_week="2024-01-01",
        final_valuation_session="2024-01-12",
        strategy=ScriptedStrategy({"2024-01-05": "BUY"}), experiment_id="actions",
    )


def test_dividend_credits_prior_close_holder_before_orders(xnys):
    result = run(market(xnys, dividend=1.0), xnys)
    event = result.corporate_action_events.iloc[0]
    quantity = result.daily_equity.set_index("session").loc["2024-01-08", "quantity"]
    assert event["type"] == "dividend"
    assert event["quantity_eligible"] == pytest.approx(quantity)
    assert event["cash_effect"] == pytest.approx(quantity)
    assert result.metrics["cumulative_dividends"] == pytest.approx(quantity)
    assert result.daily_equity.iloc[-1]["cumulative_dividends"] == pytest.approx(quantity)


def test_split_changes_units_but_not_market_value(xnys):
    result = run(market(xnys, split=2.0), xnys)
    daily = result.daily_equity.set_index("session")
    assert daily.loc["2024-01-09", "quantity"] == pytest.approx(
        daily.loc["2024-01-08", "quantity"] * 2
    )
    assert daily.loc["2024-01-09", "average_entry_price"] == pytest.approx(
        daily.loc["2024-01-08", "average_entry_price"] / 2
    )
    assert daily.loc["2024-01-09", "market_value"] == pytest.approx(
        daily.loc["2024-01-08", "market_value"]
    )
    assert result.corporate_action_events.iloc[0]["type"] == "split"
