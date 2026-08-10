from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.data import InMemoryDataProvider


@pytest.fixture(scope="session")
def xnys() -> ExchangeSchedule:
    return ExchangeSchedule()


@pytest.fixture
def synthetic_provider(xnys: ExchangeSchedule) -> InMemoryDataProvider:
    sessions = xnys.sessions("2022-12-01", "2024-07-05").tz_localize(None)
    close = pd.Series([100 + index * 0.1 for index in range(len(sessions))], index=sessions)
    frame = pd.DataFrame({
        "Open": close + 0.5, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": 1000,
    })
    return InMemoryDataProvider(dict.fromkeys(("TEST", "AAPL", "AMZN", "JPM"), frame))
