"""Offline point-in-time contract tests for historical single-date runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.analysts.news_analyst import build_news_analyst_system_message
from tradingagents.agents.utils.agent_utils import resolve_instrument_identity
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import get_balance_sheet, get_fundamentals
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.prediction_markets_tools import get_prediction_markets
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
from tradingagents.dataflows import (
    alpha_vantage_common as av_common,
    alpha_vantage_fundamentals as avf,
    alpha_vantage_stock as av_stock,
    polymarket,
    reddit,
    stocktwits,
    y_finance,
    yfinance_news as ynews,
)
from tradingagents.dataflows.config import set_config
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.capabilities import DataCapability
from tradingagents.runtime.run_context import (
    AuditTrail,
    RunContext,
    activate_run_context,
    create_run_context,
)


def _historical_audit() -> tuple[RunContext, AuditTrail]:
    context = RunContext.historical(
        "2024-03-15",
        generated_at=datetime(2026, 8, 5, 12, tzinfo=timezone.utc),
    )
    return context, AuditTrail(context)


@pytest.mark.unit
def test_historical_context_requires_explicit_as_of_and_separates_times():
    with pytest.raises(ValueError, match="requires an explicit as_of"):
        create_run_context("historical")

    context = create_run_context(
        "historical",
        as_of="2024-03-15",
        generated_at=datetime(2026, 8, 5, 12, tzinfo=timezone.utc),
    )
    assert context.mode == "historical"
    assert context.as_of.isoformat() == "2024-03-15T23:59:59.999999+00:00"
    assert context.generated_at.isoformat() == "2026-08-05T12:00:00+00:00"


@pytest.mark.unit
def test_future_ohlcv_is_filtered_before_agent_and_indicator_evidence(monkeypatch, tmp_path):
    context, audit = _historical_audit()
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-03-14", "2024-03-15", "2024-03-16"]),
            "Open": [100.0, 101.0, 999.0],
            "High": [101.0, 102.0, 1000.0],
            "Low": [99.0, 100.0, 998.0],
            "Close": [100.5, 101.5, 999.5],
            "Volume": [100, 110, 999],
        }
    )
    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _: MagicMock(history=lambda **_: frame.copy()))
    set_config({"data_cache_dir": str(tmp_path)})

    with activate_run_context(context, audit):
        output = y_finance.get_YFin_data_online("SPY", "2024-03-14", "2024-03-15")
        assert "2024-03-14" in output
        assert "2024-03-15" in output
        assert "2024-03-16" not in output

        # The cached path is also tested because technical indicators consume it.
        monkeypatch.setattr(y_finance, "load_ohlcv", lambda *_: frame.copy())
        indicator_output = y_finance.get_stock_stats_indicators_window(
            "SPY", "close_10_ema", "2024-03-15", 3
        )
        assert "2024-03-16" not in indicator_output
    assert any(r["source_name"] == "yfinance.ohlcv_cache" for r in audit.records) is False

    # The direct online path is audited by the router in normal Graph runs;
    # this test additionally proves the strict content boundary itself.


@pytest.mark.unit
def test_news_excludes_future_and_undated_articles_and_keeps_window(monkeypatch):
    context, audit = _historical_audit()
    articles = [
        {
            "title": "PAST EVENT",
            "publisher": "P",
            "link": "past",
            "providerPublishTime": int(datetime(2024, 3, 14, tzinfo=timezone.utc).timestamp()),
        },
        {
            "title": "FUTURE EVENT",
            "publisher": "P",
            "link": "future",
            "providerPublishTime": int(datetime(2024, 3, 16, tzinfo=timezone.utc).timestamp()),
        },
        {"title": "UNDATED EVENT", "publisher": "P", "link": "undated"},
    ]

    class DummyTicker:
        def get_news(self, count):
            return articles

    monkeypatch.setattr(ynews.yf, "Ticker", lambda _: DummyTicker())
    with activate_run_context(context, audit):
        output = ynews.get_news_yfinance("SPY", "2024-03-08", "2024-03-15")

    assert "PAST EVENT" in output
    assert "FUTURE EVENT" not in output
    assert "UNDATED EVENT" not in output
    assert "2024-03-08 to 2024-03-15" in output
    assert any(r["status"] == "used" and r["capability"] == DataCapability.APPROXIMATE.value for r in audit.records)


@pytest.mark.unit
@pytest.mark.parametrize(
    "csv_data",
    [
        'Date,Close\n"unterminated,1.0\n',
        "Open,Close\n2024-03-15,1.0\n",
        "Date,Close\nnot-a-date,1.0\n",
    ],
    ids=["malformed", "missing-date-column", "unparseable-date"],
)
def test_alpha_vantage_csv_filter_failure_is_fail_closed(csv_data):
    context, audit = _historical_audit()
    with activate_run_context(context, audit):
        output = av_common._filter_csv_by_date_range(
            csv_data, "2024-03-14", "2024-03-15"
        )

    assert output.startswith("DATA_UNAVAILABLE_IN_HISTORICAL_MODE:")
    assert "1.0" not in output
    record = audit.records[-1]
    assert record["source_name"] == "alpha_vantage.get_stock_data"
    assert record["mode"] == "historical"
    assert record["historical_as_of"].startswith("2024-03-15")
    assert record["status"] == "unavailable"
    assert record["requested_start"] == "2024-03-14"
    assert record["requested_end"] == "2024-03-15"
    assert "cannot prove" in record["reason"]
    assert record["generated_at"]


@pytest.mark.unit
def test_alpha_vantage_future_rows_are_filtered_at_historical_as_of(monkeypatch):
    response = (
        "timestamp,open,high,low,close,volume\n"
        "2024-03-15,100,101,99,100.5,100\n"
        "2024-03-16,999,1000,998,999.5,999\n"
    )
    monkeypatch.setattr(av_stock, "_make_api_request", lambda *_: response)
    context, audit = _historical_audit()

    with activate_run_context(context, audit):
        output = av_stock.get_stock("SPY", "2024-03-14", "2024-03-16")

    assert "2024-03-15" in output
    assert "2024-03-16" not in output
    assert "999.5" not in output


@pytest.mark.unit
def test_alpha_vantage_live_filter_keeps_compatibility_on_failure():
    csv_data = "Open,Close\n2024-03-15,1.0\n"
    with activate_run_context(
        RunContext.live(generated_at=datetime(2026, 8, 5, 12, tzinfo=timezone.utc))
    ):
        output = av_common._filter_csv_by_date_range(
            csv_data, "2024-03-14", "2024-03-15"
        )

    assert output == csv_data


@pytest.mark.unit
def test_yfinance_flat_news_timestamp_is_utc_safe(monkeypatch):
    context = RunContext.historical(
        "2025-07-18", generated_at=datetime(2026, 8, 5, 12, tzinfo=timezone.utc)
    )
    articles = [
        {
            "title": "IN WINDOW",
            "publisher": "P",
            "providerPublishTime": int(
                datetime(2025, 7, 18, 23, 30, tzinfo=timezone.utc).timestamp()
            ),
        },
        {
            "title": "AFTER CUTOFF",
            "publisher": "P",
            "providerPublishTime": int(
                datetime(2025, 7, 19, 0, 30, tzinfo=timezone.utc).timestamp()
            ),
        },
    ]

    class DummyTicker:
        def get_news(self, count):
            return articles

    monkeypatch.setattr(ynews.yf, "Ticker", lambda _: DummyTicker())
    parsed = ynews._extract_article_data(articles[0])["pub_date"]
    nested = ynews._extract_article_data(
        {"content": {"title": "NESTED", "pubDate": "2025-07-18T23:30:00Z"}}
    )["pub_date"]
    assert parsed == datetime(2025, 7, 18, 23, 30, tzinfo=timezone.utc)
    assert nested == parsed
    assert parsed.tzinfo is timezone.utc

    with activate_run_context(context):
        output = ynews.get_news_yfinance("SPY", "2025-07-18", "2025-07-18")

    assert "IN WINDOW" in output
    assert "AFTER CUTOFF" not in output


@pytest.mark.unit
def test_news_analyst_prompt_uses_relative_time_language():
    prompt = build_news_analyst_system_message("company").lower()
    assert "recession 2026" not in prompt
    assert not re.search(r"\b20\d{2}\b", prompt)
    assert "next 12 months" in prompt


@pytest.mark.unit
def test_live_only_sources_are_blocked_before_network_calls():
    context, audit = _historical_audit()
    with (
        activate_run_context(context, audit),
        patch.object(stocktwits, "urlopen") as stocktwits_open,
        patch.object(reddit, "_fetch_subreddit") as reddit_fetch,
        patch.object(polymarket, "_request") as polymarket_request,
    ):
        stocktwits_out = stocktwits.fetch_stocktwits_messages("SPY")
        reddit_out = reddit.fetch_reddit_posts("SPY")
        polymarket_out = polymarket.get_prediction_markets("Fed rate cut")

    stocktwits_open.assert_not_called()
    reddit_fetch.assert_not_called()
    polymarket_request.assert_not_called()
    for output, source in (
        (stocktwits_out, "stocktwits"),
        (reddit_out, "reddit"),
        (polymarket_out, "polymarket"),
    ):
        assert output.startswith("DATA_UNAVAILABLE_IN_HISTORICAL_MODE")
        assert any(r["source_name"] == source and r["status"] == "blocked" for r in audit.records)


@pytest.mark.unit
def test_ticker_info_is_not_called_for_historical_fundamentals(monkeypatch):
    context, audit = _historical_audit()
    ticker = MagicMock()
    monkeypatch.setattr(y_finance.yf, "Ticker", ticker)
    resolve_instrument_identity.cache_clear()
    with activate_run_context(context, audit):
        output = y_finance.get_fundamentals("SPY", "2024-03-15")
        identity = resolve_instrument_identity("SPY")

    ticker.assert_not_called()
    assert "52 Week High" not in output
    assert "Forward PE" not in output
    assert identity == {}


@pytest.mark.unit
def test_financial_statements_require_publication_time(monkeypatch):
    context, audit = _historical_audit()
    payload = json.dumps(
        {
            "symbol": "SPY",
            "quarterlyReports": [
                {
                    "fiscalDateEnding": "2023-12-31",
                    "reportedDate": "2024-04-30",  # after historical_as_of
                    "totalAssets": "future-publication",
                },
                {
                    "fiscalDateEnding": "2023-09-30",
                    "reportedDate": "2023-11-01",
                    "totalAssets": "available",
                },
            ],
        }
    )
    monkeypatch.setattr(avf, "_make_api_request", lambda *_: payload)
    with activate_run_context(context, audit):
        output = avf.get_income_statement("SPY", curr_date="2024-03-15")

    parsed = json.loads(output)
    reports = parsed["quarterlyReports"]
    assert len(reports) == 1
    assert reports[0]["totalAssets"] == "available"


@pytest.mark.unit
def test_memory_outcome_stays_pending_until_five_trading_days(monkeypatch, tmp_path):
    context, audit = _historical_audit()
    prices = pd.DataFrame(
        {"Close": [100, 101, 102, 103, 104, 105, 106]},
        index=pd.to_datetime([
            "2024-03-15", "2024-03-18", "2024-03-19", "2024-03-20",
            "2024-03-21", "2024-03-22", "2024-03-25",
        ]),
    )
    requested_ends = []

    class DummyTicker:
        def history(self, *, start, end):
            requested_ends.append(end)
            return prices.copy()

    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"benchmark_ticker": None, "benchmark_map": {"": "SPY"}}
    graph.reflector = MagicMock()
    memory_path = Path(tmp_path) / "historical.md"
    graph.memory_log = TradingMemoryLog(
        {"memory_log_path": str(memory_path), "historical_as_of": "2024-03-18"}
    )
    graph.memory_log.store_decision("SPY", "2024-03-15", "Rating: Buy")
    monkeypatch.setattr("tradingagents.graph.trading_graph.yf.Ticker", lambda _: DummyTicker())

    early_context = RunContext.historical("2024-03-18", generated_at=context.generated_at)
    with activate_run_context(early_context, audit):
        graph._resolve_pending_entries("SPY")
    assert graph.memory_log.get_pending_entries()
    graph.reflector.reflect_on_final_decision.assert_not_called()
    assert all(end <= "2024-03-19" for end in requested_ends)

    mature_context = RunContext.historical("2024-03-22", generated_at=context.generated_at)
    graph.memory_log = TradingMemoryLog(
        {"memory_log_path": str(memory_path), "historical_as_of": "2024-03-22"}
    )
    graph.reflector.reflect_on_final_decision.return_value = "Mature reflection"
    with activate_run_context(mature_context, audit):
        graph._resolve_pending_entries("SPY")
    assert graph.memory_log.get_pending_entries() == []
    assert graph.reflector.reflect_on_final_decision.called
    assert all(end <= "2024-03-23" for end in requested_ends)


@pytest.mark.unit
def test_historical_memory_namespace_isolated_by_date(tmp_path):
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"data_cache_dir": str(tmp_path), "results_dir": str(tmp_path)}
    first = graph._historical_memory_config("SPY", RunContext.historical("2024-03-15"))
    second = graph._historical_memory_config("SPY", RunContext.historical("2024-03-18"))
    assert first["memory_log_path"] != second["memory_log_path"]
    assert str(tmp_path) not in str(Path.home() / ".tradingagents" / "memory")


@pytest.mark.unit
def test_reusing_graph_does_not_leave_historical_memory_attached(tmp_path):
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
        "memory_log_path": str(tmp_path / "live.md"),
        "checkpoint_enabled": False,
    }
    graph.memory_log = TradingMemoryLog(graph.config)
    graph._checkpointer_ctx = None
    graph._run_graph = lambda *args, **kwargs: {"ok": True}

    graph.propagate("SPY", "2024-03-15", mode="historical", historical_as_of="2024-03-15")
    historical_path = graph.memory_log._log_path
    graph.propagate("SPY", "2024-03-15", mode="live")
    assert historical_path != graph.memory_log._log_path
    assert graph.memory_log._log_path == Path(tmp_path / "live.md")


@pytest.mark.unit
def test_future_mutation_does_not_change_historical_evidence(monkeypatch):
    context, _ = _historical_audit()
    base = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 999.0],
            "High": [101.0, 102.0, 1000.0],
            "Low": [99.0, 100.0, 998.0],
            "Close": [100.5, 101.5, 999.5],
            "Volume": [100, 110, 999],
        },
        index=pd.to_datetime(["2024-03-14", "2024-03-15", "2024-03-16"]),
    )
    mutated = base.copy()
    mutated.loc[pd.Timestamp("2024-03-16"), "Close"] = 1_000_000.0

    def run(frame):
        monkeypatch.setattr(y_finance.yf, "Ticker", lambda _: MagicMock(history=lambda **_: frame.copy()))
        with activate_run_context(context):
            return y_finance.get_YFin_data_online("SPY", "2024-03-14", "2024-03-15")

    first = run(base)
    second = run(mutated)
    assert first == second


@pytest.mark.unit
def test_live_mode_still_calls_social_source(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps({"messages": []}).encode()
    monkeypatch.setattr(stocktwits, "urlopen", MagicMock(return_value=response))
    output = stocktwits.fetch_stocktwits_messages("SPY")
    assert "no StockTwits messages" in output
    stocktwits.urlopen.assert_called_once()


@pytest.mark.unit
def test_spy_historical_tool_assembly_writes_complete_audit(monkeypatch, tmp_path):
    """Mocked SPY 2024-03-15 assembly path never uses current-only evidence."""
    context, audit = _historical_audit()
    ohlcv = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-03-14", "2024-03-15", "2024-03-16"]),
            "Open": [100.0, 101.0, 999.0],
            "High": [101.0, 102.0, 1000.0],
            "Low": [99.0, 100.0, 998.0],
            "Close": [100.5, 101.5, 999.5],
            "Volume": [100, 110, 999],
        }
    )

    class DummyTicker:
        def history(self, **_):
            return ohlcv.set_index("Date").copy()

        def get_news(self, count):
            return [
                {
                    "title": "PAST NEWS",
                    "publisher": "P",
                    "providerPublishTime": int(datetime(2024, 3, 14, tzinfo=timezone.utc).timestamp()),
                },
                {
                    "title": "FUTURE NEWS",
                    "publisher": "P",
                    "providerPublishTime": int(datetime(2024, 3, 16, tzinfo=timezone.utc).timestamp()),
                },
            ]

    class DummySearch:
        news = DummyTicker().get_news(10)

    monkeypatch.setattr(y_finance.yf, "Ticker", lambda _: DummyTicker())
    monkeypatch.setattr(ynews.yf, "Ticker", lambda _: DummyTicker())
    monkeypatch.setattr(ynews.yf, "Search", lambda **_: DummySearch())
    monkeypatch.setattr(y_finance, "load_ohlcv", lambda *_: ohlcv.copy())
    from tradingagents.dataflows import market_data_validator as validation

    monkeypatch.setattr(validation, "load_ohlcv", lambda *_: ohlcv.copy())
    set_config({"data_cache_dir": str(tmp_path / "cache")})

    with activate_run_context(context, audit):
        evidence = [
            get_stock_data.func("SPY", "2024-03-14", "2024-03-15"),
            get_indicators.func("SPY", "close_10_ema", "2024-03-15", 3),
            get_verified_market_snapshot.func("SPY", "2024-03-15", 3),
            get_news.func("SPY", "2024-03-08", "2024-03-15"),
            get_global_news.func("2024-03-15", 7, 10),
            get_fundamentals.func("SPY", "2024-03-15"),
            get_balance_sheet.func("SPY", "quarterly", "2024-03-15"),
            get_insider_transactions.func("SPY"),
            get_macro_indicators.func("cpi", "2024-03-15", 30),
            get_prediction_markets.func("Fed rate cut", 5),
            stocktwits.fetch_stocktwits_messages("SPY"),
            reddit.fetch_reddit_posts("SPY"),
        ]
        audit_path = audit.write(tmp_path / "historical_data_audit.json")

    joined = "\n".join(evidence)
    assert "2024-03-16" not in joined
    assert "FUTURE NEWS" not in joined
    assert all("DATA_UNAVAILABLE_IN_HISTORICAL_MODE" in item for item in evidence[5:])
    document = json.loads(audit_path.read_text())
    names = {record["source_name"] for record in document["sources"]}
    assert {"stocktwits", "reddit"} <= names
    assert any(name.startswith("polymarket") for name in names)
    assert any(name in {"yfinance.ticker_info", "yfinance.get_fundamentals"} for name in names)
    assert document["run_context"]["historical_as_of"].startswith("2024-03-15")
