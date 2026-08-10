import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from tradingagents.backtesting.cache import DecisionCache, cache_key
from tradingagents.backtesting.engine import WeeklyBacktestEngine
from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.backtesting.strategies import TradingAgentsStrategy


def test_cache_key_changes_with_state_model_and_git():
    base = {"portfolio": "a", "model": "m", "git_sha": "1"}
    key = cache_key(base)
    assert key != cache_key({**base, "portfolio": "b"})
    assert key != cache_key({**base, "model": "x"})
    assert key != cache_key({**base, "git_sha": "2"})


def _strategy_payload(
    *, graph_overrides=None, cache_overrides=None, analysts=None, context_overrides=None,
):
    class Graph:
        config = {
            "research_depth": "medium",
            "quick_think_llm": "deepseek-v4-flash",
            "deep_think_llm": "deepseek-v4-flash",
            "llm_provider": "deepseek",
            "deepseek_thinking": "disabled",
            "temperature": 0.0,
            "max_debate_rounds": 3,
            "max_risk_discuss_rounds": 3,
            "output_language": "English",
            "data_vendors": {"news_data": "yfinance"},
            "tool_vendors": {},
            "memory_mode": "experiment",
            "memory_holding_horizon_sessions": 5,
            "graph_config_sha256": "graph-a",
        }
        selected_analysts = ("market", "social", "news", "fundamentals")

    graph = Graph()
    graph.config = {**graph.config, **(graph_overrides or {})}
    if analysts is not None:
        graph.selected_analysts = tuple(analysts)
    cache_config = {
        "git_commit_sha": "git-a",
        "prompt_config_version": "prompt-a",
        "memory_namespace_version": "memory-a",
        **(cache_overrides or {}),
    }
    strategy = TradingAgentsStrategy(graph, cache_config=cache_config)
    snapshot = PortfolioSnapshot(
        datetime(2024, 1, 5, 21, tzinfo=timezone.utc), "2024-01-05", "AAPL",
        100_000, 0, 100, 0, 100_000, 0, 0, 0, 0, 0, 0, 100_000,
    )
    return strategy._cache_payload({
        "symbol": "AAPL",
        "decision_time": datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
        "market_history": pd.DataFrame(
            {"Close": [99.0, 100.0]}, index=pd.to_datetime(["2024-01-04", "2024-01-05"])
        ),
        "portfolio_snapshot": snapshot,
        "context": {
            "experiment_id": "M0", "point_in_time": True, **(context_overrides or {})
        },
    })


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("quick_think_llm", "other-quick"),
        ("deep_think_llm", "other-deep"),
        ("llm_provider", "other-provider"),
        ("deepseek_thinking", "enabled"),
        ("temperature", 0.2),
        ("research_depth", "deep"),
        ("max_debate_rounds", 5),
        ("max_risk_discuss_rounds", 5),
        ("output_language", "Chinese"),
        ("data_vendors", {"news_data": "alpha_vantage"}),
        ("tool_vendors", {"get_news": "alpha_vantage"}),
        ("memory_mode", "disabled"),
        ("memory_holding_horizon_sessions", 3),
        ("graph_config_sha256", "graph-b"),
    ),
)
def test_decision_cache_identity_changes_with_research_graph_field(field, changed):
    base = cache_key(_strategy_payload())
    changed_key = cache_key(_strategy_payload(graph_overrides={field: changed}))
    assert changed_key != base


def test_changing_thinking_mode_causes_cache_miss():
    disabled = cache_key(_strategy_payload(graph_overrides={"deepseek_thinking": "disabled"}))
    enabled = cache_key(_strategy_payload(graph_overrides={"deepseek_thinking": "enabled"}))
    assert disabled != enabled


def test_decision_cache_identity_changes_with_selected_analysts():
    assert cache_key(_strategy_payload()) != cache_key(
        _strategy_payload(analysts=("market", "news"))
    )


def test_decision_cache_identity_changes_with_point_in_time_mode():
    assert cache_key(_strategy_payload()) != cache_key(
        _strategy_payload(context_overrides={"point_in_time": False})
    )


def test_decision_cache_identity_changes_with_memory_lineage():
    assert cache_key(_strategy_payload(
        graph_overrides={"historical_memory_lineage_id": "run-a"}
    )) != cache_key(_strategy_payload(
        graph_overrides={"historical_memory_lineage_id": "run-b"}
    ))


@pytest.mark.parametrize(
    ("field", "changed"),
    (("git_commit_sha", "git-b"), ("prompt_config_version", "prompt-b")),
)
def test_decision_cache_identity_changes_with_version_field(field, changed):
    assert cache_key(_strategy_payload()) != cache_key(
        _strategy_payload(cache_overrides={field: changed})
    )


def test_only_complete_success_is_a_hit(tmp_path):
    cache = DecisionCache(tmp_path)
    key = cache_key({"case": 1})
    assert cache.load(key) is None
    cache.save_success(key, {"action": "BUY"})
    assert cache.load(key) == {"action": "BUY"}
    (cache.case_dir(key) / "run_status.json").write_text("{", encoding="utf-8")
    assert cache.load(key) is None
    (cache.case_dir(key) / "run_status.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    assert cache.load(key) is None


@pytest.mark.parametrize(
    "missing_member", ("reports", "source_audit", "llm_usage"),
)
def test_formal_case_cache_requires_complete_origin_artifacts(tmp_path, missing_member):
    cache = DecisionCache(tmp_path / "cache")
    key = cache_key({"case": "formal"})
    report_source = tmp_path / "origin-reports"
    report_source.mkdir()
    (report_source / "complete_report.md").write_text("report", encoding="utf-8")
    cache.save_success(
        key,
        {"action": "BUY"},
        {
            "source_audit": {
                "sources": [{"source_name": "mock", "status": "used"}],
            },
            "llm_usage": [{"run_id": "origin-run", "total_tokens": 1}],
            "report_path": str(report_source),
        },
    )
    assert cache.load_bundle(
        key, require_artifacts=True, require_usage=True,
    ) is not None

    member = {
        "reports": cache.case_dir(key) / "reports" / "complete_report.md",
        "source_audit": cache.case_dir(key) / "source_audit.json",
        "llm_usage": cache.case_dir(key) / "llm_usage.json",
    }[missing_member]
    member.unlink()

    assert cache.load_bundle(
        key, require_artifacts=True, require_usage=True,
    ) is None


def test_replay_from_cache_skips_agent_and_rebuilds_same_portfolio(
    tmp_path, synthetic_provider, xnys,
):
    class Graph:
        config = {
            "quick_think_llm": "q", "deep_think_llm": "d", "llm_provider": "mock",
            "max_debate_rounds": 3,
        }
        selected_analysts = ("market",)
        calls = 0

        def propagate(self, *args, **kwargs):
            self.calls += 1
            return {"final_trade_decision": "Rating: BUY"}, "Buy"

    graph = Graph()
    cache = DecisionCache(tmp_path / "cache")

    def run():
        return WeeklyBacktestEngine(data_provider=synthetic_provider, schedule=xnys).run(
            symbol="TEST", first_week="2024-01-01", final_week="2024-01-15",
            final_valuation_session="2024-01-26",
            strategy=TradingAgentsStrategy(
                graph, cache=cache,
                cache_config={"git_commit_sha": "abc", "prompt_config_version": "v1"},
            ),
            experiment_id="resume",
        )

    uninterrupted = run()
    calls_after_first = graph.calls
    resumed = run()
    assert graph.calls == calls_after_first
    assert set(resumed.decisions["status"]) == {"cached"}
    pd.testing.assert_frame_equal(uninterrupted.daily_equity, resumed.daily_equity)
    pd.testing.assert_frame_equal(uninterrupted.orders, resumed.orders)
    pd.testing.assert_frame_equal(uninterrupted.fills, resumed.fills)
