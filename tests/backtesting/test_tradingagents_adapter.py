from datetime import datetime, timezone

import pandas as pd

from tradingagents.backtesting.cache import DecisionCache
from tradingagents.backtesting.models import DecisionStatus, PortfolioSnapshot
from tradingagents.backtesting.strategies import TradingAgentsStrategy, strict_action


def snapshot():
    return PortfolioSnapshot(
        datetime.now(timezone.utc), "2024-01-05", "AAPL", 100_000, 0, 100,
        0, 100_000, 0, 0, 0, 0, 0, 0, 100_000,
    )


def test_adapter_passes_exact_historical_context_and_truncates_evidence():
    class Graph:
        def propagate(self, symbol, trade_date, **kwargs):
            self.symbol, self.trade_date, self.kwargs = symbol, trade_date, kwargs
            return {"final_trade_decision": "**Rating**: Buy"}, "Buy"

    graph = Graph()
    strategy = TradingAgentsStrategy(graph)
    close = datetime(2024, 1, 5, 21, tzinfo=timezone.utc)
    history = pd.DataFrame({"Close": [99, 100]}, index=pd.to_datetime(["2024-01-04", "2024-01-05"]))
    decision = strategy.decide(
        symbol="AAPL", decision_session="2024-01-05", decision_time=close,
        market_history=history, portfolio_snapshot=snapshot(),
        context={"experiment_id": "M0", "point_in_time": True},
    )
    context = graph.kwargs["run_context"]
    assert context.mode == "historical"
    assert context.as_of == close
    assert context.as_of.hour == 21
    assert context.experiment_id == "M0"
    assert decision.action.value == "BUY"


def test_adapter_failure_is_explicit_not_hold():
    class Graph:
        def propagate(self, *args, **kwargs):
            raise RuntimeError("LLM unavailable")

    decision = TradingAgentsStrategy(Graph()).decide(
        symbol="AAPL", decision_session="2024-01-05",
        decision_time=datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
        market_history=pd.DataFrame({"Close": [100]}), portfolio_snapshot=snapshot(),
        context={"experiment_id": "M0"},
    )
    assert decision.status is DecisionStatus.FAILED
    assert decision.action is None
    assert decision.target_weight is None


def test_strict_parser_rejects_unknown_and_conflicts():
    for raw in ("looks interesting", "Action: BUY\nAction: SELL"):
        try:
            strict_action(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous/unknown text was accepted")
    assert strict_action("Rating: Overweight").value == "BUY"
    assert strict_action("Rating: Underweight").value == "SELL"


def test_adapter_exact_cache_hit_skips_graph(tmp_path):
    class Graph:
        config = {"quick_think_llm": "q", "deep_think_llm": "d", "llm_provider": "mock"}
        selected_analysts = ("market",)
        calls = 0

        def propagate(self, *args, **kwargs):
            self.calls += 1
            return {"final_trade_decision": "Rating: BUY"}, "Buy"

    graph = Graph()
    strategy = TradingAgentsStrategy(
        graph, cache=DecisionCache(tmp_path),
        cache_config={"git_commit_sha": "abc", "prompt_config_version": "v1"},
    )
    kwargs = {
        "symbol": "AAPL", "decision_session": "2024-01-05",
        "decision_time": datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
        "market_history": pd.DataFrame({"Close": [100]}),
        "portfolio_snapshot": snapshot(), "context": {"experiment_id": "M0"},
    }
    assert strategy.decide(**kwargs).status is DecisionStatus.SUCCESS
    assert strategy.decide(**kwargs).status is DecisionStatus.CACHED
    assert graph.calls == 1
