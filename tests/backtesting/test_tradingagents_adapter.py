import json
from datetime import datetime, timezone
from pathlib import Path

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


def test_cache_hit_materializes_complete_case_artifacts_in_current_run(tmp_path):
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
            "memory_mode": "experiment",
            "memory_holding_horizon_sessions": 5,
            "graph_config_sha256": "a" * 64,
        }
        selected_analysts = ("market", "social", "news", "fundamentals")
        calls = 0

        def _audit_path(self, symbol, context):
            return tmp_path / "graph-audits" / f"{symbol}-{context.as_of.date()}.json"

        def propagate(self, symbol, trade_date, **kwargs):
            self.calls += 1
            audit = self._audit_path(symbol, kwargs["run_context"])
            audit.parent.mkdir(parents=True, exist_ok=True)
            audit.write_text(
                json.dumps({"sources": [{"source_name": "mock", "status": "used"}]}),
                encoding="utf-8",
            )
            return {"final_trade_decision": "Rating: BUY"}, "Buy"

        def save_reports(self, final_state, symbol, report_dir):
            del final_state, symbol
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            (Path(report_dir) / "complete_report.md").write_text(
                "offline report", encoding="utf-8"
            )

    graph = Graph()
    cache = DecisionCache(tmp_path / "cache")
    kwargs = {
        "symbol": "AAPL", "decision_session": "2024-01-05",
        "decision_time": datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
        "market_history": pd.DataFrame({"Close": [100.0]}),
        "portfolio_snapshot": snapshot(),
        "context": {"experiment_id": "M0", "point_in_time": True},
    }

    first_run = tmp_path / "run-one"
    first = TradingAgentsStrategy(
        graph, cache=cache, reports_root=first_run / "strategy" / "cases",
        run_root=first_run, cache_config={"git_commit_sha": "abc"},
    ).decide(**kwargs)
    second_run = tmp_path / "run-two"
    second = TradingAgentsStrategy(
        graph, cache=cache, reports_root=second_run / "strategy" / "cases",
        run_root=second_run, cache_config={"git_commit_sha": "abc"},
    ).decide(**kwargs)

    case = second_run / "strategy" / "cases" / "AAPL" / "2024-01-05"
    assert first.status is DecisionStatus.SUCCESS
    assert second.status is DecisionStatus.CACHED
    assert graph.calls == 1
    assert (case / "reports" / "complete_report.md").is_file()
    assert json.loads((case / "source_audit.json").read_text())["sources"][0][
        "status"
    ] == "used"
    for filename in (
        "decision.json", "model_config.json", "run_context.json",
        "cache_identity.json", "case_metadata.json", "llm_usage.json",
    ):
        assert (case / filename).is_file()
    assert second.metadata["report_path"].startswith("strategy/cases/")
    assert str(first_run) not in second.metadata["report_path"]
