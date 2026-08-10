from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.run_weekly_backtest as weekly_runner
from tradingagents.backtesting.cache import DecisionCache, cache_key
from tradingagents.backtesting.data import normalize_ohlcv
from tradingagents.backtesting.engine import (
    ChronologicalDecisionFailure,
    WeeklyBacktestEngine,
)
from tradingagents.backtesting.models import (
    Action,
    DecisionStatus,
    StrategyDecision,
)
from tradingagents.backtesting.portfolio import Portfolio
from tradingagents.backtesting.strategies import (
    DecisionChronology,
    TradingAgentsStrategy,
)


class MemoryWritingGraph:
    """Offline Graph double that exposes case-level Memory mutations."""

    selected_analysts = ("market", "social", "news", "fundamentals")

    def __init__(
        self, memory_root: Path, *, failures: set[tuple[str, str]] | None = None,
        lineage: str = "resume-lineage",
    ) -> None:
        self.memory_root = memory_root
        self.failures = failures or set()
        self.calls: list[tuple[str, str]] = []
        self.config = {
            "research_depth": "medium",
            "quick_think_llm": "deepseek-v4-flash",
            "deep_think_llm": "deepseek-v4-flash",
            "llm_provider": "deepseek",
            "deepseek_thinking": "disabled",
            "temperature": 0.0,
            "max_debate_rounds": 3,
            "max_risk_discuss_rounds": 3,
            "output_language": "English",
            "data_vendors": {},
            "tool_vendors": {},
            "memory_mode": "experiment",
            "memory_holding_horizon_sessions": 5,
            "historical_memory_lineage_id": lineage,
            "graph_config_sha256": "a" * 64,
        }

    def _historical_memory_config(self, symbol, run_context):
        del run_context
        return {"memory_log_path": str(self.memory_root / f"{symbol}.md")}

    def propagate(self, symbol, trade_date, **kwargs):
        del kwargs
        self.calls.append((symbol, trade_date))
        path = self.memory_root / f"{symbol}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        prior = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(prior + f"{symbol}:{trade_date}\n", encoding="utf-8")
        if (symbol, trade_date) in self.failures:
            raise RuntimeError("offline injected Agent failure")
        return {"final_trade_decision": "Rating: HOLD"}, "Hold"


def _strategy(
    graph: MemoryWritingGraph, cache: DecisionCache, *,
    chronology: DecisionChronology | None = None, force: bool = False,
) -> TradingAgentsStrategy:
    return TradingAgentsStrategy(
        graph,
        cache=cache,
        cache_config={
            "git_commit_sha": "offline-test",
            "prompt_config_version": "weekly-backtest-v1",
            "memory_namespace_version": "experiment-v3-run-lineage",
        },
        chronology=chronology,
        force=force,
    )


def _run(
    strategy: TradingAgentsStrategy, synthetic_provider, xnys, *, symbol: str,
    final_week: str,
):
    return WeeklyBacktestEngine(
        data_provider=synthetic_provider, schedule=xnys,
    ).run(
        symbol=symbol,
        first_week="2024-01-01",
        final_week=final_week,
        final_valuation_session="2024-01-26",
        strategy=strategy,
        experiment_id="M0-offline-resume",
    )


def _case_kwargs(synthetic_provider, xnys, *, symbol: str, session: str):
    first_session = "2024-01-05"
    data_start = xnys.preceding_sessions(first_session, 252)[0].date().isoformat()
    data = normalize_ohlcv(
        synthetic_provider.load(symbol, data_start, "2024-01-26")
    )
    event = next(
        item
        for item in xnys.weekly_events("2024-01-01", "2024-01-15")
        if item.decision_session == session
    )
    close = float(data.loc[pd.Timestamp(session), "Close"])
    snapshot = Portfolio(symbol, 100_000).snapshot(
        session, event.decision_close_utc.to_pydatetime(), close,
    )
    return {
        "symbol": symbol,
        "decision_session": session,
        "decision_time": event.decision_close_utc.to_pydatetime(),
        "market_history": data.loc[:pd.Timestamp(session)].copy(),
        "portfolio_snapshot": snapshot,
        "context": {"experiment_id": "M0-offline-resume", "point_in_time": True},
    }


def _seed_stale_hold(
    cache: DecisionCache, strategy: TradingAgentsStrategy, kwargs,
) -> str:
    payload = strategy._cache_payload(kwargs)
    key = cache_key(payload)
    cache.save_success(
        key,
        StrategyDecision(
            symbol=kwargs["symbol"],
            decision_session=kwargs["decision_session"],
            decision_time_utc=kwargs["decision_time"],
            action=Action.HOLD,
            target_weight=0.0,
            status=DecisionStatus.SUCCESS,
            reason="obsolete downstream decision",
            strategy_id="tradingagents",
            experiment_id=kwargs["context"]["experiment_id"],
            raw_signal="HOLD",
            metadata={"cache_key": key, "cache_status": "miss"},
        ).to_dict(),
    )
    return key


def test_first_agent_failure_rolls_back_memory_and_aborts_whole_attempt(
    tmp_path, synthetic_provider, xnys, monkeypatch,
):
    cache = DecisionCache(tmp_path / "cache")
    memory_root = tmp_path / "memory"
    created: list[TradingAgentsStrategy] = []

    def factory(name, **kwargs):
        assert name == "tradingagents"
        strategy = _strategy(
            MemoryWritingGraph(
                memory_root,
                failures={("AAPL", "2024-01-12")},
            ),
            cache,
            chronology=kwargs["chronology"],
        )
        created.append(strategy)
        return strategy

    monkeypatch.setattr(weekly_runner, "strategy_factory", factory)
    events = xnys.weekly_events("2024-01-01", "2024-01-15")
    config = {
        "first_calendar_week": "2024-01-01",
        "final_calendar_week": "2024-01-15",
        "final_valuation_session": "2024-01-26",
        "initial_cash": 100_000,
        "commission_bps": 5,
        "slippage_bps": 5,
        "fractional_shares": True,
        "risk_free_rate": 0.0,
        "annualization": 252,
    }

    with pytest.raises(ChronologicalDecisionFailure, match="AAPL:2024-01-12"):
        weekly_runner.run_accounts(
            symbols=["AAPL", "AMZN"],
            provider=synthetic_provider,
            schedule=xnys,
            config=config,
            strategy_name="tradingagents",
            experiment_id="M0-offline-resume",
            experiment_root=tmp_path,
            run_dir=tmp_path / "run",
            git_sha="offline-test",
            events=events,
            data_start=xnys.preceding_sessions(
                events[0].decision_session, 252,
            )[0].date().isoformat(),
            planned_cases=6,
            force=False,
            graph_config={},
        )

    assert len(created) == 1  # AMZN strategy construction was never reached.
    assert created[0].graph.calls == [
        ("AAPL", "2024-01-05"),
        ("AAPL", "2024-01-12"),
    ]
    assert (memory_root / "AAPL.md").read_text(encoding="utf-8") == (
        "AAPL:2024-01-05\n"
    )
    assert not (memory_root / "AMZN.md").exists()
    assert len(list((tmp_path / "cache").rglob("run_status.json"))) == 1


def test_failure_resume_replays_prefix_and_misses_obsolete_later_week_cache(
    tmp_path, synthetic_provider, xnys,
):
    cache = DecisionCache(tmp_path / "cache")
    memory_root = tmp_path / "memory"
    failed_graph = MemoryWritingGraph(
        memory_root, failures={("AAPL", "2024-01-12")},
    )
    failed_strategy = _strategy(failed_graph, cache)

    with pytest.raises(ChronologicalDecisionFailure):
        _run(
            failed_strategy, synthetic_provider, xnys,
            symbol="AAPL", final_week="2024-01-15",
        )

    valid_prefix = failed_strategy.chronology.prefix_sha256
    stale_chronology = DecisionChronology()
    stale_chronology.prefix_sha256 = valid_prefix
    stale_strategy = _strategy(
        MemoryWritingGraph(memory_root), cache, chronology=stale_chronology,
    )
    stale_key = _seed_stale_hold(
        cache,
        stale_strategy,
        _case_kwargs(
            synthetic_provider, xnys, symbol="AAPL", session="2024-01-19",
        ),
    )

    resumed_graph = MemoryWritingGraph(memory_root)
    resumed = _run(
        _strategy(resumed_graph, cache),
        synthetic_provider,
        xnys,
        symbol="AAPL",
        final_week="2024-01-15",
    )

    assert resumed.decisions["status"].tolist() == ["cached", "success", "success"]
    assert resumed_graph.calls == [
        ("AAPL", "2024-01-12"),
        ("AAPL", "2024-01-19"),
    ]
    assert resumed.decisions.iloc[2]["metadata"]["cache_status"] == "miss"
    assert resumed.decisions.iloc[2]["metadata"]["cache_key"] != stale_key
    assert (memory_root / "AAPL.md").read_text(encoding="utf-8") == (
        "AAPL:2024-01-05\nAAPL:2024-01-12\nAAPL:2024-01-19\n"
    )


def test_repaired_aapl_prefix_invalidates_obsolete_amzn_cache(
    tmp_path, synthetic_provider, xnys,
):
    cache = DecisionCache(tmp_path / "cache")
    memory_root = tmp_path / "memory"
    failed_chronology = DecisionChronology()
    failed_aapl = _strategy(
        MemoryWritingGraph(
            memory_root, failures={("AAPL", "2024-01-12")},
        ),
        cache,
        chronology=failed_chronology,
    )
    with pytest.raises(ChronologicalDecisionFailure):
        _run(
            failed_aapl, synthetic_provider, xnys,
            symbol="AAPL", final_week="2024-01-08",
        )

    stale_chronology = DecisionChronology()
    stale_chronology.prefix_sha256 = failed_chronology.prefix_sha256
    stale_amzn = _strategy(
        MemoryWritingGraph(memory_root), cache, chronology=stale_chronology,
    )
    stale_key = _seed_stale_hold(
        cache,
        stale_amzn,
        _case_kwargs(
            synthetic_provider, xnys, symbol="AMZN", session="2024-01-05",
        ),
    )

    resumed_chronology = DecisionChronology()
    resumed_aapl = _run(
        _strategy(
            MemoryWritingGraph(memory_root), cache,
            chronology=resumed_chronology,
        ),
        synthetic_provider,
        xnys,
        symbol="AAPL",
        final_week="2024-01-08",
    )
    assert resumed_aapl.decisions["status"].tolist() == ["cached", "success"]

    amzn_graph = MemoryWritingGraph(memory_root)
    resumed_amzn = _run(
        _strategy(amzn_graph, cache, chronology=resumed_chronology),
        synthetic_provider,
        xnys,
        symbol="AMZN",
        final_week="2024-01-01",
    )

    assert amzn_graph.calls == [("AMZN", "2024-01-05")]
    assert resumed_amzn.decisions.iloc[0]["metadata"]["cache_status"] == "miss"
    assert resumed_amzn.decisions.iloc[0]["metadata"]["cache_key"] != stale_key


def test_fresh_and_force_attempts_do_not_reuse_prior_prefix_cache(
    tmp_path, synthetic_provider, xnys,
):
    cache = DecisionCache(tmp_path / "cache")
    first_graph = MemoryWritingGraph(tmp_path / "memory-first", lineage="lineage-first")
    first = _run(
        _strategy(first_graph, cache), synthetic_provider, xnys,
        symbol="AAPL", final_week="2024-01-01",
    )

    independent_graph = MemoryWritingGraph(
        tmp_path / "memory-independent", lineage="lineage-independent",
    )
    independent = _run(
        _strategy(independent_graph, cache), synthetic_provider, xnys,
        symbol="AAPL", final_week="2024-01-01",
    )
    forced_graph = MemoryWritingGraph(
        tmp_path / "memory-force", lineage="lineage-force",
    )
    forced = _run(
        _strategy(forced_graph, cache, force=True), synthetic_provider, xnys,
        symbol="AAPL", final_week="2024-01-01",
    )

    assert first.decisions.iloc[0]["metadata"]["cache_status"] == "miss"
    assert independent.decisions.iloc[0]["metadata"]["cache_status"] == "miss"
    assert forced.decisions.iloc[0]["metadata"]["cache_status"] == "bypass"
    assert independent_graph.calls == [("AAPL", "2024-01-05")]
    assert forced_graph.calls == [("AAPL", "2024-01-05")]
