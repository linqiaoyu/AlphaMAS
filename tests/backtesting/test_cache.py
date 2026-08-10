import json

import pandas as pd

from tradingagents.backtesting.cache import DecisionCache, cache_key
from tradingagents.backtesting.engine import WeeklyBacktestEngine
from tradingagents.backtesting.strategies import TradingAgentsStrategy


def test_cache_key_changes_with_state_model_and_git():
    base = {"portfolio": "a", "model": "m", "git_sha": "1"}
    key = cache_key(base)
    assert key != cache_key({**base, "portfolio": "b"})
    assert key != cache_key({**base, "model": "x"})
    assert key != cache_key({**base, "git_sha": "2"})


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
