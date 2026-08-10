from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.run_context import RunContext


def graph_stub(tmp_path, memory_mode):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "historical_memory_dir": str(tmp_path / "results" / "memory"),
        "historical_memory_log_path": None,
        "memory_mode": memory_mode,
        "graph_config_sha256": "a" * 64,
    }
    return graph


def test_experiment_memory_shared_by_date_but_isolated_by_symbol_and_experiment(tmp_path):
    graph = graph_stub(tmp_path, "experiment")
    one = graph._historical_memory_config(
        "AAPL", RunContext.historical("2024-01-05", experiment_id="M0")
    )["memory_log_path"]
    later = graph._historical_memory_config(
        "AAPL", RunContext.historical("2024-01-12", experiment_id="M0")
    )["memory_log_path"]
    other_symbol = graph._historical_memory_config(
        "JPM", RunContext.historical("2024-01-12", experiment_id="M0")
    )["memory_log_path"]
    other_experiment = graph._historical_memory_config(
        "AAPL", RunContext.historical("2024-01-12", experiment_id="M1")
    )["memory_log_path"]
    assert one == later
    assert one != other_symbol != other_experiment
    assert "live" not in one


def test_experiment_memory_isolated_by_graph_config_hash(tmp_path):
    first = graph_stub(tmp_path, "experiment")
    second = graph_stub(tmp_path, "experiment")
    second.config["graph_config_sha256"] = "b" * 64
    context = RunContext.historical("2024-01-05", experiment_id="M0")

    first_path = first._historical_memory_config("AAPL", context)["memory_log_path"]
    second_path = second._historical_memory_config("AAPL", context)["memory_log_path"]

    assert first_path != second_path


def test_memory_disabled_and_unsafe_experiment_rejected(tmp_path):
    disabled = graph_stub(tmp_path, "disabled")._historical_memory_config(
        "AAPL", RunContext.historical("2024-01-05", experiment_id="M0")
    )
    assert disabled["memory_log_path"] is None
    graph = graph_stub(tmp_path, "experiment")
    try:
        graph._historical_memory_config(
            "AAPL", RunContext.historical("2024-01-05", experiment_id="../bad")
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe experiment path accepted")
