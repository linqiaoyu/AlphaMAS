import json
from pathlib import Path

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.backtesting.memory_lineage import (
    backtest_protocol_sha256,
    select_memory_lineage,
    snapshot_input_identity,
    validate_resume_data_source,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.run_context import RunContext

GRAPH_A = "a" * 64
PROTOCOL_A = "p" * 64


def graph_stub(tmp_path, memory_mode, *, graph_hash=GRAPH_A):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "data_cache_dir": str(tmp_path / "cache"),
        "historical_memory_dir": str(tmp_path / "results" / "memory"),
        "historical_memory_log_path": None,
        "memory_mode": memory_mode,
        "graph_config_sha256": graph_hash,
    }
    return graph


def historical(
    as_of: str, *, experiment_id: str = "M0", lineage: str = "run-a",
) -> RunContext:
    return RunContext.historical(
        as_of, experiment_id=experiment_id, memory_lineage_id=lineage,
    )


def test_experiment_memory_shared_by_date_but_isolated_by_symbol_and_experiment(
    tmp_path,
):
    graph = graph_stub(tmp_path, "experiment")
    one = graph._historical_memory_config(
        "AAPL", historical("2024-01-05")
    )["memory_log_path"]
    later = graph._historical_memory_config(
        "AAPL", historical("2024-01-12")
    )["memory_log_path"]
    other_symbol = graph._historical_memory_config(
        "JPM", historical("2024-01-12")
    )["memory_log_path"]
    other_experiment = graph._historical_memory_config(
        "AAPL", historical("2024-01-12", experiment_id="M1")
    )["memory_log_path"]
    assert one == later
    assert len({one, other_symbol, other_experiment}) == 3
    assert "live" not in one


def test_experiment_memory_isolated_by_graph_config_hash_and_run_lineage(tmp_path):
    first = graph_stub(tmp_path, "experiment")
    second = graph_stub(tmp_path, "experiment", graph_hash="b" * 64)

    first_path = first._historical_memory_config(
        "AAPL", historical("2024-01-05")
    )["memory_log_path"]
    second_path = second._historical_memory_config(
        "AAPL", historical("2024-01-05")
    )["memory_log_path"]
    rerun_path = first._historical_memory_config(
        "AAPL", historical("2024-01-05", lineage="run-b")
    )["memory_log_path"]

    assert len({first_path, second_path, rerun_path}) == 3


def test_independent_rerun_cannot_consume_prior_future_reflection(tmp_path):
    graph = graph_stub(tmp_path, "experiment")
    run_a_t1 = graph._historical_memory_config(
        "AAPL", historical("2024-01-05", lineage="run-a")
    )
    run_a_log = TradingMemoryLog(run_a_t1)
    run_a_log.store_decision("AAPL", "2024-01-05", "Rating: Buy")

    # Run A advances to t2 and matures a reflection derived from prices after t1.
    run_a_t2 = graph._historical_memory_config(
        "AAPL", historical("2024-01-12", lineage="run-a")
    )
    TradingMemoryLog(run_a_t2).update_with_outcome(
        "AAPL", "2024-01-05", 0.10, 0.05, 5,
        "FUTURE-DERIVED RUN A REFLECTION",
        outcome_visible_from="2024-01-12",
    )
    assert "FUTURE-DERIVED" in TradingMemoryLog(run_a_t2).get_past_context("AAPL")

    # An independent Run B starts from t1 in a distinct, empty lineage.
    run_b_t1 = graph._historical_memory_config(
        "AAPL", historical("2024-01-05", lineage="run-b")
    )
    assert run_b_t1["memory_log_path"] != run_a_t1["memory_log_path"]
    assert TradingMemoryLog(run_b_t1).get_past_context("AAPL") == ""


def test_legitimate_resume_reuses_lineage_but_honors_reflection_visibility(tmp_path):
    graph = graph_stub(tmp_path, "experiment")
    t2_config = graph._historical_memory_config(
        "AAPL", historical("2024-01-12", lineage="resume-lineage")
    )
    log = TradingMemoryLog(t2_config)
    log.store_decision("AAPL", "2024-01-05", "Rating: Buy")
    log.update_with_outcome(
        "AAPL", "2024-01-05", 0.10, 0.05, 5, "Mature lesson",
        outcome_visible_from="2024-01-12",
    )

    # A cache hole while replaying t1 must not reveal its later outcome.
    resumed_t1 = TradingMemoryLog(graph._historical_memory_config(
        "AAPL", historical("2024-01-05", lineage="resume-lineage")
    ))
    assert resumed_t1.get_past_context("AAPL") == ""
    with pytest.raises(ValueError, match="lineage conflict"):
        resumed_t1.store_decision("AAPL", "2024-01-05", "Rating: Sell")
    assert len(TradingMemoryLog(t2_config).load_entries()) == 1
    assert TradingMemoryLog(t2_config).load_entries()[0]["rating"] == "Buy"

    # At the recorded visibility session, the same lineage can use the lesson.
    resumed_t2 = TradingMemoryLog(graph._historical_memory_config(
        "AAPL", historical("2024-01-12", lineage="resume-lineage")
    ))
    assert "Mature lesson" in resumed_t2.get_past_context("AAPL")


def _write_prior_attempt(
    root: Path, *, status: str = "failed", graph_hash: str = GRAPH_A,
    protocol_hash: str = PROTOCOL_A,
) -> None:
    prior_id = "prior-run"
    prior = root / "runs" / prior_id
    prior.mkdir(parents=True)
    (prior / "run_status.json").write_text(json.dumps({
        "experiment_id": "M0", "run_id": prior_id, "status": status,
        "memory_lineage_id": "original-lineage",
    }), encoding="utf-8")
    (prior / "config.resolved.json").write_text(json.dumps({
        "experiment_id": "M0", "run_id": prior_id,
        "graph_config_sha256": graph_hash,
        "backtest_protocol_sha256": protocol_hash,
        "memory_lineage_id": "original-lineage",
    }), encoding="utf-8")
    (root / "latest.json").write_text(json.dumps({
        "experiment_id": "M0", "run_id": prior_id, "status": status,
        "memory_lineage_id": "original-lineage",
        # Resume must ignore this potentially tampered path.
        "run_path": "/tmp/not-the-resume-source",
    }), encoding="utf-8")


def _select(root: Path, **overrides):
    kwargs = {
        "experiment_root": root,
        "experiment_id": "M0",
        "run_id": "new-run",
        "graph_config_sha256": GRAPH_A,
        "backtest_protocol_sha256": PROTOCOL_A,
        "resume": False,
        "force": False,
    }
    return select_memory_lineage(**{**kwargs, **overrides})


def test_resume_continues_latest_incomplete_compatible_lineage(tmp_path):
    root = tmp_path / "M0"
    _write_prior_attempt(root)

    lineage = _select(root, resume=True)

    assert lineage.lineage_id == "original-lineage"
    assert lineage.lifecycle == "resume"
    assert lineage.resumed_from_run_id == "prior-run"


def test_independent_and_force_runs_always_receive_fresh_lineage(tmp_path):
    root = tmp_path / "M0"
    _write_prior_attempt(root, status="success")

    independent = _select(root)
    forced = _select(root, force=True)

    assert independent.lineage_id == forced.lineage_id == "new-run"
    assert independent.lifecycle == "independent_fresh"
    assert forced.lifecycle == "force_fresh"
    assert forced.lineage_id != "original-lineage"


def test_resume_rejects_completed_incompatible_or_ambiguous_attempt(tmp_path):
    root = tmp_path / "M0"
    _write_prior_attempt(root, status="success")
    with pytest.raises(ValueError, match="completed"):
        _select(root, resume=True)

    (root / "latest.json").unlink()
    for path in (root / "runs" / "prior-run").iterdir():
        path.unlink()
    (root / "runs" / "prior-run").rmdir()
    _write_prior_attempt(root, graph_hash="b" * 64)
    with pytest.raises(ValueError, match="graph configuration"):
        _select(root, resume=True)

    with pytest.raises(ValueError, match="mutually exclusive"):
        _select(root, resume=True, force=True)


def test_resume_rejects_changed_backtest_protocol(tmp_path):
    root = tmp_path / "M0"
    _write_prior_attempt(root, protocol_hash="changed")

    with pytest.raises(ValueError, match="backtest protocol"):
        _select(root, resume=True)


def test_resume_rejects_inconsistent_lineage_provenance(tmp_path):
    root = tmp_path / "M0"
    _write_prior_attempt(root)
    status_path = root / "runs" / "prior-run" / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["memory_lineage_id"] = "different-lineage"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="memory lineages disagree"):
        _select(root, resume=True)


def test_resume_rejects_null_persisted_identity(tmp_path):
    root = tmp_path / "M0"
    _write_prior_attempt(root)
    latest_path = root / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["memory_lineage_id"] = None
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string"):
        _select(root, resume=True)


def test_snapshot_content_is_part_of_resume_protocol_identity(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "AAPL.csv").write_text("first", encoding="utf-8")
    (snapshot / "SPY.csv").write_text("benchmark", encoding="utf-8")
    first_inputs = snapshot_input_identity(snapshot, ["AAPL"])
    first = backtest_protocol_sha256(
        {"data_source": "snapshot"}, planned_cases=26,
        market_input_identity=first_inputs,
    )

    (snapshot / "AAPL.csv").write_text("changed", encoding="utf-8")
    second_inputs = snapshot_input_identity(snapshot, ["AAPL"])
    second = backtest_protocol_sha256(
        {"data_source": "snapshot"}, planned_cases=26,
        market_input_identity=second_inputs,
    )
    changed_code = backtest_protocol_sha256(
        {"data_source": "snapshot"}, planned_cases=26,
        market_input_identity=first_inputs,
        implementation_identity="different-worktree",
    )

    assert first_inputs["SPY"] == second_inputs["SPY"]
    assert first_inputs["AAPL"] != second_inputs["AAPL"]
    assert first != second
    assert first != changed_code


def test_resume_rejects_mutable_yfinance_inputs():
    with pytest.raises(ValueError, match="mutable yfinance"):
        validate_resume_data_source("yfinance", resume=True)

    validate_resume_data_source("yfinance", resume=False)
    validate_resume_data_source("snapshot", resume=True)
    validate_resume_data_source("synthetic", resume=True)


def test_memory_disabled_and_unsafe_experiment_rejected(tmp_path):
    disabled = graph_stub(tmp_path, "disabled")._historical_memory_config(
        "AAPL", RunContext.historical("2024-01-05", experiment_id="M0")
    )
    assert disabled["memory_log_path"] is None

    graph = graph_stub(tmp_path, "experiment")
    with pytest.raises(ValueError):
        graph._historical_memory_config(
            "AAPL", historical("2024-01-05", experiment_id="../bad")
        )
    with pytest.raises(ValueError, match="lineage"):
        graph._historical_memory_config(
            "AAPL", RunContext.historical("2024-01-05", experiment_id="M0")
        )


def test_experiment_mode_rejects_unscoped_explicit_memory_file(tmp_path):
    graph = graph_stub(tmp_path, "experiment")
    graph.config["historical_memory_log_path"] = str(tmp_path / "shared.md")

    with pytest.raises(ValueError, match="historical_memory_log_path"):
        graph._historical_memory_config("AAPL", historical("2024-01-05"))
