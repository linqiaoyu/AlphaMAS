from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.run_weekly_backtest as weekly_runner
from tradingagents.backtesting.config import (
    FIXED_BACKTEST_CONTRACT,
    FORMAL_M0_CONTRACT,
    GRAPH_OPERATIONAL_KEYS,
    GRAPH_RESEARCH_KEYS,
    compute_graph_config_sha256,
    resolve_graph_config,
    sanitize_graph_config,
    validate_formal_m0_config,
)
from tradingagents.backtesting.data import canonical_market_csv
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.run_context import RunContext, activate_run_context

REPOSITORY = Path(__file__).resolve().parents[2]
FORMAL_CONFIG = REPOSITORY / "configs" / "backtest_m0_2024h1.json"


def load_formal_config() -> dict:
    return json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))


def test_formal_m0_contract_is_frozen() -> None:
    config = load_formal_config()

    validate_formal_m0_config(config)

    assert {key: config[key] for key in FORMAL_M0_CONTRACT} == FORMAL_M0_CONTRACT
    assert config["symbols"] == ["AAPL", "AMZN", "JPM"]
    assert config["decision_weeks"] == 26
    assert config["research_depth"] == "medium"
    assert config["max_debate_rounds"] == 3
    assert config["max_risk_discuss_rounds"] == 3
    assert config["llm_provider"] == "deepseek"
    assert config["quick_think_llm"] == "deepseek-v4-flash"
    assert config["deep_think_llm"] == "deepseek-v4-flash"
    assert config["deepseek_thinking"] == "disabled"
    assert config["temperature"] == 0.0
    assert config["memory_mode"] == "experiment"
    assert config["holding_horizon_sessions"] == 5
    assert config["memory_outcome_price_mode"] == "adjusted_close"
    assert config["data_source"] == "snapshot"
    assert "seed" not in config


@pytest.mark.parametrize(
    ("field", "unsupported"),
    (
        ("long_only", False),
        ("short_selling", True),
        ("leverage", True),
        ("buy_target_weight", 0.5),
        ("sell_target_weight", -1.0),
        ("hold", "flatten"),
        ("force_liquidate_at_end", True),
        ("point_in_time", False),
    ),
)
def test_unsupported_fixed_contract_value_fails_loudly(
    field: str, unsupported: object,
) -> None:
    config = load_formal_config()
    config[field] = unsupported

    with pytest.raises(ValueError, match=field):
        validate_formal_m0_config(config)


def test_formal_fixed_contract_values_are_explicit() -> None:
    config = load_formal_config()

    assert {key: config[key] for key in FIXED_BACKTEST_CONTRACT} == (
        FIXED_BACKTEST_CONTRACT
    )


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("max_recur_limit", 99),
        ("news_article_limit", 19),
        ("global_news_lookback_days", 8),
        ("data_vendors", {"core_stock_apis": "alpha_vantage"}),
        ("tool_vendors", {"get_news": "alpha_vantage"}),
        ("checkpoint_enabled", True),
    ),
)
def test_formal_m0_extended_graph_contract_is_frozen(field, changed) -> None:
    config = load_formal_config()
    config[field] = changed

    with pytest.raises(ValueError, match="contract mismatch"):
        validate_formal_m0_config(config)


def test_formal_m0_rejects_untracked_extra_fields() -> None:
    config = load_formal_config()
    config["seed"] = 42

    with pytest.raises(ValueError, match="unexpected fields"):
        validate_formal_m0_config(config)


def test_m0_uses_252_xnys_warmup_sessions(xnys) -> None:
    warmup = xnys.preceding_sessions("2024-01-05", 252)

    assert len(warmup) == 252
    assert warmup[0].date().isoformat() == "2023-01-04"
    assert warmup[-1].date().isoformat() == "2024-01-04"
    assert pd.Timestamp("2024-01-05", tz="UTC") not in warmup


def test_m0_temperature_is_zero() -> None:
    config = load_formal_config()

    assert config["temperature"] == 0.0


@pytest.mark.parametrize("value", (3.5, "3", True, 0, -1))
def test_memory_horizon_rejects_non_positive_integer_values(value) -> None:
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_holding_horizon_sessions": value}

    with pytest.raises(ValueError, match="positive integer"):
        graph._memory_holding_horizon_sessions()


def test_memory_horizon_comes_from_backtest_config(tmp_path: Path) -> None:
    backtest_config = load_formal_config()
    backtest_config["holding_horizon_sessions"] = 3
    graph_config = resolve_graph_config(
        backtest_config,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )

    assert graph_config["memory_holding_horizon_sessions"] == 3


def test_memory_outcome_price_mode_is_frozen_and_research_affecting(tmp_path: Path) -> None:
    config = load_formal_config()
    graph = resolve_graph_config(
        config,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )
    changed = {**graph, "memory_outcome_price_mode": "unadjusted_close"}

    assert graph["memory_outcome_price_mode"] == "adjusted_close"
    assert compute_graph_config_sha256(changed) != compute_graph_config_sha256(graph)

    instance = object.__new__(TradingAgentsGraph)
    instance.config = changed
    with pytest.raises(ValueError, match="must be 'adjusted_close'"):
        instance._memory_outcome_price_mode()


def test_graph_config_hash_is_path_stable_and_research_sensitive(tmp_path: Path) -> None:
    config = load_formal_config()
    first = resolve_graph_config(
        config,
        results_dir=tmp_path / "run-a" / "results",
        data_cache_dir=tmp_path / "run-a" / "cache",
        historical_memory_dir=tmp_path / "run-a" / "memory",
    )
    second = resolve_graph_config(
        config,
        results_dir=tmp_path / "run-b" / "results",
        data_cache_dir=tmp_path / "run-b" / "cache",
        historical_memory_dir=tmp_path / "run-b" / "memory",
    )
    changed = {**second, "deepseek_thinking": "enabled"}

    assert compute_graph_config_sha256(first) == compute_graph_config_sha256(second)
    assert compute_graph_config_sha256(first) != compute_graph_config_sha256(changed)


def test_finmultitime_input_root_is_propagated_but_not_research_identity(
    tmp_path: Path,
) -> None:
    config = {
        **load_formal_config(),
        "finmultitime_input_root": Path("/tmp/example/pilot"),
    }
    graph = resolve_graph_config(
        config,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )
    relocated = {
        **graph,
        "finmultitime_input_root": Path("/machine/b/identical-inputs"),
    }

    assert GRAPH_OPERATIONAL_KEYS == (
        "finmultitime_input_root",
        "m2_checkpoint_path",
        "m2_encoder_snapshot_path",
        "m2_encoder_snapshot_manifest_path",
        "m2_state_root",
        "m2_preformal_evidence_root",
    )
    assert graph["finmultitime_input_root"] == Path("/tmp/example/pilot")
    assert compute_graph_config_sha256(graph) == compute_graph_config_sha256(relocated)


def test_every_declared_graph_research_field_changes_identity(tmp_path: Path) -> None:
    base = resolve_graph_config(
        load_formal_config(),
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )
    base_hash = compute_graph_config_sha256(base)

    for field in GRAPH_RESEARCH_KEYS:
        changed = copy.deepcopy(base)
        value = changed[field]
        if isinstance(value, bool):
            changed[field] = not value
        elif isinstance(value, dict):
            changed[field] = {**value, "__identity_test__": "changed"}
        elif isinstance(value, list):
            changed[field] = [*value, "__identity_test__"]
        elif isinstance(value, (int, float)):
            changed[field] = value + 1
        elif value is None:
            changed[field] = "__identity_test__"
        else:
            changed[field] = f"{value}__identity_test__"

        assert compute_graph_config_sha256(changed) != base_hash, field


def test_resolved_graph_config_removes_credentials(tmp_path: Path) -> None:
    config = load_formal_config()
    graph = resolve_graph_config(
        config,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )
    graph.update({"api_key": "secret", "nested": {"client_secret": "secret"}})

    sanitized = sanitize_graph_config(graph)

    assert "api_key" not in sanitized
    assert "client_secret" not in sanitized["nested"]


def test_memory_three_session_horizon_changes_maturity_logic() -> None:
    stock = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0, 103.0]},
        index=pd.date_range("2024-01-02", periods=4, freq="D"),
    )
    benchmark = pd.DataFrame(
        {"Close": [200.0, 201.0, 202.0, 203.0]},
        index=stock.index,
    )

    def ticker(symbol: str):
        history = benchmark if symbol == "SPY" else stock
        instance = type("OfflineTicker", (), {})()
        instance.history = lambda **_kwargs: history.copy()
        return instance

    context = RunContext.historical("2024-01-05", experiment_id="memory-horizon-test")
    with patch("tradingagents.graph.trading_graph.yf.Ticker", side_effect=ticker):
        graph = object.__new__(TradingAgentsGraph)
        graph.config = {"memory_holding_horizon_sessions": 3}
        with activate_run_context(context):
            raw, alpha, days = graph._fetch_returns("AAPL", "2024-01-02")

        graph.config = {"memory_holding_horizon_sessions": 5}
        with activate_run_context(context):
            immature = graph._fetch_returns("AAPL", "2024-01-02")

    assert days == 3
    assert raw == pytest.approx(0.03)
    assert alpha == pytest.approx(0.015)
    assert immature == (None, None, None)


def test_snapshot_source_without_directory_fails_before_provider_use(
    tmp_path: Path, xnys, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weekly_runner,
        "YFinanceDataProvider",
        lambda: pytest.fail("snapshot policy must not construct yfinance"),
    )

    with pytest.raises(ValueError, match="snapshot-dir"):
        weekly_runner.materialize_inputs(
            "snapshot",
            None,
            ["AAPL"],
            xnys,
            "2024-01-02",
            "2024-01-05",
            tmp_path / "run",
            {
                "configured_warmup_sessions": 2,
                "actual_warmup_sessions": 2,
                "warmup_first_session": "2024-01-02",
                "warmup_last_session": "2024-01-03",
                "first_decision_session": "2024-01-04",
            },
        )


def test_formal_m0_execution_without_snapshot_dir_fails_before_any_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weekly_runner,
        "materialize_inputs",
        lambda *_args, **_kwargs: pytest.fail("no provider may be used without snapshot-dir"),
    )
    args = SimpleNamespace(
        config=str(FORMAL_CONFIG),
        strategy=None,
        experiment_id="missing-snapshot",
        resume=False,
        force=False,
        max_cases=None,
        dry_run=False,
        data_source=None,
        snapshot_dir=None,
        synthetic_data=False,
        output_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="snapshot-dir"):
        weekly_runner.execute(args)

    assert not (tmp_path / "missing-snapshot").exists()


def test_formal_m0_dry_run_is_offline_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        weekly_runner,
        "materialize_inputs",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not touch a data provider"),
    )
    monkeypatch.setattr(
        weekly_runner,
        "strategy_factory",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not construct a Graph"),
    )
    args = SimpleNamespace(
        config=str(FORMAL_CONFIG),
        strategy=None,
        experiment_id="formal-dry-run",
        resume=False,
        force=False,
        max_cases=None,
        dry_run=True,
        data_source=None,
        snapshot_dir=None,
        synthetic_data=False,
        output_root=str(tmp_path),
    )

    _, status = weekly_runner.execute(args)
    output = json.loads(capsys.readouterr().out)

    assert status == "dry-run"
    assert output["expected_paid_agent_cases"] == 78
    assert output["protocol_mode"] == "formal_m0"
    assert output["snapshot_required"] is True
    assert output["do_not_execute_paid_agent_cases"] is True
    assert output["thinking_mode"] == "disabled"
    assert output["temperature"] == 0.0
    assert output["debate_rounds"] == output["risk_rounds"] == 3
    assert output["warmup"]["actual_warmup_sessions"] == 252
    assert not (tmp_path / "formal-dry-run").exists()


def test_formal_dry_run_rejects_non_snapshot_agent_override(tmp_path: Path) -> None:
    args = SimpleNamespace(
        config=str(FORMAL_CONFIG),
        strategy=None,
        experiment_id="invalid-formal-dry-run",
        resume=False,
        force=False,
        max_cases=None,
        dry_run=True,
        data_source="yfinance",
        snapshot_dir=None,
        synthetic_data=False,
        output_root=str(tmp_path),
    )

    with pytest.raises(ValueError, match="requires data_source=snapshot"):
        weekly_runner.execute(args)

    assert not (tmp_path / "invalid-formal-dry-run").exists()


def test_unknown_materialization_source_never_falls_back_to_yfinance(
    tmp_path: Path, xnys, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weekly_runner,
        "YFinanceDataProvider",
        lambda: pytest.fail("unknown source must not construct yfinance"),
    )

    with pytest.raises(ValueError, match="unsupported market data source"):
        weekly_runner.materialize_inputs(
            "typo",
            None,
            ["AAPL"],
            xnys,
            "2024-01-02",
            "2024-01-05",
            tmp_path / "run",
            {
                "configured_warmup_sessions": 2,
                "actual_warmup_sessions": 2,
                "warmup_first_session": "2024-01-02",
                "warmup_last_session": "2024-01-03",
                "first_decision_session": "2024-01-04",
            },
        )


def test_valid_snapshot_policy_uses_only_snapshot_provider(
    tmp_path: Path, xnys, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    sessions = xnys.sessions("2024-01-02", "2024-01-05").tz_localize(None)
    close = pd.Series(range(100, 100 + len(sessions)), index=sessions, dtype=float)
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )
    for symbol in ("AAPL", "SPY"):
        canonical_market_csv(frame, snapshot_dir / f"{symbol}.csv")

    monkeypatch.setattr(
        weekly_runner,
        "YFinanceDataProvider",
        lambda: pytest.fail("valid snapshot policy must not construct yfinance"),
    )
    provider, frames, manifest = weekly_runner.materialize_inputs(
        "snapshot",
        str(snapshot_dir),
        ["AAPL"],
        xnys,
        "2024-01-02",
        "2024-01-05",
        tmp_path / "run",
        {
            "configured_warmup_sessions": 2,
            "actual_warmup_sessions": 2,
            "warmup_first_session": "2024-01-02",
            "warmup_last_session": "2024-01-03",
            "first_decision_session": "2024-01-04",
        },
    )

    assert set(frames) == {"AAPL", "SPY"}
    assert provider.load("AAPL", "2024-01-02", "2024-01-05").equals(frames["AAPL"])
    assert {entry["provider"] for entry in manifest["symbols"]} == {"snapshot"}
    assert all(entry["validation_status"] == "passed" for entry in manifest["symbols"])
