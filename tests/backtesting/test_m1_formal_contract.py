from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_weekly_backtest as weekly_runner
from tradingagents.backtesting.config import (
    FORMAL_M0_CONTRACT,
    FORMAL_M1_CONTRACT,
    compute_graph_config_sha256,
    resolve_graph_config,
    validate_formal_m1_config,
)
from tradingagents.backtesting.memory_lineage import backtest_protocol_sha256

REPOSITORY = Path(__file__).resolve().parents[2]
FORMAL_M0_CONFIG = REPOSITORY / "configs" / "backtest_m0_2024h1.json"
FORMAL_M1_CONFIG = REPOSITORY / "configs" / "backtest_m1_2024h1.json"
EXPERIMENT_ID = "M1_finmultitime_prompt_2024H1"


def load_m1() -> dict:
    return json.loads(FORMAL_M1_CONFIG.read_text(encoding="utf-8"))


def args(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "config": str(FORMAL_M1_CONFIG),
        "strategy": None,
        "experiment_id": EXPERIMENT_ID,
        "resume": False,
        "force": False,
        "max_cases": None,
        "dry_run": True,
        "data_source": None,
        "snapshot_dir": None,
        "finmultitime_input_root": None,
        "synthetic_data": False,
        "output_root": str(tmp_path),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_formal_m1_contract_is_exactly_frozen() -> None:
    config = load_m1()

    validate_formal_m1_config(config)

    assert config == FORMAL_M1_CONTRACT
    assert config["experiment_id_template"] == EXPERIMENT_ID
    assert config["symbols"] == ["AAPL", "AMZN", "JPM"]
    assert config["decision_weeks"] == 26
    assert config["finmultitime_bundle_scope"] == "FORMAL"
    assert config["finmultitime_verify_full_bundle_on_start"] is True
    assert "finmultitime_input_root" not in config


def test_m0_m1_config_diff_is_limited_to_frozen_evidence_identity() -> None:
    m0 = json.loads(FORMAL_M0_CONFIG.read_text(encoding="utf-8"))
    m1 = load_m1()
    intended = {
        "experiment_id_template",
        "finmultitime_evidence_enabled",
        "finmultitime_bundle_scope",
        "finmultitime_expected_contract_version",
        "finmultitime_expected_contract_sha256",
        "finmultitime_expected_packet_manifest_sha256",
        "finmultitime_expected_input_bundle_identity",
        "finmultitime_archive_commit",
        "finmultitime_verify_full_bundle_on_start",
    }

    assert {key for key in set(m0) | set(m1) if m0.get(key) != m1.get(key)} == intended
    assert {key: m1[key] for key in FORMAL_M0_CONTRACT if key != "experiment_id_template"} == {
        key: value
        for key, value in FORMAL_M0_CONTRACT.items()
        if key != "experiment_id_template"
    }


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("symbols", ["AMZN", "AAPL", "JPM"]),
        ("decision_weeks", 25),
        ("commission_bps", 4),
        ("memory_outcome_price_mode", "close"),
        ("quick_think_llm", "other"),
        ("finmultitime_bundle_scope", "PILOT"),
        ("finmultitime_expected_contract_sha256", "0" * 64),
        ("finmultitime_expected_packet_manifest_sha256", "0" * 64),
        ("finmultitime_expected_input_bundle_identity", "0" * 64),
        ("finmultitime_archive_commit", "0" * 40),
        ("finmultitime_verify_full_bundle_on_start", False),
    ),
)
def test_formal_m1_rejects_any_contract_drift(field: str, changed: object) -> None:
    config = load_m1()
    config[field] = changed

    with pytest.raises(ValueError, match="contract mismatch"):
        validate_formal_m1_config(config)


def test_formal_m1_rejects_operational_path_in_tracked_config() -> None:
    config = load_m1()
    config["finmultitime_input_root"] = "/machine/specific/path"

    with pytest.raises(ValueError, match="unexpected fields"):
        validate_formal_m1_config(config)


def test_formal_m1_dry_run_is_explicit_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        weekly_runner,
        "materialize_inputs",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not touch market data"),
    )
    monkeypatch.setattr(
        weekly_runner,
        "strategy_factory",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not construct a Graph"),
    )

    _, status = weekly_runner.execute(args(tmp_path))
    output = json.loads(capsys.readouterr().out)

    assert status == "dry-run"
    assert output["protocol_mode"] == "formal_m1"
    assert output["experiment_id"] == EXPERIMENT_ID
    assert output["symbols"] == ["AAPL", "AMZN", "JPM"]
    assert output["expected_paid_agent_cases"] == 78
    assert output["formal_execution_workers"] == 1
    assert output["parallel_symbol_runners"] == 0
    assert output["do_not_execute_paid_agent_cases"] is True
    assert not (tmp_path / EXPERIMENT_ID).exists()


def test_formal_m1_rejects_wrong_experiment_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires experiment_id"):
        weekly_runner.execute(args(tmp_path, experiment_id="renamed"))


def test_formal_m1_execution_requires_snapshot_and_evidence_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="snapshot-dir"):
        weekly_runner.execute(args(tmp_path, dry_run=False))

    with pytest.raises(ValueError, match="finmultitime-input-root"):
        weekly_runner.execute(
            args(tmp_path, dry_run=False, snapshot_dir=str(tmp_path / "snapshot"))
        )


def test_formal_m1_forbids_force_before_creating_a_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--force is forbidden"):
        weekly_runner.execute(
            args(
                tmp_path,
                dry_run=False,
                force=True,
                snapshot_dir=str(tmp_path / "snapshot"),
                finmultitime_input_root=str(tmp_path / "evidence"),
            )
        )
    assert not (tmp_path / EXPERIMENT_ID).exists()


def test_formal_m1_execution_requires_clean_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        weekly_runner,
        "git_value",
        lambda *git_args: "dirty" if git_args == ("status", "--porcelain") else "source",
    )

    with pytest.raises(ValueError, match="formal M1 execution requires a clean"):
        weekly_runner.execute(
            args(
                tmp_path,
                dry_run=False,
                snapshot_dir=str(tmp_path / "snapshot"),
                finmultitime_input_root=str(tmp_path / "evidence"),
            )
        )
    assert not (tmp_path / EXPERIMENT_ID).exists()


def test_operational_input_root_changes_neither_graph_nor_protocol_identity(
    tmp_path: Path,
) -> None:
    first_config = {**load_m1(), "finmultitime_input_root": "/machine/a/inputs"}
    second_config = {**load_m1(), "finmultitime_input_root": "/machine/b/inputs"}
    first_graph = resolve_graph_config(
        first_config,
        results_dir=tmp_path / "a/results",
        data_cache_dir=tmp_path / "a/cache",
        historical_memory_dir=tmp_path / "a/memory",
    )
    second_graph = resolve_graph_config(
        second_config,
        results_dir=tmp_path / "b/results",
        data_cache_dir=tmp_path / "b/cache",
        historical_memory_dir=tmp_path / "b/memory",
    )

    assert compute_graph_config_sha256(first_graph) == compute_graph_config_sha256(
        second_graph
    )
    assert backtest_protocol_sha256(
        first_config,
        planned_cases=78,
        market_input_identity={"AAPL": "a" * 64},
        implementation_identity="source",
    ) == backtest_protocol_sha256(
        second_config,
        planned_cases=78,
        market_input_identity={"AAPL": "a" * 64},
        implementation_identity="source",
    )


def test_research_identity_change_changes_protocol_identity() -> None:
    base = load_m1()
    changed = copy.deepcopy(base)
    changed["finmultitime_expected_input_bundle_identity"] = "0" * 64

    assert backtest_protocol_sha256(base, planned_cases=78) != backtest_protocol_sha256(
        changed, planned_cases=78
    )
