from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.run_weekly_backtest as weekly_runner
from scripts.m2.online_delayed_adaptation import (
    FAST_PARAMETER_NAMES,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    _encode_optimizer_state,
    payload_sha,
)
from scripts.m2.pa_ctppo import EXPECTED_FAST_PARAMETERS, SEED, fast_checkpoint_sha
from scripts.m2.train_global_pa_ctppo import canonical_parameter_sha, sha256_file
from tests.archive_roots import ExperimentsArchiveUnavailable, resolve_experiments_root
from tradingagents.agents.utils.memory_namespace import runtime_experiment_memory_path
from tradingagents.backtesting.config import (
    FORMAL_A2_CONTRACT,
    FORMAL_M2_CONTRACT,
    compute_graph_config_sha256,
    resolve_graph_config,
    validate_formal_a2_config,
)
from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.m2 import runtime as runtime_module
from tradingagents.m2.runtime import (
    A2_FILE_SHA,
    A2_PARAMETER_SHA,
    C09_FILE_SHA,
    C09_PARAMETER_SHA,
    M2ProductionTraderRuntime,
)

ROOT = Path(__file__).resolve().parents[2]
FORMAL_A2_CONFIG = ROOT / "configs/backtest_a2_no_global_pretraining_2024h1.json"
FORMAL_M2_CONFIG = ROOT / "configs/backtest_m2_2024h1.json"
EXPERIMENT_ID = "A2_no_global_pretraining_2024H1"
INITIAL_FAST_SHA = "49380bb69ea6fb7682fe97a3ef5e1ccf2b2429b749d6f9f9971e40725271fcf9"
INITIAL_GLOBAL_SHA = "9b83aa1382d6b41c55981f2475eb0e236248affc5b5004a151a4b01651fa646e"
INITIAL_OPTIMISER_SHA = "087df628de120d7e2271432fe99d2888097983c25a3cfc880ec60a51d3b4d623"


def _experiments_root() -> Path:
    try:
        return resolve_experiments_root(ROOT)
    except ExperimentsArchiveUnavailable as exc:
        pytest.fail(str(exc))


def _initial_checkpoint() -> Path:
    return (
        _experiments_root()
        / "experiments/M2/development/global_training_v1/initial_checkpoint/model.pt"
    )


def _c09_checkpoint() -> Path:
    return (
        _experiments_root()
        / "experiments/M2/development/global_selection_v1/selected_global_checkpoint/model.pt"
    )


def _encoder(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        row = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
        rows.append(row / np.linalg.norm(row))
    return np.stack(rows)


@pytest.fixture()
def runtime(tmp_path: Path) -> M2ProductionTraderRuntime:
    return M2ProductionTraderRuntime(
        checkpoint_path=_initial_checkpoint(),
        state_root=tmp_path / "a2_state",
        encoder=_encoder,
        checkpoint_parameter_identity=A2_PARAMETER_SHA,
        checkpoint_file_identity=A2_FILE_SHA,
        online_adaptation_enabled=True,
    )


def _snapshot(symbol: str, session: str) -> dict:
    close = runtime_module.ExchangeSchedule().session_close(session).to_pydatetime()
    return PortfolioSnapshot(
        timestamp=close,
        session=session,
        symbol=symbol,
        cash=100_000.0,
        quantity=0.0,
        close_price=100.0,
        market_value=0.0,
        equity=100_000.0,
        current_weight=0.0,
        average_entry_price=0.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        cumulative_cost=0.0,
        current_drawdown=0.0,
        peak_equity=100_000.0,
    ).to_dict()


def _graph_state(
    symbol: str = "AAPL",
    session: str = "2023-10-06",
    next_session: str | None = "2023-10-13",
) -> dict:
    return {
        "company_of_interest": symbol,
        "trade_date": session,
        "investment_plan": "Frozen pre-2024 Research Manager plan.",
        "trader_investment_plan": (
            "Frozen pre-2024 Prompt Trader proposal.\n\n"
            "FINAL TRANSACTION PROPOSAL: **BUY**"
        ),
        "m2_portfolio_snapshot": _snapshot(symbol, session),
        "m2_portfolio_reward_state": {
            "cash": 100_000.0,
            "quantity": 0.0,
            "average_entry_price": 0.0,
            "open_position_commission": 0.0,
        },
        "m2_next_decision_session": next_session,
    }


def _maturity_frame(origin: str = "2023-10-06") -> pd.DataFrame:
    sessions = runtime_module.ExchangeSchedule().calendar.sessions_window(
        origin, 6
    ).tz_localize(None)
    values = np.array([100.0, 104.0, 108.0, 112.0, 116.0, 120.0])
    return pd.DataFrame(
        {
            "Open": values,
            "High": values + 1.0,
            "Low": values - 1.0,
            "Close": values + 0.5,
            "Volume": np.full(6, 1_000_000.0),
            "Dividends": np.zeros(6),
            "Stock Splits": np.zeros(6),
        },
        index=sessions,
    )


def _global_sha(model: torch.nn.Module) -> str:
    return canonical_parameter_sha(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name not in FAST_PARAMETER_NAMES
    )


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _formal_args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values = {
        "config": str(FORMAL_A2_CONFIG),
        "strategy": None,
        "experiment_id": EXPERIMENT_ID,
        "resume": False,
        "force": False,
        "max_cases": None,
        "dry_run": True,
        "data_source": None,
        "snapshot_dir": None,
        "finmultitime_input_root": None,
        "m2_checkpoint": None,
        "m2_encoder_snapshot": None,
        "m2_encoder_snapshot_manifest": None,
        "m2_preformal_evidence_root": None,
        "synthetic_data": False,
        "output_root": str(tmp_path),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exact_initial_checkpoint_and_prompt_anchor(runtime: M2ProductionTraderRuntime) -> None:
    checkpoint = _initial_checkpoint()
    assert sha256_file(checkpoint) == A2_FILE_SHA
    assert canonical_parameter_sha(runtime.base_model) == A2_PARAMETER_SHA
    assert fast_checkpoint_sha(runtime.base_model) == INITIAL_FAST_SHA
    assert _global_sha(runtime.base_model) == INITIAL_GLOBAL_SHA
    state = runtime._load("AAPL")
    assert state.model.fast_parameter_count() == EXPECTED_FAST_PARAMETERS == 165
    assert payload_sha(_encode_optimizer_state(state.optimiser)) == INITIAL_OPTIMISER_SHA
    output = runtime.issue(_graph_state())
    metadata = output["m2_trader_handoff_metadata"]
    assert output["m2_rl_action"] == output["prompt_trader_action"] == "BUY"
    assert metadata["actor_probabilities"] == pytest.approx([2 / 3, 1 / 6, 1 / 6])
    assert metadata["checkpoint_parameter_identity"] == A2_PARAMETER_SHA
    assert metadata["checkpoint_file_identity"] == A2_FILE_SHA


def _variant_config(variant: str, parameter: str, file_sha: str) -> dict:
    return {
        "m2_checkpoint_path": _initial_checkpoint(),
        "m2_encoder_snapshot_path": "unused",
        "m2_encoder_snapshot_manifest_path": "unused",
        "m2_state_root": "unused",
        "m2_variant": variant,
        "m2_method_id": runtime_module.METHOD_ID,
        "m2_representation_identity": runtime_module.REPRESENTATION_IDENTITY,
        "m2_checkpoint_parameter_identity": parameter,
        "m2_checkpoint_file_identity": file_sha,
        "m2_online_candidate_id": runtime_module.ONLINE_CANDIDATE_ID,
        "m2_online_learning_rate": runtime_module.ONLINE_LEARNING_RATE,
        "m2_online_update_epochs": runtime_module.ONLINE_UPDATE_EPOCHS,
        "m2_online_weight_decay": WEIGHT_DECAY,
        "m2_online_gradient_clip": MAX_GRAD_NORM,
    }


def test_a2_rejects_c09_and_other_variants_keep_their_frozen_binding() -> None:
    with pytest.raises(ValueError, match="m2_checkpoint_parameter_identity"):
        M2ProductionTraderRuntime.from_config(
            _variant_config("A2_NO_GLOBAL_PRETRAINING", C09_PARAMETER_SHA, C09_FILE_SHA)
        )
    with pytest.raises(ValueError, match="m2_checkpoint_parameter_identity"):
        M2ProductionTraderRuntime.from_config(
            _variant_config("FULL_M2", A2_PARAMETER_SHA, A2_FILE_SHA)
        )
    with pytest.raises(ValueError, match="m2_checkpoint_parameter_identity"):
        M2ProductionTraderRuntime.from_config(
            _variant_config("A1_NO_ONLINE_ADAPTATION", A2_PARAMETER_SHA, A2_FILE_SHA)
        )
    assert sha256_file(_c09_checkpoint()) == C09_FILE_SHA


def test_o08_update_is_delayed_exactly_once_and_fast_only(
    runtime: M2ProductionTraderRuntime,
) -> None:
    runtime.issue(_graph_state())
    state = runtime._load("AAPL")
    before = _parameter_snapshot(state.model)
    before_fast = fast_checkpoint_sha(state.model)
    before_global = _global_sha(state.model)
    before_optimizer = payload_sha(_encode_optimizer_state(state.optimiser))

    assert runtime.mature_visible(
        "AAPL",
        visible_market_history=_maturity_frame().iloc[:-1],
        cutoff_session="2023-10-12",
        allow_update_for_later_decision=True,
    ) == []
    assert state.credits[0].status == "PENDING"
    assert state.credits[0].counterfactual_r3 is None

    same_close = runtime.issue(_graph_state(session="2023-10-13", next_session=None))
    assert runtime.decision_is_persisted("AAPL", "2023-10-13")
    same_close_fast = same_close["m2_trader_handoff_metadata"][
        "pre_decision_fast_parameter_identity"
    ]
    assert same_close_fast == before_fast

    events = runtime.mature_visible(
        "AAPL",
        visible_market_history=_maturity_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    )
    assert len(events) == 1 and events[0]["status"] == "APPLIED"
    assert state.credits[0].status == "APPLIED"
    after = _parameter_snapshot(state.model)
    mutated = {name for name in before if not torch.equal(before[name], after[name])}
    assert mutated
    assert mutated <= set(FAST_PARAMETER_NAMES)
    assert fast_checkpoint_sha(state.model) != before_fast
    assert _global_sha(state.model) == before_global
    assert payload_sha(_encode_optimizer_state(state.optimiser)) != before_optimizer
    assert len(state.update_records) == 1
    assert state.update_records[0]["pre_fast_sha"] == before_fast
    assert state.update_records[0]["post_fast_sha"] == fast_checkpoint_sha(state.model)

    assert runtime.mature_visible(
        "AAPL",
        visible_market_history=_maturity_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    ) == []
    assert len(state.update_records) == 1


def test_several_eligible_credits_produce_parameter_mutating_updates(
    runtime: M2ProductionTraderRuntime,
) -> None:
    sessions = ["2023-10-06", "2023-10-13", "2023-10-20", "2023-10-27"]
    runtime.issue(_graph_state(session=sessions[0], next_session=sessions[1]))
    for index in range(1, len(sessions)):
        next_session = sessions[index + 1] if index + 1 < len(sessions) else None
        runtime.issue(
            _graph_state(session=sessions[index], next_session=next_session)
        )
        events = runtime.mature_visible(
            "AAPL",
            visible_market_history=_maturity_frame(sessions[index - 1]),
            cutoff_session=sessions[index],
            allow_update_for_later_decision=True,
        )
        assert len(events) == 1 and events[0]["status"] == "APPLIED"
    state = runtime._load("AAPL")
    assert len(state.update_records) == 3
    assert all(
        record["pre_fast_sha"] != record["post_fast_sha"]
        for record in state.update_records
    )
    assert len(state.applied_event_ids) == 3
    assert _global_sha(state.model) == INITIAL_GLOBAL_SHA


def test_safe_resume_before_and_after_update_and_symbol_isolation(
    runtime: M2ProductionTraderRuntime,
) -> None:
    first = runtime.issue(_graph_state())
    runtime.issue(_graph_state("AMZN"))
    b_state = runtime._load("AMZN")
    b_fast = fast_checkpoint_sha(b_state.model)
    b_optimizer = payload_sha(_encode_optimizer_state(b_state.optimiser))
    b_identity = runtime.state_identity("AMZN")

    resumed = M2ProductionTraderRuntime(
        checkpoint_path=_initial_checkpoint(),
        state_root=runtime.state_root,
        encoder=_encoder,
        checkpoint_parameter_identity=A2_PARAMETER_SHA,
        checkpoint_file_identity=A2_FILE_SHA,
    )
    assert resumed.issue(_graph_state()) == first
    resumed.issue(_graph_state(session="2023-10-13", next_session=None))
    resumed.mature_visible(
        "AAPL",
        visible_market_history=_maturity_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    )
    a_state = resumed._load("AAPL")
    applied_fast = fast_checkpoint_sha(a_state.model)
    applied_optimizer = payload_sha(_encode_optimizer_state(a_state.optimiser))
    applied_id = a_state.credits[0].application_id

    after = M2ProductionTraderRuntime(
        checkpoint_path=_initial_checkpoint(),
        state_root=runtime.state_root,
        encoder=_encoder,
        checkpoint_parameter_identity=A2_PARAMETER_SHA,
        checkpoint_file_identity=A2_FILE_SHA,
    )
    reloaded = after._load("AAPL")
    assert fast_checkpoint_sha(reloaded.model) == applied_fast
    assert payload_sha(_encode_optimizer_state(reloaded.optimiser)) == applied_optimizer
    assert reloaded.credits[0].application_id == applied_id
    assert len(reloaded.applied_event_ids) == len(reloaded.update_records) == 1
    assert after.mature_visible(
        "AAPL",
        visible_market_history=_maturity_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    ) == []

    reloaded_b = after._load("AMZN")
    assert fast_checkpoint_sha(reloaded_b.model) == b_fast
    assert payload_sha(_encode_optimizer_state(reloaded_b.optimiser)) == b_optimizer
    assert after.state_identity("AMZN") == b_identity
    assert _global_sha(reloaded.model) == INITIAL_GLOBAL_SHA


def test_changed_input_resume_guard_is_preserved(runtime: M2ProductionTraderRuntime) -> None:
    runtime.issue(_graph_state())
    changed = _graph_state()
    changed["investment_plan"] = "Changed after publication."
    with pytest.raises(RuntimeError, match="changed M2 decision input"):
        runtime.issue(changed)


def test_formal_config_diff_and_isolation_are_exact(tmp_path: Path) -> None:
    a2 = json.loads(FORMAL_A2_CONFIG.read_text(encoding="utf-8"))
    m2 = json.loads(FORMAL_M2_CONFIG.read_text(encoding="utf-8"))
    validate_formal_a2_config(a2)
    assert a2 == FORMAL_A2_CONTRACT
    changed = {key for key in set(a2) | set(m2) if a2.get(key) != m2.get(key)}
    assert changed == {
        "experiment_id_template",
        "m2_variant",
        "m2_checkpoint_parameter_identity",
        "m2_checkpoint_file_identity",
    }
    a1 = {
        **FORMAL_M2_CONTRACT,
        "experiment_id_template": "A1_no_online_adaptation_2024H1",
        "m2_variant": "A1_NO_ONLINE_ADAPTATION",
    }
    graphs = []
    for index, config in enumerate((m2, a1, a2)):
        graph = resolve_graph_config(
            config,
            results_dir=tmp_path / str(index) / "results",
            data_cache_dir=tmp_path / str(index) / "cache",
            historical_memory_dir=tmp_path / str(index) / "memory",
        )
        graphs.append((config, compute_graph_config_sha256(graph)))
    assert len({identity for _, identity in graphs}) == 3
    memory_paths = {
        runtime_experiment_memory_path(
            tmp_path / "runtime",
            experiment_id=config["experiment_id_template"],
            graph_config_sha256=identity,
            memory_lineage_id="fresh-lineage",
            symbol="AAPL",
            finmultitime_evidence_enabled=True,
            finmultitime_bundle_scope="FORMAL",
            finmultitime_bundle_identity=config[
                "finmultitime_expected_input_bundle_identity"
            ],
        )
        for config, identity in graphs
    }
    assert len(memory_paths) == 3


def test_formal_a2_dry_run_is_offline_and_runs_no_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    _, status = weekly_runner.execute(_formal_args(tmp_path))
    output = json.loads(capsys.readouterr().out)
    assert status == "dry-run"
    assert output["protocol_mode"] == "formal_a2"
    assert output["expected_paid_agent_cases"] == 78
    assert output["planned_cases"] == 78
    assert output["will_execute"] is False
    assert output["do_not_execute_paid_agent_cases"] is True
    assert not (tmp_path / EXPERIMENT_ID).exists()


def test_formal_a2_rejects_identity_drift_and_force(tmp_path: Path) -> None:
    for field, changed in (
        ("m2_variant", "FULL_M2"),
        ("m2_checkpoint_parameter_identity", C09_PARAMETER_SHA),
        ("m2_online_learning_rate", 1e-4),
        ("temperature", 0.1),
    ):
        config = copy.deepcopy(FORMAL_A2_CONTRACT)
        config[field] = changed
        with pytest.raises(ValueError, match="contract mismatch"):
            validate_formal_a2_config(config)
    with pytest.raises(ValueError, match="requires experiment_id"):
        weekly_runner.execute(_formal_args(tmp_path, experiment_id="renamed"))
    with pytest.raises(ValueError, match="--force is forbidden"):
        weekly_runner.execute(
            _formal_args(
                tmp_path,
                dry_run=False,
                force=True,
                snapshot_dir=str(tmp_path / "snapshot"),
                finmultitime_input_root=str(tmp_path / "evidence"),
                m2_checkpoint=str(_initial_checkpoint()),
                m2_encoder_snapshot=str(tmp_path / "encoder"),
                m2_encoder_snapshot_manifest=str(tmp_path / "encoder.json"),
            )
        )


def test_frozen_o08_and_treatment_constants_are_unchanged() -> None:
    assert runtime_module.ONLINE_LEARNING_RATE == 1e-3
    assert runtime_module.ONLINE_UPDATE_EPOCHS == 2
    assert WEIGHT_DECAY == 1e-4
    assert MAX_GRAD_NORM == 0.5
    assert EXPECTED_FAST_PARAMETERS == 165
    assert SEED == 20260816
    assert FAST_PARAMETER_NAMES == (
        "residual_head.weight",
        "residual_head.bias",
        "gate_head.weight",
        "gate_head.bias",
        "value_head.weight",
        "value_head.bias",
    )
