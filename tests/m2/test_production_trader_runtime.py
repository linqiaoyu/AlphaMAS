from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.m2 import runtime as runtime_module
from tradingagents.m2.runtime import (
    C09_FILE_SHA,
    C09_PARAMETER_SHA,
    FrozenQwenEncoder,
    M2ProductionTraderRuntime,
    parse_authoritative_m2_action,
    parse_prompt_trader_action,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT.parent / "AlphaMAS-Experiments"
C09 = (
    EXPERIMENTS
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
    if not C09.is_file():
        pytest.fail(f"canonical C09 archive is unavailable: {C09}")
    return M2ProductionTraderRuntime(
        checkpoint_path=C09,
        state_root=tmp_path / "states",
        encoder=_encoder,
    )


def _snapshot(symbol: str = "AAPL", session: str = "2023-10-06") -> dict:
    timestamp = ExchangeSchedule().session_close(session).to_pydatetime()
    return PortfolioSnapshot(
        timestamp=timestamp,
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


def _graph_state(symbol: str = "AAPL") -> dict:
    return {
        "company_of_interest": symbol,
        "trade_date": "2023-10-06",
        "investment_plan": "Frozen Research Manager plan.",
        "trader_investment_plan": (
            "**Action**: Buy\n\n**Reasoning**: Frozen prompt reasoning.\n\n"
            "FINAL TRANSACTION PROPOSAL: **BUY**"
        ),
        "m2_portfolio_snapshot": _snapshot(symbol),
        "m2_portfolio_reward_state": {
            "cash": 100_000.0,
            "quantity": 0.0,
            "average_entry_price": 0.0,
            "open_position_commission": 0.0,
        },
        "m2_next_decision_session": None,
    }


def test_c09_and_pending_credit_boundary(runtime: M2ProductionTraderRuntime) -> None:
    assert runtime.checkpoint_parameter_identity == C09_PARAMETER_SHA
    assert runtime.checkpoint_file_identity == C09_FILE_SHA
    output = runtime.issue(_graph_state())
    metadata = output["m2_trader_handoff_metadata"]
    assert metadata["actor_observation_dimension"] == 3080
    assert metadata["portfolio_state"] == [1.0, 0.0, 0.0, 0.0]
    assert metadata["no_second_llm_call"] is True
    assert parse_authoritative_m2_action(output["trader_investment_plan"]) == output["m2_rl_action"]
    payload = json.loads((runtime.state_root / "AAPL/runtime_state.json").read_text())
    credit = payload["credits"][0]
    assert credit["status"] == "PENDING"
    assert credit["counterfactual_r3"] is None
    assert "bars" not in credit and "future" not in credit
    assert len(payload["fast_parameters"]) == 6


def test_override_handoff_has_one_parser_authority(runtime: M2ProductionTraderRuntime) -> None:
    state = runtime._load("AAPL")
    with torch.no_grad():
        state.model.gate_head.bias.fill_(1000.0)
        state.model.residual_head.bias.copy_(torch.tensor([-1000.0, -1000.0, 1000.0]))
    graph_state = _graph_state()
    graph_state["trade_date"] = "2023-10-13"
    graph_state["m2_portfolio_snapshot"] = _snapshot(session="2023-10-13")
    output = runtime.issue(graph_state)
    assert output["m2_override"] is True
    assert output["prompt_trader_action"] == "BUY"
    assert output["m2_rl_action"] == "SELL"
    assert parse_prompt_trader_action(output["prompt_trader_proposal_original"]) == "BUY"
    assert parse_authoritative_m2_action(output["trader_investment_plan"]) == "SELL"


def test_safe_resume_and_symbol_isolation(runtime: M2ProductionTraderRuntime) -> None:
    assert runtime.decision_is_persisted("AAPL", "2023-10-06") is False
    first = runtime.issue(_graph_state())
    assert runtime.decision_is_persisted("AAPL", "2023-10-06") is True
    before = runtime.state_identity("AAPL")
    resumed = M2ProductionTraderRuntime(
        checkpoint_path=C09,
        state_root=runtime.state_root,
        encoder=_encoder,
    )
    second = resumed.issue(_graph_state())
    assert first == second
    assert resumed.state_identity("AAPL") == before
    resumed.issue(_graph_state("AMZN"))
    assert resumed.state_identity("AAPL") == before
    assert (runtime.state_root / "AMZN/runtime_state.json").is_file()


def test_from_config_rejects_research_identity_drift() -> None:
    config = {
        "m2_checkpoint_path": C09,
        "m2_encoder_snapshot_path": "unused",
        "m2_encoder_snapshot_manifest_path": "unused",
        "m2_state_root": "unused",
        "m2_variant": "FULL_M2",
        "m2_method_id": "wrong",
        "m2_representation_identity": runtime_module.REPRESENTATION_IDENTITY,
        "m2_checkpoint_parameter_identity": C09_PARAMETER_SHA,
        "m2_checkpoint_file_identity": C09_FILE_SHA,
        "m2_online_candidate_id": runtime_module.ONLINE_CANDIDATE_ID,
        "m2_online_learning_rate": runtime_module.ONLINE_LEARNING_RATE,
        "m2_online_update_epochs": runtime_module.ONLINE_UPDATE_EPOCHS,
        "m2_online_weight_decay": runtime_module.WEIGHT_DECAY,
        "m2_online_gradient_clip": runtime_module.MAX_GRAD_NORM,
    }
    with pytest.raises(ValueError, match="m2_method_id"):
        M2ProductionTraderRuntime.from_config(config)


def test_terminal_credit_is_scored_and_archived_without_update(
    runtime: M2ProductionTraderRuntime,
) -> None:
    runtime.issue(_graph_state())
    sessions = ExchangeSchedule().calendar.sessions_window("2023-10-06", 6).tz_localize(None)
    frame = pd.DataFrame(
        {
            "Open": np.linspace(100.0, 105.0, 6),
            "High": np.linspace(101.0, 106.0, 6),
            "Low": np.linspace(99.0, 104.0, 6),
            "Close": np.linspace(100.0, 105.0, 6),
            "Volume": np.full(6, 1_000_000.0),
            "Dividends": np.zeros(6),
            "Stock Splits": np.zeros(6),
        },
        index=sessions,
    )
    fast_before = runtime.state_identity("AAPL")
    events = runtime.mature_visible(
        "AAPL",
        visible_market_history=frame,
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=False,
    )
    assert len(events) == 1 and events[0]["status"] == "ARCHIVED"
    payload = json.loads((runtime.state_root / "AAPL/runtime_state.json").read_text())
    assert payload["credits"][0]["status"] == "ARCHIVED"
    assert payload["update_records"] == []
    assert runtime.state_identity("AAPL") != fast_before


def test_prompt_parser_rejects_ambiguous_terminal_actions() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        parse_prompt_trader_action(
            "FINAL TRANSACTION PROPOSAL: **BUY**\n"
            "FINAL TRANSACTION PROPOSAL: **SELL**"
        )


def test_frozen_encoder_verifies_complete_snapshot_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("frozen", encoding="utf-8")
    files = [
        {
            "path": "config.json",
            "size": 6,
            "sha256": hashlib.sha256(b"frozen").hexdigest(),
        }
    ]
    identity = hashlib.sha256(
        json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"identity_sha256": identity, "files": files}), encoding="utf-8"
    )
    monkeypatch.setattr(runtime_module, "MODEL_SNAPSHOT_IDENTITY", identity)
    FrozenQwenEncoder(snapshot, manifest)
    (snapshot / "unexpected.txt").write_text("drift", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bytes differ"):
        FrozenQwenEncoder(snapshot, manifest)
