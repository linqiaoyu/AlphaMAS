"""Pre-Formal correctness gates for the preregistered A1 treatment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tradingagents.agents.utils.memory_namespace import runtime_experiment_memory_path
from tradingagents.backtesting.cache import DecisionCache, cache_key
from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.config import (
    FORMAL_A1_CONTRACT,
    FORMAL_M2_CONTRACT,
    compute_graph_config_sha256,
    resolve_graph_config,
    validate_formal_a1_config,
)
from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.m2.runtime import (
    A1_ARCHIVE_REASON,
    A1_VARIANT,
    C09_PARAMETER_SHA,
    FULL_M2_VARIANT,
    M2ProductionTraderRuntime,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT.parent / "AlphaMAS-Experiments"
C09 = (
    EXPERIMENTS
    / "experiments/M2/development/global_selection_v1/selected_global_checkpoint/model.pt"
)
FORMAL_A1 = ROOT / "configs/backtest_a1_no_online_adaptation_2024h1.json"
FORMAL_M2 = ROOT / "configs/backtest_m2_2024h1.json"
M1_BUNDLE_IDENTITY = "30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121"


def _encoder(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        row = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
        rows.append(row / np.linalg.norm(row))
    return np.stack(rows)


def _runtime(tmp_path: Path, variant: str) -> M2ProductionTraderRuntime:
    if not C09.is_file():
        pytest.fail(f"canonical C09 archive is unavailable: {C09}")
    return M2ProductionTraderRuntime(
        checkpoint_path=C09,
        state_root=tmp_path / variant,
        encoder=_encoder,
        variant=variant,
        online_adaptation_enabled=variant != A1_VARIANT,
    )


def _snapshot(symbol: str, session: str, close: float = 100.0) -> dict:
    return PortfolioSnapshot(
        timestamp=ExchangeSchedule().session_close(session).to_pydatetime(),
        session=session,
        symbol=symbol,
        cash=100_000.0,
        quantity=0.0,
        close_price=close,
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
    *,
    next_session: str | None = "2023-10-13",
) -> dict:
    return {
        "company_of_interest": symbol,
        "trade_date": session,
        "investment_plan": "Frozen pre-2024 Research Manager plan.",
        "trader_investment_plan": (
            "**Action**: Buy\n\n**Reasoning**: Frozen pre-2024 prompt.\n\n"
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


def _market_frame() -> pd.DataFrame:
    sessions = ExchangeSchedule().calendar.sessions_window("2023-10-06", 6).tz_localize(None)
    return pd.DataFrame(
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


def test_a1_formal_variant_resolves_and_starts_from_exact_c09(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(FORMAL_A1.read_text())
    validate_formal_a1_config(config)
    resolved = resolve_graph_config(
        config,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )
    resolved.update(
        {
            "m2_checkpoint_path": C09,
            "m2_encoder_snapshot_path": "unused",
            "m2_encoder_snapshot_manifest_path": "unused",
            "m2_state_root": tmp_path / "state",
        }
    )
    monkeypatch.setattr("tradingagents.m2.runtime.FrozenQwenEncoder", lambda *_args: _encoder)
    runtime = M2ProductionTraderRuntime.from_config(resolved)
    assert runtime.variant == A1_VARIANT
    assert runtime.online_adaptation_enabled is False
    assert runtime.checkpoint_parameter_identity == C09_PARAMETER_SHA
    assert runtime.policy_parameter_identity("AAPL") == C09_PARAMETER_SHA


def test_formal_a1_diff_is_only_identity_and_preregistered_treatment() -> None:
    m2 = json.loads(FORMAL_M2.read_text())
    a1 = json.loads(FORMAL_A1.read_text())
    assert m2 == FORMAL_M2_CONTRACT
    assert a1 == FORMAL_A1_CONTRACT
    diff = {name for name in a1 if a1[name] != m2[name]}
    assert diff == {"experiment_id_template", "m2_variant"}
    assert a1["m2_variant"] == A1_VARIANT
    assert a1["finmultitime_expected_input_bundle_identity"] == M1_BUNDLE_IDENTITY


def test_preupdate_actor_equivalence_and_treatment_distinct_state(
    tmp_path: Path,
) -> None:
    full = _runtime(tmp_path, FULL_M2_VARIANT)
    a1 = _runtime(tmp_path, A1_VARIANT)
    full_output = full.issue(_graph_state())
    a1_output = a1.issue(_graph_state())
    full_meta = full_output["m2_trader_handoff_metadata"]
    a1_meta = a1_output["m2_trader_handoff_metadata"]
    assert full_meta["actor_logits"] == a1_meta["actor_logits"]
    assert full_meta["actor_probabilities"] == a1_meta["actor_probabilities"]
    assert full_meta["authoritative_action"] == a1_meta["authoritative_action"]
    assert full.fast_parameter_identity("AAPL") == a1.fast_parameter_identity("AAPL")
    assert full.policy_parameter_identity("AAPL") == C09_PARAMETER_SHA
    assert a1.policy_parameter_identity("AAPL") == C09_PARAMETER_SHA
    assert full.state_identity("AAPL") != a1.state_identity("AAPL")
    assert a1_meta["m2_variant"] == A1_VARIANT
    assert a1_meta["online_adaptation_enabled"] is False


def test_matured_a1_credit_cannot_step_or_mutate_any_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path, A1_VARIANT)
    initial_fast = runtime.fast_parameter_identity("AAPL")
    initial_global = runtime.global_parameter_identity("AAPL")
    initial_policy = runtime.policy_parameter_identity("AAPL")
    initial_optimiser = runtime.optimiser_state_identity("AAPL")
    runtime.issue(_graph_state())
    state = runtime._load("AAPL")
    forbidden_step = MagicMock(side_effect=AssertionError("optimiser.step called"))
    monkeypatch.setattr(state.optimiser, "step", forbidden_step)

    events = runtime.mature_visible(
        "AAPL",
        visible_market_history=_market_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    )

    assert len(events) == 1
    event = events[0]
    assert event["status"] == "ARCHIVED"
    assert event["archive_reason"] == A1_ARCHIVE_REASON
    assert event["pre_fast_sha"] == event["post_fast_sha"] == initial_fast
    assert event["pre_global_sha"] == event["post_global_sha"] == initial_global
    assert event["pre_policy_parameter_sha"] == initial_policy
    assert event["post_policy_parameter_sha"] == C09_PARAMETER_SHA
    assert event["pre_optimiser_state_identity"] == initial_optimiser
    assert event["post_optimiser_state_identity"] == initial_optimiser
    assert runtime.parameter_mutating_update_count("AAPL") == 0
    assert runtime.fast_parameter_identity("AAPL") == initial_fast
    assert runtime.global_parameter_identity("AAPL") == initial_global
    assert runtime.policy_parameter_identity("AAPL") == C09_PARAMETER_SHA
    assert runtime.optimiser_state_identity("AAPL") == initial_optimiser
    forbidden_step.assert_not_called()


def test_a1_chronology_and_resume_are_pit_safe_and_exactly_once(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, A1_VARIANT)
    first = runtime.issue(_graph_state())
    before_maturity = runtime.state_identity("AAPL")
    resumed = M2ProductionTraderRuntime(
        checkpoint_path=C09,
        state_root=runtime.state_root,
        encoder=_encoder,
        variant=A1_VARIANT,
        online_adaptation_enabled=False,
    )
    assert resumed.issue(_graph_state()) == first
    assert resumed.state_identity("AAPL") == before_maturity
    assert (
        resumed.mature_visible(
            "AAPL",
            visible_market_history=_market_frame(),
            cutoff_session="2023-10-12",
            allow_update_for_later_decision=True,
        )
        == []
    )

    events = resumed.mature_visible(
        "AAPL",
        visible_market_history=_market_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    )
    assert len(events) == 1
    after_maturity = resumed.state_identity("AAPL")
    post_resume = M2ProductionTraderRuntime(
        checkpoint_path=C09,
        state_root=runtime.state_root,
        encoder=_encoder,
        variant=A1_VARIANT,
        online_adaptation_enabled=False,
    )
    assert post_resume.state_identity("AAPL") == after_maturity
    assert (
        post_resume.mature_visible(
            "AAPL",
            visible_market_history=_market_frame(),
            cutoff_session="2023-10-13",
            allow_update_for_later_decision=True,
        )
        == []
    )
    next_output = post_resume.issue(_graph_state(session="2023-10-13", next_session=None))
    assert next_output["m2_trader_handoff_metadata"]["m2_variant"] == A1_VARIANT
    assert post_resume.parameter_mutating_update_count("AAPL") == 0


def test_duplicate_credit_changed_input_and_cross_symbol_mutation_are_blocked(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, A1_VARIANT)
    runtime.issue(_graph_state())
    runtime.issue(_graph_state())
    payload = json.loads((runtime.state_root / "AAPL/runtime_state.json").read_text())
    assert len(payload["credits"]) == 1
    changed = _graph_state()
    changed["investment_plan"] = "Changed after torn publication."
    with pytest.raises(RuntimeError, match="changed M2 decision input"):
        runtime.issue(changed)

    aapl_fast = runtime.fast_parameter_identity("AAPL")
    aapl_state = runtime.state_identity("AAPL")
    runtime.issue(_graph_state("AMZN"))
    runtime.mature_visible(
        "AMZN",
        visible_market_history=_market_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    )
    assert runtime.fast_parameter_identity("AAPL") == aapl_fast
    assert runtime.state_identity("AAPL") == aapl_state
    assert runtime.parameter_mutating_update_count("AMZN") == 0


def test_terminal_credit_uses_same_a1_archive_semantics(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, A1_VARIANT)
    runtime.issue(_graph_state(next_session=None))
    events = runtime.mature_visible(
        "AAPL",
        visible_market_history=_market_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=False,
    )
    assert len(events) == 1
    assert events[0]["status"] == "ARCHIVED"
    assert events[0]["archive_reason"] == A1_ARCHIVE_REASON
    assert runtime.parameter_mutating_update_count("AAPL") == 0


def test_full_m2_update_path_remains_active_and_unchanged(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, FULL_M2_VARIANT)
    runtime.issue(_graph_state())
    events = runtime.mature_visible(
        "AAPL",
        visible_market_history=_market_frame(),
        cutoff_session="2023-10-13",
        allow_update_for_later_decision=True,
    )
    assert len(events) == 1 and events[0]["status"] == "APPLIED"
    assert runtime.parameter_mutating_update_count("AAPL") == 1


def test_a1_cache_and_memory_namespaces_cannot_reuse_full_m2(tmp_path: Path) -> None:
    full = _runtime(tmp_path, FULL_M2_VARIANT)
    a1 = _runtime(tmp_path, A1_VARIANT)
    full_identity = full.cache_identity_for_decision("AAPL", "2023-10-06")
    a1_identity = a1.cache_identity_for_decision("AAPL", "2023-10-06")
    common = {
        "symbol": "AAPL",
        "session": "2023-10-06",
        "evidence": "same",
        "prompt": "same",
        "portfolio": "same",
        "market_history": "same",
    }
    full_key = cache_key(common | {"m2_variant": FULL_M2_VARIANT, "runtime": full_identity})
    a1_key = cache_key(common | {"m2_variant": A1_VARIANT, "runtime": a1_identity})
    assert full_identity != a1_identity
    assert full_key != a1_key
    cache = DecisionCache(tmp_path / "full-cache")
    cache.save_success(full_key, {"action": "BUY"})
    assert cache.load(a1_key) is None

    m2_config = resolve_graph_config(
        json.loads(FORMAL_M2.read_text()),
        results_dir=tmp_path / "m2-results",
        data_cache_dir=tmp_path / "m2-cache",
        historical_memory_dir=tmp_path / "m2-memory",
    )
    a1_config = resolve_graph_config(
        json.loads(FORMAL_A1.read_text()),
        results_dir=tmp_path / "a1-results",
        data_cache_dir=tmp_path / "a1-cache",
        historical_memory_dir=tmp_path / "a1-memory",
    )
    m2_sha = compute_graph_config_sha256(m2_config)
    a1_sha = compute_graph_config_sha256(a1_config)
    assert m2_sha != a1_sha
    m2_memory = runtime_experiment_memory_path(
        tmp_path / "M2/runtime/memory",
        experiment_id="M2_agentic_rl_2024H1",
        graph_config_sha256=m2_sha,
        memory_lineage_id="m2-lineage",
        symbol="AAPL",
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope="FORMAL",
        finmultitime_bundle_identity=M1_BUNDLE_IDENTITY,
    )
    a1_memory = runtime_experiment_memory_path(
        tmp_path / "A1/runtime/memory",
        experiment_id="A1_no_online_adaptation_2024H1",
        graph_config_sha256=a1_sha,
        memory_lineage_id="a1-lineage",
        symbol="AAPL",
        finmultitime_evidence_enabled=True,
        finmultitime_bundle_scope="FORMAL",
        finmultitime_bundle_identity=M1_BUNDLE_IDENTITY,
    )
    assert m2_memory != a1_memory
    assert "A1_no_online_adaptation_2024H1" in a1_memory.parts
