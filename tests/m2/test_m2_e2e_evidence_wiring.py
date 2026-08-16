from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.agents.utils.memory_namespace import (
    experiment_memory_namespace_component,
)
from tradingagents.backtesting.config import resolve_graph_config
from tradingagents.evidence.finmultitime import (
    FrozenEvidenceError,
    FrozenFinMultiTimeEvidenceStore,
)
from tradingagents.evidence.m2_preformal import (
    CORPUS_IDENTITY,
    DECISION_SESSION,
    PACKETS,
    FrozenM2E2EEvidenceStore,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.m2.runtime import M2ProductionTraderRuntime
from tradingagents.runtime.run_context import RunContext

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT.parent / "AlphaMAS-Experiments"
E2E_CONFIG = ROOT / "configs/backtest_m2_e2e_pilot.json"
FORMAL_M2_CONFIG = ROOT / "configs/backtest_m2_2024h1.json"
FORMAL_M1_CONFIG = ROOT / "configs/backtest_m1_2024h1.json"
FORMAL_M0_CONFIG = ROOT / "configs/backtest_m0_2024h1.json"
EVIDENCE_ROOT = (
    EXPERIMENTS / "experiments/M2/development/preformal_evidence_v1"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(config: dict, tmp_path: Path) -> dict:
    return resolve_graph_config(
        config,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "cache",
        historical_memory_dir=tmp_path / "memory",
    )


class _M2StoreSelected(RuntimeError):
    pass


class _M1StoreSelected(RuntimeError):
    pass


def test_e2e_controls_and_mount_propagate_to_resolved_graph(tmp_path: Path) -> None:
    evidence_root = EVIDENCE_ROOT.resolve()
    resolved = _resolve(
        {**_load(E2E_CONFIG), "m2_preformal_evidence_root": evidence_root},
        tmp_path,
    )

    assert resolved["m2_preformal_evidence_enabled"] is True
    assert resolved["m2_preformal_evidence_identity"] == CORPUS_IDENTITY
    assert resolved["m2_preformal_evidence_role"] == "E2E_PILOT"
    assert resolved["m2_preformal_evidence_root"] == evidence_root


def test_e2e_graph_selects_frozen_m2_store(tmp_path: Path) -> None:
    resolved = _resolve(
        {
            **_load(E2E_CONFIG),
            "m2_preformal_evidence_root": EVIDENCE_ROOT.resolve(),
        },
        tmp_path,
    )

    with (
        patch.object(
            FrozenM2E2EEvidenceStore,
            "from_config",
            side_effect=_M2StoreSelected,
        ),
        patch.object(
            FrozenFinMultiTimeEvidenceStore,
            "from_config",
            side_effect=AssertionError("M1 store must not be selected for M2 E2E"),
        ),
        pytest.raises(_M2StoreSelected),
    ):
        TradingAgentsGraph(config=resolved)


def test_all_eight_exact_e2e_packets_resolve() -> None:
    store = FrozenM2E2EEvidenceStore(EVIDENCE_ROOT)

    identities = [store.case_identity(symbol, DECISION_SESSION) for symbol in PACKETS]

    assert len(identities) == 8
    assert all(item["input_bundle_identity"] == CORPUS_IDENTITY for item in identities)
    assert {item["case_id"] for item in identities} == {
        f"{symbol}:{DECISION_SESSION}" for symbol in PACKETS
    }


def test_e2e_store_fails_closed_on_session_symbol_and_identity(tmp_path: Path) -> None:
    store = FrozenM2E2EEvidenceStore(EVIDENCE_ROOT)
    with pytest.raises(FrozenEvidenceError, match="no exact"):
        store.case_identity("AAPL", "2023-10-13")
    with pytest.raises(FrozenEvidenceError, match="no exact"):
        store.case_identity("UNKNOWN", DECISION_SESSION)

    resolved = _resolve(
        {
            **_load(E2E_CONFIG),
            "m2_preformal_evidence_root": EVIDENCE_ROOT.resolve(),
            "m2_checkpoint_path": tmp_path / "model.pt",
            "m2_encoder_snapshot_path": tmp_path / "encoder",
            "m2_encoder_snapshot_manifest_path": tmp_path / "encoder.json",
            "m2_state_root": tmp_path / "state",
            "m2_preformal_evidence_identity": "corrupted",
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="frozen E2E evidence contract mismatch"):
        M2ProductionTraderRuntime.from_config(resolved)


def test_e2e_store_scope_has_an_isolated_fail_closed_memory_namespace(
    tmp_path: Path,
) -> None:
    resolved = _resolve(
        {
            **_load(E2E_CONFIG),
            "m2_preformal_evidence_root": EVIDENCE_ROOT.resolve(),
        },
        tmp_path,
    )
    graph = object.__new__(TradingAgentsGraph)
    graph.config = resolved | {"historical_memory_lineage_id": "retry-3-preflight"}
    graph.finmultitime_evidence_store = FrozenM2E2EEvidenceStore(EVIDENCE_ROOT)
    path = Path(
        graph._historical_memory_config(
            "AAPL",
            RunContext.historical(
                DECISION_SESSION,
                experiment_id="M2_e2e_pilot_2023_10_06",
                memory_lineage_id="retry-3-preflight",
            ),
        )["memory_log_path"]
    )

    assert "finmultitime-m2_e2e_pilot-3e9bb6e66fcd998c" in path.as_posix()
    with pytest.raises(ValueError, match="bundle scope"):
        experiment_memory_namespace_component(
            finmultitime_evidence_enabled=True,
            finmultitime_bundle_scope="UNKNOWN",
            finmultitime_bundle_identity=CORPUS_IDENTITY,
        )


def test_formal_m2_stays_on_frozen_m1_store(tmp_path: Path) -> None:
    resolved = _resolve(_load(FORMAL_M2_CONFIG), tmp_path)
    assert resolved["m2_preformal_evidence_enabled"] is False
    assert resolved["finmultitime_evidence_enabled"] is True

    with (
        patch.object(
            FrozenM2E2EEvidenceStore,
            "from_config",
            side_effect=AssertionError("M2 E2E store must not be selected for Formal M2"),
        ),
        patch.object(
            FrozenFinMultiTimeEvidenceStore,
            "from_config",
            side_effect=_M1StoreSelected,
        ),
        pytest.raises(_M1StoreSelected),
    ):
        TradingAgentsGraph(config=resolved)


def test_m0_and_m1_evidence_modes_are_unchanged(tmp_path: Path) -> None:
    m0 = _resolve(_load(FORMAL_M0_CONFIG), tmp_path / "m0")
    m1 = _resolve(_load(FORMAL_M1_CONFIG), tmp_path / "m1")

    assert m0["m2_preformal_evidence_enabled"] is False
    assert m0["finmultitime_evidence_enabled"] is False
    assert m1["m2_preformal_evidence_enabled"] is False
    assert m1["finmultitime_evidence_enabled"] is True
