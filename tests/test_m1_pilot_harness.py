from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.finmultitime.build_m1_pilot_inputs import (
    PILOT_DATASET_ID,
    PILOT_SESSIONS,
    _pilot_context,
)
from tradingagents.backtesting.cache import cache_key
from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.backtesting.strategies import TradingAgentsStrategy
from tradingagents.evidence.finmultitime import (
    FrozenEvidenceError,
    FrozenFinMultiTimeEvidenceStore,
)
from tradingagents.graph.checkpointer import thread_id
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.run_context import AuditTrail, RunContext, activate_run_context

REPOSITORY = Path(__file__).resolve().parents[1]
PILOT_CONFIG = REPOSITORY / "configs" / "m1_pilot_aapl_2023q4.json"
PILOT_ARCHIVE_COMMIT = "376a214a9cbd0a650b7e5ac96d6275ae7cb5974a"


def _root_from_env() -> Path:
    value = os.environ.get("ALPHAMAS_M1_PILOT_INPUT_ROOT")
    if not value:
        pytest.skip("pilot input archive is not available")
    root = Path(value).resolve()
    if not root.is_dir():
        pytest.skip(f"pilot input archive is not available: {root}")
    return root


def _identity(root: Path) -> tuple[str, str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    packet_sha = manifest["final_evidence_packet_manifest"]["sha256"]
    paths = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in (
            "manifest.json",
            "manifests/input_bundle_manifest.json",
            "manifests/input_bundle_checksums.json",
            "manifests/evidence_packet_manifest.json",
            "manifests/processed_sha256.json",
        )
    }
    bundle_identity = hashlib.sha256(
        json.dumps(paths, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return packet_sha, bundle_identity


def _pilot_store(root: Path) -> FrozenFinMultiTimeEvidenceStore:
    packet_sha, bundle_identity = _identity(root)
    return FrozenFinMultiTimeEvidenceStore(
        root,
        bundle_scope="PILOT",
        expected_packet_manifest_sha256=packet_sha,
        expected_input_bundle_identity=bundle_identity,
        expected_archive_commit=PILOT_ARCHIVE_COMMIT,
    )


def _cache_kwargs() -> dict:
    snapshot = PortfolioSnapshot(
        datetime(2023, 10, 6, 20, tzinfo=timezone.utc), "2023-10-06", "AAPL",
        100_000, 0, 100, 0, 100_000, 0, 0, 0, 0, 0, 0, 100_000,
    )
    return {
        "symbol": "AAPL",
        "decision_session": "2023-10-06",
        "decision_time": datetime(2023, 10, 6, 20, tzinfo=timezone.utc),
        "market_history": pd.DataFrame(
            {"Close": [99.0, 100.0]}, index=pd.to_datetime(["2023-10-05", "2023-10-06"])
        ),
        "portfolio_snapshot": snapshot,
        "context": {"experiment_id": "M1_pilot_aapl_2023q4_4w_v1", "point_in_time": True},
    }


def test_pilot_schedule_and_bundle_contract() -> None:
    context = _pilot_context()
    assert [event["decision_session"] for event in context["events"]] == list(PILOT_SESSIONS)
    assert all(event["execution_session"] for event in context["events"])
    assert PILOT_DATASET_ID == "finmultitime_m1_pilot_aapl_2023q4_4w_v1"


def test_pilot_config_pins_the_frozen_archive_commit() -> None:
    config = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))

    assert config["finmultitime_archive_commit"] == PILOT_ARCHIVE_COMMIT


def test_pilot_provenance_records_archive_and_route_identity() -> None:
    store = _pilot_store(_root_from_env())
    context = RunContext.historical(
        "2023-10-06", experiment_id="M1_pilot_aapl_2023q4_4w_v1"
    )
    audit = AuditTrail(context)

    with activate_run_context(context, audit):
        routed = store.get_routed_evidence("AAPL", "2023-10-06", "market")

    assert routed is not None
    identity = store.case_identity("AAPL", "2023-10-06")
    assert routed.archive_commit == PILOT_ARCHIVE_COMMIT
    assert identity["archive_commit"] == PILOT_ARCHIVE_COMMIT
    assert audit.records
    metadata = audit.records[-1]["metadata"]
    assert metadata["archive_commit"] == PILOT_ARCHIVE_COMMIT
    assert metadata["bundle_scope"] == "PILOT"
    assert metadata["input_bundle_identity"] == store.bundle_identity
    assert metadata["case_id"] == "AAPL:2023-10-06"
    assert metadata["packet_json_sha256"] == routed.packet_json_sha256
    assert metadata["route_sha256"] == routed.route_sha256


def test_archived_pilot_is_exactly_four_aapl_cases_and_missingness() -> None:
    store = _pilot_store(_root_from_env())
    assert store.bundle_scope == "PILOT"
    assert set(store._packet_entries) == {("AAPL", session) for session in PILOT_SESSIONS}
    for session in PILOT_SESSIONS:
        packet = store.get_packet("AAPL", session)
        assert packet["packet_status"] == "PILOT_FROZEN"
        assert packet["pilot_only"] is True
        assert packet["formal_eligible"] is False
        assert packet["TEXT"]["status"] == "UNAVAILABLE"
        assert packet["TABLE"]["status"] == "AVAILABLE"
        assert packet["TIME_SERIES"]["status"] == "AVAILABLE"
        assert packet["IMAGE"]["status"] == "UNAVAILABLE"
        assert packet["IMAGE"]["caption_status"] == "NOT_APPLICABLE"
        assert store.get_routed_evidence("AAPL", session, "social") is None


def test_formal_and_pilot_scope_cross_use_fails() -> None:
    pilot_root = _root_from_env()
    with pytest.raises(FrozenEvidenceError, match="scope mismatch"):
        FrozenFinMultiTimeEvidenceStore(pilot_root, bundle_scope="FORMAL")

    formal_value = os.environ.get("ALPHAMAS_M1_INPUT_ROOT")
    if not formal_value or not Path(formal_value).is_dir():
        pytest.skip("formal input archive is not available")
    formal_root = Path(formal_value).resolve()
    packet_sha, bundle_identity = _identity(pilot_root)
    with pytest.raises(FrozenEvidenceError, match="scope mismatch"):
        FrozenFinMultiTimeEvidenceStore(
            formal_root,
            bundle_scope="PILOT",
            expected_packet_manifest_sha256=packet_sha,
            expected_input_bundle_identity=bundle_identity,
        )


def test_pilot_cache_and_checkpoint_identity_isolated() -> None:
    pilot_root = _root_from_env()
    pilot = _pilot_store(pilot_root)

    class Graph:
        selected_analysts = ("market", "social", "news", "fundamentals")
        config = {"finmultitime_evidence_enabled": True, "finmultitime_bundle_scope": "PILOT"}
        finmultitime_evidence_store = pilot

    pilot_payload = TradingAgentsStrategy(Graph())._cache_payload(_cache_kwargs())
    m0_graph = Graph()
    m0_graph.config = {"finmultitime_evidence_enabled": False}
    m0_graph.finmultitime_evidence_store = None
    m0_payload = TradingAgentsStrategy(m0_graph)._cache_payload(_cache_kwargs())
    assert pilot_payload["finmultitime_bundle_scope"] == "PILOT"
    assert cache_key(pilot_payload) != cache_key(m0_payload)

    graph = object.__new__(TradingAgentsGraph)
    graph.selected_analysts = ("market", "social", "news", "fundamentals")
    graph.config = {
        "max_debate_rounds": 3,
        "max_risk_discuss_rounds": 3,
        "finmultitime_evidence_enabled": True,
        "finmultitime_bundle_scope": "PILOT",
        "finmultitime_expected_contract_sha256": pilot.expected_contract_sha256,
        "historical_memory_dir": str(pilot_root.parent / "memory"),
        "memory_mode": "experiment",
        "graph_config_sha256": "a" * 64,
        "data_cache_dir": str(pilot_root.parent / "cache"),
    }
    graph.finmultitime_evidence_store = pilot
    context = RunContext.historical(
        "2023-10-06", experiment_id="M1_pilot_aapl_2023q4_4w_v1", memory_lineage_id="pilot-run"
    )
    with activate_run_context(context):
        signature = graph._run_signature("stock", "AAPL", "2023-10-06")
        memory = graph._historical_memory_config("AAPL", context)
    assert "PILOT" in signature
    assert pilot.bundle_identity in signature
    assert pilot.expected_contract_sha256 in signature
    assert f"finmultitime-pilot-{pilot.bundle_identity[:16]}" in memory["memory_log_path"]
    assert thread_id("AAPL", "2023-10-06", signature) != thread_id(
        "AAPL", "2023-10-06", signature.replace("PILOT", "FORMAL", 1)
    )
