from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.backtesting.cache import cache_key
from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.backtesting.strategies import TradingAgentsStrategy
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evidence.finmultitime import (
    FrozenEvidenceError,
    FrozenFinMultiTimeEvidenceStore,
)
from tradingagents.evidence.prompt import (
    EVIDENCE_WRAPPER_PREFIX,
    EVIDENCE_WRAPPER_SUFFIX,
    build_analyst_local_messages,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT.parent.parent / "AlphaMAS-Experiments/experiments/M1/inputs"


@pytest.fixture(scope="session")
def frozen_input_root() -> Path:
    root = Path(
        os.environ.get("ALPHAMAS_M1_INPUT_ROOT", DEFAULT_INPUT_ROOT)
    ).resolve()
    if not root.is_dir():
        pytest.skip(f"frozen M1 input archive not available: {root}")
    return root


@pytest.fixture
def bundle_copy(tmp_path: Path, frozen_input_root: Path) -> Path:
    target = tmp_path / "inputs"
    shutil.copytree(frozen_input_root, target)
    return target


def _store(root: Path) -> FrozenFinMultiTimeEvidenceStore:
    return FrozenFinMultiTimeEvidenceStore(root)


def _state(symbol: str = "AAPL") -> dict:
    tool_call = {"name": "get_stock_data", "args": {}, "id": "call-1", "type": "tool_call"}
    return {
        "company_of_interest": symbol,
        "trade_date": "2024-01-05",
        "messages": [
            HumanMessage(content=f"Analyze {symbol}"),
            AIMessage(content="", tool_calls=[tool_call]),
            ToolMessage(content="tool result", tool_call_id="call-1"),
        ],
    }


def test_valid_bundle_load_and_exact_case_routes(frozen_input_root: Path) -> None:
    store = _store(frozen_input_root)
    assert store.bundle_identity == "30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121"

    for symbol in ("AAPL", "AMZN", "JPM"):
        packet = store.get_packet(symbol, "2024-01-05")
        assert packet["packet_status"] == "FINAL_FROZEN"
        assert packet["symbol"] == symbol
        assert packet["decision_session"] == "2024-01-05"
        for analyst_key, packet_key in {
            "news": "news_analyst",
            "fundamentals": "fundamentals_analyst",
            "market": "market_analyst",
        }.items():
            routed = store.get_routed_evidence(symbol, "2024-01-05", analyst_key)
            assert routed is not None
            assert routed.text == packet["routed_projections"][packet_key]["text"]
            assert routed.packet_json_sha256 == store.case_identity(
                symbol, "2024-01-05"
            )["packet_json_sha256"]
        assert store.get_routed_evidence(symbol, "2024-01-05", "social") is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("frozen", "input_bundle_frozen=true"),
        ("contract_version", "contract version mismatch"),
        ("contract_sha", "contract SHA mismatch"),
    ],
)
def test_bundle_identity_and_contract_fail_closed(
    bundle_copy: Path, mutation: str, match: str
) -> None:
    manifest_path = bundle_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "frozen":
        manifest["input_bundle_frozen"] = False
    elif mutation == "contract_version":
        manifest["frozen_evidence_contract_version"] = "wrong"
    else:
        manifest["evidence_contract_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FrozenEvidenceError, match=match):
        _store(bundle_copy)


def test_missing_bundle_fails_immediately(tmp_path: Path) -> None:
    with pytest.raises(FrozenEvidenceError, match="input root is missing"):
        FrozenFinMultiTimeEvidenceStore(tmp_path / "missing")


def test_graph_validates_enabled_bundle_before_constructing_llm(tmp_path: Path) -> None:
    config = {
        **DEFAULT_CONFIG,
        "finmultitime_evidence_enabled": True,
        "finmultitime_input_root": tmp_path / "missing",
    }
    with (
        patch("tradingagents.graph.trading_graph.create_llm_client") as create_client,
        pytest.raises(FrozenEvidenceError, match="input root is missing"),
    ):
        TradingAgentsGraph(config=config)
    create_client.assert_not_called()


def test_wrong_packet_manifest_identity_fails_closed(frozen_input_root: Path) -> None:
    with pytest.raises(FrozenEvidenceError, match="packet manifest SHA"):
        FrozenFinMultiTimeEvidenceStore(
            frozen_input_root,
            expected_packet_manifest_sha256="0" * 64,
        )


def test_wrong_bundle_identity_fails_closed(frozen_input_root: Path) -> None:
    with pytest.raises(FrozenEvidenceError, match="input bundle identity"):
        FrozenFinMultiTimeEvidenceStore(
            frozen_input_root,
            expected_input_bundle_identity="0" * 64,
        )


def test_missing_packet_fails_closed(bundle_copy: Path) -> None:
    (bundle_copy / "evidence_packets/AAPL/2024-01-05.json").unlink()
    with pytest.raises(FrozenEvidenceError, match="missing frozen packet"):
        _store(bundle_copy)


def test_wrong_packet_sha_fails_closed(bundle_copy: Path) -> None:
    packet = bundle_copy / "evidence_packets/AAPL/2024-01-05.json"
    packet.write_text(packet.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FrozenEvidenceError, match="frozen packet SHA mismatch"):
        _store(bundle_copy)


def test_packet_sha_is_rechecked_on_access(bundle_copy: Path) -> None:
    store = _store(bundle_copy)
    packet = bundle_copy / "evidence_packets/AAPL/2024-01-05.json"
    original = packet.read_bytes()
    try:
        packet.write_bytes(original + b"\n")
        with pytest.raises(FrozenEvidenceError, match="frozen packet SHA mismatch"):
            store.get_packet("AAPL", "2024-01-05")
    finally:
        packet.write_bytes(original)


def test_exact_lookup_rejects_nearest_or_cross_symbol_fallback(
    frozen_input_root: Path,
) -> None:
    store = _store(frozen_input_root)
    with pytest.raises(FrozenEvidenceError, match="no exact frozen packet"):
        store.get_packet("AAPL", "2024-01-06")
    with pytest.raises(FrozenEvidenceError, match="no exact frozen packet"):
        store.get_packet("MSFT", "2024-01-05")


def test_path_traversal_is_rejected() -> None:
    from tradingagents.evidence.finmultitime import _safe_relative_path

    root = Path("/tmp/frozen-m1-root").resolve()
    with pytest.raises(FrozenEvidenceError, match="escapes"):
        _safe_relative_path(root, "../outside.json", label="test path")


def test_routes_match_frozen_representative_missingness_and_modalities(
    frozen_input_root: Path,
) -> None:
    store = _store(frozen_input_root)
    for symbol, news_status in (("AAPL", "UNAVAILABLE"), ("AMZN", "AVAILABLE"), ("JPM", "UNAVAILABLE")):
        packet = store.get_packet(symbol, "2024-01-05")
        news = store.get_routed_evidence(symbol, "2024-01-05", "news")
        fundamentals = store.get_routed_evidence(symbol, "2024-01-05", "fundamentals")
        market = store.get_routed_evidence(symbol, "2024-01-05", "market")
        assert news and fundamentals and market
        assert f"Status: {news_status}" in news.text
        assert news.text == packet["routed_projections"]["news_analyst"]["text"]
        assert fundamentals.text == packet["routed_projections"]["fundamentals_analyst"]["text"]
        assert market.text == packet["routed_projections"]["market_analyst"]["text"]
        assert store.get_routed_evidence(symbol, "2024-01-05", "social") is None


def test_prompt_disabled_is_identical_and_does_not_call_provider() -> None:
    state = _state()
    messages_before = list(state["messages"])
    actual = build_analyst_local_messages(state, None, "news")
    assert actual == messages_before
    assert state["messages"] == messages_before


def test_prompt_enabled_is_one_analyst_local_data_message_before_history(
    frozen_input_root: Path,
) -> None:
    store = _store(frozen_input_root)
    state = _state("AAPL")
    original = list(state["messages"])
    local = build_analyst_local_messages(state, store, "news")
    assert len(local) == len(original) + 1
    assert local[0].content.startswith(EVIDENCE_WRAPPER_PREFIX)
    assert local[0].content.endswith(EVIDENCE_WRAPPER_SUFFIX)
    assert "Status: UNAVAILABLE" in local[0].content
    assert "raw_model_output" not in local[0].content
    assert local[1:] == original
    assert state["messages"] == original

    prompt = ChatPromptTemplate.from_messages(
        [("system", "UNCHANGED SYSTEM"), MessagesPlaceholder(variable_name="messages")]
    )
    formatted = prompt.format_messages(messages=local)
    assert formatted[0].type == "system"
    assert formatted[1].type == "human"
    assert formatted[2].type == "human"
    assert formatted[3].type == "ai"
    assert formatted[4].type == "tool"
    assert formatted[3].tool_calls[0]["id"] == "call-1"
    assert formatted[4].tool_call_id == "call-1"


def test_social_prompt_has_no_finmultitime_route() -> None:
    class ExplodingProvider:
        def get_routed_evidence(self, *_args):
            raise AssertionError("social analyst must not access FinMultiTime")

    state = _state()
    assert build_analyst_local_messages(state, ExplodingProvider(), "social") == state["messages"]


def _cache_kwargs() -> dict:
    snapshot = PortfolioSnapshot(
        datetime(2024, 1, 5, 21, tzinfo=timezone.utc), "2024-01-05", "AAPL",
        100_000, 0, 100, 0, 100_000, 0, 0, 0, 0, 0, 0, 100_000,
    )
    return {
        "symbol": "AAPL",
        "decision_session": "2024-01-05",
        "decision_time": datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
        "market_history": pd.DataFrame(
            {"Close": [99.0, 100.0]}, index=pd.to_datetime(["2024-01-04", "2024-01-05"])
        ),
        "portfolio_snapshot": snapshot,
        "context": {"experiment_id": "M1", "point_in_time": True},
    }


def test_m0_and_m1_decision_cache_identity_is_separate(frozen_input_root: Path) -> None:
    class Graph:
        selected_analysts = ("market", "social", "news", "fundamentals")
        config = {
            "quick_think_llm": "q", "deep_think_llm": "d", "llm_provider": "mock",
            "max_debate_rounds": 3, "max_risk_discuss_rounds": 3,
            "finmultitime_evidence_enabled": False,
        }
        finmultitime_evidence_store = None

    m0 = TradingAgentsStrategy(Graph())._cache_payload(_cache_kwargs())

    m1_graph = Graph()
    m1_graph.config = {**m1_graph.config, "finmultitime_evidence_enabled": True}
    m1_graph.finmultitime_evidence_store = FrozenFinMultiTimeEvidenceStore(frozen_input_root)
    m1 = TradingAgentsStrategy(m1_graph)._cache_payload(_cache_kwargs())
    assert cache_key(m0) != cache_key(m1)
    assert m1["finmultitime_packet_json_sha256"] == (
        "fa645f18040da61765076d3531a8a1bb1b6fdb80f5dd0a009241ddb04e296049"
    )
    assert m1["finmultitime_input_bundle_identity"] == m1_graph.finmultitime_evidence_store.bundle_identity


def test_packet_identity_changes_cache_identity_without_mutating_archive(
    frozen_input_root: Path,
) -> None:
    class Graph:
        selected_analysts = ("market",)
        config = {"finmultitime_evidence_enabled": True}
        finmultitime_evidence_store = FrozenFinMultiTimeEvidenceStore(frozen_input_root)

    strategy = TradingAgentsStrategy(Graph())
    base = strategy._cache_payload(_cache_kwargs())
    changed = {**base, "finmultitime_packet_json_sha256": "0" * 64}
    assert cache_key(base) != cache_key(changed)
    assert base["finmultitime_packet_json_sha256"] != changed["finmultitime_packet_json_sha256"]


def test_frozen_packet_and_route_identities_are_stable(frozen_input_root: Path) -> None:
    store_a = FrozenFinMultiTimeEvidenceStore(frozen_input_root)
    store_b = FrozenFinMultiTimeEvidenceStore(frozen_input_root)
    assert store_a.case_identity("AMZN", "2024-01-05") == store_b.case_identity("AMZN", "2024-01-05")
    route = store_a.get_routed_evidence("AMZN", "2024-01-05", "market")
    assert route is not None
    assert route.route_sha256 == hashlib.sha256(route.text.encode()).hexdigest()
