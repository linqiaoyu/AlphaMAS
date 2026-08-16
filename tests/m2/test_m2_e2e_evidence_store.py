from __future__ import annotations

from pathlib import Path

import pytest

from tradingagents.evidence.m2_preformal import (
    CORPUS_IDENTITY,
    DECISION_SESSION,
    PACKETS,
    FrozenM2E2EEvidenceStore,
)

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (
    ROOT.parent
    / "AlphaMAS-Experiments/experiments/M2/development/preformal_evidence_v1"
)


@pytest.fixture(scope="module")
def store() -> FrozenM2E2EEvidenceStore:
    if not ARCHIVE.is_dir():
        pytest.fail(f"canonical M2 evidence archive is unavailable: {ARCHIVE}")
    return FrozenM2E2EEvidenceStore(ARCHIVE)


def test_population_is_exactly_the_frozen_eight_cases(
    store: FrozenM2E2EEvidenceStore,
) -> None:
    assert store.bundle_identity == CORPUS_IDENTITY
    assert tuple(PACKETS) == (
        "AAPL", "AEMD", "AGI", "AMZN", "ARR", "EML", "JBSS", "JPM"
    )
    identities = [store.case_identity(symbol, DECISION_SESSION) for symbol in PACKETS]
    assert len(identities) == 8
    assert {item["case_id"] for item in identities} == {
        f"{symbol}:{DECISION_SESSION}" for symbol in PACKETS
    }
    assert all(
        item["packet_identity_sha256"]
        == PACKETS[item["case_id"].split(":")[0]][2]
        for item in identities
    )


def test_only_analyst_local_routes_are_exposed(
    store: FrozenM2E2EEvidenceStore,
) -> None:
    for analyst in ("market", "news", "fundamentals"):
        routed = store.get_routed_evidence("AAPL", DECISION_SESSION, analyst)
        assert routed is not None and routed.text
        assert routed.bundle_scope == "M2_E2E_PILOT"
    assert store.get_routed_evidence("AAPL", DECISION_SESSION, "social") is None


def test_protected_or_unregistered_case_fails_closed(
    store: FrozenM2E2EEvidenceStore,
) -> None:
    with pytest.raises(ValueError, match="no exact"):
        store.get_packet("AAPL", "2023-07-28")
