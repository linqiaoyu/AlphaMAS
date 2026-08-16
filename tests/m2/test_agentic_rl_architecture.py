from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_inheritance_and_method_boundaries() -> None:
    representation = json.loads(
        (ROOT / "docs/m2/m2_semantic_state_representation_freeze.json").read_text()
    )
    reward = json.loads((ROOT / "docs/m2/m2_reward_selection_freeze.json").read_text())
    assert representation["representation_identity_sha256"] == (
        "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe"
    )
    assert representation["actor_observation_dimension"] == 3080
    assert reward["selected_reward_id"] == "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"
    assert reward["market_snapshot_identity_sha256"] == (
        "3afc723888666e0ca3a12219a57dc577d6cf3953717da731b774634f4aff1445"
    )


def test_no_production_graph_wiring_in_m2_10() -> None:
    setup = (ROOT / "tradingagents/graph/setup.py").read_text()
    assert "M2-PA-CTPPO-v1" not in setup
    assert "PromptAnchoredActorCritic" not in setup


def test_frozen_runtime_components_do_not_import_rl_method() -> None:
    for relative in (
        "scripts/m2/reward_simulator.py",
        "tradingagents/agents/risk_mgmt_agent.py",
        "tradingagents/agents/managers.py",
    ):
        path = ROOT / relative
        if path.exists():
            text = path.read_text()
            assert "pa_ctppo" not in text
            assert "rl_environment" not in text
