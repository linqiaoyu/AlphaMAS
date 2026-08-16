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


def test_corrected_delayed_credit_method_freeze() -> None:
    contract = json.loads(
        (ROOT / "docs/m2/m2_agentic_rl_method_freeze.json").read_text()
    )
    assert contract["task_id"] == "M2-10A"
    assert contract["superseded_method"] == "M2-PA-CTPPO-v1"
    assert contract["method_id"] == "M2-PA-CTPPO-v2"
    assert contract["tree_contract"] == "M2_TRAIN_COUNTERFACTUAL_TREE-v2"
    assert contract["state_reward_time_decoupled"] is True
    assert contract["bellman_bootstrap"] is False
    assert contract["gamma"] is None
    assert contract["gae"] is False
    assert contract["nonterminal_transitions"] == 8736
    assert contract["after_child_maturity_edges"] == 648
    assert contract["trainable_parameters"] == 20197
    assert contract["fast_parameters_per_symbol"] == 165
    assert contract["validation_performance_used"] is False
    assert contract["final_holdout_used"] is False


def test_v2_source_has_no_bellman_r3_bootstrap() -> None:
    method = (ROOT / "scripts/m2/pa_ctppo.py").read_text()
    assert "GAMMA" not in method
    assert "q_values" not in method
    assert "rewards[node] +" not in method
    assert "exact_local_credit_policy_evaluation" in method


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
