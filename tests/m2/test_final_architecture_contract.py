from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tradingagents.backtesting.config import (
    FORMAL_M1_CONTRACT,
    FORMAL_M2_CONTRACT,
    validate_formal_m2_config,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("relative", "identity_field"),
    [
        ("docs/m2/m2_final_architecture_freeze.json", "architecture_identity_sha256"),
        ("docs/m2/m2_ablation_preregistration.json", "preregistration_identity_sha256"),
    ],
)
def test_freeze_identity_is_recomputable(relative: str, identity_field: str) -> None:
    artifact = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    actual = hashlib.sha256(
        json.dumps(
            artifact["identity_payload"], separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    assert actual == artifact[identity_field]


def test_ablation_preregistration_removes_exactly_one_mechanism() -> None:
    artifact = json.loads(
        (ROOT / "docs/m2/m2_ablation_preregistration.json").read_text(encoding="utf-8")
    )
    assert artifact["status"] == "PREREGISTERED_BEFORE_E2E_PILOT"
    assert artifact["pilot_performance_used"] is False
    assert artifact["formal_performance_used"] is False
    assert artifact["branches_modified"] is False
    a1 = artifact["ablations"]["A1_NO_ONLINE_ADAPTATION"]
    a2 = artifact["ablations"]["A2_NO_GLOBAL_PRETRAINING"]
    assert a1["global_parameter_sha256"] == FORMAL_M2_CONTRACT[
        "m2_checkpoint_parameter_identity"
    ]
    assert a1["online_parameter_updates"] is False
    assert a2["initial_parameter_sha256"] == (
        "60a0fec7b69ef2d0576a9c0894be09c377d573585db827c162279fb27483303e"
    )
    assert a2["retuned"] is False
    assert a2["online_learning_rate"] == 1e-3
    assert a2["online_update_epochs"] == 2


def test_formal_m2_changes_only_the_registered_trader_policy_fields() -> None:
    config = json.loads((ROOT / "configs/backtest_m2_2024h1.json").read_text())
    validate_formal_m2_config(config)
    assert config == FORMAL_M2_CONTRACT
    unchanged = set(FORMAL_M1_CONTRACT) - {"experiment_id_template"}
    assert all(config[key] == FORMAL_M1_CONTRACT[key] for key in unchanged)
    assert config["m2_trader_enabled"] is True
    assert config["m2_variant"] == "FULL_M2"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("m2_variant", "A1_NO_ONLINE_ADAPTATION"),
        ("m2_online_learning_rate", 0.0001),
        ("m2_online_update_epochs", 1),
        ("m2_representation_identity", "0" * 64),
        ("temperature", 0.1),
        ("max_risk_discuss_rounds", 2),
    ],
)
def test_formal_m2_contract_rejects_research_drift(field: str, value: object) -> None:
    config = dict(FORMAL_M2_CONTRACT)
    config[field] = value
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_formal_m2_config(config)
