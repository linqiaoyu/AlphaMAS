from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/m2/m2_aws_environment_contract.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT.read_text())


def test_aws_environment_contract_freezes_verified_infrastructure() -> None:
    contract = _contract()
    assert contract["region"] == "eu-west-2"
    assert contract["availability_zone"] == "eu-west-2a"
    assert contract["instance_type"] == "g5.xlarge"
    assert contract["gpu_name"] == "NVIDIA A10G"
    assert contract["gpu_count"] == 1
    assert contract["auto_stop_minutes"] <= 90
    assert contract["security_group_inbound_rule_count"] == 0
    assert contract["s3_public_access_block"] is True
    assert contract["persistent_ebs_encrypted"] is True
    assert contract["persistent_ebs_gib"] == 64


def test_aws_environment_contract_freezes_exact_research_lineage() -> None:
    contract = _contract()
    assert contract["verified_runtime_source_sha"] == (
        "1594ef850999ecb29f150cb4a8eedfe20c6ad7a6"
    )
    assert contract["verified_experiments_sha"] == (
        "6c9e18d7d0ea1a2b91fd4ac5eefe829160a15cac"
    )
    assert contract["evidence_archive_commit"] == (
        "3a8e63a9abe89a80787b68aabb9b3523453966fc"
    )
    assert contract["selected_reward"] == "R3_HOLD_RELATIVE_DRAWDOWN_UTILITY"


def test_aws_environment_contract_records_complete_linux_and_cuda_pass() -> None:
    contract = _contract()
    assert contract["cross_platform_results"] == {
        "backtesting_passed": 205,
        "m1_formal_passed": 21,
        "m1_runtime_passed": 23,
        "m2_passed": 86,
        "pilot_archive_target_passed": 1,
        "pilot_archive_target_skipped": 0,
        "pit_passed": 19,
        "reward_simulator_passed": 44,
        "reward_study_passed": 11,
        "synthetic_semantic_equivalence": "PASS",
    }
    assert contract["torch_cuda_smoke_passed"] is True
    assert contract["final_training_torch_frozen"] is False
    assert contract["running_alphamas_gpu_instances_at_freeze"] == 0
    assert contract["deepseek_calls"] == 0
    assert contract["qwen_calls"] == 0
