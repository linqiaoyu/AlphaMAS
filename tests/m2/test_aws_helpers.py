from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AWS_SCRIPTS = ROOT / "scripts/m2/aws"
COST_GUARD = ROOT / "configs/m2/aws_cost_guard.json"


def _script(name: str) -> str:
    return (AWS_SCRIPTS / name).read_text()


def test_aws_cost_guard_freezes_safe_m2_limits() -> None:
    guard = json.loads(COST_GUARD.read_text())
    assert guard == {
        "auto_stop_required": True,
        "instance_count": 1,
        "instance_type": "g5.xlarge",
        "m2_05_incremental_cost_ceiling_usd": 5,
        "m2_05_max_running_minutes": 90,
        "profile": "alphamas",
        "purchasing_model": "on-demand",
        "region": "eu-west-2",
        "whole_project_aws_budget_usd": 50,
    }


def test_helpers_enforce_explicit_profile_region_and_tag_scope() -> None:
    common = _script("common.sh")
    stop = _script("stop.sh")
    assert 'readonly M2_PROFILE="alphamas"' in common
    assert 'readonly M2_REGION="eu-west-2"' in common
    assert '${AWS_PROFILE:?Set AWS_PROFILE=alphamas}' in common
    assert '${AWS_REGION:?Set AWS_REGION=eu-west-2}' in common
    assert "Name=tag:Project,Values=AlphaMAS" in common
    assert "Name=tag:Stage,Values=M2" in common
    assert "Name=tag:ProvisionedByTask,Values=$M2_TASK_TAG" in common
    assert "instance_id=$(single_instance_id)" in stop
    assert "aws ec2 stop-instances --instance-ids \"$instance_id\"" in stop


def test_helpers_require_autostop_and_never_default_to_world_open_ssh() -> None:
    provision = _script("provision.sh")
    user_data = _script("user_data.sh")
    combined = "\n".join(path.read_text() for path in AWS_SCRIPTS.glob("*.sh"))
    assert "--count 1" in provision
    assert "--instance-initiated-shutdown-behavior stop" in provision
    assert "OnBootSec=90min" in user_data
    assert "systemctl enable --now alphamas-autostop.timer" in user_data
    assert "0.0.0.0/0" not in combined


def test_helpers_contain_no_credentials_or_dynamic_resource_ids() -> None:
    combined = "\n".join(path.read_text() for path in AWS_SCRIPTS.glob("*.sh"))
    forbidden = (
        r"AKIA[0-9A-Z]{16}",
        r"ASIA[0-9A-Z]{16}",
        r"AWS_SECRET_ACCESS_KEY=",
        r"AWS_SESSION_TOKEN=",
        r"BEGIN [A-Z ]*PRIVATE KEY",
        r"\b(?:i|vol|sg|subnet|vpc)-[0-9a-f]{8,}\b",
        r"\b\d{12}\b",
    )
    for pattern in forbidden:
        assert re.search(pattern, combined) is None


def test_teardown_is_non_automatic_even_with_confirmation() -> None:
    teardown = _script("teardown.sh")
    assert '"${1:-}" != "--confirm-destroy"' in teardown
    for destructive_call in (
        "terminate-instances",
        "delete-volume",
        "delete-bucket",
        "delete-security-group",
        "delete-role",
    ):
        assert destructive_call not in teardown
