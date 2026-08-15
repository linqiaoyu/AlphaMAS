from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AWS_SCRIPTS = ROOT / "scripts/m2/aws"
COST_GUARD = ROOT / "configs/m2/aws_cost_guard.json"


def _script(name: str) -> str:
    return (AWS_SCRIPTS / name).read_text()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _lineage_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "--initial-branch=main")
    _git(remote, "config", "user.name", "M2 Test")
    _git(remote, "config", "user.email", "m2-test@example.invalid")
    tracked = remote / "tracked.txt"
    tracked.write_text("commit-a\n")
    _git(remote, "add", "tracked.txt")
    _git(remote, "commit", "-m", "commit A")
    commit_a = _git(remote, "rev-parse", "HEAD")
    tracked.write_text("commit-b\n")
    _git(remote, "commit", "-am", "commit B")
    commit_b = _git(remote, "rev-parse", "HEAD")
    return remote, commit_a, commit_b


def _sync_repository(repository: Path, remote: Path, sha: str) -> subprocess.CompletedProcess[str]:
    command = (
        f'source "{AWS_SCRIPTS / "sync_repos.sh"}"; '
        'sync_repository "$1" "$2" "$3"'
    )
    return subprocess.run(
        ["bash", "-c", command, "lineage-test", str(repository), str(remote), sha],
        capture_output=True,
        text=True,
    )


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


def test_sync_repos_requires_explicit_validated_exact_lineage() -> None:
    sync = _script("sync_repos.sh")
    assert "77058329fdc3e7cc089fcc73680a206f5cd36876" not in sync
    assert "6c9e18d7d0ea1a2b91fd4ac5eefe829160a15cac" not in sync
    assert '${M2_SOURCE_SHA:?Set M2_SOURCE_SHA' in sync
    assert '${M2_EXPERIMENTS_SHA:?Set M2_EXPERIMENTS_SHA' in sync
    assert "^[0-9a-fA-F]{40}$" in sync
    assert 'cat-file -e "${requested_sha}^{commit}"' in sync
    assert 'checkout --detach "$requested_sha"' in sync
    assert "rev-parse HEAD" in sync
    assert "origin/baseline-m2" not in sync
    assert "origin/main" not in sync
    assert "reset --hard" not in sync
    assert "clean -fd" not in sync


def test_sync_repos_missing_or_invalid_sha_fails_before_git(tmp_path: Path) -> None:
    script = AWS_SCRIPTS / "sync_repos.sh"
    valid = "a" * 40
    cases = (
        {},
        {"M2_EXPERIMENTS_SHA": valid},
        {"M2_SOURCE_SHA": valid},
        {"M2_SOURCE_SHA": "abc", "M2_EXPERIMENTS_SHA": valid},
        {"M2_SOURCE_SHA": "latest", "M2_EXPERIMENTS_SHA": valid},
        {"M2_SOURCE_SHA": valid, "M2_EXPERIMENTS_SHA": "baseline-m2"},
    )
    for supplied in cases:
        environment = {"PATH": "/usr/bin:/bin", "M2_ROOT": str(tmp_path / "root")}
        environment.update(supplied)
        result = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, env=environment
        )
        assert result.returncode != 0
        assert not (tmp_path / "root").exists()


def test_sync_repository_checks_out_exact_commits_and_ignores_branch_tip(
    tmp_path: Path,
) -> None:
    remote, commit_a, commit_b = _lineage_fixture(tmp_path)
    checkout = tmp_path / "checkout"

    assert _sync_repository(checkout, remote, commit_a).returncode == 0
    assert _git(checkout, "rev-parse", "HEAD") == commit_a
    assert _sync_repository(checkout, remote, commit_b.upper()).returncode == 0
    assert _git(checkout, "rev-parse", "HEAD") == commit_b

    (remote / "tracked.txt").write_text("branch-tip-moved\n")
    _git(remote, "commit", "-am", "move branch tip")
    assert _sync_repository(checkout, remote, commit_a).returncode == 0
    assert _git(checkout, "rev-parse", "HEAD") == commit_a


def test_sync_repository_rejects_nonexistent_commit_and_dirty_checkout(
    tmp_path: Path,
) -> None:
    remote, commit_a, commit_b = _lineage_fixture(tmp_path)
    checkout = tmp_path / "checkout"

    missing = _sync_repository(checkout, remote, "f" * 40)
    assert missing.returncode != 0
    assert _sync_repository(checkout, remote, commit_a).returncode == 0
    (checkout / "tracked.txt").write_text("unsaved task output\n")

    dirty = _sync_repository(checkout, remote, commit_b)
    assert dirty.returncode != 0
    assert "Refusing to change dirty repository" in dirty.stderr
    assert (checkout / "tracked.txt").read_text() == "unsaved task output\n"

    untracked_checkout = tmp_path / "untracked-checkout"
    assert _sync_repository(untracked_checkout, remote, commit_a).returncode == 0
    (untracked_checkout / "task-output.txt").write_text("preserve me\n")
    untracked = _sync_repository(untracked_checkout, remote, commit_b)
    assert untracked.returncode != 0
    assert "Refusing to change dirty repository" in untracked.stderr
    assert (untracked_checkout / "task-output.txt").read_text() == "preserve me\n"
