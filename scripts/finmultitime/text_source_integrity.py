"""Frozen, fail-closed TEXT source-integrity policy for formal M1 inputs.

The semantic audit is recorded in ``docs/m1/m1_text_source_integrity_audit.json``.
This module only applies the resulting symbol-level policy.  It never fetches
web content, repairs article bodies, or selects replacement articles.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_RELATIVE_PATH = "docs/m1/m1_text_source_integrity_audit.json"
AUDIT_PATH = REPO_ROOT / AUDIT_RELATIVE_PATH
AUDIT_STATUSES = {"VERIFIED_MATCH", "VERIFIED_MISMATCH", "UNVERIFIABLE"}
FORMAL_TEXT_POLICIES = {"AVAILABLE", "UNAVAILABLE"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_integrity_policy(
    contract: dict[str, Any], *, audit_path: Path = AUDIT_PATH
) -> dict[str, dict[str, Any]]:
    """Validate and return the contract's frozen symbol-level policy.

    The contract owns the active policy, while the audit artifact provides the
    evidence and is hash-pinned by the contract.  Any disagreement fails
    closed before raw data is selected.
    """

    identity = contract.get("text_source_integrity")
    if not isinstance(identity, dict):
        raise ValueError("frozen TEXT source-integrity policy is missing")
    expected_sha = identity.get("audit_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("frozen TEXT source-integrity audit SHA is missing")
    if not audit_path.is_file() or sha256_file(audit_path) != expected_sha:
        raise ValueError("TEXT source-integrity audit SHA does not match frozen contract")

    audit = read_json(audit_path)
    policy = identity.get("policy_by_symbol")
    audited_policy = audit.get("policy_by_symbol")
    if not isinstance(policy, dict) or not isinstance(audited_policy, dict):
        raise ValueError("TEXT source-integrity policy is malformed")
    if policy != audited_policy:
        raise ValueError("contract and TEXT source-integrity audit policy disagree")

    result: dict[str, dict[str, Any]] = {}
    for symbol, item in sorted(policy.items()):
        if not isinstance(item, dict):
            raise ValueError(f"TEXT source-integrity policy is malformed for {symbol}")
        audit_status = item.get("audit_status")
        formal_policy = item.get("formal_text_policy")
        if audit_status not in AUDIT_STATUSES:
            raise ValueError(f"invalid TEXT source-integrity audit status for {symbol}")
        if formal_policy not in FORMAL_TEXT_POLICIES:
            raise ValueError(f"invalid TEXT formal policy for {symbol}")
        result[symbol] = item
    return result


def unavailable_text_section(
    symbol: str,
    source_member: str | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Return the canonical preprocessed UNAVAILABLE TEXT section."""

    return {
        "status": "UNAVAILABLE",
        "reason": policy["reason"],
        "selected_records": [],
        "selected_record_hashes": [],
        "dedup_removed_count": 0,
        "same_day_ambiguous_rejected": [],
        "same_day_ambiguous_rejected_count": 0,
        "latest_safe_date": None,
        "source_member": source_member,
        "integrity_status": policy["audit_status"],
        "integrity_policy": policy["formal_text_policy"],
        "integrity_audit_record_count": policy.get("formal_use_record_count", 0),
    }
