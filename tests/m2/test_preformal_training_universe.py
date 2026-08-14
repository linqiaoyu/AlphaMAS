from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from scripts.m2 import audit_preformal_training_universe as audit

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/m2"


def _protocol() -> dict:
    return json.loads((DOCS / audit.PROTOCOL_JSON).read_text())


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (DOCS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _cases() -> list[dict[str, str]]:
    return _csv_rows(audit.CASE_PLAN_CSV)


def test_checked_in_artifacts_use_deterministic_serialization() -> None:
    protocol = _protocol()
    symbols = _csv_rows(audit.SYMBOL_AUDIT_CSV)
    cases = _cases()

    for name, expected in audit.artifact_bytes(protocol, symbols, cases).items():
        assert (DOCS / name).read_bytes() == expected


def test_all_decisions_and_maturities_are_preformal() -> None:
    assert all(row["decision_session"] < "2024-01-01" for row in _cases())
    assert all(row["maturity_session"] < "2024-01-01" for row in _cases())


def test_roles_are_disjoint_and_embargoed() -> None:
    protocol = _protocol()
    roles = {
        role: set(details["case_ids"])
        for role, details in protocol["split_definitions"].items()
    }
    for role, ids in roles.items():
        assert not any(ids & other for other_role, other in roles.items() if other_role != role)

    for boundary in protocol["embargo_boundaries"]:
        assert boundary["strict_order_passed"]
        assert (
            boundary["left_last_maturity_session"]
            < boundary["right_first_decision_session"]
        )
        assert boundary["embargo_xnys_sessions"]


def test_budget_tiers_are_strictly_nested_and_within_cap() -> None:
    tiers = _protocol()["budget_tiers"]
    compact = set(tiers["COMPACT"]["case_ids"])
    standard = set(tiers["STANDARD"]["case_ids"])
    maximum = set(tiers["MAXIMUM"]["case_ids"])

    assert compact < standard < maximum
    assert [len(compact), len(standard), len(maximum)] == [72, 80, 96]
    assert len(maximum) <= 104
    for tier in tiers.values():
        assert set(tier["temporal_blocks"]) == set(audit.ROLE_WEEKS)
        assert all(
            block["decision_sessions"] and block["maturity_sessions"]
            for block in tier["temporal_blocks"].values()
        )


def test_budget_tiers_vary_train_only_with_fixed_evaluation() -> None:
    protocol = _protocol()
    cases = _cases()
    by_id = {row["case_id"]: row for row in cases}
    tiers = {
        name: set(details["case_ids"])
        for name, details in protocol["budget_tiers"].items()
    }
    evaluation_by_role = {
        role: {row["case_id"] for row in cases if row["split_role"] == role}
        for role in ("VALIDATION", "FINAL_HOLDOUT", "E2E_PILOT")
    }
    fixed_evaluation = set().union(*evaluation_by_role.values())

    assert protocol["budget_tier_policy"] == {
        "all_tiers_use_all_selected_symbols": True,
        "evaluation_membership_fixed_across_tiers": True,
        "right_aligned_train_windows": True,
        "tier_variable": "TRAIN_CASE_COUNT_ONLY",
    }
    for tier_ids in tiers.values():
        assert fixed_evaluation <= tier_ids
        assert {by_id[case_id]["symbol"] for case_id in tier_ids} == set(
            protocol["selected_symbols"]
        )
        for role_ids in evaluation_by_role.values():
            assert tier_ids & role_ids == role_ids
    for larger, smaller in (("STANDARD", "COMPACT"), ("MAXIMUM", "STANDARD")):
        assert {
            by_id[case_id]["split_role"]
            for case_id in tiers[larger] - tiers[smaller]
        } == {"TRAIN"}


def test_train_windows_and_counts_are_right_aligned() -> None:
    protocol = _protocol()
    cases = _cases()
    expected = {
        "COMPACT": (32, set(audit.TIER_TRAIN_SESSIONS["COMPACT"])),
        "STANDARD": (40, set(audit.TIER_TRAIN_SESSIONS["STANDARD"])),
        "MAXIMUM": (56, set(audit.TIER_TRAIN_SESSIONS["MAXIMUM"])),
    }
    for tier, (count, sessions) in expected.items():
        included = [
            row
            for row in cases
            if row["split_role"] == "TRAIN"
            and row[f"{tier.lower()}_included"] == "true"
        ]
        assert len(included) == count
        assert {row["decision_session"] for row in included} == sessions
        assert max(row["decision_session"] for row in included) == "2023-06-16"
        assert max(row["maturity_session"] for row in included) == "2023-06-26"
        assert protocol["budget_tiers"][tier]["split_counts"]["TRAIN"] == count


def test_fixed_evaluation_identities_are_canonical() -> None:
    protocol = _protocol()
    cases = _cases()
    identity = protocol["fixed_evaluation_identity"]

    validation = [row for row in cases if row["split_role"] == "VALIDATION"]
    holdout = [row for row in cases if row["split_role"] == "FINAL_HOLDOUT"]
    pilot = [row for row in cases if row["split_role"] == "E2E_PILOT"]
    evaluation = validation + holdout + pilot
    assert identity["case_count"] == 40
    assert identity["validation_case_list_sha256"] == audit.sha256_bytes(
        audit.case_list_bytes(validation)
    )
    assert identity["final_holdout_case_list_sha256"] == audit.sha256_bytes(
        audit.case_list_bytes(holdout)
    )
    assert identity["e2e_pilot_case_list_sha256"] == audit.sha256_bytes(
        audit.case_list_bytes(pilot)
    )
    assert identity["combined_case_list_sha256"] == audit.sha256_bytes(
        audit.case_list_bytes(evaluation)
    )


def test_training_caps_and_nonformal_diversity() -> None:
    rows = [row for row in _cases() if row["split_role"] == "TRAIN"]
    counts = Counter(row["symbol"] for row in rows)
    formal_count = sum(row["formal_symbol"] == "true" for row in rows)
    selected = set(_protocol()["selected_symbols"])

    assert max(counts.values()) / len(rows) <= 0.25
    assert formal_count / len(rows) <= 0.50
    assert set(audit.FORMAL_SYMBOLS) <= selected
    assert len(selected - set(audit.FORMAL_SYMBOLS)) >= 3
    assert len(selected) == 8


def test_holdout_identity_and_hash_are_frozen() -> None:
    protocol = _protocol()
    holdout = [row for row in _cases() if row["split_role"] == "FINAL_HOLDOUT"]
    identity = protocol["final_holdout_identity"]

    assert sorted(row["case_id"] for row in holdout) == identity["case_ids"]
    assert audit.sha256_bytes(audit.case_list_bytes(holdout)) == identity[
        "canonical_holdout_case_list_sha256"
    ]
    assert not identity["performance_inspected"]
    assert identity["protected_until_task"] == "M2-15"
    assert identity["canonical_holdout_case_list_sha256"] == (
        "f8e011558f59f56db730702700ef3d419b353b63d4f35278ee78ddb70fedabfe"
    )


def test_selected_symbol_universe_and_erratum_provenance_are_frozen() -> None:
    protocol = _protocol()
    assert set(protocol["selected_symbols"]) == {
        "AAPL",
        "AMZN",
        "JPM",
        "JBSS",
        "EML",
        "AGI",
        "ARR",
        "AEMD",
    }
    assert protocol["schema_version"] == "1.1"
    assert protocol["erratum_task"] == "M2-02B"
    assert audit.STARTING_SHA == "cf189c53a3600030911efcd1ceb5afdad1e06765"
    assert audit.PARENT_PROTOCOL_COMMIT == (
        "cf189c53a3600030911efcd1ceb5afdad1e06765"
    )
    assert audit.ENVIRONMENT_RECOVERY_SHA == (
        "c1bebcbd73d1d30bfd081e4529181f7163450c49"
    )
    assert protocol["starting_sha"] == audit.STARTING_SHA
    assert protocol["parent_protocol_commit"] == audit.PARENT_PROTOCOL_COMMIT
    assert protocol["environment_recovery_sha"] == audit.ENVIRONMENT_RECOVERY_SHA


def test_future_source_integrity_policy_fails_closed_without_reselection() -> None:
    policy = _protocol()["future_source_integrity_policy"]
    assert policy["modality_failure"] == "MARK_MODALITY_UNAVAILABLE_AND_RETAIN_CASE"
    assert policy["automatic_symbol_reselection"] is False
    assert policy["performance_driven_symbol_replacement"] is False
    assert policy["whole_case_correctness_failure"] == (
        "BLOCK_FOR_EXPLICIT_RESEARCH_REVIEW"
    )


def test_price_trajectory_cannot_change_eligibility() -> None:
    first = date(2023, 1, 1)
    sessions = [(first + timedelta(days=offset)).isoformat() for offset in range(66)]
    event = audit.WeeklyCandidate(
        decision_session=sessions[60],
        maturity_session=sessions[65],
        required_history=tuple(sessions[:61]),
        required_outcome=tuple(sessions[61:]),
    )

    def trajectory(reverse: bool) -> list[dict[str, str | float]]:
        result = []
        for index, session in enumerate(sessions):
            base = 200 - index if reverse else 100 + index
            result.append(
                {
                    "Date": session,
                    "Open": base,
                    "High": base + 2,
                    "Low": base - 2,
                    "Close": base + (1 if reverse else -1),
                    "Volume": 1_000 + index,
                }
            )
        return result

    rising = audit.eligible_case_ids_from_price_rows("syn", trajectory(False), [event])
    falling = audit.eligible_case_ids_from_price_rows("syn", trajectory(True), [event])
    assert rising == falling == [f"SYN:{sessions[60]}"]


def test_tie_break_is_case_insensitive_and_hash_ordered() -> None:
    symbols = ["ZZZ", "aaa", "MMM"]
    ordered = sorted(symbols, key=lambda symbol: (audit.tie_break_digest(symbol), symbol))

    assert audit.tie_break_digest("aapl") == audit.tie_break_digest("AAPL")
    assert audit.tie_break_digest("AAPL") == (
        "814740cb5f5d16289d1be2dee50f481b678a773ed2c0e9b85807fc60ad3ad787"
    )
    assert ordered == sorted(ordered, key=lambda symbol: (audit.tie_break_digest(symbol), symbol))
    assert len({audit.tie_break_digest(symbol) for symbol in symbols}) == len(symbols)


def test_case_plan_has_no_performance_fields() -> None:
    cases = _cases()
    symbols = _csv_rows(audit.SYMBOL_AUDIT_CSV)
    assert set(cases[0]) == set(audit.CASE_PLAN_FIELDS)
    assert audit.no_performance_fields(cases)
    assert audit.no_performance_fields(symbols)
    assert {field for field in symbols[0] if "reward" in field} == {
        "pre2024_matured_reward_cases"
    }
    assert not {
        "return",
        "reward",
        "direction",
        "outcome",
        "profit",
        "label",
    } & set(cases[0])


def test_research_validity_and_raw_guard_pass() -> None:
    protocol = _protocol()
    assert protocol["raw_dataset"]["read_only"]
    assert protocol["raw_dataset"]["mutation_guard_passed"]
    assert not any(protocol["research_validity"].values())


def test_aapl_modalities_follow_frozen_m1_contract() -> None:
    rows = [row for row in _cases() if row["symbol"] == "AAPL"]
    assert rows
    assert {row["text_status"] for row in rows} == {"UNAVAILABLE_SOURCE_INTEGRITY"}
    assert {row["image_status"] for row in rows} == {"UNAVAILABLE_M1_FROZEN_POLICY"}
    assert all(row["modality_profile"].endswith("I0") for row in rows)


def test_selected_cases_retain_multiple_modality_profiles() -> None:
    profiles = {row["modality_profile"] for row in _cases()}
    assert len(profiles) >= 3
