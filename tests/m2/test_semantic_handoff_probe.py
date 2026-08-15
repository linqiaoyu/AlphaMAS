from __future__ import annotations

import copy
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.m2.run_semantic_handoff_probe import (
    M2_07_HARD_CEILING_CNY,
    PROBE_POSITIONS,
    SemanticUsageRecorder,
    normalize_action,
    project_tiers,
    request_cost_cny,
    select_probe_cases,
    select_tier,
    validate_actor_visible_state,
)

EXPERIMENTS = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-Experiments")
CORPUS = EXPERIMENTS / "experiments/M2/development/preformal_evidence_v1"


def actor_state() -> dict:
    import hashlib

    plan = "**Recommendation**: Hold"
    proposal = (
        "**Action**: Hold\n\n**Reasoning**: Balanced.\n\n"
        "FINAL TRANSACTION PROPOSAL: **HOLD**"
    )
    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()
    return {
        "schema_version": "M2-SEMANTIC-HANDOFF-v1",
        "case_id": "AAPL:2023-05-26",
        "symbol": "AAPL",
        "decision_session": "2023-05-26",
        "role": "TRAIN",
        "evidence_packet_sha256": "a" * 64,
        "research_manager": {
            "investment_plan": plan,
            "investment_plan_sha256": digest(plan),
        },
        "prompt_trader": {
            "rendered_proposal": proposal,
            "normalized_action": "HOLD",
            "proposal_sha256": digest(proposal),
        },
        "context": {
            "company_of_interest": "AAPL",
            "ticker": "AAPL",
            "instrument_context": "The instrument to analyze is `AAPL`.",
            "decision_session": "2023-05-26",
            "deterministic_temporal_context": {"run_mode": "historical"},
        },
        "provenance": {
            "source_sha": "b" * 40,
            "experiments_sha": "c" * 40,
            "model_id": "deepseek-v4-flash",
            "model_configuration_identity": "d" * 64,
            "run_identity": "e" * 64,
        },
        "reusable_for_M2_08": True,
    }


def test_actor_visible_allowed_fields_and_reuse_contract():
    validate_actor_visible_state(actor_state())


@pytest.mark.parametrize(
    "field",
    ["market_report", "investment_debate_state", "risk_debate_state", "reward", "cash"],
)
def test_actor_visible_forbidden_fields_fail_closed(field):
    value = actor_state()
    value[field] = "leak"
    with pytest.raises(ValueError, match="forbidden Actor-visible"):
        validate_actor_visible_state(value)


def test_actor_and_provenance_are_distinct():
    actor = actor_state()
    assert "analyst_outputs" not in actor
    trace = {"actor_visible": False, "analyst_outputs": {"market": "raw"}}
    assert trace["actor_visible"] is False


def test_action_parser_is_deterministic_and_ambiguous_output_blocks():
    assert normalize_action(actor_state()["prompt_trader"]["rendered_proposal"]) == "HOLD"
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_action("Action: Buy")


def test_probe_selection_is_exact_compact_train_quantiles():
    probes = select_probe_cases(CORPUS)
    assert [row["quantile_position"] for row in probes] == list(PROBE_POSITIONS)
    assert len(probes) == 6
    assert all(row["role"] == "TRAIN" and row["in_compact"] for row in probes)
    assert [
        (row["packet_bytes"], row["case_id"]) for row in probes
    ] == sorted((row["packet_bytes"], row["case_id"]) for row in probes)


def test_cost_formula_uses_cache_split_and_conservative_fallback():
    prices = {
        "input_cache_hit_price": "0.02",
        "input_cache_miss_price": "1",
        "output_price": "2",
    }
    split = {
        "prompt_tokens": 1000,
        "prompt_cache_hit_tokens": 800,
        "prompt_cache_miss_tokens": 200,
        "completion_tokens": 100,
    }
    assert request_cost_cny(split, prices) == Decimal("0.000416")
    fallback = {
        "prompt_tokens": 1000,
        "prompt_cache_hit_tokens": None,
        "prompt_cache_miss_tokens": None,
        "completion_tokens": 100,
    }
    assert request_cost_cny(fallback, prices) == Decimal("0.0012")
    assert fallback["cost_method"] == "ALL_INPUT_AS_CACHE_MISS"


def test_tier_projection_includes_four_cny_reserve():
    projections = project_tiers(Decimal("1"), Decimal("0.1"))
    compact = projections[0]
    assert compact["reserve_cny"] == "4.00"
    assert Decimal(compact["projected_with_reserve_cny"]) == Decimal("11.6")


def test_tier_priority_is_maximum_then_standard_then_compact():
    projections = project_tiers(Decimal("1"), Decimal("0.1"))
    assert select_tier(projections) == ("MAXIMUM", "TARGET_BUDGET")


def test_target_rule_can_select_standard():
    projections = project_tiers(Decimal("1"), Decimal("0.42"))
    assert select_tier(projections) == ("STANDARD", "TARGET_BUDGET")


def test_compact_hard_budget_fallback_and_over_fifty_block():
    fallback = project_tiers(Decimal("1"), Decimal("0.55"))
    assert select_tier(fallback) == ("COMPACT", "HARD_BUDGET_FALLBACK")
    blocked = project_tiers(Decimal("1"), Decimal("0.70"))
    with pytest.raises(RuntimeError, match="exceeds CNY 50"):
        select_tier(blocked)


def test_performance_fields_are_forbidden_from_selector():
    projections = project_tiers(Decimal("1"), Decimal("0.1"))
    projections[0]["reward"] = 1
    with pytest.raises(ValueError, match="performance fields"):
        select_tier(projections)


def test_usage_recorder_does_not_mutate_messages_or_kwargs():
    recorder = SemanticUsageRecorder()
    messages = [[{"role": "user", "content": "unchanged"}]]
    kwargs = {
        "invocation_params": {
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
    }
    before_messages, before_kwargs = copy.deepcopy(messages), copy.deepcopy(kwargs)
    run_id = uuid4()
    recorder.on_chat_model_start({}, messages, run_id=run_id, **kwargs)
    recorder.on_llm_error(RuntimeError("mock"), run_id=run_id)
    assert messages == before_messages
    assert kwargs == before_kwargs
    assert recorder.rows[0]["request_success"] is False
    assert Decimal("5.00") == M2_07_HARD_CEILING_CNY
