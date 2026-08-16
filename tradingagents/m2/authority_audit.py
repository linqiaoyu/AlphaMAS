"""Deterministic audit of M2 Trader authority versus portfolio authority.

M2 owns the Trader-slot handoff.  The unchanged Portfolio Manager owns the
final policy decision, and the backtesting adapter deterministically maps that
decision to the execution action.  A Portfolio Manager disagreement with M2 is
therefore valid; only handoff, provenance, downstream-consumption, or execution
corruption is a structural authority failure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from scripts.m2.semantic_state_representation import text_sha256
from tradingagents.backtesting.models import Action
from tradingagents.backtesting.strategies import strict_action
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.m2.runtime import (
    parse_authoritative_m2_action,
    parse_prompt_trader_action,
)

VALID_ACTIONS = frozenset({"BUY", "HOLD", "SELL"})
PROVENANCE_BEGIN = "--- BEGIN NON-AUTHORITATIVE PROMPT TRADER PROVENANCE ---"
PROVENANCE_END = "--- END NON-AUTHORITATIVE PROMPT TRADER PROVENANCE ---"


@dataclass(frozen=True)
class AuthorityAuditResult:
    """Machine-readable outcome of one corrected authority audit."""

    status: str
    structural_mismatch_count: int
    violations: tuple[str, ...]
    prompt_action: str
    m2_action: str
    normalized_portfolio_manager_action: str | None
    executed_action: str
    m2_override: bool
    portfolio_manager_disagrees_with_m2: bool | None
    silent_prompt_restoration: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _provenance_body(handoff: str) -> str | None:
    if handoff.count(PROVENANCE_BEGIN) != 1 or handoff.count(PROVENANCE_END) != 1:
        return None
    prefix, remainder = handoff.split(PROVENANCE_BEGIN, 1)
    body, suffix = remainder.split(PROVENANCE_END, 1)
    if not prefix or suffix:
        return None
    return body.removeprefix("\n").removesuffix("\n")


def audit_trader_portfolio_authority(
    *,
    prompt_trader_proposal_original: str,
    prompt_trader_action: str,
    m2_rl_action: str,
    m2_override: bool,
    trader_investment_plan: str,
    m2_trader_handoff_metadata: Mapping[str, Any],
    risk_trader_input: str,
    portfolio_manager_trader_input: str,
    final_trade_decision: str,
    executed_action: str | Action,
) -> AuthorityAuditResult:
    """Audit one M2 decision without requiring Portfolio Manager agreement.

    ``risk_trader_input`` and ``portfolio_manager_trader_input`` are captured
    values of ``state["trader_investment_plan"]`` at those boundaries.  They
    must equal the M2 handoff byte-for-byte.  ``executed_action`` must instead
    equal the normalized Portfolio Manager decision.
    """

    violations: list[str] = []
    prompt_action = str(prompt_trader_action).upper()
    m2_action = str(m2_rl_action).upper()
    execution_action = (
        executed_action.value if isinstance(executed_action, Action) else str(executed_action)
    ).upper()

    if prompt_action not in VALID_ACTIONS:
        violations.append("PROMPT_ACTION_INVALID")
    try:
        parsed_prompt = parse_prompt_trader_action(prompt_trader_proposal_original)
    except ValueError:
        parsed_prompt = None
        violations.append("PROMPT_PROPOSAL_TERMINAL_ACTION_INVALID")
    if parsed_prompt is not None and parsed_prompt != prompt_action:
        violations.append("PROMPT_ACTION_METADATA_MISMATCH")

    if m2_action not in VALID_ACTIONS:
        violations.append("M2_ACTION_INVALID")
    try:
        parsed_m2 = parse_authoritative_m2_action(trader_investment_plan)
    except ValueError:
        parsed_m2 = None
        violations.append("M2_HANDOFF_AUTHORITY_FIELD_INVALID")
    if parsed_m2 is not None and parsed_m2 != m2_action:
        violations.append("M2_HANDOFF_ACTION_MISMATCH")

    if bool(m2_override) != (m2_action != prompt_action):
        violations.append("M2_OVERRIDE_FLAG_FALSE")

    metadata = dict(m2_trader_handoff_metadata)
    expected_metadata = {
        "authoritative_action": m2_action,
        "action_source": "FROZEN_M2_ACTOR",
        "prompt_action": prompt_action,
        "override": bool(m2_override),
        "prompt_trader_proposal_sha256": text_sha256(prompt_trader_proposal_original),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            violations.append(f"M2_HANDOFF_METADATA_{field.upper()}_MISMATCH")

    provenance = _provenance_body(trader_investment_plan)
    if provenance is None:
        violations.append("PROMPT_PROVENANCE_BOUNDARY_INVALID")
    elif provenance != prompt_trader_proposal_original:
        violations.append("PROMPT_PROPOSAL_BYTES_NOT_PRESERVED")

    if trader_investment_plan == prompt_trader_proposal_original:
        violations.append("POST_M2_TRADER_HANDOFF_RESTORED_TO_PROMPT")
    if risk_trader_input != trader_investment_plan:
        violations.append("RISK_DID_NOT_CONSUME_M2_HANDOFF")
    if portfolio_manager_trader_input != trader_investment_plan:
        violations.append("PORTFOLIO_MANAGER_DID_NOT_CONSUME_M2_HANDOFF")

    normalized_pm_action: str | None
    try:
        rating = SignalProcessor().process_signal(final_trade_decision)
        normalized_pm_action = strict_action(final_trade_decision, rating).value
    except ValueError:
        normalized_pm_action = None
        violations.append("PORTFOLIO_MANAGER_DECISION_NOT_DETERMINISTIC")
    if normalized_pm_action is not None and execution_action != normalized_pm_action:
        violations.append("EXECUTION_DID_NOT_FOLLOW_PORTFOLIO_MANAGER")

    mismatch_count = len(violations)
    return AuthorityAuditResult(
        status="PASS" if mismatch_count == 0 else "FAIL",
        structural_mismatch_count=mismatch_count,
        violations=tuple(violations),
        prompt_action=prompt_action,
        m2_action=m2_action,
        normalized_portfolio_manager_action=normalized_pm_action,
        executed_action=execution_action,
        m2_override=bool(m2_override),
        portfolio_manager_disagrees_with_m2=(
            normalized_pm_action != m2_action if normalized_pm_action is not None else None
        ),
        silent_prompt_restoration=any(
            violation
            in {
                "POST_M2_TRADER_HANDOFF_RESTORED_TO_PROMPT",
                "RISK_DID_NOT_CONSUME_M2_HANDOFF",
                "PORTFOLIO_MANAGER_DID_NOT_CONSUME_M2_HANDOFF",
            }
            for violation in violations
        ),
    )
