"""Regression tests for M2 Trader-slot and Portfolio Manager authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.backtesting.models import Action
from tradingagents.backtesting.strategies import strict_action
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.m2.authority_audit import audit_trader_portfolio_authority
from tradingagents.m2.runtime import M2ProductionTraderRuntime

ROOT = Path(__file__).resolve().parents[2]
C09 = (
    ROOT.parent
    / "AlphaMAS-Experiments/experiments/M2/development/global_selection_v1"
    / "selected_global_checkpoint/model.pt"
)
EXPERIMENTS = ROOT.parent / "AlphaMAS-Experiments"
ARCHITECTURE_IDENTITY = "35c5a46616f654ce70b0badc01cd59fb0afd433dcdcd99e5e0b8f2419ec4d153"
HISTORICAL_AUDIT_SHA = "72507326754207b95e565e28cd0fb6b1a1adc4beda4c405806b6d502cf3cacb0"
PROMPT = (
    "**Action**: Buy\n\n**Reasoning**: Frozen prompt reasoning.\n\n"
    "FINAL TRANSACTION PROPOSAL: **BUY**"
)


def _encoder(texts: list[str]) -> np.ndarray:
    rows = []
    for value in texts:
        seed = int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
        row = np.random.default_rng(seed).standard_normal(1024).astype(np.float32)
        rows.append(row / np.linalg.norm(row))
    return np.stack(rows)


def _snapshot(session: str) -> dict:
    return {
        "timestamp": f"{session}T20:00:00+00:00",
        "session": session,
        "symbol": "AAPL",
        "cash": 100_000.0,
        "quantity": 0.0,
        "close_price": 100.0,
        "market_value": 0.0,
        "equity": 100_000.0,
        "current_weight": 0.0,
        "average_entry_price": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "cumulative_cost": 0.0,
        "current_drawdown": 0.0,
        "peak_equity": 100_000.0,
    }


def _forced_override(tmp_path: Path) -> dict:
    runtime = M2ProductionTraderRuntime(
        checkpoint_path=C09,
        state_root=tmp_path / "states",
        encoder=_encoder,
    )
    state = runtime._load("AAPL")
    with torch.no_grad():
        state.model.gate_head.bias.fill_(1000.0)
        state.model.residual_head.bias.copy_(torch.tensor([-1000.0, -1000.0, 1000.0]))
    return runtime.issue(
        {
            "company_of_interest": "AAPL",
            "trade_date": "2023-10-06",
            "investment_plan": "Frozen Research Manager plan.",
            "trader_investment_plan": PROMPT,
            "m2_portfolio_snapshot": _snapshot("2023-10-06"),
            "m2_portfolio_reward_state": {
                "cash": 100_000.0,
                "quantity": 0.0,
                "average_entry_price": 0.0,
                "open_position_commission": 0.0,
            },
            "m2_next_decision_session": None,
        }
    )


def _downstream_state(handoff: str) -> dict:
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2023-10-06",
        "investment_plan": "Frozen Research Manager plan.",
        "trader_investment_plan": handoff,
        "market_report": "Frozen market report.",
        "sentiment_report": "Frozen sentiment report.",
        "news_report": "Frozen news report.",
        "fundamentals_report": "Frozen fundamentals report.",
        "risk_debate_state": {
            "history": "Frozen risk history.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "",
            "latest_speaker": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
    }


def _pm_llm(captured: dict) -> MagicMock:
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt)
        or PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Buy after independent portfolio review.",
            investment_thesis="The Portfolio Manager accepts the portfolio-level risk.",
        )
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def test_a_m2_node_replaces_trader_handoff_and_preserves_prompt(tmp_path: Path) -> None:
    output = _forced_override(tmp_path)
    assert output["prompt_trader_proposal_original"] == PROMPT
    assert output["trader_investment_plan"] != PROMPT
    assert output["m2_rl_action"] == "SELL"
    assert output["m2_override"] is True


def test_b_risk_receives_m2_handoff_not_prompt_only(tmp_path: Path) -> None:
    output = _forced_override(tmp_path)
    captured: dict[str, str] = {}
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or MagicMock(content="Risk response.")
    )
    create_aggressive_debator(llm)(_downstream_state(output["trader_investment_plan"]))
    assert output["trader_investment_plan"] in captured["prompt"]
    assert f"Here is the trader's decision:\n\n{PROMPT}\n\nYour task" not in captured["prompt"]


def test_c_portfolio_manager_receives_m2_handoff(tmp_path: Path) -> None:
    output = _forced_override(tmp_path)
    captured: dict[str, str] = {}
    result = create_portfolio_manager(_pm_llm(captured))(
        _downstream_state(output["trader_investment_plan"])
    )
    assert output["trader_investment_plan"] in captured["prompt"]
    assert "**Rating**: Buy" in result["final_trade_decision"]


def test_d_portfolio_manager_disagreement_is_correctness_valid(tmp_path: Path) -> None:
    output = _forced_override(tmp_path)
    handoff = output["trader_investment_plan"]
    final_decision = "**Rating**: Buy\n\nPortfolio-level decision after risk review."
    result = audit_trader_portfolio_authority(
        prompt_trader_proposal_original=output["prompt_trader_proposal_original"],
        prompt_trader_action=output["prompt_trader_action"],
        m2_rl_action=output["m2_rl_action"],
        m2_override=output["m2_override"],
        trader_investment_plan=handoff,
        m2_trader_handoff_metadata=output["m2_trader_handoff_metadata"],
        risk_trader_input=handoff,
        portfolio_manager_trader_input=handoff,
        final_trade_decision=final_decision,
        executed_action="BUY",
    )
    assert result.status == "PASS"
    assert result.portfolio_manager_disagrees_with_m2 is True
    assert result.silent_prompt_restoration is False


def test_e_actual_silent_prompt_restoration_fails(tmp_path: Path) -> None:
    output = _forced_override(tmp_path)
    result = audit_trader_portfolio_authority(
        prompt_trader_proposal_original=output["prompt_trader_proposal_original"],
        prompt_trader_action=output["prompt_trader_action"],
        m2_rl_action=output["m2_rl_action"],
        m2_override=output["m2_override"],
        trader_investment_plan=output["trader_investment_plan"],
        m2_trader_handoff_metadata=output["m2_trader_handoff_metadata"],
        risk_trader_input=PROMPT,
        portfolio_manager_trader_input=PROMPT,
        final_trade_decision="**Rating**: Buy\n\nInvalid restored-input path.",
        executed_action="BUY",
    )
    assert result.status == "FAIL"
    assert result.silent_prompt_restoration is True
    assert "RISK_DID_NOT_CONSUME_M2_HANDOFF" in result.violations
    assert "PORTFOLIO_MANAGER_DID_NOT_CONSUME_M2_HANDOFF" in result.violations


def test_f_m0_m1_keep_portfolio_manager_execution_semantics() -> None:
    processor = SignalProcessor()
    expected = {
        "Buy": Action.BUY,
        "Overweight": Action.BUY,
        "Hold": Action.HOLD,
        "Underweight": Action.SELL,
        "Sell": Action.SELL,
    }
    for rating, action in expected.items():
        decision = f"**Rating**: {rating}\n\nUnchanged Portfolio Manager output."
        assert strict_action(decision, processor.process_signal(decision)) is action

    setup = GraphSetup(
        MagicMock(), MagicMock(), {"market": MagicMock()}, ConditionalLogic(1, 1)
    )
    m0_m1_graph = setup.setup_graph(selected_analysts=["market"])
    assert ("Trader", "Aggressive Analyst") in m0_m1_graph.edges
    assert ("Portfolio Manager", "__end__") in m0_m1_graph.edges
    assert all("M2 Trader Policy" not in edge for edge in m0_m1_graph.edges)


def test_authority_erratum_preserves_architecture_and_historical_audit() -> None:
    source_erratum = json.loads(
        (ROOT / "docs/m2/m2_trader_portfolio_authority_erratum.json").read_text()
    )
    historical_path = (
        EXPERIMENTS
        / "experiments/M2/development/e2e_pilot_v1/retry3_blocked_audit.json"
    )
    assert source_erratum["classification"] == "AUDIT_SEMANTICS_CORRECTION"
    assert source_erratum["architecture_change"] is False
    assert source_erratum["architecture_identity_sha256"] == ARCHITECTURE_IDENTITY
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == HISTORICAL_AUDIT_SHA


def test_preformal_freeze_binds_corrected_eight_case_audit() -> None:
    freeze = json.loads((ROOT / "docs/m2/m2_preformal_environment_freeze.json").read_text())
    audit_path = (
        EXPERIMENTS
        / "experiments/M2/development/e2e_pilot_v1/retry3_corrected_authority_audit.json"
    )
    audit = json.loads(audit_path.read_text())
    assert freeze["status"] == "FROZEN_FOR_M2_15"
    assert freeze["frozen_identities"]["architecture"] == ARCHITECTURE_IDENTITY
    assert audit["result"]["structural_mismatches"] == 0
    assert audit["result"]["correctness_valid_cases"] == 8
    assert audit["result"]["trajectories_regenerated"] == 0
    assert (
        hashlib.sha256(audit_path.read_bytes()).hexdigest()
        == freeze["audits"]["corrected_authority_audit_sha256"]
    )
