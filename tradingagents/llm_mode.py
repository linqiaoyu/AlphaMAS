"""Agent-level LLM mode routing for TradingAgents."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

VALID_LLM_MODES = {"quick", "deep"}

# Paper-aligned default: analysis/decision agents use deep-thinking models.
PAPER_BASELINE_AGENT_LLM_MODES: Dict[str, str] = {
    "market_analyst": "deep",
    "social_analyst": "deep",
    "news_analyst": "deep",
    "fundamentals_analyst": "deep",
    "bull_researcher": "deep",
    "bear_researcher": "deep",
    "research_manager": "deep",
    "trader": "deep",
    "aggressive_risk_analyst": "deep",
    "neutral_risk_analyst": "deep",
    "conservative_risk_analyst": "deep",
    "portfolio_manager": "deep",
}

AGENT_ROLE_LABELS: Tuple[Tuple[str, str], ...] = (
    ("Market Analyst", "market_analyst"),
    ("Social Analyst", "social_analyst"),
    ("News Analyst", "news_analyst"),
    ("Fundamentals Analyst", "fundamentals_analyst"),
    ("Bull Researcher", "bull_researcher"),
    ("Bear Researcher", "bear_researcher"),
    ("Research Manager", "research_manager"),
    ("Trader", "trader"),
    ("Aggressive Risk Analyst", "aggressive_risk_analyst"),
    ("Neutral Risk Analyst", "neutral_risk_analyst"),
    ("Conservative Risk Analyst", "conservative_risk_analyst"),
    ("Portfolio Manager", "portfolio_manager"),
)


def resolve_agent_llm_modes(overrides: Dict[str, str] | None = None) -> Dict[str, str]:
    """Merge user overrides into paper-aligned defaults."""
    resolved = PAPER_BASELINE_AGENT_LLM_MODES.copy()
    if not overrides:
        return resolved

    for role, mode in overrides.items():
        if role not in resolved:
            continue
        normalized = str(mode).strip().lower()
        if normalized in VALID_LLM_MODES:
            resolved[role] = normalized
    return resolved


def apply_mode_to_roles(
    roles: Iterable[str],
    mode: str,
    base: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """Assign one mode to multiple roles."""
    normalized = str(mode).strip().lower()
    if normalized not in VALID_LLM_MODES:
        raise ValueError(f"Invalid LLM mode '{mode}'. Must be one of: {sorted(VALID_LLM_MODES)}")

    result = resolve_agent_llm_modes(base)
    for role in roles:
        if role in result:
            result[role] = normalized
    return result
