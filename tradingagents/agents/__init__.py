from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "create_msg_delete": (".utils.agent_utils", "create_msg_delete"),
    "AgentState": (".utils.agent_states", "AgentState"),
    "InvestDebateState": (".utils.agent_states", "InvestDebateState"),
    "RiskDebateState": (".utils.agent_states", "RiskDebateState"),
    "FinancialSituationMemory": (".utils.memory", "FinancialSituationMemory"),
    "create_fundamentals_analyst": (".analysts.fundamentals_analyst", "create_fundamentals_analyst"),
    "create_market_analyst": (".analysts.market_analyst", "create_market_analyst"),
    "create_news_analyst": (".analysts.news_analyst", "create_news_analyst"),
    "create_social_media_analyst": (".analysts.social_media_analyst", "create_social_media_analyst"),
    "create_bear_researcher": (".researchers.bear_researcher", "create_bear_researcher"),
    "create_bull_researcher": (".researchers.bull_researcher", "create_bull_researcher"),
    "create_aggressive_debator": (".risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    "create_conservative_debator": (".risk_mgmt.conservative_debator", "create_conservative_debator"),
    "create_neutral_debator": (".risk_mgmt.neutral_debator", "create_neutral_debator"),
    "create_research_manager": (".managers.research_manager", "create_research_manager"),
    "create_portfolio_manager": (".managers.portfolio_manager", "create_portfolio_manager"),
    "create_trader": (".trader.trader", "create_trader"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)


if TYPE_CHECKING:
    from .utils.agent_utils import create_msg_delete
    from .utils.agent_states import AgentState, InvestDebateState, RiskDebateState
    from .utils.memory import FinancialSituationMemory
    from .analysts.fundamentals_analyst import create_fundamentals_analyst
    from .analysts.market_analyst import create_market_analyst
    from .analysts.news_analyst import create_news_analyst
    from .analysts.social_media_analyst import create_social_media_analyst
    from .researchers.bear_researcher import create_bear_researcher
    from .researchers.bull_researcher import create_bull_researcher
    from .risk_mgmt.aggressive_debator import create_aggressive_debator
    from .risk_mgmt.conservative_debator import create_conservative_debator
    from .risk_mgmt.neutral_debator import create_neutral_debator
    from .managers.research_manager import create_research_manager
    from .managers.portfolio_manager import create_portfolio_manager
    from .trader.trader import create_trader
