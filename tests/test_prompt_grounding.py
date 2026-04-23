import inspect
import unittest

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_utils import (
    get_report_evidence_guardrail_instruction,
    get_report_only_instruction,
    get_tool_evidence_guardrail_instruction,
)


class PromptGroundingTests(unittest.TestCase):
    def test_report_only_instruction_blocks_trade_language(self):
        instruction = get_report_only_instruction()
        self.assertIn("Do not recommend BUY, HOLD, or SELL", instruction)
        self.assertIn("FINAL TRANSACTION PROPOSAL", instruction)

    def test_tool_evidence_instruction_blocks_external_knowledge(self):
        instruction = get_tool_evidence_guardrail_instruction()
        self.assertIn("Use only facts explicitly returned by the tools", instruction)
        self.assertIn("Do not add external background knowledge", instruction)
        self.assertIn("Do not infer the cause or purpose of R&D", instruction)
        self.assertIn("report the numeric change and timing only", instruction)

    def test_report_evidence_instruction_requires_source_tags(self):
        instruction = get_report_evidence_guardrail_instruction(["Market", "Debate", "Process"])
        self.assertIn("[Market]", instruction)
        self.assertIn("[Debate]", instruction)
        self.assertIn("[Process]", instruction)
        self.assertIn("outside knowledge", instruction)
        self.assertIn("no positive catalyst", instruction)
        self.assertIn("absence of insider buying is notable", instruction)
        self.assertIn("must not be used as new market facts", instruction)
        self.assertIn("tag it as [Process]", instruction)

    def test_analysts_use_report_only_and_tool_guardrails(self):
        for factory in (
            create_market_analyst,
            create_social_media_analyst,
            create_news_analyst,
            create_fundamentals_analyst,
        ):
            source = inspect.getsource(factory)
            self.assertIn("get_tool_evidence_guardrail_instruction()", source)
            self.assertIn("get_report_only_instruction()", source)
            self.assertNotIn(
                "If you or any other assistant has the FINAL TRANSACTION PROPOSAL",
                source,
            )

    def test_downstream_roles_use_report_grounding_guardrail(self):
        for factory in (
            create_bull_researcher,
            create_bear_researcher,
            create_research_manager,
            create_trader,
            create_aggressive_debator,
            create_conservative_debator,
            create_neutral_debator,
            create_portfolio_manager,
        ):
            source = inspect.getsource(factory)
            self.assertIn("get_report_evidence_guardrail_instruction(", source)


if __name__ == "__main__":
    unittest.main()
