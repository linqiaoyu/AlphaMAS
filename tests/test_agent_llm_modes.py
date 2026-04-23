import unittest

from tradingagents.llm_mode import PAPER_BASELINE_AGENT_LLM_MODES, resolve_agent_llm_modes


class AgentLlmModeTests(unittest.TestCase):
    def test_resolve_defaults_to_paper_baseline(self):
        resolved = resolve_agent_llm_modes()
        self.assertEqual(resolved, PAPER_BASELINE_AGENT_LLM_MODES)

    def test_resolve_applies_valid_overrides(self):
        resolved = resolve_agent_llm_modes(
            {
                "market_analyst": "quick",
                "trader": "Quick",
                "portfolio_manager": "DEEP",
            }
        )
        self.assertEqual(resolved["market_analyst"], "quick")
        self.assertEqual(resolved["trader"], "quick")
        self.assertEqual(resolved["portfolio_manager"], "deep")

    def test_resolve_ignores_invalid_roles_and_modes(self):
        resolved = resolve_agent_llm_modes(
            {
                "unknown_role": "quick",
                "news_analyst": "unsupported",
            }
        )
        self.assertEqual(
            resolved["news_analyst"],
            PAPER_BASELINE_AGENT_LLM_MODES["news_analyst"],
        )
        self.assertNotIn("unknown_role", resolved)


if __name__ == "__main__":
    unittest.main()
