import inspect
import unittest

from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst
from tradingagents.agents.utils.agent_utils import get_news_data_guardrail_instruction


class NewsGuardrailTests(unittest.TestCase):
    def test_guardrail_instruction_mentions_structured_tags(self):
        instruction = get_news_data_guardrail_instruction()
        self.assertIn("[NEWS_UNAVAILABLE]", instruction)
        self.assertIn("[GLOBAL_NEWS_EMPTY]", instruction)
        self.assertIn("must not infer", instruction)
        self.assertIn("no positive catalyst", instruction)

    def test_social_analyst_uses_news_guardrail_instruction(self):
        source = inspect.getsource(create_social_media_analyst)
        self.assertIn("get_news_data_guardrail_instruction()", source)

    def test_news_analyst_uses_news_guardrail_instruction(self):
        source = inspect.getsource(create_news_analyst)
        self.assertIn("get_news_data_guardrail_instruction()", source)


if __name__ == "__main__":
    unittest.main()
