import functools

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_report_evidence_guardrail_instruction,
)


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": (
                f"Based on a comprehensive analysis by a team of analysts, here is an investment plan "
                f"tailored for {company_name}. {instrument_context} Use it as a foundation for "
                "evaluating your next trading decision.\n\n"
                f"[InvestmentPlan]\n{investment_plan}\n\n"
                f"[Market]\n{market_research_report}\n\n"
                f"[Sentiment]\n{sentiment_report}\n\n"
                f"[News]\n{news_report}\n\n"
                f"[Fundamentals]\n{fundamentals_report}\n"
            ),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "End with a firm decision and always conclude your response with "
                    "'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. "
                    "Apply lessons from past decisions only as process discipline, not as market evidence. "
                    "Include a short 'Key Evidence' section and tag each concrete claim with an inline "
                    "source label such as [InvestmentPlan], [Market], [Sentiment], [News], "
                    "[Fundamentals], or [Process]. Do not cite [Process] as proof of a market fact, "
                    "price pattern, probability, or company attribute. "
                    f"Here are process reflections from similar situations you traded in and the lessons "
                    f"learned: {past_memory_str}"
                    f"{get_report_evidence_guardrail_instruction(['InvestmentPlan', 'Market', 'Sentiment', 'News', 'Fundamentals', 'Process'])}"
                ),
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
