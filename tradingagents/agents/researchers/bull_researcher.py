

from tradingagents.agents.utils.agent_utils import get_report_evidence_guardrail_instruction


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case using only the favorable signals that appear in the supplied analyst reports and debate history. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Positive Indicators: Highlight concrete favorable signals such as stronger price action, supportive technical readings, resilient margins, improving cash generation, or clearly stated positive news.
- Bull Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the favorable interpretation is better supported by the supplied materials.
- Evidence Discipline: Do not introduce product narratives, customer metrics, market-share claims, or recurring price-pattern claims unless they already appear in the supplied materials.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.
- Every concrete claim must be grounded in the supplied analyst reports or debate history.
- Past reflections are process reminders only, not market evidence.
- When citing evidence, append source tags such as [Market], [Sentiment], [News], [Fundamentals], [Debate], or [Process].

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Process reminders from similar situations and lessons learned (not market facts): {past_memory_str}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
{get_report_evidence_guardrail_instruction(["Market", "Sentiment", "News", "Fundamentals", "Debate", "Process"])}
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
