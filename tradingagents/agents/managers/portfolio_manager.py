from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_report_evidence_guardrail_instruction,
)


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Allowed Ratings** (use exactly one paper-aligned action):
- **BUY**: Enter or add to a long position, or cover an existing short
- **HOLD**: Maintain the current stance with no trade
- **SELL**: Reduce long exposure, exit a long position, or establish/add to a short position

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
- Process lessons from past decisions (not market evidence): **{past_memory_str}**

**Source-of-truth analyst reports:**
[Market]
{market_research_report}

[Sentiment]
{sentiment_report}

[News]
{news_report}

[Fundamentals]
{fundamentals_report}

**Required Output Structure:**
1. `Rating: BUY` or `Rating: HOLD` or `Rating: SELL`
2. `Key Evidence:` A concise bullet list of the facts you actually used, each with inline source tags.
3. `Executive Summary:` A concise action plan covering entry strategy, risk levels, and time horizon.
4. `Investment Thesis:` Detailed reasoning anchored in the analysts' debate and past reflections.

---

**Risk Analysts Debate History:**
{history}

---

Do not use Overweight, Underweight, or any other label outside BUY / HOLD / SELL.
Be decisive and ground every conclusion in specific evidence from the analysts.
Do not cite process lessons as proof of a market fact, recurring pattern, or expected return.
{get_report_evidence_guardrail_instruction(["Market", "Sentiment", "News", "Fundamentals", "ResearchPlan", "Trader", "RiskDebate", "Process"])}
{get_language_instruction()}"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return portfolio_manager_node
