from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_temporal_prompt_instruction,
)
from tradingagents.evidence.finmultitime import FrozenFinMultiTimeEvidenceStore
from tradingagents.evidence.prompt import build_analyst_local_messages


def build_news_analyst_system_message(asset_label: str) -> str:
    """Build a time-neutral news prompt for both live and historical runs."""
    return (
        "You are a news researcher tasked with analyzing recent news and trends over "
        "the past week. Please write a comprehensive report of the current state of "
        "the world that is relevant for trading and macroeconomics. Use the available "
        f"tools: get_news(ticker, start_date, end_date) for {asset_label}-specific news "
        "by ticker symbol, get_global_news(curr_date, look_back_days, limit) for broader "
        "macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) "
        "to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', "
        "'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and "
        "get_prediction_markets(topic, limit) for live market-implied probabilities "
        "of forward-looking events (e.g. 'Fed rate cut', 'recession risk within the "
        "next 12 months', geopolitical or sector events). Provide specific, actionable "
        "insights with supporting evidence to help traders make informed decisions."
    )


def create_news_analyst(
    llm, evidence_provider: FrozenFinMultiTimeEvidenceStore | None = None
):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        temporal_instruction = get_temporal_prompt_instruction(state)

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        system_message = (
            build_news_analyst_system_message(asset_label)
            + " Make sure to append a Markdown table at the end of the report to "
            "organize key points in the report, organized and easy to read."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " {temporal_instruction} {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(temporal_instruction=temporal_instruction)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(
            build_analyst_local_messages(state, evidence_provider, "news")
        )

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
