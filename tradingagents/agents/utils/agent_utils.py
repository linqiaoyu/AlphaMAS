from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )


def get_news_data_guardrail_instruction() -> str:
    """Return instructions for handling unavailable or empty news data."""
    return (
        " If a news tool returns `[NEWS_UNAVAILABLE]`, `[GLOBAL_NEWS_UNAVAILABLE]`, "
        "or similar unavailability text, you must explicitly state that the news source "
        "was unavailable for the requested date and you must not infer or summarize any "
        "missing articles. If a tool returns `[NEWS_EMPTY]` or `[GLOBAL_NEWS_EMPTY]`, "
        "state that no matching articles were returned and treat news evidence as missing "
        "or weak rather than neutral, bullish, or bearish by default. Do not reinterpret "
        "missing or empty news as 'no catalyst', 'no bad news', 'no positive catalyst', "
        "or any other directional signal."
    )


def get_tool_evidence_guardrail_instruction() -> str:
    """Return instructions that force analysts to stay inside tool outputs."""
    return (
        " Use only facts explicitly returned by the tools you call in this run. "
        "Do not add external background knowledge, generic company lore, rumors, "
        "future events, or unstated statistics. If a tool does not provide a fact, "
        "state that the evidence is missing or unavailable instead of filling the gap. "
        "Do not infer the cause or purpose of R&D, capex, taxes, margin changes, "
        "inventory changes, buybacks, or cash-flow changes unless the tool output "
        "explicitly states it. Do not map generic financial changes to specific "
        "products, AI initiatives, launches, regions, or future catalysts without "
        "explicit support in the tool output. If tools show R&D, capex, or inventory "
        "changes, report the numeric change and timing only unless the returned data "
        "explicitly names the cause."
    )


def get_report_only_instruction() -> str:
    """Return instructions for upstream analysts that should not issue trades."""
    return (
        " You are an upstream evidence-gathering analyst, not the final decision-maker. "
        "Do not recommend BUY, HOLD, or SELL. Do not include a `Rating:` field. "
        "Do not use the phrase `FINAL TRANSACTION PROPOSAL`. Summarize evidence, "
        "uncertainty, and key takeaways only."
    )


def get_report_evidence_guardrail_instruction(allowed_labels=None) -> str:
    """Return instructions for downstream agents to stay grounded in current-run inputs."""
    if allowed_labels is None:
        allowed_labels = [
            "Market",
            "Sentiment",
            "News",
            "Fundamentals",
            "Debate",
            "Trader",
            "Process",
        ]

    label_text = ", ".join(f"[{label}]" for label in allowed_labels)

    return (
        " Use only claims explicitly supported by the provided materials from this run. "
        "Treat unsupported statements inside debate history or plans as unverified unless "
        "they can be traced back to the supplied analyst reports. Do not add outside "
        "knowledge, unstated historical statistics, future catalysts, rumors, executive "
        "quotes, product expectations, partnerships, regional growth claims, or customer "
        "metrics unless they already appear in the supplied materials. If News or "
        "Sentiment reports say data is unavailable or empty, keep those sources missing "
        "and do not fill the gap. Missing or unavailable news, sentiment, or insider "
        "results are unknown, not directional. Do not recast them as 'no catalyst', "
        "'no bad news', 'no positive catalyst', 'no panic selling', 'absence of insider "
        "buying is notable', or any similar signal. Do not infer specific product cycles, "
        "AI initiatives, or future growth engines from generic R&D, capex, or inventory "
        "figures unless the supplied materials explicitly make that connection. Past "
        "reflections may guide process and risk discipline only. They must not be used as "
        "new market facts, empirical pattern claims, probabilities, technical analogies, "
        "or evidence in Key Evidence. If you mention a reflection at all, keep it to a "
        "process note such as caution, sizing discipline, or avoiding overconfidence, and "
        "tag it as [Process] rather than market evidence. When making a concrete claim, "
        f"append an inline source tag chosen from {label_text}. If a claim cannot be tied "
        "to one of these sources, omit it."
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
