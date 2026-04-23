import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    # LLM settings
    "llm_provider": "deepseek",
    "deep_think_llm": "deepseek-reasoner",
    "quick_think_llm": "deepseek-chat",
    "backend_url": "https://api.deepseek.com",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Agent-level LLM mode routing:
    # "deep" uses deep_think_llm, "quick" uses quick_think_llm.
    # Empty means paper-aligned defaults (analysis/decision agents use deep).
    "agent_llm_modes": {},
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "yfinance",  # Override category default
    },
    # Simulated execution defaults used for single-date runs and future backtests.
    "simulated_exchange_enabled": True,
    "simulated_exchange_initial_cash": 100000.0,
    "simulated_exchange_fill_policy": "next_open",
    "simulated_exchange_buy_fraction": 1.0,
    "simulated_exchange_commission_bps": 0.0,
    "simulated_exchange_slippage_bps": 0.0,
    "simulated_exchange_allow_fractional_shares": False,
}
