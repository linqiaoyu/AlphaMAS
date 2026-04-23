# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.backtesting import SimulatedExchange
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        # Copy the config so backtest-specific mutations stay local to this instance.
        self.config = dict(config) if config is not None else DEFAULT_CONFIG.copy()
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
            self.config.get("agent_llm_modes"),
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)
        self.simulated_exchange = SimulatedExchange(
            starting_cash=self.config.get("simulated_exchange_initial_cash", 100000.0),
            fill_policy=self.config.get("simulated_exchange_fill_policy", "next_open"),
            commission_bps=self.config.get("simulated_exchange_commission_bps", 0.0),
            slippage_bps=self.config.get("simulated_exchange_slippage_bps", 0.0),
            buy_fraction=self.config.get("simulated_exchange_buy_fraction", 1.0),
            allow_fractional_shares=self.config.get(
                "simulated_exchange_allow_fractional_shares", False
            ),
        )

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def reset_portfolio(self, starting_cash: Optional[float] = None) -> None:
        """Reset the simulated exchange to a clean account state."""
        self.simulated_exchange.reset(starting_cash=starting_cash)

    def load_portfolio_snapshot(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Restore the simulated exchange from a saved snapshot."""
        self.simulated_exchange.load_snapshot(snapshot)

    def get_portfolio_snapshot(self, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """Return the current portfolio marked on the requested date."""
        if as_of_date is None:
            as_of_date = self.config.get("backtest_as_of_date") or date.today().isoformat()
        return self.simulated_exchange.snapshot(str(as_of_date)).to_dict()

    def get_trade_log(self) -> List[Dict[str, Any]]:
        """Return the simulated exchange trade history recorded in this session."""
        return self.simulated_exchange.get_trade_log()

    def pin_backtest_as_of_date(self, trade_date: str) -> str:
        """Pin dataflow tool access to one as-of date for the current run."""
        trade_date_str = str(trade_date)
        self.config["backtest_as_of_date"] = trade_date_str
        set_config({"backtest_as_of_date": trade_date_str})
        return trade_date_str

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, List[Any]]:
        """Create tool registries for different analyst roles."""
        return {
            "market": [
                # Core stock data tools
                get_stock_data,
                # Technical indicators
                get_indicators,
            ],
            "social": [
                # News tools for social media analysis
                get_news,
            ],
            "news": [
                # News and insider information
                get_news,
                get_global_news,
                get_insider_transactions,
            ],
            "fundamentals": [
                # Fundamental analysis tools
                get_fundamentals,
                get_balance_sheet,
                get_cashflow,
                get_income_statement,
            ],
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name
        trade_date_str = self.pin_backtest_as_of_date(trade_date)

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date_str
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        signal = self.process_signal(final_state["final_trade_decision"])
        final_state = self.attach_execution_state(final_state, company_name, trade_date_str, signal)
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date_str, final_state)

        # Return decision and processed signal
        return final_state, signal

    def attach_execution_state(
        self,
        final_state: Dict[str, Any],
        company_name: str,
        trade_date: str,
        signal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach simulated execution and updated portfolio state to a final graph state."""
        if not self.config.get("simulated_exchange_enabled", True):
            final_state["simulated_execution"] = {
                "symbol": company_name,
                "signal": signal or self.process_signal(final_state.get("final_trade_decision", "")),
                "requested_trade_date": str(trade_date),
                "status": "disabled",
                "reason": "Simulated exchange is disabled in config.",
            }
            final_state["portfolio_snapshot"] = self.get_portfolio_snapshot(str(trade_date))
            return final_state

        resolved_signal = signal or self.process_signal(final_state.get("final_trade_decision", ""))
        execution_report, portfolio_snapshot = self.simulated_exchange.execute_signal(
            company_name,
            resolved_signal,
            str(trade_date),
        )
        final_state["simulated_execution"] = execution_report.to_dict()
        final_state["portfolio_snapshot"] = portfolio_snapshot.to_dict()
        return final_state

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
            "simulated_execution": final_state.get("simulated_execution", {}),
            "portfolio_snapshot": final_state.get("portfolio_snapshot", {}),
        }

        # Save to file
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
