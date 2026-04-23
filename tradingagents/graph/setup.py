# TradingAgents/graph/setup.py

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import Any, Dict, List
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.llm_mode import resolve_agent_llm_modes

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, List[Any]],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        conditional_logic: ConditionalLogic,
        agent_llm_modes: Dict[str, str] | None = None,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.conditional_logic = conditional_logic
        self.agent_llm_modes = resolve_agent_llm_modes(agent_llm_modes)

    def _llm_for_role(self, role: str):
        mode = self.agent_llm_modes.get(role, "deep")
        return self.quick_thinking_llm if mode == "quick" else self.deep_thinking_llm

    def _invoke_tool(self, tools_by_name: Dict[str, Any], tool_call: Any) -> tuple[str, str]:
        """Execute one tool call and return (tool_name, output)."""
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
        else:
            tool_name = getattr(tool_call, "name", "")
            tool_args = getattr(tool_call, "args", {})

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                # Some providers may emit raw strings; pass through unchanged.
                pass

        tool = tools_by_name.get(tool_name)
        if tool is None:
            return tool_name, f"Tool '{tool_name}' is not available."

        try:
            output = tool.invoke(tool_args)
        except Exception as exc:
            output = f"Tool execution error for '{tool_name}': {exc}"
        return tool_name, str(output)

    def _run_analyst_until_completion(
        self,
        analyst_node: Any,
        report_key: str,
        tools: List[Any],
        state: AgentState,
        max_tool_iterations: int = 30,
    ) -> str:
        """Run one analyst in an isolated local tool loop."""
        local_state = {
            "messages": [HumanMessage(content=state["company_of_interest"])],
            "company_of_interest": state["company_of_interest"],
            "trade_date": state["trade_date"],
        }
        tools_by_name = {tool.name: tool for tool in tools}

        for _ in range(max_tool_iterations):
            result = analyst_node(local_state)
            messages = result.get("messages", [])

            if not messages:
                return str(result.get(report_key, ""))

            analyst_msg = messages[-1]
            local_state["messages"].append(analyst_msg)

            tool_calls = getattr(analyst_msg, "tool_calls", None) or []
            if len(tool_calls) == 0:
                return str(result.get(report_key, ""))

            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    call_id = tool_call.get("id")
                else:
                    call_id = getattr(tool_call, "id", None)

                tool_name, tool_output = self._invoke_tool(tools_by_name, tool_call)
                local_state["messages"].append(
                    ToolMessage(
                        content=tool_output,
                        tool_call_id=call_id or f"{tool_name}_call",
                        name=tool_name,
                    )
                )

        return (
            f"Error: analyst did not finish within {max_tool_iterations} tool iterations."
        )

    def _create_parallel_analyst_team_node(self, analyst_specs: List[Dict[str, Any]]):
        """Create a fan-out/fan-in analyst stage that runs selected analysts concurrently."""

        def analyst_team_node(state: AgentState) -> dict:
            updates = {}
            if len(analyst_specs) == 1:
                spec = analyst_specs[0]
                updates[spec["report_key"]] = self._run_analyst_until_completion(
                    spec["node"], spec["report_key"], spec["tools"], state
                )
                return updates

            with ThreadPoolExecutor(max_workers=len(analyst_specs)) as executor:
                future_to_spec = {
                    executor.submit(
                        self._run_analyst_until_completion,
                        spec["node"],
                        spec["report_key"],
                        spec["tools"],
                        state,
                    ): spec
                    for spec in analyst_specs
                }

                for future in as_completed(future_to_spec):
                    spec = future_to_spec[future]
                    report_key = spec["report_key"]
                    try:
                        updates[report_key] = future.result()
                    except Exception as exc:
                        updates[report_key] = (
                            f"Error in {spec['name']} analyst execution: {exc}"
                        )

            return updates

        return analyst_team_node

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst specs for concurrent fan-out/fan-in execution.
        analyst_specs = []
        if "market" in selected_analysts:
            analyst_specs.append(
                {
                    "name": "market",
                    "node": create_market_analyst(self._llm_for_role("market_analyst")),
                    "report_key": "market_report",
                    "tools": self.tool_nodes["market"],
                }
            )

        if "social" in selected_analysts:
            analyst_specs.append(
                {
                    "name": "social",
                    "node": create_social_media_analyst(self._llm_for_role("social_analyst")),
                    "report_key": "sentiment_report",
                    "tools": self.tool_nodes["social"],
                }
            )

        if "news" in selected_analysts:
            analyst_specs.append(
                {
                    "name": "news",
                    "node": create_news_analyst(self._llm_for_role("news_analyst")),
                    "report_key": "news_report",
                    "tools": self.tool_nodes["news"],
                }
            )

        if "fundamentals" in selected_analysts:
            analyst_specs.append(
                {
                    "name": "fundamentals",
                    "node": create_fundamentals_analyst(self._llm_for_role("fundamentals_analyst")),
                    "report_key": "fundamentals_report",
                    "tools": self.tool_nodes["fundamentals"],
                }
            )

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self._llm_for_role("bull_researcher"), self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self._llm_for_role("bear_researcher"), self.bear_memory
        )
        research_manager_node = create_research_manager(
            self._llm_for_role("research_manager"), self.invest_judge_memory
        )
        trader_node = create_trader(self._llm_for_role("trader"), self.trader_memory)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self._llm_for_role("aggressive_risk_analyst"))
        neutral_analyst = create_neutral_debator(self._llm_for_role("neutral_risk_analyst"))
        conservative_analyst = create_conservative_debator(self._llm_for_role("conservative_risk_analyst"))
        portfolio_manager_node = create_portfolio_manager(
            self._llm_for_role("portfolio_manager"), self.portfolio_manager_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Analyst stage: parallel fan-out/fan-in to align with paper workflow.
        workflow.add_node(
            "Analyst Team",
            self._create_parallel_analyst_team_node(analyst_specs),
        )

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        workflow.add_edge(START, "Analyst Team")
        workflow.add_edge("Analyst Team", "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        # Compile and return
        return workflow.compile()
