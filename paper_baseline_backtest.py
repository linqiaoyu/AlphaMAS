import argparse
from pathlib import Path

from tradingagents.backtesting import (
    SingleAssetBacktester,
    TradingAgentsStrategy,
    build_rule_based_baselines,
    run_strategy_suite,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def build_graph_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "deepseek"
    config["deep_think_llm"] = "deepseek-reasoner"
    config["quick_think_llm"] = "deepseek-chat"
    config["backend_url"] = "https://api.deepseek.com"
    config["simulated_exchange_enabled"] = False
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simplified DeepSeek+yfinance paper baseline backtest suite.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "GOOGL", "AMZN"],
        help="Ticker symbols to backtest.",
    )
    parser.add_argument("--start-date", required=True, help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument(
        "--output-dir",
        default=str(Path("reports") / "paper_backtests"),
        help="Directory used for summary and per-run artifacts.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Run only rule-based baselines and skip TradingAgents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategies = build_rule_based_baselines()
    if not args.skip_agent:
        graph = TradingAgentsGraph(debug=False, config=build_graph_config())
        strategies["TradingAgents"] = TradingAgentsStrategy(graph)

    engine = SingleAssetBacktester(
        initial_cash=DEFAULT_CONFIG["simulated_exchange_initial_cash"],
        fill_policy=DEFAULT_CONFIG["simulated_exchange_fill_policy"],
        commission_bps=DEFAULT_CONFIG["simulated_exchange_commission_bps"],
        slippage_bps=DEFAULT_CONFIG["simulated_exchange_slippage_bps"],
        buy_fraction=DEFAULT_CONFIG["simulated_exchange_buy_fraction"],
    )

    summary = run_strategy_suite(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        strategies=strategies,
        backtester=engine,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))
    print(f"\nSaved backtest artifacts to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
