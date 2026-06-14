# AlphaMAS

AlphaMAS is a multi-agent LLM financial trading research framework based on the
TradingAgents architecture. It coordinates specialized analyst, researcher,
trader, risk management, and portfolio management agents to produce structured
market analysis and simulated trading decisions.

The project is intended for research and experimentation only. Outputs are not
financial, investment, or trading advice.

## Project Structure

- `tradingagents/`: core agent graph, data flows, LLM clients, and backtesting
  utilities.
- `cli/`: interactive command line interface.
- `tests/`: regression tests for model selection, signal processing, backtests,
  data handling, and CLI behavior.
- `reports/`: reproduction reports and generated analysis artifacts.
- `paper_baseline_backtest.py`: paper-aligned baseline backtest script.
- `plot_paper_results.py`: plotting utilities for reproduction results.

## Installation

Create and activate a Python environment, then install the project:

```bash
pip install .
```

The package requires Python 3.10 or newer.

## Configuration

Set API keys for the providers you plan to use:

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
export ANTHROPIC_API_KEY=...
export XAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...
export ZHIPU_API_KEY=...
export OPENROUTER_API_KEY=...
export ALPHA_VANTAGE_API_KEY=...
```

You can also copy `.env.example` to `.env` and fill in the relevant values.
Enterprise provider settings can be placed in `.env.enterprise`.

## CLI Usage

Launch the interactive CLI:

```bash
tradingagents
python -m cli.main
```

The CLI lets you choose tickers, an analysis date, LLM provider, research depth,
analyst set, and related runtime options.

## Python Usage

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "deepseek"
config["deep_think_llm"] = "deepseek-reasoner"
config["quick_think_llm"] = "deepseek-chat"

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

See `tradingagents/default_config.py` for all available configuration options.

## Docker

Run the CLI with Docker:

```bash
cp .env.example .env
docker compose run --rm tradingagents
```

For local models with Ollama:

```bash
docker compose --profile ollama run --rm tradingagents-ollama
```

## Tests

Run the test suite from an environment with the project dependencies installed:

```bash
pytest
```
