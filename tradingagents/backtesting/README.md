# Weekly walk-forward contract

This package runs one long-only account per symbol. Each calendar week's final
XNYS session closes before a strategy is called. The strategy receives OHLCV
only through that close. A changed target creates an order for the next XNYS
session's actual open; it can never fill on the decision bar. Every session is
then valued at its close. `BUY`, `HOLD`, and `SELL` mean target 100%, preserve,
and target 0%, respectively. No-op rebalances have no fill or cost.

Fractional buys reserve both slipped notional and commission:
`quantity = cash / (fill_price * (1 + commission_rate))`. Sells liquidate the
existing non-negative quantity. Missing opens reject the intended-session order;
they are never replaced with closes and are never silently carried forward.

Metrics use daily close equity and 252 sessions per year. Maximum drawdown is
the magnitude of the worst daily equity/peak decline. Turnover is
`sum(abs(fill_notional)) / average_daily_equity`; exposure is the mean daily
position market value/equity; time in market is the fraction of valuation
sessions with positive quantity. Undefined Sharpe, Sortino, and Calmar values
are JSON `null`, never infinity.

Historical memory defaults to safe per-date isolation. Walk-forward experiments
may select `memory_mode=experiment`, which namespaces the log by validated
experiment ID and symbol. `disabled` creates no memory log. The existing Graph
continues to enforce five-trading-session outcome maturity and idempotent
reflection updates.

TradingAgents cache identity covers experiment, symbol, exact UTC decision
close, analysts, debate depth, models/provider/temperature, data vendors,
point-in-time mode, git and prompt versions, portfolio-state hash, and memory
namespace version. Only atomically completed successful cases are cache hits;
failed and partial cases remain retryable.

The current Graph emits a structured five-tier rating. The adapter collapses
that explicit vocabulary deterministically: Buy/Overweight become `BUY`, Hold
becomes `HOLD`, and Underweight/Sell become `SELL`. Unknown or conflicting text
is a failed decision, never a synthetic hold.
