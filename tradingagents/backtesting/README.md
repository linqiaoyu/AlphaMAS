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

Metrics use daily close equity and the configured annualization factor (252 by
default). The configured annual risk-free rate is converted to a simple daily
rate before excess returns are calculated. Maximum drawdown is
the magnitude of the worst daily equity/peak decline. Turnover is
`sum(abs(fill_notional)) / average_daily_equity`; exposure is the mean daily
position market value/equity; time in market is the fraction of valuation
sessions with positive quantity. Transaction cost rate is
`total_transaction_cost / initial_equity`. Sortino downside deviation is
`sqrt(mean(min(daily_return - daily_rf, 0) ** 2))`, including zero components
for non-downside days. Undefined Sharpe, Sortino, and Calmar values are JSON
`null`, never infinity.

Raw, unadjusted OHLC prices are used. At the effective-session open, before a
new order, cash dividends are credited to the quantity held at the preceding
close. Splits multiply quantity and divide average entry price before the open
order. Strategy, same-stock buy-and-hold, and SPY buy-and-hold use this same
policy.

Each invocation creates an immutable
`results/backtests/<experiment_id>/runs/<run_id>` bundle. `--force` recomputes
but still creates a new run. Runtime decision cache and experiment memory live
once under `<experiment_id>/runtime`, outside result bundles. `--resume` means
replay from the first valuation session while successful TradingAgents
decisions are restored from that shared content-addressed cache; it is not a
checkpointed portfolio resume. `latest.json` points to the latest attempt and
`run_status.json` is atomically changed from `running` to `success` or `failed`.

Use `--data-source yfinance` to download once and archive canonical CSVs, then
use `--data-source snapshot --snapshot-dir <prior-run>/inputs/market_data` for
strict offline replay. Snapshot CSVs are normalized and validated again;
missing data fails closed and is never downloaded or forward-filled.

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

Research depth has one shared mapping for the CLI and backtester: shallow is 1,
medium is 3, and deep is 5 debate and risk-discussion rounds.

The current Graph emits a structured five-tier rating. The adapter collapses
that explicit vocabulary deterministically: Buy/Overweight become `BUY`, Hold
becomes `HOLD`, and Underweight/Sell become `SELL`. Unknown or conflicting text
is a failed decision, never a synthetic hold.
