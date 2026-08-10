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

`realized_pnl` is completed round-trip price P&L net of both the entry and exit
commissions, with each commission attributed exactly once. The entry commission
remains attached to the open position until liquidation. `unrealized_pnl`
is the open position's mark-to-fill-price movement net of that entry commission.
`cumulative_cost` separately reports every commission already deducted from
cash and equity; it must not be subtracted from the P&L fields again. At every
valuation, `equity = initial_cash + realized_pnl + unrealized_pnl +
cumulative_dividends`. Dividends are not trading P&L and remain separate in
`cumulative_dividends`.

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
but still creates a new run and a fresh historical-memory lineage. Runtime
storage lives under `<experiment_id>/runtime`, outside result bundles, but
experiment memory is further isolated by graph hash, run lineage, and symbol.
Normal reruns also receive a fresh lineage. `--resume` is mutually exclusive
with `--force` and may continue only the attempt named by `latest.json` when
that attempt is still `running` or `failed` and has the same graph and complete
backtest-protocol identity. A completed or incompatible attempt is rejected.
Resume replays from the first valuation session and restores successful
TradingAgents decisions from lineage-scoped content-addressed cache entries; it
is not a checkpointed portfolio resume. `latest.json` points to the latest
attempt and `run_status.json` is atomically changed from `running` to `success`
or `failed`.

Use `--data-source yfinance` to download once and archive canonical CSVs, then
use `--data-source snapshot --snapshot-dir <prior-run>/inputs/market_data` for
strict offline replay. Snapshot CSVs are normalized and validated again;
missing data fails closed and is never downloaded or forward-filled.
`--resume` rejects the mutable `yfinance` source because identical inputs cannot
be proven; resume the archived snapshot instead. Deterministic synthetic runs
remain resumable for engineering validation.

Historical memory defaults to safe per-date isolation. Walk-forward experiments
may select `memory_mode=experiment`, which requires a validated lineage and
namespaces the log by experiment ID, graph hash, lineage, and symbol. `disabled`
creates no memory log. Every resolved reflection records the first historical
cutoff at which it existed; an earlier decision replay hides that future-derived
entry even when it belongs to the resumed lineage. Legacy resolved entries that
lack this provenance fail closed in historical mode. The Graph continues to
enforce the configured trading-session outcome maturity and idempotent updates.

TradingAgents cache identity covers experiment, symbol, exact UTC decision
close, analysts, debate depth, models/provider/temperature, data vendors,
point-in-time mode, git and prompt versions, portfolio-state hash, and memory
namespace version and lineage. Consequently, cache hits can rebuild a legitimate
resume but cannot silently bridge independent historical runs. Only atomically
completed successful cases are cache hits; failed and partial cases remain
retryable.

Research depth has one shared mapping for the CLI and backtester: shallow is 1,
medium is 3, and deep is 5 debate and risk-discussion rounds.

The current Graph emits a structured five-tier rating. The adapter collapses
that explicit vocabulary deterministically: Buy/Overweight become `BUY`, Hold
becomes `HOLD`, and Underweight/Sell become `SELL`. Unknown or conflicting text
is a failed decision, never a synthetic hold.
