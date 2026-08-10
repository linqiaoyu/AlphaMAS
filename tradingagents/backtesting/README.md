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
Every fill reports commission, signed implementation-shortfall slippage
`quantity * (fill_price - raw_open_price)`, and their sum as total transaction
cost. This formula is positive for both the adverse buys and adverse sells
produced by the broker. Daily state reports the corresponding cumulative
components; `cumulative_cost` is a compatibility alias for
`cumulative_transaction_cost`. These values are informational: commission is
already deducted from cash, and slippage is already borne through fill price,
so neither is subtracted from equity again. At every valuation,
`equity = initial_cash + realized_pnl + unrealized_pnl + cumulative_dividends`.
Dividends are not trading P&L and remain separate in `cumulative_dividends`.

Metrics use daily close equity and the configured annualization factor (252 by
default). The configured annual risk-free rate is converted to a simple daily
rate before excess returns are calculated. Maximum drawdown is
the magnitude of the worst daily equity/peak decline. Turnover is
`sum(abs(fill_notional)) / average_daily_equity`; exposure is the mean daily
position market value/equity; time in market is the fraction of valuation
sessions with positive quantity. Transaction cost rate is
`total_transaction_cost / initial_equity`, where total transaction cost is
commission plus slippage recomputed from archived fills. Sortino downside deviation is
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
or `failed`. The first failed TradingAgents decision terminates the whole
attempt before any later week or symbol can run. Its runtime Memory mutation is
rolled back, leaving the earlier successful cases as the only reusable prefix.

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
Memory outcome returns use yfinance daily `Close` with `auto_adjust=True` and
all output-shaping history options explicit. The authoritative XNYS calendar
defines SPY-benchmarked horizons; other benchmarks use the asset's
exchange-local session dates. Both price series must have every required date.
The decision-date adjusted close is observation zero and the fifth subsequent
session is observation five. Raw return is the asset
adjusted-close ratio minus one; benchmark return uses the identical two dates;
alpha is raw return minus benchmark return. Historical requests end at the
exclusive day after the latest completed session allowed by
`historical_as_of`, and returned rows after that cutoff are discarded before
alignment or price validation. Timestamped SPY
cutoffs are gated by the actual XNYS close; another benchmark's timestamped
cutoff conservatively excludes its current date. Live resolution excludes the
current UTC date so an in-progress daily bar cannot mature an outcome. Outcome
visibility is persisted as an exact UTC timestamp; legacy date-only visibility
tags mean end-of-day and therefore fail closed for earlier same-day replays.
After a successful TradingAgents experiment, the final log for every symbol
with a completed Agent decision is copied into `memory/symbols/` inside that run
bundle (all of AAPL, AMZN, and JPM for formal M0). This is a one-way archival
copy: runtime and resume never read it. `memory/manifest.json` binds the symbol
files to the experiment, completing run, lineage, Graph hash, and per-file
SHA-256 values; the top-level manifest also records the Memory manifest
checksum. The archive therefore remains analysis-ready if operational runtime
Memory and decision caches are later removed.

TradingAgents cache identity covers experiment, symbol, exact UTC decision
close, analysts, debate depth, models/provider/temperature, data vendors,
point-in-time mode, git and prompt versions, portfolio-state hash, and memory
namespace version and lineage. It also includes a rolling hash of every earlier
successful Agent case in actual execution order, shared across all symbols in
the attempt. Consequently, cache hits can rebuild a legitimate resume but
cannot silently bridge independent historical runs or reuse a downstream
decision after a repaired failure changes the prefix. Only atomically completed
successful cases are cache hits; failed and partial cases remain retryable.

Research depth has one shared mapping for the CLI and backtester: shallow is 1,
medium is 3, and deep is 5 debate and risk-discussion rounds.

The current Graph emits a structured five-tier rating. The adapter collapses
that explicit vocabulary deterministically: Buy/Overweight become `BUY`, Hold
becomes `HOLD`, and Underweight/Sell become `SELL`. Unknown or conflicting text
is a failed decision, never a synthetic hold.
