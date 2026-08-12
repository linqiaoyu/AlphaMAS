# FinMultiTime Time-Series Price Semantics

## Scope and reference

This is a deterministic, read-only consistency investigation. It compares the target FinMultiTime time-series CSVs with the local frozen M0 market snapshot at `results/backtests/M0_original_prompt_2024H1/runs/20260812T082530978211Z_2535896c/inputs/market_data`. The M0 manifest records `auto_adjust=false`, includes corporate actions, and covers `2023-01-04` through `2024-07-05`.

No live yfinance query, web source, fresh download, forward return, or trading outcome was used. The comparison is diagnostic; it is not intended to force the two data sources to match.

## Deterministic date set

The set includes four ordinary non-corporate-action dates per target, every FinMultiTime dividend event whose date overlaps the M0 snapshot plus its adjacent sessions, and every target split event plus adjacent sessions. Split-event rows outside the M0 snapshot are retained with `NOT_AVAILABLE_OUTSIDE_FROZEN_SNAPSHOT` rather than filled from another source.

The resulting table has 69 rows. Exact values, action fields, and per-field relative differences are in `finmultitime_timeseries_consistency.csv`.

## Findings

| Symbol | Direct comparison rows | FinMultiTime minus M0 close relative-difference range | Median volume relative difference |
|---|---:|---:|---:|
| AAPL | 22 | -0.0114933 to -0.0033496 | 0 |
| AMZN | 4 | -0 to -0 | -0.00022107 |
| JPM | 24 | -0.06224199 to -0.0172508 | 0 |

AAPL and JPM show time-varying price differences relative to the raw M0 snapshot, including on ordinary and dividend-event dates, while volume is usually unchanged or only very slightly different. AMZN OHLC is raw-equivalent over the audited overlap. These observations are empirical comparisons, not proof of a universal adjustment policy.

## Contract conclusion

FinMultiTime does not explicitly document the adjustment semantics of the target OHLC series. Empirical comparison is consistent with dividend-adjusted historical prices for AAPL/JPM and raw-equivalent OHLC for AMZN over the audited period. The M1 contract therefore treats FinMultiTime OHLC as source-native descriptive data with adjustment semantics not contractually guaranteed.
FinMultiTime OHLC remains source-native descriptive data: no silent normalisation, no repair to force equality with M0, and no use for execution or valuation. If used for descriptive summaries, retain the source identity and hash.

The 12 impossible-OHLC rows are a separate structural issue and are listed in `finmultitime_ohlc_anomalies.csv`; all are outside the relevant M0 warm-up, formal span, and proposed 60-session lookback.
