# ARMA11 pre-Formal source freeze

The implementation of `ARMA11_FIXED_V1` was frozen before any 2024H1 ARMA
trading performance was computed or inspected.

- Branch: `compare-with-adft`
- Frozen methodology source SHA (`ARMA_PREFORMAL_SOURCE_SHA`):
  `d33bf08a90f9cb5bb6f4511e66fda8babd9d65e0`
- Frozen Full-M2 parent SHA: `6306ea4ea20cda501c6238db80c34d27bbc16bea`
- Formal experiment: `ARMA11_2024H1`
- Config SHA256: `fa193e253716faeb4a3235c0a1ff51dc9f75f95bf01564153ccec7a3e5eb1fc5`
- Model: `statsmodels.tsa.arima.model.ARIMA`, `order=(1,0,1)`, `trend="c"`,
  stationarity and invertibility enforced
- Fit: deterministic statespace path, `maxiter=200`
- Input: exactly 252 daily simple PIT total returns derived from 253 consecutive
  raw-price observations and session-local dividends/splits
- PIT return formula: `(Close_t * split_factor_t + Dividend_t) / Close_(t-1) - 1`
- Vendor `Adj Close`: not used
- Forecast: exactly five daily returns, compounded as `prod(1+r_hat_h)-1`
- Signal: positive cumulative forecast targets 1.0; zero or negative targets 0.0
- Hard failure: `MODEL_FAILURE_HOLD_FALLBACK`, preserving the current position;
  no alternate model, optimizer, trend, window, or retry
- Trading: the common weekly close/next-XNYS-open engine, $100,000 per stock,
  fractional shares, long-only, no leverage, 5 bps commission, 5 bps slippage,
  daily mark-to-market, no forced final close
- Snapshot identities: the four frozen AAPL/AMZN/JPM/SPY hashes in the Formal config
- Memory/RL/DecisionCache/FinMultiTime/MAS: none
- Paid compute: none

Pre-Formal validation passed 109 ARMA/common-backtester regression tests. The
dedicated structural PIT audit inspected all 78 candidate Formal windows without
fitting a model or computing performance: 78 exact 252-return windows, zero
future-input violations, zero insufficient windows, and zero duplicate cases.

Environment frozen for execution: Python 3.12.10, statsmodels 0.14.5, NumPy
2.5.2, SciPy 1.18.0, pandas 2.3.3, and exchange-calendars 4.13.2.

No model or numerical setting may be changed based on the Formal result.
