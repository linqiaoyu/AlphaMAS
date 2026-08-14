# M2-04 Reward Selection Freeze

**Status:** FROZEN  
**Scope:** Pre-Formal development only; not Formal M2  
**Selected raw economic reward:** `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY`

## 1. Scope and research-validity boundary

M2-04 compared the three M2-03 candidates once on the frozen pre-Formal MAXIMUM TRAIN/VALIDATION population. It did not train a policy or inspect Final Holdout, E2E Pilot, Formal 2024H1, B&H, SPY, or M1 performance. The outcome snapshot creates matured reward labels only; it is never agent evidence or a model input.

## 2. Market source and immutable snapshot

The M0/M1 audit identified `yfinance==1.5.2` as the execution-market provider family. M2-04 reused `YFinanceDataProvider.load` with `auto_adjust=False`, `actions=True`, raw Open/Close prices, timezone-naive normalized XNYS labels, and explicit Dividends/Stock Splits. Corporate actions are applied before next-open execution, matching the frozen backtester. FinMultiTime was not used as an execution/reward price source.

- Snapshot archive commit: `5c71719e8cde6a78088a47024a263f58378b83e2`
- Snapshot identity: `3afc723888666e0ca3a12219a57dc577d6cf3953717da731b774634f4aff1445`
- Result archive commit: `6c9e18d7d0ea1a2b91fd4ac5eefe829160a15cac`
- Latest retained session: `2023-07-14`
- Snapshot mutation after Phase A: no

## 3. Population and state construction

The population contains exactly 56 TRAIN and 16 VALIDATION windows for AAPL, AEMD, AGI, AMZN, ARR, EML, JBSS, and JPM. Each decision uses its close and exactly five subsequent XNYS sessions, producing 72 windows, 144 CASH/LONG states, 432 BUY/HOLD/SELL counterfactual outcomes, and 1,296 raw reward values.

`CASH` is `$100,000` cash with zero shares. `LONG` is zero cash and `100000 / decision_close` shares carried from the decision close, with average entry equal to that close and zero open-position commission. Corporate actions are applied at the next open before E0 is recomputed, so the overnight move is not attributed to the new action.

## 4. Candidate diagnostics

All rewards were finite. Standard deviations use `ddof=0`; reported quantiles use linear interpolation.

| Candidate | TRAIN spread > 1e-6 | VALIDATION spread > 1e-6 | Overall mean | Median absolute | P95 absolute | P99 absolute | Relative no-op max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | 1.000 | 1.000 | 0.00383789 | 0.00100025 | 0.06107451 | 0.12644178 | n/a |
| R2 | 1.000 | 1.000 | -0.00033333 | 0 | 0.05025794 | 0.10471937 | 0 |
| R3 | 1.000 | 1.000 | -0.00496457 | 0 | 0.06585843 | 0.12659672 | 0 |

TRAIN/VALIDATION means were respectively 0.00363690/0.00454137 for R1, -0.00033333/-0.00033333 for R2, and -0.00486128/-0.00532605 for R3. All three had a non-trivial spread in every state and every symbol. Cross-symbol standard deviations varied with the empirical price paths, but finite rates and action discrimination remained complete across all eight symbols.

## 5. Correctness, no-op, costs, and market drift

There were 288 economic no-op outcomes and 144 traded outcomes. R2 and R3 were exactly zero on every no-op, satisfying the `1e-12` neutrality tolerance. R1's no-op CASH reward was zero, while no-op LONG paths had median absolute reward 0.02096879 and maximum 0.14470549; this quantifies its passive-market-drift exposure and is the pre-registered semantic limitation of R1.

Formal 5 bps commission and 5 bps adverse slippage were retained. Across all outcomes, median cost was zero because two-thirds were no-ops; traded actions carried the frozen costs. No alternative cost assumption was tested.

## 6. R3 downside behaviour

R3's excess-drawdown penalty activated in 67/432 outcomes (15.5093%): 15.4762% in TRAIN and 15.6250% in VALIDATION. Among non-zero penalties, the median was 0.02210033, mean 0.02986109, and P95 0.06795139.

R2 and R3 best-action sets differed in 11/144 states, exceeding the required 8. One difference occurred in VALIDATION. Differences were distributed across six symbols; AAPL contributed the largest share, 3/11 (27.27%), below the 50% ceiling. R3/R2 P99 absolute scale was 1.208914, below the 2.0 ceiling.

## 7. Gate results and pre-registered selection

| Candidate | Gate 1 correctness | Gate 2 semantics | Gate 3 non-degeneracy | Gate 4 stability | Gate 5 interpretation |
|---|---|---|---|---|---|
| R1 | PASS | BASELINE/FALLBACK ONLY | PASS | PASS | passive-drift limitation |
| R2 | PASS | PASS | PASS | PASS | local-credit PASS |
| R3 | PASS | PASS | PASS | PASS | added-value PASS |

R3 passed every pre-registered added-value condition: correctness, non-degeneracy, at least eight overall best-set changes, at least one VALIDATION change, cross-asset distribution, and P99 scale stability. The frozen hierarchy therefore selects R3 over correctness-valid R2. R1 is rejected as a fallback-only absolute-return baseline; R2 is rejected only because R3 demonstrated the required downside-sensitive decision effect.

## 8. Final freeze

The Full-M2 raw economic reward is:

\[
R3(a)=\ln(E_T^a/E_T^{HOLD})-\max\{0,-\ln(1-MDD(P_a))-[-\ln(1-MDD(P_H))]\}.
\]

This selection does not claim to maximize Formal return. It is frozen for Full M2, A1, and A2 unless a genuine correctness bug invalidates M2-04. The PPO training transform remains `DEFERRED`.

**Final verdict:** PASS — M2 empirical reward study completed and final raw economic reward research-frozen; ready for M2-05.
