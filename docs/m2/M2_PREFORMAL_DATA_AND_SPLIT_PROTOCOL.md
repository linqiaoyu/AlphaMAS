# M2 Pre-Formal Data and Split Protocol

## 1. Objective

This protocol freezes the performance-blind FinMultiTime input universe and exact
TRAIN, VALIDATION, FINAL_HOLDOUT, and E2E_PILOT membership for Full M2, A1, and
A2. It selects inputs only: no semantic state, reward, return, model, or policy
was generated or evaluated.

## 2. Inherited architecture boundary

The inherited M2-01A boundary is unchanged. Future Actor semantics may use only
the existing M1 Research Manager `investment_plan`, unchanged Prompt Trader
proposal/action/reasoning, permitted instrument/time/PIT context, and permitted
endogenous portfolio state. Raw analyst reports, debates, Risk/PM output, and
future outcomes remain forbidden direct inputs.

## 3. Read-only dataset policy

The logical source is `FinMultiTime:S&P500:local-read-only` at
`/Volumes/Jackson/Dataset/FinMultiTime`. ZIP members were streamed or inspected via
their central directories; images were inventoried in place. No raw member was
extracted, rewritten, renamed, or deleted. The pre/post mutation guard passed:
**TRUE**.

## 4. Candidate universe

The complete 4,213-row FinMultiTime S&P500 description partition was audited
against physical text/table/time-series members and image inventory. Time-series
eligibility uses only XNYS continuity and structural validity. Missing modalities
remain legitimate. Correctness-eligible: 2983;
correctness-excluded: 1230.

## 5. Frozen PIT eligibility rules

- TEXT: `article_date < decision_session`, with the frozen 30-calendar-day lookback.
- TABLE: `filed_date < decision_session`; reporting period end is never availability.
- TIME_SERIES: 61 structurally valid completed XNYS rows through decision.
- IMAGE: inferred half-year end must be strictly before decision.
- OUTCOME: the fifth subsequent XNYS session is maturity and must be pre-2024.

`UNAVAILABLE` is retained as a legitimate modality status. AAPL TEXT remains
fail-closed under the frozen M1 source-integrity erratum, and AAPL IMAGE remains
unavailable under the frozen M1 contract even if later physical inventory differs.

## 6. Performance-blind selection design

Prices are inspected only for missing/non-finite values, positivity, duplicate
sessions, volume sign, and impossible OHLC relations. No return, direction,
volatility, drawdown, reward, or profitability is calculated. Formal M0/M1
performance artifacts were not inspected.

The algorithm `M2-PREFORMAL-UNIVERSE-v1` requires correctness-eligible AAPL, AMZN,
and JPM. Non-Formal candidates are ordered by
`SHA256("AlphaMAS-M2-preformal-universe-v1" + "|" + upper(symbol))`; a first pass adds new
reliable sectors and a second pass fills strictly by that hash order. The exact
rank is recorded in `m2_preformal_symbol_audit.csv`.

## 7. Selected symbols and modality diversity

MAXIMUM symbols: `AAPL, AMZN, JPM, JBSS, EML, AGI, ARR, AEMD`. The three Formal symbols contribute
distinct legitimate availability profiles (including AAPL image/TEXT restrictions
and JPM missing TEXT), while deterministic non-Formal selection supplies
cross-asset and sector diversity. Formal symbols are 3/8; non-Formal symbols are
5/8.

## 8. Sequential split definitions

Each symbol contributes contiguous weekly blocks within each role:

- **TRAIN:** 56 cases; decisions `2023-05-05`–`2023-06-16`; maturities `2023-05-12`–`2023-06-26`.
- **VALIDATION:** 16 cases; decisions `2023-06-30`–`2023-07-07`; maturities `2023-07-10`–`2023-07-14`.
- **FINAL_HOLDOUT:** 16 cases; decisions `2023-07-21`–`2023-07-28`; maturities `2023-07-28`–`2023-08-04`.
- **E2E_PILOT:** 8 cases; decisions `2023-10-06`–`2023-10-06`; maturities `2023-10-13`–`2023-10-13`.

TRAIN may later fit approved components. VALIDATION is for pre-Formal selection
only. FINAL_HOLDOUT is protected until M2-15. E2E_PILOT is engineering-only and
includes the provenance-intact AAPL `2023-10-06` case from the frozen M1 pilot;
its performance cannot drive design.

## 9. Embargo logic

Every left-hand maturity precedes the next protected decision:

- `TRAIN_TO_VALIDATION`: `2023-06-26` < `2023-06-30`; embargo sessions: `2023-06-27, 2023-06-28, 2023-06-29`
- `VALIDATION_TO_FINAL_HOLDOUT`: `2023-07-14` < `2023-07-21`; embargo sessions: `2023-07-17, 2023-07-18, 2023-07-19, 2023-07-20`
- `FINAL_HOLDOUT_TO_E2E_PILOT`: `2023-08-04` < `2023-10-06`; embargo sessions: `2023-08-07, 2023-08-08, 2023-08-09, 2023-08-10, 2023-08-11, 2023-08-14, 2023-08-15, 2023-08-16, 2023-08-17, 2023-08-18, 2023-08-21, 2023-08-22, 2023-08-23, 2023-08-24, 2023-08-25, 2023-08-28, 2023-08-29, 2023-08-30, 2023-08-31, 2023-09-01, 2023-09-05, 2023-09-06, 2023-09-07, 2023-09-08, 2023-09-11, 2023-09-12, 2023-09-13, 2023-09-14, 2023-09-15, 2023-09-18, 2023-09-19, 2023-09-20, 2023-09-21, 2023-09-22, 2023-09-25, 2023-09-26, 2023-09-27, 2023-09-28, 2023-09-29, 2023-10-02, 2023-10-03, 2023-10-04, 2023-10-05`

These are XNYS session boundaries, not calendar-day approximations.

## 10. Frozen semantic-corpus budget tiers

Whole-symbol trajectories are added to preserve strict nesting:

- **COMPACT:** 72 cases; SHA-256 `46cf3d886c89339097e9d0995639feae2ac32962ff1955c2b45fc1292799884e`; symbols `AAPL, AGI, AMZN, EML, JBSS, JPM`.
- **STANDARD:** 84 cases; SHA-256 `e91e83c47362b49786cd86de98cfc5c81d22158cf6da384685c26e795eaefea8`; symbols `AAPL, AGI, AMZN, ARR, EML, JBSS, JPM`.
- **MAXIMUM:** 96 cases; SHA-256 `68fbc780cbb454f78d0ae719f91665737ed6e04928eba91f4c9c0530745d601f`; symbols `AAPL, AEMD, AGI, AMZN, ARR, EML, JBSS, JPM`.

COMPACT is a strict subset of STANDARD, which is a strict subset of MAXIMUM.
M2-07 may choose among them using measured API cost per case only. Performance
cannot trigger hand-picked additions or tier expansion.

## 11. Final Holdout identity and protection

FINAL_HOLDOUT contains 16 cases across
`AAPL, AEMD, AGI, AMZN, ARR, EML, JBSS, JPM`, with decisions
`2023-07-21, 2023-07-28` and maturity sessions
`2023-07-28, 2023-08-04`. Canonical SHA-256:
`f8e011558f59f56db730702700ef3d419b353b63d4f35278ee78ddb70fedabfe`. Its outcomes remain blocked
from development/model-selection code until M2-15. No Holdout performance was
calculated or inspected.

## 12. Environment and reproducibility

Audit execution used resident CPython 3.12.10 through
`.venv-m2-locked/bin/python`, backed by
`/Users/yulinqiao/.local/share/alphamas/venvs/m2-locked`, with NumPy 2.5.2,
pandas 2.3.3, exchange-calendars 4.13.2, yfinance 1.5.2, and tradingagents
0.3.1. JSON is sorted and CSV fields/order are fixed. Exact case identities and
canonical hashes are in `m2_preformal_data_and_split_protocol.json` and `m2_preformal_semantic_case_plan.csv`.

## 13. Limitations

FinMultiTime's partition name is retained as supplied and includes 4,213 symbols;
the stock-description availability flags are advisory, so physical members are
authoritative. Non-Formal news members were date/PIT audited but not subjected to
the separate M1 article-content integrity investigation. Image availability is
inferred conservatively from half-year filenames. OHLC adjustment semantics are
not used for selection.

## 14. Research validity and cost

2024 decision leakage: **NO**. 2024 reward-label leakage: **NO**.
Performance-based selection: **NO**. Formal-result selection: **NO**.
Split outcome overlap: **NO**. Raw dataset mutation: **NO**.

DeepSeek API calls: **0**; DeepSeek API cost: **¥0**; Qwen inference calls:
**0**; AWS GPU hours: **0**; AWS incremental cost: **$0**.

## 15. Final freeze verdict

**PASS — M2 pre-Formal universe and temporal split frozen; ready for M2-03**
