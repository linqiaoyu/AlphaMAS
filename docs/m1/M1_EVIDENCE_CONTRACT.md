# M1 FinMultiTime Evidence Contract

**Packet version:** `M1-FINMULTITIME-v1.0.1`
**Status:** FROZEN
**Final verdict:** M1 EVIDENCE CONTRACT ERRATUM PASSED — CONTRACT READY FOR PREPROCESSING
**Parent draft:** `M1-FINMULTITIME-DRAFT-0.1`
**Research-review decision:** `M1 EVIDENCE CONTRACT RESEARCH REVIEW PASSED WITH REQUIRED REVISIONS`
**Freeze date:** `2026-08-13`

This patch-level erratum supersedes `M1-FINMULTITIME-v1.0` without erasing it. Reason: Rename year_to_date_h1/year_to_date_h2 to the precise year_to_date_6m/year_to_date_9m duration classes; selection behaviour is unchanged. The deterministic equivalence report records no research-relevant change.

This contract freezes selection and representation rules only. It does not build the processed subset, generate final Evidence Packets, download/run Qwen, modify M0 behavior, or run M1.

## Research relationship and hard controls

`M1 evidence = M0 historical-safe evidence + FinMultiTime Evidence Packet`. FinMultiTime is additive. M0 historical-safe evidence, Trader policy, Memory, backtester, execution, evaluation prices, metrics, and all 78 formal cases remain unchanged.

Every packet always contains TEXT, TABLE, TIME_SERIES, and IMAGE. Each has explicit `AVAILABLE` or `UNAVAILABLE` status. Unsafe observed items may appear only as provenance-only `AMBIGUOUS_REJECTED` and are never Agent-visible.

## Residual data validation and price semantics

The deterministic scan found 12 impossible-OHLC rows: 1 AAPL, 0 AMZN, and 11 JPM. They are outside the M0 warm-up, formal decision span, and reachable 60-session summary lookback. Raw rows are not repaired.

FinMultiTime does not explicitly document the adjustment semantics of the target OHLC series. Empirical comparison is consistent with dividend-adjusted historical prices for AAPL/JPM and raw-equivalent OHLC for AMZN over the audited period. FinMultiTime OHLC is therefore treated as source-native descriptive data with adjustment semantics not contractually guaranteed.

The contract treats FinMultiTime OHLC as source-native descriptive data: no silent normalisation, no repair to force equality with M0, and no use for execution or valuation. The validated M0 market path remains authoritative.

The audit has 69 deterministic comparison rows. Close relative-difference ranges are AAPL `-0.0114933` to `-0.0033496`, AMZN `-0` to `-0`, and JPM `-0.06224199` to `-0.0172508`.

## TEXT

PIT gate: `Date < decision_session_date`. Same-day date-only news is `AMBIGUOUS_REJECTED`; no external news, live web, or cross-symbol filling is allowed. Deduplicate exact records, then exact URLs, retain removed/kept hashes, order newest-first deterministically, and select at most 8 articles from the fixed 30-calendar-day lookback. JPM is always `UNAVAILABLE`.

| Symbol | Window | Cases with article | Mean count | Maximum | No-article cases |
|---|---:|---:|---:|---:|---:|
| AAPL | 7 days | 0/26 | 0.0 | 0 | 26/26 |
| AAPL | 14 days | 0/26 | 0.0 | 0 | 26/26 |
| AAPL | 30 days | 2/26 | 11.115 | 212 | 24/26 |
| AMZN | 7 days | 0/26 | 0.0 | 0 | 26/26 |
| AMZN | 14 days | 0/26 | 0.0 | 0 | 26/26 |
| AMZN | 30 days | 2/26 | 14.115 | 261 | 24/26 |
| JPM | 7 days | 0/26 | 0.0 | 0 | 26/26 |
| JPM | 14 days | 0/26 | 0.0 | 0 | 26/26 |
| JPM | 30 days | 0/26 | 0.0 | 0 | 26/26 |

## TABLE

PIT gate: `filed_date < decision_session_date`; `period_end` is never an availability gate. Same-day filings are `AMBIGUOUS_REJECTED`.

The six fixed concepts are:

| Concept | Interpretation | Cross-asset coverage | Concept-level unavailable cases |
|---|---|---:|---:|
| `Assets` | balance-sheet total assets | 3/3 | 0 |
| `Liabilities` | balance-sheet total liabilities | 2/3 | 26 |
| `StockholdersEquity` | stockholders' equity | 3/3 | 0 |
| `NetCashProvidedByUsedInOperatingActivities` | cash flow from operations | 3/3 | 0 |
| `NetCashProvidedByUsedInInvestingActivities` | cash flow from investing | 3/3 | 0 |
| `NetCashProvidedByUsedInFinancingActivities` | cash flow from financing | 3/3 | 0 |

Each selected fact exposes taxonomy, concept, value, unit, form, fy, fp, period start, period end, inclusive `period_duration_days`, filed date, accession number, source member, and source/provenance hash. Balance-sheet facts remain point-in-time. Cash-flow facts retain the actual quarterly, year-to-date, or annual source-reported horizon; no annualisation, interpolation, derivation, or artificial period conversion is allowed.

Duration classes use inclusive calendar-day ranges with tolerance for 13-week quarters and 52/53-week fiscal calendars: `quarterly` = 45–120 days, `year_to_date_6m` = 121–210, `year_to_date_9m` = 211–300, `annual` = 301+, `point_in_time` = no source period start, and `other_duration` = 1–44 days.

Selection is deterministic: apply PIT; resolve restatements only among versions already filed before the decision; select the latest economic period by period end; at a shared latest period end prefer the duration class matching `fp/form` (`Q1` → `quarterly`, `Q2` → `year_to_date_6m`, `Q3` → `year_to_date_9m`, `FY/Q4/10-K` → `annual`). Conflicting values at identical filing metadata or a non-canonical selection produce concept-level `UNAVAILABLE`. Diagnostics are in `m1_evidence_contract_table_selection_diagnostics.csv`.

## TIME_SERIES

PIT gate: completed session rows with `session <= decision_session`; the source-selection logic retains at least 61 completed rows through the decision session. The packet contains no raw rows, future labels, targets, or forecasts.

The off-by-one issue is fixed. Exact formulas are:

- 5-session cumulative return: `Close[t] / Close[t-5] - 1` (6 closes).
- 20-session cumulative return: `Close[t] / Close[t-20] - 1` (21 closes).
- 60-session cumulative return: `Close[t] / Close[t-60] - 1` (61 closes).
- 20-session realised volatility: `sample_std(ddof=1)(last 20 one-session returns) * sqrt(252)`, with `r_i = Close[i] / Close[i-1] - 1` (21 closes).
- 20-session high-low range: `max(High[t-19:t]) / min(Low[t-19:t]) - 1` (20 completed bars).
- Drawdown from 60-session peak: `Close[t] / max(Close[t-59:t]) - 1` (60 completed closes including the decision session).
- Relative volume: `Volume[t] / mean(Volume[t-20:t-1])` (current volume versus the previous 20 completed sessions; 21 rows).

M0 validated market data remains authoritative for execution, valuation, and corporate-action accounting. FinMultiTime prices are descriptive evidence only.

## IMAGE and future Qwen adapter

AAPL is explicitly `UNAVAILABLE`. AMZN/JPM may use their latest completed half-year chart: H1 nominal end is June 30, H2 nominal end is December 31, and inferred end must be strictly before the decision. No additional 30/60/90/180-day staleness cutoff exists. Agent-visible metadata includes filename/source identity, inferred chart period, inferred period end, and evidence age.

The simulation selects 2 unique image files for 52 repeated references. Evidence age is 5–180 days (median 92.0); AAPL has no eligible file.

| Symbol | References | Unique files | Age <=30d | <=90d | <=180d | <=365d |
|---|---:|---:|---:|---:|---:|---:|
| AAPL | 0 | 0 | 0/26 | 0/26 | 0/26 | 0/26 |
| AMZN | 26 | 1 | 4/26 | 13/26 | 26/26 | 26/26 |
| JPM | 26 | 1 | 4/26 | 13/26 | 26/26 | 26/26 |

Later Qwen use is offline preprocessing only: `FinMultiTime image -> frozen Qwen3-VL-2B-Instruct caption -> Market Analyst`. Qwen is not an Agent, is not trained, is not an experiment variable, and is not called during Formal M1 runtime. The prompt must say: “Describe only information visually observable in the supplied financial chart.” It must prohibit external/company knowledge not visible in the image, subsequent events, future returns, forecasts, predictions, price targets, and BUY/HOLD/SELL recommendations. No ticker/company identity or additional text context is supplied. The frozen short schema is `trend`, `momentum_visual`, `volatility_visual`, `candlestick_structure`, `notable_gap_or_reversal`, `support_resistance_visual`, `volume_visual`, `other_visible_pattern`, `confidence`, with a maximum caption size of 900 characters. The exact model revision and runtime/generation environment remain intentionally unfrozen until caption preprocessing.

## Routing and budget

Routing is modality-specific: TEXT → News Analyst; TABLE → Fundamentals Analyst; TIME_SERIES → Market Analyst; IMAGE caption → Market Analyst. Social Analyst receives no new FinMultiTime-specific modality. The complete raw packet is not injected into Bull/Bear Researchers, Research Manager, Trader, Risk agents, or Portfolio Manager; derived information travels through the existing analyst-report flow.

| Analyst | Min additional chars | Median | Max additional chars |
|---|---:|---:|---:|
| News Analyst | 19 | 19.0 | 8841 |
| Fundamentals Analyst | 2482 | 2952.0 | 2977 |
| Market Analyst | 527 | 573.5 | 587 |
| Social Analyst | 0 | 0.0 | 0 |

| Section | Deterministic maximum | Representation |
|---|---:|---|
| TEXT | 12,000 chars | max 8 newest records; title <= 200; body <= 900 chars |
| TABLE | 3,200 chars | six fixed concepts, one latest eligible fact each |
| TIME_SERIES | 3,200 chars | seven fixed summary fields; no raw rows |
| IMAGE | 1,600 chars | one bounded caption, if later approved |
| Total packet | 22,000 chars | includes a fixed envelope/provenance budget; UTF-8 chars are the review proxy |

The raw audit corpus contains 14871 article records; raw article body length is 133 / 4073 / 32767 characters (min/median/max). Tables contain 183906 observations; serialized observation length is median 432 and maximum 480 characters. The local M0 run recorded 2262 API calls with prompt-token range 661–25309 (median 9609); no new API call was made for this analysis.

### 78-case deterministic simulation

The simulation contains 78/78 cases. PIT violations: 0. Ambiguous rejected observations: 4242. Estimated packet characters: 3664–12847 (median 4125.0). Exact metadata is in `m1_evidence_contract_case_simulation.csv`; no Agent, outcome, target, or future trading performance was used.

| Modality | AVAILABLE | UNAVAILABLE |
|---|---:|---:|
| TEXT | 4 | 74 |
| TABLE | 78 | 0 |
| TIME_SERIES | 78 | 0 |
| IMAGE | 52 | 26 |

## Validation and boundaries

| Invariant | Result |
|---|---|
| every_modality_has_explicit_status | PASS |
| formal_cases_are_78 | PASS |
| future_pit_violations_are_zero | PASS |
| no_future_label_or_target_included | PASS |
| no_later_restatement_visible_early | PASS |
| no_m0_execution_or_evaluation_input_replaced | PASS |
| no_raw_finmultitime_row_modified | PASS |
| no_same_day_date_only_news_admitted | PASS |
| no_same_day_filed_table_fact_admitted | PASS |
| no_unfinished_half_year_image_admitted | PASS |
| routing_follows_frozen_analyst_map | PASS |
| time_series_has_required_history | PASS |

The following remain deliberately unfrozen: processed three-stock subset, image captions, final Evidence Packets, M1 input SHA bundle, M1 runtime integration, M1 pilot environment, and Formal M1 result.

Raw FinMultiTime modified: **NO**. Final processed subset built: **NO**. Final Evidence Packets built: **NO**. Qwen downloaded: **NO**. Qwen run: **NO**. DeepSeek calls: **0**. Paid API calls: **0**. Trader, Memory, and execution modified: **NO**. Formal M1 run: **NO**. M2 / Agentic RL started: **NO**. AlphaMAS-Experiments modified: **NO**.

Freeze artifacts include `M1_EVIDENCE_CONTRACT.md`, `m1_evidence_contract.json`, `m1_evidence_contract_case_simulation.csv`, `m1_evidence_contract_table_selection_diagnostics.csv`, `m1_contract_erratum_equivalence.json`, and `m1_evidence_contract_freeze.json`. The freeze manifest records SHA-256 hashes for the final contract, simulation, equivalence evidence, and required audit/semantics reports.
