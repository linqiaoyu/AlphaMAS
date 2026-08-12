# M1 FinMultiTime Evidence Contract Draft

**Packet version:** `M1-FINMULTITIME-DRAFT-0.1`
**Status:** PROPOSED — NOT FORMALLY FROZEN
**Final verdict:** M1 EVIDENCE CONTRACT PROPOSAL READY FOR RESEARCH REVIEW

This document proposes the controlled FinMultiTime augmentation contract. It is a design-stage artifact for human/research review. It does not build the final M1 dataset, create formal Evidence Packets, call an LLM, download/run Qwen, modify Trader/Memory/backtesting behavior, or run Formal M1.

## Research relationship and hard controls

`M1 evidence = M0 historical-safe evidence + FinMultiTime Evidence Packet`. FinMultiTime supplements M0; it does not replace M0 safe evidence, execution prices, valuation prices, backtesting, Memory, Trader policy, or metrics. All 78 formal decisions remain mandatory.

The proposed packet always contains TEXT, TABLE, TIME_SERIES, and IMAGE sections. Each starts with `AVAILABLE` or `UNAVAILABLE`; an observed but unsafe item may be recorded only as `AMBIGUOUS_REJECTED` in provenance and is not Agent-visible.

## Residual data validation

The deterministic scan found 12 impossible-OHLC rows: 1 AAPL, 0 AMZN, and 11 JPM. All are outside the 252-session M0 warm-up, outside the Formal 2024H1 calendar/decision span, and not reachable by the proposed 60-session time-series summary lookback. Raw rows are not repaired.

| Symbol | Anomaly dates | Inside M0 warm-up | Inside Formal 2024H1 | Reachable by proposed lookback |
|---|---|---:|---:|---:|
| AAPL | 2008-03-19 | 0 | 0 | 0 |
| AMZN | none | 0 | 0 | 0 |
| JPM | 2002-03-25, 2002-08-14, 2002-12-20, 2003-02-12, 2003-10-20, 2006-06-26, 2007-08-09, 2009-05-22, 2009-10-21, 2009-12-24, 2011-08-02 | 0 | 0 | 0 |

### Time-series price semantics

The frozen M0 snapshot reference is the local archived M0 run input at `results/backtests/M0_original_prompt_2024H1/runs/20260812T082530978211Z_2535896c/inputs/market_data`, whose manifest records `auto_adjust=false` and corporate actions enabled. No live yfinance or web data was used.

The comparison is not a forced-match test. On ordinary and dividend-event dates in the overlapping 2023-01-04 through 2024-07-05 range, FinMultiTime volumes are generally equal or nearly equal to M0 while dividend-paying AAPL/JPM OHLC prices are consistently below raw M0 prices by time-varying factors. AMZN OHLC is raw-like in that window. This is consistent with adjusted historical prices for AAPL/JPM but raw-like prices for AMZN; the universal source policy is therefore **partially adjusted / inconsistent across target symbols and unresolved**. The contract must declare this assumption and must not silently call FinMultiTime raw or adjusted for every symbol.

Deterministic consistency rows: 69. AAPL close relative-difference range: `-0.0114933` to `-0.0033496`; AMZN: `-0` to `-0`; JPM: `-0.06224199` to `-0.0172508`. Split-event rows outside the M0 snapshot are retained with an explicit no-reference status.

See `finmultitime_ohlc_anomalies.csv` and `finmultitime_timeseries_consistency.csv` for the exact rows.

## M0 versus FinMultiTime evidence map

The detailed mapping is in `m0_vs_finmultitime_evidence_map.csv`. The main additive candidates are: (1) a PIT-safe historical text corpus with explicit missingness and deterministic deduplication; (2) six compact filed-date-gated financial facts; (3) fixed multi-window time-series summaries with source/session provenance; and (4) optionally, conservatively gated chart images. OHLCV and named technical indicators are primarily duplicated and remain M0-controlled.

## Proposed TEXT contract

PIT gate: `news.Date < decision_session_date`. The date-only source cannot prove same-day publication timing, so same-day records are `AMBIGUOUS_REJECTED` and excluded. JPM has no valid text member and is `UNAVAILABLE` in every case; no external web fill is permitted.

Deduplication is deterministic: exact canonical record identity, then exact duplicate URL, with removed hashes and the kept hash preserved in provenance. The removed-record audit is in `finmultitime_news_deduplication.csv`. No semantic or LLM deduplication is used. Eligible records are ordered newest date first, then URL, title, and record hash; at most 8 are represented with bounded title/body characters.

### News lookback analysis

Counts below are after the contract deduplication pass and use only the frozen audit corpus; no strategy performance or forward return was inspected.

| Symbol | Window | Formal cases with article | Average article count | Maximum | No-article cases |
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

Recommendation: fixed 30 calendar days. Seven and fourteen days produce zero safe articles in all formal cases; 30 days produces sparse coverage (2/26 AAPL, 2/26 AMZN, 0/26 JPM) while preserving recency and a concise deterministic context. A no-article case remains `TEXT = UNAVAILABLE`; the lookback is not extended to manufacture coverage.

## Proposed TABLE contract

PIT gate: `filed_date < decision_session_date`. `period_end` describes the economic period and is never an availability gate. Same-day filed observations are `AMBIGUOUS_REJECTED` because filing time-of-day is absent.

For the same taxonomy/concept/unit/start/end economic fact, retain only the latest eligible filed version, using deterministic accession/form/member tie-breaks and retaining `concept`, `unit`, `period`, `filed`, `accn`, and provenance. A later restatement can never appear in an earlier decision.

The proposed compact schema is:

| Concept | Interpretation | Cross-asset coverage |
|---|---|---:|
| `Assets` | balance-sheet total assets | 3/3 |
| `Liabilities` | balance-sheet total liabilities | 2/3 |
| `StockholdersEquity` | stockholders' equity | 3/3 |
| `NetCashProvidedByUsedInOperatingActivities` | cash flow from operations | 3/3 |
| `NetCashProvidedByUsedInInvestingActivities` | cash flow from investing | 3/3 |
| `NetCashProvidedByUsedInFinancingActivities` | cash flow from financing | 3/3 |

The table contract exposes at most one latest eligible fact per selected concept per case. A missing field is `UNAVAILABLE`; no future substitution or cross-stock substitution is allowed. Full candidate coverage and revision diagnostics are in `finmultitime_table_concept_coverage.csv`.

## Proposed TIME_SERIES contract

PIT gate: at a valid XNYS session-close decision, completed rows through `session <= decision_session` are eligible, and no later session is eligible. The packet contains no raw OHLCV rows and no target labels.

The fixed summary uses 5-, 20-, and 60-session cumulative returns, 20-session annualised realised volatility, 20-session high-low range, drawdown from the 60-session peak, and current volume relative to the 20-session mean. This is intentionally compact and overlaps with M0 technical analysis; the M0 snapshot and indicator evidence remain authoritative, and the FinMultiTime summary is labelled as augmentation.

## Proposed IMAGE contract

An image filename is interpreted conservatively: H1 nominal end is June 30 and H2 nominal end is December 31. The image is eligible only when the inferred end is strictly before the decision. A half-year still in progress is never used. The metadata convention is inferred from filenames and is not a machine-readable chart end timestamp.

Formal simulation selects 2 unique image files for 52 repeated case references, totaling 88838 bytes. Image age across selected references is 5 to 180 calendar days (median 92.0). AAPL is unavailable; AMZN and JPM each reuse their 2023 H2 image across all 26 formal cases.

| Symbol | Repeated case references | Unique files | Coverage <=30d | <=90d | <=180d | <=365d |
|---|---:|---:|---:|---:|---:|---:|
| AAPL | 0 | 0 | 0/26 | 0/26 | 0/26 | 0/26 |
| AMZN | 26 | 1 | 4/26 | 13/26 | 26/26 | 26/26 |
| JPM | 26 | 1 | 4/26 | 13/26 | 26/26 | 26/26 |

No additional fixed staleness threshold is proposed: 30/90-day thresholds would make most otherwise eligible completed images unavailable, while the strict window-end gate already prevents in-progress charts. Coverage is reported for review. If images are retained after review, the previously agreed offline architecture remains `image -> frozen Qwen3-VL-2B-Instruct -> structured caption`; no Qwen was downloaded or run here. With only two genuinely eligible unique files, Qwen is not necessary for contract validation, but removing the image modality is not proposed at this stage.

## Controlled packet schema and budget

Every packet has the version, symbol, decision time, four modality sections, source identity, source availability date/session, evidence age, SHA-256 provenance reference, deterministic selection rule, and explicit missingness reason. The machine-readable contract is in `m1_evidence_contract_draft.json`.

| Section | Deterministic maximum | Representation |
|---|---:|---|
| TEXT | 12,000 chars | max 8 newest records; title <= 200; body <= 900 chars |
| TABLE | 3,200 chars | six fixed concepts, one latest eligible fact each |
| TIME_SERIES | 3,200 chars | seven fixed summary fields; no raw rows |
| IMAGE | 1,600 chars | one bounded caption, if later approved |
| Total packet | 22,000 chars | includes a fixed envelope/provenance budget; UTF-8 chars are the review proxy |

The raw audit corpus contains 14871 article records; raw article body length is 133 / 4073 / 32767 characters (min/median/max). Tables contain 183906 observations; serialized observation length is median 432 and maximum 480 characters. The local M0 run recorded 2262 API calls with prompt-token range 661–25309 (median 9609); no new API call was made for this analysis.

### 78-case dry-run metadata

The simulation contains 78/78 cases. PIT violations: 0. Ambiguous rejected observations: 4242. Estimated packet characters: 1867–10712 (median 1967.0).

| Modality | AVAILABLE | UNAVAILABLE |
|---|---:|---:|
| TEXT | 4 | 74 |
| TABLE | 78 | 0 |
| TIME_SERIES | 78 | 0 |
| IMAGE | 52 | 26 |

The exact case metadata is in `m1_evidence_contract_case_simulation.csv`. It contains all 26 weekly sessions for each of AAPL, AMZN, and JPM, with no trading outcomes or Agent prompts.

## Research-validity review

- **Augmentation:** FinMultiTime is additive; M0 evidence and execution/valuation inputs remain unchanged.
- **Controlled variable:** Trader, Memory, execution, backtester, and metrics are outside this proposal.
- **PIT:** strict text/table/image gates and session cutoff prevent evidence later than decision time; simulation reports zero violations.
- **Missingness:** unavailable modalities remain explicit, never deleted, backfilled, web-filled, or cross-stock substituted.
- **No target leakage:** no forward returns, predictions, labels, or outcome fields enter the packet.
- **No outcome tuning:** lookbacks, concepts, and image rules use audit coverage and financial interpretability only.
- **Reusability:** once reviewed and frozen, the same input-selection rules can be reused unchanged by M2, A1, and A2.

## Review gates before freezing

1. Confirm the unresolved FinMultiTime price-semantics declaration is acceptable.
2. Approve the fixed 30-day text lookback and sparse-coverage behavior.
3. Approve the six-concept table schema and latest-eligible restatement rule.
4. Decide whether to retain images and whether later Qwen captioning is worth two unique files.
5. Freeze the reviewed contract before building any processed M1 inputs or formal packets.

## Boundary confirmation

- Raw FinMultiTime modified: **NO**
- Final processed subset built: **NO**
- Formal Evidence Packets generated: **NO**
- Qwen downloaded/run: **NO**
- DeepSeek calls: **0**
- Paid API calls: **0**
- Formal M1: **NOT RUN**
- M2 / Agentic RL: **NOT STARTED**
- AlphaMAS-Experiments modified: **NO**
- Evidence Contract formally frozen: **NO — review remains required**
