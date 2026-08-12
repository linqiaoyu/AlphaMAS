# FinMultiTime Data Audit

Read-only audit of the local raw FinMultiTime directory. No archive members were extracted and no files under the raw dataset were written.

**Raw path:** `/Volumes/Jackson/Dataset/FinMultiTime`
**Physical size (including macOS sidecars):** 15.89 GiB
**Logical size excluding `._*` sidecars:** 14.77 GiB
**Physical files:** 578477
**Logical files excluding sidecars:** 284963

## Verdict

**M1 DATA AUDIT PASSED WITH PIT/AVAILABILITY GAPS — CONTRACT MUST HANDLE EXPLICIT UNAVAILABLE DATA**

The dataset supports the M1 augmentation concept and the 78-case schedule, but the evidence contract must represent unavailable/ambiguous modalities explicitly. The highest-impact gaps are missing JPM news, missing AAPL images, date-only news/table availability, and image window metadata that is only inferable from half-year filenames.

## Dataset Structure

The local release contains top-level `.cache`, `image`, `table`, `text`, and `time_series` directories plus README/description metadata. The S&P500 text, table, and time-series data remain in ZIP archives. S&P500 images are expanded into `image/image/S&P500_image_a` through `_z`; the sibling image directory also contains an HS300 ZIP archive. The physical inventory found 578,477 files; the largest top-level directories are `image` (8.78 GB), `text` (3.66 GB), `table` (3.37 GB), `.cache` (667 MB), and `time_series` (584 MB).

| Modality / partition | Source | Files or members | Compressed / physical | Uncompressed / logical |
|---|---|---:|---:|---:|
| text / S&P500 | `text/sp500_news.zip` | 4694 | 3.36 GiB | 12.99 GiB |
| text / HS300 | `text/hs300news_summary.zip` | 892 | 54.09 MiB | 413.54 MiB |
| table / S&P500 | `table/SP500_tabular.zip` | 8028 | 3.08 GiB | 84.04 GiB |
| table / HS300 | `table/hs300_tabular.zip` | 2430 | 55.87 MiB | 568.20 MiB |
| time_series / S&P500 | `time_series/S&P500_time_series.zip` | 4213 | 459.99 MiB | 1.83 GiB |
| time_series / HS300 | `time_series/HS300_time_series.zip` | 810 | 96.20 MiB | 345.82 MiB |
| image / S&P500 extracted + HS300 archive | `image/image` | 142470 | 2.01 GiB | 7.62 GiB |

Archive central-directory counts include only file members, not directory entries. The physical total counts both the expanded S&P500 images and the HS300 ZIP because both are present locally. The largest physical files are `text/sp500_news.zip` (3.36 GiB), `table/SP500_tabular.zip` (3.08 GiB), `image/HS300_image.zip` (2.01 GiB), and `time_series/S&P500_time_series.zip` (460.65 MiB).

### Metadata and documentation

The local documentation consists of `README.md`, `image/image.md`, `sp500stock_data_description.csv`, `hs300stock_data_description.csv`, and `.gitattributes`; macOS `._*` sidecars are also present. The stock description CSV is an availability flag, not a verified per-file inventory.

## Target Stocks and Modality Coverage

| Symbol | Modality | Present | Records/files | Date or period span | Availability / notes |
|---|---|---:|---:|---|---|
| AAPL | text | true | 9311 | 2022-06-03 → 2025-04-18 | File-level symbol mapping with Stock_symbol field; no publication-time or revision field |
| AAPL | table | true | 105966 | period end 2006-09-30 → 2024-12-28; filed 2009-07-22 → 2025-01-31 | SEC-style company facts; period end is distinct from filed date; restatement groups detected |
| AAPL | time_series | true | 6348 | 2000-01-03 → 2025-03-28 | No Adj Close column; values appear adjusted but source metadata does not state auto_adjust semantics |
| AAPL | image | false | 0 | nominal None → None | AAPL description flag is not backed by an actual extracted AAPL image directory |
| AMZN | text | true | 5560 | 2023-03-09 → 2025-04-21 | File-level symbol mapping with Stock_symbol field; no publication-time or revision field |
| AMZN | table | true | 70980 | period end 2006-12-31 → 2024-12-31; filed 2009-07-24 → 2025-02-07 | SEC-style company facts; period end is distinct from filed date; restatement groups detected |
| AMZN | time_series | true | 6348 | 2000-01-03 → 2025-03-28 | No Adj Close column; values appear adjusted but source metadata does not state auto_adjust semantics |
| AMZN | image | true | 51 | nominal 2000-H1 → 2025-H1 | Filename encodes ticker/year/half-year; chart start/end not stored separately |
| JPM | text | false | 0 | None → None | No matching ZIP member |
| JPM | table | true | 6960 | period end 2007-12-31 → 2024-12-31; filed 2009-08-10 → 2025-02-14 | SEC-style company facts; period end is distinct from filed date; restatement groups detected |
| JPM | time_series | true | 6348 | 2000-01-03 → 2025-03-28 | No Adj Close column; values appear adjusted but source metadata does not state auto_adjust semantics |
| JPM | image | true | 51 | nominal 2000-H1 → 2025-H1 | Filename encodes ticker/year/half-year; chart start/end not stored separately |

All three symbols have table and time-series members. Text exists for AAPL and AMZN only; the JPM news member is absent. Images exist for AMZN and JPM only; no `aapl` image directory/file exists even though the description CSV flags `image=1`.

## Temporal Coverage and Formal Schedule

The exact Formal M0 schedule was regenerated from `configs/backtest_m0_2024h1.json` and the repository XNYS calendar implementation. It contains 26 weekly decisions from `2024-01-05` through `2024-06-28`, with execution sessions `2024-01-08` through `2024-07-01` and final valuation `2024-07-05`. The 252-session warm-up spans `2023-01-04` through `2024-01-05` exclusive.

The daily time series begin on 2000-01-03 for all targets and extend beyond 2024H1. Tables contain historical period ends back to 2008/2009-era observations and filed dates through 2025. News and image spans are modality-specific and are documented in the target CSV.

| Symbol | Time-series 2023 rows | Time-series 2024H1 rows | Time-series rows through 2024-07-05 | Formal XNYS session gaps |
|---|---:|---:|---:|---:|
| AAPL | 250 | 124 | 6165 | 0 |
| AMZN | 250 | 124 | 6165 | 0 |
| JPM | 250 | 124 | 6165 | 0 |

The time series therefore provide ample pre-formal history for the 252-session reference. Daily rows are interpreted as completed session bars when used at a session-close decision; the source itself stores a timezone-bearing midnight timestamp rather than an explicit bar close timestamp.

## Point-in-Time Findings

### News/text

- **AAPL:** 9311 JSONL records, `2022-06-03` → `2025-04-18`, fields `Article, Article_title, Date, Stock_symbol, Url`; 0 missing dates, 0 invalid dates, 0 exact duplicate records, and 0 duplicate URLs. Dates are date-only, with no timezone or publication time. `Url` provides a source domain, `Article_title` is present, and `Stock_symbol` values are `{'aapl': 495, 'AAPL': 8816}`; no revision/version field exists.
- **AMZN:** 5560 JSONL records, `2023-03-09` → `2025-04-21`, fields `Article, Article_title, Date, Stock_symbol, Url`; 0 missing dates, 0 invalid dates, 307 exact duplicate records, and 307 duplicate URLs. Dates are date-only, with no timezone or publication time. `Url` provides a source domain, `Article_title` is present, and `Stock_symbol` values are `{'amzn': 798, 'AMZN': 4762}`; no revision/version field exists.
- **JPM:** no news member; text must be unavailable.

A same-calendar-date article cannot be assumed available at the market-open or close decision because the dataset does not say when it was published. The audit therefore treats only dates strictly earlier than the decision session as safely time-ordered and records same-day dates as ambiguous.

### Structured tables / fundamentals

- **AAPL:** 3 SEC-style JSON files and 105966 fact observations; parent `filing_date` fields `2014-10-27` → `2025-01-31`, period ends `2006-09-30` → `2024-12-28`, fact availability `2009-07-22` → `2025-01-31`; 1126 repeated concept/unit/period groups indicate identifiable revisions/restatements; 0 observations lack `filed` (parent filing-date fallback missing for 0).
- **AMZN:** 3 SEC-style JSON files and 70980 fact observations; parent `filing_date` fields `2018-10-26` → `2025-02-07`, period ends `2006-12-31` → `2024-12-31`, fact availability `2009-07-24` → `2025-02-07`; 1285 repeated concept/unit/period groups indicate identifiable revisions/restatements; 0 observations lack `filed` (parent filing-date fallback missing for 0).
- **JPM:** 3 SEC-style JSON files and 6960 fact observations; parent `filing_date` fields `2024-05-01` → `2025-02-14`, period ends `2007-12-31` → `2024-12-31`, fact availability `2009-08-10` → `2025-02-14`; 712 repeated concept/unit/period groups indicate identifiable revisions/restatements; 0 observations lack `filed` (parent filing-date fallback missing for 0).

The tables distinguish fiscal/reporting period (`start`/`end`) from availability (`filed`, plus parent filing-object `filing_date`). `filed` is date-only with no time or timezone, so it is a conservative day-level gate, not an intraday release timestamp. Fiscal period end must not be used as availability.

### Time series

- **AAPL:** 6348 daily rows, `2000-01-03` → `2025-03-28`, columns `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`; timezone offsets `-05:00, -04:00`; duplicate dates 0, non-monotonic order `False`, missing-value rows 0, non-finite rows 0, non-positive-price rows 0, impossible OHLC rows 1.
- **AMZN:** 6348 daily rows, `2000-01-03` → `2025-03-28`, columns `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`; timezone offsets `-05:00, -04:00`; duplicate dates 0, non-monotonic order `False`, missing-value rows 0, non-finite rows 0, non-positive-price rows 0, impossible OHLC rows 0.
- **JPM:** 6348 daily rows, `2000-01-03` → `2025-03-28`, columns `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`; timezone offsets `-05:00, -04:00`; duplicate dates 0, non-monotonic order `False`, missing-value rows 0, non-finite rows 0, non-positive-price rows 0, impossible OHLC rows 11.

No explicit future-return, forecast, direction, target, or prediction column appears in the target time-series headers. The CSV has no Adj Close column and no adjustment-policy metadata; it includes Dividends and Stock Splits, and values appear adjusted, so M1 must carry that source assumption explicitly.

### Images / K-line charts

The images are PNG candlestick charts named `<ticker>_<year>_H1|H2_candlestick.png`, stored under ticker folders in the S&P500 shards. Visual inspection of representative AMZN and JPM 2024H1 charts found a title and date axis for the half-year, but no separate machine-readable start/end metadata. The chart convention appears to cover the named half-year; this is an inference and not a formal PIT guarantee.

| Symbol | All target images | Nominally safe by final decision | In 252-session warm-up + formal span | Latest-safe references across 26 cases | Total bytes | Dimensions |
|---|---:|---:|---:|---:|---:|---|
| AAPL | 0 | 0 | 0 | 0 | 0.00 B | {} |
| AMZN | 51 | 48 | 2 | 26 | 2.10 MiB | {'800x600': 51} |
| JPM | 51 | 48 | 2 | 26 | 2.10 MiB | {'800x600': 51} |

The workload metric is unique files, not repeated case references. AMZN and JPM each have 51 half-year images from 2000H1 through 2025H1; 48 each have nominal ends no later than the final 2024-06-28 decision, while two each fall inside the 2023-01-04 warm-up through 2024-06-28 span under the conservative filename-end rule. AAPL has zero. If each formal case selects its latest nominally safe image, there are 52 possible existing-stock references, reusing files across weeks. PNG target files were header-readable with consistent dimensions; no target duplicate binary hashes were found.

## Future-looking Fields and Leakage Risks

| Source | Field/file type | Classification | Required handling |
|---|---|---|---|
| time_series | `future_return / next_return / return_5d / target / label / direction / forecast / prediction / horizon result` | prohibited_if_present | No such target field appears in the target CSV headers; the only columns are Date, Open, High, Low, Close, Volume, Dividends, Stock Splits. |
| table | `facts.*.*.units.*[].val` | safe_candidate_after_filed_gate | SEC fact values are historical observations, not forecast labels; they remain PIT-sensitive and must be gated by filed date. |
| table | `facts.*.*.units.*[].end / start` | metadata_only_for_availability | Period dates describe the reporting period and are not availability timestamps. |
| text | `Article` | safe_candidate_content_but_not_label | No explicit target/label field exists; article prose may discuss future outcomes, which is ordinary text content rather than a dataset-derived target. |
| image | `*_YYYY_H[1|2]_candlestick.png` | safe_only_after_window_end_gate | Filename/title encodes a chart period but no target/prediction label; the entire chart is prohibited until its inferred window end is no later than the decision. |

The audit found no explicit future target field in the target schemas. The principal leakage risks are temporal: same-day date-only news, same-day date-only filing availability, using fiscal period ends as availability, treating a half-year image as available before its inferred window end, and accidentally exposing later table revisions. The full prohibited/controlled-field register is in `finmultitime_prohibited_future_fields.csv`.

## Formal 78-case Coverage

The machine-readable case table contains one row for each AAPL × 26, AMZN × 26, and JPM × 26 case. Counts are descriptive and do not freeze M1 lookback windows. News 1/3/7/14/30-day counts use valid article dates in the half-open calendar interval `[decision - window, decision)`; same-day records are reported separately as ambiguous. Table counts use observations with `filed < decision date` and exclude same-day filed observations.

| Modality | Overview |
|---|---|
| News | AAPL and AMZN have date-only records; JPM is unavailable. Same-day records are not treated as safe. |
| Tables | All three symbols have filed-date-gated SEC facts; filed dates are day-only and same-day facts are ambiguous. |
| Time series | All three have long daily histories and 0 case rows with a formal XNYS gap. |
| Images | AMZN/JPM have nominally gated chart candidates; AAPL is unavailable. The filename-derived end-date rule deliberately leaves early/final weeks without a current-half-year chart. |

See `finmultitime_formal_case_coverage.csv` for all 78 rows, including latest safe dates, ages, recent news counts, time-series history counts, candidate image counts, and PIT flags.

## Data Quality

The audit reports issues without cleaning upstream data. Text parsing checks malformed JSON, missing/invalid dates, empty article bodies, exact duplicate records, duplicate URLs, and obvious duplicate article keys. Table parsing checks JSON errors, missing filed dates, duplicate fact rows, inconsistent repeated periods/restatement groups, and schema keys. Time-series checks parse errors, ordering, duplicate sessions, missing values, non-finite values, non-positive prices, impossible OHLC relationships, negative volume, and formal XNYS gaps. Target PNG headers and dimensions were checked and binary hashes compared. Detailed counts are in the CSV/JSON outputs.

## Processed-subset Estimate

A future experiment-specific subset should be built only after the Evidence Contract selects lookbacks. As an empirical upper bound for the full 2023-01-04 through 2024-07-05 M0 warm-up/formal/valuation span, the source sizes are:

- **Text:** AAPL + AMZN only; JPM unavailable. The raw target news members are approximately the sizes recorded in `finmultitime_target_stock_coverage.csv`; a filtered date subset would be smaller.
- **Tables:** all nine target statement members; approximately the sum of their uncompressed member sizes in the target CSV. Filtering concepts/filings will reduce this substantially.
- **Time series:** three CSV members, approximately the sum of their uncompressed member sizes; the formal span is a small fraction of the 2000–2025 history.
- **Images:** zero for AAPL and two nominally usable support-span images each for AMZN/JPM under the conservative rule; existing target image bytes are a measured upper bound and captioning 52 possible latest-safe case references would reuse those files.

Using complete target members as a conservative pre-filter upper bound, the measured three-stock source bytes are approximately **66.2 MiB text** (AAPL + AMZN; JPM unavailable), **88.7 MiB tables**, and **2.1 MiB time series**. The two AMZN and two JPM support-span images add about **0.17 MiB**, for an upper-bound working estimate of approximately **157.2 MiB** before concept/news filtering and serialization overhead. A filtered subset should be smaller; the exact source bytes are in the generated JSON and CSV. This estimate does not copy or build the subset.

## Audit Boundaries

- Raw FinMultiTime modified: **NO**
- Full dataset copied into AlphaMAS: **NO**
- Final processed subset built: **NO**
- Qwen downloaded: **NO**
- Qwen run: **NO**
- Image captions generated: **NO**
- Evidence Packet frozen: **NO**
- DeepSeek calls: **0**
- Paid API calls: **0**
- Formal M1 run: **NO**
- M2 / Agentic RL started: **NO**
- AlphaMAS-Experiments modified: **NO**

## Outputs

- `scripts/finmultitime/audit_finmultitime.py`
- `docs/m1/finmultitime_data_audit.md`
- `docs/m1/finmultitime_data_audit.json`
- `docs/m1/finmultitime_modality_inventory.csv`
- `docs/m1/finmultitime_target_stock_coverage.csv`
- `docs/m1/finmultitime_formal_case_coverage.csv`
- `docs/m1/finmultitime_pit_risks.csv`
- `docs/m1/finmultitime_prohibited_future_fields.csv`

The JSON report contains the full structured audit, including raw source paths, exact schedule events, schema findings, and serialized row-level statistics.
