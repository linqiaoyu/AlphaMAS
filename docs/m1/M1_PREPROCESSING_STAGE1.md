# M1 preprocessing Stage 1

This report records the local, pre-caption M1 working subset built on
`baseline-m1`. It is not a final Evidence Packet archive.

## Environment and Phase A

- Starting HEAD: `544c71b7d5b33e8cd31c71b2096d734d5e26f7f8`
- Python: 3.12.10, installed with `uv sync --frozen --extra dev --python 3.12`
- `uv.lock` SHA-256: `beec81b017ae7608e7ff8a529476ed637163af7004efb7c7111a8bd081ae5d29`
- pytest: 9.1.1; pluggy: 1.6.0; auto-loaded plugins: anyio 4.14.2 and langsmith 0.10.17
- Frozen M0 base: `2535896c8b1070b19c06fa6a936663babb4356f7`

The collection stall was reproduced under the pre-existing Python 3.14
environment, not under the required Python 3.12 environment. Faulthandler
identified importlib source loading in the eager M0 contract-test path through
`tradingagents.graph` and `tradingagents.agents`. Plugin-disable controls did
not identify a plugin as the cause. The test-only session guard in
`tests/conftest.py` now exits with an actionable error before that unsupported
interpreter can enter collection. No production runtime code, dependency, or
lock file changed. Full diagnostic evidence is in
`docs/m1/pytest_hang_diagnosis.md`.

Before preprocessing, the complete frozen-environment suite passed:

```text
762 passed, 2 skipped, 19 warnings, 69 subtests passed in 23.00s
```

## Deterministic working subset

- Raw source: `/Volumes/Jackson/Dataset/FinMultiTime` (read-only)
- Processed working path: `data/processed/finmultitime_3stocks_2024h1_v1/`
- Contract: `M1-FINMULTITIME-v1.0.1`
- Contract SHA-256: `13563ba0c829addde44d602cf8b9ac0e2879d8091832ee120a5caefa4c843ab3`
- Case simulation SHA-256: `29556d622656b63225e42a07f3815fa1ab6175636ec6a755fddfdc7e1cc50617`
- Formal schedule: `XNYS_2024H1_26W_DECISION_CLOSES_FINAL_2024-07-05`
- Formal case records: 78 (26 decisions × AAPL, AMZN, JPM)
- Regular files in local output: 90; checksum inventory entries: 87
- Size: approximately 2.9 MB
- `caption_status`: `NOT_GENERATED`
- `final_evidence_packets`: `NOT_GENERATED`
- `input_bundle_frozen`: `false`

The build uses a temporary staging tree, validates it, and atomically replaces
the local target. Per-case records are `preprocessed_case_record` JSON files.
Each case contains all four modality sections and the frozen contract
projection. Time series are stored once per stock in `time_series/*.json`
(181 source rows spanning 2023-10-10 through 2024-06-28), while each case stores
its selected session-date references and derived summaries.

## Modality results

| Symbol | TEXT | TABLE | TIME_SERIES | IMAGE |
| --- | --- | --- | --- | --- |
| AAPL | 2 available / 24 unavailable | 26 available | 26 available | 26 unavailable |
| AMZN | 2 available / 24 unavailable | 26 available; `Liabilities` unavailable in all cases | 26 available | 26 available |
| JPM | 26 unavailable | 26 available | 26 available | 26 available |

TEXT is bounded original source content only: fixed 30-calendar-day PIT
selection, same-day rejection, exact-record and URL deduplication, deterministic
ordering, and at most eight articles. TABLE contains exactly the six frozen
concepts, source-reported duration metadata, and PIT-safe `filed < decision`
selection; no missing facts are derived or annualised. TIME_SERIES retains
source-native values and the seven frozen summaries, including the required
61 completed rows per case. It is not used for execution, valuation, P&L, or
corporate-action accounting.

## Provenance and images

`manifests/source_manifest.json` hashes only the target source members: the
AAPL/AMZN news members, all three target time-series members, nine target table
members, and the two staged image files. ZIP provenance records the outer
archive, member, compressed/uncompressed sizes, CRC32, and member SHA-256.
The processed inventory is in `manifests/processed_sha256.json`.

The unique image inputs are copied byte-for-byte:

| Image | Bytes | SHA-256 |
| --- | ---: | --- |
| `amzn_2023_H2_candlestick.png` | 45042 | `215c01f8a03dc55719558644992b14c28d8b9f604d44e3504ef7df0f99997bb8` |
| `jpm_2023_H2_candlestick.png` | 43796 | `b9e218ab002a8bfd6c0cbf86dd53abe9bcfeda4744a785754cb1b9173cc85611` |

The Qwen manifest contains provenance and later-run references, but the future
runner is instructed to pass only the image and the frozen visual-only prompt.
Qwen was not downloaded or run; captions remain pending.

## Contract equivalence and validation

`docs/m1/m1_preprocessing_contract_equivalence.json` reports:

```text
cases_compared = 78
research_relevant_differences = 0
status = PASS
```

The `--verify-determinism` build created two independent staging trees from the
unchanged raw source and compared research-relevant hashes. The result was
`deterministic_rerun = PASS` with zero mismatched hashes.

The 9 targeted preprocessing tests cover frozen-contract rejection, TEXT boundaries
and deduplication, same-day PIT rejection, TABLE duration/PIT/missing concepts,
61-row time-series selection, image eligibility and pending captions,
four-modality missingness, ZIP-member hashing, and stable research hashes.

The full suite was rerun after preprocessing under Python 3.12 and completed
with zero failures:

```text
771 passed, 2 skipped, 19 warnings, 69 subtests passed in 25.29s
```

Static validation included Ruff, Python compilation, JSON and CSV parsing,
processed-manifest checksum verification, raw ZIP-member and image SHA-256
verification, and Git-ignore checks.

## Research boundary

Raw FinMultiTime was not modified. The processed dataset is Git-ignored and is
not committed. No Agent, Trader, Memory, execution, valuation, metrics, or
Formal M1 behavior was changed or run. No DeepSeek, Qwen, or other paid API
calls were made. No final Evidence Packets were generated, and nothing was
archived to `AlphaMAS-Experiments`. M2 / Agentic RL was not started.
