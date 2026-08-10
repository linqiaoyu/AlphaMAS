# M0 Final Readiness Audit

Audit date: 2026-08-10

Base branch: `baseline-m0`

Base SHA: `352aa9db4ddb425a7974609b823e4f955f75d3ab`

Working branch: `chore/final-m0-readiness-audit`

Scope: M0 infrastructure freeze only

## Executive status

All identified BLOCKER and SHOULD_FIX items are fixed. The required offline test
suites, formal M0 dry-run, snapshot-backed SMA readiness run, and pull-request CI
matrix pass without any real LLM call.

- Real LLM API calls during this task: **0**
- Yahoo or other market-data downloads during final verification: **0**
- Stage 3: **NOT RUN**
- Formal M0: **NOT RUN**
- M1/M2/M3, FinMultiTime, Agentic RL: **NOT RUN / NOT IMPLEMENTED**

At audit start, `baseline-m0` pointed to `352aa9d`; the only pre-existing untracked
workspace item was `.DS_Store`. It was preserved, and `.DS_Store` is now ignored.

## BLOCKER findings and resolutions

| ID | Finding | Resolution | Evidence |
|---|---|---|---|
| B01 | The formal preset did not explicitly control DeepSeek thinking, and no provider-specific switch reached the HTTP request. | Added strict `enabled`/`disabled` validation, environment override, Graph wiring, and DeepSeek-only `extra_body` injection. | `tests/test_deepseek_thinking.py` intercepts the final OpenAI-compatible HTTP JSON with `httpx.MockTransport`. |
| B02 | Formal Agent behavior inherited mutable `DEFAULT_CONFIG` values and therefore was not fully frozen. | The checked-in preset now explicitly supplies every research-affecting Graph field; `validate_formal_m0_config()` exact-matches the whole preset and rejects missing, changed, or extra fields. | `test_formal_m0_contract_is_frozen`, extended-contract mutation tests, and unexpected-field test. |
| B03 | Formal market data could be ambiguous or fall back to Yahoo. | Formal preset is `snapshot`; real execution without `--snapshot-dir` fails before run-directory/provider creation. Unknown source names fail instead of falling through to Yahoo. Formal TradingAgents source overrides are rejected even in dry-run. | Snapshot-policy and no-provider-use tests. |
| B04 | `warmup_sessions=252` was decorative while the runner used an approximate 400-calendar-day offset. | The runner now asks XNYS for exactly the 252 sessions strictly before the first decision. The general engine fallback uses the same exact-session rule. | 252 sessions, `2023-01-04` through `2024-01-04`; first decision `2024-01-05`. |
| B05 | Decision Memory contained an independent default/hard-coded five-day horizon. | Formal `holding_horizon_sessions` resolves once to `memory_holding_horizon_sessions`; maturity, realized return indexing, and reflection all use it. Non-positive and non-integer values fail. | Three-session mock matures at 3 and remains immature at 5. |
| B06 | Process-global dataflow config merged nested dictionaries, so `tool_vendors={}` could retain a stale override from an earlier Graph while resolved config/hash claimed it was empty. | Added explicit replacement semantics; a Graph installs its complete resolved config verbatim. Partial `set_config()` merge behavior remains available to existing callers. | Stale nested-override reset test. |
| B07 | A formal Agent bundle did not contain aggregate case, availability, or raw usage tables, and cache-hit source reports could remain only in runtime cache. | Added canonical per-case artifacts plus `case_index.csv`, `data_availability.csv`, and `llm_usage.csv`. Fresh and cache-hit runs materialize reports and source audits into the immutable run bundle. | Artifact aggregation, cache-hit materialization, fixed-schema, and completeness tests. |
| B08 | Graph hash checks could accept the same arbitrary 64-character string copied into three documents. | The validator now recomputes the research identity from `graph_config.resolved.json`, compares manifest/config/Graph/case/model/cache values, and recomputes every cache key. | Parameterized Graph/case/model/cache tamper tests. |
| B09 | An incomplete cache entry could be treated as a formal hit, silently losing reports, source audit, or request usage. | Formal artifact-producing cache hits require a complete report tree, non-empty source audit, and preserved origin usage; otherwise the Graph reruns. | Parameterized incomplete-cache tests. |
| B10 | A live DeepSeek test could run if a key happened to exist. | The live test now needs both `RUN_LIVE_LLM_TESTS=1` and a non-placeholder key. Final test commands explicitly removed both the opt-in and key. | Full suite reports the live test skipped. |

## SHOULD_FIX findings and resolutions

| ID | Finding | Resolution |
|---|---|---|
| S01 | Usage copied from a cache origin could be misrepresented as a request made by the replay run. | Added `usage_source` and `origin_run_id`. Current bundle identity remains in `run_id`; the paid request's true origin remains explicit. |
| S02 | A HOLD decision used a `current_weight > 0.5` threshold and could flatten an existing position after a large drawdown. | HOLD now preserves the discrete long-only position by quantity. |
| S03 | `--force` was recorded as a normal cache miss. | Forced recomputation is recorded as `cache_status=bypass`. |
| S04 | A partial run using the formal config could be mistaken for formal M0. | `protocol_mode=formal_m0` is emitted only for full TradingAgents + snapshot + no `--max-cases`; SMA and partial checks are `engineering_validation`. |
| S05 | Memory from a changed Graph config could remain under the same experiment namespace. | Experiment memory paths now include `graph_config_sha256`; configuration variants cannot share memory. |
| S06 | Only the core M0 values were exact-validated. Article limits, queries, vendors, benchmark map, retry/checkpoint settings could drift while still passing formal validation. | The complete checked-in M0 JSON is now the exact formal contract. |
| S07 | Synthetic Agent-specific tables could be absent or have unstable headers. | They are always emitted as header-only CSV files with the declared schema. |
| S08 | A bundle validator checked only global non-emptiness for Agent availability/usage. | It now checks unique schedule coverage and per-case parity with canonical source-audit and usage JSON. |

## ACCEPTABLE_LIMITATION

1. Exact raw tool-call evidence capture was not added. It would require a broad Graph/tool
   redesign that could change prompts or reasoning semantics. Immutable reports, structured
   decisions, source audits, snapshots, orders, fills, equity, metrics, and usage are retained.
2. Provider-omitted token fields stay null. Tokens are never estimated.
3. API prices are not embedded in research code or results. Cost can be calculated later from
   raw usage and experiment-time pricing metadata.
4. `temperature=0.0` reduces sampling variation but is not claimed to make a hosted LLM
   bit-for-bit deterministic.
5. The execution and valuation feed is the frozen snapshot. Agent research tools and memory
   outcome resolution can still consult their configured point-in-time-safe vendors; every
   used, blocked, unavailable, or errored source is recorded in the source audit.
6. A partial runtime resume must preserve its decision cache and experiment memory together.
   Graph-hash namespacing prevents cross-config contamination; immutable completed bundles do
   not depend on runtime cache or memory for later analysis.
7. Pre-schema-1.1 historical bundles are not backfilled. The validator applies to new runs.

## Formal M0 configuration freeze

The formal preset resolves to:

| Area | Frozen value |
|---|---|
| Symbols / schedule | `AAPL`, `AMZN`, `JPM`; XNYS; 26 calendar weeks |
| Decisions | `2024-01-05` through `2024-06-28`, including `2024-03-28` |
| Execution / valuation | next XNYS open; first `2024-01-08`; last `2024-07-01`; valuation `2024-07-05` |
| Warmup | exactly 252 prior XNYS sessions, `2023-01-04` through `2024-01-04` |
| Capital / costs | 100,000 initial cash; 5 bps commission; 5 bps slippage; fractional shares |
| Position contract | long-only; no shorting/leverage; BUY 1.0; SELL 0.0; HOLD preserve; no forced liquidation |
| Memory | `experiment`; five trading-session holding horizon; no entry cap |
| Research | Medium; debate rounds 3; risk rounds 3; recurrence limit 100 |
| LLM | DeepSeek; quick/deep `deepseek-v4-flash`; thinking disabled; temperature 0.0; English |
| Analysts | market, social, news, fundamentals |
| Data | execution source `snapshot`; point-in-time true; explicit data/tool vendors |
| Metrics | risk-free rate 0.0; annualization 252 |
| Checkpoint | disabled |
| Seed | absent; no unsupported determinism claim |

`deepseek-chat` and `deepseek-reasoner` are not used by the formal preset.

## DeepSeek request-boundary proof

The mock transport test invokes the real LangChain/OpenAI SDK request-building path and
captures the final outbound JSON. The asserted subset for formal M0 is:

```json
{
  "model": "deepseek-v4-flash",
  "temperature": 0.0,
  "thinking": {"type": "disabled"}
}
```

It also proves:

- `enabled` reaches `thinking.type=enabled`;
- invalid values raise before a request;
- OpenAI, Google, and Anthropic do not receive the DeepSeek field;
- quick and deep clients receive the same thinking and temperature values.

No real DeepSeek endpoint is contacted by these tests.

## Decorative/dead configuration audit

| Field | Effective behavior |
|---|---|
| `warmup_sessions` | exact XNYS calculation, requested input start, resolved config, and data manifest validation |
| `long_only`, `short_selling`, `leverage` | fixed-contract validation; unsupported values fail |
| `buy_target_weight`, `sell_target_weight`, `hold` | fixed-contract validation against implemented 1.0/0.0/preserve mapping |
| `force_liquidate_at_end` | fixed false; unsupported true fails |
| `holding_horizon_sessions` | drives memory maturity, return lookup, and reflection |
| `point_in_time` | fixed true; propagated into run context, Graph config, cache identity, and audits |
| `seed` | removed because no supported provider/local stochastic component consumed it |
| `research_depth` | resolves Medium to 3 and must agree with explicit debate/risk rounds |
| `risk_free_rate`, `annualization` | passed into metric computation |
| `fractional_shares`, cash, commission, slippage | passed into portfolio/broker engine |
| provider/models/thinking/temperature/analysts/language | passed into both Graph clients and recorded in artifacts/cache identity |
| article/query/vendor/benchmark/checkpoint/retry fields | explicit, exact-frozen, resolved into Graph, and included in Graph identity |
| `data_source` | snapshot policy enforced; effective CLI source is reflected in resolved Graph config |

No formal preset field remains decorative.

## Resolved Graph identity

`graph_config.resolved.json` is the sanitized configuration actually passed to
`TradingAgentsGraph`. Secret keys and runtime credentials are recursively removed. Runtime
paths are retained for operational provenance but excluded from the stable research hash.

`graph_config_sha256` is computed from the canonical JSON projection of every declared
research-affecting Graph field. It is stable across run directories and changes when any one
of those fields changes. The value is stored in:

- `manifest.json`
- `config.resolved.json`
- `graph_config.resolved.json`
- every `case_index.csv` row
- every case `model_config.json`
- every case cache identity and therefore its cache key
- the experiment-memory namespace

The offline snapshot readiness run produced the consistent hash:

`07e1fd8251befb360e232e080440726a8f8d55cb3432f0649280586ca176ac95`

## Decision cache identity

The canonical key covers:

- experiment ID, symbol, exact UTC decision time, and strategy ID;
- full portfolio-state hash and visible market-history hash;
- selected analysts;
- provider, quick/deep models, thinking mode, and temperature;
- research depth, debate rounds, and risk rounds;
- output language, data vendors, and tool vendors;
- point-in-time mode;
- memory mode, holding horizon, and memory namespace version;
- complete `graph_config_sha256`;
- Git SHA and prompt/config version.

Changing thinking from disabled to enabled produces a different key and a cache miss.

## Final artifact tree

```text
<run>/
├── manifest.json
├── config.resolved.json
├── graph_config.resolved.json
├── environment.json
├── artifact_schema.json
├── run_status.json
├── schedule.csv
├── inputs/
│   ├── data_manifest.json
│   ├── market_data/<AAPL|AMZN|JPM|SPY>.csv
│   └── corporate_actions/<AAPL|AMZN|JPM|SPY>.csv
├── strategy/
│   ├── <symbol>/{decisions,orders,fills,daily_equity,corporate_action_events,metrics}.{csv|json}
│   ├── combined/*.csv
│   └── cases/<symbol>/<decision_session>/
│       ├── decision.json
│       ├── model_config.json
│       ├── run_context.json
│       ├── cache_identity.json
│       ├── source_audit.json
│       ├── llm_usage.json
│       ├── cached_origin_llm_usage.json  # cache hit only
│       ├── case_metadata.json
│       └── reports/complete_report.md + component reports
├── benchmarks/
│   ├── <symbol>_buy_and_hold/*
│   └── SPY_buy_and_hold/*
├── aggregate/{equal_weight_equity.csv,equal_weight_metrics.json}
├── analysis_ready/
│   ├── equity_curves.csv
│   ├── drawdowns.csv
│   ├── weekly_performance.csv
│   ├── decision_timeline.csv
│   ├── metrics_long.csv
│   ├── action_summary.csv
│   ├── case_index.csv
│   ├── data_availability.csv
│   └── llm_usage.csv
├── validation/validation_report.json
└── failures/failures.jsonl
```

## Analysis-ready Agent schemas

`case_index.csv` (one row per Agent decision):

```text
experiment_id, run_id, case_id, symbol, decision_session, decision_time_utc,
execution_session, action, decision_status, rebalance_status, provider, quick_model,
deep_model, thinking_mode, temperature, research_depth, debate_rounds, risk_rounds,
graph_config_sha256, portfolio_equity_before, portfolio_weight_before, report_path,
source_audit_path, cache_key, cache_status, wall_clock_seconds, prompt_tokens,
prompt_cache_hit_tokens, prompt_cache_miss_tokens, completion_tokens, reasoning_tokens,
total_tokens
```

`data_availability.csv` (long form):

```text
experiment_id, run_id, symbol, decision_session, source_name, capability, status,
requested_start, requested_end, latest_event_time, latest_available_time, reason
```

`llm_usage.csv` (one provider request or explicitly identified cache origin per row):

```text
experiment_id, run_id, case_id, symbol, decision_session, usage_source, origin_run_id,
agent_node, provider, model, thinking_mode, prompt_tokens, prompt_cache_hit_tokens,
prompt_cache_miss_tokens, completion_tokens, reasoning_tokens, total_tokens,
latency_seconds
```

## Artifact completeness validation

`validate_artifact_bundle()` is offline and read-only. It verifies:

- all required root/input/strategy/benchmark/aggregate/analysis/validation/failure files;
- all nine analysis-ready schemas exactly, including header-only synthetic Agent tables;
- every market snapshot SHA256 against `data_manifest.json`;
- the three root Graph hashes and a recomputed hash from resolved Graph content;
- expected unique symbol/session Agent coverage;
- successful/cached decisions and valid actions for a successful Agent run;
- canonical case report/audit paths and complete report trees;
- model, decision metadata, case index, and cache-identity Graph hashes;
- canonical cache-key recomputation;
- per-case source-audit and usage aggregation parity;
- current-vs-origin usage provenance;
- passed validation report and `run_status=success`.

A missing core Agent artifact prevents a run from remaining successful.

## Offline verification evidence

All test commands used Python 3.12 from a fresh editable dev installation. The live-test
opt-in and DeepSeek key were explicitly removed from the environment.

| Check | Result |
|---|---|
| Fresh Python 3.12 `pip install -e ".[dev]"` | passed |
| `pytest -q tests/backtesting` | **104 passed** |
| `pytest -q tests/test_point_in_time_contract.py` | **19 passed** |
| `pytest -q` | **692 passed, 2 skipped, 69 subtests passed** |
| Live DeepSeek test | skipped by explicit dual opt-in guard |
| `ruff check .` | **passed** |
| `git diff --check` | **passed** |
| Formal M0 dry-run | **passed; no provider/Graph/data access** |
| Snapshot-backed SMA readiness run | **success; bundle validator all checks true** |

The SMA readiness run used only the existing Stage 2 snapshot and emitted
`protocol_mode=engineering_validation`. It did not execute TradingAgents. Its local ignored
artifact path was:

`results/backtests/final-readiness-offline-v2/runs/20260810T134645566151Z_352aa9db`

## Formal dry-run evidence

The formal dry-run reported:

- symbols: AAPL, AMZN, JPM;
- decision weeks: 26;
- expected paid Agent cases: 78;
- first / Good-Friday-week / last decisions: 2024-01-05 / 2024-03-28 / 2024-06-28;
- first / last executions: 2024-01-08 / 2024-07-01;
- final valuation: 2024-07-05;
- Medium = debate 3 and risk 3;
- quick = deep = `deepseek-v4-flash`;
- thinking = disabled;
- temperature = 0.0;
- snapshot required, not supplied;
- actual warmup = 252 XNYS sessions;
- `will_execute=false` and `do_not_execute_paid_agent_cases=true`.

## CI

The repository workflow defines:

- Python 3.10, 3.11, 3.12, and 3.13 full test jobs;
- Python 3.12 clean-install smoke;
- strict full-repository ruff.

PR [#14](https://github.com/linqiaoyu/AlphaMAS/pull/14), CI run
[31395117771](https://github.com/linqiaoyu/AlphaMAS/actions/runs/31395117771):

| CI job | Result |
|---|---|
| tests (Python 3.10) | passed |
| tests (Python 3.11) | passed |
| tests (Python 3.12) | passed |
| tests (Python 3.13) | passed |
| clean-install smoke | passed |
| ruff (strict, full repo) | passed |

## Readiness conclusion

All identified BLOCKER and SHOULD_FIX findings are resolved, all required local checks pass,
and the complete PR CI matrix is green.

**M0 INFRASTRUCTURE FROZEN AND READY FOR STAGE 3.**

This is an infrastructure-readiness conclusion only. Stage 3 and formal M0 remain explicitly
not run and require a separate, deliberate execution authorization.
