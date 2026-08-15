# M2 Semantic Hand-off TRAIN+VALIDATION Freeze

## Status and purpose

Task `M2-08` materialised the pre-Formal development semantic-state corpus for
later Agentic RL design. It froze information states only: 56 TRAIN and 16
VALIDATION hand-offs under `M2-SEMANTIC-HANDOFF-v1`. It did not select or run a
semantic encoder, Actor, Critic, or PPO-family method, and it did not inspect
reward, future return, portfolio outcome, or policy performance.

## Lineage

- Starting AlphaMAS SHA: `0a93008da45e0b2c066827ba423f226eae3cc793`.
- Starting AlphaMAS-Experiments SHA: `123a0b412c91379a1c2279ca9e31a5c19743c40f`.
- Frozen M2-07 generation primitive SHA: `6d3c46b96b8935944e0f232b73020262d870d953`.
- Frozen runner byte SHA256: `f4aacbfaf6accc34632012bcacc330578518c9d2e6149dd4eee1d71d6c40d5ef`.
- M2-08 orchestrator SHA: `d225316aa622fadbac116698c6deb19b4e19600c`.
- M2-08 plan SHA: `b4f3d22978048d079600752981694a5a7d0041ef`.
- M2-08 corpus SHA: `d7ba1a3d8a77ede15be82c1b700abd5092066526`.
- Frozen M2-06 Evidence Corpus identity:
  `3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420`.
- Selected tier: `MAXIMUM`, 96 cases, identity
  `68fbc780cbb454f78d0ae719f91665737ed6e04928eba91f4c9c0530745d601f`.

The M2-07 runner, `tradingagents/`, Prompt Trader, Research Manager, analysts,
semantic contract, selected reward, and formal protocols were unchanged.

## Population and protected boundary

The generation plan was derived only from frozen MAXIMUM membership, case role,
Evidence Packet identity, and the six reusable M2-07 identities. Canonical
generation order was TRAIN then VALIDATION, preserving manifest order within
each role.

| Role | Materialised | Reused M2-07 | Generated M2-08 | Deferred |
|---|---:|---:|---:|---:|
| TRAIN | 56 | 6 | 50 | 0 |
| VALIDATION | 16 | 0 | 16 | 0 |
| FINAL_HOLDOUT | 0 | 0 | 0 | 16 |
| E2E_PILOT | 0 | 0 | 0 | 8 |

The six reused TRAIN cases were copied byte-for-byte without provenance edits:

- `AAPL:2023-06-09`
- `JBSS:2023-06-02`
- `AEMD:2023-06-02`
- `EML:2023-06-09`
- `AGI:2023-06-02`
- `AMZN:2023-06-02`

No reusable case was regenerated. All 50 remaining TRAIN cases completed before
the first VALIDATION call. All 66 newly paid cases use their first complete
correctness-valid trajectory; no retry, alternative-output selection, or
overwrite occurred.

FINAL_HOLDOUT DeepSeek calls were zero. FINAL_HOLDOUT outcomes and rewards were
not inspected. E2E_PILOT DeepSeek calls were zero. The plan retains all 24
protected cases as deferred MAXIMUM members, but the archive contains no state
files for either protected role.

## Frozen semantic path and model configuration

The unchanged path was Analysts (market, social, news, fundamentals),
three-round Bull/Bear research debate, Research Manager, Prompt Trader, then
stop. Each `upstream_trace.json` is provenance-only with `actor_visible=false`.
Risk Debate, Portfolio Manager, Broker, and trading execution did not run.

- Provider/model: `deepseek` / `deepseek-v4-flash` for quick and deep calls.
- Thinking: disabled.
- Temperature: 0.
- Research depth: Medium.
- Research debate rounds: 3.
- Output language: English.

Temperature zero was not treated as deterministic. The first complete
correctness-valid trajectory is authoritative.

## Pricing and no-cache preflight

Official DeepSeek CNY pricing was retrieved on `2026-08-15T17:46:20Z`.
Per one million tokens, `deepseek-v4-flash` remained ¥0.02 cache-hit input,
¥1.00 cache-miss input, and ¥2.00 output. Pricing drift from M2-07 was `NO`.
Pricing snapshot SHA256:
`f40dc2a3671114cb1264485c5b71ae07fa355fb427b13ef052456b51f2653286`.

Pricing all input tokens as cache misses for each M2-07 probe produced a maximum
case guard of ¥0.221348. Therefore:

- 66-case no-cache projection: ¥14.608968.
- M2-07 plus projected M2-08: ¥15.56617444.
- Projection plus ¥4 reserve: ¥19.56617444.
- Projection plus reserve at or below ¥40: YES.
- Projected M2-08 spend at or below ¥18: YES.
- Projected M2-07 plus M2-08 spend at or below ¥19: YES.

## Actual API use and cost

- New correctness-valid cases: 66.
- DeepSeek LLM requests: 1,241.
- Failed billed attempts: 0.
- Input tokens: 11,308,914.
- Output tokens: 1,816,143.
- M2-08 actual spend: ¥10.99448128.
- M2-07 actual spend: ¥0.95720644.
- Cumulative M2-07 plus M2-08 spend: ¥11.95168772.
- Maximum M2-08 case cost: ¥0.20778572.

M2-08 remained below its ¥15 target and ¥18 hard ceiling. Cumulative task spend
remained below ¥19, the global pre-Formal target remains ¥40, and the absolute
global ceiling remains ¥50.

## Corpus audit and identity

Every actor-visible state has the correct case identity, role, Evidence Packet
SHA, Research Manager plan and SHA, complete Prompt Trader proposal and SHA,
normalized BUY/HOLD/SELL action, and allowed historical context. Structured
forbidden actor fields and outcome metadata violations were zero.

Two independent audits of frozen bytes agreed on case membership, roles, all
actor-state hashes, and corpus identity. Secret scanning passed. No reward table,
future return, benchmark, or policy metric is present in this archive.

Semantic TRAIN+VALIDATION corpus identity:
`cc1fea692489a5b5791cae1fd96386bfbcf1f0a07f2ea3e22637dd3c43533a43`.

The committed archive is
`experiments/M2/development/semantic_handoff_trainval_v1/` at
AlphaMAS-Experiments SHA `d7ba1a3d8a77ede15be82c1b700abd5092066526`.

## Regression gate

With explicit AlphaMAS-Experiments, formal M1 input, and pilot M1 input roots,
the post-freeze regression gate passed without skips: M1 runtime evidence 23,
M1 Formal 21, PIT 19, M2 134, and backtesting 205 tests.

## Resource and research invariance

AWS EC2 starts and GPU hours were zero; incremental AWS compute cost was USD 0.
Qwen calls, model loads, and regenerated captions were zero. Raw FinMultiTime was
not accessed or modified. The M2-06 Evidence Corpus and six M2-07 states were not
changed. R1, R2, R3, the selected reward, MAXIMUM membership, semantic contract,
Prompt Trader, Research Manager, analysts, and Formal M1/M2 inputs and protocol
were unchanged.

## M2-09 boundary

The semantic encoder, embedding dimension, Actor, Critic, and PPO-family method
remain deferred. M2-09 may transform the frozen 72-state TRAIN+VALIDATION
semantic corpus into a compact learnable representation. It must not regenerate
these DeepSeek trajectories and must not use FINAL_HOLDOUT. M2-09 is not started
by this freeze.
