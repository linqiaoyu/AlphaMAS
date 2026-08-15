# M2 Pre-Formal API Cost and Tier Freeze

## Lineage and boundary

Task `M2-07` started from AlphaMAS
`3ddf043b4bed5cc53197d33a91f652cdb75339d8` and AlphaMAS-Experiments
`5d79a199e77a94f29faba32d6a4f320d109a0f7c`. The frozen 96-case M2-06
corpus identity is
`3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420`;
it was not modified.

- M2-07 runner SHA: `6d3c46b96b8935944e0f232b73020262d870d953`
- Probe-inputs SHA: `2bf79ffce2dedf15e157f48a5e3727eef611d0e9`
- Calibration archive SHA: `123a0b412c91379a1c2279ca9e31a5c19743c40f`
- Contract: `M2-SEMANTIC-HANDOFF-v1`

The unchanged M1 market/social/news/fundamentals analysts, three-round Bull/Bear
research debate, Research Manager, and Prompt Trader ran in historical/PIT mode.
Capture stopped after Prompt Trader. Risk Debate, Portfolio Manager, trading,
reward evaluation, VALIDATION, FINAL_HOLDOUT, and E2E_PILOT were not executed.

## Pre-registered probes

COMPACT contained exactly 32 TRAIN cases. Frozen rendered Evidence Packets were
sorted by `(UTF-8 packet bytes, case_id)` and zero-based positions
`0, 6, 12, 18, 24, 31` were selected before generation.

| Position | Case | Bytes | Packet SHA256 |
|---:|---|---:|---|
| 0 | AAPL:2023-06-09 | 2,858 | `b273de525eb697f1142007b0882f9053b1f78f3624fb0bd81aa845859b60ee4e` |
| 6 | JBSS:2023-06-02 | 3,296 | `6e952e90b820c5f395216ffc2da246c11656091563d6f170c50226e0a1d9d62c` |
| 12 | AEMD:2023-06-02 | 3,569 | `92c535ac40f3416c707182015c34da8d9d89fd932ebf40d4e44c0a41b139954b` |
| 18 | EML:2023-06-09 | 4,778 | `6e3e8408812d4aa07f408b8214482c24b758d290a0fe706f6b9a3c26132d8370` |
| 24 | AGI:2023-06-02 | 11,249 | `96a0fd89e4959e7d475d1c6d805b63995b13387ed4ba08602409c40fce6d6584` |
| 31 | AMZN:2023-06-02 | 13,040 | `feb689864ca1ab89e9f0f15a983e5fa2a827b5f60f1ac8de510f90f8fe41ec7f` |

## Pricing and measured cost

The official DeepSeek CNY pricing page was retrieved at
`2026-08-15T16:51:09Z`. For `deepseek-v4-flash`, prices per one million tokens
were ¥0.02 cache-hit input, ¥1.00 cache-miss input, and ¥2.00 output. If a call
did not expose a complete cache split, all input tokens were conservatively
priced as cache misses. Pricing snapshot SHA256:
`f27fd57a4e63fa415f0e799b57a29b64bfae8a39ef85240ecddb36d88b4be9f3`.

| Case | Calls | Input tokens | Output tokens | Cost CNY |
|---|---:|---:|---:|---:|
| AAPL:2023-06-09 | 18 | 154,028 | 26,971 | 0.16155720 |
| JBSS:2023-06-02 | 18 | 150,453 | 26,225 | 0.15724284 |
| AEMD:2023-06-02 | 19 | 167,587 | 25,869 | 0.16237524 |
| EML:2023-06-09 | 19 | 165,542 | 27,903 | 0.16803600 |
| AGI:2023-06-02 | 18 | 156,339 | 25,811 | 0.15289284 |
| AMZN:2023-06-02 | 18 | 166,990 | 25,228 | 0.15510232 |

All six outputs were first complete correctness-valid trajectories. There were
110 successful LLM requests, zero failed billed attempts, 960,939 input tokens,
158,007 output tokens, and ¥0.95720644 actual spend. Per-case cost was minimum
¥0.15289284, median ¥0.15940002, mean ¥0.1595344066666666666666666667,
and maximum ¥0.16803600. The maximum is frozen as
`GUARD_COST_PER_CASE_CNY`.

## Projection and selection

For a tier of `N` cases:

`PROJECTED_TIER_COST_CNY = ACTUAL_PROBE_SPEND_CNY + (N - 6) × GUARD_COST_PER_CASE_CNY`

`PROJECTED_TIER_WITH_RESERVE_CNY = PROJECTED_TIER_COST_CNY + 4.00`

Actual probe spend includes billed failed-attempt cost, if any, so failed cost
is disclosed separately and never double-counted. Here it is zero.

| Tier | N | Remaining | Semantic cost CNY | Reserve | With reserve | ≤ ¥40 | ≤ ¥50 |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| COMPACT | 72 | 66 | 12.04758244 | 4.00 | 16.04758244 | YES | YES |
| STANDARD | 80 | 74 | 13.39187044 | 4.00 | 17.39187044 | YES | YES |
| MAXIMUM | 96 | 90 | 16.08044644 | 4.00 | 20.08044644 | YES | YES |

Evaluating `MAXIMUM → STANDARD → COMPACT`, MAXIMUM is selected with
`TARGET_BUDGET` status. Its frozen identity is
`68fbc780cbb454f78d0ae719f91665737ed6e04928eba91f4c9c0530745d601f`.
Selection used API cost only. Performance, reward, future price/return, action
quality, VALIDATION results, and FINAL_HOLDOUT outcomes were not used or
inspected.

## M2-08 boundary

M2-08 may generate the remaining selected-tier semantic hand-offs using the
frozen runner, staged according to role protections. It must reuse these six
actor-visible states byte-for-byte and must not regenerate them. FINAL_HOLDOUT
generation remains deferred until its designated later boundary. The semantic
encoder, embedding dimension, Actor, Critic, and PPO-family method remain
deferred.

M2-07 used no AWS, no Qwen, and did not access raw FinMultiTime.
