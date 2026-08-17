# A2 Pre-Formal Freeze

`A2_NO_GLOBAL_PRETRAINING` is frozen as the sibling ablation that removes only C09 global TRAIN-only pretraining from Full M2. It starts from the exact archived M2-11 initial model while retaining O08 delayed per-symbol online adaptation without retuning. The preregistration identity is `e7e2cc8520f9033e57c442b04c882baffd7c51b497bbc9495b71098f64a83b0e`; the Full-M2 source base is `6306ea4ea20cda501c6238db80c34d27bbc16bea`.

## Frozen checkpoint and treatment

The archived checkpoint is `experiments/M2/development/global_training_v1/initial_checkpoint/model.pt`. Its file SHA is `4d65dd2c1563b144aee8e79878171feb1ecd990a8bd8e2c584547eaa3a546c9f` and its canonical complete-parameter SHA is `60a0fec7b69ef2d0576a9c0894be09c377d573585db827c162279fb27483303e`. The initial fast SHA is `49380bb69ea6fb7682fe97a3ef5e1ccf2b2429b749d6f9f9971e40725271fcf9`; its canonical fast SHA is `859d91374903768e82c8d3621ca18ea6e8ba3b0002e9e915a579df13664e27d2`. The canonical non-fast/global SHA is `9b83aa1382d6b41c55981f2475eb0e236248affc5b5004a151a4b01651fa646e`, and the empty initial AdamW state identity is `087df628de120d7e2271432fe99d2888097983c25a3cfc880ec60a51d3b4d623`.

A2 validation fails closed if paired with C09. Full M2 and A1 continue to require C09. Frozen A2 variant support was already present at the Full-M2 base, so `tradingagents/m2/runtime.py` was not changed.

## O08 and online-update proof

O08 remains enabled with learning rate `1e-3`, two epochs per matured credit, AdamW, weight decay `1e-4`, gradient clipping `0.5`, deterministic seed `20260816`, and exactly 165 fast parameters across the residual, gate, and value heads. No setting was retuned.

Three eligible pre-2024 credits were tested. All three followed `PENDING → MATURED → APPLIED`, and all three produced non-zero parameter changes. The fast SHA sequence was:

1. `49380bb69ea6fb7682fe97a3ef5e1ccf2b2429b749d6f9f9971e40725271fcf9`
2. `25e6bd6a6317ceb04816f7e43d671c116668c6c2d6329ea82d80a6ad75d3659a`
3. `b42fc8169e119302798bf7f7dd6fa25a91d1910483ec0dd78737c6eba8c6fcfd`
4. `ae204d1c1f43d59fb1b83c0a3721d207da073864592e5a5ffa7ac90b1ce68b4b`

Only the authorized 165 fast values changed. The global SHA remained `9b83aa13…`, giving zero non-fast mutation. AdamW moments evolved, exactly-once application survived reload, and an AAPL update caused zero AMZN fast, optimizer, credit, or global mutation.

## Chronology, resume, and isolation

Rewards remain invisible until five valid XNYS sessions mature. The same-close action is issued and persisted before maturity processing, so an update can affect only a later decision. Tests cover resume before maturity, at maturity, and immediately after update, duplicate-application rejection, changed-input/torn-publication protection, terminal archival, and per-symbol isolation.

The A2 experiment ID, graph identity, DecisionCache identity, Memory path, RL state root, and run lineage are distinct from Full M2 and A1. A2-02 must create fresh dynamic state; only immutable frozen inputs and checkpoint bytes may be reused.

## Preserved controls

The M1 bundle (`30596a54…`), representation (`6e3b1186…`), exact Qwen encoder/revision, 3080-dimensional observation, `M2-PA-CTPPO-v2`, R3 reward, Prompt prior, Actor/Critic architecture, Risk Debate, Portfolio Manager authority, execution, metrics, market snapshots, schedule, LLM/MAS settings, and all Formal controls remain unchanged.

The Formal config SHA is `d8982296be9d632be996d23836cfebb7fb2fb363b6fd496590056025251a07da`. Its machine diff contains exactly four approved fields and no unapproved differences.

## Verification and resource boundary

Focused A2 tests: 10 passed. M2 plus backtesting regression: 462 passed. Complete repository regression: 1129 passed, 6 skipped, 19 warnings, and 69 subtests passed; skips cover an optional AWS package, an opt-in live LLM test, and unavailable historical M1 pilot inputs, none of which is an A2 gate.

DeepSeek calls, Qwen calls, new embeddings, AWS starts, GPU-hours, Formal A2 decisions, global training work, and paid cost were all zero. Neither M2 nor A1 Formal performance was used for design or tuning.

> A2-02 must use the first complete correctness-valid Formal trajectory as the sole official A2 result.
