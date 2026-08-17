# A1 Pre-Formal Freeze

## Scientific purpose and treatment

`A1_NO_ONLINE_ADAPTATION` isolates the causal contribution of delayed per-symbol online fast-parameter adaptation. It is exactly Full frozen M2 with that mechanism disabled. The preregistration identity is `e7e2cc8520f9033e57c442b04c882baffd7c51b497bbc9495b71098f64a83b0e`; the exact Full-M2 starting commit is `6306ea4ea20cda501c6238db80c34d27bbc16bea`.

Matured credits remain PIT-safe, are scored using frozen R3, and are auditable. In A1 they transition `PENDING -> MATURED -> ARCHIVED` exactly once with immutable reason `ONLINE_UPDATE_DISABLED_BY_ABLATION`. A guarded `_apply()` rejects A1 directly; `optimizer.step()` is never called; no alternative adaptation exists.

## Frozen controls

- M1 evidence bundle: `30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121`; exact frozen captions, evidence contract, PIT routing, and unavailable-evidence behavior are unchanged.
- Representation: `6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe`; `Qwen/Qwen3-Embedding-0.6B` at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; observation dimension 3080.
- C09: parameter SHA `6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841`; model file SHA `56dc52128e1df9c9ddcf79fa6f7b293393bd61306ba4ef032f96cad6bf92126c`.
- RL/reward: `M2-PA-CTPPO-v2` and `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY`; prompt prior, residual bound, gate, Actor, Critic, deterministic action, and 165-fast-parameter boundary are unchanged.
- O08 provenance remains visible: learning rate `1e-3`, 2 epochs, AdamW, weight decay `1e-4`, clip `0.5`. These settings cannot produce an A1 update.
- Downstream authority remains Prompt Trader → frozen Actor → Risk Debate → Portfolio Manager → execution. Risk and Portfolio Manager are unchanged; Portfolio Manager remains final authority.
- Backtester, execution, daily valuation, metrics, Memory algorithm, market provider, reward, evidence, and all M0/M1/M2 controls are unchanged.

## Formal configuration freeze

The mechanically derived config is `configs/backtest_a1_no_online_adaptation_2024h1.json`, SHA-256 `503fb7c3294f2a1e25154d30e249f3d58241d2b29967cce09275c3b89d5b5498`. Relative to frozen Full M2, only `experiment_id_template` (`ADMINISTRATIVE_IDENTITY`) and `m2_variant` (`PREREGISTERED_A1_TREATMENT`) differ. The machine audit is `docs/a1/a1_formal_config_diff.json`; it reports zero unapproved differences.

The frozen protocol remains AAPL/AMZN/JPM, 2024H1, XNYS, 26 weekly decisions per symbol, 78 total, 252-session warm-up, $100,000 per account, next-valid-session open execution, 5 bps commission, 5 bps slippage, fractional long-only positions without leverage, and final valuation 2024-07-05 without forced liquidation. DeepSeek V4 Flash, disabled thinking, temperature zero, Medium/3/3, four analysts, and English output remain configured but were not invoked in A1-01.

Frozen snapshot identities remain AAPL `5428fc2c672f3b68c7c3e83b4a22bd5b7330c95a8b4194695762539d9d8a5af3`, AMZN `c4b5c747d75ba658c6f6833348783e3f8a8c571380c930de20cf9fb7dd6b1444`, JPM `74cf77b77b0a83ce8e6246578d4da30bf7622558e8973bda71344b99b9dfd6fc`, and SPY `22e6996ebf963787f40d54bfc59e1ca088fa698cb82b639768504dbdbb2d25ac`. A1-02 must execute snapshot-only.

## Correctness evidence

All evidence used deterministic pre-2024 local CPU fixtures. Initial and final complete-policy SHA equal C09. Initial A1 fast-parameter SHA is `365d06abf2d30f9ef3c283fb383712b3bc1b7439851e3c828f25e9dd676df2ec`; the non-fast/global parameter SHA is `35dd8185eb04d9662531b81c571ba6df964805b8fa4af9281e414b0c1a3429bc`. Before and after every tested A1 maturity, fast, global, complete-policy, and optimiser-state identities are equal. Matured credits tested: 4. Parameter-mutating updates: 0. Global mutations: 0.

Before the first Full-M2 update boundary, A1 and Full M2 produced identical logits, probabilities, authoritative action, and C09 fast identity. Runtime identities differed only by treatment namespace. Existing authority tests confirm unchanged Risk/PM routing. Post-update equality is not required.

Chronology passed: no reward was visible before five valid XNYS sessions; the same-close action was durably issued before maturity processing; credits were archived exactly once; terminal boundary behavior was preserved. Resume passed before maturity, at maturity, immediately after maturity, and at the changed-input/torn-publication guard. The next decision remained deterministic.

Cache isolation passed: A1 and Full M2 treatment/runtime identities generate different keys, and an M2 cache entry is not an A1 hit. Memory isolation passed: the A1 experiment ID, graph hash, experiment root, and lineage generate a fresh namespace. M2 Memory, pending credits, issued actions, optimiser state, runtime state, and DecisionCache are not imported. Cross-symbol mutation tests passed.

Focused tests: 34 passed. Full regression: 1112 passed, 22 skipped, 69 subtests passed, 0 failures. Skips were pre-existing and explicit: optional `langchain_aws` unavailable, the opt-in live DeepSeek test disabled, and documented legacy M1/pilot archives unavailable at their expected paths. No fixture was fabricated.

## Resource and research guard

DeepSeek calls: 0. Qwen3-VL calls: 0. Qwen embedding generation: 0. Formal Agent decisions: 0. AWS starts: 0. GPU-hours: 0. Paid cost: 0.

Formal 2024H1 performance, actions, rewards, portfolio paths, and metrics were not accessed or used. No tuning, parameter selection, or new hyperparameter occurred. A1-01 did not run Formal A1. A1-02 must use the first complete correctness-valid Formal run as the sole official A1 trajectory.

## Source diff inventory

- `tradingagents/m2/runtime.py`: A1 variant/state provenance, fail-closed update guard, and cryptographic archive immutability evidence.
- `tradingagents/backtesting/config.py`: dedicated exact A1 Formal contract and validation.
- `scripts/run_weekly_backtest.py`: recognition of the dedicated config by the existing Formal safety gates.
- `configs/backtest_a1_no_online_adaptation_2024h1.json`: mechanically derived Formal configuration.
- `tests/m2/test_a1_no_online_adaptation.py`: focused pre-2024 correctness gates.
- `docs/a1/*`: treatment/config/freeze documentation only.

No Prompt Trader, Research Manager, Risk, Portfolio Manager, FinMultiTime, Memory algorithm, market provider, execution, metrics, reward, C09, representation, or O08 hyperparameter implementation file changed.

The machine-readable companion is `docs/a1/a1_preformal_freeze.json`. The source commit containing this contract is the canonical `A1_PREFORMAL_SOURCE_SHA` recorded after commit and in the independent AlphaMAS-Experiments archive.
