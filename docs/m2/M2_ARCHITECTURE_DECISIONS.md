# M2 Architecture Decision Matrix

**Task:** M2-01
**Status:** Contract freeze for progression to M2-02
**Starting SHA:** `ac0d1b006d8019748702fda38399a4316befb9b0`

This matrix is authoritative for subsequent M2 tasks. A later task must not silently convert a `DEFERRED`, `FORBIDDEN`, or `REQUIRES LATER VALIDATION` item into a frozen research choice.

## FROZEN

| Decision | Contract |
|---|---|
| Research comparison | M2 is compared with frozen M1; the intended difference is Agentic RL applied only at the Trader hand-off. |
| Formal Prompt Trader | DeepSeek V4 Flash remains unchanged, untrained, and explicitly observable. |
| RL placement | External policy boundary immediately after the Prompt Trader and before the existing risk debate. |
| Policy family | Compact external semantic encoder + proposal-anchored discrete Actor + value/Critic boundary; no local generative LLM as principal Trader. |
| Prompt prior | Enabled conceptually: the Prompt Trader proposal/action is an explicit prior/anchor, and the Actor may follow, retain, or override it. |
| Action space | Exactly BUY, HOLD, SELL. No continuous sizing, leverage, shorting, or Actor-side execution. |
| Execution mapping | BUY → 1.0 target weight; SELL → 0.0; HOLD → preserve; next valid XNYS open; existing costs and no forced liquidation unchanged. |
| Downstream authority | Existing risk debate and Portfolio Manager remain the downstream decision path; PM remains final before execution. |
| Observation classes | Frozen M1 semantic state plus permitted endogenous portfolio control state only. No new Formal market information. |
| Portfolio-state interpretation | Portfolio state is an M2-only endogenous control-state interface, not an additional market-information source. |
| Formal online boundary | If enabled, only a fast-adaptation component may update; global backbone/checkpoint is frozen and each symbol starts from identical initial adapter parameters. |
| Symbol isolation | AAPL, AMZN, and JPM have independent policy state, reward/update state, ledgers, and lineage. No cross-symbol update or future leakage. |
| Reward maturity boundary | Five valid XNYS outcome sessions must be observable before a reward can mature; terminal Formal rewards may remain pending. |
| Memory separation | Existing prose TradingMemory and numeric RL experience/policy state are logically distinct, though future execution must commit them at one chronological transaction boundary. |
| Experience lifecycle | `PENDING → MATURED → APPLIED`, with durable update event IDs and exactly-once application. |
| Provenance | Preserve distinct Prompt Trader, RL Trader, PM final, and executed action fields. |
| Formal information identities | M1 FinMultiTime version, contract hash, packet manifest hash, bundle identity, archive commit, and packet text remain exact. |
| Formal execution identities | M1/M0 market inputs, XNYS schedule, costs, target-weight semantics, daily valuation, and final valuation remain exact. |
| Training boundary | Architecture development and all training/selection/validation use data strictly before `2024-01-01`; Formal 2024H1 is test-only. |
| DeepSeek budget for M2-01 | Zero calls and `¥0`; no runtime API or GPU activity. |
| Failure/resume direction | Prefer an append-only/event-sourced RL ledger with atomic commit semantics and state/ledger prefix identities. |
| M2-01 scope | Documentation only; no runtime RL code, Formal M2 config, training, experiment, checkpoint, or artifact. |

## DEFERRED

| Decision | Owner/boundary |
|---|---|
| Final reward formula | M2-03/M2-04. |
| Training universe | M2-02. |
| Train/validation/holdout split | M2-02. |
| Exact semantic encoder model | Later representation/resource evaluation. |
| Exact feature representation and normalisation | Later state-contract implementation/validation. |
| Exact Actor/Critic dimensions and network details | Later implementation/validation. |
| PPO-family hyperparameters | Later development/validation. |
| Final scalar vs distributional Critic | Later pre-Formal validation only. |
| Fast-adaptation enablement | Later pre-Formal validation; isolation boundary is already frozen. |
| A1/A2 mechanisms | Only after Full M2 is research-frozen. |

## FORBIDDEN

| Prohibited change | Reason |
|---|---|
| Fine-tune DeepSeek or replace it with Qwen/Llama/another generative LLM in Formal M2 | Violates the M1-controlled comparison and Formal LLM contract. |
| Replace the Prompt Trader with the RL Actor | The Prompt Trader must remain the explicit observable proposal prior. |
| Add raw market history, new indicators, future prices, or post-decision outcomes to the Formal RL/Trader observation | Violates the frozen information environment and PIT contract. |
| Change BUY/HOLD/SELL, target weights, long-only, no leverage, or no forced liquidation | Violates the frozen execution protocol. |
| Modify M1 Evidence Packets, captions, lookback, preprocessing, routing, or unavailable-evidence policy | Violates frozen M1 FinMultiTime identities. |
| Use Formal 2024H1 outcomes or M1 observed results to tune reward, architecture, encoder, hyperparameters, or Critic choice | Makes M2 test leakage. |
| Let one symbol’s reward update another symbol’s policy or adapter | Violates Formal online isolation. |
| Apply a reward before its full five-session outcome horizon is visible | Violates PIT safety. |
| Treat a cache hit as permission to skip RL state reconstruction or reward application | Breaks exactly-once transitions and reproducible resume. |
| Double-apply a reward or retain a partial RL update after a failed case | Breaks transaction and rollback semantics. |
| Write PPO/Actor/Critic runtime code, RL storage, or Formal M2 config in M2-01 | Explicit task boundary; later tasks own implementation. |
| Call DeepSeek, run Qwen, start EC2, allocate GPUs, or modify AlphaMAS-Experiments | M2-01 cost and scope boundary. |
| Modify `baseline-m1`, `baseline-m0`, M1 configs, M1 runtime, backtester runtime, Memory runtime, or Formal inputs | Preserves the frozen source and controlled comparison. |

## REQUIRES LATER VALIDATION

| Candidate | Required validation constraint |
|---|---|
| Scalar Critic | Compare only on pre-2024 development/validation/holdout data. |
| Distributional Critic | Compare only on pre-2024 data; it is not assumed superior. |
| Prior implementation | Validate follow/retain/override behaviour without discarding Prompt Trader provenance. |
| Encoder representation | Validate resource use, state sufficiency, PIT safety, and reproducibility before Formal. |
| Fast-adaptation head | Validate delayed updates, per-symbol isolation, rollback, and exact-once semantics before enabling Formal updates. |
| PPO-family optimisation | Validate locally with pre-2024 data; PPO is an implementation family, not a frozen novelty claim. |
| Reward implementation | Must implement the later M2-03/M2-04 reward specification and full maturity/visibility audit. |
| Cache/resume integration | Must reconstruct policy state, ledger prefix, verbal Memory prefix, and update events on replay. |
| Artifact extension | Must preserve all M1 provenance and add four-way action provenance plus RL state/update lineage. |
| A1/A2 | Decide only after Full M2 architecture and Formal controls are research-frozen. |
