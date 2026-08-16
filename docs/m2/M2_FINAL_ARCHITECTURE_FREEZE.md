# M2 Final Architecture Freeze

## Freeze status

This is the final M2 production architecture frozen in M2-14 before any E2E_PILOT DeepSeek call or performance inspection. Its canonical identity is `35c5a46616f654ce70b0badc01cd59fb0afd433dcdcd99e5e0b8f2419ec4d153` (`M2-FINAL-ARCHITECTURE-v1`). Poor pilot performance cannot change it.

## Production flow

The complete unchanged M1 pipeline runs through the Prompt Trader. A deterministic local node then constructs the frozen M2-09 semantic state from the verbatim Research Manager plan and complete Prompt Trader proposal, appends the actual four-feature PIT portfolio state, executes the C09 Actor with the current symbol's isolated fast state, and produces the sole authoritative Trader handoff. The unchanged Risk Debate, Portfolio Manager, execution engine, costs, and next-session-open backtester then continue normally.

`Analysts → Research Debate → Research Manager → Prompt Trader → M2 semantic state → C09 + symbol fast state → authoritative M2 Trader handoff → Risk Debate → Portfolio Manager → execution`

The Prompt Trader remains the only generative Trader. Its original proposal is byte-preserved as non-authoritative provenance. The M2 node makes no LLM call and performs no explanatory rewrite. The authoritative parser reads only `M2 AUTHORITATIVE TRADER ACTION`; the old Prompt action inside the preserved provenance cannot become authoritative.

## Frozen semantic and policy identities

- M1 Formal input bundle: `30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121`.
- M2-08 semantic handoff corpus: `cc1fea692489a5b5791cae1fd96386bfbcf1f0a07f2ea3e22637dd3c43533a43`.
- Encoder: `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, float32 eager, batch size one, frozen and untrained.
- Representation: `6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe`; 3,076 semantic features plus `is_cash`, `is_long`, `entry_log_return`, and `current_drawdown`, for 3,080 Actor inputs.
- Method: `M2-PA-CTPPO-v2`; TRAIN tree `ff5e1bf21ef90b9eeec96d257c59e83656957f32ec4d2f94396a0a5d4bf1db13`; reward `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY`.
- C09 parameters: `6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841`; model file `56dc52128e1df9c9ddcf79fa6f7b293393bd61306ba4ef032f96cad6bf92126c`.
- Actor/Critic: 20,197 parameters; shared 1,024→16 semantic adapter, independent 56→32 trunks, bounded three-action residual plus gate Actor heads, and scalar value head. The frozen Prompt prior is 2/3 on the Prompt action and 1/6 on each alternative.

## Online state and chronology

O08 is frozen at AdamW LR `1e-3`, two epochs per matured credit, weight decay `1e-4`, gradient clipping `0.5`, and seed `20260816`. Only the six residual/gate/value head tensors—165 scalar parameters per symbol—may update. The global backbone is verified unchanged around each update.

Every symbol owns an atomic canonical-JSON checkpoint containing fast tensors, persistent AdamW moments, all credit records, applied event IDs, issued actions, next expected decision, and full architecture/checkpoint lineage. A valid same-lineage checkpoint is reused; it is never silently replaced with C09. Symbol paths and optimiser states are isolated.

Credits obey `PENDING → MATURED → APPLIED`; terminal score-only credits become `ARCHIVED`. A pending credit holds only origin-time state. On a shared decision/maturity close, the current action is issued and durably saved before the older credit matures or applies. Exactly-once event IDs prohibit duplicates, and terminal maturation never performs a useless post-horizon update.

## Unchanged system boundary

M0/M1 evidence rules, analysts, Research Manager, Prompt Trader prompt, debate rounds, Risk Debate, Portfolio Manager, Memory semantics, execution, commission, slippage, metrics, schedule, and Formal LLM configuration remain unchanged. The RL Actor is neither a risk layer nor a portfolio executor. Formal remains long-only, unlevered, and next-session-open.

The machine-readable binding is `docs/m2/m2_final_architecture_freeze.json`. Architecture changes after this point require a separately documented demonstrated correctness bug; E2E return or action quality is never such a bug.
