# M2-13 Online Delayed Adaptation Freeze

M2-13 Retry #1 froze the delayed per-symbol online adaptation rule for
`M2-PA-CTPPO-v2`. The first complete correctness-valid TRAIN-only run selected
**O08: online learning rate `1e-3`, two update epochs per matured credit**.

## Frozen lineage and inputs

- Phase-A AlphaMAS SHA: `95709a50ecea79e4d37f515681925f45dba7d77b`
- Phase-A Experiments SHA: `a5602b02e88e015b585b99166f51572a1392df42`
- Global checkpoint: C09, iteration 100, global-training LR `1e-3`
- C09 parameter SHA: `6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841`
- Actor SHA: `5af9a28baaf2dc25687e65a7bf8bbefb047fa1a9664b05257bfbf78d2ac5b14d`
- Critic SHA: `e55f942699d47eeab16d92b8ad3ff7277306239a8d1ba7d542f14080c86d5416`
- C09 `model.pt` SHA-256: `56dc52128e1df9c9ddcf79fa6f7b293393bd61306ba4ef032f96cad6bf92126c`
- Representation: `6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe`
- TRAIN tree: `ff5e1bf21ef90b9eeec96d257c59e83656957f32ec4d2f94396a0a5d4bf1db13`
- Reward: `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY`

## Runtime contract

Each symbol begins from an in-memory copy of C09 and owns an independent set
of 165 fast parameters and an independent persistent AdamW optimiser. The only
online-updatable tensors are `residual_head.weight`, `residual_head.bias`,
`gate_head.weight`, `gate_head.bias`, `value_head.weight`, and
`value_head.bias`. Weight decay is `1e-4`, gradient clipping is `0.5`, and the
deterministic seed is `20260816`.

Credits follow `PENDING → MATURED → APPLIED`. A same-close maturity occurs
after that close's action is permanently issued, so the update can affect only
a later weekly action. Pending records contain origin identities and frozen
`pi_old`, but no R3 vector or future outcome. Terminal credits that cannot
affect another TRAIN action mature only for scoring and are archived without a
post-horizon update.

Canonical safe-resume state includes fast tensors, optimiser moments, pending
and matured credits, applied event identities, issued actions, and the next
expected node/session. Canonical JSON identities make save/load deterministic
and exactly-once application auditable.

## Selection

All nine preregistered candidates were valid and produced 56 decisions. All
nine had primary mean sequential TRAIN R3 `0.0044633290897862065`, worst-symbol
cumulative R3 `0.0`, and 40 applied updates. O01–O07 and O09 had Prompt override
rate `0.25`; O08 had `0.23214285714285715`. Thus the primary and first
worst-symbol tie-breaks tied, and the preregistered lower Prompt-override-rate
tie-break uniquely selected O08. No human override occurred.

The `AUDIT_ONLY` replay compared all 504 records with zero action, score, or
chronology mismatches and reproduced O08. Memorial Day chronology, per-symbol
isolation, global-backbone immutability, and safe resume immediately before and
after a maturity boundary all passed with zero mismatches.

Historical M2-12 VALIDATION selection remains authorised and unchanged. M2-13
made no new read of sealed M2-12 validation-result contents and used no
VALIDATION, FINAL_HOLDOUT, E2E_PILOT, Formal 2024H1, raw FinMultiTime, live
market, model API, or AWS result/resource in selection. Candidate-specific
TRAIN endpoints are not checkpoints and must not initialise M2-14.

