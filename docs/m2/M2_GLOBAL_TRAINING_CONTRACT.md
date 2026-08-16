# M2-11 Global PA-CTPPO-v2 Training Contract

M2-11 trains the frozen `M2-PA-CTPPO-v2` method on the frozen
`M2_TRAIN_COUNTERFACTUAL_TREE-v2` population. It is an offline, TRAIN-only,
full-tree optimisation stage. It does not alter the production MAS graph or the
later Formal online-adaptation protocol.

## Bound identities

- Representation: `6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe`
- Tree: `ff5e1bf21ef90b9eeec96d257c59e83656957f32ec4d2f94396a0a5d4bf1db13`
- Reward: `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY`
- Population: 8,744 TRAIN nodes and 26,232 local action credits
- Model: 20,197 globally trainable parameters

The trainer reconstructs each Actor observation from its frozen TRAIN semantic
row and frozen portfolio snapshot, verifies the stored observation SHA, and
asserts that the only role present is `TRAIN`. It has no Validation, Final
Holdout, E2E Pilot, or Formal result loader.

## Optimisation chronology

At every outer iteration, the current model is frozen as `pi_old`. The runner
evaluates all 8,744 nodes, propagates exact old-policy occupancy from eight roots,
and computes local counterfactual values and advantages from raw R3. Those
probabilities, occupancies, values, and advantages remain frozen for exactly four
full-batch AdamW epochs. The post-update model must become the next iteration's
old policy. A state-machine audit rejects both within-block refresh and reuse of
the iteration-zero policy.

The surrogate is the exact three-action PPO clipped objective with epsilon 0.20,
weighted by old action probability and old state occupancy. The Critic target is
`V_cf(s)`, with coefficient 0.5. There is no Bellman bootstrap, gamma, GAE,
sampling, minibatching, shuffling, scheduler, warm-up, entropy term, Prompt-KL
term, reward transform, or advantage transform.

## Candidate and replay freeze

One canonical initial checkpoint is generated with seed 20260816 and loaded by
all three LR lineages. The only learning rates are `1e-4`, `3e-4`, and `1e-3`;
the only checkpoints are outer iterations 25, 50, and 100, producing C01–C09.
All three canonical runs finish before audit replay. Replay artifacts are marked
`AUDIT_ONLY` and `NOT_SELECTION_ELIGIBLE`; their canonical parameter SHAs must
match the nine canonical checkpoints exactly.

Every completed outer iteration writes an exact safe-resume checkpoint containing
model and optimizer state, completed boundary, CPU/CUDA RNG state, and all lineage
bindings. Candidate selection is not performed in M2-11 and is deferred to M2-12.
