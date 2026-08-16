# M2 Final Pre-Formal Environment Freeze

Status: **FROZEN FOR M2-15**
Freeze time: `2026-08-16T14:30:17Z`

M2-14 is complete under the corrected Trader/Portfolio authority semantics.
The eight existing Retry #3 trajectories are retained as the first complete
correctness-valid E2E population.  No Agent case was rerun and no performance
result was used to select, replace, or alter a trajectory.

## Frozen authority boundary

`Prompt Trader → M2 RL Trader policy → authoritative M2 Trader handoff → unchanged Risk Debate → unchanged Portfolio Manager → unchanged execution/backtester`

- M2 Actor direct portfolio executor: **NO**
- Portfolio Manager remains final policy decision: **YES**
- Portfolio Manager may differ from M2 Trader action: **YES**
- Corrected `AUDIT_ONLY` structural mismatches: **0**
- Correctness-valid cases: **8/8**

ARR and AEMD are correctness-valid.  Their M2 overrides remained authoritative
at the Trader handoff; the unchanged Portfolio Manager was entitled to produce a
different portfolio-level decision, and execution followed the Portfolio
Manager's deterministic rating/action mapping.

## Frozen research identities

- Architecture: `35c5a46616f654ce70b0badc01cd59fb0afd433dcdcd99e5e0b8f2419ec4d153`
- M2-06 evidence: `3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420`
- M2-08 semantic corpus: `cc1fea692489a5b5791cae1fd96386bfbcf1f0a07f2ea3e22637dd3c43533a43`
- M2-09 representation: `6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe`
- Reward: `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY`
- M2-10A tree: `ff5e1bf21ef90b9eeec96d257c59e83656957f32ec4d2f94396a0a5d4bf1db13`
- C09 parameters: `6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841`
- C09 actor: `5af9a28baaf2dc25687e65a7bf8bbefb047fa1a9664b05257bfbf78d2ac5b14d`
- C09 critic: `e55f942699d47eeab16d92b8ad3ff7277306239a8d1ba7d542f14080c86d5416`
- C09 model: `56dc52128e1df9c9ddcf79fa6f7b293393bd61306ba4ef032f96cad6bf92126c`
- O08: AdamW, LR `1e-3`, epochs `2`, weight decay `1e-4`, clip `0.5`, seed `20260816`, `165` fast parameters per symbol
- A1/A2 preregistration: `e7e2cc8520f9033e57c442b04c882baffd7c51b497bbc9495b71098f64a83b0e`

## Existing E2E evidence

- Run: `20260816T131429511226Z_1accde90`
- Memory lineage: `20260816T122459032810Z_1accde90`
- Trajectories reused: `8`
- Trajectories regenerated during authority correction: `0`
- Corrected audit: `retry3_corrected_authority_audit.json`, SHA-256
  `69e23595ea3a3d7c9bcc6a94e4dc1e1b33de5b179ed3e871b5cf2badfd8cd101`
- Historical blocked audit retained unchanged: SHA-256
  `72507326754207b95e565e28cd0fb6b1a1adc4beda4c405806b6d502cf3cacb0`

The Retry #3 raw trajectory bundle remains on stopped EBS.  This resolution did
not reopen it because AWS starts were forbidden.  The corrected replay uses the
already-archived Retry #3 observations and validation records together with the
unchanged deterministic execution-source control flow.  It does not invent
missing token or cost values.

All eight terminal credits were recorded once and archived without a
post-horizon update; duplicate, premature, cross-symbol, and global-backbone
mutations are zero.  Safe resume preserved three completed cache hits, safely
rolled back the unpublished JBSS partial state, and replaced zero completed
cases.

## Protected and resource gates

FINAL_HOLDOUT Agent executions and DeepSeek calls are zero; outcomes and rewards
were not inspected.  Formal 2024H1 Agent executions are zero and results were
not inspected.  Raw FinMultiTime was not accessed.

This resolution used zero DeepSeek calls, zero Qwen calls, zero AWS starts, zero
GPU runtime, zero live market-data calls, and USD `0` incremental AWS cost.  The
recorded cumulative M2-14 AWS estimate remains USD `2.971508`.  EC2 must remain
`STOPPED`, with zero running AlphaMAS g5.xlarge instances.

## Research invariance

M0, M1, M2-06, M2-08, M2-09, R3, M2-10A, C09, O08, Prompt Trader,
Research Manager, Risk Debate, Portfolio Manager, execution mapping,
backtester, A1, A2, Formal protocol, E2E population, and architecture identity
are unchanged.  The sole change is the corrected audit interpretation of
Trader-slot authority versus downstream Portfolio Manager execution authority.
