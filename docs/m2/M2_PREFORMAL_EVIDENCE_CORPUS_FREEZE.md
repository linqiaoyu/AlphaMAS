# M2-06 Pre-Formal Evidence Corpus Freeze

## Purpose and lineage

M2-06 freezes the deterministic, point-in-time pre-Formal evidence inputs from which M2-07 may later create semantic Trader states. It is preprocessing only: no Agent, policy, reward, outcome, or DeepSeek execution occurred.

- Starting AlphaMAS: `bc67a5cda93e08b40a97a37fd88feee9e16441b7`
- Builder AlphaMAS: `791c88a7ba927cebe3713d893b8b111c5f54bde5`
- Starting AlphaMAS-Experiments: `3a8e63a9abe89a80787b68aabb9b3523453966fc`
- Pre-Qwen inputs: `a9d3c554a9eb0ecf401a2064d12d5fb901e55165`
- Final evidence archive: `6b2406f1e12e1988c27b44880a1e153a9b750c2e`

The universe is the frozen 96-case MAXIMUM corpus across AAPL, AEMD, AGI, AMZN, ARR, EML, JBSS, and JPM: TRAIN 56, VALIDATION 16, FINAL_HOLDOUT 16, and E2E_PILOT 8.

## Tier identities

The tier identities are frozen subsets of this maximum-first construction:

- COMPACT: 72 cases, `f35488ed0f910f73b11713bb7eadf00191f13f5e5b711b309dc879aca8075d62`
- STANDARD: 80 cases, `f1253ab0ed8e23d9ae5656fc4250d7dca63bd6f9342e2fd9a2d84eac8e377452`
- MAXIMUM: 96 cases, `68fbc780cbb454f78d0ae719f91665737ed6e04928eba91f4c9c0530745d601f`

Budget tier selected: **NO**. Status: **DEFERRED TO M2-07**.

## Evidence rules and source integrity

`/Volumes/Jackson/Dataset/FinMultiTime` was read-only. The final mutation guard rehashed all 52 accessed source members and found no difference. Evidence follows frozen M1 packet version `M1-FINMULTITIME-v1.0.2`, rule identity `46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08`.

The source-local TEXT integrity audit examined 271 candidate records: 175 were safe and 96 were rejected. It used no outcomes, performed no automatic symbol reselection, and used no external replacement. AAPL remains fail-closed TEXT UNAVAILABLE under the M1 erratum; JPM and JBSS are TEXT UNAVAILABLE because their source members are absent. Across cases, TEXT is AVAILABLE 44 / UNAVAILABLE 52.

TABLE selects the frozen six concepts using `filed_date < decision session`, resolves eligible versions/restatements deterministically, preserves source duration semantics, and fails closed rather than deriving or substituting values. Coverage is AVAILABLE 84 / UNAVAILABLE 12.

TIME_SERIES uses completed sessions only (`session <= decision session`), the frozen M1 61-row formulas, and emits summaries rather than raw bars. Coverage is AVAILABLE 96 / UNAVAILABLE 0.

IMAGE uses only the latest completed half-year ending strictly before the decision, with exact image provenance. AAPL IMAGE remains frozen UNAVAILABLE; no replacement was introduced. Coverage is AVAILABLE 84 / UNAVAILABLE 12.

## Qwen caption freeze

- Model: `Qwen/Qwen3-VL-2B-Instruct`
- Revision: `89644892e4d85e24eaac8bacfd4f463576704203`
- Prompt SHA256: `284c6e52763796a47f7d30fd2e44cfe68d9211db6831d89ec5ed436920c34df9`
- Schema SHA256: `bf8f04330ffb1bd8468b9bf01eb96bec6b35bb4bad29c8e3f1ad6c47cf0ca8e4`
- Settings: batch size 1, bfloat16, sampling disabled, one beam, 256 maximum new tokens; Python/torch/CUDA seed 0; cuDNN benchmark off and deterministic on; deterministic algorithms enabled warn-only.
- Unique eligible images: 14; frozen M1 captions reused: 0; new captions: 14; successful calls: 14; case-level repeats: 0; manual edits: 0.
- Historical synthetic compatibility: PASS for image identity, raw output identity, and canonical output identity.

Qwen trained: **NO**  
Qwen fine-tuned: **NO**  
Qwen Agent: **NO**  
Qwen experimental variable: **NO**

## Packet freeze and protection

All 96 structured and rendered packets were produced twice with identical output. Packet bounds passed with zero violations; the largest canonical structured packet is 43,793 bytes. PIT audit: PASS, zero violations. Future-label/outcome exclusion audit: PASS, zero violations. The final corpus identity is:

`3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420`

FINAL_HOLDOUT input evidence was materialised once so preprocessing cannot change after model selection. Its outcome and reward were not inspected, no Agent executed on it, and it remains protected. E2E_PILOT is routing/integration evidence only and was not used for selection or performance inspection.

The pre-Formal 2023 corpus is separate from Formal M2. Formal M2 continues to reuse the exact frozen M1 2024H1 FinMultiTime inputs, Qwen captions, and Evidence Packets; none was regenerated here.

## Cost and M2-07 boundary

The existing AWS A10G instance was reused for 15 minutes 42 seconds. Estimated incremental Qwen compute cost is USD 0.334 at USD 1.277/hour, below the USD 2.50 ceiling. No new EC2, EBS, or S3 resource was created. Persistent idle storage cost remains approximately USD 0.43/day.

DeepSeek calls: **0**. DeepSeek cost: **¥0**.

M2-07 owns DeepSeek semantic-state calibration, measured API cost per case, the affordability decision, and the semantic-state generation plan. M2-07 may choose among COMPACT, STANDARD, and MAXIMUM using measured DeepSeek API cost only. It may not use returns, reward distributions, policy performance, or validation trading performance, and it must use this frozen corpus without evidence reprocessing.

## Verdict

**PASS — M2 pre-Formal 96-case evidence corpus, source-integrity decisions and unique-image Qwen captions are frozen without DeepSeek or outcome leakage; ready for M2-07 semantic-state API-cost calibration and budget-tier selection**
