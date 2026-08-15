# M2-06A Final Evidence Corpus Lineage Reproduction Audit

## Purpose

M2-06 froze the pre-Formal evidence corpus while recording pre-Qwen builder
commit `791c88a7ba927cebe3713d893b8b111c5f54bde5` and final AlphaMAS freeze
commit `3cef79b9536e4714cc5f532eada84f66fd6e8142`. The latter includes final-packet
PIT-audit compatibility fixes that were not committed at the former. M2-06A
therefore tested whether clean `3cef79...`, without reconstructing upstream data
or captions, can reproduce the frozen corpus exactly.

This is a **PRE-FORMAL DEVELOPMENT — M2-06 LINEAGE AUDIT**. It does not modify
the frozen corpus or replace the M2-06 freeze records.

## Starting and frozen lineage

- AlphaMAS start and clean final-corpus reproducer:
  `3cef79b9536e4714cc5f532eada84f66fd6e8142`
- AlphaMAS-Experiments start and frozen final corpus:
  `6b2406f1e12e1988c27b44880a1e153a9b750c2e`
- Original M2-06 pre-Qwen builder:
  `791c88a7ba927cebe3713d893b8b111c5f54bde5`
- Frozen pre-Qwen inputs:
  `a9d3c554a9eb0ecf401a2064d12d5fb901e55165`
- Published Experiments lineage audit:
  `5d79a199e77a94f29faba32d6a4f320d109a0f7c`
- Frozen corpus identity:
  `3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420`

Both canonical repositories passed the mandatory clean-worktree, branch, HEAD,
and remote-ref gates before the audit. The reproduction used detached temporary
worktrees at the exact source, pre-Qwen, and frozen-reference commits.

## Reproduction method

The locked M2 Python environment ran the `final` mode of
`scripts/m2/build_preformal_evidence_corpus.py` from clean `3cef79...`. Its only
research inputs were the archived `preformal_evidence_v1/` tree from
`a9d3c...` and the frozen `captions/` tree from `6b2406...`; no `--raw-root` was
provided. A second invocation used the built-in `--verify-determinism` mode.

Before building, 127 archived pre-Qwen research files were compared between
`a9d3c...` and `6b2406...`, including all `inputs/` files and the required
audits/manifests. Missing files, extra files, and hash drift were all zero. Each
of the 14 frozen captions was also checked against its image SHA256, raw-output
SHA256, canonical-caption SHA256, model repository and revision, prompt SHA256,
and schema SHA256. All checks passed; the M2-06 freeze records zero manual edits.

## Exact comparison results

- Rebuilt corpus identity:
  `3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420`
- Structured packets: 96 compared; 0 missing; 0 extra; 0 hash mismatches.
- Rendered packets: 96 compared; 0 missing; 0 extra; 0 hash mismatches.
- Research files under `packets/`, `captions/`, `inputs/`, `audits/`, and
  `manifests/`: 355 compared; 0 missing; 0 extra; 0 hash mismatches.
- `final_sha256.json`: 353 entries in each archive and byte-identical.
- Generated `README.md`: byte-identical.
- Built-in double generation: PASS with the frozen corpus identity.
- Source worktree after both builds: clean.

The complete per-path size and SHA256 comparison is frozen separately in the
Experiments lineage-audit commit above. The original frozen corpus remains
byte-identical to its `6b2406...` version.

## Nature of the source delta

The three-file delta from `791c88...` to `3cef79...` comprises final-packet PIT
audit compatibility, audit normalization/import-path support, and additional
correctness regression coverage. The exact reproduction demonstrates that this
delta did not alter any frozen research evidence.

## Protection, non-execution, and invariance

Raw FinMultiTime accessed: **NO**

Raw FinMultiTime modified: **NO**

Qwen inference calls: **0**

Qwen model loads: **0**

Qwen captions regenerated: **0**

Qwen captions manually edited: **0**

DeepSeek API calls: **0**

DeepSeek API cost: **¥0**

Agent executions: **0**

AWS EC2 starts: **0**

AWS GPU hours: **0**

New AWS resources: **0**

AWS incremental compute cost: **USD 0**

FINAL_HOLDOUT outcomes, rewards, policy performance, and trading performance
were not inspected. No performance metric or reward distribution was inspected.
The 96-case, COMPACT, STANDARD, and MAXIMUM memberships; packet and caption
bytes; TEXT, TABLE, TIME_SERIES, and IMAGE evidence; PIT rules; Holdout
protection; R1, R2, R3, selected reward; and Formal M1/M2 inputs are unchanged.

Budget tier selected: **NO**

Status: **DEFERRED TO M2-07**

## Historical limitation and verdict

Historical exact working-tree state during original final generation:
**NOT INDEPENDENTLY PROVABLE**. This audit does not rewrite history or claim
that the original run definitely used a clean `3cef79...` checkout.

**PASS — The frozen M2-06 final corpus is byte-identically reproducible from
clean commit `3cef79b9536e4714cc5f532eada84f66fd6e8142` using frozen pre-Qwen
inputs `a9d3c554a9eb0ecf401a2064d12d5fb901e55165` and the frozen Qwen captions
from `6b2406f1e12e1988c27b44880a1e153a9b750c2e`. The frozen corpus remains
authoritative and unchanged.**
