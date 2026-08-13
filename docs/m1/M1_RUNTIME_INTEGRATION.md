# M1 Runtime Integration

## Scope and research boundary

This change integrates the already-frozen FinMultiTime Evidence Packets into
the TradingAgents analyst runtime. It is an additive M1 runtime capability:

```text
M1 = M0 historical-safe evidence + frozen FinMultiTime evidence
```

The integration does not run an Agent, DeepSeek, Qwen, a paid API, Formal M1,
Memory updates, a backtest, or a trading decision. `M1 runtime research-freeze`
remains `NOT YET` and `Formal M1` remains `NOT RUN`.

## Frozen input identity

| Identity | Value |
|---|---|
| Archive source | `AlphaMAS-Experiments/experiments/M1/inputs/` |
| Verified archive commit | `3750fa50224ba46ab1d4bf5511cb5e8fa514445b` |
| Contract | `M1-FINMULTITIME-v1.0.2` |
| Contract SHA-256 | `46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08` |
| Packet manifest SHA-256 | `05f10129b430475bad3d5dee9dfceffba463f8a99cdd3f087ff4b755a869d63d` |
| Computed input bundle identity | `30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121` |
| Final Evidence Packets | `78` |
| `input_bundle_frozen` | `true` |
| `formal_m1_run` | `false` |

`FrozenFinMultiTimeEvidenceStore` verifies the manifest chain, contract,
checksum inventory, 78-case structure, every packet JSON SHA, and every packet
text representation before an enabled graph is built. On access it rechecks
the exact packet SHA, `FINAL_FROZEN` status, symbol, decision session, case ID,
and contract version. Missing or mismatched evidence fails closed.

The input root is configuration-driven. The runtime never reads
`AlphaMAS/data/processed/...`, the original FinMultiTime dataset, or an
external drive, and it never writes to the frozen bundle.

## Runtime routing

```text
TEXT                    → News Analyst
TABLE                   → Fundamentals Analyst
TIME_SERIES + IMAGE     → Market Analyst
Social Analyst          → no FinMultiTime-specific evidence
```

The store returns only the packet's existing `routed_projections` string. It
does not reconstruct or re-render evidence. The two immutable Qwen captions
are exposed only through the frozen Market projection; raw model output and
model metadata are not exposed.

## Prompt integration

The default configuration is `finmultitime_evidence_enabled = false`. In that
mode no bundle is loaded, no evidence message is created, and the existing
analyst system message, tools, temporal instructions, and tool loop are
unchanged.

When enabled, the three relevant analyst factories receive an explicit
`FrozenFinMultiTimeEvidenceStore`. Each invocation prepends one deterministic
`HumanMessage` containing the fixed neutral wrapper and the exact frozen route
to the analyst-local prompt input. The shared `state["messages"]` list is not
modified. The message is placed before the existing history so an existing
`AI tool call → ToolMessage` adjacency remains intact. The Social/Sentiment
factory is not given an evidence provider and its prompt remains unchanged.

The Market Analyst's existing `get_verified_market_snapshot` source-of-truth
instruction is unchanged; TIME_SERIES is descriptive additional evidence only.

## Symbol and decision-session identity

The formal path already initializes `company_of_interest` with the canonical
ticker used by the framework. The adapter uses that deterministic field (upper
cased and stripped only) and `state["trade_date"]` as the exact
`decision_session`. It performs no timezone conversion and no nearest-date,
previous-week, later-week, or cross-symbol fallback.

## Cache, checkpoint, resume, and Memory namespace

M1 decision cache payloads include:

```text
finmultitime_enabled
input bundle identity
contract SHA
case ID
packet JSON SHA
news/fundamentals/market route SHAs
```

The checkpoint thread signature includes the enabled flag, bundle identity, and
case packet SHA. Therefore the same symbol/date cannot reuse an M0 decision or
resume checkpoint, and a changed packet cannot reuse an existing M1 prefix.
The graph configuration identity includes the M1 protocol fields while
excluding the machine-specific input path. Experiment Memory remains in the
existing graph-config/lineage namespace; the Memory algorithm itself is
unchanged.

## Provenance and audit

Each routed access records a structured `finmultitime.frozen_evidence_packet`
source-audit entry containing the enabled flag, contract version/SHA, bundle
identity, case ID, packet SHA, analyst route, and route SHA. Cache case metadata
contains the same lightweight identities. The complete bundle is never copied
into per-case artifacts.

## Deterministic integration probe

Run:

```bash
uv run --frozen python scripts/finmultitime/probe_m1_runtime_integration.py \
  --input-root /path/to/AlphaMAS-Experiments/experiments/M1/inputs
```

The probe formats prompt-local messages for `AAPL`, `AMZN`, and `JPM` on
`2024-01-05` and reports packet/route SHA identities. It asserts exact frozen
projection equality, Social route absence, and correct message ordering. It
does not invoke an LLM or an Agent.

## Controlled difference

Allowed changed classes are limited to:

- isolated frozen FinMultiTime evidence adapter and prompt wrapper;
- explicit M1 configuration;
- analyst-local evidence dependency injection;
- cache/checkpoint evidence identity;
- structured runtime provenance;
- deterministic probe, tests, and this documentation.

Trader, Memory learning/maturity/outcome logic, execution, valuation,
portfolio accounting, transaction costs, metrics, formal schedule, action
mapping, debate/risk rounds, and LLM provider/model settings are unchanged.

## Research state

```text
M1 frozen inputs              = YES
M1 runtime integration       = IMPLEMENTED
M1 runtime research-freeze   = NOT YET
M1 pilot                    = NOT RUN
Formal M1                   = NOT RUN
M2/RL                       = NOT STARTED
```
