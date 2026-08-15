# M2 Semantic Hand-off Contract

Contract ID and schema version: `M2-SEMANTIC-HANDOFF-v1`.

## Observation boundary

The static semantic state is the deterministic record at the existing M1
Research Manager → Prompt Trader boundary. The capture executes the unchanged
market, social, news, and fundamentals analysts, the unchanged three-round
Bull/Bear research debate, the unchanged Research Manager, and the unchanged
DeepSeek Prompt Trader, then stops immediately after Prompt Trader.

Actor-visible fields are the exact Research Manager `investment_plan` and its
SHA256; the complete byte-preserved rendered Prompt Trader proposal, its SHA256,
and a deterministically parsed BUY/HOLD/SELL action; ticker/instrument and
historical decision context; frozen Evidence Packet identity; and stable source,
model-configuration, experiments, and run identities.

The Prompt Trader proposal is never reduced to its action. The exact plan is the
one consumed by the unchanged Prompt Trader. Direct analyst reports, Bull/Bear
transcripts, Risk Debate state, Portfolio Manager output, orders, fills, reward,
future price/return, outcomes, and Formal/B&H results are forbidden Actor inputs.

## Separation and deferred choices

Upstream text may appear only in a distinct `upstream_trace.json` that declares
`actor_visible=false`. Future Actor code must never load that provenance-only
text. Cash, quantity, weight, equity, and prior position are dynamic endogenous
environment state and are not part of this static corpus.

The semantic encoder, embedding dimension, Actor, Critic, and PPO-family method
remain deferred. The capture uses `deepseek-v4-flash` for quick and deep calls,
thinking disabled, temperature zero, Medium research depth, three research
debate rounds, and English output. Every case runs in historical/PIT mode using
its frozen decision time; unavailable evidence remains unavailable and no live
replacement evidence is introduced.

Risk Debate, Portfolio Manager, broker, backtester execution, and reward
evaluation are outside the capture boundary. The six M2-07 probe states are
authoritative first correctness-valid outputs and must be reused byte-for-byte
by M2-08 rather than regenerated.
