# M2 Agentic RL Architecture Contract

**Task:** M2-01 — Agentic RL Architecture Contract Freeze
**Project:** AlphaMAS MSc Dissertation
**Status:** Architecture contract frozen for progression to M2-02; this is not the final Full M2 research freeze.
**Starting branch:** `baseline-m2`
**Starting SHA:** `ac0d1b006d8019748702fda38399a4316befb9b0`
**M1 inheritance SHA:** `ac0d1b006d8019748702fda38399a4316befb9b0`

This document is the authoritative high-level contract for subsequent M2 tasks. It freezes interfaces, research controls, and provenance requirements. It does not implement an Actor, Critic, policy state, experience store, Formal M2 configuration, training run, or experiment.

## 1. Scope and research question

The formal research structure is:

```text
M0 → M1 → M2 + A1/A2
```

M2 asks:

> In an otherwise identical information and execution environment, can Agentic Reinforcement Learning applied specifically to the Trader Agent improve the multi-agent trading system beyond M1?

The controlled comparison is `M2 vs M1`. M2 is defined as:

> research-frozen M1 + Agentic RL applied only to the Trader Agent.

The contribution is the external Agentic RL Trader policy and its delayed, point-in-time-safe adaptation boundary. M2 does not claim to invent PPO, a general RL algorithm, or a new generative language model.

M2-01 is repository analysis and documentation only. No model is trained, no experiment is executed, no checkpoint or Formal artifact is produced, and no file in `AlphaMAS-Experiments` is changed.

## 2. Frozen M1 inheritance

M2 inherits the M1 source and Formal protocol at the frozen starting SHA. `baseline-m1` and `baseline-m0` are outside the scope of M2 development and must not be modified.

### 2.1 Formal information environment

Formal M2 must consume the exact frozen M1 FinMultiTime Evidence Packets. The following identities are immutable:

| Identity | Frozen value |
|---|---|
| Contract version | `M1-FINMULTITIME-v1.0.2` |
| Contract SHA-256 | `46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08` |
| Packet manifest SHA-256 | `05f10129b430475bad3d5dee9dfceffba463f8a99cdd3f087ff4b755a869d63d` |
| Input bundle identity | `30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121` |
| Input archive commit | `3750fa50224ba46ab1d4bf5511cb5e8fa514445b` |

Formal M2 must not regenerate Qwen captions, modify packet text, alter lookback or FinMultiTime preprocessing, reroute analyst evidence, supplement unavailable evidence with web data, add future information, or change any M1 packet identity. Training/development data outside Formal 2024H1 may be constructed later as a separately identified pre-2024 dataset.

### 2.2 Formal execution and evaluation environment

Formal M2 keeps the M1/M0 execution and valuation semantics unchanged: raw execution inputs, benchmark construction, corporate actions, trading costs, mark-to-market, and no forced final liquidation. The frozen market-input identities are:

| Symbol | SHA-256 |
|---|---|
| AAPL | `5428fc2c672f3b68c7c3e83b4a22bd5b7330c95a8b4194695762539d9d8a5af3` |
| AMZN | `c4b5c747d75ba658c6f6833348783e3f8a8c571380c930de20cf9fb7dd6b1444` |
| JPM | `74cf77b77b0a83ce8e6246578d4da30bf7622558e8973bda71344b99b9dfd6fc` |
| SPY | `22e6996ebf963787f40d54bfc59e1ca088fa698cb82b639768504dbdbb2d25ac` |

### 2.3 Formal protocol inherited by M2

- Symbols: AAPL, AMZN, JPM; calendar: XNYS.
- Window: 26 weekly decisions per symbol, 78 decisions total; final valuation: `2024-07-05`; warm-up: 252 sessions.
- Decision: final valid XNYS session close; execution: next valid XNYS session open.
- Account: `$100,000` per symbol, fractional shares, long-only, no shorting, no leverage.
- Mapping: BUY → target weight `1.0`; SELL → target weight `0.0`; HOLD → preserve.
- Costs: 5 bps commission and 5 bps slippage; daily mark-to-market; no forced final liquidation.
- Formal LLM/MAS: DeepSeek V4 Flash for quick/deep calls, thinking disabled, temperature 0, Medium research depth, 3 research debate rounds, 3 risk rounds, market/social/news/fundamentals analysts, English output.
- Verbal Memory: experiment mode, five trading-session outcome horizon, adjusted-close outcome semantics, and reflection/update only after the outcome is observable.

## 3. Repository-grounded current architecture

The source inspection found the following live path. The backtester is the execution owner; the Graph returns the Portfolio Manager decision, not a Trader order.

```text
WeeklyBacktestEngine
  → TradingAgentsStrategy.decide
    → TradingAgentsGraph.propagate(mode="historical")
      → analysts: market / social / news / fundamentals
      → Bull Researcher ↔ Bear Researcher
      → Research Manager
      → Trader
      → Aggressive / Conservative / Neutral risk debate
      → Portfolio Manager
    → SignalProcessor.parse_rating
    → strict_action: 5-tier PM rating → BUY/HOLD/SELL
    → target_for: BUY=1.0, SELL=0.0, HOLD=preserve
    → next-XNYS-open Order and Broker fill
    → daily Portfolio snapshots and final valuation
```

### 3.1 Trader contract in the current code

`tradingagents/agents/trader/trader.py` constructs the Trader with the quick-thinking LLM. It reads:

- `company_of_interest`;
- deterministic `instrument_context`;
- `investment_plan` from the Research Manager; and
- the point-in-time temporal instruction.

It does not read `past_context` or directly consume the verbal Memory log. `past_context` is injected into the initial state, but the current Portfolio Manager is the node that places prior lessons into its prompt.

The structured schema in `tradingagents/agents/schemas.py` is `TraderProposal`:

- `action`: `TraderAction.BUY`, `TraderAction.HOLD`, or `TraderAction.SELL`;
- `reasoning`;
- optional `entry_price` and `stop_loss`; and
- optional `position_sizing`.

The current node renders that proposal to Markdown, including `**Action**: ...` and `FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`, then stores the rendered text in `state["trader_investment_plan"]`. This rendered text is the current Trader proposal representation.

### 3.2 Graph and downstream contracts

`tradingagents/graph/setup.py` creates the Trader and connects it directly to `Aggressive Analyst`. The risk router cycles through the three risk analysts and eventually routes to `Portfolio Manager`. The safest M2 insertion boundary is therefore the single edge immediately after the unchanged `Trader` node and immediately before the first risk node:

```text
Research Manager → unchanged Prompt Trader → M2 RL Trader policy boundary → Aggressive Analyst
```

The M2 layer must preserve the original rendered proposal and action as an explicit field. For compatibility with the existing risk prompts, the RL-adapted proposal may eventually be supplied through the existing Trader-proposal hand-off, but it must be accompanied by separate provenance fields; the two proposals must never be silently collapsed.

The three risk analysts currently read `state["trader_investment_plan"]` and the existing analyst reports. The Portfolio Manager reads the Research Manager plan, Trader proposal, risk-debate history, and any current `past_context`. It emits a `PortfolioDecision` with the five-tier rating `Buy`, `Overweight`, `Hold`, `Underweight`, or `Sell` into `final_trade_decision`.

`SignalProcessor` deterministically extracts that five-tier rating. The backtesting adapter maps it to the three execution actions: Buy/Overweight → BUY, Hold → HOLD, and Underweight/Sell → SELL. Therefore, the Portfolio Manager remains the final policy decision before execution; the RL Trader action is a proposal that risk and portfolio management can accept, modify, or reject.

### 3.3 Backtester and chronology findings

`WeeklyBacktestEngine.run` is a daily event loop for one symbol. At each session it applies corporate actions, executes only the pending order intended for that session’s open, marks the portfolio at close, and calls the strategy at weekly decision closes with market history sliced through that close. The strategy receives a `PortfolioSnapshot` containing cash, quantity, equity, current weight, and prior-position information.

The runner in `scripts/run_weekly_backtest.py` is symbol-major: `run_accounts` completes the full decision sequence for one symbol before starting the next symbol. Formal M1 execution is explicitly single-worker and non-parallel. This is safe for M2 only if policy state, pending experiences, and delayed updates are isolated per symbol. A global mutable adapter or shared reward queue would make the current chronology unsafe and is forbidden. A future implementation must either retain symbol-major execution with fully independent symbol state or explicitly prove a date-major scheduler without changing the Formal trading protocol; M2-01 freezes the former as the compatible boundary.

## 4. M1 → M2 controlled difference

The only intended Formal difference is a compact external RL policy layer applied to the Trader hand-off:

```text
Frozen M1 analysts
  → frozen bull/bear debate
  → frozen Research Manager
  → unchanged DeepSeek V4 Flash Prompt Trader
  → explicit Prompt Trader proposal/prior
  → new external RL Actor and value boundary
  → RL-adapted Trader proposal
  → frozen risk debate
  → frozen Portfolio Manager
  → frozen execution/backtester
```

The controlled information path is therefore:

```text
M1:
upstream MAS → Research Manager → unchanged Prompt Trader

M2:
same upstream MAS → same Research Manager → same Prompt Trader → RL calibration
```

The RL Actor is not given all hidden MAS state. It receives semantic information
already present at the existing M1 Trader hand-off, plus the explicitly permitted
endogenous portfolio control state. Upstream analysts and debate may influence the
Actor indirectly through the unchanged Research Manager `investment_plan` and
Prompt Trader proposal, but they must not become a new direct Actor input channel.

The risk and portfolio components remain the M1 implementation and semantic contract. Their input proposal is the intended M2 adapted proposal, but their prompts, model identity, rounds, output ratings, and execution mapping are not redesigned. The M2 layer must not modify the Formal action space, add raw market information, fine-tune or replace DeepSeek, or alter the Evidence Packets.

## 5. Agentic RL operational definition

For this dissertation, Agentic RL means an external sequential decision policy operating inside the existing Trader slot, using semantic information already present at the existing M1 Trader hand-off, a Prompt Trader proposal prior, permitted endogenous portfolio control state, delayed outcome feedback, and reproducible policy/experience lineage. The mechanism is agentic because it observes the current agent-generated proposal, chooses whether to follow/retain/override it, and can adapt from mature outcomes under a strict point-in-time boundary.

The label does not imply that DeepSeek is trained. DeepSeek remains the
unchanged generator of the M1 Trader hand-off and Prompt Trader proposal in
Formal M2; it is not a source of a new direct upstream-MAS input to the Actor.
PPO-family optimisation is an implementation candidate rather than the claimed
novelty.

## 6. Proposal-Prior Actor contract

### 6.1 Prompt Trader remains observable and unchanged

The DeepSeek V4 Flash Prompt Trader is retained exactly as the M1 Trader. Its prompt, model, temperature, thinking mode, structured schema, and output semantics are frozen. Its normalized action is recorded as `prompt_trader_action`; its full rendered proposal and a stable hash are recorded as the Prompt Trader proposal identity.

The Prompt Trader is not replaced by a local generative LLM and is not fine-tuned. The original proposal remains available for baseline comparison, credit assignment, override analysis, and future ablations.

### 6.2 External policy interface

The trainable policy is a compact external model, not a generative LLM. The architecture family is:

- a frozen local semantic encoder applied to the frozen M1 Trader hand-off state;
- a proposal-anchored discrete Actor over BUY/HOLD/SELL;
- a value/Critic component; and
- an optional small fast-adaptation head for delayed per-symbol Formal updates.

The Actor consumes only the explicitly bounded semantic representation of the existing M1 Trader hand-off, the complete unchanged Prompt Trader proposal/action/reasoning, deterministic instrument/time context already available to the M1 Trader, and the permitted endogenous portfolio state. It produces action scores/logits/probabilities and a selected `rl_trader_action` in the unchanged three-action space. This boundary does not reduce the Actor to the final Prompt Trader action alone: the Research Manager `investment_plan` and the richer unchanged Prompt Trader proposal/reasoning remain permitted semantic inputs.

The Prompt Trader action is an explicit policy prior/anchor. The Actor must be able to follow it, retain the current position through HOLD, or override it. The exact prior implementation—feature conditioning, logit prior, regularisation, or another validated equivalent—is intentionally not frozen in M2-01. What is frozen is the role: the Prompt Trader proposal remains an observable prior and is not discarded.

The Actor’s output is an RL-adapted Trader proposal consumed by the unchanged downstream risk path. The Actor does not bypass risk debate, does not call the Broker, and does not directly determine the executed action.

## 7. Critic architecture boundary

The Critic is an external value component trained against the same point-in-time observation boundary as the Actor. A scalar value Critic and a distributional Critic remain architecture-level candidates. The distributional option may represent return uncertainty and support a later risk-sensitive analysis, but it is not selected here.

The final scalar-versus-distributional choice must be based only on pre-Formal validation. No Formal 2024H1 outcome, M1 result, or performance-driven selection may determine it. Exact encoder model, network dimensions, layer count, dropout, activation, optimiser, and PPO-family hyperparameters remain deferred.

## 8. Delayed online adaptation boundary

If Formal online adaptation is enabled after pre-Formal validation, it is restricted to a small fast-adaptation component. The global pretrained semantic encoder, Actor/Critic backbone, and Formal initial checkpoint are frozen. Each symbol starts from the same frozen initial fast-adapter parameters and maintains an independent:

- policy state;
- experience ledger and applied-event prefix;
- reward/update queue; and
- policy lineage.

An AAPL mature reward cannot update AMZN or JPM. No update can occur at the decision that generated the experience. The earliest valid update is after the complete five-session outcome is visible and the experience transitions from `MATURED` to `APPLIED`. Pending or unavailable outcomes remain pending.

The current symbol-major runner is compatible with this boundary because each symbol can be completed using its own state. It is not compatible with a shared mutable online policy; cross-symbol state sharing is prohibited.

## 9. Preliminary M2 observation/state contract

The final feature representation and encoder are deferred, but the information classes are frozen.

### 9.1 Frozen semantic information

The Formal RL Actor's semantic observation is restricted to semantic information
already present at the existing M1 Trader hand-off. The preliminary observation may
include stable identities, hashes, or encoded representations of:

- the Research Manager `investment_plan`, including its deterministic
  structured/rendered representation and stable identity/hash;
- the complete unchanged Prompt Trader structured/rendered proposal, including its
  BUY/HOLD/SELL action, reasoning, permitted optional proposal fields, and stable
  proposal/action identity/hash;
- deterministic context already available to the M1 Trader, including instrument
  context, company/ticker identity, decision session, decision timestamp, and the
  existing temporal/point-in-time instruction/context.

This is a hand-off boundary, not a licence to expose arbitrary upstream MAS state.
Raw analyst reports and upstream debate transcripts that the M1 Prompt Trader did
not directly consume must not be passed directly to the Actor. Those sources may
influence the Actor only indirectly through the unchanged M1 path:

```text
Analysts / Debate → Research Manager investment_plan → Prompt Trader proposal → RL Actor
```

The richer Research Manager plan and Prompt Trader proposal/reasoning are allowed;
the Actor is not restricted to a three-value copy of the Prompt Trader action.
Evidence Packet text, M1 lookbacks, analyst routing, and point-in-time rules remain
unchanged.

The following are forbidden as direct Formal RL Actor inputs:

- raw `market_report`, `sentiment_report`, `news_report`, or
  `fundamentals_report`;
- raw bull-researcher or bear-researcher transcripts, including complete
  bull/bear debate history;
- Research Manager internal debate state beyond the final `investment_plan`;
- raw Risk Debate content before the RL action or Portfolio Manager output;
- future outcomes/reflections, future rewards, or future prices;
- new technical indicators, new web evidence, or modified FinMultiTime evidence.

This prohibition prevents M2 from becoming an RL policy with a privileged direct
view of upstream MAS internals. The intended controlled difference remains
`M2 − M1 = Agentic RL applied to the Trader hand-off`, not RL plus a new semantic
information interface.

### 9.2 Permitted endogenous portfolio state

The preliminary observation may include:

- cash;
- position quantity;
- current asset weight;
- portfolio equity;
- prior executed position state; and
- other explicitly versioned state derived from the account’s own prior actions.

Portfolio state is an **M2-only endogenous control-state interface, not an additional market-information source**. Raw OHLC, new indicators, future prices, execution-session prices, reward outcomes, or post-decision reflections must not be added to the Formal Trader observation.

Every semantic and portfolio sub-state must eventually have a deterministic schema identity and SHA-256 identity. The exact feature list, normalisation, encoder, dimensions, and missing-value policy are deferred to later validation.

## 10. Action and provenance contract

The action space remains exactly:

```text
BUY | HOLD | SELL
```

There is no continuous sizing, leverage, shorting, or Actor-side execution. The unchanged backtester semantics remain BUY → target weight 1.0, SELL → target weight 0.0, and HOLD → preserve the prior position.

Future artifacts must retain at least these distinct fields:

1. `prompt_trader_action` — normalized action from the unchanged Prompt Trader;
2. `rl_trader_action` — selected action from the external Actor;
3. `portfolio_manager_final_action` — normalized three-action interpretation of the PM’s final five-tier rating, with the raw PM rating retained separately; and
4. `executed_action` — action actually submitted to the execution mapping, alongside order status, no-op status, and fill provenance.

These fields must not be collapsed into one `action` field. A PM override, an execution no-op, and an unfilled/rejected order must remain distinguishable from an Actor decision.

## 11. Memory and experience lifecycle

### 11.1 Existing verbal Memory

The repository’s `TradingMemoryLog` is an append-only Markdown log. At the end of a successful Graph run, the final PM decision is stored as `pending`. At the start of a later same-ticker run, `_resolve_pending_entries` attempts to obtain a complete five-session adjusted-close outcome, computes raw and benchmark-relative returns, generates a short verbal reflection, and atomically resolves the entry with an `outcome_visible_from` timestamp. Historical reads fail closed when a resolved reflection was not visible by the historical cutoff.

This verbal Memory remains logically separate from the numeric RL policy/experience store. Its prose reflection, cross-ticker lesson formatting, and LLM lifecycle must not be treated as the Actor’s numeric experience ledger. Future integration must commit both through one chronological transaction boundary without making them the same data structure.

### 11.2 Logical RL experience schema

M2-01 freezes the logical schema only; it does not implement the store. Each future experience record must be capable of recording:

| Field group | Required contents |
|---|---|
| Case identity | experiment ID, RL policy lineage ID, symbol, decision session, decision time |
| State identity | semantic-state identity/hash, portfolio-state identity/hash, feature/representation schema identity |
| Proposal/action | Prompt Trader proposal/action, RL scores/logits/probabilities, selected RL action, PM final action, executed action |
| Transition provenance | policy-state-before hash, policy checkpoint/encoder identity, implementation/config hashes |
| Outcome lifecycle | pending reward status, reward horizon, reward maturity session, outcome source identity, reward value, reward-visible timestamp |
| Update provenance | update-applied flag, unique update event ID, policy-state-after hash, applied-event prefix identity |

The logical status machine is:

```text
PENDING  -- full horizon observable -->  MATURED  -- exactly-once update -->  APPLIED
```

`PENDING` means no reward may yet be read. `MATURED` means the full outcome is observable under the runtime cutoff and the reward is recorded but not yet applied. `APPLIED` means one validated update event has committed and its before/after policy-state hashes are recorded. A failed or interrupted case must not create a partially applied transition.

The outcome source identity must bind to the exact permitted price/outcome inputs and their hashes. The numerical reward formula is not frozen here.

## 12. Point-in-time rules

The following rules are mandatory:

1. No reward may be calculated before the full outcome horizon is observable.
2. Formal reward/update may use only data available as of that decision’s runtime cutoff.
3. No future outcome may enter Actor input, Critic runtime input, Prompt Trader, risk layer, or Portfolio Manager.
4. Online updates may use only already-matured experiences.
5. Terminal experiences whose reward is not observable before Formal final valuation remain `PENDING`.
6. No Formal test result may trigger architecture, reward, hyperparameter, Critic, encoder, or model-selection changes.

For the inherited 2024H1 schedule, the final decision is `2024-06-28`. Its five subsequent valid XNYS sessions are `2024-07-01`, `2024-07-02`, `2024-07-03`, `2024-07-05`, and `2024-07-08`. Formal valuation stops at `2024-07-05`, so the final experience is not mature at the Formal cutoff and must remain pending. The final valuation must not be extended merely to mature that reward.

## 13. Training and Formal separation

All architecture development, reward selection, hyperparameter tuning, Critic selection, model selection, training, validation, and holdout evaluation must use data strictly before `2024-01-01`. The 2024H1 Formal window is test-only.

The intended later pipeline is:

```text
Pre-2024 FinMultiTime / historical-safe evidence
        ↓
Frozen M1 Trader hand-off state generation
        ↓
Frozen Trader hand-off semantic corpus
        ↓
Frozen local encoder
        ↓
Counterfactual BUY/HOLD/SELL outcomes
        ↓
Critic/value pretraining
        ↓
Proposal-anchored Actor optimisation
        ↓
PPO-family training
        ↓
Optional delayed fast-adaptation training
        ↓
Frozen Full M2 checkpoint
```

DeepSeek may later be called to generate unique pre-Formal M1 Trader hand-off
states, including the Research Manager plan and unchanged Prompt Trader proposal,
but RL optimisation must be reusable locally without repeatedly calling DeepSeek.
Full M2, A1, and A2 should later share the same frozen pre-Formal Trader hand-off
semantic corpus. M2-01 itself makes zero DeepSeek calls.

## 14. Cache, resume, and lineage requirements

The current `DecisionCache` is content-addressed and atomic, while `TradingAgentsStrategy` already binds decisions to a rolling chronological prefix and the runner’s experiment Memory is identified by graph configuration, protocol, lineage, and symbol. Those mechanisms are necessary but not sufficient for RL state transitions.

Future M2 decision-cache identity must include both:

### 14.1 Static RL research identity

- architecture version;
- feature/representation schema identity;
- reward-spec identity;
- Actor and Critic checkpoint identities;
- training-data/corpus identity;
- update-rule identity;
- frozen development hyperparameters; and
- encoder identity.

### 14.2 Dynamic RL execution identity

- RL lineage ID;
- policy-state-before SHA-256;
- experience-ledger prefix SHA-256;
- already-applied reward-event prefix;
- per-symbol adapter state identity; and
- the M1 graph/config/evidence/portfolio state identities already required by the repository.

Cache replay must never return an old action without reconstructing the corresponding RL state transition. It must not skip a required reward application or double-apply a reward. A cache hit must be semantically equivalent to replaying the exact decision, ledger-prefix, policy-state, and outcome-transition event.

## 15. Failure and resume semantics

Verbal Memory, RL policy state, and the RL experience ledger form one logical chronological transaction boundary for a future M2 execution. The repository already provides useful foundations: per-ticker Graph checkpoints, atomic Memory writes, atomic cache publication, and fail-fast chronological decision handling.

The recommended RL implementation is an append-only/event-sourced ledger with deterministic event IDs and commit markers. For each Agent case:

- append the pending experience and the verbal Memory pending decision as one commit unit;
- apply a mature reward and policy update at most once, guarded by its update event ID and ledger prefix;
- publish policy-state-after only after the update event is durable; and
- on failure, roll back or ignore the entire uncommitted unit.

If an Agent case fails, no partial RL update may survive, no partial experience append may survive, and resume must not duplicate a reward application. Resume must reconstruct the last committed verbal-Memory prefix, policy-state hash, experience-ledger prefix, and cache chronology before recomputing the failed case. The existing symbol-major runner requires these prefixes to be symbol-scoped for RL adaptation, while the experiment-level run lineage may still bind the full run for reproducibility.

## 16. Artifact and provenance requirements

The repository currently writes per-case decision, model-config, run-context, cache-identity, source-audit, and LLM-usage artifacts, plus decision timelines, validation reports, and a final symbol-specific Memory archive. Future M2 RL artifacts must extend this provenance without replacing the M1 fields.

At minimum, future M2 artifacts must preserve:

- all four action-provenance fields and the raw Prompt Trader proposal;
- semantic and portfolio state hashes and schema identities;
- RL architecture, encoder, Actor, Critic, reward-spec, and update-rule identities;
- policy lineage, policy-state-before/after hashes, and per-symbol adapter identity;
- experience ID, ledger prefix, status, reward horizon/maturity/visibility, outcome-source identity, and update event ID;
- cache static/dynamic identity and whether the case was a fresh computation or replay;
- M1 Evidence Packet and execution-input identities; and
- rollback/resume status and validation results.

The future artifact schema must make PM overrides, Actor overrides, no-op execution, rejected orders, pending terminal rewards, and cache replay independently auditable. M2-01 creates no runtime artifact and does not change the existing artifact schema.

## 17. API and AWS budget boundaries

This task performed:

```text
DeepSeek API calls: 0
DeepSeek API cost: ¥0
AWS GPU hours: 0
AWS incremental cost: $0
```

The future combined pre-Formal M2/A1/A2 incremental DeepSeek budget is targeted at `¥20–40`, with a normal hard stop at `¥40` and an absolute ceiling of `¥50`. The `¥40–50` reserve is for replacement of correctness-invalid calls only, never performance-driven expansion. Formal M2/A1/A2 API expenditure is a separate budget.

M2-01 does not start EC2, allocate GPUs, run Qwen, train anything, or create AWS infrastructure. AWS provisioning belongs to M2-05.

## 18. Deferred decisions

The following decisions are intentionally not frozen in M2-01:

- final reward formula: deferred to M2-03/M2-04;
- training universe: deferred to M2-02;
- train/validation/holdout split: deferred to M2-02;
- exact semantic encoder model: deferred until representation/resource evaluation;
- exact network dimensions, layer count, dropout, activation, and optimiser parameters;
- PPO-family hyperparameters, including learning rate, clipping, entropy/KL coefficients, batch size, epochs, GAE lambda, and gamma;
- final scalar-versus-distributional Critic choice, subject only to pre-Formal validation;
- exact feature representation and normalisation within the frozen observation classes;
- whether the optional Formal fast-adaptation head is enabled, subject to pre-Formal correctness/validation, while its per-symbol isolation boundary is frozen; and
- A1/A2 mechanisms, which are selected only after Full M2 is research-frozen.

## 19. Risks and limitations

- A proposal prior can anchor the Actor too strongly; the prior strength must be validated without using Formal results.
- DeepSeek-generated M1 Trader hand-off states may be expensive or sparse; the local hand-off semantic corpus must be frozen before policy optimisation.
- Only 26 Formal decisions per symbol are available, so Formal online adaptation is an evaluation condition, not a basis for tuning.
- The symbol-major runner makes shared online state especially dangerous; per-symbol lineage must be mechanically enforced.
- Delayed rewards create pending terminal experiences and require durable exactly-once update semantics.
- The PM may override the RL Trader proposal, so Actor performance and end-to-end execution performance must be analysed separately.
- No architecture at M2-01 guarantees improvement over Buy & Hold, M1, or any benchmark.

## 20. A1/A2 boundary

A1 and A2 are not selected by M2-01. Future A1/A2 work must start only after the complete M2 architecture is research-frozen, use the same pre-Formal frozen semantic corpus where applicable, and preserve the M1 information/execution controls unless a separately approved controlled-difference contract says otherwise.

## 21. Research-validity audit

| Question | M2-01 answer |
|---|---|
| Is M2 controlled against M1? | **YES.** The only intended difference is the external Trader policy layer; M1 information, reasoning models, risk/PM path, and execution remain frozen. |
| Does Formal M2 add new market information? | **NO.** Portfolio state is an endogenous M2 control-state interface, not an additional market-information source. |
| Is DeepSeek replaced or fine-tuned? | **NO.** DeepSeek V4 Flash remains unchanged and untrained. |
| Does the action space change? | **NO.** BUY/HOLD/SELL and the existing target-weight mapping remain unchanged. |
| Is Formal 2024H1 used for training/model selection? | **NO.** It is test-only. |
| Are M1 observed 2024H1 performance results used to choose reward or architecture? | **NO.** M2-01 does not inspect or optimise against Formal performance. |
| Is reward design already frozen? | **NO — DEFERRED TO M2-03/M2-04.** |
| Are A1/A2 already selected? | **NO.** They are deferred until after Full M2 research freeze. |
| Is AWS/GPU required in M2-01? | **NO.** GPU hours and incremental AWS cost are zero. |

## 22. Architecture-freeze verdict

Repository inspection found no correctness conflict requiring a change to the Formal action space, M1 information environment, execution protocol, or DeepSeek boundary. The Proposal-Anchored RL Trader is therefore frozen as the high-level M2 architecture for progression to M2-02, subject to all deferred decisions and later pre-Formal validation remaining open as specified above.

**PASS — M2 architecture contract frozen for progression to M2-02**
