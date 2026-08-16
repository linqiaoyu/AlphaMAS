# M2 Agentic RL Method Freeze

## M2-10A Correctness Erratum

`M2-PA-CTPPO-v1` is superseded before training. It incorrectly used the end of
the five-session R3 credit window as the next weekly portfolio state. This is
point-in-time invalid around holidays: the action selected at the 2023-05-26
close executes at the 2023-05-30 open, the next frozen weekly decision is the
2023-06-02 close, and its R3 credit matures only at the 2023-06-05 close. Using
the maturity portfolio on June 2 would leak future information across 216 nodes
and 648 action edges.

The corrected `M2-PA-CTPPO-v2` method separates weekly state-transition time
from local reward-maturity time. This is a correctness correction made before
policy training, VALIDATION-performance inspection, or Holdout use. It does not
alter the frozen split, representation, reward, Prompt prior, weekly decisions,
or trading protocol.

## Weekly state tree and delayed credit

`M2_TRAIN_COUNTERFACTUAL_TREE-v2` is a complete, history-preserving tree of all
weekly portfolio states reachable through BUY, HOLD, and SELL. Eight independent
symbol episodes start with $100,000 and no position. Seven frozen weekly depths
produce 8,744 decision nodes, 26,232 action edges, 17,496 terminal action leaves,
and 8,736 nonterminal weekly transitions. Histories and no-op branches are never
merged or pruned.

For a nonterminal edge, the selected action executes at the next XNYS open and
the actual sequential portfolio advances only through the next frozen weekly
decision close. The child state carries cash, quantity, true average entry,
open-position commission, peak equity, current equity, and drawdown. Corporate
actions, 5 bps commission, 5 bps slippage, fractional shares, and long-only
rules use the frozen Broker and Portfolio primitives.

Independently, each node/action receives one five-session
`R3_HOLD_RELATIVE_DRAWDOWN_UTILITY` credit probe from the exact parent state.
This probe does not advance or overwrite the weekly portfolio and cannot depend
on descendant actions. Reward maturity may be before, on, or after a child
decision. The 648 Memorial Day edges mature after their June 2 child state and
are valid delayed-credit overlaps. A June-5 poison test changes the parent
credit identity while leaving the June-2 child state and Actor observation
byte-identical.

The tree is used only to enumerate reachable weekly states and propagate exact
old-policy occupancy. It is not used to chain overlapping R3 probes into a
Bellman return.

## Frozen state and architecture

The source Actor observation remains the frozen 3,080-dimensional
`M2-SEMANTIC-STATE-v1` vector: three 1,024-dimensional semantic channels,
agreement, three-way Prompt action, and four portfolio features. Symbol, depth,
date, identity, maturity status, reward, and future information are excluded.

One shared bias-free `Linear(1024,16)` projection with tanh transforms Research
Manager, Prompt Trader, and residual embeddings. Their 48 projected features,
agreement, Prompt one-hot, and portfolio state form a 56-dimensional bottleneck.
Separate `Linear(56,32)+Tanh` Actor and Critic trunks share only the projection.
The Actor has a `Linear(32,3)` bounded residual and `Linear(32,1)` sigmoid gate;
the scalar Critic has a `Linear(32,1)` value head.

The Prompt action prior remains 2/3 for the selected Prompt action and 1/6 for
each alternative. Policy logits are `log(pi_prompt) + gate * ln(4) *
tanh(raw_residual)`. Zero-initialized residual and gate heads make the initial
policy exactly equal to the full-support Prompt prior. Deterministic inference
uses argmax with Prompt-first, then BUY/HOLD/SELL tie-breaking.

The architecture has exactly 20,197 trainable parameters: 16,384 projection,
1,824 Actor trunk, 1,824 Critic trunk, 99 residual head, 33 gate head, and 33
value head. Formal online adaptation freezes the global projection and trunks;
only the 165 residual, gate, and value-head parameters evolve independently per
symbol.

## Corrected PA-CTPPO-v2 objective

At each reachable weekly state under frozen old policy:

```text
V_cf(s)   = sum_a pi_old(a|s) R3(s,a)
A_cf(s,a) = R3(s,a) - V_cf(s)
```

The exact BUY/HOLD/SELL expectation makes `sum_a pi_old A_cf = 0`. The scalar
Critic target is local `V_cf`, not a Bellman return or weekly reward-to-go.
There is no Bellman bootstrap, discount factor, GAE, or action sampling.

Each symbol root has occupancy 1/8. Weekly child occupancy is
`d_old(child)=d_old(parent)*pi_old(action|parent)`, summing to one at every
depth. The policy objective is the occupancy-weighted exact three-action PPO
clipped surrogate with ratio `pi_theta/pi_old` and clip 0.20. The value loss is
occupancy-weighted MSE to `V_cf`; total loss is `-L_policy + 0.5 L_value`.
Raw R3 is used without clipping, normalization, ranking, sign transformation,
or advantage standardization. Entropy and Prompt-KL coefficients are zero.

Future global training remains AdamW with weight decay `1e-4`, gradient norm
0.5, seed 20260816, 100 outer iterations and four optimization epochs. Only
learning rates `1e-4`, `3e-4`, and `1e-3`, at checkpoints 25, 50, and 100, may
form the nine candidates. Future VALIDATION selection remains equal-weight mean
sequential local R3 across 16 decisions with the preregistered deterministic
tie-break. No VALIDATION performance was inspected here.

## Delayed online adaptation

Formal experience follows `PENDING -> MATURED -> APPLIED`. A decision cannot use
its own future credit. If an older credit matures after a newer decision, it can
affect only a later decision; if it matures at the current decision close, the
older update may be applied after marking the portfolio and before the current
calibration. If maturity coincides with a newer execution session, that order
executes at the open before the older reward matures at the close. Applied
events are immutable and per-symbol adapters never share updates. Online
learning rate and update epochs remain deferred.

The deterministic adapted proposal preserves the original Prompt Trader text
and SHA, changes only BUY/HOLD/SELL with a RETAIN/OVERRIDE marker, and exposes no
reward, value, probability, gate, checkpoint, or training internals downstream.
Risk Debate, Portfolio Manager, execution, and the production graph remain
unchanged; the Actor never bypasses risk controls.

## Protected boundary and M2-11

VALIDATION performance, FINAL_HOLDOUT, E2E_PILOT performance, and Formal 2024
results were not used. DeepSeek, Qwen inference, AWS, and raw FinMultiTime were
not accessed. The semantic representation, evidence corpus, captions, TRAIN and
VALIDATION membership, weekly dates, R3 horizon/definition, Prompt Trader,
Research Manager, Risk Debate, Portfolio Manager, execution semantics, and
Formal protocol remain unchanged.

M2-11 may begin TRAIN-only global `M2-PA-CTPPO-v2` optimization using weekly
state-tree occupancy and independently matured local R3 advantages. It must not
alter this frozen method, expand the candidate set, or inspect FINAL_HOLDOUT.
