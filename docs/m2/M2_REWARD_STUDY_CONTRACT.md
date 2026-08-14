# M2-03 Reward Study Contract

## 1. Scope

M2-03 freezes the economic reward semantics, counterfactual simulator, three-candidate family, synthetic correctness tests, and the M2-04 decision process for the Agentic RL Trader. It does not choose the Full M2 reward, train an Actor/Critic/PPO component, apply a reward training transform, or inspect realised historical performance. The final reward is **DEFERRED TO M2-04**.

The framework is local and deterministic. M2-03 uses source inspection, calendar semantics, frozen split metadata, and synthetic price paths only. It makes no DeepSeek, Qwen, paid-LLM, network, or AWS call.

## 2. Inherited M2 architecture and data boundary

This contract inherits the architecture at `ea9ca73c15ed94ba3fa37a86e3ee145960d94bc2` and the authoritative post-M2-02C split-contract state at `8574067d039f6288042ec85cebc4517f3c3da49c`. The split lineage began with the original M2-02 split at `cf189c53a3600030911efcd1ceb5afdad1e06765`, followed by the M2-02B fixed-evaluation correction at `71a9a106b328c4e9a967c0275b705a11382450bc`; M2-03 inherits the resulting post-M2-02C state, not the original split commit. It does not reopen the eight-symbol selection (`AAPL`, `AMZN`, `JPM`, `JBSS`, `EML`, `AGI`, `ARR`, `AEMD`), the COMPACT/STANDARD/MAXIMUM case budgets (72/80/96), or evaluation membership. Final Holdout identity `f8e011558f59f56db730702700ef3d419b353b63d4f35278ee78ddb70fedabfe` remains protected until M2-15.

## 3. Frozen execution timing and accounting

The frozen backtester makes a decision at the close of the final valid XNYS session of a calendar week. Any changed target is submitted for the next valid XNYS session open. At that open, dividends and splits affecting the carried position are applied before the new order, after which the portfolio is marked to market at each session close.

- BUY means target weight 1.0. From the reachable CASH state, the broker spends available cash on the maximum affordable long position after adverse buy slippage and commission.
- SELL means target weight 0.0 and liquidates the complete long position.
- HOLD preserves the discrete position: positive quantity remains long and zero quantity remains cash.
- The frozen engine treats any quantity above `Portfolio.tolerance` as its discrete long target. Therefore BUY from an existing fractional-exposure robustness state is a no-op; that state is not a new Formal action.
- Fractional shares are enabled. Shorting and leverage are forbidden.
- Commission is 5 bp of fill notional. Slippage is 5 bp adverse to the action and is already reflected in fill price, cash, and equity; reporting counters do not deduct it twice.
- Cash earns no interest. There is no forced liquidation.
- Dividends increase cash for shares eligible at that open. Splits change quantity and average entry price without changing economic value. The M2-03 synthetic artifact uses no corporate actions, but the simulator preserves these frozen primitives.

## 4. Attribution baseline

For every candidate, `E0` is portfolio equity at the execution-session raw open immediately before the counterfactual action is applied. Corporate actions due at that open have already affected the carried state. The decision-close-to-execution-open overnight move is outside the selected action's economic reward because the action could not yet have executed.

The post-execution path begins with net equity valued at the raw execution open after the trade and its costs, followed by session-close valuations through maturity. Pre-trade equity is not inserted as the drawdown peak.

## 5. Counterfactual HOLD

Each BUY, HOLD, and SELL outcome starts from an identical copy of cash, shares, entry accounting, execution-open market, corporate actions, and subsequent path. The counterfactual HOLD branch makes no new trade. It is not a benchmark strategy and sees no future information at decision time; the outcome is calculated only when the complete horizon is observable.

## 6. R1 — absolute net log return

Candidate ID: `R1_ABSOLUTE_NET_LOG_RETURN`

\[
r_{abs}(a)=\ln\left(\frac{E_T^a}{E_0}\right)
\]

`E_T^a` is terminal net equity after execution costs. R1 is dimensionless and equity-scale invariant, with no benchmark subtraction. It is the conventional economic baseline. Its known limitation is that an unchanged long portfolio can receive non-zero reward from passive market movement that is not uniquely attributable to the Trader's current action.

## 7. R2 — HOLD-relative log advantage

Candidate ID: `R2_HOLD_RELATIVE_LOG_ADVANTAGE`

\[
r_{adv}(a)=\ln\left(\frac{E_T^a}{E_T^{HOLD}}\right)
\]

Equivalently, this is `R1(action) - R1(HOLD)` when both use the same `E0`. Positive reward means the selected action increased terminal net wealth relative to doing nothing; negative reward means it reduced wealth. HOLD relative to itself and any path-identical no-op are exactly zero within floating-point tolerance. Trading costs naturally penalise unnecessary switching.

## 8. R3 — HOLD-relative drawdown utility

Candidate ID: `R3_HOLD_RELATIVE_DRAWDOWN_UTILITY`

For a positive post-execution net-equity path `P`:

\[
MDD(P)=\max_{j\leq k}\left(1-\frac{P_k}{\max_{i\leq k}P_i}\right),
\qquad D(P)=-\ln(1-MDD(P))
\]

and:

\[
r_{risk}(a)=r_{adv}(a)-\max(0,D(P_a)-D(P_{HOLD}))
\]

The action receives a penalty only for drawdown beyond HOLD. Reducing drawdown does not create a separate bonus. Both terms are dimensionless; no tunable lambda is present or permitted in M2-03. R3 should differ from R2 only when the action creates greater local drawdown than HOLD.

## 9. Deliberately excluded reward families

B&H-relative, SPY-relative, and M1-relative rewards are excluded because Formal benchmarks and an observed predecessor are evaluation references, not Trader training targets. Arbitrary composites of return, Sharpe, drawdown, or turnover with tuned coefficients are excluded. Differential Sharpe was considered but is excluded from the primary family because it adds persistent estimator state, a smoothing parameter, weaker locality to the five-session decision, and confounding complexity. There is no fourth primary candidate.

## 10. Maturity and lifecycle

The authoritative convention is the decision session as observation zero and the fifth subsequent valid XNYS session as maturity. Execution occurs at the first subsequent session open; the reward path ends at the fifth-subsequent session close. The simulator requires every XNYS session from execution through maturity.

Before that complete horizon is observable, status is `PENDING` and no scalar or outcome object is emitted. At completion, status is `MATURED` and all three action outcomes are emitted. The future RL ledger may later transition a matured outcome to `APPLIED`; M2-03 does not implement that transition. Observations after maturity are neither validated nor read.

## 11. Transaction costs and no-op semantics

The simulator imports the frozen `Broker` and `Portfolio` directly. BUY and SELL use the same 5 bp commission, 5 bp adverse slippage, fractional-share calculation, tolerance, cash clamp, and long-only assertions as Formal backtesting. Reported slippage is informational because its economic effect is already in the fill. HOLD never creates an order. CASH+SELL and LONG+BUY are frozen no-ops with no artificial cost, including BUY from the synthetic 0.5-exposure robustness state because the engine's current target is discrete.

## 12. Synthetic scenarios

The deterministic sanity suite covers flat CASH and LONG paths; monotonic rise and fall from both states; V and inverted-V/whipsaw paths; paired pre-execution overnight gaps with identical post-open paths; a high-cost stress used only in tests; equity scales of USD 10,000, 100,000, and 1,000,000; no-op states; a 0.5 fractional exposure interface case; appended post-maturity prices; and maturity gating. The machine-readable artifact contains no historical return or leaderboard.

## 13. Simulator/backtester equivalence

The adapter does not define independent trade arithmetic. It creates the same frozen `Order`, executes it through the same `Broker`, carries state in the same `Portfolio`, and values through `Portfolio.snapshot`. Tests also run direct frozen primitive paths for CASH+BUY, LONG+SELL, and HOLD and compare terminal shares, cash, commission, slippage, and terminal equity at relative tolerance `1e-12` and absolute tolerance `1e-10`.

## 14. M2-04 allowed data and state templates

M2-04 may use exactly 56 MAXIMUM TRAIN and 16 VALIDATION pre-Formal cases: 72 decision windows. It may calculate BUY/HOLD/SELL outcomes for the CASH and LONG templates at common nominal equity USD 100,000. This permits approximately `72 × 2 × 3 = 432` matured price-only action outcomes with no LLM call. A fractional exposure remains synthetic robustness only.

## 15. M2-04 forbidden data

M2-04 reward selection must not use FINAL_HOLDOUT, E2E_PILOT, Formal 2024H1, Formal M0/M1 performance, B&H performance, SPY performance, or M1 action profitability. M2-03 likewise inspected none of these outcomes. Existing archived M1 inputs are used only by foundational regression tests.

## 16. Pre-registered diagnostics

For each candidate, M2-04 must report finite rate, minimum, maximum, mean, median, standard deviation, P05/P25/P75/P95/P99, median absolute reward, and extreme-to-median scale ratio where defined. Per state it must report the distribution of maximum-minus-minimum action reward and the fraction with non-trivial spread.

It must also report no-op count/fraction and relative reward magnitude; traded versus no-trade behaviour and synthetic cost monotonicity; CASH-versus-LONG market-drift exposure; R3 penalty activation, magnitude, and relationship to R2; and reward scale/distribution per symbol without ranking symbols by profitability.

## 17. Selection hierarchy

1. Correctness rejects non-finite values, PIT or cost-monotonicity violations, scale failures, or execution mismatch.
2. Semantic suitability requires a clear interpretation attributable to the Trader decision.
3. Non-degeneracy requires meaningful pre-Formal TRAIN/VALIDATION action discrimination.
4. Stability rejects pathological scales or operationally unreliable outliers.
5. Risk/incentive interpretation asks whether explicit downside shaping adds information rather than unstable noise.
6. If candidates remain comparably suitable, prefer the simpler reward with clearer Trader-local credit and fewer assumptions.

No weighted composite score may be invented after results are observed. The prior hypotheses are only that R1 is simple but less local, R2 may offer strong local credit assignment, and R3 may or may not justify its extra complexity.

## 18. Deferred training transform

Reward scaling, standardisation, whitening, clipping, and running normalisation belong to PPO optimiser numerics and remain deferred. M2-04 compares the raw economic candidates. It must not blur the reward definition with a training transform.

## 19. Research-validity rules

The candidate family, formulas, scenarios, diagnostics, and selection hierarchy are frozen before any historical candidate comparison. No candidate may be selected for producing the highest Formal return, for appearing most likely to beat B&H, or through lambda/hyperparameter search. Future evidence that a tunable downside coefficient is scientifically necessary requires an explicit stop and research discussion rather than a silent change.

## 20. Cost and verdict

```text
DeepSeek API calls: 0
DeepSeek API cost: ¥0
Qwen inference calls: 0
AWS GPU hours: 0
AWS incremental cost: $0
```

M2-03 verdict: **PASS — reward framework, simulator semantics, frozen candidates, and pre-registered M2-04 study process are internally specified; final reward selection remains deferred.**

`DEFERRED — final M2 reward will be selected in M2-04 only`
