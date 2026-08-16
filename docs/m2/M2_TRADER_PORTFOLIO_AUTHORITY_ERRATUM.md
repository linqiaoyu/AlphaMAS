# M2 Trader / Portfolio Authority Audit Erratum

Classification: **AUDIT_SEMANTICS_CORRECTION — NOT AN ARCHITECTURE CHANGE**.

M2-14 Retry #3 used an over-strict `AUDIT_ONLY` criterion.  It treated an
authoritative M2 Trader override followed by a Portfolio Manager decision equal
to the original Prompt Trader direction as proof that the Prompt proposal had
been silently restored.  That criterion conflated two deliberately separate
authority boundaries.

The frozen M2 contract remains:

`Prompt Trader → M2 RL Trader policy → authoritative M2 Trader handoff → unchanged Risk Debate → unchanged Portfolio Manager → unchanged execution/backtester`

The M2 Actor is authoritative for the Trader-slot handoff only.  It is not the
Risk layer, Portfolio Manager, Broker, or direct portfolio executor.  The
Portfolio Manager remains the final policy decision and may accept, modify, or
reject the M2 Trader proposal.  Consequently, Portfolio Manager agreement with
the original Prompt direction is not evidence of restoration when Risk and the
Portfolio Manager received the intact M2 handoff and execution followed the
Portfolio Manager output.

## Corrected failure definition

A silent Prompt restoration is present only when the M2 handoff or its
authoritative metadata is lost or replaced before downstream consumption.  The
permanent audit therefore fails on any of the following:

- the post-M2 `trader_investment_plan` becomes the Prompt-only proposal;
- Risk Debate or Portfolio Manager receives anything other than the exact M2
  handoff;
- the authoritative M2 action, source, override flag, or Prompt provenance
  identity is corrupted;
- execution does not equal the deterministic normalization of the Portfolio
  Manager's final decision.

It deliberately does not compare the executed action with `m2_rl_action` as a
validity condition.

## Retry #3 correction

The historical Experiments record
`experiments/M2/development/e2e_pilot_v1/retry3_blocked_audit.json` is preserved
byte-for-byte.  Its observations remain historical evidence; only its
interpretation is superseded.  Under the corrected structural criterion, ARR
and AEMD are correctness-valid because their differing final actions were
Portfolio Manager decisions on the intact M2 handoff, and execution followed
those decisions.  No trajectory was regenerated.

This correction was made from the frozen contracts, unchanged execution-source
control flow, and existing Retry #3 audit/publication artifacts.  The raw
trajectory bundle remained on the stopped EBS and was not reopened because this
resolution authorizes zero AWS starts.  No missing token or cost value is
invented.

## Invariance

No performance result caused this correction.  M0, M1, M2-06, M2-08, M2-09,
R3, M2-10A, C09, O08, Prompt Trader, Research Manager, Risk Debate, Portfolio
Manager, execution mapping, backtester, A1, A2, Formal protocol, and the E2E
population are unchanged.  Architecture identity remains
`35c5a46616f654ce70b0badc01cd59fb0afd433dcdcd99e5e0b8f2419ec4d153`.

Resolution resource use: zero Agent reruns, zero DeepSeek calls, zero Qwen
calls, zero AWS starts, zero GPU runtime, zero live market-data calls, and no raw
FinMultiTime access.
