# M2 E2E Evidence-Store Wiring Erratum

M2-14 Retry #2 stopped before its first Agent decision because the resolved Graph
configuration omitted the frozen M2 pre-Formal evidence controls. The Graph therefore
received the default disabled value and selected the generic frozen M1 Formal evidence
store, which correctly had no `AAPL:2023-10-06` packet.

The eight exact M2 E2E packets were already present and byte-valid in the frozen M2-06
pre-Formal corpus. This correctness patch propagates
`m2_preformal_evidence_enabled`, `m2_preformal_evidence_identity`, and
`m2_preformal_evidence_role` through the central backtest-to-Graph configuration
boundary. The operational `m2_preformal_evidence_root` mount remains path-only.

No Agent trajectory, DeepSeek request, decision artifact, research output, or protected
evaluation access occurred before this correction. No packet, prompt, research method,
model parameter, reward, execution rule, E2E population, A1 definition, or A2 definition
changed. Formal M2 continues to keep the M2 E2E store disabled and uses the frozen M1
Formal evidence store.

Classification: **PRE-AGENT RUNTIME EVIDENCE-WIRING CORRECTNESS FIX — NO RESEARCH-SEMANTIC CHANGE**.
