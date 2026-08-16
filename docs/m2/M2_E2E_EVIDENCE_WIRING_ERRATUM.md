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

The first post-propagation dry execution then stopped before any LLM request because
the shared experiment-Memory namespace still admitted only the legacy M1 `PILOT` and
`FORMAL` scopes. The frozen M2 store deliberately identifies itself as
`M2_E2E_PILOT`; the namespace allowlist now admits that exact third scope while
continuing to reject every unknown scope. The store scope and packet bytes were not
changed.

After the eight official trajectories and ordinary run validation completed, final
Memory publication exposed the same distinction at a second non-Graph consumer: the
archiver derived its namespace from the unchanged legacy M1 `FORMAL` config fields
instead of the evidence store actually selected for M2 E2E. A shared effective-Memory
identity resolver now binds both final archiving and bundle validation to
`M2_E2E_PILOT` plus the frozen M2 corpus identity while leaving
`finmultitime_bundle_scope` and all Formal M2 semantics unchanged. The already-complete
trajectories were preserved and were not rerun.

No Agent trajectory, DeepSeek request, decision artifact, research output, or protected
evaluation access occurred before this correction. No packet, prompt, research method,
model parameter, reward, execution rule, E2E population, A1 definition, or A2 definition
changed. Formal M2 continues to keep the M2 E2E store disabled and uses the frozen M1
Formal evidence store.

Classification: **RUNTIME/PUBLICATION EVIDENCE-WIRING CORRECTNESS FIX — NO RESEARCH-SEMANTIC CHANGE**.
