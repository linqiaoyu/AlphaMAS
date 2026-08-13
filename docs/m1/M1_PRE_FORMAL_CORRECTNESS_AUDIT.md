# M1 Pre-Formal Correctness Audit

**Overall status:** `PASS`
**Contract:** `M1-FINMULTITIME-v1.0.2`
**Scope:** corrected M1 preprocessing, frozen Qwen-caption integration, packet serialization, and protocol/PIT checks before Formal M1.

Formal M1 was not run. A `NOT_APPLICABLE` execution-boundary check records that this is intentionally a pre-formal audit.

| Check | Status | Evidence summary |
|---|---|---|
| contract_and_erratum_identity | PASS | `{"audit_sha256": "abcc33b4e68a85ca7aceae3ab268f91b957ca931466257bb55b1e20fa3c5f397", "contract_sha256": "46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08", "contract_version": "M1-FINMULTITIME-v1.0.2"}` |
| TEXT_source_integrity_and_fail_closed_policy | PASS | `{"audit_record_status_counts": {"VERIFIED_MATCH": 8, "VERIFIED_MISMATCH": 8}, "case_selected_text_record_count": 16, "case_text_available_by_symbol": {"AMZN": 2}, "no_external_replacement": true}` |
| TABLE_PIT_and_source_provenance | PASS | `{"status_counts": {"AVAILABLE": 78}, "violations": []}` |
| TIME_SERIES_PIT_history_and_summary_contract | PASS | `{"required_rows": 61, "status_counts": {"AVAILABLE": 78}}` |
| IMAGE_bytes_and_Qwen_caption_provenance | PASS | `{"caption_hashes": {"AMZN": "131c2904cd82c850f94b00cc58e2b97b2ef69c7e805fa562ef2c8f2e00a568fe", "JPM": "44e980fa7768c0cf36961d6c42ea769157e4b989d02b60b37618256028b04d70"}, "image_hashes": {"AMZN": "215c01f8a03dc55719558644992b14c28d8b9f604d44e3504ef7df0f99997bb8", "JPM": "b9e218ab002a8bfd6c0cbf86dd53abe9bcfeda4744a785754cb1b9173cc85611"}, "status_counts": {"AVAILABLE": 52, "UNAVAILABLE": 26}}` |
| PIT_and_future_leakage_global | PASS | `{"future_field_violation_count": 0, "pit_violation_count": 0}` |
| routing_and_packet_serialization | PASS | `{"case_count": 78, "json_packet_count": 78, "packet_violations": [], "routing_violations": [], "text_packet_count": 78}` |
| deterministic_build_and_manifests | PASS | `{"deterministic_build": {"json_hash_mismatches": 0, "packets_compared": 78, "status": "PASS", "text_hash_mismatches": 0}, "packet_count": 78, "qwen_input_caption_status_counts": {"GENERATED": 52, "NOT_APPLICABLE": 26, "PENDING": 0}}` |
| formal_m1_execution_boundary | PASS | `{"formal_m1_run": false, "result": "NOT_APPLICABLE — pre-formal audit only"}` |

## Modality counts

| Modality | Status counts |
|---|---|
| TEXT | `{"AVAILABLE": 2, "UNAVAILABLE": 76}` |
| TABLE | `{"AVAILABLE": 78}` |
| TIME_SERIES | `{"AVAILABLE": 78}` |
| IMAGE | `{"AVAILABLE": 52, "UNAVAILABLE": 26}` |

TEXT source-integrity audit SHA-256: `abcc33b4e68a85ca7aceae3ab268f91b957ca931466257bb55b1e20fa3c5f397`.
Contract SHA-256: `46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08`.

No raw FinMultiTime file was modified, no external article was substituted, no future label or outcome was admitted, and no formal M1 result is represented by this audit.
