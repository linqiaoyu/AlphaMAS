from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "docs/m2/m2_preformal_evidence_corpus_lineage_audit.json"


def test_preformal_evidence_lineage_audit_contract() -> None:
    audit = json.loads(AUDIT_PATH.read_text())

    assert audit["task_id"] == "M2-06A"
    assert audit["original_pre_qwen_builder_sha"] == (
        "791c88a7ba927cebe3713d893b8b111c5f54bde5"
    )
    assert audit["clean_final_corpus_reproducer_sha"] == (
        "3cef79b9536e4714cc5f532eada84f66fd6e8142"
    )
    assert audit["pre_qwen_inputs_sha"] == (
        "a9d3c554a9eb0ecf401a2064d12d5fb901e55165"
    )
    assert audit["frozen_evidence_corpus_sha"] == (
        "6b2406f1e12e1988c27b44880a1e153a9b750c2e"
    )
    assert audit["frozen_corpus_identity_sha256"] == (
        "3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420"
    )
    assert audit["structured_hash_mismatches"] == 0
    assert audit["rendered_hash_mismatches"] == 0
    assert audit["research_file_hash_mismatches"] == 0
    assert audit["qwen_calls"] == 0
    assert audit["deepseek_calls"] == 0
    assert audit["raw_dataset_accessed"] is False
    assert audit["frozen_corpus_modified"] is False
    assert audit["status"] == "PASS"
