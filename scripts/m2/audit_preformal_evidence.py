#!/usr/bin/env python3
"""Fail-closed audit of an M2-06 pre-Qwen or final evidence archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.m2.build_preformal_evidence_corpus import (  # noqa: E402
    EXPECTED_ROLE_COUNTS,
    EvidenceBuildError,
    _pit_violations,
    _walk_forbidden,
    read_json,
    source_mutation_report,
)


def audit(root: Path) -> dict[str, object]:
    packet_paths = sorted((root / "packets/structured").glob("**/*.json"))
    if packet_paths:
        packets = [read_json(path) for path in packet_paths]
        if len(packets) != 96:
            raise EvidenceBuildError("final archive does not contain 96 packets")
        cases = packets
        forbidden = [{"case_id": p["case_id"], "paths": _walk_forbidden(p)} for p in packets]
        forbidden = [item for item in forbidden if item["paths"]]
    else:
        skeleton_paths = sorted((root / "inputs/pre_qwen_packet_skeletons").glob("**/*.json"))
        cases = [read_json(path) for path in skeleton_paths]
        if len(cases) != 96:
            raise EvidenceBuildError("pre-Qwen archive does not contain 96 skeletons")
        forbidden = []
    role_counts = {}
    for role in EXPECTED_ROLE_COUNTS:
        role_counts[role] = sum((case.get("role") or case.get("split_role")) == role for case in cases)
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise EvidenceBuildError("role count drift")
    normalized = [case if "sections" in case else {"case_id": case["case_id"], "decision_session": case["decision_session"], "sections": {name: case[name] for name in ("TEXT", "TABLE", "TIME_SERIES", "IMAGE")}} for case in cases]
    pit = _pit_violations(normalized)
    provenance = read_json(root / "manifests/source_provenance.json")
    mutation = source_mutation_report(provenance)
    status = "PASS" if not pit and not forbidden and mutation["status"] == "PASS" else "FAIL"
    return {"status": status, "case_count": 96, "role_counts": role_counts, "pit_violations": pit, "future_label_violations": forbidden, "source_mutation": mutation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
