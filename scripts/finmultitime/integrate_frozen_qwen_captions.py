#!/usr/bin/env python3
"""Integrate frozen Qwen caption identities without running Qwen.

The script copies only already-frozen caption/image metadata, validates the
canonical caption and image hashes, and updates the 78 preprocessed case
records deterministically.  It never imports a model or provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "finmultitime_3stocks_2024h1_v1"
EXPECTED_CAPTIONS = {
    "AMZN": ("amzn_2023_H2_candlestick.png", "131c2904cd82c850f94b00cc58e2b97b2ef69c7e805fa562ef2c8f2e00a568fe"),
    "JPM": ("jpm_2023_H2_candlestick.png", "44e980fa7768c0cf36961d6c42ea769157e4b989d02b60b37618256028b04d70"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_frozen_qwen(source_root: Path, destination_root: Path, contract_version: str, contract_sha: str) -> None:
    source_qwen = source_root / "qwen"
    destination_qwen = destination_root / "qwen"
    destination_qwen.mkdir(parents=True, exist_ok=True)
    (destination_qwen / "captions").mkdir(parents=True, exist_ok=True)
    for path in (source_qwen / "caption_manifest.json", source_qwen / "qwen_input_manifest.json"):
        shutil.copy2(path, destination_qwen / path.name)
    for path in sorted((source_qwen / "captions").glob("*.json")):
        shutil.copy2(path, destination_qwen / "captions" / path.name)
    for path in [destination_qwen / "caption_manifest.json", destination_qwen / "qwen_input_manifest.json", *sorted((destination_qwen / "captions").glob("*.json"))]:
        value = read_json(path)
        if "evidence_contract_version" in value:
            value["evidence_contract_version"] = contract_version
        if "evidence_contract_sha256" in value:
            value["evidence_contract_sha256"] = contract_sha
        if path.name in {"caption_manifest.json", "qwen_input_manifest.json"}:
            value["pre_formal_correctness_erratum"] = "Qwen canonical caption reused unchanged; only contract wrapper provenance updated"
        write_json(path, value)


def integrate(processed_root: Path, *, frozen_qwen_root: Path | None = None) -> dict[str, Any]:
    contract = read_json(REPO_ROOT / "docs/m1/m1_evidence_contract.json")
    contract_version = contract["packet_version"]
    contract_sha = sha256_file(REPO_ROOT / "docs/m1/m1_evidence_contract.json")
    if frozen_qwen_root is not None:
        copy_frozen_qwen(frozen_qwen_root, processed_root, contract_version, contract_sha)

    caption_manifest = read_json(processed_root / "qwen/caption_manifest.json")
    qwen_input_manifest = read_json(processed_root / "qwen/qwen_input_manifest.json")
    if qwen_input_manifest.get("caption_status_counts") != {"GENERATED": 52, "NOT_APPLICABLE": 26, "PENDING": 0}:
        raise ValueError("frozen Qwen input manifest does not prove 52/26/0 states")
    identities = {}
    for item in caption_manifest.get("captions", []):
        symbol = item["symbol"]
        filename, expected_sha = EXPECTED_CAPTIONS.get(symbol, (None, None))
        if filename is None or item.get("image_filename") != filename:
            raise ValueError(f"unexpected frozen caption identity for {symbol}")
        if item.get("canonical_caption_sha256") != expected_sha:
            raise ValueError(f"canonical caption SHA drift for {symbol}")
        caption_path = processed_root / "qwen" / Path(item["caption_ref"]).relative_to("qwen")
        caption = read_json(caption_path)
        canonical = caption.get("canonical_caption")
        if not isinstance(canonical, str) or sha256_bytes(canonical.encode("utf-8")) != expected_sha:
            raise ValueError(f"frozen canonical caption content drift for {symbol}")
        image_path = processed_root / "image" / filename
        if sha256_file(image_path) != item["image_sha256"]:
            raise ValueError(f"frozen image SHA drift for {symbol}")
        identities[symbol] = {"caption_ref": item["caption_ref"], "caption_sha256": expected_sha}

    cases = sorted((processed_root / "cases").glob("*/*.json"))
    if len(cases) != 78:
        raise ValueError(f"expected 78 preprocessed cases, found {len(cases)}")
    changes = []
    for path in cases:
        case = read_json(path)
        image = case["IMAGE"]
        before = (image.get("caption_status"), image.get("caption_ref"), image.get("caption_sha256"))
        if image["status"] == "UNAVAILABLE":
            image.update({"caption_status": "NOT_APPLICABLE", "caption_ref": None, "caption_sha256": None})
        else:
            identity = identities.get(case["symbol"])
            if identity is None:
                raise ValueError(f"available image lacks frozen caption identity: {path}")
            image.update({"caption_status": "GENERATED", **identity})
        after = (image.get("caption_status"), image.get("caption_ref"), image.get("caption_sha256"))
        if before != after:
            changes.append({"case": f"{case['symbol']}:{case['decision_session']}", "before": before, "after": after})
        write_json(path, case)

    statuses = Counter()
    for path in cases:
        statuses[read_json(path)["IMAGE"]["caption_status"]] += 1
    if statuses != Counter({"GENERATED": 52, "NOT_APPLICABLE": 26}):
        raise ValueError(f"caption integration status mismatch: {statuses}")
    return {
        "schema_version": "1.0",
        "verdict": "PASS",
        "contract_version": contract_version,
        "contract_sha256": contract_sha,
        "cases_compared": len(cases),
        "caption_status_counts": dict(sorted(statuses.items())),
        "caption_reference_sha_counts": {
            identity["caption_sha256"]: sum(
                read_json(path)["IMAGE"].get("caption_sha256") == identity["caption_sha256"]
                for path in cases
            )
            for identity in identities.values()
        },
        "canonical_caption_hashes_unchanged": True,
        "image_hashes_unchanged": True,
        "qwen_inference_calls": 0,
        "case_changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data/processed" / DATASET_ID)
    parser.add_argument("--frozen-qwen-root", type=Path)
    args = parser.parse_args()
    report = integrate(args.processed_root.resolve(), frozen_qwen_root=args.frozen_qwen_root.resolve() if args.frozen_qwen_root else None)
    write_json(REPO_ROOT / "docs/m1/m1_qwen_formal_caption_integration_equivalence.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
