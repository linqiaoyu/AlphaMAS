#!/usr/bin/env python3
"""Run the frozen M1 Qwen implementation once per new M2-06 image SHA."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.run_qwen_caption import (  # noqa: E402
    MODEL_REPO,
    MODEL_REVISION,
    run_inference,
    sha256_file,
    sha256_text,
)
from scripts.m2.build_preformal_evidence_corpus import (  # noqa: E402
    PROMPT_SHA256,
    SCHEMA_SHA256,
    SYNTHETIC_CANONICAL_SHA256,
    SYNTHETIC_IMAGE_SHA256,
    SYNTHETIC_RAW_SHA256,
    EvidenceBuildError,
    read_json,
    write_json,
)

EXPECTED_PACKAGES = {
    "python": "3.13.14",
    "torch": "2.12.1+cu130",
    "transformers": "4.57.6",
    "huggingface_hub": "0.36.2",
    "safetensors": "0.8.0",
    "pillow": "12.3.0",
    "torchvision": "0.27.1+cu130",
    "accelerate": "1.14.0",
    "tokenizers": "0.22.2",
}


def validate_environment(contract: dict[str, Any], environment: dict[str, Any], model_snapshot: Path) -> dict[str, Any]:
    qwen = environment["qwen"]
    if qwen["repo_id"] != MODEL_REPO or qwen["revision"] != MODEL_REVISION:
        raise EvidenceBuildError("frozen Qwen model/revision mismatch")
    if environment["contract"]["prompt_sha256"] != PROMPT_SHA256 or environment["contract"]["schema_sha256"] != SCHEMA_SHA256:
        raise EvidenceBuildError("frozen prompt/schema mismatch")
    prompt = contract["qwen_image_adapter"]["prompt"]
    schema = contract["qwen_image_adapter"]["caption_schema"]
    if sha256_text(prompt) != PROMPT_SHA256:
        raise EvidenceBuildError("contract prompt identity mismatch")
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    if sha256_text(schema_text) != SCHEMA_SHA256:
        raise EvidenceBuildError("contract schema identity mismatch")
    frozen_packages = environment["environment"]
    for name, expected in EXPECTED_PACKAGES.items():
        if str(frozen_packages.get(name)) != expected:
            raise EvidenceBuildError(f"frozen package mismatch: {name}")
    generation = environment["generation"]
    required = {
        "dtype": "bfloat16", "batch_size": 1, "do_sample": False,
        "num_beams": 1, "max_new_tokens": 256, "python_seed": 0,
        "torch_seed": 0, "cuda_seed": 0, "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    for name, expected in required.items():
        if generation.get(name) != expected:
            raise EvidenceBuildError(f"generation setting mismatch: {name}")
    if not model_snapshot.is_dir() or model_snapshot.name != MODEL_REVISION:
        raise EvidenceBuildError("exact local model snapshot missing")
    result = json.loads(json.dumps(environment))
    result["qwen"]["local_snapshot_path"] = str(model_snapshot)
    return result


def validate_smoke(result: dict[str, Any], synthetic_image: Path) -> None:
    required = {
        "image": (sha256_file(synthetic_image), SYNTHETIC_IMAGE_SHA256),
        "raw": (result["raw_output_sha256"], SYNTHETIC_RAW_SHA256),
        "canonical": (result["canonical_caption_sha256"], SYNTHETIC_CANONICAL_SHA256),
    }
    failures = [name for name, (actual, expected) in required.items() if actual != expected]
    if failures:
        raise EvidenceBuildError(f"historical synthetic Qwen identity failed: {failures}")


def _durable_caption_write(root: Path, item: dict[str, Any], result: dict[str, Any], source: str) -> dict[str, Any]:
    image_sha = item["image_sha256"]
    raw_path = root / "raw_outputs" / f"{image_sha}.txt"
    canonical_path = root / "canonical_captions" / f"{image_sha}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(result["raw_output"], encoding="utf-8")
    canonical_path.write_text(result["canonical_caption"], encoding="utf-8")
    if sha256_text(raw_path.read_text(encoding="utf-8")) != result["raw_output_sha256"]:
        raise EvidenceBuildError("durable raw output identity mismatch")
    if sha256_text(canonical_path.read_text(encoding="utf-8")) != result["canonical_caption_sha256"]:
        raise EvidenceBuildError("durable canonical output identity mismatch")
    return {
        "image_id": item["image_id"],
        "image_filename": item["image_filename"],
        "image_sha256": image_sha,
        "caption_source": source,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "prompt_sha256": PROMPT_SHA256,
        "schema_sha256": SCHEMA_SHA256,
        "raw_output": result["raw_output"],
        "raw_output_sha256": result["raw_output_sha256"],
        "canonical_caption": result["canonical_caption"],
        "canonical_caption_sha256": result["canonical_caption_sha256"],
        "duration_seconds": result.get("duration_seconds", 0),
        "started_at_utc": result.get("started_at_utc"),
        "ended_at_utc": result.get("ended_at_utc"),
        "schema_valid": True,
        "inference_count": 0 if source == "REUSED_FROZEN_M1" else 1,
    }


def run_batch(pre_qwen_root: Path, output: Path, contract_path: Path, environment_path: Path, model_snapshot: Path, synthetic_image: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise EvidenceBuildError("Qwen output must start empty")
    output.mkdir(parents=True, exist_ok=True)
    contract = read_json(contract_path)
    environment = validate_environment(contract, read_json(environment_path), model_snapshot)
    if sha256_file(synthetic_image) != SYNTHETIC_IMAGE_SHA256:
        raise EvidenceBuildError("historical synthetic image identity mismatch")
    smoke = run_inference(synthetic_image, contract, environment)
    validate_smoke(smoke, synthetic_image)
    write_json(output / "synthetic_compatibility.json", {
        "status": "PASS", "image_sha256": SYNTHETIC_IMAGE_SHA256,
        "raw_output_sha256": smoke["raw_output_sha256"],
        "canonical_caption_sha256": smoke["canonical_caption_sha256"],
    })
    manifest = read_json(pre_qwen_root / "manifests/unique_image_manifest.json")
    captions = []
    for item in manifest["images"]:
        if item["qwen_status"] == "REUSED_FROZEN":
            frozen = item["reused_caption"]
            result = {
                "raw_output": frozen.get("raw_model_output", frozen.get("raw_output", "")),
                "raw_output_sha256": frozen["raw_output_sha256"],
                "canonical_caption": frozen["canonical_caption"],
                "canonical_caption_sha256": frozen["canonical_caption_sha256"],
            }
            captions.append(_durable_caption_write(output, item, result, "REUSED_FROZEN_M1"))
            continue
        staged = pre_qwen_root / item["staged_path"]
        if sha256_file(staged) != item["image_sha256"]:
            raise EvidenceBuildError("research image identity changed before Qwen")
        result = run_inference(staged, contract, environment)
        captions.append(_durable_caption_write(output, item, result, "NEW_QWEN_M2_06"))
        # Checkpoint after every first complete output for safe infrastructure resume evidence.
        write_json(output / "caption_manifest.partial.json", {"captions": captions})
    final = {
        "schema_version": "1.0",
        "task_id": "M2-06",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "prompt_sha256": PROMPT_SHA256,
        "schema_sha256": SCHEMA_SHA256,
        "generation_settings": environment["generation"],
        "official_caption_policy": "first_complete_correctness_valid_output",
        "manual_caption_editing": False,
        "multiple_outputs_cherry_picked": False,
        "unique_eligible_images": len(captions),
        "reused_frozen_m1_captions": sum(item["caption_source"] == "REUSED_FROZEN_M1" for item in captions),
        "new_qwen_images": sum(item["caption_source"] == "NEW_QWEN_M2_06" for item in captions),
        "successful_qwen_calls": sum(item["inference_count"] for item in captions),
        "captions": captions,
    }
    if final["successful_qwen_calls"] != final["new_qwen_images"]:
        raise EvidenceBuildError("Qwen call accounting mismatch")
    write_json(output / "caption_manifest.json", final)
    if (output / "caption_manifest.partial.json").exists():
        (output / "caption_manifest.partial.json").unlink()
    sha = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "sha256.json"):
        sha[path.relative_to(output).as_posix()] = sha256_file(path)
    write_json(output / "sha256.json", sha)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-qwen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=REPO_ROOT / "docs/m1/m1_evidence_contract.json")
    parser.add_argument("--environment", type=Path, default=REPO_ROOT / "docs/m1/m1_qwen_aws_environment_freeze.json")
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--synthetic-image", type=Path, required=True)
    args = parser.parse_args()
    result = run_batch(args.pre_qwen_root, args.output, args.contract, args.environment, args.model_snapshot, args.synthetic_image)
    print(json.dumps({key: result[key] for key in ("unique_eligible_images", "reused_frozen_m1_captions", "new_qwen_images", "successful_qwen_calls")}, sort_keys=True))


if __name__ == "__main__":
    main()
