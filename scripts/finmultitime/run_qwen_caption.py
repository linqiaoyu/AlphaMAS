#!/usr/bin/env python3
"""Run the frozen M1 Qwen caption adapter in isolated smoke or formal mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_REPO = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_CLASS = "Qwen3VLForConditionalGeneration"
PROCESSOR_CLASS = "Qwen3VLProcessor"
CAPTION_FIELDS = (
    "trend",
    "momentum_visual",
    "volatility_visual",
    "candlestick_structure",
    "notable_gap_or_reversal",
    "support_resistance_visual",
    "volume_visual",
    "other_visible_pattern",
    "confidence",
)
MAX_CAPTION_CHARS = 900


class FreezeValidationError(ValueError):
    """Raised when a frozen research identity does not match."""


class CaptionValidationError(ValueError):
    """Raised when raw model output violates the canonical caption contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FreezeValidationError(f"expected a JSON object: {path}")
    return value


def canonical_schema_json(fields: list[str] | tuple[str, ...]) -> str:
    return json.dumps(list(fields), ensure_ascii=False, separators=(",", ":"))


def load_and_validate_identities(
    contract_path: Path,
    environment_path: Path,
    qwen_manifest_path: Path,
    *,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = load_json(contract_path)
    environment = load_json(environment_path)
    qwen_manifest = load_json(qwen_manifest_path)

    expected_statuses = {"FROZEN"} if mode == "FORMAL" else {"FROZEN_FOR_SMOKE", "FROZEN"}
    if environment.get("status") not in expected_statuses:
        raise FreezeValidationError("Qwen environment is not frozen for the selected mode")

    qwen = environment.get("qwen", {})
    if qwen.get("repo_id") != MODEL_REPO or qwen.get("revision") != MODEL_REVISION:
        raise FreezeValidationError("exact Qwen repository/revision mismatch")
    if qwen.get("model_class") != MODEL_CLASS or qwen.get("processor_class") != PROCESSOR_CLASS:
        raise FreezeValidationError("Qwen model or processor class mismatch")

    adapter = contract.get("qwen_image_adapter", {})
    prompt = adapter.get("prompt")
    schema = adapter.get("caption_schema")
    if not isinstance(prompt, str) or schema != list(CAPTION_FIELDS):
        raise FreezeValidationError("Evidence Contract prompt/schema is invalid")
    if adapter.get("max_caption_chars") != MAX_CAPTION_CHARS:
        raise FreezeValidationError("Evidence Contract caption limit mismatch")

    identities = {
        "contract_version": contract.get("packet_version"),
        "contract_sha256": sha256_file(contract_path),
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_text(canonical_schema_json(schema)),
        "max_caption_chars": adapter.get("max_caption_chars"),
    }
    frozen_contract = environment.get("contract", {})
    for key, value in identities.items():
        if frozen_contract.get(key) != value:
            raise FreezeValidationError(f"environment/contract identity mismatch: {key}")

    manifest_expected = {
        "evidence_contract_version": identities["contract_version"],
        "evidence_contract_sha256": identities["contract_sha256"],
        "prompt_sha256": identities["prompt_sha256"],
        "caption_schema_sha256": identities["schema_sha256"],
        "max_caption_chars": identities["max_caption_chars"],
    }
    for key, value in manifest_expected.items():
        if qwen_manifest.get(key) != value:
            raise FreezeValidationError(f"Qwen input manifest mismatch: {key}")
    if qwen_manifest.get("caption_schema") != list(CAPTION_FIELDS):
        raise FreezeValidationError("Qwen input manifest schema/order mismatch")
    if qwen_manifest.get("caption_status_counts", {}).get("GENERATED") != 0:
        raise FreezeValidationError("formal captions already exist")

    generation = environment.get("generation", {})
    required_generation = {
        "device": "cuda",
        "dtype": "bfloat16",
        "batch_size": 1,
        "do_sample": False,
        "num_beams": 1,
        "python_seed": 0,
        "torch_seed": 0,
        "cuda_seed": 0,
    }
    for key, value in required_generation.items():
        if generation.get(key) != value:
            raise FreezeValidationError(f"generation setting mismatch: {key}")
    if not isinstance(generation.get("max_new_tokens"), int):
        raise FreezeValidationError("max_new_tokens must be frozen")

    return contract, environment, qwen_manifest


def build_model_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    """Build the only allowed model input: one image and the exact frozen prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptionValidationError(f"duplicate caption key: {key}")
        result[key] = value
    return result


def _parse_caption_object(raw_output: str) -> dict[str, Any]:
    stripped = raw_output.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped, object_pairs_hook=_unique_object)
        except CaptionValidationError:
            raise
        except json.JSONDecodeError as exc:
            raise CaptionValidationError("caption is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise CaptionValidationError("caption JSON must be an object")
        return parsed

    parsed_lines: dict[str, Any] = {}
    for line in stripped.splitlines():
        if not line.strip() or ":" not in line:
            raise CaptionValidationError("narrative or malformed content outside schema")
        key, value = line.split(":", 1)
        key = key.strip().strip('"')
        if key in parsed_lines:
            raise CaptionValidationError(f"duplicate caption key: {key}")
        parsed_lines[key] = value.strip()
    return parsed_lines


def parse_canonical_caption(raw_output: str) -> str:
    parsed = _parse_caption_object(raw_output)
    actual = set(parsed)
    expected = set(CAPTION_FIELDS)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise CaptionValidationError(f"caption keys mismatch; missing={missing}, unknown={unknown}")

    canonical: dict[str, str] = {}
    for key in CAPTION_FIELDS:
        value = parsed[key]
        if not isinstance(value, str) or not value.strip():
            raise CaptionValidationError(f"caption value must be a non-empty string: {key}")
        canonical[key] = " ".join(value.split())
    serialized = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) > MAX_CAPTION_CHARS:
        raise CaptionValidationError("canonical caption exceeds 900 characters")
    return serialized


def validate_caption_content(canonical_caption: str) -> None:
    lowered = canonical_caption.casefold()
    prohibited = (
        r"\b(?:buy|hold|sell)\b",
        r"price\s+target",
        r"future\s+return",
        r"\bforecast(?:s|ed|ing)?\b",
        r"\bpredict(?:s|ed|ion|ions|ing)?\b",
    )
    for pattern in prohibited:
        if re.search(pattern, lowered):
            raise CaptionValidationError(f"prohibited caption content: {pattern}")


def current_git_head(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_mode_and_image(
    mode: str,
    image_path: Path,
    environment: dict[str, Any],
    qwen_manifest: dict[str, Any],
    *,
    captioning_code_sha: str | None,
    repository: Path,
) -> None:
    image_sha = sha256_file(image_path)
    formal_hashes = {item["sha256"] for item in qwen_manifest.get("images", [])}
    if mode == "SMOKE":
        if image_sha in formal_hashes:
            raise FreezeValidationError("SMOKE mode refuses formal image bytes")
        return

    if image_sha not in formal_hashes:
        raise FreezeValidationError("FORMAL mode accepts only frozen formal image bytes")
    if not captioning_code_sha or not re.fullmatch(r"[0-9a-f]{40}", captioning_code_sha):
        raise FreezeValidationError("FORMAL mode requires a frozen captioning_code_sha")
    if current_git_head(repository) != captioning_code_sha:
        raise FreezeValidationError("running Git HEAD does not match captioning_code_sha")
    if environment.get("status") != "FROZEN" or not environment.get("smoke", {}).get(
        "canonical_outputs_identical"
    ):
        raise FreezeValidationError("FORMAL mode requires a repeatability-validated environment")


def run_inference(
    image_path: Path,
    contract: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    generation = environment["generation"]
    qwen = environment["qwen"]
    snapshot = Path(qwen["local_snapshot_path"])
    if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
        raise FreezeValidationError("resolved model snapshot path/revision mismatch")

    random.seed(generation["python_seed"])
    torch.manual_seed(generation["torch_seed"])
    torch.cuda.manual_seed_all(generation["cuda_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    if processor.__class__.__name__ != PROCESSOR_CLASS:
        raise FreezeValidationError("resolved processor class mismatch")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
    )
    model.eval()

    prompt = contract["qwen_image_adapter"]["prompt"]
    messages = build_model_messages(image_path, prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=generation["max_new_tokens"],
            use_cache=True,
        )
    torch.cuda.synchronize()
    duration_seconds = time.perf_counter() - wall_start
    ended_at = datetime.now(UTC)
    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    canonical = parse_canonical_caption(raw_output)
    validate_caption_content(canonical)
    return {
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "duration_seconds": round(duration_seconds, 6),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "canonical_caption": canonical,
        "canonical_caption_sha256": sha256_text(canonical),
        "caption_length_chars": len(canonical),
        "schema_valid": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("SMOKE", "FORMAL"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--environment-freeze", type=Path, required=True)
    parser.add_argument("--qwen-input-manifest", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--captioning-code-sha")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FreezeValidationError(f"image does not exist: {args.image}")
    contract, environment, qwen_manifest = load_and_validate_identities(
        args.contract,
        args.environment_freeze,
        args.qwen_input_manifest,
        mode=args.mode,
    )
    validate_mode_and_image(
        args.mode,
        args.image,
        environment,
        qwen_manifest,
        captioning_code_sha=args.captioning_code_sha,
        repository=args.repository,
    )
    result = {
        "mode": args.mode,
        "image_sha256": sha256_file(args.image),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        **run_inference(args.image, contract, environment),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
