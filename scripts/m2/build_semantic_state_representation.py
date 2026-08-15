#!/usr/bin/env python3
"""Build the canonical M2 semantic-state archive with a frozen local encoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.m2.semantic_state_representation import (
    ACTION_ORDER,
    CANDIDATE_DIMENSIONS,
    CONTRACT,
    MODEL_REPO,
    MODEL_REVISION,
    REFERENCE_DIMENSION,
    SOURCE_CONTRACT,
    SOURCE_CORPUS_IDENTITY,
    build_semantic_base,
    extract_economic_payload,
    mrl_prefix_normalize,
    select_embedding_dimension,
    sha256_bytes,
    text_sha256,
    validate_token_count,
)

MAX_TOKENS = 32768
EXPECTED_COUNTS = {"TRAIN": 56, "VALIDATION": 16}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(corpus: Path) -> list[dict[str, Any]]:
    manifest = read_json(corpus / "manifests/corpus_manifest.json")
    if manifest.get("contract") != SOURCE_CONTRACT:
        raise RuntimeError("source semantic contract differs")
    if manifest.get("semantic_handoff_trainval_corpus_identity_sha256") != SOURCE_CORPUS_IDENTITY:
        raise RuntimeError("source semantic corpus identity differs")
    if manifest.get("train") != 56 or manifest.get("validation") != 16:
        raise RuntimeError("source semantic population differs")
    if manifest.get("final_holdout", {}).get("materialised") != 0:
        raise RuntimeError("FINAL_HOLDOUT is materialised")
    if manifest.get("e2e_pilot", {}).get("materialised") != 0:
        raise RuntimeError("E2E_PILOT is materialised")
    rows: list[dict[str, Any]] = []
    counts = {"TRAIN": 0, "VALIDATION": 0}
    for index, case in enumerate(manifest["cases"]):
        role = case["role"]
        if role not in counts:
            raise RuntimeError("protected role appeared in materialised corpus manifest")
        relative = Path("cases") / role.lower() / case["case_id"].replace(":", "_") / "actor_visible_state.json"
        path = corpus / relative
        if file_sha256(path) != case["actor_visible_state_sha256"]:
            raise RuntimeError(f"actor state integrity differs: {case['case_id']}")
        state = read_json(path)
        payload = extract_economic_payload(state)
        rows.append(
            {
                "row_index": index,
                "case_id": case["case_id"],
                "role": role,
                "symbol": case["symbol"],
                "decision_session": case["decision_session"],
                "actor_state_sha256": case["actor_visible_state_sha256"],
                "manager_text": payload.research_manager_text,
                "trader_text": payload.prompt_trader_text,
                "action": payload.prompt_action,
            }
        )
        counts[role] += 1
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"source population differs: {counts}")
    return rows


def last_token_pool(hidden_states: Any, attention_mask: Any) -> Any:
    """Official Qwen3 left-padding-compatible last-token pooling path."""
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[range(batch_size), sequence_lengths]


def encode_texts(model: Any, tokenizer: Any, texts: list[str]) -> tuple[np.ndarray, list[int]]:
    import torch

    vectors: list[np.ndarray] = []
    token_counts: list[int] = []
    model.eval()
    with torch.inference_mode():
        for text in texts:  # canonical batch_size = 1
            encoded = tokenizer(text, padding=False, truncation=False, return_tensors="pt")
            count = int(encoded["input_ids"].shape[1])
            validate_token_count(count)
            token_counts.append(count)
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vector = pooled[0].detach().cpu().numpy().astype(np.float32)
            if vector.shape != (REFERENCE_DIMENSION,):
                raise RuntimeError(f"unexpected encoder dimension: {vector.shape}")
            if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1.0) > 1e-5:
                raise RuntimeError("invalid encoder output")
            vectors.append(vector)
    return np.stack(vectors), token_counts


def snapshot_manifest(snapshot: Path) -> tuple[list[dict[str, Any]], str]:
    files = []
    for path in sorted(candidate for candidate in snapshot.rglob("*") if candidate.is_file()):
        resolved = path.resolve()
        relative = path.relative_to(snapshot).as_posix()
        files.append({"path": relative, "size": resolved.stat().st_size, "sha256": file_sha256(resolved)})
    identity = sha256_bytes(json.dumps(files, separators=(",", ":"), sort_keys=True).encode())
    return files, identity


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def nvidia_driver_version() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0].strip()


def array_row_sha256(array: np.ndarray) -> list[str]:
    return [sha256_bytes(np.ascontiguousarray(row, dtype=np.float32).tobytes()) for row in array]


def save_diagnostics(output: Path, diagnostics: dict[str, Any], selected: int) -> None:
    payload = {
        "schema_version": "M2-DIMENSION-SELECTION-v1",
        "candidate_order": list(CANDIDATE_DIMENSIONS),
        "selected_dimension": selected,
        "all_candidate_metrics": diagnostics,
        "selection_algorithm": "smallest candidate passing every threshold in every TRAIN view",
        "performance_used": False,
        "reward_used": False,
        "validation_used_for_selection": False,
        "manual_override": False,
    }
    write_json(output / "diagnostics/dimension_selection.json", payload)
    write_json(output / "diagnostics/dimension_fidelity.json", diagnostics)
    csv_path = output / "diagnostics/dimension_fidelity.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dimension", "view", "spearman", "cosine_mae", "top5_overlap", "pass"))
        for dimension in CANDIDATE_DIMENSIONS:
            for view in ("manager", "trader", "joint"):
                metric = diagnostics[str(dimension)][view]
                writer.writerow((dimension, view, metric["spearman"], metric["cosine_mae"], metric["top5_overlap"], metric["pass"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    rows = load_rows(args.corpus)

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = args.snapshot or Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION))
    files, model_snapshot_identity = snapshot_manifest(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side="left")
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    train = [row for row in rows if row["role"] == "TRAIN"]
    validation = [row for row in rows if row["role"] == "VALIDATION"]
    train_manager, train_manager_tokens = encode_texts(model, tokenizer, [row["manager_text"] for row in train])
    train_trader, train_trader_tokens = encode_texts(model, tokenizer, [row["trader_text"] for row in train])
    selected, diagnostics = select_embedding_dimension(train_manager, train_trader)
    validation_manager, validation_manager_tokens = encode_texts(model, tokenizer, [row["manager_text"] for row in validation])
    validation_trader, validation_trader_tokens = encode_texts(model, tokenizer, [row["trader_text"] for row in validation])
    manager_full = np.concatenate((train_manager, validation_manager))
    trader_full = np.concatenate((train_trader, validation_trader))

    # Exactly one second complete inference run for the reproducibility audit.
    audit_manager, audit_manager_tokens = encode_texts(model, tokenizer, [row["manager_text"] for row in rows])
    audit_trader, audit_trader_tokens = encode_texts(model, tokenizer, [row["trader_text"] for row in rows])
    audit_selected, audit_diagnostics = select_embedding_dimension(audit_manager[:56], audit_trader[:56])
    maximum_difference = max(
        float(np.max(np.abs(manager_full - audit_manager))),
        float(np.max(np.abs(trader_full - audit_trader))),
    )
    minimum_cosine = min(
        float(np.min(np.sum(manager_full * audit_manager, axis=1))),
        float(np.min(np.sum(trader_full * audit_trader, axis=1))),
    )
    if maximum_difference > 1e-6 or minimum_cosine < 0.999999 or selected != audit_selected:
        raise RuntimeError("double-run encoder reproducibility tolerance failed")
    if train_manager_tokens + validation_manager_tokens != audit_manager_tokens:
        raise RuntimeError("manager token audit changed between runs")
    if train_trader_tokens + validation_trader_tokens != audit_trader_tokens:
        raise RuntimeError("trader token audit changed between runs")

    manager_selected = mrl_prefix_normalize(manager_full, selected)
    trader_selected = mrl_prefix_normalize(trader_full, selected)
    semantic_base = build_semantic_base(manager_selected, trader_selected, [row["action"] for row in rows])
    embeddings = args.output / "embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    np.save(embeddings / "research_manager_full_1024.npy", manager_full, allow_pickle=False)
    np.save(embeddings / "prompt_trader_full_1024.npy", trader_full, allow_pickle=False)
    np.save(embeddings / "research_manager_selected.npy", manager_selected, allow_pickle=False)
    np.save(embeddings / "prompt_trader_selected.npy", trader_selected, allow_pickle=False)
    np.save(embeddings / "semantic_base.npy", semantic_base, allow_pickle=False)
    save_diagnostics(args.output, diagnostics, selected)

    manager_tokens = train_manager_tokens + validation_manager_tokens
    trader_tokens = train_trader_tokens + validation_trader_tokens
    manager_full_hashes = array_row_sha256(manager_full)
    trader_full_hashes = array_row_sha256(trader_full)
    manager_selected_hashes = array_row_sha256(manager_selected)
    trader_selected_hashes = array_row_sha256(trader_selected)
    semantic_hashes = array_row_sha256(semantic_base)
    text_manifest = []
    semantic_manifest = []
    row_index = []
    for index, row in enumerate(rows):
        row_index.append({key: row[key] for key in ("row_index", "case_id", "role", "symbol", "decision_session", "actor_state_sha256")})
        text_manifest.append({
            "row_index": index,
            "case_id": row["case_id"],
            "role": row["role"],
            "research_manager_text_sha256": text_sha256(row["manager_text"]),
            "prompt_trader_text_sha256": text_sha256(row["trader_text"]),
            "research_manager_token_count": manager_tokens[index],
            "prompt_trader_token_count": trader_tokens[index],
        })
        semantic_manifest.append({
            **text_manifest[-1],
            "research_manager_full_embedding_sha256": manager_full_hashes[index],
            "prompt_trader_full_embedding_sha256": trader_full_hashes[index],
            "research_manager_selected_embedding_sha256": manager_selected_hashes[index],
            "prompt_trader_selected_embedding_sha256": trader_selected_hashes[index],
            "semantic_base_row_sha256": semantic_hashes[index],
            "prompt_action": row["action"],
        })
    write_json(args.output / "text_manifest.json", text_manifest)
    write_json(args.output / "manifests/row_index.json", row_index)
    write_json(args.output / "manifests/semantic_state_manifest.json", semantic_manifest)
    write_json(args.output / "model_snapshot_sha256.json", {"identity_sha256": model_snapshot_identity, "files": files})
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_driver": nvidia_driver_version(),
        "transformers": package_version("transformers"),
        "sentence_transformers": package_version("sentence-transformers"),
        "huggingface_hub": package_version("huggingface-hub"),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "attention_implementation": "eager",
        "deterministic_algorithms": True,
        "allow_tf32": False,
        "inference_dtype": "float32",
        "batch_size": 1,
        "padding_side": tokenizer.padding_side,
        "max_sequence_length": MAX_TOKENS,
    }
    environment_identity = sha256_bytes(json.dumps(environment, separators=(",", ":"), sort_keys=True).encode())
    environment["identity_sha256"] = environment_identity
    write_json(args.output / "encoder_environment.json", environment)
    write_json(args.output / "encoder_identity.json", {
        "repository": MODEL_REPO,
        "resolved_revision_sha": MODEL_REVISION,
        "model_snapshot_identity_sha256": model_snapshot_identity,
        "frozen": True,
        "fine_tuned": False,
        "custom_instruction": False,
    })
    write_json(args.output / "reproducibility_audit.json", {
        "canonical_runs": 2,
        "byte_identical": bool(maximum_difference == 0),
        "maximum_absolute_difference": maximum_difference,
        "minimum_corresponding_cosine": minimum_cosine,
        "selected_dimension_identical": selected == audit_selected,
        "audit_diagnostics": audit_diagnostics,
    })
    write_json(args.output / "runtime.json", {"wall_seconds": time.monotonic() - started})
    identity_payload = {
        "source_corpus_identity": SOURCE_CORPUS_IDENTITY,
        "contract": CONTRACT,
        "encoder_repo": MODEL_REPO,
        "encoder_revision": MODEL_REVISION,
        "model_snapshot_identity": model_snapshot_identity,
        "environment_identity": environment_identity,
        "selected_dimension": selected,
        "selection_rule": "smallest passing [256,512,1024] on every TRAIN view",
        "manager_selected_sha256": file_sha256(embeddings / "research_manager_selected.npy"),
        "trader_selected_sha256": file_sha256(embeddings / "prompt_trader_selected.npy"),
        "semantic_base_sha256": file_sha256(embeddings / "semantic_base.npy"),
        "action_order": ACTION_ORDER,
        "portfolio_features": ["is_cash", "is_long", "entry_log_return", "current_drawdown"],
    }
    write_json(args.output / "representation_identity.json", {
        "semantic_state_representation_identity_sha256": sha256_bytes(json.dumps(identity_payload, separators=(",", ":"), sort_keys=True).encode()),
        "bound_fields": identity_payload,
    })
    sha_manifest = {}
    for path in sorted(candidate for candidate in args.output.rglob("*") if candidate.is_file()):
        if path.name != "sha256.json":
            sha_manifest[path.relative_to(args.output).as_posix()] = file_sha256(path)
    write_json(args.output / "manifests/sha256.json", {"algorithm": "SHA-256", "files": sha_manifest})
    print(json.dumps({"selected_dimension": selected, "semantic_base_dimension": 3 * selected + 4, "actor_observation_dimension": 3 * selected + 8}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
