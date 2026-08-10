"""Immutable final-memory artifacts for completed experiment backtests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tradingagents.backtesting.recorder import write_json
from tradingagents.dataflows.utils import safe_ticker_component

MEMORY_ARCHIVE_SCHEMA_VERSION = "1.0"
MEMORY_ARCHIVE_MANIFEST_PATH = "memory/manifest.json"
MEMORY_ARCHIVE_FORMAT = "trading-memory-log-markdown-v1"


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity_component(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    try:
        return safe_ticker_component(value, max_len=128)
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc


def _safe_symbols(symbols: Iterable[str]) -> list[str]:
    safe_symbols = [safe_ticker_component(symbol) for symbol in symbols]
    if not safe_symbols:
        raise ValueError("memory archive requires at least one symbol")
    if len(set(safe_symbols)) != len(safe_symbols):
        raise ValueError("memory archive symbols must be unique")
    return safe_symbols


def runtime_experiment_memory_path(
    runtime_memory_dir: str | Path, *, experiment_id: str,
    graph_config_sha256: str, memory_lineage_id: str, symbol: str,
) -> Path:
    """Resolve the existing Graph experiment-memory file without mutating it."""
    experiment = _identity_component(experiment_id, field="experiment_id")
    lineage = _identity_component(memory_lineage_id, field="memory_lineage_id")
    if not _valid_sha256(graph_config_sha256):
        raise ValueError("graph_config_sha256 must be a valid SHA-256")
    safe_symbol = safe_ticker_component(symbol)
    return (
        Path(runtime_memory_dir)
        / "historical_memory"
        / experiment
        / graph_config_sha256
        / lineage
        / f"{safe_symbol}.md"
    )


def archive_final_experiment_memory(
    *, run_dir: str | Path, runtime_memory_dir: str | Path,
    experiment_id: str, run_id: str, memory_lineage_id: str,
    memory_lifecycle: str, memory_resumed_from_run_id: str | None,
    graph_config_sha256: str, symbols: Iterable[str],
) -> dict[str, Any]:
    """Copy final runtime Memory into a run-local, content-addressed artifact.

    The copy direction is deliberately one-way. Runtime execution and resume
    continue to use the lineage namespace beneath ``runtime_memory_dir``; no
    runtime code reads the returned archive paths.
    """
    run_root = Path(run_dir)
    runtime_root = Path(runtime_memory_dir)
    run_resolved = run_root.resolve()
    runtime_resolved = runtime_root.resolve()
    if (
        run_resolved == runtime_resolved
        or run_resolved in runtime_resolved.parents
        or runtime_resolved in run_resolved.parents
    ):
        raise ValueError("runtime Memory and immutable run bundle must be separate")

    experiment = _identity_component(experiment_id, field="experiment_id")
    current_run = _identity_component(run_id, field="run_id")
    lineage = _identity_component(memory_lineage_id, field="memory_lineage_id")
    if memory_lifecycle not in {"independent_fresh", "force_fresh", "resume"}:
        raise ValueError(f"invalid memory_lifecycle {memory_lifecycle!r}")
    resumed_from = (
        _identity_component(
            memory_resumed_from_run_id, field="memory_resumed_from_run_id"
        )
        if memory_resumed_from_run_id is not None else None
    )
    if memory_lifecycle == "resume" and resumed_from is None:
        raise ValueError("resume memory archive requires memory_resumed_from_run_id")
    if memory_lifecycle != "resume" and resumed_from is not None:
        raise ValueError("fresh memory archive cannot have memory_resumed_from_run_id")
    if memory_lifecycle != "resume" and lineage != current_run:
        raise ValueError("fresh memory archive lineage must equal run_id")
    if not _valid_sha256(graph_config_sha256):
        raise ValueError("graph_config_sha256 must be a valid SHA-256")
    safe_symbols = _safe_symbols(symbols)

    # Read and validate every source before creating the archive directory. A
    # successful run cannot be published with a partial symbol archive.
    payloads: dict[str, bytes] = {}
    source_errors: list[str] = []
    for symbol in safe_symbols:
        source = runtime_experiment_memory_path(
            runtime_root,
            experiment_id=experiment,
            graph_config_sha256=graph_config_sha256,
            memory_lineage_id=lineage,
            symbol=symbol,
        )
        try:
            payload = source.read_bytes()
        except OSError as exc:
            source_errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
            continue
        if not payload:
            source_errors.append(f"{symbol}: runtime Memory is empty")
            continue
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            source_errors.append(f"{symbol}: runtime Memory is not valid UTF-8")
            continue
        payloads[symbol] = payload
    if source_errors:
        raise ValueError(
            "cannot archive complete final experiment Memory: "
            + "; ".join(source_errors)
        )

    archive_root = run_root / "memory"
    archive_root.mkdir(parents=True, exist_ok=False)
    symbol_records: list[dict[str, Any]] = []
    for symbol in safe_symbols:
        relative = f"memory/symbols/{symbol}.md"
        destination = run_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = payloads[symbol]
        destination.write_bytes(payload)
        symbol_records.append({
            "symbol": symbol,
            "path": relative,
            "format": MEMORY_ARCHIVE_FORMAT,
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        })

    archive_manifest = {
        "schema_version": MEMORY_ARCHIVE_SCHEMA_VERSION,
        "artifact_type": "final_experiment_memory",
        "artifact_role": "read_only_research_artifact",
        "runtime_consumed": False,
        "experiment_id": experiment,
        "run_id": current_run,
        "memory_lineage_id": lineage,
        "memory_lifecycle": memory_lifecycle,
        "memory_resumed_from_run_id": resumed_from,
        "graph_config_sha256": graph_config_sha256,
        "runtime_namespace": {
            "storage_role": "operational_runtime_only",
            "layout_version": "historical-memory-v1",
            "experiment_id": experiment,
            "memory_lineage_id": lineage,
            "graph_config_sha256": graph_config_sha256,
        },
        "symbols": symbol_records,
    }
    archive_manifest_path = run_root / MEMORY_ARCHIVE_MANIFEST_PATH
    write_json(archive_manifest_path, archive_manifest)
    manifest_payload = archive_manifest_path.read_bytes()
    return {
        "schema_version": MEMORY_ARCHIVE_SCHEMA_VERSION,
        "manifest_path": MEMORY_ARCHIVE_MANIFEST_PATH,
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "symbol_count": len(symbol_records),
    }


def _bundle_member(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    relative = Path(value)
    if relative.is_absolute():
        return None
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        return None
    return candidate


def validate_final_memory_archive(
    *, run_dir: str | Path, descriptor: Any, experiment_id: Any,
    run_id: Any, memory_lineage_id: Any, memory_lifecycle: Any,
    memory_resumed_from_run_id: Any, graph_config_sha256: Any,
    symbols: Iterable[str],
) -> dict[str, list[str]]:
    """Validate archive completeness, provenance, isolation, and checksums."""
    root = Path(run_dir)
    errors: list[str] = []
    missing_files: list[str] = []
    checksum_errors: list[str] = []
    try:
        expected_symbols = _safe_symbols(symbols)
    except ValueError as exc:
        return {
            "errors": [str(exc)],
            "missing_files": [],
            "checksum_errors": [],
        }

    if not isinstance(descriptor, Mapping):
        return {
            "errors": ["manifest.json is missing memory_archive provenance"],
            "missing_files": [MEMORY_ARCHIVE_MANIFEST_PATH],
            "checksum_errors": [],
        }
    if descriptor.get("schema_version") != MEMORY_ARCHIVE_SCHEMA_VERSION:
        errors.append("memory_archive descriptor has an unsupported schema_version")
    if descriptor.get("manifest_path") != MEMORY_ARCHIVE_MANIFEST_PATH:
        errors.append("memory_archive descriptor has a non-canonical manifest_path")
    if descriptor.get("symbol_count") != len(expected_symbols):
        errors.append("memory_archive descriptor symbol_count does not match the run")
    expected_manifest_hash = descriptor.get("manifest_sha256")
    if not _valid_sha256(expected_manifest_hash):
        checksum_errors.append("memory archive manifest has an invalid checksum")

    archive_manifest_path = root / MEMORY_ARCHIVE_MANIFEST_PATH
    if not archive_manifest_path.is_file() or archive_manifest_path.is_symlink():
        missing_files.append(MEMORY_ARCHIVE_MANIFEST_PATH)
        return {
            "errors": errors,
            "missing_files": missing_files,
            "checksum_errors": checksum_errors,
        }
    manifest_payload = archive_manifest_path.read_bytes()
    if (
        _valid_sha256(expected_manifest_hash)
        and _sha256_bytes(manifest_payload) != expected_manifest_hash
    ):
        checksum_errors.append("memory archive manifest checksum does not match")
    try:
        archive_manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"memory archive manifest is invalid JSON: {exc}")
        return {
            "errors": errors,
            "missing_files": missing_files,
            "checksum_errors": checksum_errors,
        }
    if not isinstance(archive_manifest, Mapping):
        errors.append("memory archive manifest must be a JSON object")
        return {
            "errors": errors,
            "missing_files": missing_files,
            "checksum_errors": checksum_errors,
        }

    expected_provenance = {
        "schema_version": MEMORY_ARCHIVE_SCHEMA_VERSION,
        "artifact_type": "final_experiment_memory",
        "artifact_role": "read_only_research_artifact",
        "runtime_consumed": False,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "memory_lineage_id": memory_lineage_id,
        "memory_lifecycle": memory_lifecycle,
        "memory_resumed_from_run_id": memory_resumed_from_run_id,
        "graph_config_sha256": graph_config_sha256,
    }
    for field, expected in expected_provenance.items():
        if archive_manifest.get(field) != expected:
            errors.append(f"memory archive {field} does not match the run bundle")
    runtime_namespace = archive_manifest.get("runtime_namespace")
    expected_runtime_namespace = {
        "storage_role": "operational_runtime_only",
        "layout_version": "historical-memory-v1",
        "experiment_id": experiment_id,
        "memory_lineage_id": memory_lineage_id,
        "graph_config_sha256": graph_config_sha256,
    }
    if runtime_namespace != expected_runtime_namespace:
        errors.append("memory archive runtime_namespace provenance does not match the run")

    records = archive_manifest.get("symbols")
    if not isinstance(records, list):
        errors.append("memory archive symbols must be a list")
        records = []
    observed_symbols: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("every memory archive symbol record must be an object")
            continue
        symbol = record.get("symbol")
        if not isinstance(symbol, str):
            errors.append("memory archive symbol must be a string")
            continue
        observed_symbols.append(symbol)
        canonical_relative = f"memory/symbols/{symbol}.md"
        if record.get("path") != canonical_relative:
            errors.append(f"{symbol}: memory archive path is not canonical")
        if record.get("format") != MEMORY_ARCHIVE_FORMAT:
            errors.append(f"{symbol}: memory archive format is unsupported")
        unresolved_member = root / str(record.get("path", ""))
        member = _bundle_member(root, record.get("path"))
        if (
            unresolved_member.is_symlink()
            or member is None
            or not member.is_file()
            or member.is_symlink()
        ):
            missing_files.append(canonical_relative)
            continue
        payload = member.read_bytes()
        if not payload:
            errors.append(f"{symbol}: archived Memory is empty")
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{symbol}: archived Memory is not valid UTF-8")
        if record.get("size_bytes") != len(payload):
            checksum_errors.append(f"{symbol}: archived Memory size does not match")
        expected_hash = record.get("sha256")
        if not _valid_sha256(expected_hash):
            checksum_errors.append(f"{symbol}: archived Memory checksum is invalid")
        elif _sha256_bytes(payload) != expected_hash:
            checksum_errors.append(f"{symbol}: archived Memory checksum does not match")

    if len(set(observed_symbols)) != len(observed_symbols):
        errors.append("memory archive contains duplicate symbol records")
    if set(observed_symbols) != set(expected_symbols):
        errors.append("memory archive symbol coverage does not match the run")
    return {
        "errors": errors,
        "missing_files": missing_files,
        "checksum_errors": checksum_errors,
    }
