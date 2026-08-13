"""Canonical runtime experiment-Memory namespace and path resolution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tradingagents.dataflows.utils import safe_ticker_component

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FINMULTITIME_SCOPES = frozenset({"PILOT", "FORMAL"})


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def experiment_memory_namespace_component(
    *,
    finmultitime_evidence_enabled: bool,
    finmultitime_bundle_scope: str | None = None,
    finmultitime_bundle_identity: str | None = None,
) -> str | None:
    """Return the optional namespace component for experiment Memory.

    FinMultiTime identity is deliberately validated here so enabled runs can
    never silently fall back to the unscoped M0 Memory namespace.
    """
    if not finmultitime_evidence_enabled:
        return None
    if not isinstance(finmultitime_bundle_scope, str):
        raise ValueError(
            "FinMultiTime experiment Memory requires bundle scope PILOT or FORMAL"
        )
    scope = finmultitime_bundle_scope.strip().upper()
    if scope not in _FINMULTITIME_SCOPES:
        raise ValueError(
            "FinMultiTime experiment Memory requires bundle scope PILOT or FORMAL"
        )
    if not _valid_sha256(finmultitime_bundle_identity):
        raise ValueError(
            "FinMultiTime experiment Memory requires a valid SHA-256 bundle identity"
        )
    namespace = f"finmultitime-{scope.lower()}-{finmultitime_bundle_identity[:16].lower()}"
    return safe_ticker_component(namespace, max_len=128)


def runtime_experiment_memory_path(
    runtime_memory_dir: str | Path,
    *,
    experiment_id: str,
    graph_config_sha256: str,
    memory_lineage_id: str,
    symbol: str,
    finmultitime_evidence_enabled: bool = False,
    finmultitime_bundle_scope: str | None = None,
    finmultitime_bundle_identity: str | None = None,
) -> Path:
    """Resolve the canonical runtime experiment-Memory file path."""
    experiment = safe_ticker_component(experiment_id, max_len=128)
    lineage = safe_ticker_component(memory_lineage_id, max_len=128)
    if not _valid_sha256(graph_config_sha256):
        raise ValueError("graph_config_sha256 must be a valid SHA-256")
    safe_symbol = safe_ticker_component(symbol)
    namespace = experiment_memory_namespace_component(
        finmultitime_evidence_enabled=finmultitime_evidence_enabled,
        finmultitime_bundle_scope=finmultitime_bundle_scope,
        finmultitime_bundle_identity=finmultitime_bundle_identity,
    )
    components = [Path(runtime_memory_dir), "historical_memory"]
    if namespace is not None:
        components.append(namespace)
    components.extend([experiment, graph_config_sha256, lineage, f"{safe_symbol}.md"])
    return Path(*components)
