"""Fail-closed lifecycle selection for historical experiment memory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradingagents.dataflows.utils import safe_ticker_component


def validate_resume_data_source(source: str, *, resume: bool) -> None:
    """Reject resume when identical historical inputs cannot be proven."""
    if resume and source == "yfinance":
        raise ValueError(
            "--resume is not supported with mutable yfinance input; "
            "resume from the archived snapshot instead"
        )


def snapshot_input_identity(
    snapshot_dir: str | Path, symbols: list[str],
) -> dict[str, str]:
    """Return exact per-symbol hashes for a frozen snapshot input bundle."""
    root = Path(snapshot_dir)
    identity: dict[str, str] = {}
    for symbol in dict.fromkeys([*symbols, "SPY"]):
        safe_symbol = safe_ticker_component(symbol)
        path = root / f"{safe_symbol}.csv"
        if not path.is_file():
            raise ValueError(f"snapshot is missing {symbol}: {path}")
        identity[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def backtest_protocol_sha256(
    effective_config: dict[str, Any], *, planned_cases: int,
    market_input_identity: dict[str, str] | None = None,
    implementation_identity: str | None = None,
) -> str:
    """Hash the stable research/execution contract relevant to a resume."""
    payload = json.dumps(
        {
            "config": effective_config,
            "planned_cases": planned_cases,
            "market_input_identity": market_input_identity,
            "implementation_identity": implementation_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryLineage:
    """Identity and provenance for one chronological historical-memory run."""

    lineage_id: str
    lifecycle: str
    resumed_from_run_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "memory_lineage_id": self.lineage_id,
            "memory_lifecycle": self.lifecycle,
            "memory_resumed_from_run_id": self.resumed_from_run_id,
        }


def _read_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume: invalid {description} at {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"cannot resume: {description} must be a JSON object")
    return value


def _resume_component(value: Any, *, field: str) -> str:
    """Validate persisted resume IDs without coercing JSON null/non-strings."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"cannot resume: {field} must be a non-empty string")
    try:
        return safe_ticker_component(value, max_len=128)
    except ValueError as exc:
        raise ValueError(f"cannot resume: invalid {field}") from exc


def select_memory_lineage(
    *,
    experiment_root: str | Path,
    experiment_id: str,
    run_id: str,
    graph_config_sha256: str,
    backtest_protocol_sha256: str,
    resume: bool,
    force: bool,
) -> MemoryLineage:
    """Choose a fresh lineage or explicitly continue the latest failed attempt.

    ``--resume`` is deliberately bound to ``latest.json`` rather than searching
    older runs. The referenced run must be incomplete and graph-compatible.
    ``latest.json.run_path`` is ignored; the candidate is reconstructed beneath
    the current experiment root from a validated run ID.
    """
    safe_ticker_component(experiment_id, max_len=128)
    current_run_id = safe_ticker_component(run_id, max_len=128)
    if resume and force:
        raise ValueError("--resume and --force are mutually exclusive")
    if not resume:
        return MemoryLineage(
            current_run_id,
            "force_fresh" if force else "independent_fresh",
        )

    root = Path(experiment_root)
    latest = _read_object(root / "latest.json", description="latest.json")
    if latest.get("experiment_id") != experiment_id:
        raise ValueError("cannot resume: latest experiment_id does not match")
    prior_run_id = _resume_component(
        latest.get("run_id"), field="latest run_id"
    )
    if prior_run_id == current_run_id:
        raise ValueError("cannot resume the current run as its own predecessor")
    prior_run = root / "runs" / prior_run_id
    status = _read_object(
        prior_run / "run_status.json", description="prior run_status.json"
    )
    if latest.get("status") != status.get("status"):
        raise ValueError("cannot resume: latest.json and run_status.json disagree")
    if status.get("status") not in {"failed", "running"}:
        raise ValueError("cannot resume a completed run; start an independent rerun")
    if (
        status.get("experiment_id") != experiment_id
        or status.get("run_id") != prior_run_id
    ):
        raise ValueError("cannot resume: prior run identity does not match")
    latest_lineage_id = _resume_component(
        latest.get("memory_lineage_id"), field="latest memory_lineage_id"
    )
    status_lineage_id = _resume_component(
        status.get("memory_lineage_id"), field="status memory_lineage_id"
    )
    if latest_lineage_id != status_lineage_id:
        raise ValueError("cannot resume: latest and status memory lineages disagree")

    resolved = _read_object(
        prior_run / "config.resolved.json",
        description="prior config.resolved.json",
    )
    if resolved.get("experiment_id") != experiment_id:
        raise ValueError("cannot resume: prior resolved experiment_id does not match")
    if resolved.get("run_id") != prior_run_id:
        raise ValueError("cannot resume: prior resolved run_id does not match")
    if resolved.get("graph_config_sha256") != graph_config_sha256:
        raise ValueError("cannot resume: prior graph configuration is incompatible")
    if resolved.get("backtest_protocol_sha256") != backtest_protocol_sha256:
        raise ValueError("cannot resume: prior backtest protocol is incompatible")

    lineage_id = _resume_component(
        resolved.get("memory_lineage_id"), field="resolved memory_lineage_id"
    )
    if lineage_id != status_lineage_id:
        raise ValueError("cannot resume: resolved and status memory lineages disagree")
    return MemoryLineage(lineage_id, "resume", prior_run_id)
