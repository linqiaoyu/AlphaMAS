"""Explicit run context and a small machine-readable source audit trail.

The context is carried in a :class:`contextvars.ContextVar` so LangGraph's
existing tool schemas do not need an extra hidden argument.  Graph runs set it
at their boundary and always reset it afterwards; callers of an individual
dataflow can use ``activate_run_context`` in tests or integrations.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Literal

UTC = timezone.utc
RunMode = Literal["live", "historical"]


def _as_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime, rejecting neither old nor new callers."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def coerce_as_of(value: date | datetime | str) -> datetime:
    """Parse an as-of value; a date-only value means the end of that UTC day."""
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.max, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        raise ValueError("historical as_of must not be empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"historical as_of must be an ISO date or datetime, got {value!r}"
        ) from exc
    if "T" not in text and " " not in text:
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return _as_utc(parsed)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class RunContext:
    """Immutable temporal identity of one analysis run.

    ``generated_at`` is when this run was actually created.  ``as_of`` is the
    latest time the evidence is allowed to describe.  They intentionally remain
    separate: a historical report is expected to be generated after its
    historical cutoff.
    """

    mode: RunMode
    as_of: datetime
    generated_at: datetime
    experiment_id: str | None = None
    memory_lineage_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("live", "historical"):
            raise ValueError(f"unsupported run mode: {self.mode!r}")
        if self.mode == "historical" and self.as_of is None:
            raise ValueError("historical runs require an explicit as_of")
        as_of = self.as_of
        if not isinstance(as_of, datetime):
            as_of = coerce_as_of(as_of)  # type: ignore[arg-type]
        object.__setattr__(self, "as_of", _as_utc(as_of))
        object.__setattr__(self, "generated_at", _as_utc(self.generated_at))

    @property
    def as_of_date(self) -> date:
        return self.as_of.date()

    @property
    def historical_as_of(self) -> str | None:
        return self.as_of.isoformat() if self.mode == "historical" else None

    @classmethod
    def live(cls, generated_at: datetime | None = None) -> RunContext:
        generated = _as_utc(generated_at or datetime.now(UTC))
        return cls("live", generated, generated)

    @classmethod
    def historical(
        cls,
        as_of: date | datetime | str,
        generated_at: datetime | None = None,
        experiment_id: str | None = None,
        memory_lineage_id: str | None = None,
    ) -> RunContext:
        generated = _as_utc(generated_at or datetime.now(UTC))
        return cls(
            "historical", coerce_as_of(as_of), generated,
            experiment_id, memory_lineage_id,
        )


def create_run_context(
    mode: RunMode = "live",
    *,
    as_of: date | datetime | str | None = None,
    generated_at: datetime | None = None,
    experiment_id: str | None = None,
    memory_lineage_id: str | None = None,
) -> RunContext:
    """Construct a validated context, requiring ``as_of`` in historical mode."""
    if mode == "historical":
        if as_of is None:
            raise ValueError("historical mode requires an explicit as_of")
        return RunContext.historical(
            as_of, generated_at, experiment_id, memory_lineage_id,
        )
    if as_of is not None:
        raise ValueError("live mode does not accept a historical as_of")
    if experiment_id is not None:
        generated = _as_utc(generated_at or datetime.now(UTC))
        return RunContext("live", generated, generated, experiment_id)
    return RunContext.live(generated_at)


@dataclass
class AuditTrail:
    """Small append-only in-memory audit collector for one run."""

    context: RunContext
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        source_name: str,
        capability: str,
        status: str,
        requested_start: Any = None,
        requested_end: Any = None,
        latest_event_time: Any = None,
        latest_available_time: Any = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "source_name": source_name,
            "mode": self.context.mode,
            "requested_start": _iso(requested_start),
            "requested_end": _iso(requested_end),
            "historical_as_of": self.context.historical_as_of,
            "capability": capability,
            "status": status,
            "latest_event_time": _iso(latest_event_time),
            "latest_available_time": _iso(latest_available_time),
            "reason": reason,
            "generated_at": self.context.generated_at.isoformat(),
        }
        if metadata is not None:
            record["metadata"] = metadata
        self.records.append(record)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_context": {
                "mode": self.context.mode,
                "as_of": self.context.as_of.isoformat(),
                "historical_as_of": self.context.historical_as_of,
                "generated_at": self.context.generated_at.isoformat(),
                "experiment_id": self.context.experiment_id,
                "memory_lineage_id": self.context.memory_lineage_id,
            },
            "sources": list(self.records),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return target


_context_var: ContextVar[RunContext | None] = ContextVar("tradingagents_run_context", default=None)
_audit_var: ContextVar[AuditTrail | None] = ContextVar("tradingagents_audit_trail", default=None)


def current_run_context() -> RunContext:
    """Return the active context, preserving the project's live default."""
    context = _context_var.get()
    return context if context is not None else RunContext.live()


def current_audit() -> AuditTrail | None:
    return _audit_var.get()


@contextmanager
def activate_run_context(
    context: RunContext,
    audit: AuditTrail | None = None,
) -> Iterator[AuditTrail | None]:
    """Activate and reliably clean up one run's context and audit collector."""
    context_token = _context_var.set(context)
    audit_obj = audit or AuditTrail(context)
    audit_token = _audit_var.set(audit_obj)
    try:
        yield audit_obj
    finally:
        _audit_var.reset(audit_token)
        _context_var.reset(context_token)


def audit_source(**kwargs: Any) -> None:
    """Record a source event when an audit collector is active."""
    audit = _audit_var.get()
    if audit is not None:
        audit.record(**kwargs)
