"""Provider-reported, request-level LLM usage collection for backtest artifacts."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any
from uuid import UUID

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.backtesting.artifacts import build_llm_usage

_TOKEN_COLUMNS = (
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _token_count(value: Any) -> int | None:
    """Accept a provider's integer count without estimating missing values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for part in path:
        current_mapping = _mapping(current)
        if part not in current_mapping:
            return None
        current = current_mapping[part]
    return current


def _normalise_usage(value: Any) -> dict[str, int | None]:
    usage = _mapping(value)
    return {
        "prompt_tokens": _token_count(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        ),
        "prompt_cache_hit_tokens": _token_count(
            usage.get(
                "prompt_cache_hit_tokens",
                _nested(usage, "prompt_tokens_details", "cached_tokens")
                if "prompt_tokens_details" in usage
                else _nested(usage, "input_token_details", "cache_read"),
            )
        ),
        "prompt_cache_miss_tokens": _token_count(
            usage.get("prompt_cache_miss_tokens")
        ),
        "completion_tokens": _token_count(
            usage.get("completion_tokens", usage.get("output_tokens"))
        ),
        "reasoning_tokens": _token_count(
            usage.get(
                "reasoning_tokens",
                _nested(usage, "completion_tokens_details", "reasoning_tokens")
                if "completion_tokens_details" in usage
                else _nested(usage, "output_token_details", "reasoning"),
            )
        ),
        "total_tokens": _token_count(usage.get("total_tokens")),
    }


def _message_candidates(response: Any) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    for generations in getattr(response, "generations", ()) or ():
        for generation in generations or ():
            message = getattr(generation, "message", None)
            if message is None:
                continue
            response_metadata = _mapping(getattr(message, "response_metadata", None))
            for key in ("token_usage", "usage"):
                candidate = _mapping(response_metadata.get(key))
                if candidate:
                    candidates.append(candidate)
            usage_metadata = _mapping(getattr(message, "usage_metadata", None))
            if usage_metadata:
                candidates.append(usage_metadata)
    return candidates


def extract_provider_usage(response: Any) -> dict[str, int | None]:
    """Extract only counts returned by LangChain/provider response metadata."""
    candidates: list[Mapping[str, Any]] = []
    llm_output = _mapping(getattr(response, "llm_output", None))
    for key in ("token_usage", "usage"):
        candidate = _mapping(llm_output.get(key))
        if candidate:
            candidates.append(candidate)
    candidates.extend(_message_candidates(response))

    merged: dict[str, int | None] = dict.fromkeys(_TOKEN_COLUMNS)
    for candidate in candidates:
        normalized = _normalise_usage(candidate)
        for key, value in normalized.items():
            if merged[key] is None and value is not None:
                merged[key] = value
    return merged


def _first_value(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)


def _records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = build_llm_usage(rows).astype(object)
    return frame.where(pd.notna(frame), None).to_dict("records")


class LLMUsageCallback(BaseCallbackHandler):
    """Thread-safe callback collecting one fixed-schema row per real request.

    The callback never tokenizes prompts and never derives missing cache-token
    counts. A provider-omitted field remains null in the emitted artifact.
    """

    def __init__(
        self, *, run_id: str | None = None, provider: str | None = None,
        thinking_mode: str | None = None, model: str | None = None,
        agent_node: str | None = None,
    ) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._defaults: dict[str, Any] = {
            "experiment_id": None,
            "run_id": run_id,
            "case_id": None,
            "symbol": None,
            "decision_session": None,
            "usage_source": "live_request",
            "origin_run_id": run_id,
            "agent_node": agent_node,
            "provider": provider,
            "model": model,
            "thinking_mode": thinking_mode,
        }
        self._starts: dict[UUID | str, dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] = []
        self._active_case_start: int | None = None
        self._last_case_rows: list[dict[str, Any]] = []

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Return all completed rows with the public fixed schema."""
        with self._lock:
            return _records(self._rows)

    def start_case(
        self, experiment_id: str, case_id: str, symbol: str, decision_session: str,
        *, run_id: str | None = None,
    ) -> None:
        """Bind subsequent request callbacks to one backtest case."""
        with self._lock:
            if self._active_case_start is not None:
                raise RuntimeError("finish_case() must be called before starting another case")
            self._defaults.update({
                "experiment_id": experiment_id,
                "case_id": case_id,
                "symbol": symbol,
                "decision_session": decision_session,
            })
            if run_id is not None:
                self._defaults["run_id"] = run_id
                self._defaults["origin_run_id"] = run_id
            self._active_case_start = len(self._rows)
            self._last_case_rows = []

    def finish_case(self) -> list[dict[str, Any]]:
        """Return rows completed since ``start_case`` and clear case identity."""
        with self._lock:
            if self._active_case_start is None:
                return [dict(row) for row in self._last_case_rows]
            start = self._active_case_start
            rows = _records(self._rows[start:])
            self._last_case_rows = rows
            self._active_case_start = None
            self._defaults.update({
                "experiment_id": None,
                "case_id": None,
                "symbol": None,
                "decision_session": None,
            })
            return [dict(row) for row in rows]

    def set_context(self, **values: Any) -> None:
        """Update stable callback context such as run/provider/thinking mode."""
        unknown = set(values) - set(self._defaults)
        if unknown:
            raise ValueError(f"unsupported LLM usage context fields: {sorted(unknown)}")
        with self._lock:
            self._defaults.update(values)

    @contextmanager
    def case(
        self, experiment_id: str, case_id: str, symbol: str, decision_session: str,
        *, run_id: str | None = None,
    ):
        """Context-manager alias for callers that can scope a case lexically."""
        self.start_case(
            experiment_id, case_id, symbol, decision_session, run_id=run_id,
        )
        try:
            yield self
        finally:
            self.finish_case()

    def _start_request(
        self, serialized: Mapping[str, Any], run_id: UUID | str,
        *, tags: list[str] | None, metadata: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> None:
        metadata = _mapping(metadata)
        invocation = _mapping(kwargs.get("invocation_params"))
        serialized_kwargs = _mapping(serialized.get("kwargs"))
        with self._lock:
            context = dict(self._defaults)
            context["provider"] = _first_value(
                context["provider"], metadata.get("provider"), metadata.get("ls_provider"),
            )
            context["model"] = _first_value(
                metadata.get("ls_model_name"),
                metadata.get("model"),
                invocation.get("model"),
                invocation.get("model_name"),
                serialized_kwargs.get("model"),
                serialized_kwargs.get("model_name"),
                context["model"],
            )
            context["agent_node"] = _first_value(
                metadata.get("langgraph_node"),
                metadata.get("agent_node"),
                metadata.get("agent"),
                context["agent_node"],
            )
            # Some LangChain versions invoke both chat-model and LLM start hooks.
            # Preserve the first timestamp/context for a given provider request.
            self._starts.setdefault(run_id, {**context, "started": time.monotonic()})

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], *,
        run_id: UUID, parent_run_id: UUID | None = None,
        tags: list[str] | None = None, metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del messages, parent_run_id
        self._start_request(
            serialized, run_id, tags=tags, metadata=metadata, kwargs=kwargs,
        )

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *,
        run_id: UUID, parent_run_id: UUID | None = None,
        tags: list[str] | None = None, metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del prompts, parent_run_id
        self._start_request(
            serialized, run_id, tags=tags, metadata=metadata, kwargs=kwargs,
        )

    def _finish_request(self, run_id: UUID | str, response: Any = None) -> None:
        with self._lock:
            started = self._starts.pop(run_id, None)
            if started is None:
                started = {**self._defaults, "started": None}
            llm_output = _mapping(getattr(response, "llm_output", None))
            started["model"] = _first_value(
                started.get("model"), llm_output.get("model_name"), llm_output.get("model"),
            )
            start_time = started.pop("started")
            latency = time.monotonic() - start_time if start_time is not None else None
            usage = extract_provider_usage(response) if response is not None else dict.fromkeys(
                _TOKEN_COLUMNS
            )
            self._rows.append({**started, **usage, "latency_seconds": latency})

    def on_llm_end(
        self, response: Any, *, run_id: UUID,
        parent_run_id: UUID | None = None, tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, tags, kwargs
        self._finish_request(run_id, response)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID,
        parent_run_id: UUID | None = None, tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del error, parent_run_id, tags, kwargs
        self._finish_request(run_id)
