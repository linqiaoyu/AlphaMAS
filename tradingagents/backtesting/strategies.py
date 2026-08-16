"""Unified deterministic and TradingAgents weekly strategies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from tradingagents.backtesting.cache import DecisionCache, cache_key
from tradingagents.backtesting.config import graph_config_identity
from tradingagents.backtesting.models import Action, DecisionStatus, StrategyDecision
from tradingagents.backtesting.recorder import write_json
from tradingagents.runtime.run_context import RunContext


def target_for(action: Action, current_weight: float, quantity: float = 0.0) -> float:
    if action is Action.BUY:
        return 1.0
    if action is Action.SELL:
        return 0.0
    # HOLD preserves the discrete long-only position. Using a 0.5 weight
    # threshold would silently turn a large drawdown into a SELL.
    return 1.0 if quantity > 0.0 else 0.0


class Strategy(Protocol):
    strategy_id: str

    def decide(
        self, *, symbol: str, decision_session: str, decision_time: datetime,
        market_history: pd.DataFrame, portfolio_snapshot: Any, context: dict[str, Any],
    ) -> StrategyDecision: ...


class BaseStrategy:
    strategy_id = "base"

    def _decision(
        self, action: Action, *, symbol: str, decision_session: str,
        decision_time: datetime, portfolio_snapshot: Any, context: dict[str, Any],
        reason: str, market_history: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyDecision:
        return StrategyDecision(
            symbol=symbol, decision_session=decision_session,
            decision_time_utc=decision_time, action=action,
            target_weight=target_for(
                action, portfolio_snapshot.current_weight, portfolio_snapshot.quantity
            ),
            status=DecisionStatus.SUCCESS, reason=reason, strategy_id=self.strategy_id,
            experiment_id=context["experiment_id"], raw_signal=action.value,
            metadata=metadata or {},
        )


class ScriptedStrategy(BaseStrategy):
    strategy_id = "scripted"

    def __init__(self, actions: dict[str, str | Action] | None = None, default: str = "HOLD"):
        self.actions = actions or {}
        self.default = Action(default.upper())

    def decide(self, **kwargs: Any) -> StrategyDecision:
        configured = self.actions.get(kwargs["decision_session"], self.default)
        action = configured if isinstance(configured, Action) else Action(str(configured).upper())
        return self._decision(action, reason="scripted action", **kwargs)


class BuyAndHoldStrategy(BaseStrategy):
    strategy_id = "buy_and_hold"

    def decide(self, **kwargs: Any) -> StrategyDecision:
        has_position = kwargs["portfolio_snapshot"].quantity > 0
        action = Action.HOLD if has_position else Action.BUY
        return self._decision(action, reason="first weekly execution then hold", **kwargs)


class SMAStrategy(BaseStrategy):
    strategy_id = "sma"

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if not 0 < short_window < long_window:
            raise ValueError("SMA windows must satisfy 0 < short < long")
        self.short_window = short_window
        self.long_window = long_window

    def decide(self, **kwargs: Any) -> StrategyDecision:
        close = kwargs["market_history"]["Close"].astype(float)
        if len(close) < self.long_window:
            action, reason = Action.HOLD, "insufficient history"
        else:
            short = float(close.iloc[-self.short_window:].mean())
            long = float(close.iloc[-self.long_window:].mean())
            action = Action.BUY if short > long else Action.SELL
            reason = "short SMA above long SMA" if action is Action.BUY else "short SMA not above long SMA"
        return self._decision(action, reason=reason, **kwargs)


_EXPLICIT_ACTION = re.compile(
    r"(?:rating|action)\s*[:\-]\s*\**(buy|overweight|hold|underweight|sell)\b", re.I
)
_RATING_TO_ACTION = {
    "BUY": "BUY", "OVERWEIGHT": "BUY", "HOLD": "HOLD",
    "UNDERWEIGHT": "SELL", "SELL": "SELL",
}


def strict_action(raw: str, processed: str | None = None) -> Action:
    """Accept one unambiguous BUY/HOLD/SELL; never turn unknown text into HOLD."""
    candidates: set[str] = set()
    processed_rating = (processed or "").strip().upper()
    if processed_rating in _RATING_TO_ACTION:
        candidates.add(_RATING_TO_ACTION[processed_rating])
    candidates.update(
        _RATING_TO_ACTION[match.upper()] for match in _EXPLICIT_ACTION.findall(raw or "")
    )
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one explicit action, found {sorted(candidates)}")
    return Action(candidates.pop())


_DECISION_PREFIX_SEED = hashlib.sha256(
    b"TradingAgents chronological decision prefix v1"
).hexdigest()


@dataclass
class DecisionChronology:
    """Attempt-local chain binding every Agent cache key to its valid prefix."""

    prefix_sha256: str = field(default=_DECISION_PREFIX_SEED, init=False)
    failed_case: str | None = field(default=None, init=False)

    def require_active(self) -> None:
        if self.failed_case is not None:
            raise RuntimeError(
                "TradingAgents chronology already terminated at "
                f"{self.failed_case}; future decisions are forbidden"
            )

    def record_success(self, decision: StrategyDecision, *, cache_key_value: str) -> None:
        """Advance identically for a fresh success and its exact cache replay."""
        self.require_active()
        payload = {
            "prior_prefix_sha256": self.prefix_sha256,
            "cache_key": cache_key_value,
            "symbol": decision.symbol,
            "decision_session": decision.decision_session,
            "action": decision.action.value if decision.action else None,
            "target_weight": decision.target_weight,
            "raw_signal": decision.raw_signal,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        self.prefix_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def record_failure(self, symbol: str, session: str) -> None:
        self.failed_case = f"{symbol}:{session}"


@dataclass(frozen=True)
class _MemoryFileSnapshot:
    path: Path
    existed: bool
    content: bytes


class TradingAgentsStrategy(BaseStrategy):
    strategy_id = "tradingagents"
    fail_fast_on_decision_failure = True

    def __init__(
        self, graph: Any, *, reports_root: str | Path | None = None,
        cache: DecisionCache | None = None, cache_config: dict[str, Any] | None = None,
        usage_callback: Any | None = None, run_root: str | Path | None = None,
        force: bool = False, chronology: DecisionChronology | None = None,
    ) -> None:
        self.graph = graph
        self.reports_root = Path(reports_root) if reports_root else None
        self.cache = cache
        self.cache_config = cache_config or {}
        self.usage_callback = usage_callback
        self.run_root = Path(run_root) if run_root else None
        self.force = force
        self.chronology = chronology or DecisionChronology()

    def _cache_payload(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        snapshot = kwargs["portfolio_snapshot"]
        state = snapshot.to_dict() if hasattr(snapshot, "to_dict") else {
            "cash": snapshot.cash, "quantity": snapshot.quantity,
            "equity": snapshot.equity, "average_entry_price": snapshot.average_entry_price,
        }
        graph_config = getattr(self.graph, "config", {})
        finmultitime_store = getattr(self.graph, "finmultitime_evidence_store", None)
        finmultitime_enabled = finmultitime_store is not None
        m2_runtime = getattr(self.graph, "m2_trader_runtime", None)
        finmultitime_case = None
        if finmultitime_store is not None:
            decision_session = kwargs.get("decision_session")
            if decision_session is None:
                raise ValueError(
                    "FinMultiTime cache identity requires an exact decision_session"
                )
            finmultitime_case = finmultitime_store.case_identity(
                kwargs["symbol"], str(decision_session)
            )
        history_payload = kwargs["market_history"].to_csv(
            lineterminator="\n", float_format="%.12g"
        )
        return {
            "experiment_id": kwargs["context"]["experiment_id"],
            "symbol": kwargs["symbol"],
            "decision_time_utc": kwargs["decision_time"].isoformat(),
            "strategy_id": self.strategy_id,
            "selected_analysts": list(getattr(self.graph, "selected_analysts", ())),
            "research_depth": graph_config.get("research_depth"),
            "debate_rounds": graph_config.get("max_debate_rounds"),
            "risk_rounds": graph_config.get("max_risk_discuss_rounds"),
            "quick_model": graph_config.get("quick_think_llm"),
            "deep_model": graph_config.get("deep_think_llm"),
            "provider": graph_config.get("llm_provider"),
            "thinking_mode": graph_config.get("deepseek_thinking"),
            "temperature": graph_config.get("temperature"),
            "output_language": graph_config.get("output_language"),
            "data_vendor_config": graph_config.get("data_vendors"),
            "tool_vendor_config": graph_config.get("tool_vendors"),
            "point_in_time": bool(kwargs["context"].get("point_in_time")),
            "memory_mode": graph_config.get("memory_mode"),
            "memory_lineage_id": graph_config.get(
                "historical_memory_lineage_id"
            ),
            "holding_horizon_sessions": graph_config.get(
                "memory_holding_horizon_sessions"
            ),
            "graph_config_sha256": graph_config.get("graph_config_sha256"),
            "git_commit_sha": self.cache_config.get("git_commit_sha"),
            "prompt_config_version": self.cache_config.get("prompt_config_version", "v1"),
            "portfolio_state_hash": hashlib.sha256(
                json.dumps(state, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "market_history_sha256": hashlib.sha256(history_payload.encode()).hexdigest(),
            "memory_namespace_version": self.cache_config.get("memory_namespace_version", "v1"),
            "finmultitime_evidence_enabled": finmultitime_enabled,
            "finmultitime_bundle_scope": (
                finmultitime_case["bundle_scope"] if finmultitime_case else None
            ),
            "finmultitime_input_bundle_identity": (
                finmultitime_case["input_bundle_identity"]
                if finmultitime_case else None
            ),
            "finmultitime_contract_sha256": (
                finmultitime_case["contract_sha256"] if finmultitime_case else None
            ),
            "finmultitime_case_id": (
                finmultitime_case["case_id"] if finmultitime_case else None
            ),
            "finmultitime_packet_json_sha256": (
                finmultitime_case["packet_json_sha256"] if finmultitime_case else None
            ),
            "finmultitime_route_sha256": (
                finmultitime_case["route_sha256"] if finmultitime_case else None
            ),
            # This rolling chain covers all earlier successful Agent cases in
            # the attempt, including prior symbols. A repaired failed point
            # therefore invalidates every obsolete downstream cache entry.
            "decision_prefix_sha256": self.chronology.prefix_sha256,
            "m2_trader_enabled": bool(graph_config.get("m2_trader_enabled", False)),
            "m2_variant": graph_config.get("m2_variant"),
            "m2_runtime_state_identity": (
                m2_runtime.cache_identity_for_decision(
                    kwargs["symbol"], kwargs["decision_session"]
                )
                if m2_runtime is not None else None
            ),
        }

    def _capture_memory(
        self, symbol: str, run_context: RunContext,
    ) -> _MemoryFileSnapshot | None:
        """Snapshot the exact runtime Memory file before an uncached Agent case."""
        resolver = getattr(self.graph, "_historical_memory_config", None)
        if not callable(resolver):
            return None
        memory_config = resolver(symbol, run_context)
        path_value = memory_config.get("memory_log_path")
        if not path_value:
            return None
        path = Path(path_value)
        if path.is_file():
            return _MemoryFileSnapshot(path, True, path.read_bytes())
        if path.exists():
            raise ValueError(f"historical Memory path is not a file: {path}")
        return _MemoryFileSnapshot(path, False, b"")

    @staticmethod
    def _restore_memory(snapshot: _MemoryFileSnapshot | None) -> str:
        """Atomically roll an unsuccessful case back to its Memory prefix."""
        if snapshot is None:
            return "not_applicable"
        path = snapshot.path
        if not snapshot.existed:
            if path.exists():
                if not path.is_file():
                    raise ValueError(f"historical Memory path is not a file: {path}")
                path.unlink()
            return "restored"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.rollback.", dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(snapshot.content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return "restored"

    def record_chronological_success(self, decision: StrategyDecision) -> None:
        """Commit a decision to the attempt chain after the engine accepts it."""
        key = decision.metadata.get("cache_key")
        if not isinstance(key, str) or not key:
            raise ValueError("TradingAgents decision is missing its cache key")
        self.chronology.record_success(decision, cache_key_value=key)

    def _case_dir(self, symbol: str, session: str) -> Path | None:
        if self.reports_root is None:
            return None
        path = self.reports_root / symbol / session
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        if self.run_root is not None:
            try:
                return path.relative_to(self.run_root).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    def _model_config(self) -> dict[str, Any]:
        config = getattr(self.graph, "config", {})
        return {
            **graph_config_identity(config),
            "graph_config_sha256": config.get("graph_config_sha256"),
        }

    def _write_case(
        self, case_dir: Path | None, *, decision: StrategyDecision,
        cache_payload: dict[str, Any], key: str, source_audit: dict[str, Any],
        usage: list[dict[str, Any]], cache_status: str, wall_clock_seconds: float,
        run_context: RunContext,
    ) -> None:
        if case_dir is None:
            return
        write_json(case_dir / "decision.json", decision.to_dict())
        write_json(case_dir / "model_config.json", self._model_config())
        write_json(case_dir / "run_context.json", {
            "mode": run_context.mode,
            "as_of": run_context.as_of.isoformat(),
            "generated_at": run_context.generated_at.isoformat(),
            "experiment_id": run_context.experiment_id,
            "memory_lineage_id": run_context.memory_lineage_id,
        })
        write_json(case_dir / "cache_identity.json", {
            "cache_key": key, "identity": cache_payload,
        })
        write_json(case_dir / "source_audit.json", source_audit)
        write_json(case_dir / "llm_usage.json", usage)
        write_json(case_dir / "case_metadata.json", {
            "case_id": f"{decision.symbol}:{decision.decision_session}",
            "cache_key": key,
            "cache_status": cache_status,
            "wall_clock_seconds": wall_clock_seconds,
            "report_path": decision.metadata.get("report_path"),
            "source_audit_path": decision.metadata.get("source_audit_path"),
        })

    @staticmethod
    def _load_source_audit(graph: Any, symbol: str, context: RunContext) -> dict[str, Any]:
        if not hasattr(graph, "_audit_path"):
            return {}
        audit_path = Path(graph._audit_path(symbol, context))
        if not audit_path.is_file():
            return {}
        try:
            return json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def decide(self, **kwargs: Any) -> StrategyDecision:
        self.chronology.require_active()
        started = time.monotonic()
        symbol = kwargs["symbol"]
        session = kwargs["decision_session"]
        decision_time = kwargs["decision_time"]
        context = kwargs["context"]
        visibility = context.get("market_history_visibility")
        visibility_metadata = (
            {"market_history_visibility": dict(visibility)}
            if isinstance(visibility, Mapping) else {}
        )
        cache_payload = self._cache_payload(kwargs)
        key = cache_key(cache_payload)
        case_dir = self._case_dir(symbol, session)
        run_context = RunContext.historical(
            decision_time, generated_at=datetime.now(timezone.utc),
            experiment_id=context["experiment_id"],
            memory_lineage_id=getattr(self.graph, "config", {}).get(
                "historical_memory_lineage_id"
            ),
        )
        m2_runtime = getattr(self.graph, "m2_trader_runtime", None)
        m2_cache_ready = (
            m2_runtime is None
            or m2_runtime.decision_is_persisted(symbol, session)
        )
        if self.cache and not self.force and m2_cache_ready:
            bundle = self.cache.load_bundle(
                key,
                require_artifacts=case_dir is not None,
                require_usage=case_dir is not None and self.usage_callback is not None,
            )
            if bundle:
                cached = bundle["decision"]
                cached_action = Action(cached["action"])
                if case_dir is not None:
                    reports = Path(bundle["path"]) / "reports"
                    if reports.is_dir():
                        shutil.copytree(reports, case_dir / "reports", dirs_exist_ok=True)
                    if bundle.get("llm_usage"):
                        write_json(
                            case_dir / "cached_origin_llm_usage.json", bundle["llm_usage"]
                        )
                elapsed = time.monotonic() - started
                metadata = {
                    **cached.get("metadata", {}),
                    **visibility_metadata,
                    "cache_key": key,
                    "cache_status": "hit",
                    "wall_clock_seconds": elapsed,
                    "report_path": self._artifact_path(
                        case_dir / "reports" if case_dir else None
                    ),
                    "source_audit_path": self._artifact_path(
                        case_dir / "source_audit.json" if case_dir else None
                    ),
                    "model_config": self._model_config(),
                    "decision_prefix_sha256": cache_payload[
                        "decision_prefix_sha256"
                    ],
                }
                decision = StrategyDecision(
                    symbol=symbol, decision_session=session,
                    decision_time_utc=decision_time, action=cached_action,
                    target_weight=cached["target_weight"], status=DecisionStatus.CACHED,
                    reason="exact cache hit", strategy_id=self.strategy_id,
                    experiment_id=context["experiment_id"],
                    raw_signal=cached.get("raw_signal", cached_action.value),
                    metadata=metadata,
                )
                self._write_case(
                    case_dir, decision=decision, cache_payload=cache_payload, key=key,
                    source_audit=bundle.get("source_audit", {}), usage=[],
                    cache_status="hit", wall_clock_seconds=elapsed,
                    run_context=run_context,
                )
                return decision
        if decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        case_id = f"{symbol}:{session}"
        if self.usage_callback is not None:
            self.usage_callback.start_case(
                context["experiment_id"], case_id, symbol, session
            )
        usage: list[dict[str, Any]] = []
        source_audit: dict[str, Any] = {}
        cache_status = "bypass" if self.force else "miss"
        memory_snapshot: _MemoryFileSnapshot | None = None
        try:
            memory_snapshot = self._capture_memory(symbol, run_context)
            final_state, processed = self.graph.propagate(
                symbol,
                session,
                mode="historical",
                run_context=run_context,
                portfolio_snapshot=kwargs["portfolio_snapshot"].to_dict(),
                portfolio_reward_state=context.get("portfolio_reward_state"),
                next_decision_session=context.get("next_decision_session"),
            )
            if case_dir is not None:
                report_dir = case_dir / "reports"
                self.graph.save_reports(final_state, symbol, report_dir)
            else:
                report_dir = None
            raw = str(final_state.get("final_trade_decision", ""))
            action = strict_action(raw, str(processed))
            source_audit = self._load_source_audit(self.graph, symbol, run_context)
            elapsed = time.monotonic() - started
            metadata = {
                "wall_clock_seconds": elapsed,
                "run_context": {"mode": "historical", "as_of": decision_time.isoformat()},
                **visibility_metadata,
                "report_path": self._artifact_path(report_dir),
                "source_audit_path": self._artifact_path(
                    case_dir / "source_audit.json" if case_dir else None
                ),
                "cache_key": key,
                "cache_status": cache_status,
                "model_config": self._model_config(),
                "decision_prefix_sha256": cache_payload[
                    "decision_prefix_sha256"
                ],
            }
            m2_runtime = getattr(self.graph, "m2_trader_runtime", None)
            if m2_runtime is not None:
                maturity_events = m2_runtime.mature_visible(
                    symbol,
                    visible_market_history=kwargs["market_history"],
                    cutoff_session=session,
                    allow_update_for_later_decision=bool(
                        context.get("next_decision_session")
                    ),
                )
                metadata["m2_trader_handoff"] = final_state.get(
                    "m2_trader_handoff_metadata", {}
                )
                metadata["m2_prompt_trader_proposal_original"] = final_state.get(
                    "prompt_trader_proposal_original", ""
                )
                metadata["m2_maturity_events_after_action"] = maturity_events
            decision = self._decision(
                action, reason="TradingAgents structured decision", metadata=metadata, **kwargs
            )
            if self.usage_callback is not None:
                usage = self.usage_callback.finish_case()
            self._write_case(
                case_dir, decision=decision, cache_payload=cache_payload, key=key,
                source_audit=source_audit, usage=usage, cache_status=cache_status,
                wall_clock_seconds=elapsed, run_context=run_context,
            )
            if self.cache:
                self.cache.save_success(
                    key, decision.to_dict(), {
                        "wall_clock_seconds": metadata["wall_clock_seconds"],
                        "source_audit": source_audit,
                        "report_path": str(report_dir) if report_dir else None,
                        "llm_usage": usage,
                    }
                )
            return decision
        except Exception as exc:
            self.chronology.record_failure(symbol, session)
            rollback_status = self._restore_memory(memory_snapshot)
            if self.usage_callback is not None:
                usage = self.usage_callback.finish_case()
            source_audit = self._load_source_audit(self.graph, symbol, run_context)
            elapsed = time.monotonic() - started
            metadata = {
                "wall_clock_seconds": elapsed,
                **visibility_metadata,
                "cache_key": key,
                "cache_status": cache_status,
                "decision_prefix_sha256": cache_payload[
                    "decision_prefix_sha256"
                ],
                "memory_rollback_status": rollback_status,
                "report_path": self._artifact_path(
                    case_dir / "reports" if case_dir and (case_dir / "reports").is_dir()
                    else None
                ),
                "source_audit_path": self._artifact_path(
                    case_dir / "source_audit.json" if case_dir else None
                ),
                "model_config": self._model_config(),
            }
            decision = StrategyDecision(
                symbol=symbol, decision_session=session, decision_time_utc=decision_time,
                action=None, target_weight=None, status=DecisionStatus.FAILED,
                reason=f"{type(exc).__name__}: {exc}", strategy_id=self.strategy_id,
                experiment_id=context["experiment_id"], raw_signal="",
                metadata=metadata,
            )
            self._write_case(
                case_dir, decision=decision, cache_payload=cache_payload, key=key,
                source_audit=source_audit, usage=usage, cache_status=cache_status,
                wall_clock_seconds=elapsed, run_context=run_context,
            )
            return decision

    def finalize_symbol(
        self,
        *,
        symbol: str,
        final_session: str,
        market_history: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Score/archive terminal M2 credits without a post-horizon update."""

        runtime = getattr(self.graph, "m2_trader_runtime", None)
        if runtime is None:
            return []
        return runtime.mature_visible(
            symbol,
            visible_market_history=market_history,
            cutoff_session=final_session,
            allow_update_for_later_decision=False,
        )
