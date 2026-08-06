"""Unified deterministic and TradingAgents weekly strategies."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from tradingagents.backtesting.cache import DecisionCache, cache_key
from tradingagents.backtesting.models import Action, DecisionStatus, StrategyDecision
from tradingagents.runtime.run_context import RunContext


def target_for(action: Action, current_weight: float) -> float:
    if action is Action.BUY:
        return 1.0
    if action is Action.SELL:
        return 0.0
    return 1.0 if current_weight > 0.5 else 0.0


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
            target_weight=target_for(action, portfolio_snapshot.current_weight),
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


class TradingAgentsStrategy(BaseStrategy):
    strategy_id = "tradingagents"

    def __init__(
        self, graph: Any, *, reports_root: str | Path | None = None,
        cache: DecisionCache | None = None, cache_config: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        self.graph = graph
        self.reports_root = Path(reports_root) if reports_root else None
        self.cache = cache
        self.cache_config = cache_config or {}
        self.force = force

    def _cache_payload(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        snapshot = kwargs["portfolio_snapshot"]
        state = {
            "cash": snapshot.cash, "quantity": snapshot.quantity,
            "equity": snapshot.equity, "average_entry_price": snapshot.average_entry_price,
        }
        graph_config = getattr(self.graph, "config", {})
        return {
            "experiment_id": kwargs["context"]["experiment_id"],
            "symbol": kwargs["symbol"],
            "decision_time_utc": kwargs["decision_time"].isoformat(),
            "strategy_id": self.strategy_id,
            "selected_analysts": list(getattr(self.graph, "selected_analysts", ())),
            "research_depth": graph_config.get("max_debate_rounds"),
            "quick_model": graph_config.get("quick_think_llm"),
            "deep_model": graph_config.get("deep_think_llm"),
            "provider": graph_config.get("llm_provider"),
            "temperature": graph_config.get("temperature"),
            "data_vendor_config": graph_config.get("data_vendors"),
            "point_in_time": True,
            "git_commit_sha": self.cache_config.get("git_commit_sha"),
            "prompt_config_version": self.cache_config.get("prompt_config_version", "v1"),
            "portfolio_state_hash": hashlib.sha256(
                json.dumps(state, sort_keys=True).encode()
            ).hexdigest(),
            "memory_namespace_version": self.cache_config.get("memory_namespace_version", "v1"),
        }

    def decide(self, **kwargs: Any) -> StrategyDecision:
        started = time.monotonic()
        symbol = kwargs["symbol"]
        session = kwargs["decision_session"]
        decision_time = kwargs["decision_time"]
        context = kwargs["context"]
        key = cache_key(self._cache_payload(kwargs))
        if self.cache and not self.force:
            cached = self.cache.load(key)
            if cached:
                cached_action = Action(cached["action"])
                return StrategyDecision(
                    symbol=symbol, decision_session=session,
                    decision_time_utc=decision_time, action=cached_action,
                    target_weight=cached["target_weight"], status=DecisionStatus.CACHED,
                    reason="exact cache hit", strategy_id=self.strategy_id,
                    experiment_id=context["experiment_id"],
                    raw_signal=cached.get("raw_signal", cached_action.value),
                    metadata={**cached.get("metadata", {}), "cache_key": key},
                )
        if decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        run_context = RunContext.historical(
            decision_time, generated_at=datetime.now(timezone.utc),
            experiment_id=context["experiment_id"],
        )
        try:
            final_state, processed = self.graph.propagate(
                symbol, session, mode="historical", run_context=run_context,
            )
            raw = str(final_state.get("final_trade_decision", ""))
            action = strict_action(raw, str(processed))
            if self.reports_root:
                report_dir = self.reports_root / symbol / session / "reports"
                self.graph.save_reports(final_state, symbol, report_dir)
            metadata = {
                "wall_clock_seconds": time.monotonic() - started,
                "run_context": {"mode": "historical", "as_of": decision_time.isoformat()},
                "report_path": str(report_dir) if self.reports_root else None,
                "model_config": {
                    key: getattr(self.graph, "config", {}).get(key)
                    for key in (
                        "llm_provider", "quick_think_llm", "deep_think_llm",
                        "temperature", "max_debate_rounds", "max_risk_discuss_rounds",
                    )
                },
            }
            decision = self._decision(
                action, reason="TradingAgents structured decision", metadata=metadata, **kwargs
            )
            if self.cache:
                source_audit = {}
                audit_path = None
                if hasattr(self.graph, "_audit_path"):
                    audit_path = self.graph._audit_path(symbol, run_context)
                if audit_path and Path(audit_path).is_file():
                    source_audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
                self.cache.save_success(
                    key, decision.to_dict(), {
                        "wall_clock_seconds": metadata["wall_clock_seconds"],
                        "source_audit": source_audit,
                        "report_path": metadata["report_path"],
                    }
                )
            return decision
        except Exception as exc:
            return StrategyDecision(
                symbol=symbol, decision_session=session, decision_time_utc=decision_time,
                action=None, target_weight=None, status=DecisionStatus.FAILED,
                reason=f"{type(exc).__name__}: {exc}", strategy_id=self.strategy_id,
                experiment_id=context["experiment_id"], raw_signal="",
                metadata={"wall_clock_seconds": time.monotonic() - started},
            )
