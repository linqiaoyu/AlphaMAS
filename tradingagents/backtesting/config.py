"""Frozen research contracts and graph-configuration identity helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

RESEARCH_DEPTH_ROUNDS = {"shallow": 1, "medium": 3, "deep": 5}

# Values supported by the first weekly engine. They remain explicit in the
# experiment preset, but are validated instead of pretending unsupported
# shorting, leverage, sizing, or liquidation policies already exist.
FIXED_BACKTEST_CONTRACT = {
    "long_only": True,
    "short_selling": False,
    "leverage": False,
    "buy_target_weight": 1.0,
    "sell_target_weight": 0.0,
    "hold": "preserve",
    "force_liquidate_at_end": False,
    "point_in_time": True,
}

FORMAL_M0_CONTRACT = {
    "experiment_id_template": "M0_original_prompt_2024H1",
    "symbols": ["AAPL", "AMZN", "JPM"],
    "calendar": "XNYS",
    "first_calendar_week": "2024-01-01",
    "final_calendar_week": "2024-06-24",
    "decision_weeks": 26,
    "final_valuation_session": "2024-07-05",
    "warmup_sessions": 252,
    "initial_cash": 100000,
    "commission_bps": 5,
    "slippage_bps": 5,
    "fractional_shares": True,
    **FIXED_BACKTEST_CONTRACT,
    "memory_mode": "experiment",
    "memory_log_max_entries": None,
    "holding_horizon_sessions": 5,
    "research_depth": "medium",
    "max_debate_rounds": 3,
    "max_risk_discuss_rounds": 3,
    "max_recur_limit": 100,
    "llm_provider": "deepseek",
    "quick_think_llm": "deepseek-v4-flash",
    "deep_think_llm": "deepseek-v4-flash",
    "backend_url": None,
    "deepseek_thinking": "disabled",
    "temperature": 0.0,
    "selected_analysts": ["market", "social", "news", "fundamentals"],
    "output_language": "English",
    "llm_max_retries": None,
    "news_article_limit": 20,
    "global_news_article_limit": 10,
    "global_news_lookback_days": 7,
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
        "macro_data": "fred",
        "prediction_markets": "polymarket",
    },
    "tool_vendors": {},
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS": "^NSEI",
        ".BO": "^BSESN",
        ".T": "^N225",
        ".HK": "^HSI",
        ".L": "^FTSE",
        ".TO": "^GSPTSE",
        ".AX": "^AXJO",
        ".SS": "000001.SS",
        ".SZ": "399001.SZ",
        "": "SPY",
    },
    "checkpoint_enabled": False,
    "strategy": "tradingagents",
    "risk_free_rate": 0.0,
    "annualization": 252,
    "data_source": "snapshot",
}

# Every mutable default that can affect graph output is copied into the formal
# preset and overlaid explicitly when constructing the graph. Operational paths
# are supplied separately and are intentionally excluded from the stable hash.
GRAPH_RESEARCH_KEYS = (
    "llm_provider",
    "quick_think_llm",
    "deep_think_llm",
    "backend_url",
    "deepseek_thinking",
    "temperature",
    "llm_max_retries",
    "selected_analysts",
    "output_language",
    "research_depth",
    "max_debate_rounds",
    "max_risk_discuss_rounds",
    "max_recur_limit",
    "news_article_limit",
    "global_news_article_limit",
    "global_news_lookback_days",
    "global_news_queries",
    "data_vendors",
    "tool_vendors",
    "benchmark_ticker",
    "benchmark_map",
    "memory_mode",
    "memory_log_max_entries",
    "memory_holding_horizon_sessions",
    "checkpoint_enabled",
    "point_in_time",
    "execution_data_source",
)

GRAPH_HASH_EXCLUDED_KEYS = frozenset({
    "graph_config_sha256",
    "project_dir",
    "results_dir",
    "data_cache_dir",
    "memory_log_path",
    "historical_memory_log_path",
    "historical_memory_dir",
    "historical_audit_path",
})

_SECRET_KEYS = frozenset({
    "api_key",
    "access_key",
    "access_token",
    "auth_token",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "secret",
    "secret_key",
})


def _contract_equal(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def resolve_research_rounds(depth: str) -> int:
    try:
        return RESEARCH_DEPTH_ROUNDS[depth.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown research depth: {depth}") from exc


def validate_fixed_backtest_contract(config: dict[str, Any]) -> None:
    """Reject values that the current engine cannot truthfully implement."""
    for key, expected in FIXED_BACKTEST_CONTRACT.items():
        if key in config and not _contract_equal(config[key], expected):
            raise ValueError(
                f"unsupported {key}={config[key]!r}; current fixed contract requires {expected!r}"
            )


def validate_formal_m0_config(config: dict[str, Any]) -> None:
    """Fail if the checked-in formal M0 protocol has drifted or is ambiguous."""
    missing = [key for key in FORMAL_M0_CONTRACT if key not in config]
    if missing:
        raise ValueError(f"formal M0 config is missing required fields: {missing}")
    unexpected = sorted(set(config) - set(FORMAL_M0_CONTRACT))
    if unexpected:
        raise ValueError(f"formal M0 config has unexpected fields: {unexpected}")
    mismatches = {
        key: {"expected": expected, "actual": config[key]}
        for key, expected in FORMAL_M0_CONTRACT.items()
        if not _contract_equal(config[key], expected)
    }
    if mismatches:
        raise ValueError(f"formal M0 config contract mismatch: {mismatches}")
    if resolve_research_rounds(config["research_depth"]) != 3:
        raise ValueError("formal M0 medium research depth must resolve to 3 rounds")
    derived_graph_keys = {"memory_holding_horizon_sessions", "execution_data_source"}
    missing_graph = [
        key for key in GRAPH_RESEARCH_KEYS
        if key not in derived_graph_keys and key not in config
    ]
    if missing_graph:
        raise ValueError(f"formal M0 config leaves graph defaults implicit: {missing_graph}")
    validate_fixed_backtest_contract(config)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if str(key).lower() not in _SECRET_KEYS
            and not str(key).lower().endswith("_api_key")
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sanitize_graph_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe graph config with credential fields removed."""
    return _sanitize(copy.deepcopy(config))


def graph_config_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Return the stable, research-affecting graph identity projection."""
    sanitized = sanitize_graph_config(config)
    return {
        key: sanitized.get(key)
        for key in GRAPH_RESEARCH_KEYS
        if key not in GRAPH_HASH_EXCLUDED_KEYS
    }


def compute_graph_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        graph_config_identity(config), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_graph_config(
    backtest_config: dict[str, Any], *, results_dir: str | Path,
    data_cache_dir: str | Path, historical_memory_dir: str | Path,
) -> dict[str, Any]:
    """Resolve exactly what ``TradingAgentsGraph`` will receive for a run."""
    from tradingagents.default_config import DEFAULT_CONFIG

    graph = copy.deepcopy(DEFAULT_CONFIG)
    for key in GRAPH_RESEARCH_KEYS:
        if key in backtest_config:
            graph[key] = copy.deepcopy(backtest_config[key])
    graph.update({
        "results_dir": str(results_dir),
        "data_cache_dir": str(data_cache_dir),
        "historical_memory_dir": str(historical_memory_dir),
        "memory_mode": backtest_config["memory_mode"],
        "memory_holding_horizon_sessions": backtest_config["holding_horizon_sessions"],
        "research_depth": backtest_config["research_depth"],
        "max_debate_rounds": int(backtest_config["max_debate_rounds"]),
        "max_risk_discuss_rounds": int(backtest_config["max_risk_discuss_rounds"]),
        "selected_analysts": list(backtest_config["selected_analysts"]),
        "point_in_time": bool(backtest_config["point_in_time"]),
        "execution_data_source": backtest_config["data_source"],
    })
    graph["graph_config_sha256"] = compute_graph_config_sha256(graph)
    return graph
