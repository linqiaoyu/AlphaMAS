"""Frozen research contracts and graph-configuration identity helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

RESEARCH_DEPTH_ROUNDS = {"shallow": 1, "medium": 3, "deep": 5}

M2_GRAPH_KEYS = {
    "m2_trader_enabled",
    "m2_variant",
    "m2_method_id",
    "m2_representation_identity",
    "m2_checkpoint_parameter_identity",
    "m2_checkpoint_file_identity",
    "m2_online_candidate_id",
    "m2_online_learning_rate",
    "m2_online_update_epochs",
    "m2_online_weight_decay",
    "m2_online_gradient_clip",
    "m2_preformal_evidence_enabled",
    "m2_preformal_evidence_identity",
    "m2_preformal_evidence_role",
}

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
    "memory_outcome_price_mode": "adjusted_close",
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

FORMAL_M1_CONTRACT = {
    **FORMAL_M0_CONTRACT,
    "experiment_id_template": "M1_finmultitime_prompt_2024H1",
    "finmultitime_evidence_enabled": True,
    "finmultitime_bundle_scope": "FORMAL",
    "finmultitime_expected_contract_version": "M1-FINMULTITIME-v1.0.2",
    "finmultitime_expected_contract_sha256": (
        "46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08"
    ),
    "finmultitime_expected_packet_manifest_sha256": (
        "05f10129b430475bad3d5dee9dfceffba463f8a99cdd3f087ff4b755a869d63d"
    ),
    "finmultitime_expected_input_bundle_identity": (
        "30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121"
    ),
    "finmultitime_archive_commit": "3750fa50224ba46ab1d4bf5511cb5e8fa514445b",
    "finmultitime_verify_full_bundle_on_start": True,
}

FORMAL_M2_CONTRACT = {
    **FORMAL_M1_CONTRACT,
    "experiment_id_template": "M2_agentic_rl_2024H1",
    "m2_trader_enabled": True,
    "m2_variant": "FULL_M2",
    "m2_method_id": "M2-PA-CTPPO-v2",
    "m2_representation_identity": "6e3b11863bc3ec214444326a269477e465101afb866f30e80698f37c7148d2fe",
    "m2_checkpoint_parameter_identity": "6baafc03b0b63512b3a66a1ae8f1ce1ce7e774395787626b49905e7b72cd1841",
    "m2_checkpoint_file_identity": "56dc52128e1df9c9ddcf79fa6f7b293393bd61306ba4ef032f96cad6bf92126c",
    "m2_online_candidate_id": "O08",
    "m2_online_learning_rate": 1e-3,
    "m2_online_update_epochs": 2,
    "m2_online_weight_decay": 1e-4,
    "m2_online_gradient_clip": 0.5,
    "m2_preformal_evidence_enabled": False,
    "m2_preformal_evidence_identity": "3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420",
    "m2_preformal_evidence_role": "E2E_PILOT",
}

FORMAL_A1_CONTRACT = {
    **FORMAL_M2_CONTRACT,
    "experiment_id_template": "A1_no_online_adaptation_2024H1",
    "m2_variant": "A1_NO_ONLINE_ADAPTATION",
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
    "memory_outcome_price_mode",
    "checkpoint_enabled",
    "point_in_time",
    "execution_data_source",
    # Frozen M1 evidence identity is research-affecting. The input path is
    # excluded below, while the byte identity and exact case packet hash are
    # bound into cache/checkpoint identities by the runtime.
    "finmultitime_evidence_enabled",
    "finmultitime_expected_contract_version",
    "finmultitime_expected_contract_sha256",
    "finmultitime_expected_packet_manifest_sha256",
    "finmultitime_expected_input_bundle_identity",
    "finmultitime_bundle_scope",
    "finmultitime_archive_commit",
    "finmultitime_verify_full_bundle_on_start",
    # M2 Trader-only intervention. Operational model/state mount paths are
    # excluded from the stable graph hash below.
    "m2_trader_enabled",
    "m2_variant",
    "m2_method_id",
    "m2_representation_identity",
    "m2_checkpoint_parameter_identity",
    "m2_checkpoint_file_identity",
    "m2_online_candidate_id",
    "m2_online_learning_rate",
    "m2_online_update_epochs",
    "m2_online_weight_decay",
    "m2_online_gradient_clip",
    "m2_preformal_evidence_enabled",
    "m2_preformal_evidence_identity",
    "m2_preformal_evidence_role",
)

# Runtime values that must follow a backtest config into the Graph without
# becoming part of its research identity. Equivalent frozen bytes may be
# mounted at different absolute paths on different machines.
GRAPH_OPERATIONAL_KEYS = (
    "finmultitime_input_root",
    "m2_checkpoint_path",
    "m2_encoder_snapshot_path",
    "m2_encoder_snapshot_manifest_path",
    "m2_state_root",
    "m2_preformal_evidence_root",
)

GRAPH_HASH_EXCLUDED_KEYS = frozenset({
    "graph_config_sha256",
    "project_dir",
    "results_dir",
    "data_cache_dir",
    "memory_log_path",
    "historical_memory_log_path",
    "historical_memory_dir",
    "historical_memory_lineage_id",
    "historical_audit_path",
    "finmultitime_input_root",
    "m2_checkpoint_path",
    "m2_encoder_snapshot_path",
    "m2_encoder_snapshot_manifest_path",
    "m2_state_root",
    "m2_preformal_evidence_root",
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
    # These M1 controls deliberately default to the disabled M0-compatible
    # values. They become explicit in a future M1 formal config without
    # changing the frozen M0 config contract.
    derived_graph_keys = {
        "memory_holding_horizon_sessions",
        "execution_data_source",
        "finmultitime_evidence_enabled",
        "finmultitime_expected_contract_version",
        "finmultitime_expected_contract_sha256",
        "finmultitime_expected_packet_manifest_sha256",
        "finmultitime_expected_input_bundle_identity",
        "finmultitime_bundle_scope",
        "finmultitime_archive_commit",
        "finmultitime_verify_full_bundle_on_start",
        *M2_GRAPH_KEYS,
    }
    missing_graph = [
        key for key in GRAPH_RESEARCH_KEYS
        if key not in derived_graph_keys and key not in config
    ]
    if missing_graph:
        raise ValueError(f"formal M0 config leaves graph defaults implicit: {missing_graph}")
    validate_fixed_backtest_contract(config)


def validate_formal_m1_config(config: dict[str, Any]) -> None:
    """Fail if the checked-in Formal M1 protocol has drifted or is ambiguous."""
    missing = [key for key in FORMAL_M1_CONTRACT if key not in config]
    if missing:
        raise ValueError(f"formal M1 config is missing required fields: {missing}")
    unexpected = sorted(set(config) - set(FORMAL_M1_CONTRACT))
    if unexpected:
        raise ValueError(f"formal M1 config has unexpected fields: {unexpected}")
    mismatches = {
        key: {"expected": expected, "actual": config[key]}
        for key, expected in FORMAL_M1_CONTRACT.items()
        if not _contract_equal(config[key], expected)
    }
    if mismatches:
        raise ValueError(f"formal M1 config contract mismatch: {mismatches}")
    if resolve_research_rounds(config["research_depth"]) != 3:
        raise ValueError("formal M1 medium research depth must resolve to 3 rounds")
    derived_graph_keys = {
        "memory_holding_horizon_sessions",
        "execution_data_source",
        *M2_GRAPH_KEYS,
    }
    missing_graph = [
        key for key in GRAPH_RESEARCH_KEYS
        if key not in derived_graph_keys and key not in config
    ]
    if missing_graph:
        raise ValueError(f"formal M1 config leaves graph defaults implicit: {missing_graph}")
    validate_fixed_backtest_contract(config)


def validate_formal_m2_config(config: dict[str, Any]) -> None:
    """Fail if the final frozen Formal M2 protocol is incomplete or changed."""

    missing = [key for key in FORMAL_M2_CONTRACT if key not in config]
    if missing:
        raise ValueError(f"formal M2 config is missing required fields: {missing}")
    unexpected = sorted(set(config) - set(FORMAL_M2_CONTRACT))
    if unexpected:
        raise ValueError(f"formal M2 config has unexpected fields: {unexpected}")
    mismatches = {
        key: {"expected": expected, "actual": config[key]}
        for key, expected in FORMAL_M2_CONTRACT.items()
        if not _contract_equal(config[key], expected)
    }
    if mismatches:
        raise ValueError(f"formal M2 config contract mismatch: {mismatches}")
    if resolve_research_rounds(config["research_depth"]) != 3:
        raise ValueError("formal M2 medium research depth must resolve to 3 rounds")
    missing_graph = [
        key
        for key in GRAPH_RESEARCH_KEYS
        if key not in {"memory_holding_horizon_sessions", "execution_data_source"}
        and key not in config
    ]
    if missing_graph:
        raise ValueError(f"formal M2 config leaves graph defaults implicit: {missing_graph}")
    validate_fixed_backtest_contract(config)


def validate_formal_a1_config(config: dict[str, Any]) -> None:
    """Fail if Formal A1 differs from frozen M2 beyond its registered treatment."""

    missing = [key for key in FORMAL_A1_CONTRACT if key not in config]
    if missing:
        raise ValueError(f"formal A1 config is missing required fields: {missing}")
    unexpected = sorted(set(config) - set(FORMAL_A1_CONTRACT))
    if unexpected:
        raise ValueError(f"formal A1 config has unexpected fields: {unexpected}")
    mismatches = {
        key: {"expected": expected, "actual": config[key]}
        for key, expected in FORMAL_A1_CONTRACT.items()
        if not _contract_equal(config[key], expected)
    }
    if mismatches:
        raise ValueError(f"formal A1 config contract mismatch: {mismatches}")
    if resolve_research_rounds(config["research_depth"]) != 3:
        raise ValueError("formal A1 medium research depth must resolve to 3 rounds")
    missing_graph = [
        key
        for key in GRAPH_RESEARCH_KEYS
        if key not in {"memory_holding_horizon_sessions", "execution_data_source"}
        and key not in config
    ]
    if missing_graph:
        raise ValueError(f"formal A1 config leaves graph defaults implicit: {missing_graph}")
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
    for key in GRAPH_OPERATIONAL_KEYS:
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
