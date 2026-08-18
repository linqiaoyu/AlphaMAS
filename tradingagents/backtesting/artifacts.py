"""Analysis-ready, versioned artifacts for weekly backtest runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.agents.utils.memory_namespace import graph_memory_evidence_identity
from tradingagents.backtesting.cache import cache_key
from tradingagents.backtesting.config import compute_graph_config_sha256
from tradingagents.backtesting.memory_archive import (
    MEMORY_ARCHIVE_MANIFEST_PATH,
    validate_final_memory_archive,
)
from tradingagents.backtesting.metrics import compute_metrics, transaction_cost_components
from tradingagents.backtesting.recorder import equal_weight_aggregate, write_json

ARTIFACT_SCHEMA_VERSION = "1.3"
RESULT_TABLES = {
    "decisions.csv": "decisions",
    "orders.csv": "orders",
    "fills.csv": "fills",
    "daily_equity.csv": "daily_equity",
    "corporate_action_events.csv": "corporate_action_events",
}

CASE_INDEX_COLUMNS = (
    "experiment_id",
    "run_id",
    "case_id",
    "symbol",
    "decision_session",
    "decision_time_utc",
    "execution_session",
    "action",
    "decision_status",
    "rebalance_status",
    "provider",
    "quick_model",
    "deep_model",
    "thinking_mode",
    "temperature",
    "research_depth",
    "debate_rounds",
    "risk_rounds",
    "graph_config_sha256",
    "portfolio_equity_before",
    "portfolio_weight_before",
    "report_path",
    "source_audit_path",
    "cache_key",
    "cache_status",
    "wall_clock_seconds",
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
)

DATA_AVAILABILITY_COLUMNS = (
    "experiment_id",
    "run_id",
    "symbol",
    "decision_session",
    "source_name",
    "capability",
    "status",
    "requested_start",
    "requested_end",
    "latest_event_time",
    "latest_available_time",
    "reason",
)

LLM_USAGE_COLUMNS = (
    "experiment_id",
    "run_id",
    "case_id",
    "symbol",
    "decision_session",
    "usage_source",
    "origin_run_id",
    "agent_node",
    "provider",
    "model",
    "thinking_mode",
    "prompt_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "latency_seconds",
)

ANALYSIS_READY_SCHEMAS = {
    "equity_curves.csv": (
        "session", "symbol", "series", "equity", "normalized_equity",
    ),
    "drawdowns.csv": ("session", "symbol", "series", "drawdown"),
    "weekly_performance.csv": (
        "symbol",
        "decision_session",
        "next_evaluation_session",
        "action",
        "strategy_forward_return",
        "same_stock_buy_hold_forward_return",
        "underlying_forward_close_return",
        "SPY_forward_return",
        "strategy_excess_vs_stock",
        "strategy_excess_vs_spy",
        "commission_cost_in_period",
        "slippage_cost_in_period",
        "transaction_cost_in_period",
        "average_exposure_in_period",
    ),
    "decision_timeline.csv": (
        "symbol",
        "decision_session",
        "decision_time_utc",
        "action",
        "status",
        "reason",
        "target_weight",
        "portfolio_weight_before",
        "equity_at_decision",
        "rebalance_status",
        "execution_session",
        "order_status",
        "raw_open_price",
        "fill_price",
        "fill_quantity",
        "commission",
        "slippage_cost",
        "total_transaction_cost",
        "position_after",
        "cache_status",
    ),
    "metrics_long.csv": (
        "experiment_id", "run_id", "symbol", "series", "metric", "value",
    ),
    "action_summary.csv": (
        "symbol",
        "buy_count",
        "hold_count",
        "sell_count",
        "failed_count",
        "no_op_count",
        "actual_trades",
        "time_in_market",
    ),
    "case_index.csv": CASE_INDEX_COLUMNS,
    "data_availability.csv": DATA_AVAILABILITY_COLUMNS,
    "llm_usage.csv": LLM_USAGE_COLUMNS,
}

AnalysisRecords = Iterable[Mapping[str, Any]] | pd.DataFrame | None


def _fixed_schema_frame(records: AnalysisRecords, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return records in a stable column order, including headers when empty."""
    if records is None:
        return pd.DataFrame(columns=columns)
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame.from_records(list(records))
    return frame.reindex(columns=columns)


def build_case_index(records: AnalysisRecords = None) -> pd.DataFrame:
    """Build the one-row-per-Agent-case analysis table."""
    return _fixed_schema_frame(records, CASE_INDEX_COLUMNS)


def build_data_availability(records: AnalysisRecords = None) -> pd.DataFrame:
    """Build the long-form source availability table."""
    return _fixed_schema_frame(records, DATA_AVAILABILITY_COLUMNS)


def build_llm_usage(records: AnalysisRecords = None) -> pd.DataFrame:
    """Build the request-level raw provider usage table."""
    return _fixed_schema_frame(records, LLM_USAGE_COLUMNS)


def flatten_source_audit(
    audit: Mapping[str, Any], *, experiment_id: str, run_id: str,
    symbol: str, decision_session: str,
) -> list[dict[str, Any]]:
    """Add case identity to every source record without changing audit semantics."""
    rows = []
    sources = audit.get("sources", ())
    if not isinstance(sources, list | tuple):
        raise ValueError("source audit 'sources' must be a list")
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("every source audit record must be an object")
        rows.append({
            "experiment_id": experiment_id,
            "run_id": run_id,
            "symbol": symbol,
            "decision_session": decision_session,
            **{column: source.get(column) for column in DATA_AVAILABILITY_COLUMNS[4:]},
        })
    return rows


def write_agent_analysis_tables(
    root: str | Path, *, case_index: AnalysisRecords = None,
    data_availability: AnalysisRecords = None, llm_usage: AnalysisRecords = None,
) -> dict[str, Path]:
    """Write fixed-schema Agent tables; deterministic runs get header-only files."""
    output_root = Path(root)
    output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "case_index.csv": build_case_index(case_index),
        "data_availability.csv": build_data_availability(data_availability),
        "llm_usage.csv": build_llm_usage(llm_usage),
    }
    paths = {}
    for filename, frame in tables.items():
        path = output_root / filename
        frame.to_csv(path, index=False)
        paths[filename] = path
    return paths


def _case_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid case artifact {path}: {exc}") from exc


def _usage_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {}
    for column in (
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        values = [
            value for value in (record.get(column) for record in records)
            if value is not None and not pd.isna(value)
        ]
        totals[column] = sum(values) if values else None
    return totals


def collect_agent_analysis_tables(
    run_root: str | Path, *, experiment_id: str, run_id: str,
    results: Mapping[str, Any], graph_config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate canonical per-case artifacts without scanning them during analysis later."""
    root = Path(run_root)
    if graph_config is None:
        loaded = _case_json(root / "graph_config.resolved.json", {})
        graph_config = loaded if isinstance(loaded, Mapping) else {}
    try:
        schedule = pd.read_csv(root / "schedule.csv").set_index("decision_session")
    except (OSError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError):
        schedule = pd.DataFrame()

    case_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    for symbol, result in results.items():
        orders = result.orders.copy()
        daily = result.daily_equity.set_index("session")
        for _, decision in result.decisions.iterrows():
            if decision.get("strategy_id") != "tradingagents":
                continue
            session = str(decision["decision_session"])
            case_root = root / "strategy" / "cases" / symbol / session
            case_metadata = _case_json(case_root / "case_metadata.json", {})
            if not isinstance(case_metadata, Mapping):
                case_metadata = {}
            metadata = decision.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            case_id = str(case_metadata.get("case_id") or f"{symbol}:{session}")

            source_audit = _case_json(case_root / "source_audit.json", {})
            if not isinstance(source_audit, Mapping):
                raise ValueError(f"source audit must be an object: {case_root}")
            availability_rows.extend(flatten_source_audit(
                source_audit,
                experiment_id=experiment_id,
                run_id=run_id,
                symbol=symbol,
                decision_session=session,
            ))

            case_usage = _case_json(case_root / "llm_usage.json", [])
            usage_source = "live_request"
            if not case_usage:
                case_usage = _case_json(case_root / "cached_origin_llm_usage.json", [])
                usage_source = "cache_origin"
            if not isinstance(case_usage, list):
                raise ValueError(f"LLM usage must be a list: {case_root}")
            normalized_usage = []
            for raw_usage in case_usage:
                if not isinstance(raw_usage, Mapping):
                    raise ValueError(f"LLM usage rows must be objects: {case_root}")
                row = dict(raw_usage)
                origin_run_id = row.get("origin_run_id") or row.get("run_id")
                row.update({
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "case_id": case_id,
                    "symbol": symbol,
                    "decision_session": session,
                    "usage_source": usage_source,
                    "origin_run_id": origin_run_id or (
                        run_id if usage_source == "live_request" else None
                    ),
                    "provider": row.get("provider") or graph_config.get("llm_provider"),
                    "thinking_mode": (
                        row.get("thinking_mode") or graph_config.get("deepseek_thinking")
                    ),
                })
                normalized_usage.append(row)
            usage_rows.extend(normalized_usage)

            order = pd.Series(dtype=object)
            if not orders.empty:
                matches = orders.loc[
                    orders["created_at"].astype(str).str[:10] == session
                ]
                if not matches.empty:
                    order = matches.iloc[0]
            execution_session = order.get("intended_execution_session", "")
            if not execution_session and not schedule.empty and session in schedule.index:
                execution_session = schedule.loc[session].get("execution_session", "")
            portfolio = daily.loc[session]
            case_rows.append({
                "experiment_id": experiment_id,
                "run_id": run_id,
                "case_id": case_id,
                "symbol": symbol,
                "decision_session": session,
                "decision_time_utc": decision.get("decision_time_utc"),
                "execution_session": execution_session,
                "action": decision.get("action"),
                "decision_status": decision.get("status"),
                "rebalance_status": decision.get("rebalance_status", ""),
                "provider": graph_config.get("llm_provider"),
                "quick_model": graph_config.get("quick_think_llm"),
                "deep_model": graph_config.get("deep_think_llm"),
                "thinking_mode": graph_config.get("deepseek_thinking"),
                "temperature": graph_config.get("temperature"),
                "research_depth": graph_config.get("research_depth"),
                "debate_rounds": graph_config.get("max_debate_rounds"),
                "risk_rounds": graph_config.get("max_risk_discuss_rounds"),
                "graph_config_sha256": graph_config.get("graph_config_sha256"),
                "portfolio_equity_before": portfolio.get("equity"),
                "portfolio_weight_before": portfolio.get("current_weight"),
                "report_path": metadata.get("report_path") or case_metadata.get("report_path"),
                "source_audit_path": (
                    metadata.get("source_audit_path")
                    or case_metadata.get("source_audit_path")
                ),
                "cache_key": metadata.get("cache_key") or case_metadata.get("cache_key"),
                "cache_status": (
                    metadata.get("cache_status") or case_metadata.get("cache_status")
                ),
                "wall_clock_seconds": (
                    metadata.get("wall_clock_seconds")
                    if metadata.get("wall_clock_seconds") is not None
                    else case_metadata.get("wall_clock_seconds")
                ),
                **_usage_totals(normalized_usage),
            })
    return (
        build_case_index(case_rows),
        build_data_availability(availability_rows),
        build_llm_usage(usage_rows),
    )


def collect_agent_analysis_records(
    experiment_id: str, run_id: str, results: Mapping[str, Any],
    cases_root: str | Path, graph_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Record-list API used by the runner after canonical case artifacts exist."""
    cases_path = Path(cases_root)
    run_root = cases_path.parent.parent
    frames = collect_agent_analysis_tables(
        run_root,
        experiment_id=experiment_id,
        run_id=run_id,
        results=results,
        graph_config=graph_config,
    )
    output = []
    for frame in frames:
        normalized = frame.astype(object).where(pd.notna(frame), None)
        output.append(normalized.to_dict("records"))
    return output[0], output[1], output[2]


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(payload.encode()).hexdigest()


def result_hashes(result: Any) -> dict[str, str]:
    hashes = {name: frame_hash(getattr(result, attr)) for name, attr in RESULT_TABLES.items()}
    encoded = json.dumps(result.metrics, sort_keys=True, allow_nan=False).encode()
    hashes["metrics.json"] = hashlib.sha256(encoded).hexdigest()
    return hashes


def save_result(root: Path, result: Any) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for filename, attribute in RESULT_TABLES.items():
        getattr(result, attribute).to_csv(root / filename, index=False)
    write_json(root / "metrics.json", result.metrics)
    pd.DataFrame([result.metrics]).to_csv(root / "metrics.csv", index=False)


def _combined(results: dict[str, Any], attribute: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    empty_template: pd.DataFrame | None = None
    for symbol, result in results.items():
        frame = getattr(result, attribute).copy()
        if "account_symbol" not in frame:
            frame.insert(0, "account_symbol", symbol)
        if frame.empty:
            empty_template = frame
        else:
            frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return empty_template if empty_template is not None else pd.DataFrame()


def save_strategy(root: Path, results: dict[str, Any]) -> None:
    for symbol, result in results.items():
        save_result(root / symbol, result)
    combined = root / "combined"
    combined.mkdir(parents=True, exist_ok=False)
    for filename, attribute in RESULT_TABLES.items():
        _combined(results, attribute).to_csv(combined / filename, index=False)
    pd.DataFrame([
        {"symbol": symbol, **result.metrics} for symbol, result in results.items()
    ]).to_csv(combined / "metrics.csv", index=False)


def aggregate_outputs(results: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregate = equal_weight_aggregate(results)
    first_metrics = next(iter(results.values())).metrics
    metrics = compute_metrics(
        aggregate["normalized_equity"], initial_equity=1.0,
        risk_free_rate=first_metrics["risk_free_rate"],
        annualization=first_metrics["annualization_factor"],
    )
    total_initial = sum(float(result.daily_equity.iloc[0]["equity"]) for result in results.values())
    commission_cost = sum(
        float(result.metrics["total_commission_cost"])
        for result in results.values()
    )
    slippage_cost = sum(
        float(result.metrics["total_slippage_cost"])
        for result in results.values()
    )
    total_cost = commission_cost + slippage_cost
    total_notional = sum(
        float(result.fills.get("notional", pd.Series(dtype=float)).abs().sum())
        for result in results.values()
    )
    total_average_equity = sum(
        float(result.daily_equity["equity"].mean()) for result in results.values()
    )
    sum_keys = (
        "trade_count", "decision_count", "decision_failure_count", "cumulative_dividends",
        "buy_decision_count", "hold_decision_count", "sell_decision_count",
        "noop_rebalance_count", "filled_order_count", "rejected_order_count",
        "model_failure_count",
    )
    metrics.update({key: sum(float(result.metrics.get(key, 0)) for result in results.values())
                    for key in sum_keys})
    for integer_key in set(sum_keys) - {"cumulative_dividends"}:
        metrics[integer_key] = int(metrics[integer_key])
    metrics.update({
        "decision_failure_rate": (
            metrics["decision_failure_count"] / metrics["decision_count"]
            if metrics["decision_count"] else 0.0
        ),
        "model_failure_rate": (
            metrics["model_failure_count"] / metrics["decision_count"]
            if metrics["decision_count"] else 0.0
        ),
        "total_commission_cost": commission_cost,
        "total_slippage_cost": slippage_cost,
        "total_transaction_cost": total_cost,
        "commission_cost_rate": commission_cost / total_initial,
        "slippage_cost_rate": slippage_cost / total_initial,
        "transaction_cost_rate": total_cost / total_initial,
        "turnover": total_notional / total_average_equity,
        "average_exposure": float(np.mean([
            result.metrics["average_exposure"] for result in results.values()
        ])),
        "time_in_market": float(np.mean([
            result.metrics["time_in_market"] for result in results.values()
        ])),
        "metric_scope": {
            "return_and_risk": "equal_weight_aggregate_daily_equity",
            "transactions_and_behaviour": "sum_or_equal_weight_mean_of_constituent_accounts",
        },
    })
    return aggregate, metrics


def _curve(frame: pd.DataFrame, symbol: str, series: str) -> pd.DataFrame:
    output = frame.loc[:, ["session", "equity"]].copy()
    output.insert(1, "symbol", symbol)
    output.insert(2, "series", series)
    output["normalized_equity"] = output["equity"] / output["equity"].iloc[0]
    return output


def analysis_ready(
    root: Path, *, experiment_id: str, run_id: str, results: dict[str, Any],
    stock_benchmarks: dict[str, Any], spy: Any, aggregate: pd.DataFrame,
    aggregate_metrics: dict[str, Any], market_data: dict[str, pd.DataFrame],
    case_index: AnalysisRecords = None, data_availability: AnalysisRecords = None,
    llm_usage: AnalysisRecords = None,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    curves: list[pd.DataFrame] = []
    for symbol, result in results.items():
        curves.append(_curve(result.daily_equity, symbol, "strategy"))
        curves.append(_curve(
            stock_benchmarks[symbol].daily_equity, symbol, "same_stock_buy_and_hold"
        ))
    curves.append(_curve(spy.daily_equity, "SPY", "SPY"))
    aggregate_curve = aggregate.rename(columns={"normalized_equity": "equity"}).copy()
    curves.append(_curve(aggregate_curve, "ALL", "equal_weight_strategy"))
    equity_curves = pd.concat(curves, ignore_index=True)
    equity_curves.to_csv(root / "equity_curves.csv", index=False)
    drawdowns = equity_curves.loc[:, ["session", "symbol", "series"]].copy()
    drawdowns["drawdown"] = equity_curves.groupby(
        ["symbol", "series"], sort=False
    )["equity"].transform(lambda values: values / values.cummax() - 1)
    drawdowns.to_csv(root / "drawdowns.csv", index=False)

    metric_sets: list[tuple[str, str, dict[str, Any]]] = []
    for symbol, result in results.items():
        metric_sets.extend((
            (symbol, "strategy", result.metrics),
            (symbol, "same_stock_buy_and_hold", stock_benchmarks[symbol].metrics),
        ))
    metric_sets.extend((("SPY", "SPY", spy.metrics),
                        ("ALL", "equal_weight_strategy", aggregate_metrics)))
    metric_rows = []
    for symbol, series, metrics in metric_sets:
        for metric, value in metrics.items():
            if metric != "metric_scope":
                metric_rows.append({
                    "experiment_id": experiment_id, "run_id": run_id,
                    "symbol": symbol, "series": series, "metric": metric, "value": value,
                })
    pd.DataFrame(metric_rows).to_csv(root / "metrics_long.csv", index=False)
    _decision_timeline(results).to_csv(root / "decision_timeline.csv", index=False)
    _weekly_performance(results, stock_benchmarks, spy, market_data).to_csv(
        root / "weekly_performance.csv", index=False
    )
    pd.DataFrame([{
        "symbol": symbol,
        "buy_count": result.metrics["buy_decision_count"],
        "hold_count": result.metrics["hold_decision_count"],
        "sell_count": result.metrics["sell_decision_count"],
        "failed_count": result.metrics["decision_failure_count"],
        "no_op_count": result.metrics["noop_rebalance_count"],
        "actual_trades": result.metrics["trade_count"],
        "time_in_market": result.metrics["time_in_market"],
    } for symbol, result in results.items()]).to_csv(root / "action_summary.csv", index=False)
    if case_index is None or data_availability is None or llm_usage is None:
        generated = collect_agent_analysis_tables(
            root.parent,
            experiment_id=experiment_id,
            run_id=run_id,
            results=results,
        )
        if case_index is None:
            case_index = generated[0]
        if data_availability is None:
            data_availability = generated[1]
        if llm_usage is None:
            llm_usage = generated[2]
    write_agent_analysis_tables(
        root, case_index=case_index, data_availability=data_availability,
        llm_usage=llm_usage,
    )


def _decision_timeline(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, result in results.items():
        orders = result.orders.copy()
        fills = result.fills.copy()
        for _, decision in result.decisions.iterrows():
            metadata = decision.get("metadata")
            cache_status = (
                metadata.get("cache_status")
                if isinstance(metadata, Mapping) else ""
            )
            order = pd.Series(dtype=object)
            if not orders.empty:
                match = orders.loc[
                    orders["created_at"].astype(str).str[:10] == decision["decision_session"]
                ]
                if not match.empty:
                    order = match.iloc[0]
            fill = pd.Series(dtype=object)
            if not order.empty and not fills.empty:
                match = fills.loc[fills["order_id"] == order["order_id"]]
                if not match.empty:
                    fill = match.iloc[0]
            daily = result.daily_equity.set_index("session").loc[decision["decision_session"]]
            rows.append({
                "symbol": symbol, "decision_session": decision["decision_session"],
                "decision_time_utc": decision["decision_time_utc"],
                "action": decision.get("action"), "status": decision["status"],
                "reason": decision["reason"], "target_weight": decision.get("target_weight"),
                "portfolio_weight_before": daily["current_weight"],
                "equity_at_decision": daily["equity"],
                "rebalance_status": decision.get("rebalance_status", ""),
                "execution_session": order.get("intended_execution_session", ""),
                "order_status": order.get("status", ""),
                "raw_open_price": fill.get("raw_open_price", ""),
                "fill_price": fill.get("fill_price", ""),
                "fill_quantity": fill.get("quantity", ""),
                "commission": fill.get("commission", ""),
                "slippage_cost": fill.get("slippage_cost", ""),
                "total_transaction_cost": fill.get("total_transaction_cost", ""),
                "position_after": fill.get("position_after", ""),
                # Cache provenance is independent of decision success/cached status:
                # forced recomputation is a bypass, not a miss.
                "cache_status": cache_status if cache_status is not None else "",
            })
    return pd.DataFrame(rows)


def _period_return(frame: pd.DataFrame, start: str, end: str) -> float:
    values = frame.set_index("session")["equity"]
    return float(values.loc[end] / values.loc[start] - 1)


def _weekly_performance(
    results: dict[str, Any], stock_benchmarks: dict[str, Any], spy: Any,
    market_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spy_daily = spy.daily_equity
    for symbol, result in results.items():
        decisions = result.decisions.reset_index(drop=True)
        for index, decision in decisions.iterrows():
            start = decision["decision_session"]
            end = (
                decisions.iloc[index + 1]["decision_session"]
                if index + 1 < len(decisions) else result.daily_equity.iloc[-1]["session"]
            )
            data = market_data[symbol]
            start_close, end_close = data.loc[start, "Close"], data.loc[end, "Close"]
            fills = result.fills
            dates = fills.get("execution_time", pd.Series(index=fills.index, dtype=str))
            period_fills = fills.loc[
                (dates.astype(str).str[:10] > start)
                & (dates.astype(str).str[:10] <= end)
            ]
            period_costs = transaction_cost_components(period_fills)
            commission_cost = float(period_costs["commission_cost"].sum())
            slippage_cost = float(period_costs["slippage_cost"].sum())
            period_cost = float(period_costs["total_transaction_cost"].sum())
            daily = result.daily_equity.set_index("session").loc[start:end]
            stock_return = _period_return(stock_benchmarks[symbol].daily_equity, start, end)
            strategy_return = _period_return(result.daily_equity, start, end)
            spy_return = _period_return(spy_daily, start, end)
            rows.append({
                "symbol": symbol, "decision_session": start,
                "next_evaluation_session": end, "action": decision.get("action"),
                "strategy_forward_return": strategy_return,
                "same_stock_buy_hold_forward_return": stock_return,
                "underlying_forward_close_return": float(end_close / start_close - 1),
                "SPY_forward_return": spy_return,
                "strategy_excess_vs_stock": strategy_return - stock_return,
                "strategy_excess_vs_spy": strategy_return - spy_return,
                "commission_cost_in_period": commission_cost,
                "slippage_cost_in_period": slippage_cost,
                "transaction_cost_in_period": period_cost,
                "average_exposure_in_period": float(daily["current_weight"].mean()),
            })
    return pd.DataFrame(rows)


def artifact_schema() -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "time": {
            "session": "XNYS session date (YYYY-MM-DD)",
            "decision": "last actual XNYS close in each calendar week",
            "execution": "next actual XNYS session open; never same-bar",
        },
        "definitions": {
            "daily_return": "close-to-close portfolio equity percentage change",
            "drawdown": "equity / running_peak_equity - 1",
            "turnover": "sum(abs(fill_notional)) / average_daily_equity",
            "total_commission_cost": "sum(fill commission)",
            "total_slippage_cost": (
                "sum(signed quantity * (fill_price - raw_open_price))"
            ),
            "total_transaction_cost": (
                "total_commission_cost + total_slippage_cost; informational only because "
                "commission and slipped fill prices already affect equity"
            ),
            "transaction_cost_rate": "total_transaction_cost / initial_equity",
            "sortino_ratio": (
                "mean(daily_return-daily_rf) / sqrt(mean(min(daily_return-daily_rf,0)^2)) "
                "* sqrt(annualization); null when denominator is zero"
            ),
            "corporate_actions": (
                "raw prices; dividends credited and splits applied at effective-session open "
                "before orders, based on the prior-close position"
            ),
            "same_stock_benchmark": "100% long same symbol from first available next-open",
            "SPY_benchmark": "100% long SPY from first available next-open",
            "final_valuation": "marked at configured final XNYS close; no forced liquidation",
            "final_memory_archive": (
                "one-way byte copies of the final symbol-specific experiment Memory; "
                "read-only research artifacts that are never inputs to runtime or resume"
            ),
        },
        "memory_artifacts": {
            "memory/manifest.json": (
                "run, lineage, Graph, and symbol provenance plus per-file SHA-256; "
                "its own SHA-256 is bound into the top-level manifest"
            ),
            "memory/symbols/<symbol>.md": (
                "exact final TradingMemoryLog state copied out of the operational "
                "runtime namespace"
            ),
        },
        "csv_files": {
            "inputs/market_data/<symbol>.csv": "canonical exact raw OHLCV plus actions",
            "strategy/*": "strategy decisions, orders, fills, actions and daily portfolio state",
            "benchmarks/*": "benchmark artifacts with the same accounting policy",
            "analysis_ready/equity_curves.csv": "long-form daily equity and normalized equity",
            "analysis_ready/drawdowns.csv": "long-form daily drawdowns",
            "analysis_ready/weekly_performance.csv": "realized decision-close forward periods",
            "analysis_ready/decision_timeline.csv": "one enriched row per weekly decision",
            "analysis_ready/metrics_long.csv": "long-form metrics for paper tables",
            "analysis_ready/action_summary.csv": "per-symbol action and trading counts",
            "analysis_ready/case_index.csv": "one row per TradingAgents decision case",
            "analysis_ready/data_availability.csv": (
                "one row per historical source-audit record"
            ),
            "analysis_ready/llm_usage.csv": (
                "one row per real LLM request using provider-reported usage"
            ),
        },
        "analysis_ready_schemas": {
            filename: list(columns) for filename, columns in ANALYSIS_READY_SCHEMAS.items()
        },
        "columns": {
            "Date": "market session date",
            "Open/High/Low/Close": "raw, unadjusted session prices",
            "Volume": "reported shares traded",
            "Dividends": "cash dividend per eligible share on the session",
            "Stock Splits": "effective split ratio; zero means no split",
            "decision_session": "session whose close generated the decision",
            "decision_time_utc": "actual decision-session close in UTC",
            "action": "BUY, HOLD, SELL, or empty for a failed decision",
            "status": "success, cached, failed, pending, filled, rejected, or noop",
            "target_weight": "requested long-only portfolio weight",
            "execution_session": "next XNYS session intended for open execution",
            "raw_open_price": "unadjusted market open before slippage",
            "fill_price": "execution price after directional slippage",
            "quantity": "signed fill shares bought (positive) or sold (negative)",
            "fill_quantity": "decision-timeline alias of signed fill quantity",
            "commission": "cash commission charged on the fill",
            "slippage_cost": (
                "signed fill quantity times (fill price - raw open price); positive for "
                "the broker's adverse buys and sells"
            ),
            "total_transaction_cost": "commission plus slippage cost",
            "equity": "cash plus close-marked position market value",
            "realized_pnl": (
                "completed round-trip price P&L net of entry and exit commissions exactly once"
            ),
            "unrealized_pnl": (
                "open-position mark-to-fill-price P&L net of its entry commission"
            ),
            "cumulative_cost": (
                "compatibility alias for cumulative_transaction_cost; informational and not "
                "subtracted from equity again"
            ),
            "cumulative_commission_cost": "running sum of fill commissions",
            "cumulative_slippage_cost": "running sum of fill slippage costs",
            "cumulative_transaction_cost": (
                "running commission plus slippage; equity already bears both components"
            ),
            "normalized_equity": "equity divided by the series first equity",
            "drawdown": "signed decline from running peak (zero at a peak)",
            "cash_effect": "cash credited by a corporate-action event",
            "strategy_forward_return": "equity return from decision close to next evaluation close",
            "commission_cost_in_period": (
                "commissions on fills after the decision through period end"
            ),
            "slippage_cost_in_period": (
                "recomputed slippage on fills after the decision through period end"
            ),
            "transaction_cost_in_period": (
                "commission plus slippage on fills after the decision through period end"
            ),
            "average_exposure_in_period": "mean daily position market value/equity in the period",
            "metric": "stable metric name",
            "value": "numeric metric value or empty when undefined",
            "case_id": "stable identity of one symbol and decision-session Agent case",
            "decision_status": "success, cached, or failed Agent decision status",
            "source_audit_path": "run-relative path to the immutable source audit",
            "report_path": "run-relative path to the immutable report tree",
            "cache_key": "content-addressed decision cache identity",
            "cache_status": "miss, hit, or bypass for the current case",
            "market_history_visibility": (
                "engine-owned audit of the exact strategy-visible history, including its "
                "maximum session, corresponding XNYS close time, row count, and SHA-256"
            ),
            "wall_clock_seconds": (
                "elapsed time for the current case materialization; cache hits measure replay"
            ),
            "graph_config_sha256": "canonical hash of the research-affecting graph config",
            "agent_node": "LangGraph node or Agent identity when provider metadata exposes it",
            "usage_source": (
                "live_request for a request made in this run; cache_origin for the "
                "preserved request that originally generated a replayed decision"
            ),
            "origin_run_id": "run in which the provider request was actually made",
            "prompt_tokens": (
                "provider-reported input token count; empty when unavailable; case_index "
                "cache-hit totals describe the preserved origin request"
            ),
            "prompt_cache_hit_tokens": (
                "provider-reported cached input tokens; empty when unavailable"
            ),
            "prompt_cache_miss_tokens": (
                "provider-reported uncached input tokens; empty when unavailable"
            ),
            "completion_tokens": "provider-reported output token count; empty when unavailable",
            "reasoning_tokens": "provider-reported reasoning tokens; empty when unavailable",
            "total_tokens": "provider-reported total token count; empty when unavailable",
            "latency_seconds": "local monotonic request duration in seconds",
        },
        "units": {
            "prices_cash_equity_transaction_costs": "account currency",
            "quantity": "shares (fractional when configured)",
            "rates_returns_weights_drawdowns": "decimal fraction",
            "slippage_bps": "basis points",
        },
    }


def graph_config_sha256(config: Mapping[str, Any]) -> str:
    """Compatibility alias for the canonical research-identity graph hash."""
    return compute_graph_config_sha256(dict(config))


def _read_json_document(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
        return None


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _bundle_member(root: Path, value: Any) -> Path | None:
    """Resolve a portable run-relative member and reject absolute/path traversal values."""
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    relative = Path(str(value))
    if relative.is_absolute():
        return None
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        return None
    return candidate


def validate_artifact_bundle(
    run_dir: str | Path, *, symbols: Iterable[str] | None = None,
    strategy_name: str | None = None, require_success: bool = True,
    require_validation_report: bool = True,
) -> dict[str, Any]:
    """Validate the formal immutable bundle without invoking data vendors or an LLM.

    The function is read-only and returns a machine-readable report. Callers must
    refuse to publish a successful run when ``status`` is ``"failed"``.
    """
    root = Path(run_dir)
    missing_files: list[str] = []
    invalid_json: list[str] = []
    schema_errors: list[str] = []
    checksum_errors: list[str] = []
    agent_artifact_errors: list[str] = []
    lineage_errors: list[str] = []

    required = {
        "manifest.json",
        "config.resolved.json",
        "graph_config.resolved.json",
        "environment.json",
        "artifact_schema.json",
        "run_status.json",
        "schedule.csv",
        "inputs/data_manifest.json",
        "aggregate/equal_weight_equity.csv",
        "aggregate/equal_weight_metrics.json",
        "failures/failures.jsonl",
        *(f"analysis_ready/{filename}" for filename in ANALYSIS_READY_SCHEMAS),
    }
    if require_validation_report:
        required.add("validation/validation_report.json")

    base_documents: dict[str, Any] = {}
    for relative in (
        "manifest.json",
        "config.resolved.json",
        "graph_config.resolved.json",
        "environment.json",
        "artifact_schema.json",
        "run_status.json",
        "inputs/data_manifest.json",
    ):
        path = root / relative
        if path.is_file():
            base_documents[relative] = _read_json_document(path, invalid_json)

    config = base_documents.get("config.resolved.json")
    if not isinstance(config, Mapping):
        config = {}
    configured_symbols = config.get("symbols", ())
    symbol_list = list(symbols if symbols is not None else configured_symbols)
    if not symbol_list:
        schema_errors.append("config.resolved.json: symbols must be a non-empty list")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbol_list):
        schema_errors.append("symbols must contain non-empty strings")
        symbol_list = [symbol for symbol in symbol_list if isinstance(symbol, str) and symbol]

    strategy = strategy_name or str(config.get("strategy", ""))
    require_memory_archive = strategy == "tradingagents" and require_success
    if require_memory_archive:
        required.add(MEMORY_ARCHIVE_MANIFEST_PATH)
    result_files = [*RESULT_TABLES, "metrics.json", "metrics.csv"]
    for symbol in symbol_list:
        required.update(f"strategy/{symbol}/{filename}" for filename in result_files)
        required.update(
            f"benchmarks/{symbol}_buy_and_hold/{filename}" for filename in result_files
        )
    required.update(f"strategy/combined/{filename}" for filename in [*RESULT_TABLES, "metrics.csv"])
    required.update(f"benchmarks/SPY_buy_and_hold/{filename}" for filename in result_files)

    data_manifest = base_documents.get("inputs/data_manifest.json")
    manifest_entries: list[Any] = []
    if isinstance(data_manifest, Mapping) and isinstance(data_manifest.get("symbols"), list):
        manifest_entries = data_manifest["symbols"]
    else:
        schema_errors.append("inputs/data_manifest.json: symbols must be a list")
    manifest_symbols = {
        entry.get("symbol") for entry in manifest_entries if isinstance(entry, Mapping)
    }
    missing_manifest_symbols = {*symbol_list, "SPY"} - manifest_symbols
    if missing_manifest_symbols:
        schema_errors.append(
            "inputs/data_manifest.json: missing symbols "
            f"{sorted(missing_manifest_symbols)}"
        )
    for entry in manifest_entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("symbol"), str):
            schema_errors.append("inputs/data_manifest.json: every symbol entry must be an object")
            continue
        symbol = entry["symbol"]
        market_relative = f"inputs/market_data/{symbol}.csv"
        required.add(market_relative)
        required.add(f"inputs/corporate_actions/{symbol}.csv")
        market_path = root / market_relative
        expected_hash = entry.get("sha256")
        if not _valid_sha256(expected_hash):
            checksum_errors.append(f"{symbol}: invalid or missing manifest sha256")
        elif market_path.is_file():
            actual_hash = hashlib.sha256(market_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                checksum_errors.append(
                    f"{symbol}: expected {expected_hash}, observed {actual_hash}"
                )

    for relative in sorted(required):
        if not (root / relative).is_file():
            missing_files.append(relative)

    for relative in sorted(required):
        path = root / relative
        if path.is_file() and path.suffix == ".json" and relative not in base_documents:
            _read_json_document(path, invalid_json)

    analysis_frames: dict[str, pd.DataFrame] = {}
    for filename, expected_columns in ANALYSIS_READY_SCHEMAS.items():
        path = root / "analysis_ready" / filename
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pandas raises several parser/empty-file exception types
            schema_errors.append(f"analysis_ready/{filename}: {type(exc).__name__}: {exc}")
            continue
        analysis_frames[filename] = frame
        if tuple(frame.columns) != tuple(expected_columns):
            schema_errors.append(
                f"analysis_ready/{filename}: columns must exactly match the declared schema"
            )

    schedule_path = root / "schedule.csv"
    schedule_frame: pd.DataFrame | None = None
    if schedule_path.is_file():
        try:
            schedule_frame = pd.read_csv(schedule_path)
            schedule_columns = set(schedule_frame.columns)
        except Exception as exc:
            schema_errors.append(f"schedule.csv: {type(exc).__name__}: {exc}")
        else:
            missing_schedule_columns = {
                "decision_session", "execution_session",
            } - schedule_columns
            if missing_schedule_columns:
                schema_errors.append(
                    f"schedule.csv: missing columns {sorted(missing_schedule_columns)}"
                )

    artifact_schema_document = base_documents.get("artifact_schema.json")
    if (
        not isinstance(artifact_schema_document, Mapping)
        or artifact_schema_document.get("schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        schema_errors.append(
            f"artifact_schema.json: schema_version must be {ARTIFACT_SCHEMA_VERSION}"
        )

    graph_hash_documents = {
        relative: base_documents.get(relative)
        for relative in ("manifest.json", "config.resolved.json", "graph_config.resolved.json")
    }
    graph_hashes = {
        relative: document.get("graph_config_sha256")
        for relative, document in graph_hash_documents.items()
        if isinstance(document, Mapping)
    }
    resolved_graph = base_documents.get("graph_config.resolved.json")
    recomputed_graph_hash = (
        compute_graph_config_sha256(dict(resolved_graph))
        if isinstance(resolved_graph, Mapping) else None
    )
    graph_hash_valid = (
        len(graph_hashes) == 3
        and all(_valid_sha256(value) for value in graph_hashes.values())
        and len(set(graph_hashes.values())) == 1
        and recomputed_graph_hash == next(iter(graph_hashes.values()), None)
    )
    if not graph_hash_valid:
        schema_errors.append(
            "graph_config_sha256 must be a matching 64-character hash in manifest, "
            "config.resolved, and graph_config.resolved, and must equal the "
            "recomputed resolved Graph identity"
        )

    manifest_document = base_documents.get("manifest.json")
    run_status_document = base_documents.get("run_status.json")
    base_provenance_documents = {
        "manifest.json": manifest_document,
        "config.resolved.json": config,
        "run_status.json": run_status_document,
    }

    def require_consistent_provenance(
        field: str, document_names: tuple[str, ...],
    ) -> None:
        values: list[tuple[str, Any]] = []
        for document_name in document_names:
            document = base_provenance_documents.get(document_name)
            if not isinstance(document, Mapping) or field not in document:
                lineage_errors.append(
                    f"{document_name} is missing provenance field {field}"
                )
            else:
                values.append((document_name, document[field]))
        if len(values) == len(document_names) and any(
            value != values[0][1] for _, value in values[1:]
        ):
            lineage_errors.append(
                f"{field} disagrees across {', '.join(document_names)}"
            )

    for provenance_field in (
        "experiment_id",
        "run_id",
        "memory_lineage_id",
        "memory_lifecycle",
        "memory_resumed_from_run_id",
    ):
        require_consistent_provenance(
            provenance_field,
            ("manifest.json", "config.resolved.json", "run_status.json"),
        )
    for provenance_field in (
        "backtest_protocol_sha256",
        "implementation_identity",
        "market_input_identity",
    ):
        require_consistent_provenance(
            provenance_field, ("manifest.json", "config.resolved.json")
        )

    expected_experiment_id = config.get("experiment_id")
    expected_run_id = config.get("run_id")
    expected_lineage_id = config.get("memory_lineage_id")
    expected_memory_lifecycle = config.get("memory_lifecycle")
    expected_resumed_from = config.get("memory_resumed_from_run_id")
    memory_archive_errors: list[str] = []
    memory_archive_checksum_errors: list[str] = []
    memory_archive_missing_files: list[str] = []
    if not isinstance(expected_experiment_id, str) or not expected_experiment_id:
        lineage_errors.append("experiment_id must be a non-empty string")
    if not isinstance(expected_run_id, str) or not expected_run_id:
        lineage_errors.append("run_id must be a non-empty string")
    if not isinstance(expected_lineage_id, str) or not expected_lineage_id:
        lineage_errors.append("memory_lineage_id must be a non-empty string")
    if expected_memory_lifecycle not in {
        "independent_fresh", "force_fresh", "resume",
    }:
        lineage_errors.append(
            f"invalid memory_lifecycle {expected_memory_lifecycle!r}"
        )
    elif expected_memory_lifecycle in {"independent_fresh", "force_fresh"}:
        if expected_resumed_from is not None:
            lineage_errors.append(
                "fresh memory lifecycle cannot have memory_resumed_from_run_id"
            )
        if expected_lineage_id != expected_run_id:
            lineage_errors.append(
                "fresh memory_lineage_id must equal the current run_id"
            )
    elif not isinstance(expected_resumed_from, str) or not expected_resumed_from:
        lineage_errors.append(
            "resume lifecycle requires a non-empty memory_resumed_from_run_id"
        )
    expected_protocol_hash = config.get("backtest_protocol_sha256")
    if not _valid_sha256(expected_protocol_hash):
        lineage_errors.append("backtest_protocol_sha256 must be a valid SHA-256")
    implementation_identity = config.get("implementation_identity")
    if not isinstance(implementation_identity, str) or not implementation_identity:
        lineage_errors.append("implementation_identity must be a non-empty string")
    market_input_identity = config.get("market_input_identity")
    if market_input_identity is not None and (
        not isinstance(market_input_identity, Mapping)
        or not market_input_identity
        or any(
            not isinstance(symbol, str) or not _valid_sha256(digest)
            for symbol, digest in market_input_identity.items()
        )
    ):
        lineage_errors.append(
            "market_input_identity must be null or a non-empty symbol/SHA-256 mapping"
        )
    graph_lineage_id = (
        resolved_graph.get("historical_memory_lineage_id")
        if isinstance(resolved_graph, Mapping) else None
    )
    if graph_lineage_id != expected_lineage_id:
        lineage_errors.append(
            "graph_config.resolved historical_memory_lineage_id does not match the bundle"
        )

    if require_memory_archive:
        memory_symbols = symbol_list
        expected_memory_cases = config.get("actual_decision_count")
        if (
            isinstance(expected_memory_cases, int)
            and not isinstance(expected_memory_cases, bool)
            and schedule_frame is not None
            and "decision_session" in schedule_frame
        ):
            remaining_memory_cases = max(expected_memory_cases, 0)
            sessions_per_symbol = len(schedule_frame)
            memory_symbols = []
            for symbol in symbol_list:
                symbol_cases = min(sessions_per_symbol, remaining_memory_cases)
                if symbol_cases > 0:
                    memory_symbols.append(symbol)
                remaining_memory_cases -= symbol_cases
        try:
            memory_enabled, memory_scope, memory_identity = (
                graph_memory_evidence_identity(
                    resolved_graph if isinstance(resolved_graph, Mapping) else {}
                )
            )
        except ValueError as exc:
            memory_archive_errors.append(
                f"resolved graph Memory evidence identity is invalid: {exc}"
            )
            memory_enabled, memory_scope, memory_identity = False, None, None
        memory_validation = validate_final_memory_archive(
            run_dir=root,
            descriptor=(
                manifest_document.get("memory_archive")
                if isinstance(manifest_document, Mapping) else None
            ),
            experiment_id=expected_experiment_id,
            run_id=expected_run_id,
            memory_lineage_id=expected_lineage_id,
            memory_lifecycle=expected_memory_lifecycle,
            memory_resumed_from_run_id=expected_resumed_from,
            graph_config_sha256=next(iter(graph_hashes.values()), None),
            symbols=memory_symbols,
            finmultitime_evidence_enabled=memory_enabled,
            finmultitime_bundle_scope=memory_scope,
            finmultitime_bundle_identity=memory_identity,
        )
        memory_archive_errors.extend(memory_validation["errors"])
        memory_archive_checksum_errors.extend(
            memory_validation["checksum_errors"]
        )
        memory_archive_missing_files.extend(memory_validation["missing_files"])
        for relative in memory_archive_missing_files:
            if relative not in missing_files:
                missing_files.append(relative)

    if strategy == "tradingagents":
        case_index = analysis_frames.get("case_index.csv")
        decision_timeline = analysis_frames.get("decision_timeline.csv")
        availability = analysis_frames.get("data_availability.csv")
        usage = analysis_frames.get("llm_usage.csv")
        expected_cases = config.get("actual_decision_count")
        expected_graph_hash = next(iter(graph_hashes.values()), None) if graph_hash_valid else None
        observed_case_keys: list[tuple[str, str]] = []
        observed_case_ids: list[str] = []
        expected_audit_counts: dict[tuple[str, str], int] = {}
        expected_usage_counts: dict[str, int] = {}
        case_cache_statuses: dict[str, str] = {}
        valid_cache_statuses = {"hit", "miss", "bypass"}
        timeline_case_keys: list[tuple[str, str]] = []
        timeline_cache_statuses: dict[tuple[str, str], str] = {}
        timeline_decision_statuses: dict[tuple[str, str], str] = {}
        if decision_timeline is None:
            agent_artifact_errors.append("decision_timeline.csv is not readable")
        else:
            for _, timeline_row in decision_timeline.iterrows():
                timeline_key = (
                    str(timeline_row.get("symbol", "")),
                    str(timeline_row.get("decision_session", "")),
                )
                timeline_status = str(timeline_row.get("cache_status", ""))
                timeline_case_keys.append(timeline_key)
                timeline_cache_statuses[timeline_key] = timeline_status
                timeline_decision_statuses[timeline_key] = str(
                    timeline_row.get("status", "")
                )
                if timeline_status not in valid_cache_statuses:
                    agent_artifact_errors.append(
                        f"{timeline_key}: decision_timeline has invalid cache_status "
                        f"{timeline_status!r}"
                    )
            if len(set(timeline_case_keys)) != len(timeline_case_keys):
                agent_artifact_errors.append(
                    "decision_timeline contains duplicate symbol/session cases"
                )
        if case_index is None:
            agent_artifact_errors.append("case_index.csv is not readable")
        else:
            if (
                isinstance(expected_cases, int)
                and not isinstance(expected_cases, bool)
                and len(case_index) != expected_cases
            ):
                agent_artifact_errors.append(
                    f"case_index.csv has {len(case_index)} rows; expected {expected_cases}"
                )
            for _, case in case_index.iterrows():
                symbol = str(case.get("symbol", ""))
                session = str(case.get("decision_session", ""))
                case_id = str(case.get("case_id", ""))
                case_key = (symbol, session)
                case_decision_status = str(case.get("decision_status", ""))
                case_cache_status = str(case.get("cache_status", ""))
                observed_case_keys.append((symbol, session))
                observed_case_ids.append(case_id)
                case_cache_statuses[case_id] = case_cache_status
                if case_cache_status not in valid_cache_statuses:
                    agent_artifact_errors.append(
                        f"{case_id}: case_index has invalid cache_status "
                        f"{case_cache_status!r}"
                    )
                if (
                    expected_memory_lifecycle == "independent_fresh"
                    and case_cache_status != "miss"
                ):
                    lineage_errors.append(
                        f"{case_id}: independent_fresh case must be a cache miss"
                    )
                elif (
                    expected_memory_lifecycle == "force_fresh"
                    and case_cache_status != "bypass"
                ):
                    lineage_errors.append(
                        f"{case_id}: force_fresh case must bypass cache"
                    )
                elif (
                    expected_memory_lifecycle == "resume"
                    and case_cache_status not in {"hit", "miss"}
                ):
                    lineage_errors.append(
                        f"{case_id}: resume case cache_status must be hit or miss"
                    )
                if (
                    (case_cache_status == "hit")
                    != (case_decision_status == "cached")
                ):
                    agent_artifact_errors.append(
                        f"{case_id}: cache hit must correspond exactly to cached decision status"
                    )
                if timeline_cache_statuses.get(case_key) != case_cache_status:
                    agent_artifact_errors.append(
                        f"{case_id}: decision_timeline cache_status does not match case_index"
                    )
                if timeline_decision_statuses.get(case_key) != case_decision_status:
                    agent_artifact_errors.append(
                        f"{case_id}: decision_timeline status does not match case_index"
                    )
                case_root = root / "strategy" / "cases" / symbol / session
                case_documents: dict[str, Any] = {}
                for filename in (
                    "decision.json",
                    "model_config.json",
                    "run_context.json",
                    "cache_identity.json",
                    "source_audit.json",
                    "llm_usage.json",
                    "case_metadata.json",
                ):
                    case_file = case_root / filename
                    if not case_file.is_file():
                        agent_artifact_errors.append(
                            f"missing canonical case artifact: {case_file.relative_to(root)}"
                        )
                    else:
                        case_documents[filename] = _read_json_document(
                            case_file, invalid_json,
                        )

                case_run_context = case_documents.get("run_context.json")
                if not isinstance(case_run_context, Mapping):
                    lineage_errors.append(
                        f"{case_id}: run_context.json must contain lineage provenance"
                    )
                else:
                    if case_run_context.get("memory_lineage_id") != expected_lineage_id:
                        lineage_errors.append(
                            f"{case_id}: run_context memory_lineage_id does not match bundle"
                        )
                    if case_run_context.get("experiment_id") != expected_experiment_id:
                        lineage_errors.append(
                            f"{case_id}: run_context experiment_id does not match bundle"
                        )

                if require_success and str(case.get("decision_status")) not in {
                    "success", "cached",
                }:
                    agent_artifact_errors.append(
                        f"{case_id}: successful Agent bundle has an unsuccessful decision"
                    )
                if require_success and str(case.get("action")) not in {"BUY", "HOLD", "SELL"}:
                    agent_artifact_errors.append(
                        f"{case_id}: successful Agent bundle has no valid action"
                    )
                if expected_graph_hash is not None and case.get(
                    "graph_config_sha256"
                ) != expected_graph_hash:
                    agent_artifact_errors.append(
                        f"{case_id}: case_index graph_config_sha256 does not match the bundle"
                    )
                if expected_experiment_id is not None and case.get(
                    "experiment_id"
                ) != expected_experiment_id:
                    agent_artifact_errors.append(
                        f"{case_id}: case_index experiment_id does not match config.resolved"
                    )
                if expected_run_id is not None and case.get("run_id") != expected_run_id:
                    agent_artifact_errors.append(
                        f"{case_id}: case_index run_id does not match config.resolved"
                    )

                graph_field_map = {
                    "provider": "llm_provider",
                    "quick_model": "quick_think_llm",
                    "deep_model": "deep_think_llm",
                    "thinking_mode": "deepseek_thinking",
                    "temperature": "temperature",
                    "research_depth": "research_depth",
                    "debate_rounds": "max_debate_rounds",
                    "risk_rounds": "max_risk_discuss_rounds",
                }
                if isinstance(resolved_graph, Mapping):
                    for case_field, graph_field in graph_field_map.items():
                        actual = case.get(case_field)
                        expected = resolved_graph.get(graph_field)
                        if pd.isna(actual) or actual != expected:
                            agent_artifact_errors.append(
                                f"{case_id}: {case_field} does not match resolved Graph config"
                            )

                model_config = case_documents.get("model_config.json")
                if not isinstance(model_config, Mapping):
                    agent_artifact_errors.append(f"{case_id}: model_config.json must be an object")
                elif model_config.get("graph_config_sha256") != expected_graph_hash:
                    agent_artifact_errors.append(
                        f"{case_id}: model_config graph_config_sha256 does not match the bundle"
                    )

                identity_document = case_documents.get("cache_identity.json")
                identity: Any = None
                identity_key: Any = None
                if isinstance(identity_document, Mapping):
                    identity = identity_document.get("identity")
                    identity_key = identity_document.get("cache_key")
                if not isinstance(identity, Mapping) or not _valid_sha256(identity_key):
                    agent_artifact_errors.append(
                        f"{case_id}: cache_identity.json is missing a valid identity/key"
                    )
                else:
                    if identity.get("graph_config_sha256") != expected_graph_hash:
                        agent_artifact_errors.append(
                            f"{case_id}: cache identity graph hash does not match the bundle"
                        )
                    if cache_key(dict(identity)) != identity_key:
                        agent_artifact_errors.append(
                            f"{case_id}: cache key does not match its canonical identity"
                        )
                    if case.get("cache_key") != identity_key:
                        agent_artifact_errors.append(
                            f"{case_id}: case_index cache_key does not match cache identity"
                        )
                if not isinstance(identity, Mapping):
                    lineage_errors.append(
                        f"{case_id}: cache identity must contain lineage provenance"
                    )
                else:
                    if identity.get("memory_lineage_id") != expected_lineage_id:
                        lineage_errors.append(
                            f"{case_id}: cache identity memory_lineage_id does not match bundle"
                        )
                    if identity.get("experiment_id") != expected_experiment_id:
                        lineage_errors.append(
                            f"{case_id}: cache identity experiment_id does not match bundle"
                        )

                case_metadata = case_documents.get("case_metadata.json")
                if not isinstance(case_metadata, Mapping):
                    agent_artifact_errors.append(f"{case_id}: case_metadata.json must be an object")
                else:
                    if case_metadata.get("case_id") != case_id:
                        agent_artifact_errors.append(
                            f"{case_id}: case_metadata case_id does not match case_index"
                        )
                    if case_metadata.get("cache_key") != case.get("cache_key"):
                        agent_artifact_errors.append(
                            f"{case_id}: case_metadata cache_key does not match case_index"
                        )
                    if case_metadata.get("cache_status") != case_cache_status:
                        agent_artifact_errors.append(
                            f"{case_id}: case_metadata cache_status does not match case_index"
                        )

                decision_document = case_documents.get("decision.json")
                if not isinstance(decision_document, Mapping):
                    agent_artifact_errors.append(f"{case_id}: decision.json must be an object")
                else:
                    if decision_document.get("status") != case_decision_status:
                        agent_artifact_errors.append(
                            f"{case_id}: decision.json status does not match case_index"
                        )
                    metadata = decision_document.get("metadata")
                    if not isinstance(metadata, Mapping):
                        agent_artifact_errors.append(
                            f"{case_id}: decision metadata must be an object"
                        )
                    elif (
                        not isinstance(metadata.get("model_config"), Mapping)
                        or metadata["model_config"].get("graph_config_sha256")
                        != expected_graph_hash
                    ):
                        agent_artifact_errors.append(
                            f"{case_id}: decision metadata graph hash does not match the bundle"
                        )
                    if (
                        isinstance(metadata, Mapping)
                        and metadata.get("cache_status") != case_cache_status
                    ):
                        agent_artifact_errors.append(
                            f"{case_id}: decision metadata cache_status does not match case_index"
                        )

                source_audit = case_documents.get("source_audit.json")
                sources = source_audit.get("sources") if isinstance(source_audit, Mapping) else None
                source_run_context = (
                    source_audit.get("run_context")
                    if isinstance(source_audit, Mapping) else None
                )
                if not isinstance(source_run_context, Mapping):
                    lineage_errors.append(
                        f"{case_id}: source_audit run_context must contain lineage provenance"
                    )
                else:
                    if source_run_context.get("memory_lineage_id") != expected_lineage_id:
                        lineage_errors.append(
                            f"{case_id}: source_audit memory_lineage_id does not match bundle"
                        )
                    if source_run_context.get("experiment_id") != expected_experiment_id:
                        lineage_errors.append(
                            f"{case_id}: source_audit experiment_id does not match bundle"
                        )
                if not isinstance(sources, list) or not sources:
                    agent_artifact_errors.append(
                        f"{case_id}: source_audit.json must contain source records"
                    )
                else:
                    expected_audit_counts[(symbol, session)] = len(sources)

                current_usage = case_documents.get("llm_usage.json")
                if not isinstance(current_usage, list):
                    agent_artifact_errors.append(f"{case_id}: llm_usage.json must be a list")
                    current_usage = []
                origin_usage: Any = []
                if not current_usage and str(case.get("cache_status")) == "hit":
                    origin_path = case_root / "cached_origin_llm_usage.json"
                    if origin_path.is_file():
                        origin_usage = _read_json_document(origin_path, invalid_json)
                    if not isinstance(origin_usage, list) or not origin_usage:
                        agent_artifact_errors.append(
                            f"{case_id}: cache hit lacks preserved origin LLM usage"
                        )
                        origin_usage = []
                expected_usage_counts[case_id] = len(current_usage or origin_usage)

                expected_report_relative = (
                    f"strategy/cases/{symbol}/{session}/reports"
                )
                expected_audit_relative = (
                    f"strategy/cases/{symbol}/{session}/source_audit.json"
                )
                report_member = _bundle_member(root, case.get("report_path"))
                if (
                    case.get("report_path") != expected_report_relative
                    or report_member is None
                    or not report_member.is_dir()
                ):
                    agent_artifact_errors.append(
                        f"{case_id}: report_path is not the canonical case report directory"
                    )
                elif not (report_member / "complete_report.md").is_file():
                    agent_artifact_errors.append(
                        f"{case_id}: report tree lacks complete_report.md"
                    )
                audit_member = _bundle_member(root, case.get("source_audit_path"))
                if (
                    case.get("source_audit_path") != expected_audit_relative
                    or audit_member is None
                    or not audit_member.is_file()
                ):
                    agent_artifact_errors.append(
                        f"{case_id}: source_audit_path is not the canonical case audit"
                    )

            if len(set(observed_case_keys)) != len(observed_case_keys):
                agent_artifact_errors.append("case_index contains duplicate symbol/session cases")
            if len(set(observed_case_ids)) != len(observed_case_ids):
                agent_artifact_errors.append("case_index contains duplicate case_id values")
            if set(timeline_case_keys) != set(observed_case_keys):
                agent_artifact_errors.append(
                    "decision_timeline case coverage does not match case_index"
                )
            if (
                schedule_frame is not None
                and isinstance(expected_cases, int)
                and not isinstance(expected_cases, bool)
            ):
                expected_keys: list[tuple[str, str]] = []
                remaining = expected_cases
                sessions = schedule_frame["decision_session"].astype(str).tolist()
                for expected_symbol in symbol_list:
                    take = min(len(sessions), max(remaining, 0))
                    expected_keys.extend(
                        (expected_symbol, expected_session)
                        for expected_session in sessions[:take]
                    )
                    remaining -= take
                if set(observed_case_keys) != set(expected_keys):
                    agent_artifact_errors.append(
                        "case_index does not cover the expected symbol/session schedule"
                    )
        if isinstance(expected_cases, int) and expected_cases > 0:
            if availability is None or availability.empty:
                agent_artifact_errors.append("data_availability.csv must contain Agent rows")
            else:
                availability_counts = availability.groupby(
                    ["symbol", "decision_session"], dropna=False,
                ).size().to_dict()
                if set(availability_counts) != set(expected_audit_counts):
                    agent_artifact_errors.append(
                        "data_availability case coverage does not match case_index"
                    )
                for case_key, expected_count in expected_audit_counts.items():
                    if availability_counts.get(case_key, 0) != expected_count:
                        agent_artifact_errors.append(
                            f"{case_key}: data_availability rows do not match source audit"
                        )
                valid_statuses = {"used", "blocked", "unavailable", "error"}
                invalid_statuses = set(availability["status"].dropna()) - valid_statuses
                if invalid_statuses:
                    agent_artifact_errors.append(
                        f"data_availability has invalid statuses: {sorted(invalid_statuses)}"
                    )
                if expected_experiment_id is not None and not availability[
                    "experiment_id"
                ].eq(expected_experiment_id).all():
                    agent_artifact_errors.append(
                        "data_availability experiment_id does not match config.resolved"
                    )
                if expected_run_id is not None and not availability["run_id"].eq(
                    expected_run_id
                ).all():
                    agent_artifact_errors.append(
                        "data_availability run_id does not match config.resolved"
                    )
            if usage is None or usage.empty:
                agent_artifact_errors.append("llm_usage.csv must contain provider usage rows")
            else:
                usage_counts = usage.groupby("case_id", dropna=False).size().to_dict()
                if set(usage_counts) != set(expected_usage_counts):
                    agent_artifact_errors.append(
                        "llm_usage case coverage does not match case_index"
                    )
                for expected_case_id, expected_count in expected_usage_counts.items():
                    if expected_count <= 0 or usage_counts.get(expected_case_id, 0) != expected_count:
                        agent_artifact_errors.append(
                            f"{expected_case_id}: llm_usage rows do not match canonical usage"
                        )
                invalid_sources = set(usage["usage_source"].dropna()) - {
                    "live_request", "cache_origin",
                }
                if invalid_sources:
                    agent_artifact_errors.append(
                        f"llm_usage has invalid usage_source values: {sorted(invalid_sources)}"
                    )
                for _, usage_row in usage.iterrows():
                    origin_run_id = usage_row.get("origin_run_id")
                    if pd.isna(origin_run_id) or not str(origin_run_id).strip():
                        agent_artifact_errors.append(
                            "llm_usage row lacks origin_run_id provenance"
                        )
                        break
                    if expected_experiment_id is not None and usage_row.get(
                        "experiment_id"
                    ) != expected_experiment_id:
                        agent_artifact_errors.append(
                            "llm_usage experiment_id does not match config.resolved"
                        )
                    if expected_run_id is not None and usage_row.get("run_id") != expected_run_id:
                        agent_artifact_errors.append(
                            "llm_usage run_id does not match config.resolved"
                        )
                    if isinstance(resolved_graph, Mapping):
                        if usage_row.get("provider") != resolved_graph.get("llm_provider"):
                            agent_artifact_errors.append(
                                "llm_usage provider does not match resolved Graph config"
                            )
                        if usage_row.get("thinking_mode") != resolved_graph.get(
                            "deepseek_thinking"
                        ):
                            agent_artifact_errors.append(
                                "llm_usage thinking_mode does not match resolved Graph config"
                            )
                        if usage_row.get("model") not in {
                            resolved_graph.get("quick_think_llm"),
                            resolved_graph.get("deep_think_llm"),
                        }:
                            agent_artifact_errors.append(
                                "llm_usage model is not a resolved quick/deep model"
                            )
                    usage_source = usage_row.get("usage_source")
                    usage_case_id = str(usage_row.get("case_id", ""))
                    if (
                        usage_source == "live_request"
                        and expected_run_id is not None
                        and origin_run_id != expected_run_id
                    ):
                        agent_artifact_errors.append(
                            "live_request usage origin_run_id must equal the current run"
                        )
                    if (
                        usage_source == "cache_origin"
                        and case_cache_statuses.get(usage_case_id) != "hit"
                    ):
                        agent_artifact_errors.append(
                            "cache_origin usage must belong to a cache-hit case"
                        )
                    latency = usage_row.get("latency_seconds")
                    if pd.notna(latency) and (
                        not isinstance(latency, (int, float, np.integer, np.floating))
                        or latency < 0
                    ):
                        agent_artifact_errors.append(
                            "llm_usage has invalid latency_seconds value"
                        )
                    for token_column in (
                        "prompt_tokens",
                        "prompt_cache_hit_tokens",
                        "prompt_cache_miss_tokens",
                        "completion_tokens",
                        "reasoning_tokens",
                        "total_tokens",
                    ):
                        value = usage_row.get(token_column)
                        if pd.notna(value) and (
                            not isinstance(value, (int, float, np.integer, np.floating))
                            or value < 0
                        ):
                            agent_artifact_errors.append(
                                f"llm_usage has invalid {token_column} value"
                            )
                            break

    validation_status_valid = True
    if require_validation_report:
        validation_document = _read_json_document(
            root / "validation" / "validation_report.json", invalid_json,
        )
        validation_status_valid = (
            isinstance(validation_document, Mapping)
            and validation_document.get("status") == "passed"
        )

    run_status_document = base_documents.get("run_status.json")
    observed_status = (
        run_status_document.get("status") if isinstance(run_status_document, Mapping) else None
    )
    run_status_valid = (
        observed_status == "success"
        if require_success
        else observed_status in {"running", "success"}
    )

    analysis_schema_errors = [
        error for error in schema_errors if error.startswith("analysis_ready/")
    ]
    metadata_schema_errors = [
        error for error in schema_errors if not error.startswith("analysis_ready/")
    ]
    checks = {
        "required_files_exist": not missing_files,
        "json_documents_valid": not invalid_json,
        "artifact_metadata_valid": not metadata_schema_errors,
        "analysis_ready_schemas_valid": not analysis_schema_errors,
        "market_snapshot_sha256_valid": not checksum_errors,
        "graph_config_sha256_present_and_consistent": graph_hash_valid,
        "memory_lineage_provenance_consistent": not lineage_errors,
        "final_memory_archive_complete_and_valid": not (
            memory_archive_errors
            or memory_archive_checksum_errors
            or memory_archive_missing_files
        ),
        "agent_case_artifacts_complete": not agent_artifact_errors,
        "validation_report_passed": validation_status_valid,
        "run_status_valid": run_status_valid,
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "checks": checks,
        "missing_files": missing_files,
        "invalid_json": invalid_json,
        "schema_errors": schema_errors,
        "checksum_errors": checksum_errors,
        "lineage_errors": lineage_errors,
        "memory_archive_errors": memory_archive_errors,
        "memory_archive_checksum_errors": memory_archive_checksum_errors,
        "agent_artifact_errors": agent_artifact_errors,
    }
