"""Analysis-ready, versioned artifacts for weekly backtest runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.backtesting.metrics import compute_metrics
from tradingagents.backtesting.recorder import equal_weight_aggregate, write_json

ARTIFACT_SCHEMA_VERSION = "1.0"
RESULT_TABLES = {
    "decisions.csv": "decisions",
    "orders.csv": "orders",
    "fills.csv": "fills",
    "daily_equity.csv": "daily_equity",
    "corporate_action_events.csv": "corporate_action_events",
}


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
    total_cost = sum(float(result.metrics["total_transaction_cost"]) for result in results.values())
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
    )
    metrics.update({key: sum(float(result.metrics[key]) for result in results.values())
                    for key in sum_keys})
    for integer_key in set(sum_keys) - {"cumulative_dividends"}:
        metrics[integer_key] = int(metrics[integer_key])
    metrics.update({
        "decision_failure_rate": (
            metrics["decision_failure_count"] / metrics["decision_count"]
            if metrics["decision_count"] else 0.0
        ),
        "total_transaction_cost": total_cost,
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


def _decision_timeline(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, result in results.items():
        orders = result.orders.copy()
        fills = result.fills.copy()
        for _, decision in result.decisions.iterrows():
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
                "position_after": fill.get("position_after", ""),
                "cache_status": "hit" if decision["status"] == "cached" else "miss",
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
            if fills.empty:
                period_cost = 0.0
            else:
                dates = fills["execution_time"].astype(str).str[:10]
                period_cost = float(fills.loc[(dates > start) & (dates <= end), "commission"].sum())
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
            "fill_quantity": "signed shares bought (positive) or sold (negative)",
            "commission": "cash commission charged on the fill",
            "equity": "cash plus close-marked position market value",
            "normalized_equity": "equity divided by the series first equity",
            "drawdown": "signed decline from running peak (zero at a peak)",
            "cash_effect": "cash credited by a corporate-action event",
            "strategy_forward_return": "equity return from decision close to next evaluation close",
            "transaction_cost_in_period": "commissions on fills after the decision through period end",
            "average_exposure_in_period": "mean daily position market value/equity in the period",
            "metric": "stable metric name",
            "value": "numeric metric value or empty when undefined",
        },
        "units": {
            "prices_cash_equity_commission": "account currency",
            "quantity": "shares (fractional when configured)",
            "rates_returns_weights_drawdowns": "decimal fraction",
            "slippage_bps": "basis points",
        },
    }
