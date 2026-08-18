#!/usr/bin/env python3
"""Structural 78-window ARMA audit that never fits a model or computes performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.backtesting.arma11 import (  # noqa: E402
    ARMA11_PRICE_COUNT,
    ARMA11_RETURN_CONVENTION,
    pit_safe_total_returns,
)
from tradingagents.backtesting.calendar import ExchangeSchedule  # noqa: E402
from tradingagents.backtesting.data import CSVSnapshotDataProvider  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "backtest_arma11_2024h1.json"),
    )
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    snapshot_dir = Path(args.snapshot_dir).resolve()
    expected_hashes = config["market_snapshot_sha256"]
    observed_hashes = {
        symbol: sha256(snapshot_dir / f"{symbol}.csv")
        for symbol in [*config["symbols"], "SPY"]
    }
    if observed_hashes != expected_hashes:
        raise ValueError("frozen market snapshot identities do not match the ARMA contract")
    schedule = ExchangeSchedule(config["calendar"])
    events = schedule.weekly_events(
        config["first_calendar_week"], config["final_calendar_week"]
    )
    if len(events) != config["decision_weeks"]:
        raise ValueError("formal schedule does not match the frozen decision count")
    provider = CSVSnapshotDataProvider(snapshot_dir)
    rows: list[dict[str, object]] = []
    future_violations = 0
    insufficient = 0
    for symbol in config["symbols"]:
        history = provider.load(
            symbol,
            pd.Timestamp("1990-01-01").date().isoformat(),
            config["final_valuation_session"],
        )
        for event in events:
            visible = history.loc[:pd.Timestamp(event.decision_session)].copy()
            try:
                window = pit_safe_total_returns(
                    visible, decision_session=event.decision_session
                )
                max_input = window.returns.index.max().date().isoformat()
                future = max_input > event.decision_session
                count = len(window.returns)
                insufficient_case = count != config["input"]["return_observations"]
                row = {
                    "symbol": symbol,
                    "decision_session": event.decision_session,
                    "decision_time_utc": event.decision_close_utc.isoformat(),
                    "execution_session": event.execution_session,
                    "price_start_session": window.price_start_session,
                    "input_start_session": window.input_start_session,
                    "input_end_session": window.input_end_session,
                    "max_model_input_session": max_input,
                    "n_price_observations": ARMA11_PRICE_COUNT,
                    "n_returns": count,
                    "input_sha256": window.sha256,
                    "future_input_violation": future,
                    "insufficient_window": insufficient_case,
                    "status": "passed" if not future and not insufficient_case else "failed",
                }
                future_violations += int(future)
                insufficient += int(insufficient_case)
            except ValueError as exc:
                insufficient += 1
                row = {
                    "symbol": symbol,
                    "decision_session": event.decision_session,
                    "decision_time_utc": event.decision_close_utc.isoformat(),
                    "execution_session": event.execution_session,
                    "status": "failed",
                    "future_input_violation": False,
                    "insufficient_window": True,
                    "error": str(exc),
                }
            rows.append(row)
    duplicates = len(rows) - len({(row["symbol"], row["decision_session"]) for row in rows})
    exact_windows = sum(
        row.get("n_returns") == config["input"]["return_observations"]
        for row in rows
    )
    report: dict[str, object] = {
        "audit_identity": "ARMA11_PREFORMAL_PIT_WINDOW_AUDIT_V1",
        "performance_computed": False,
        "model_fits_executed": 0,
        "return_convention": ARMA11_RETURN_CONVENTION,
        "corporate_action_leakage_control": (
            "Each return uses only Close_t, split_t, dividend_t, and Close_(t-1) from "
            "the frozen snapshot truncated at the decision session; no future-derived "
            "vendor adjustment factor is used."
        ),
        "config_sha256": sha256(config_path),
        "snapshot_sha256": observed_hashes,
        "case_count": len(rows),
        "exact_252_windows": int(exact_windows),
        "future_input_violations": future_violations,
        "insufficient_windows": insufficient,
        "duplicate_cases": duplicates,
        "status": (
            "passed"
            if len(rows) == 78
            and exact_windows == 78
            and future_violations == 0
            and insufficient == 0
            and duplicates == 0
            else "failed"
        ),
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False, lineterminator="\n")
    if report["status"] != "passed":
        raise RuntimeError("ARMA pre-Formal PIT-window audit failed")
    return report


def main() -> int:
    report = execute(parse_args())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
