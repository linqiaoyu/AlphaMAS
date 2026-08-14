"""Freeze the M2-04 pre-Formal reward-market snapshot.

This Phase-A program deliberately contains no reward formulas.  It resolves the
frozen MAXIMUM TRAIN/VALIDATION case population, downloads raw yfinance market
bars through the same provider/normalisation path as M0/M1, retains only the
sessions needed by those cases, and writes a content-addressed archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.data import YFinanceDataProvider

ALLOWED_ROLES = frozenset({"TRAIN", "VALIDATION"})
FORBIDDEN_ROLES = frozenset({"FINAL_HOLDOUT", "E2E_PILOT", "FORMAL_2024H1"})
EXPECTED_ROLE_COUNTS = {"TRAIN": 56, "VALIDATION": 16}
EXPECTED_SYMBOLS = ("AAPL", "AEMD", "AGI", "AMZN", "ARR", "EML", "JBSS", "JPM")
LATEST_ALLOWED_SESSION = "2023-07-14"
SNAPSHOT_COLUMNS = ("Date", "Open", "Close", "Dividends", "Stock Splits")
STARTING_SHA = "06b30f05749c53c8122d3997595aad556cbf0136"
SPLIT_CONTRACT_SHA = "8574067d039f6288042ec85cebc4517f3c3da49c"
REWARD_FRAMEWORK_SHA = STARTING_SHA


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_json(path: Path, payload: Any) -> str:
    data = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def load_allowed_cases(case_plan: Path) -> list[dict[str, str]]:
    with case_plan.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    required = {"case_id", "symbol", "decision_session", "maturity_session", "split_role", "maximum_included"}
    if not all_rows or not required.issubset(all_rows[0]):
        raise ValueError(f"case plan lacks columns: {sorted(required)}")
    rows = [
        row for row in all_rows
        if row["split_role"] in ALLOWED_ROLES and row["maximum_included"].lower() == "true"
    ]
    counts = {role: sum(row["split_role"] == role for row in rows) for role in ALLOWED_ROLES}
    if counts != EXPECTED_ROLE_COUNTS or len(rows) != 72:
        raise ValueError(f"unexpected allowed population: {counts}, total={len(rows)}")
    if len({row["case_id"] for row in rows}) != 72:
        raise ValueError("allowed case IDs must be unique")
    symbols = tuple(sorted({row["symbol"] for row in rows}))
    if symbols != EXPECTED_SYMBOLS:
        raise ValueError(f"unexpected symbols: {symbols}")
    if any(row["split_role"] in FORBIDDEN_ROLES for row in rows):
        raise ValueError("protected role entered allowed case population")
    return sorted(rows, key=lambda row: (row["symbol"], row["decision_session"]))


def required_sessions_by_symbol(
    cases: list[dict[str, str]], schedule: ExchangeSchedule,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {symbol: set() for symbol in EXPECTED_SYMBOLS}
    for row in cases:
        decision = row["decision_session"]
        if not schedule.calendar.is_session(decision):
            raise ValueError(f"decision is not XNYS session: {row['case_id']}")
        window = schedule.calendar.sessions_window(decision, 6).tz_localize(None)
        labels = tuple(item.date().isoformat() for item in window)
        if labels[-1] != row["maturity_session"]:
            raise ValueError(f"case maturity does not equal fifth subsequent session: {row['case_id']}")
        result[row["symbol"]].update(labels)
    frozen = {symbol: tuple(sorted(labels)) for symbol, labels in result.items()}
    if max(label for labels in frozen.values() for label in labels) != LATEST_ALLOWED_SESSION:
        raise ValueError("latest required session is not the frozen 2023-07-14 boundary")
    return frozen


def state_templates() -> dict[str, Any]:
    return {
        "normalised_decision_time_equity_usd": 100000.0,
        "CASH": {
            "cash": 100000.0,
            "quantity": 0.0,
            "average_entry_price": 0.0,
            "open_position_commission": 0.0,
        },
        "LONG": {
            "cash": 0.0,
            "quantity": "100000 / frozen raw decision-session Close",
            "average_entry_price": "frozen raw decision-session Close",
            "open_position_commission": 0.0,
        },
        "execution_attribution": "E0 recomputed after next-session corporate actions at execution open",
    }


def long_template(decision_close: float) -> dict[str, float]:
    close = float(decision_close)
    if not pd.notna(close) or close <= 0:
        raise ValueError("decision close must be finite and positive")
    return {
        "cash": 0.0,
        "quantity": 100000.0 / close,
        "average_entry_price": close,
        "open_position_commission": 0.0,
    }


def _canonical_market_bytes(frame: pd.DataFrame) -> bytes:
    output = frame.loc[:, SNAPSHOT_COLUMNS].copy()
    output["Date"] = pd.to_datetime(output["Date"]).dt.strftime("%Y-%m-%d")
    return output.to_csv(index=False, lineterminator="\n", float_format="%.10g").encode()


def validate_and_select(
    frame: pd.DataFrame, required_sessions: tuple[str, ...], symbol: str,
) -> pd.DataFrame:
    data = frame.reset_index(names="Date")
    data["Date"] = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d")
    selected = data[data["Date"].isin(required_sessions)].copy()
    if selected["Date"].duplicated().any():
        raise ValueError(f"duplicate market session for {symbol}")
    actual = tuple(sorted(selected["Date"]))
    if actual != required_sessions:
        missing = sorted(set(required_sessions).difference(actual))
        raise ValueError(f"missing required sessions for {symbol}: {missing}")
    selected = selected.sort_values("Date").loc[:, SNAPSHOT_COLUMNS]
    numeric = selected.loc[:, SNAPSHOT_COLUMNS[1:]].apply(pd.to_numeric, errors="raise")
    if not numeric.map(math.isfinite).all().all():
        raise ValueError(f"non-finite required value for {symbol}")
    if (numeric[["Open", "Close"]] <= 0).any().any():
        raise ValueError(f"non-positive Open/Close for {symbol}")
    if (numeric[["Dividends", "Stock Splits"]] < 0).any().any():
        raise ValueError(f"invalid corporate action for {symbol}")
    if selected["Date"].max() > LATEST_ALLOWED_SESSION or selected["Date"].str.startswith("2024").any():
        raise ValueError(f"protected future row for {symbol}")
    return selected


def snapshot_identity_payload(
    cases: list[dict[str, str]], file_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "record_type": "M2_REWARD_MARKET_SNAPSHOT_IDENTITY_V1",
        "case_population": [
            {key: row[key] for key in ("case_id", "symbol", "decision_session", "maturity_session", "split_role")}
            for row in cases
        ],
        "files": [
            {key: record[key] for key in ("symbol", "path", "sha256", "row_count", "first_session", "last_session")}
            for record in file_records
        ],
        "provider": {
            "family": "yfinance",
            "version": metadata.version("yfinance"),
            "query": {"auto_adjust": False, "actions": True, "progress": False, "end_exclusive_plus_one_day": True},
        },
        "normalisation": {
            "path": "tradingagents.backtesting.data.YFinanceDataProvider.load -> normalize_ohlcv",
            "timezone": "timezone removed and session label normalized",
            "prices": "raw unadjusted Open/Close",
            "corporate_actions": "explicit Dividends and Stock Splits",
            "serialization": "UTF-8 CSV, LF, %.10g",
        },
        "state_templates": state_templates(),
    }


def materialize(case_plan: Path, output_root: Path, source_script: Path) -> dict[str, Any]:
    cases = load_allowed_cases(case_plan)
    schedule = ExchangeSchedule()
    required = required_sessions_by_symbol(cases, schedule)
    market_dir = output_root / "inputs/market_snapshot"
    manifest_dir = output_root / "manifests"
    market_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    provider = YFinanceDataProvider()
    file_records: list[dict[str, Any]] = []
    queries: list[dict[str, str]] = []
    for symbol in EXPECTED_SYMBOLS:
        labels = required[symbol]
        frame = provider.load(symbol, labels[0], labels[-1])
        selected = validate_and_select(frame, labels, symbol)
        target = market_dir / f"{symbol}.csv"
        payload = _canonical_market_bytes(selected)
        target.write_bytes(payload)
        file_records.append({
            "symbol": symbol,
            "path": f"inputs/market_snapshot/{symbol}.csv",
            "sha256": sha256_bytes(payload),
            "row_count": len(selected),
            "first_session": selected["Date"].min(),
            "last_session": selected["Date"].max(),
            "required_field_completeness": True,
        })
        queries.append({"symbol": symbol, "requested_start": labels[0], "requested_end_inclusive": labels[-1]})
    identity_payload = snapshot_identity_payload(cases, file_records)
    identity = sha256_bytes(canonical_json_bytes(identity_payload))
    manifest = {
        "record_type": "M2_REWARD_MARKET_SNAPSHOT_V1",
        "formal_eligible": False,
        "development_only": True,
        "task_id": "M2-04",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_provider": "yfinance",
        "source_provider_version": metadata.version("yfinance"),
        "source_query_semantics": identity_payload["provider"]["query"],
        "normalisation_semantics": identity_payload["normalisation"],
        "alphamas_source_starting_sha": STARTING_SHA,
        "reward_contract_lineage_sha": REWARD_FRAMEWORK_SHA,
        "split_contract_sha": SPLIT_CONTRACT_SHA,
        "materializer_script_sha256": sha256_bytes(source_script.read_bytes()),
        "case_counts": {**EXPECTED_ROLE_COUNTS, "total": len(cases)},
        "case_ids": [row["case_id"] for row in cases],
        "symbols": list(EXPECTED_SYMBOLS),
        "symbol_count": len(EXPECTED_SYMBOLS),
        "latest_retained_session": max(record["last_session"] for record in file_records),
        "holdout_rows_included": False,
        "pilot_rows_included": False,
        "formal_2024_rows_included": False,
        "market_snapshot_is_agent_evidence": False,
        "state_templates": state_templates(),
        "queries": queries,
        "files": file_records,
        "identity_payload": identity_payload,
        "reward_market_snapshot_identity_sha256": identity,
    }
    manifest_path = manifest_dir / "market_snapshot_manifest.json"
    manifest_sha = write_json(manifest_path, manifest)
    readme = (
        "# M2 reward study v1\n\n"
        "PRE-FORMAL DEVELOPMENT — NOT FORMAL M2.\n\n"
        "The frozen market snapshot is used only to calculate matured TRAIN/VALIDATION "
        "reward labels. It is not agent evidence or a model input. Phase B must not fetch market data.\n"
    ).encode()
    (output_root / "README.md").write_bytes(readme)
    inventory = {
        "record_type": "M2_REWARD_MARKET_SNAPSHOT_PHASE_A_SHA256_V1",
        "files": {
            "README.md": sha256_bytes(readme),
            "manifests/market_snapshot_manifest.json": manifest_sha,
            **{record["path"]: record["sha256"] for record in file_records},
        },
        "reward_market_snapshot_identity_sha256": identity,
    }
    write_json(manifest_dir / "phase_a_sha256.json", inventory)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(args.case_plan, args.output_root, Path(__file__).resolve())
    print(manifest["reward_market_snapshot_identity_sha256"])


if __name__ == "__main__":
    main()
