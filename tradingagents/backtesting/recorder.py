"""Machine-readable backtest artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class BacktestRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def save(self, result: Any, manifest: dict[str, Any], schedule: list[dict[str, Any]]) -> Path:
        root = self.output_dir
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "manifest.json", manifest)
        pd.DataFrame(schedule).to_csv(root / "schedule.csv", index=False)
        tables = {
            "decisions.csv": result.decisions,
            "orders.csv": result.orders,
            "fills.csv": result.fills,
            "daily_equity.csv": result.daily_equity,
        }
        for filename, frame in tables.items():
            frame.to_csv(root / filename, index=False)
        write_json(root / "metrics.json", result.metrics)
        pd.DataFrame([result.metrics]).to_csv(root / "summary.csv", index=False)
        failures = result.decisions.loc[result.decisions.get("status") == "failed"]
        with (root / "failures.jsonl").open("w", encoding="utf-8") as handle:
            for item in failures.to_dict("records"):
                handle.write(json.dumps(item, default=str) + "\n")
        return root


def equal_weight_aggregate(results: dict[str, Any]) -> pd.DataFrame:
    if not results:
        raise ValueError("aggregate requires at least one result")
    normalized: list[pd.Series] = []
    expected_index = None
    for symbol, result in results.items():
        frame = result.daily_equity.set_index("session")
        if expected_index is None:
            expected_index = frame.index
        elif not frame.index.equals(expected_index):
            raise ValueError(f"incomplete session data for {symbol}")
        equity = frame["equity"].astype(float)
        normalized.append(equity / equity.iloc[0])
    aggregate = pd.concat(normalized, axis=1).mean(axis=1)
    return pd.DataFrame({"session": aggregate.index, "normalized_equity": aggregate.values})
