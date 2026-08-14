from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.m2.materialize_reward_market_snapshot import (
    EXPECTED_ROLE_COUNTS,
    EXPECTED_SYMBOLS,
    LATEST_ALLOWED_SESSION,
    canonical_json_bytes,
    load_allowed_cases,
    long_template,
    required_sessions_by_symbol,
    snapshot_identity_payload,
    state_templates,
    validate_and_select,
)
from scripts.m2.reward_simulator import (
    MarketBar,
    PortfolioState,
    RewardStatus,
    RewardWindow,
    simulate_counterfactuals,
)
from tradingagents.backtesting.calendar import ExchangeSchedule

ROOT = Path(__file__).resolve().parents[2]
CASE_PLAN = ROOT / "docs/m2/m2_preformal_semantic_case_plan.csv"


def test_exact_allowed_population_and_sessions() -> None:
    cases = load_allowed_cases(CASE_PLAN)
    assert len(cases) == 72
    assert {role: sum(row["split_role"] == role for row in cases) for role in EXPECTED_ROLE_COUNTS} == EXPECTED_ROLE_COUNTS
    assert tuple(sorted({row["symbol"] for row in cases})) == EXPECTED_SYMBOLS
    assert not {row["split_role"] for row in cases}.intersection({"FINAL_HOLDOUT", "E2E_PILOT", "FORMAL_2024H1"})
    sessions = required_sessions_by_symbol(cases, ExchangeSchedule())
    assert max(label for labels in sessions.values() for label in labels) == LATEST_ALLOWED_SESSION
    assert all(not label.startswith("2024") for labels in sessions.values() for label in labels)


def test_state_templates_are_exact_and_long_is_normalised() -> None:
    templates = state_templates()
    assert templates["CASH"] == {"cash": 100000.0, "quantity": 0.0, "average_entry_price": 0.0, "open_position_commission": 0.0}
    assert long_template(125.0) == {"cash": 0.0, "quantity": 800.0, "average_entry_price": 125.0, "open_position_commission": 0.0}
    assert long_template(125.0)["quantity"] * 125.0 == 100000.0
    with pytest.raises(ValueError):
        long_template(0)


def test_long_overnight_carry_and_corporate_actions_match_simulator() -> None:
    template = long_template(100.0)
    state = PortfolioState(**template)
    sessions = ("2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12", "2023-10-13")
    window = RewardWindow(
        symbol="SYNTH",
        decision_session="2023-10-06",
        decision_close_price=100.0,
        bars=tuple(
            MarketBar(
                session=session,
                open_price=40.0 if index == 0 else 41.0,
                close_price=41.0,
                dividend_per_share=1.0 if index == 0 else 0.0,
                split_ratio=2.0 if index == 0 else 0.0,
            )
            for index, session in enumerate(sessions)
        ),
    )
    first = simulate_counterfactuals(window, state)
    second = simulate_counterfactuals(window, state)
    assert first == second and first.status is RewardStatus.MATURED
    assert first.outcomes is not None
    # Dividend eligibility uses the carried 1,000 shares, then the 2:1 split
    # precedes E0 at the execution open: 1,000 cash + 2,000 * 40 = 81,000.
    assert first.outcomes["HOLD"].pre_execution_equity == 81000.0
    assert first.outcomes["HOLD"].pre_execution_equity != 100000.0


def test_market_selection_rejects_missing_and_future_rows() -> None:
    frame = pd.DataFrame({
        "Open": [10.0, 11.0], "High": [11.0, 12.0], "Low": [9.0, 10.0],
        "Close": [10.5, 11.5], "Volume": [1.0, 1.0], "Dividends": [0.0, 0.0],
        "Stock Splits": [0.0, 0.0],
    }, index=pd.to_datetime(["2023-07-13", "2023-07-14"]))
    selected = validate_and_select(frame, ("2023-07-13", "2023-07-14"), "TEST")
    assert list(selected.columns) == ["Date", "Open", "Close", "Dividends", "Stock Splits"]
    with pytest.raises(ValueError, match="missing required"):
        validate_and_select(frame, ("2023-07-12", "2023-07-13"), "TEST")
    future = frame.copy()
    future.index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    with pytest.raises(ValueError, match="protected future"):
        validate_and_select(future, ("2024-01-02", "2024-01-03"), "TEST")
    infinite = frame.copy()
    infinite.iloc[0, infinite.columns.get_loc("Open")] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_and_select(infinite, ("2023-07-13", "2023-07-14"), "TEST")


def test_identity_serialization_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.m2.materialize_reward_market_snapshot.metadata.version", lambda _: "1.5.2")
    cases = load_allowed_cases(CASE_PLAN)
    files = [{"symbol": symbol, "path": f"inputs/market_snapshot/{symbol}.csv", "sha256": "a" * 64, "row_count": 1, "first_session": "2023-05-05", "last_session": "2023-05-05"} for symbol in EXPECTED_SYMBOLS]
    payload = snapshot_identity_payload(cases, files)
    first = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    second = hashlib.sha256(canonical_json_bytes(json.loads(canonical_json_bytes(payload)))).hexdigest()
    assert first == second
