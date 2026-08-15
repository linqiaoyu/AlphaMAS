from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts.m2.build_semantic_state_representation import load_rows
from scripts.m2.semantic_state_representation import (
    ACTION_ORDER,
    EconomicPayload,
    SemanticBaseStore,
    action_onehot,
    build_actor_observation,
    build_portfolio_state,
    build_semantic_base,
    extract_economic_payload,
    mrl_prefix_normalize,
    select_embedding_dimension,
    text_sha256,
    validate_token_count,
)
from tradingagents.backtesting.models import PortfolioSnapshot
from tradingagents.backtesting.portfolio import Portfolio

EXPERIMENTS = Path("/Users/yulinqiao/.local/share/alphamas/repos/AlphaMAS-Experiments")
CORPUS = EXPERIMENTS / "experiments/M2/development/semantic_handoff_trainval_v1"
ROOT = Path(__file__).resolve().parents[2]


def _actor_state(**metadata: object) -> dict[str, object]:
    return {
        "schema_version": "M2-SEMANTIC-HANDOFF-v1",
        "case_id": metadata.get("case_id", "AAPL:2023-05-05"),
        "role": metadata.get("role", "TRAIN"),
        "symbol": metadata.get("symbol", "AAPL"),
        "ticker": metadata.get("ticker", "AAPL"),
        "company_of_interest": metadata.get("company_of_interest", "Apple"),
        "provenance": metadata.get("provenance", {"source_sha": "abc", "run_identity": "one"}),
        "research_manager": {"investment_plan": "Exact Manager text: AAPL $100."},
        "prompt_trader": {
            "rendered_proposal": "Exact Trader text: BUY AAPL at $100.",
            "normalized_action": "BUY",
        },
    }


def _snapshot(
    *, quantity: float = 0.0, close: float = 100.0, entry: float = 0.0, drawdown: float = 0.0
) -> PortfolioSnapshot:
    market_value = quantity * close
    return PortfolioSnapshot(
        timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
        session="2023-01-01",
        symbol="SYNTH",
        cash=100_000.0 - market_value,
        quantity=quantity,
        close_price=close,
        market_value=market_value,
        equity=100_000.0,
        current_weight=market_value / 100_000.0,
        average_entry_price=entry,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        cumulative_cost=0.0,
        current_drawdown=drawdown,
        peak_equity=100_000.0,
    )


def test_two_channel_payload_is_exact_and_exclusive() -> None:
    payload = extract_economic_payload(_actor_state())
    assert payload == EconomicPayload(
        "Exact Manager text: AAPL $100.",
        "Exact Trader text: BUY AAPL at $100.",
        "BUY",
    )
    assert payload.research_manager_text != payload.prompt_trader_text


def test_exact_frozen_source_population_and_order() -> None:
    rows = load_rows(CORPUS)
    assert len(rows) == 72
    assert sum(row["role"] == "TRAIN" for row in rows) == 56
    assert sum(row["role"] == "VALIDATION" for row in rows) == 16
    assert all(row["role"] == "TRAIN" for row in rows[:56])
    assert all(row["role"] == "VALIDATION" for row in rows[56:])


def test_metadata_and_split_cannot_change_payload_or_observation() -> None:
    first = extract_economic_payload(_actor_state())
    second = extract_economic_payload(
        _actor_state(
            role="VALIDATION",
            case_id="JPM:2099-01-01",
            symbol="JPM",
            ticker="JPM",
            company_of_interest="JPMorgan",
            provenance={"source_sha": "different", "run_identity": "different"},
        )
    )
    assert first == second
    semantic = np.arange(16, dtype=np.float32)
    snapshot = _snapshot()
    assert build_actor_observation(semantic, snapshot).tobytes() == build_actor_observation(
        semantic, replace(snapshot, symbol="JPM", session="2099-01-01")
    ).tobytes()


def test_one_character_text_change_changes_identity() -> None:
    assert text_sha256("investment plan") != text_sha256("investment Plan")


@pytest.mark.parametrize(
    ("action", "expected"),
    [("BUY", [1, 0, 0]), ("HOLD", [0, 1, 0]), ("SELL", [0, 0, 1])],
)
def test_action_prior_order(action: str, expected: list[int]) -> None:
    assert ACTION_ORDER == ("BUY", "HOLD", "SELL")
    np.testing.assert_array_equal(action_onehot(action), expected)


def test_mrl_is_prefix_then_l2_normalize() -> None:
    full = np.zeros((1, 1024), dtype=np.float32)
    full[0, :3] = [3, 4, 0]
    selected = mrl_prefix_normalize(full, 256)
    np.testing.assert_allclose(selected[0, :3], [0.6, 0.8, 0.0], atol=1e-7)
    assert np.linalg.norm(selected[0]) == pytest.approx(1.0, abs=1e-6)


def test_dimension_selector_uses_smallest_passing_candidate() -> None:
    rng = np.random.default_rng(7)
    manager = np.zeros((56, 1024), dtype=np.float32)
    trader = np.zeros((56, 1024), dtype=np.float32)
    manager[:, :256] = rng.normal(size=(56, 256))
    trader[:, :256] = rng.normal(size=(56, 256))
    selected, diagnostics = select_embedding_dimension(manager, trader)
    assert selected == 256
    assert all(view["pass"] for view in diagnostics["256"].values())


def test_dimension_selector_falls_back_to_1024() -> None:
    rng = np.random.default_rng(12)
    manager = rng.normal(size=(56, 1024)).astype(np.float32)
    trader = rng.normal(size=(56, 1024)).astype(np.float32)
    manager[:, :512] *= 0.01
    trader[:, :512] *= 0.01
    selected, diagnostics = select_embedding_dimension(manager, trader)
    assert selected == 1024
    assert not all(view["pass"] for view in diagnostics["256"].values())
    assert not all(view["pass"] for view in diagnostics["512"].values())
    assert all(view["pass"] for view in diagnostics["1024"].values())


def test_validation_geometry_is_not_an_input_to_selector() -> None:
    rng = np.random.default_rng(17)
    manager = np.zeros((72, 1024), dtype=np.float32)
    trader = np.zeros((72, 1024), dtype=np.float32)
    manager[:56, :256] = rng.normal(size=(56, 256))
    trader[:56, :256] = rng.normal(size=(56, 256))
    selected_before, metrics_before = select_embedding_dimension(manager[:56], trader[:56])
    manager[56:] = rng.normal(size=(16, 1024)) * 1_000_000
    trader[56:] = rng.normal(size=(16, 1024)) * -1_000_000
    selected_after, metrics_after = select_embedding_dimension(manager[:56], trader[:56])
    assert selected_before == selected_after == 256
    assert metrics_before == metrics_after


def test_semantic_base_row_integrity() -> None:
    manager = np.array([[1.0, 0.0]], dtype=np.float32)
    trader = np.array([[0.0, 1.0]], dtype=np.float32)
    base = build_semantic_base(manager, trader, ["SELL"])
    np.testing.assert_array_equal(base[0, :2], manager[0])
    np.testing.assert_array_equal(base[0, 2:4], trader[0])
    np.testing.assert_array_equal(base[0, 4:6], trader[0] - manager[0])
    assert base[0, 6] == np.dot(manager[0], trader[0])
    np.testing.assert_array_equal(base[0, -3:], [0, 0, 1])
    assert base.shape == (1, 10)
    assert base.dtype == np.float32


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (_snapshot(), [1, 0, 0, 0]),
        (_snapshot(quantity=100, close=100, entry=100), [0, 1, 0, 0]),
        (_snapshot(quantity=100, close=110, entry=100), [0, 1, np.log(1.1), 0]),
        (_snapshot(quantity=100, close=90, entry=100), [0, 1, np.log(0.9), 0]),
        (_snapshot(quantity=100, close=90, entry=100, drawdown=-0.2), [0, 1, np.log(0.9), -0.2]),
        (_snapshot(quantity=200, close=50, entry=50), [0, 1, 0, 0]),
    ],
)
def test_portfolio_state_contract(snapshot: PortfolioSnapshot, expected: list[float]) -> None:
    result = build_portfolio_state(snapshot)
    np.testing.assert_allclose(result, expected, atol=1e-7)
    assert result.dtype == np.float32
    assert result[0] + result[1] == 1


def test_portfolio_tolerance_classifies_cash() -> None:
    state = build_portfolio_state(_snapshot(quantity=Portfolio.tolerance, close=100, entry=0))
    np.testing.assert_array_equal(state[:3], [1, 0, 0])


@pytest.mark.parametrize(
    "snapshot",
    [
        _snapshot(quantity=1, close=0, entry=100),
        _snapshot(quantity=1, close=100, entry=0),
        _snapshot(drawdown=0.01),
        _snapshot(quantity=float("nan")),
    ],
)
def test_invalid_portfolio_state_is_rejected(snapshot: PortfolioSnapshot) -> None:
    with pytest.raises(ValueError):
        build_portfolio_state(snapshot)


def test_actor_observation_shape_and_nonfinite_rejection() -> None:
    semantic = np.ones(3 * 256 + 4, dtype=np.float32)
    observation = build_actor_observation(semantic, _snapshot())
    assert observation.shape == (3 * 256 + 8,)
    assert observation.dtype == np.float32
    semantic[0] = np.inf
    with pytest.raises(ValueError):
        build_actor_observation(semantic, _snapshot())


def test_token_ceiling_is_fail_closed() -> None:
    assert validate_token_count(32768) == 32768
    with pytest.raises(ValueError, match="truncation is forbidden"):
        validate_token_count(32769)


def test_loader_exposes_only_semantic_rows(tmp_path) -> None:
    path = tmp_path / "semantic_base.npy"
    np.save(path, np.ones((72, 772), dtype=np.float32), allow_pickle=False)
    store = SemanticBaseStore(path)
    assert store.shape == (72, 772)
    assert store.semantic_base(0).shape == (772,)
    assert not hasattr(store, "metadata")


def test_machine_freeze_binds_final_archive_and_dimensions() -> None:
    import json

    freeze = json.loads(
        (ROOT / "docs/m2/m2_semantic_state_representation_freeze.json").read_text()
    )
    assert freeze["schema_version"] == "M2-SEMANTIC-STATE-FREEZE-v1"
    assert freeze["representation_archive_sha"] == (
        "1a94fdd20c6c3c004cbe5c2340171e86922c49f7"
    )
    assert freeze["selected_dimension"] == 1024
    assert freeze["semantic_base_dimension"] == 3076
    assert freeze["actor_observation_dimension"] == 3080
    assert freeze["selection_train_cases"] == 56
    assert freeze["selection_validation_cases"] == 0
    assert freeze["reward_used"] is False
    assert freeze["final_holdout_used"] is False
