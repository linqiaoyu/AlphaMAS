from __future__ import annotations

import math
import statistics
from datetime import date, timedelta

from scripts.finmultitime.design_m1_contract import (
    MIN_TIME_SERIES_ROWS,
    choose_table_fact_with_diagnostic,
    duration_class,
    ts_summary,
)


def _table_row(
    *,
    start: str | None,
    end: str,
    filed: str,
    value: int,
    form: str = "10-Q",
    fp: str = "Q2",
    accession: str = "0000000000-23-000001",
) -> dict:
    return {
        "member": "financial_reports/aapl/test.json",
        "taxonomy": "us-gaap",
        "concept": "NetCashProvidedByUsedInOperatingActivities",
        "unit": "USD",
        "start": date.fromisoformat(start) if start else None,
        "end": date.fromisoformat(end),
        "filed": date.fromisoformat(filed),
        "raw_start": start,
        "raw_end": end,
        "raw_filed": filed,
        "val": value,
        "accn": accession,
        "form": form,
        "fy": 2023,
        "fp": fp,
        "frame": None,
    }


def test_q2_prefers_six_month_ytd_and_retains_duration() -> None:
    # Same latest period end, but one quarterly and one six-month YTD observation.
    quarterly = _table_row(
        start="2023-04-01",
        end="2023-06-30",
        filed="2023-08-01",
        value=30,
        accession="0000000000-23-000002",
    )
    ytd = _table_row(
        start="2023-01-01",
        end="2023-06-30",
        filed="2023-08-01",
        value=60,
        accession="0000000000-23-000003",
    )
    tables = {"AAPL": {"observations": [quarterly, ytd]}}

    selected, diagnostic = choose_table_fact_with_diagnostic(
        tables, "AAPL", quarterly["concept"], date(2023, 9, 1)
    )

    assert selected is not None
    assert selected["value"] == 60
    assert selected["period_start"] == "2023-01-01"
    assert selected["period_duration_days"] == 181
    assert selected["period_duration_class"] == "year_to_date_6m"
    assert diagnostic["selection_status"] == "AVAILABLE"
    assert {item["period_duration_days"] for item in diagnostic["duration_candidates_at_latest_end"]} == {91, 181}


def test_q1_prefers_quarterly_duration() -> None:
    quarterly = _table_row(
        start="2023-01-01", end="2023-03-31", filed="2023-05-01", value=30,
        fp="Q1", accession="0000000000-23-000010",
    )
    other = _table_row(
        start="2022-10-01", end="2023-03-31", filed="2023-05-01", value=60,
        fp="Q1", accession="0000000000-23-000011",
    )
    selected, _ = choose_table_fact_with_diagnostic(
        {"AAPL": {"observations": [other, quarterly]}},
        "AAPL", quarterly["concept"], date(2023, 6, 1),
    )
    assert selected is not None
    assert selected["value"] == 30
    assert selected["period_duration_class"] == "quarterly"


def test_q3_prefers_nine_month_ytd_amzn_like_example() -> None:
    quarterly = _table_row(
        start="2023-07-01", end="2023-09-30", filed="2023-10-27", value=30,
        fp="Q3", accession="0000000000-23-000012",
    )
    ytd_9m = _table_row(
        start="2023-01-01", end="2023-09-30", filed="2023-10-27", value=90,
        fp="Q3", accession="0000000000-23-000013",
    )
    rolling = _table_row(
        start="2022-10-01", end="2023-09-30", filed="2023-10-27", value=120,
        fp="Q3", accession="0000000000-23-000014",
    )
    selected, _ = choose_table_fact_with_diagnostic(
        {"AAPL": {"observations": [quarterly, rolling, ytd_9m]}},
        "AAPL", quarterly["concept"], date(2023, 11, 1),
    )
    assert selected is not None
    assert selected["value"] == 90
    assert selected["period_start"] == "2023-01-01"
    assert selected["period_end"] == "2023-09-30"
    assert selected["period_duration_class"] == "year_to_date_9m"


def test_fy_10k_prefers_annual_duration_for_53_week_calendar() -> None:
    quarterly = _table_row(
        start="2023-07-02", end="2023-09-30", filed="2023-11-03", value=30,
        form="10-K", fp="FY", accession="0000000000-23-000015",
    )
    annual = _table_row(
        start="2022-09-25", end="2023-09-30", filed="2023-11-03", value=120,
        form="10-K", fp="FY", accession="0000000000-23-000016",
    )
    selected, _ = choose_table_fact_with_diagnostic(
        {"AAPL": {"observations": [quarterly, annual]}},
        "AAPL", annual["concept"], date(2023, 11, 10),
    )
    assert selected is not None
    assert selected["period_duration_days"] == 371
    assert selected["period_duration_class"] == "annual"


def test_point_in_time_balance_sheet_fact() -> None:
    point = _table_row(
        start=None, end="2023-09-30", filed="2023-11-03", value=100,
        form="10-K", fp="FY", accession="0000000000-23-000017",
    )
    assert duration_class(point) == "point_in_time"


def test_duration_boundaries_are_tolerant_and_deterministic() -> None:
    expected = {
        44: "other_duration", 45: "quarterly", 120: "quarterly",
        121: "year_to_date_6m", 210: "year_to_date_6m",
        211: "year_to_date_9m", 300: "year_to_date_9m", 301: "annual",
        364: "annual", 371: "annual",
    }
    for days, expected_class in expected.items():
        start = date(2023, 1, 1)
        row = _table_row(
            start=start.isoformat(),
            end=(start + timedelta(days=days - 1)).isoformat(),
            filed="2025-01-01",
            value=days,
        )
        assert duration_class(row) == expected_class


def test_later_restatement_is_not_visible_before_filing_date() -> None:
    original = _table_row(
        start="2023-01-01",
        end="2023-03-31",
        filed="2023-05-01",
        value=100,
        fp="Q1",
        accession="0000000000-23-000004",
    )
    restated = _table_row(
        start="2023-01-01",
        end="2023-03-31",
        filed="2023-08-01",
        value=110,
        fp="Q1",
        accession="0000000000-23-000005",
    )
    tables = {"AAPL": {"observations": [original, restated]}}

    before, _ = choose_table_fact_with_diagnostic(
        tables, "AAPL", original["concept"], date(2023, 6, 1)
    )
    after, _ = choose_table_fact_with_diagnostic(
        tables, "AAPL", original["concept"], date(2023, 9, 1)
    )

    assert before is not None and before["value"] == 100
    assert after is not None and after["value"] == 110


def test_conflicting_identical_filing_metadata_is_unavailable() -> None:
    first = _table_row(
        start="2023-01-01", end="2023-06-30", filed="2023-08-01", value=100,
        accession="0000000000-23-000020",
    )
    conflicting = {**first, "val": 101}
    selected, diagnostic = choose_table_fact_with_diagnostic(
        {"AAPL": {"observations": [first, conflicting]}},
        "AAPL", first["concept"], date(2023, 9, 1),
    )
    assert selected is None
    assert diagnostic["selection_status"] == "UNAVAILABLE"
    assert diagnostic["selection_reason"] == "conflicting values at identical filing metadata"


def test_time_series_formulas_use_n_plus_one_rows_and_prior_volume_window() -> None:
    assert MIN_TIME_SERIES_ROWS == 61
    rows = []
    for index in range(MIN_TIME_SERIES_ROWS):
        close = 100.0 + index
        rows.append(
            {
                "session_date": date(2024, 1, 1) + timedelta(days=index),
                "numeric": {
                    "Close": close,
                    "High": close + 2.0,
                    "Low": close - 2.0,
                    "Volume": 100.0 + index,
                },
            }
        )
    summary = ts_summary({"rows": rows})

    assert math.isclose(summary["cumulative_return_5d"], 160 / 155 - 1)
    assert math.isclose(summary["cumulative_return_20d"], 160 / 140 - 1)
    assert math.isclose(summary["cumulative_return_60d"], 160 / 100 - 1)
    returns = [rows[i]["numeric"]["Close"] / rows[i - 1]["numeric"]["Close"] - 1 for i in range(1, 61)]
    assert math.isclose(
        summary["realised_volatility_20d_annualised"],
        statistics.stdev(returns[-20:]) * math.sqrt(252),
    )
    assert math.isclose(summary["relative_volume_vs_20d_mean"], 160 / statistics.mean(range(140, 160)))
