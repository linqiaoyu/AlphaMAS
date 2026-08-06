def test_m0_schedule_contract(xnys):
    events = xnys.weekly_events("2024-01-01", "2024-06-24")
    assert len(events) == 26
    assert events[0].decision_session == "2024-01-05"
    assert events[0].execution_session == "2024-01-08"
    assert events[12].decision_session == "2024-03-28"
    assert events[-1].decision_session == "2024-06-28"
    assert events[-1].execution_session == "2024-07-01"
    assert events[0].decision_close_utc.tzinfo is not None
    assert events[0].execution_open_utc.tzinfo is not None
    assert events[0].decision_close_ny.hour == 16
    assert events[0].execution_open_ny.hour == 9
