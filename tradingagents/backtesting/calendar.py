"""XNYS session scheduling; no hand-written weekdays or holidays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd


@dataclass(frozen=True)
class WeeklyEvent:
    week_start: str
    decision_session: str
    decision_close_utc: pd.Timestamp
    decision_close_ny: pd.Timestamp
    execution_session: str
    execution_open_utc: pd.Timestamp
    execution_open_ny: pd.Timestamp

    def to_dict(self) -> dict:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, pd.Timestamp):
                result[key] = value.isoformat()
        return result


class ExchangeSchedule:
    def __init__(
        self, calendar_name: str = "XNYS", *, start: str = "1990-01-01",
        end: str = "2050-12-31",
    ) -> None:
        self.name = calendar_name
        self.calendar = xcals.get_calendar(calendar_name, start=start, end=end)
        self.ny = ZoneInfo("America/New_York")

    def sessions(self, start: str | date, end: str | date) -> pd.DatetimeIndex:
        return self.calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))

    def session_open(self, session: str | pd.Timestamp) -> pd.Timestamp:
        return self.calendar.session_open(pd.Timestamp(session)).tz_convert("UTC")

    def session_close(self, session: str | pd.Timestamp) -> pd.Timestamp:
        return self.calendar.session_close(pd.Timestamp(session)).tz_convert("UTC")

    def next_session(self, session: str | pd.Timestamp) -> pd.Timestamp:
        return self.calendar.next_session(pd.Timestamp(session))

    def weekly_events(self, first_week: str, final_week: str) -> list[WeeklyEvent]:
        first = pd.Timestamp(first_week).normalize()
        last = pd.Timestamp(final_week).normalize()
        if first.weekday() != 0 or last.weekday() != 0:
            raise ValueError("calendar week boundaries must be Mondays")
        events: list[WeeklyEvent] = []
        for week_start in pd.date_range(first, last, freq="7D"):
            week_sessions = self.sessions(week_start, week_start + pd.Timedelta(days=6))
            if week_sessions.empty:
                raise ValueError(f"no {self.name} session in week {week_start.date()}")
            decision = week_sessions[-1]
            execution = self.next_session(decision)
            close = self.session_close(decision)
            opened = self.session_open(execution)
            events.append(WeeklyEvent(
                week_start=week_start.date().isoformat(),
                decision_session=decision.date().isoformat(),
                decision_close_utc=close,
                decision_close_ny=close.tz_convert(self.ny),
                execution_session=execution.date().isoformat(),
                execution_open_utc=opened,
                execution_open_ny=opened.tz_convert(self.ny),
            ))
        return events
