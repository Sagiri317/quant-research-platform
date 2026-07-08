"""Generic market session-window planning based on explicit sessions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow, require_date_window


def normalize_session_dates(values: Iterable[object]) -> tuple[pd.Timestamp, ...]:
    """Normalize provided session dates to sorted unique daily timestamps."""

    if isinstance(values, str):
        raise ValueError("invalid_session_dates")
    try:
        normalized = {pd.Timestamp(value).normalize() for value in values}
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_session_dates") from exc
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class SessionWindow:
    core: TimeWindow
    read: TimeWindow
    core_sessions: tuple[pd.Timestamp, ...]
    read_sessions: tuple[pd.Timestamp, ...]
    reason: str


@dataclass(frozen=True)
class SessionWindowPlanner:
    """Plan trading-session core/read windows without resource-budget knowledge."""

    sessions: tuple[pd.Timestamp, ...]
    target_core_session_count: int
    warmup_sessions: int = 0
    lookahead_sessions: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(
            self.target_core_session_count,
            error="invalid_target_core_session_count",
        )
        _require_non_negative_int(
            self.warmup_sessions,
            error="invalid_warmup_sessions",
        )
        _require_non_negative_int(
            self.lookahead_sessions,
            error="invalid_lookahead_sessions",
        )
        object.__setattr__(
            self,
            "sessions",
            normalize_session_dates(self.sessions),
        )

    @classmethod
    def from_sessions(
        cls,
        sessions: Iterable[object],
        *,
        target_core_session_count: int,
        warmup_sessions: int = 0,
        lookahead_sessions: int = 0,
    ) -> SessionWindowPlanner:
        return cls(
            sessions=normalize_session_dates(sessions),
            target_core_session_count=target_core_session_count,
            warmup_sessions=warmup_sessions,
            lookahead_sessions=lookahead_sessions,
        )

    def windows(
        self,
        time_window: TimeWindow,
        *,
        reason: str,
    ) -> tuple[SessionWindow, ...]:
        require_date_window(time_window, context="session_window_planner")
        target_sessions = tuple(
            session
            for session in self.sessions
            if time_window.start <= session.date() <= time_window.end
        )
        if not target_sessions:
            raise ValueError("no_trading_sessions_in_time_window")
        first_index = self.sessions.index(target_sessions[0])
        last_index = self.sessions.index(target_sessions[-1])
        result: list[SessionWindow] = []
        for start_index in range(
            first_index,
            last_index + 1,
            self.target_core_session_count,
        ):
            end_index = min(
                start_index + self.target_core_session_count,
                last_index + 1,
            )
            core = self.sessions[start_index:end_index]
            if not core:
                continue
            read_start_index = max(0, start_index - self.warmup_sessions)
            read_end_index = min(
                len(self.sessions),
                end_index + self.lookahead_sessions,
            )
            read = self.sessions[read_start_index:read_end_index]
            result.append(
                SessionWindow(
                    core=TimeWindow(core[0].date(), core[-1].date()),
                    read=TimeWindow(read[0].date(), read[-1].date()),
                    core_sessions=core,
                    read_sessions=read,
                    reason=reason,
                )
            )
        return tuple(result)


__all__ = [
    "SessionWindow",
    "SessionWindowPlanner",
    "normalize_session_dates",
]


def _require_positive_int(value: object, *, error: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(error)


def _require_non_negative_int(value: object, *, error: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(error)
