"""Market calendar helpers based on explicit trading sessions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow


def session_dates(values: Iterable[object]) -> tuple[pd.Timestamp, ...]:
    """Normalize provided session dates to sorted unique daily timestamps."""

    normalized = {pd.Timestamp(value).normalize() for value in values}
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
        if self.target_core_session_count <= 0:
            raise ValueError("invalid_target_core_session_count")
        if self.warmup_sessions < 0:
            raise ValueError("invalid_warmup_sessions")
        if self.lookahead_sessions < 0:
            raise ValueError("invalid_lookahead_sessions")
        object.__setattr__(self, "sessions", session_dates(self.sessions))

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
            sessions=session_dates(sessions),
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


__all__ = ["SessionWindow", "SessionWindowPlanner", "session_dates"]
