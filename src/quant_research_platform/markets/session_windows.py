"""Generic market session-window planning based on explicit sessions."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow, require_date_window


def normalize_session_dates(values: Iterable[object]) -> pd.DatetimeIndex:
    """Normalize provided session dates into one compact ordered state."""

    if isinstance(values, str):
        raise ValueError("invalid_session_dates")
    if isinstance(values, pd.DatetimeIndex) and type(values) is not pd.DatetimeIndex:
        raise ValueError("invalid_session_dates")
    if isinstance(values, pd.DatetimeIndex) and values.tz is not None:
        raise ValueError("invalid_session_dates")
    if isinstance(values, pd.DatetimeIndex) and values.isna().any():
        raise ValueError("invalid_session_dates")
    if (
        isinstance(values, pd.DatetimeIndex)
        and values.tz is None
        and values.is_normalized
        and values.is_monotonic_increasing
        and values.is_unique
    ):
        return values
    try:
        normalized = pd.DatetimeIndex(values).normalize()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid_session_dates") from exc
    if normalized.tz is not None:
        raise ValueError("invalid_session_dates")
    if normalized.isna().any():
        raise ValueError("invalid_session_dates")
    return normalized.drop_duplicates().sort_values()


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

    sessions: pd.DatetimeIndex
    target_core_session_count: int
    warmup_sessions: int = 0
    lookahead_sessions: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.target_core_session_count) is not int
            or self.target_core_session_count <= 0
        ):
            raise ValueError("invalid_target_core_session_count")
        _require_non_negative_int(
            self.warmup_sessions,
            error="invalid_warmup_sessions",
        )
        _require_non_negative_int(
            self.lookahead_sessions,
            error="invalid_lookahead_sessions",
        )
        normalized = normalize_session_dates(self.sessions)
        owned = normalized.copy(deep=True)
        object.__setattr__(self, "sessions", owned)

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
            sessions=sessions,
            target_core_session_count=target_core_session_count,
            warmup_sessions=warmup_sessions,
            lookahead_sessions=lookahead_sessions,
        )

    def windows(
        self,
        time_window: TimeWindow,
        *,
        reason: str,
    ) -> Iterator[SessionWindow]:
        require_date_window(time_window, context="session_window_planner")
        sessions = self.sessions.copy(deep=True)
        first_index = bisect_left(sessions, pd.Timestamp(time_window.start))
        end_index = bisect_right(sessions, pd.Timestamp(time_window.end))
        if first_index == end_index:
            raise ValueError("no_trading_sessions_in_time_window")

        def iter_windows() -> Iterator[SessionWindow]:
            for start_index in range(
                first_index,
                end_index,
                self.target_core_session_count,
            ):
                core_end_index = min(
                    start_index + self.target_core_session_count,
                    end_index,
                )
                core = tuple(sessions[start_index:core_end_index])
                if not core:
                    continue
                read_start_index = max(0, start_index - self.warmup_sessions)
                read_end_index = min(
                    len(sessions),
                    core_end_index + self.lookahead_sessions,
                )
                read = tuple(sessions[read_start_index:read_end_index])
                yield SessionWindow(
                    core=TimeWindow(core[0].date(), core[-1].date()),
                    read=TimeWindow(read[0].date(), read[-1].date()),
                    core_sessions=core,
                    read_sessions=read,
                    reason=reason,
                )

        return iter_windows()



def _require_non_negative_int(value: object, *, error: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(error)


def require_non_negative_session_offset(value: object, *, error: str) -> None:
    """Require one exact non-negative session offset at its owner boundary."""

    _require_non_negative_int(value, error=error)
