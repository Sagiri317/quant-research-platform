"""Market calendar helpers based on explicit trading sessions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


def session_dates(values: Iterable[object]) -> tuple[pd.Timestamp, ...]:
    """Normalize provided session dates to sorted unique daily timestamps."""

    normalized = {pd.Timestamp(value).normalize() for value in values}
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class SessionChunk:
    core_start: pd.Timestamp
    core_end: pd.Timestamp
    read_start: pd.Timestamp
    read_end: pd.Timestamp
    core_sessions: tuple[pd.Timestamp, ...]
    read_sessions: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class ChunkPlanner:
    """Plan chunks by provided trading sessions, not calendar-day spans."""

    sessions: tuple[pd.Timestamp, ...]
    chunk_sessions: int
    warmup_sessions: int = 0
    lookahead_sessions: int = 0

    def __post_init__(self) -> None:
        if self.chunk_sessions <= 0:
            raise ValueError("invalid_chunk_sessions")
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
        chunk_sessions: int,
        warmup_sessions: int = 0,
        lookahead_sessions: int = 0,
    ) -> ChunkPlanner:
        return cls(
            sessions=session_dates(sessions),
            chunk_sessions=chunk_sessions,
            warmup_sessions=warmup_sessions,
            lookahead_sessions=lookahead_sessions,
        )

    def chunks(self) -> tuple[SessionChunk, ...]:
        result: list[SessionChunk] = []
        for start_index in range(0, len(self.sessions), self.chunk_sessions):
            end_index = min(start_index + self.chunk_sessions, len(self.sessions))
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
                SessionChunk(
                    core_start=core[0],
                    core_end=core[-1],
                    read_start=read[0],
                    read_end=read[-1],
                    core_sessions=core,
                    read_sessions=read,
                )
            )
        return tuple(result)


__all__ = ["ChunkPlanner", "SessionChunk", "session_dates"]
