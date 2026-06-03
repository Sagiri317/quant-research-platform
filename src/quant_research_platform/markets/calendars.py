"""Market calendar helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def session_dates(start: str, end: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(start=start, end=end, freq="B"))


@dataclass(frozen=True)
class ChunkPlanner:
    """Plan chunks by trading sessions, not calendar-day spans."""

    sessions: tuple[pd.Timestamp, ...]
    chunk_size: int

    @classmethod
    def from_range(cls, start: str, end: str, *, chunk_size: int) -> ChunkPlanner:
        if chunk_size <= 0:
            raise ValueError("invalid_chunk_size")
        return cls(session_dates(start, end), chunk_size)

    def chunks(self) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
        result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for index in range(0, len(self.sessions), self.chunk_size):
            chunk = self.sessions[index : index + self.chunk_size]
            if chunk:
                result.append((chunk[0], chunk[-1]))
        return tuple(result)


__all__ = ["ChunkPlanner", "session_dates"]
