"""Prediction extent shared by native computation and persisted consumers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)


@dataclass(frozen=True)
class PredictionWindow:
    """Prediction extent, optionally through the end of the supplied input."""

    start: date | datetime
    end: date | datetime | None

    def __post_init__(self) -> None:
        finite = TimeWindow(self.start, self.start if self.end is None else self.end)
        object.__setattr__(self, "start", finite.start)
        if self.end is not None:
            object.__setattr__(self, "end", finite.end)


def prediction_window_mask(timestamps: pd.Series, window) -> pd.Series:
    """Select actual timestamps without inventing an endpoint for open prediction."""
    if window.end is None:
        return timestamps.ge(pd.Timestamp(window.start))
    lower, upper, closed = time_window_timestamp_interval(window)
    return timestamps.ge(pd.Timestamp(lower)) & (
        timestamps.le(pd.Timestamp(upper))
        if closed
        else timestamps.lt(pd.Timestamp(upper))
    )
