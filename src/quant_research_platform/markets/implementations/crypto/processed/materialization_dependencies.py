"""Crypto processed dependency-window planning."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_HORIZONS,
)

FACTOR_WARMUP_MINUTES = 4
LABEL_LOOKAHEAD_EXTRA_MINUTES = max(CRYPTO_FORWARD_RETURN_HORIZONS) + 1


def factor_read_window(core: TimeWindow) -> TimeWindow:
    core_start, core_end = _minute_core_bounds(core)
    start = core_start - pd.Timedelta(minutes=FACTOR_WARMUP_MINUTES)
    return TimeWindow(start.to_pydatetime(), core_end.to_pydatetime())


def label_read_window(core: TimeWindow) -> TimeWindow:
    core_start, core_end = _minute_core_bounds(core)
    end = core_end + pd.Timedelta(minutes=LABEL_LOOKAHEAD_EXTRA_MINUTES)
    return TimeWindow(core_start.to_pydatetime(), end.to_pydatetime())


def panel_dependency_window(core: TimeWindow) -> TimeWindow:
    factor = factor_read_window(core)
    label = label_read_window(core)
    return TimeWindow(
        min(pd.Timestamp(factor.start), pd.Timestamp(label.start)).to_pydatetime(),
        max(pd.Timestamp(factor.end), pd.Timestamp(label.end)).to_pydatetime(),
    )


def _minute_core_bounds(core: TimeWindow) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_bound, end_bound, end_inclusive = time_window_timestamp_interval(core)
    core_start = pd.Timestamp(start_bound)
    core_end = pd.Timestamp(end_bound)
    if not end_inclusive:
        core_end = core_end - pd.Timedelta(minutes=1)
    return core_start, core_end


__all__ = [
    "FACTOR_WARMUP_MINUTES",
    "LABEL_LOOKAHEAD_EXTRA_MINUTES",
    "factor_read_window",
    "label_read_window",
    "panel_dependency_window",
]
