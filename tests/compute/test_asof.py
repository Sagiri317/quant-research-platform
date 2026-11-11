"""Dependency clocks retain exact instants and unknown rolling history."""

import pandas as pd

from quant_research_platform.compute.dataframe.asof import rolling_available_at


def test_rolling_dependency_clock_uses_latest_known_instant_without_compressing_gaps():
    origin = pd.Timestamp("2024-01-02 09:30", tz="Asia/Shanghai")
    clocks = pd.Series(
        [
            origin + pd.Timedelta(nanoseconds=offset) if offset is not None else pd.NaT
            for offset in (3, 1, 2, None, 8, 5, 6, 4)
        ],
        index=list("abcdefgh"),
        dtype="datetime64[ns, Asia/Shanghai]",
    )
    expected = pd.Series(
        [
            origin.tz_convert("UTC").tz_localize(None)
            + pd.Timedelta(nanoseconds=offset)
            if offset is not None
            else pd.NaT
            for offset in (None, None, 3, None, None, None, 8, 6)
        ],
        index=clocks.index,
        dtype="datetime64[ns]",
    )
    pd.testing.assert_series_equal(rolling_available_at(clocks, 3), expected)
