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


def test_rolling_dependency_clocks_across_blocks_match_independent_windows():
    origin = pd.Timestamp("1965-01-01")
    clocks = pd.Series(
        [origin + pd.Timedelta(offset, unit="ns") for offset in [8, 5, 7, 4, 11, 6, 2]]
    )
    clocks.iloc[2] = pd.NaT
    for window in (1, 2, 3, 4, len(clocks), len(clocks) + 1):
        expected = pd.Series(
            [
                part.max() if len(part) == window and part.notna().all() else pd.NaT
                for i in range(len(clocks))
                for part in [clocks.iloc[max(0, i - window + 1) : i + 1]]
            ],
            dtype="datetime64[ns]",
        )
        pd.testing.assert_series_equal(rolling_available_at(clocks, window), expected)
