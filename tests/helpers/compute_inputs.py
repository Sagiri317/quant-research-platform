from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def make_cross_section_compute_panel(
    *,
    dates: Sequence[str] = ("2024-01-02", "2024-01-03"),
    include_alt_signal: bool = True,
    timestamps_as_datetime: bool = True,
) -> pd.DataFrame:
    repeated_dates = [date for date in dates for _ in ("A", "B", "C")]
    timestamps: object = (
        pd.to_datetime(repeated_dates) if timestamps_as_datetime else repeated_dates
    )
    data: dict[str, object] = {
        "timestamp": timestamps,
        "asset": ["A", "B", "C"] * len(dates),
        "factor": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        "forward_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
    }
    if include_alt_signal:
        data["factor_alt"] = [3.0, 2.0, 1.0, 1.0, 2.0, 3.0]
    return pd.DataFrame(data)


def make_perfect_cross_section_compute_panel() -> pd.DataFrame:
    return make_cross_section_compute_panel(
        dates=("2026-01-01", "2026-01-02"),
        include_alt_signal=False,
        timestamps_as_datetime=False,
    )


def make_two_asset_portfolio_compute_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": ["A", "B"],
            "factor": [1.0, 2.0],
            "label": [0.01, -0.02],
            "segment": ["s1", "s1"],
        }
    )
