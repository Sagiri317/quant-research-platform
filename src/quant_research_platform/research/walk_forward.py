"""Protocol-owned expanding walk-forward quarter boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardQuarter:
    quarter_id: str
    training_window: tuple[pd.Timestamp, pd.Timestamp]
    validation_window: tuple[pd.Timestamp, pd.Timestamp]
    test_window: tuple[pd.Timestamp, pd.Timestamp]
    phase: str


def a_share_walk_forward_quarters(
    sessions: pd.DatetimeIndex,
) -> tuple[WalkForwardQuarter, ...]:
    axis = pd.DatetimeIndex(sessions).normalize().drop_duplicates().sort_values()
    development = axis[(axis.year >= 2014) & (axis.year <= 2023)]
    holdout = axis[axis.year >= 2024]
    if development.empty or holdout.empty:
        raise ValueError("a_share_walk_forward_sessions_incomplete")
    quarters: list[WalkForwardQuarter] = []
    for period in pd.PeriodIndex(development.to_period("Q")).unique():
        if period < pd.Period("2019Q1"):
            continue
        test = development[development.to_period("Q") == period]
        validation = development[
            (development >= period.start_time - pd.DateOffset(months=12))
            & (development < period.start_time)
        ]
        training = (
            development[development < validation.min()]
            if not validation.empty
            else development[development < test.min()]
        )
        if training.empty or validation.empty or test.empty:
            raise ValueError("a_share_walk_forward_window_incomplete")
        quarters.append(
            WalkForwardQuarter(
                quarter_id=str(period),
                training_window=(training[0], training[-1]),
                validation_window=(validation[0], validation[-1]),
                test_window=(test[0], test[-1]),
                phase="development",
            )
        )
    if holdout.empty:
        raise ValueError("a_share_holdout_sessions_missing")
    return tuple(quarters)
