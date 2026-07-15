from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.session_windows import (
    SessionWindowPlanner,
    normalize_session_dates,
)


def test_normalize_session_dates_returns_sorted_unique_daily_timestamps() -> None:
    assert normalize_session_dates(
        ["2014-01-03 12:00", "2014-01-02", "2014-01-03"]
    ) == (pd.Timestamp("2014-01-02"), pd.Timestamp("2014-01-03"))


def test_session_window_planner_maps_non_trading_bounds_inside_closed_window() -> None:
    windows = SessionWindowPlanner.from_sessions(
        ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-10"],
        target_core_session_count=2,
    ).windows(TimeWindow("2014-01-04", "2014-01-11"), reason="research_assembly")

    assert [
        (window.core.start.isoformat(), window.core.end.isoformat())
        for window in windows
    ] == [
        ("2014-01-06", "2014-01-10"),
    ]
    assert windows[0].core_sessions == tuple(
        pd.Timestamp(value) for value in ("2014-01-06", "2014-01-10")
    )


def test_session_window_planner_fails_closed_when_window_has_no_sessions() -> None:
    planner = SessionWindowPlanner.from_sessions(
        ["2014-01-02", "2014-01-03"],
        target_core_session_count=1,
    )

    with pytest.raises(ValueError, match="no_trading_sessions_in_time_window"):
        planner.windows(TimeWindow("2014-01-04", "2014-01-05"), reason="factor_warmup")


def test_session_window_planner_covers_core_sessions_once() -> None:
    windows = SessionWindowPlanner.from_sessions(
        [
            "2014-01-02",
            "2014-01-03",
            "2014-01-06",
            "2014-01-10",
            "2014-01-13",
        ],
        target_core_session_count=2,
        warmup_sessions=1,
        lookahead_sessions=1,
    ).windows(TimeWindow("2014-01-02", "2014-01-13"), reason="factor_warmup")

    core_sessions = [session for window in windows for session in window.core_sessions]
    assert [session.date().isoformat() for session in core_sessions] == [
        "2014-01-02",
        "2014-01-03",
        "2014-01-06",
        "2014-01-10",
        "2014-01-13",
    ]
    assert len(core_sessions) == len(set(core_sessions))
    assert windows[0].read.end.isoformat() == "2014-01-06"
    assert windows[1].read.start.isoformat() == "2014-01-03"
    assert windows[1].read.end.isoformat() == "2014-01-13"
    assert windows[0].read_sessions[-1] in windows[1].read_sessions
    assert windows[0].reason == "factor_warmup"


def test_session_window_planner_core_window_may_cross_years() -> None:
    windows = SessionWindowPlanner.from_sessions(
        ["2014-12-31", "2015-01-05"],
        target_core_session_count=2,
    ).windows(TimeWindow("2014-12-31", "2015-01-05"), reason="panel_next_session")

    assert len(windows) == 1
    assert windows[0].core.start.isoformat() == "2014-12-31"
    assert windows[0].core.end.isoformat() == "2015-01-05"


def test_session_windows_map_non_trading_bounds_and_cover_core_once() -> None:
    sessions = [
        "2014-12-31",
        "2015-01-05",
        "2015-01-06",
        "2015-01-07",
        "2015-01-08",
    ]
    windows = SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=2,
        warmup_sessions=1,
        lookahead_sessions=1,
    ).windows(TimeWindow("2015-01-01", "2015-01-10"), reason="factor_warmup")

    core_sessions = [session for window in windows for session in window.core_sessions]
    assert [session.date().isoformat() for session in core_sessions] == [
        "2015-01-05",
        "2015-01-06",
        "2015-01-07",
        "2015-01-08",
    ]
    assert len(core_sessions) == len(set(core_sessions))
    assert windows[0].read.start.isoformat() == "2014-12-31"
    assert windows[0].read.end.isoformat() == "2015-01-07"
    assert windows[1].read.start.isoformat() == "2015-01-06"
    assert windows[1].read.end.isoformat() == "2015-01-08"
    assert windows[0].core.end < windows[1].core.start
    assert windows[0].read.end >= windows[1].core.start


def test_session_windows_fail_closed_when_time_window_has_no_sessions() -> None:
    planner = SessionWindowPlanner.from_sessions(
        ["2015-01-05", "2015-01-06"],
        target_core_session_count=1,
    )

    with pytest.raises(ValueError, match="no_trading_sessions_in_time_window"):
        planner.windows(TimeWindow("2015-01-03", "2015-01-04"), reason="label")


def test_session_windows_reject_bare_str_sessions() -> None:
    with pytest.raises(ValueError, match="invalid_session_dates"):
        normalize_session_dates("2015-01-05")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("target_core_session_count", True, "invalid_target_core_session_count"),
        ("target_core_session_count", 1.5, "invalid_target_core_session_count"),
        ("warmup_sessions", True, "invalid_warmup_sessions"),
        ("lookahead_sessions", 1.5, "invalid_lookahead_sessions"),
    ],
)
def test_session_windows_reject_non_strict_session_counts(
    field: str,
    value: object,
    error: str,
) -> None:
    kwargs = {
        "target_core_session_count": 1,
        "warmup_sessions": 0,
        "lookahead_sessions": 0,
        field: value,
    }

    with pytest.raises(ValueError, match=error):
        SessionWindowPlanner(
            sessions=(pd.Timestamp("2015-01-05"),),
            **kwargs,  # type: ignore[arg-type]
        )
