from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.calendars import SessionWindowPlanner


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
