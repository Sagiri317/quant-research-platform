from __future__ import annotations

import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.calendars import SessionWindowPlanner
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
)


def test_resource_chunk_planner_uses_budget_only_not_trading_calendar() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * 1024 * 1024,
        environ={"QRP_MEMORY_BUDGET_MB": "1"},
    )
    planner = ResourceChunkPlanner(policy)

    plan = planner.plan_sessions(
        axis=ChunkAxis.TIME,
        session_count=20,
        estimated_bytes_per_session=100_000,
    )

    assert policy.memory_budget_fraction == 0.80
    assert plan.chunk_count > 1
    assert "calendar" not in ResourceChunkPlanner.plan_sessions.__annotations__
    assert "sessions" not in ResourceChunkPlanner.plan_sessions.__annotations__


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
