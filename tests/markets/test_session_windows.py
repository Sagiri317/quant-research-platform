from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets import session_windows
from quant_research_platform.markets.session_windows import (
    SessionWindowPlanner,
    normalize_session_dates,
)


def test_normalize_session_dates_returns_sorted_unique_daily_timestamps() -> None:
    sessions = normalize_session_dates(["2014-01-03 12:00", "2014-01-02", "2014-01-03"])

    assert isinstance(sessions, pd.DatetimeIndex)
    assert sessions.equals(pd.DatetimeIndex(("2014-01-02", "2014-01-03")))


def test_session_window_planner_owns_normalized_session_state() -> None:
    sessions = normalize_session_dates(pd.date_range("2014-01-01", periods=10))

    planner = SessionWindowPlanner(
        sessions,
        target_core_session_count=2,
    )

    assert planner.sessions is not sessions
    assert planner.sessions.equals(sessions)


def test_session_window_planner_isolated_from_mutation_before_windows() -> None:
    sessions = pd.date_range("2014-01-01", periods=3)
    planner = SessionWindowPlanner(
        sessions=sessions,
        target_core_session_count=1,
    )
    sessions._data._ndarray[0] = np.datetime64("2099-01-01")  # noqa: SLF001

    windows = tuple(
        planner.windows(
            TimeWindow("2014-01-01", "2014-01-03"),
            reason="research_assembly",
        )
    )

    assert tuple(window.core.start.isoformat() for window in windows) == (
        "2014-01-01",
        "2014-01-02",
        "2014-01-03",
    )


def test_session_window_iterator_isolated_from_mutation_between_pulls() -> None:
    sessions = pd.date_range("2014-01-01", periods=3)
    planner = SessionWindowPlanner(
        sessions,
        target_core_session_count=1,
    )
    windows = planner.windows(
        TimeWindow("2014-01-01", "2014-01-03"),
        reason="research_assembly",
    )
    sessions._data._ndarray[1:] = np.array(  # noqa: SLF001
        [np.datetime64("2099-01-02"), np.datetime64("2099-01-03")]
    )

    assert next(windows).core.start.isoformat() == "2014-01-01"
    sessions._data._ndarray[0] = np.datetime64("2099-01-01")  # noqa: SLF001
    assert tuple(window.core.start.isoformat() for window in windows) == (
        "2014-01-02",
        "2014-01-03",
    )


def test_session_window_planner_rejects_hostile_datetime_index_subclass() -> None:
    class AliasingArray(np.ndarray):
        def copy(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return self

    class DataProxy:
        def __init__(self, owner) -> None:  # noqa: ANN001
            self.owner = owner

        @property
        def _ndarray(self):  # noqa: ANN202
            return self.owner._ndarray.view(AliasingArray)  # noqa: SLF001

        def __getattr__(self, name: str):  # noqa: ANN204
            return getattr(self.owner, name)

    class AliasingDatetimeIndex(pd.DatetimeIndex):
        @property
        def _data(self):  # noqa: ANN202
            return DataProxy(self.__dict__["_hostile_data"])

        @_data.setter
        def _data(self, value) -> None:  # noqa: ANN001
            self.__dict__["_hostile_data"] = value

    sessions = AliasingDatetimeIndex(pd.date_range("2014-01-01", periods=3))

    with pytest.raises(ValueError, match="^invalid_session_dates$"):
        SessionWindowPlanner(
            sessions=sessions,
            target_core_session_count=1,
        )


@pytest.mark.parametrize(
    "sessions",
    (
        pd.DatetimeIndex([pd.NaT]),
        [None],
        [pd.NaT],
        [np.datetime64("NaT")],
        ["2014-01-02", pd.NaT],
        pd.DatetimeIndex(["2015-01-05"], tz="UTC"),
        [pd.Timestamp("2015-01-05", tz="Asia/Shanghai")],
    ),
)
def test_session_dates_reject_invalid_values(sessions: object) -> None:
    with pytest.raises(ValueError, match="invalid_session_dates"):
        normalize_session_dates(sessions)  # type: ignore[arg-type]


def test_session_window_planner_maps_non_trading_bounds_inside_closed_window() -> None:
    windows = tuple(
        SessionWindowPlanner(
            ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-10"],
            target_core_session_count=2,
        ).windows(TimeWindow("2014-01-04", "2014-01-11"), reason="research_assembly")
    )

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
    planner = SessionWindowPlanner(
        ["2014-01-02", "2014-01-03"],
        target_core_session_count=1,
    )

    with pytest.raises(ValueError, match="no_trading_sessions_in_time_window"):
        planner.windows(TimeWindow("2014-01-04", "2014-01-05"), reason="factor_warmup")


def test_session_window_planner_validates_at_call_time_and_returns_lazy_iterator() -> (
    None
):
    planner = SessionWindowPlanner(
        ["2014-01-02", "2014-01-03"],
        target_core_session_count=1,
    )

    windows = planner.windows(
        TimeWindow("2014-01-02", "2014-01-03"),
        reason="factor_warmup",
    )

    assert isinstance(windows, Iterator)
    assert next(windows).core_sessions == (pd.Timestamp("2014-01-02"),)


def test_session_window_planner_normalizes_input_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalize = session_windows.normalize_session_dates
    calls = 0

    def observe(values):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        return normalize(values)

    monkeypatch.setattr(session_windows, "normalize_session_dates", observe)

    planner = SessionWindowPlanner(
        ["2014-01-03", "2014-01-02"],
        target_core_session_count=1,
    )

    assert planner.sessions.equals(pd.DatetimeIndex(("2014-01-02", "2014-01-03")))
    assert calls == 1


def test_session_window_planner_covers_core_sessions_once() -> None:
    windows = tuple(
        SessionWindowPlanner(
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
    )

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
    windows = tuple(
        SessionWindowPlanner(
            ["2014-12-31", "2015-01-05"],
            target_core_session_count=2,
        ).windows(TimeWindow("2014-12-31", "2015-01-05"), reason="panel_next_session")
    )

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
    windows = tuple(
        SessionWindowPlanner(
            sessions,
            target_core_session_count=2,
            warmup_sessions=1,
            lookahead_sessions=1,
        ).windows(TimeWindow("2015-01-01", "2015-01-10"), reason="factor_warmup")
    )

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
    planner = SessionWindowPlanner(
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


def test_session_windows_owns_snapshot_for_iterator() -> None:
    planner = SessionWindowPlanner(
        ["2015-01-05", "2015-01-06"],
        target_core_session_count=1,
    )
    windows = planner.windows(TimeWindow("2015-01-05", "2015-01-06"), reason="factor")
    planner.sessions._data._ndarray[0] = pd.Timestamp("2099-01-01").to_datetime64()

    first = next(windows)

    assert first.core.start.isoformat() == "2015-01-05"


class _SessionOffsetIntSubclass(int):
    pass


@pytest.mark.parametrize("value", (True, 1.5, -1, _SessionOffsetIntSubclass(1)))
def test_session_offset_owner_requires_exact_non_negative_int(value: object) -> None:
    with pytest.raises(ValueError, match="^invalid_test_session_offset$"):
        session_windows.require_non_negative_session_offset(
            value, error="invalid_test_session_offset"
        )
