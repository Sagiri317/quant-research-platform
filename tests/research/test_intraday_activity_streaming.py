"""Daily activity aggregation retains full session clocks and observed cutoffs."""

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import intraday
from quant_research_platform.research.factors.intraday_context import (
    calculate,
    calculate_activity,
)
from tests.helpers.a_share_sources import _calendar_input
from tests.research.test_prepare_selection import _input

ACTIVITY = (
    "opening_relative_activity_5m",
    "opening_relative_activity_15m",
    "opening_relative_activity_30m",
    "late_session_relative_activity",
)


def _observations():
    days = pd.bdate_range("2024-01-02", periods=24).as_unit("ns")
    pieces = []
    for i, day in enumerate(days):
        for asset, first, close in (("a", 0, "6h57min"), ("b", 3, "7h")):
            if i < first or (asset == "a" and i == 5):
                continue
            times = day + pd.to_timedelta(
                [*(90 + np.arange(31)), pd.Timedelta(close).total_seconds() / 60 - 15],
                unit="min",
            )
            times = times.append(
                pd.DatetimeIndex([times[-1] + pd.Timedelta(seconds=s) for s in (3, 6)])
            )
            pieces.append(
                pd.DataFrame(
                    dict(
                        market_id="a_share",
                        asset_id=asset,
                        timestamp=times,
                        observed_trade_volume=float(i + 1),
                        observed_signed_trade_volume=0.0,
                        mid=10 + np.arange(len(times)) / 1000,
                        segment_id=f"{asset}:{i}",
                        bid_px1=9.99,
                        ask_px1=10.01,
                        bid_vol1=100.0,
                        ask_vol1=100.0,
                        exchange="SZSE" if asset == "a" else "SSE",
                        instrument_type="stock" if asset == "a" else "equity_etf",
                    )
                )
            )
    return days, pd.concat(pieces, ignore_index=True)


def test_activity_daily_totals_keep_missing_sessions_and_late_listing():
    days, frame = _observations()
    history = None
    outputs = []
    for day in days:
        current = frame.loc[frame.timestamp.dt.normalize().eq(day)]
        value, history = calculate_activity(
            current, columns=ACTIVITY, session_days=days, history=history
        )
        outputs.append(value)
    actual = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["asset_id", "timestamp"])
        .reset_index(drop=True)
    )
    expected = calculate(frame, columns=ACTIVITY, session_days=days)
    pd.testing.assert_frame_equal(actual, expected)
    # B starts on session 3, so session 23 first has twenty prior observed days.
    last = actual.loc[
        actual.asset_id.eq("b") & actual.timestamp.dt.normalize().eq(days[-1])
    ]
    for name in ACTIVITY:
        assert last[name].dropna().eq(24 / np.mean(np.arange(4.0, 24.0))).all()
        assert last[name].notna().any()
    # A's missing session 5 remains missing in the twenty-session reference.
    assert actual.loc[actual.asset_id.eq("a"), list(ACTIVITY)].isna().all(axis=None)
    assert min(key[1] for key in history[0]) >= days[-20]
    assert history[1].index.get_level_values("day").min() >= days[-20]


def test_native_activity_reads_prefix_once_and_respects_mixed_factor_cutoff(
    tmp_path, monkeypatch
):
    days, frame = _observations()
    inputs = {
        "hft_panel": _input(tmp_path, "hft_panel", frame, native=True),
        "trading_calendar": _calendar_input(tmp_path, {1: days}),
    }
    columns = (*ACTIVITY, "intraday_positive_return_fraction")
    base = intraday.processed_spec()
    spec = replace(
        base,
        column_roles={
            **dict.fromkeys(intraday.KEY, "key"),
            "timestamp": "timestamp",
            **dict.fromkeys(columns, "factor"),
        },
        column_units={name: intraday.UNITS[name] for name in columns},
        parameters_json="{}",
    )
    reads = []
    original = intraday.materialize_dataset_session

    def observe(session):
        value = original(session)
        if session.spec["data_id"] == "hft_panel":
            reads.append(value)
        return value

    monkeypatch.setattr(intraday, "materialize_dataset_session", observe)
    start = days[20] + pd.Timedelta(hours=1, minutes=32)
    end = days[-1] + pd.Timedelta(hours=1, minutes=40, seconds=30)
    with intraday.produce(spec, inputs=inputs, time_window=TimeWindow(start, end)) as (
        frames,
        _,
    ):
        actual = pd.concat(frames, ignore_index=True)
    consumed = frame.loc[frame.timestamp.le(end)]
    expected = calculate(consumed, columns=columns, session_days=days)
    expected = expected.loc[expected.timestamp.between(start, end)].sort_values(
        ["timestamp", "asset_id"], ignore_index=True
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert sum(len(read) for read in reads) == len(consumed)
    assert set(reads[0]) == {
        *intraday.KEY,
        "observed_trade_volume",
        "exchange",
        "instrument_type",
    }
    final = actual.loc[actual.timestamp.dt.normalize().eq(days[-1])]
    assert final[[ACTIVITY[1], ACTIVITY[2], ACTIVITY[3]]].isna().all(axis=None)


def test_activity_missing_flows_are_local_and_unknown_calendar_retains_no_state():
    days, frame = _observations()
    broken = frame.copy()
    missing = broken.asset_id.eq("b") & broken.timestamp.eq(
        days[-1] + pd.Timedelta(hours=1, minutes=31)
    )
    broken.loc[missing, "observed_trade_volume"] = np.nan
    actual, _ = calculate_activity(broken, columns=ACTIVITY, session_days=days)
    final = actual.loc[
        actual.asset_id.eq("b") & actual.timestamp.dt.normalize().eq(days[-1])
    ]
    assert final[list(ACTIVITY[:3])].isna().all(axis=None)
    assert final[ACTIVITY[3]].dropna().eq(24 / np.mean(np.arange(4.0, 24.0))).all()
    for axis in (None, pd.DatetimeIndex([])):
        unknown, history = calculate_activity(
            frame, columns=ACTIVITY, session_days=axis
        )
        assert unknown[list(ACTIVITY)].isna().all(axis=None)
        assert history == ({}, None)
