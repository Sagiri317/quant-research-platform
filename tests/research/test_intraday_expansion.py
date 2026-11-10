"""Opening snapshots and realized variation have independent hand references."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.intraday_context import (
    calculate,
    trade_profiles,
)
from quant_research_platform.research.factors.microstructure import compute_factors


def panel():
    times = pd.date_range("2024-01-02 01:30", periods=1201, freq="3s")
    return pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=times,
            segment_id=0,
            mid=np.exp(np.arange(len(times)) * 0.0001),
            observed_trade_volume=10.0,
            observed_signed_trade_volume=2.0,
        )
    )


def test_opening_path_freezes_at_observation_boundary_and_is_scale_invariant():
    frame = panel()
    names = (
        "opening_observed_return_5m",
        "opening_path_efficiency_5m",
        "opening_cumulative_flow_imbalance_5m",
    )
    result = compute_factors(frame, columns=names)
    assert result.loc[:99, list(names)].isna().all().all()
    assert result.loc[100, names[0]] == pytest.approx(np.exp(0.01) - 1)
    assert result.loc[100, names[1]] == pytest.approx(1)
    assert result.loc[100, names[2]] == pytest.approx(0.2)
    changed = frame.copy()
    changed.loc[101:, "mid"] *= 10
    pd.testing.assert_frame_equal(result, compute_factors(changed, columns=names))
    incomplete = frame.iloc[3:]
    assert compute_factors(incomplete, columns=names)[list(names)].isna().all().all()


def test_semivariance_and_bipower_use_observed_log_returns():
    frame = panel().iloc[:4].copy()
    frame["mid"] = np.exp([0, 0.01, -0.01, 0.02])
    names = (
        "hf_upside_realized_variance_1d",
        "hf_downside_realized_variance_1d",
        "hf_downside_variance_fraction_1d",
        "daily_continuous_variation",
        "daily_jump_variation_fraction",
    )
    last = compute_factors(frame, columns=names).iloc[-1]
    assert last[names[0]] == pytest.approx(0.01**2 + 0.03**2)
    assert last[names[1]] == pytest.approx(0.02**2)
    assert last[names[2]] == pytest.approx(0.0004 / 0.0014)
    assert last[names[3]] == pytest.approx(np.pi / 2 * (0.01 * 0.02 + 0.02 * 0.03))
    assert last[names[4]] == pytest.approx(max(1 - last[names[3]] / 0.0014, 0))
    first = compute_factors(frame.iloc[:2], columns=names).iloc[-1]
    assert pd.isna(first.daily_continuous_variation)
    assert pd.isna(first.daily_jump_variation_fraction)


def test_liquidity_recovery_needs_its_own_complete_baseline():
    frame = (
        panel()
        .iloc[:41]
        .assign(bid_px1=1.0, ask_px1=1.01, bid_vol1=100.0, ask_vol1=100.0)
    )
    frame.loc[20, ["ask_px1", "bid_vol1", "ask_vol1"]] = [1.04, 25.0, 25.0]
    frame.loc[30, ["ask_px1", "bid_vol1", "ask_vol1"]] = [1.02, 75.0, 75.0]
    names = ["observed_spread_recovery_30s", "observed_depth_recovery_30s"]
    last = compute_factors(frame, columns=names).iloc[30]
    assert last[names[0]] == pytest.approx(2 / 3)
    assert last[names[1]] == pytest.approx(2 / 3)
    frame.loc[10, "ask_px1"] = np.nan
    last = compute_factors(frame, columns=names).iloc[30]
    assert pd.isna(last[names[0]])
    assert last[names[1]] == pytest.approx(2 / 3)


def test_industry_catchup_excludes_self_and_unknown_calendar_does_not_gate_peers():
    blocks = []
    for asset, slope in [("a", 0.0001), ("b", 0.0002), ("c", 0.0003)]:
        block = panel().iloc[:41].assign(asset_id=asset)
        block["mid"] = np.exp(np.arange(len(block)) * 3 * slope) * 10
        block = block.assign(
            bid_px1=block.mid - 0.01,
            ask_px1=block.mid + 0.01,
            bid_vol1=100.0,
            ask_vol1=100.0,
            exchange="SSE",
            instrument_type="stock",
        )
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    context = pd.DataFrame(
        dict(
            asset_id=["a", "b", "c"],
            timestamp=pd.Timestamp("2024-01-02 01:00"),
            signal_session=pd.Timestamp("2023-12-29"),
            industry_standard_id="CSRC2012",
            industry_level1_code="C",
            close=10.0,
        )
    )
    names = [
        "industry_catchup_30s",
        "intraday_positive_return_fraction",
        "opening_relative_activity_5m",
    ]
    result = calculate(frame, columns=names, daily_context=context)
    row = result.loc[result.asset_id.eq("a")].iloc[-1]
    expected = (np.exp(0.0002 * 30) - 1 + np.exp(0.0003 * 30) - 1) / 2 - (
        np.exp(0.0001 * 30) - 1
    )
    assert row.industry_catchup_30s == pytest.approx(expected)
    assert row.intraday_positive_return_fraction == 1
    assert result.opening_relative_activity_5m.isna().all()
    future = context.assign(timestamp=pd.Timestamp("2024-01-02 02:00"))
    assert (
        calculate(frame, columns=names, daily_context=future)
        .industry_catchup_30s.isna()
        .all()
    )


def test_trade_profile_uses_recorded_volume_and_never_reads_future_events():
    quotes = panel().iloc[:4]
    trades = pd.DataFrame(
        dict(
            asset_id="a",
            timestamp=pd.to_datetime(
                ["2024-01-02 01:30:00", "2024-01-02 01:30:03", "2024-01-02 01:30:30"]
            ),
            source_row=[1, 2, 3],
            event_kind="trade",
            price=[1.0, 1.0, 100.0],
            volume=[10.0, 20.0, 999.0],
        )
    )
    name = "trade_profile_concentration_1d"
    result = trade_profiles(quotes, trades, columns=[name])
    np.testing.assert_allclose(result[name], 1)
    pd.testing.assert_frame_equal(
        result, trade_profiles(quotes, trades.iloc[:2], columns=[name])
    )
    unknown = trades.copy()
    unknown.loc[1, "event_kind"] = "unknown"
    changed = trade_profiles(quotes, unknown, columns=[name])
    assert changed[name].iloc[0] == 1
    assert changed[name].iloc[1:].isna().all()


def test_intraday_return_does_not_cross_overnight_when_segment_ids_repeat():
    one = (
        panel()
        .iloc[:21]
        .assign(
            bid_px1=0.99,
            ask_px1=1.01,
            bid_vol1=100.0,
            ask_vol1=100.0,
            exchange="SSE",
            instrument_type="stock",
            mid=1.0,
        )
    )
    two = one.assign(timestamp=one.timestamp + pd.Timedelta(days=1), mid=2.0)
    frame = pd.concat([one, two], ignore_index=True)
    result = calculate(frame, columns=["intraday_positive_return_fraction"])
    first = result.loc[result.timestamp.dt.date == pd.Timestamp("2024-01-03").date()]
    assert first.intraday_positive_return_fraction.iloc[:10].isna().all()
    assert first.intraday_positive_return_fraction.iloc[-1] == 0


def test_late_relative_volume_uses_twenty_declared_sessions_at_same_time():
    days = pd.bdate_range("2024-01-02", periods=21)
    blocks = []
    for i, date in enumerate(days):
        block = (
            panel()
            .iloc[:4]
            .assign(
                timestamp=date + pd.to_timedelta([5400, 24120, 24123, 24126], unit="s"),
                bid_px1=0.99,
                ask_px1=1.01,
                bid_vol1=100.0,
                ask_vol1=100.0,
                exchange="SSE",
                instrument_type="stock",
                observed_trade_volume=20.0 if i == 20 else 10.0,
            )
        )
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    name = "late_session_relative_activity"
    result = calculate(frame, columns=[name], session_days=days)
    assert result[name].iloc[:-4].isna().all()
    assert result[name].iloc[-1] == pytest.approx(2)
    missing = frame.loc[frame.timestamp.dt.normalize().ne(days[5])]
    assert calculate(missing, columns=[name], session_days=days)[name].isna().all()


def test_native_intraday_publication_does_not_require_unused_optional_inputs(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.test_prepare_selection import _input

    frame = (
        panel()
        .iloc[:21]
        .assign(
            bid_px1=0.99,
            ask_px1=1.01,
            bid_vol1=100.0,
            ask_vol1=100.0,
            exchange="SSE",
            instrument_type="stock",
        )
    )
    name = "intraday_positive_return_fraction"
    path = prepare(
        dict(
            market_id="a_share",
            data_id="hft_research_factors",
            inputs={"hft_panel": _input(tmp_path, "hft_panel", frame, native=True)},
            factor_columns=[name, "opening_relative_activity_5m"],
            time_window=dict(start="2024-01-02", end="2024-01-02"),
            output_root=str(tmp_path / "result"),
        )
    )
    result = read_dataset(path)
    assert result[name].iloc[-1] == 1
    assert result.opening_relative_activity_5m.isna().all()
