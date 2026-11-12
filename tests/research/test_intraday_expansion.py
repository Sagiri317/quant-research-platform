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


@pytest.mark.parametrize("direction", (1, -1))
def test_intraday_acceleration_has_quadratic_reference_and_no_future_prices(direction):
    frame = panel().iloc[:112].copy()
    x = np.arange(len(frame), dtype=float)
    frame["mid"] = 10 * np.exp(direction * (0.0001 * x + 0.000001 * x**2))
    names = [f"price_{side}_acceleration_30s" for side in ("up", "down")]
    result = compute_factors(frame, columns=names)
    selected = names[0 if direction > 0 else 1]
    assert result.loc[:9, names].isna().all(axis=None)
    # Eleven prices cover ten 3-second increments; log-price curvature is 2e-6.
    rms = np.sqrt(np.mean((0.0001 + 0.000001 * (2 * np.arange(1, 11) - 1)) ** 2))
    assert result.loc[10, selected] == pytest.approx(2e-6 / rms, rel=1e-8)
    assert result.loc[10, names[1 if direction > 0 else 0]] == 0
    prefix = compute_factors(frame.iloc[:31], columns=names)
    pd.testing.assert_frame_equal(result.iloc[:31].reset_index(drop=True), prefix)
    np.testing.assert_allclose(
        result[names],
        compute_factors(frame.assign(mid=frame.mid * 5), columns=names)[names],
        equal_nan=True,
        atol=1e-10,
    )
    missing = frame.drop(index=range(3, 7))
    assert compute_factors(missing, columns=names).loc[6, names].isna().all()
    assert (
        compute_factors(frame.assign(mid=1.0), columns=names)[names]
        .isna()
        .all(axis=None)
    )


@pytest.mark.parametrize("direction", (1, -1))
def test_price_flow_divergence_keeps_one_signed_expression(direction):
    frame = panel().iloc[:12].copy()
    frame["mid"] = np.exp(direction * np.arange(len(frame)) * 0.001)
    frame["observed_signed_trade_volume"] = -direction * 2.0
    name = "price_flow_divergence_30s"
    result = compute_factors(frame, columns=(name,))
    assert result[name].iloc[10] == pytest.approx(direction * 0.01 * 0.2)
    frame["observed_signed_trade_volume"] *= -1
    assert compute_factors(frame, columns=(name,))[name].iloc[10] == 0


def test_drawdown_activity_recovery_uses_prior_known_context_and_local_missingness():
    frame = (
        panel()
        .iloc[:112]
        .assign(
            bid_px1=0.99,
            ask_px1=1.01,
            bid_vol1=100.0,
            ask_vol1=100.0,
            exchange="SSE",
            instrument_type="stock",
        )
    )
    frame.loc[90, ["bid_vol1", "ask_vol1"]] = 25.0
    frame.loc[101:, "observed_trade_volume"] = 20.0
    context = pd.DataFrame(
        dict(
            asset_id=["a"],
            timestamp=pd.Timestamp("2024-01-02 01:00"),
            signal_session=pd.Timestamp("2023-12-29"),
            peak_drawdown_20d=0.2,
            risk_residual_drawdown_20d=0.1,
            cn_fund_holder_concentration=0.6,
        )
    )
    names = [
        "prior_drawdown_x_activity_x_depth_recovery",
        "prior_residual_drawdown_x_activity_x_depth_recovery",
    ]
    result = calculate(frame, columns=names, daily_context=context)
    np.testing.assert_allclose(result.loc[110, names].to_numpy(float), [0.4, 0.2])
    missing = calculate(
        frame,
        columns=names,
        daily_context=context.drop(columns="risk_residual_drawdown_20d"),
    )
    assert missing[names[0]].iloc[110] == pytest.approx(0.4)
    assert missing[names[1]].isna().all()
    future = context.assign(timestamp=pd.Timestamp("2024-01-02 02:00"))
    assert (
        calculate(frame, columns=names, daily_context=future)[names]
        .isna()
        .all(axis=None)
    )
    name = "prior_ownership_crowding_x_intraday_liquidity_stress"
    assert calculate(frame, columns=[name], daily_context=context)[name].iloc[90] == 0.6
    assert calculate(frame, columns=[name])[name].isna().all()


def test_current_day_quote_profiles_use_five_trade_days_without_mixing_daily_volume():
    days = pd.bdate_range("2024-01-02", periods=5)
    quotes = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[days[-1] + pd.Timedelta(hours=1, minutes=30, seconds=1)],
            mid=10.0,
        )
    )
    trades = pd.DataFrame(
        dict(
            asset_id="a",
            timestamp=days + pd.Timedelta(hours=1, minutes=30),
            source_row=np.arange(5),
            event_kind="trade",
            price=[11.0, 11.0, 11.0, 11.0, 10.0],
            volume=10.0,
        )
    )
    columns = ["trade_profile_concentration_1d", "trade_profile_concentration_5d"]
    result = trade_profiles(quotes, trades, columns=columns, session_days=days)
    np.testing.assert_allclose(result[columns].to_numpy(float), [[1.0, 0.8]])
    trades.loc[2, "event_kind"] = "unknown"
    partial = trade_profiles(quotes, trades, columns=columns, session_days=days)
    assert partial[columns[0]].iloc[0] == 1
    assert pd.isna(partial[columns[1]].iloc[0])


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


def test_batched_path_factors_keep_asset_and_segment_coordinates():
    x = np.arange(111, dtype=float)
    blocks = []
    for asset, direction in (("a", 1), ("b", -1)):
        for segment, hour in enumerate((1, 5)):
            blocks.append(
                pd.DataFrame(
                    dict(
                        market_id="a_share",
                        asset_id=asset,
                        timestamp=pd.date_range(
                            f"2024-01-02 {hour:02d}:30", periods=len(x), freq="3s"
                        ),
                        segment_id=segment,
                        mid=(10 + 10 * segment)
                        * np.exp(direction * (0.0001 * x + 0.000001 * x**2)),
                    )
                )
            )
    frame = pd.concat(blocks, ignore_index=True).iloc[::-1]
    names = (
        "opening_observed_return_5m",
        "price_up_acceleration_30s",
        "hf_upside_realized_variance_1d",
    )
    result = compute_factors(frame, columns=names)
    assert list(result) == ["market_id", "asset_id", "timestamp", *names]
    increments = 0.0001 + 0.000001 * (2 * np.arange(1, len(x)) - 1)
    curvature = 2e-6 / np.sqrt(np.mean(increments[:10] ** 2))
    for asset, direction in (("a", 1), ("b", -1)):
        own = result.loc[result.asset_id.eq(asset)].reset_index(drop=True)
        assert own[names[0]].iloc[-1] == pytest.approx(np.expm1(direction * 0.02))
        assert own[names[1]].iloc[111:121].isna().all()
        assert own[names[1]].iloc[121] == pytest.approx(
            curvature if direction > 0 else 0, abs=1e-10
        )
        # The afternoon repricing is not a within-segment observed return.
        assert own[names[2]].iloc[-1] == pytest.approx(
            2 * np.sum(increments**2) if direction > 0 else 0
        )


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


def test_intraday_breadth_has_numeric_partial_history_and_hand_counted_stress():
    blocks = []
    for asset in ("a", "b"):
        block = (
            panel()
            .iloc[:22]
            .assign(
                asset_id=asset,
                bid_px1=1.0,
                ask_px1=1.01,
                bid_vol1=100.0,
                ask_vol1=100.0,
            )
        )
        if asset == "a":
            block.loc[20:, ["ask_px1", "bid_vol1", "ask_vol1"]] = [1.04, 25.0, 25.0]
        blocks.append(block)
    names = [
        "intraday_positive_return_fraction",
        "intraday_positive_flow_fraction",
        "intraday_liquidity_stress_fraction",
    ]
    result = calculate(pd.concat(blocks, ignore_index=True), columns=names)
    assert all(pd.api.types.is_numeric_dtype(result[name]) for name in names)
    early = result.timestamp.lt(result.timestamp.min() + pd.Timedelta(seconds=60))
    assert result.loc[early, names[-1]].isna().all()
    last = result.loc[result.timestamp.eq(result.timestamp.max()), names]
    np.testing.assert_allclose(last.to_numpy(float), [[1, 1, 0.5], [1, 1, 0.5]])


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


def test_trade_profiles_group_each_assets_own_recorded_trades():
    quotes = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["b", "missing", "a"],
            timestamp=pd.Timestamp("2024-01-02 01:30:03"),
            mid=[20.0, 10.0, 11.0],
        )
    )
    trades = pd.DataFrame(
        dict(
            asset_id=["a", "b", "a", "b", "b"],
            timestamp=pd.to_datetime(
                ["2024-01-02 01:30:03", "2024-01-02 01:30:00"] * 2
                + ["2024-01-02 01:30:02"]
            ),
            source_row=[2, 1, 1, 2, 3],
            event_kind=["trade", "trade", "trade", "trade", "cancel"],
            price=[11.0, 20.0, 10.0, 20.0, 21.0],
            volume=[300.0, 50.0, 100.0, 50.0, 1000.0],
        )
    )
    name = "trade_profile_concentration_1d"
    result = trade_profiles(quotes, trades, columns=[name])
    # A has 300/400 in one bin; B's two trades share a bin, excluding cancellation.
    np.testing.assert_allclose(result[name], [1.0, np.nan, 0.75], equal_nan=True)
    pd.testing.assert_frame_equal(result.drop(columns=name), quotes.drop(columns="mid"))


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


def test_native_intraday_skips_unobserved_day_with_history_prefix(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_sources import _calendar_input
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
    first = frame.iloc[:8]
    second = frame.assign(timestamp=frame.timestamp + pd.Timedelta(days=1))
    source = pd.concat([first, second], ignore_index=True)
    name = "intraday_positive_return_fraction"
    path = prepare(
        dict(
            market_id="a_share",
            data_id="hft_research_factors",
            inputs=dict(
                hft_panel=_input(tmp_path, "hft_panel", source, native=True),
                trading_calendar=str(
                    _calendar_input(
                        tmp_path, {1: pd.bdate_range("2024-01-02", periods=3)}
                    )
                ),
            ),
            factor_columns=[name, "opening_relative_activity_5m"],
            time_window=dict(start="2024-01-02", end="2024-01-04"),
            output_root=str(tmp_path / "fractions"),
        )
    )
    result = read_dataset(path)
    assert result.loc[result.timestamp.dt.day.eq(2), name].isna().all()
    assert result[name].iloc[-1] == 1
    assert not result.timestamp.dt.day.eq(4).any()


def test_native_intraday_keeps_complete_cross_sections_across_reader_batches(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.runtime.dataframes import iter_complete_time_frames
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.reading import (
        iter_dataset_chunks,
        open_dataset_session,
    )
    from tests.research.test_prepare_selection import _input

    blocks = []
    for asset, direction in (("a", 1), ("b", -1)):
        for offset in (0, 1):
            block = (
                panel()
                .iloc[:12]
                .assign(
                    asset_id=asset,
                    bid_px1=0.99,
                    ask_px1=1.01,
                    bid_vol1=100.0,
                    ask_vol1=100.0,
                    exchange="SSE",
                    instrument_type="stock",
                )
            )
            block["mid"] = np.exp(direction * np.arange(len(block)) * 0.001)
            block["timestamp"] = (block.timestamp + pd.Timedelta(days=offset)).astype(
                "datetime64[ns]"
            )
            blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    name = "intraday_positive_return_fraction"
    path = prepare(
        dict(
            market_id="a_share",
            data_id="hft_research_factors",
            inputs={"hft_panel": _input(tmp_path, "hft_panel", frame, native=True)},
            factor_columns=[name],
            time_window=dict(start="2024-01-02", end="2024-01-03"),
            output_root=str(tmp_path / "result"),
        )
    )
    result = read_dataset(path)
    assert result.timestamp.is_monotonic_increasing
    keys = ["market_id", "asset_id", "timestamp"]
    pd.testing.assert_frame_equal(
        result[keys].sort_values(keys).reset_index(drop=True),
        frame[keys].sort_values(keys).reset_index(drop=True),
    )
    # One rising and one falling price path give breadth one half after 30s.
    assert result[name].notna().sum() == 8
    np.testing.assert_allclose(result[name].dropna(), 0.5)
    session = open_dataset_session(path)
    recovered = pd.concat(
        iter_complete_time_frames(
            iter_dataset_chunks(session, batch_rows=7), timestamp_column="timestamp"
        ),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(recovered, result)


@pytest.mark.parametrize(
    ("name", "inputs", "expected"),
    (
        ("intraday_positive_return_fraction", ("mid",), 1.0),
        (
            "intraday_positive_flow_fraction",
            ("observed_trade_volume", "observed_signed_trade_volume"),
            1.0,
        ),
        (
            "intraday_liquidity_stress_fraction",
            ("bid_px1", "ask_px1", "bid_vol1", "ask_vol1"),
            0.0,
        ),
        ("opening_gap_repair_fraction_5m", ("mid",), -np.expm1(0.01)),
    ),
)
def test_selected_intraday_context_uses_only_its_observation_inputs(
    name, inputs, expected
):
    frame = (
        panel()
        .iloc[:111]
        .assign(bid_px1=0.99, ask_px1=1.01, bid_vol1=100.0, ask_vol1=100.0)
    )
    frame = frame.loc[:, ["market_id", "asset_id", "timestamp", "segment_id", *inputs]]
    daily = pd.DataFrame(
        dict(
            asset_id=["a"],
            timestamp=pd.Timestamp("2024-01-02 01:00"),
            signal_session=pd.Timestamp("2023-12-29"),
            close=0.5,
        )
    )
    result = calculate(frame, columns=[name], daily_context=daily)
    # Positive price/flow histories give breadth one; a constant book has no
    # shock. Opening price is 1 against prior close .5, so repair is -return.
    assert result[name].iloc[-1] == pytest.approx(expected)
    assert pd.isna(result[name].iloc[0])
    assert list(result) == ["market_id", "asset_id", "timestamp", name]
