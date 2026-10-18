"""Public native computations consume only their own evidenced calendar halos."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.producer import (
    _calendar_panel_axis,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.factors.engine import factor_lookback_bars
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_LIBRARY,
    PRICE_VOLUME_PATH_FACTORS,
    price_volume_factor_definitions,
)
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.storage.datasets import read_dataset, write_dataset


def _panel(dates, asset="a", market=1):
    price = 100.0 + np.arange(len(dates))
    return pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=asset,
            timestamp=pd.DatetimeIndex(dates).as_unit("ns"),
            listing_date=pd.Timestamp("2000-01-01"),
            market_type=market,
            adj_open=price,
            adj_close=price,
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            limit_status=0,
            is_trading=True,
            filling_flag=0,
            period_return=0.01,
            amount=1e8,
        )
    )


def _calendar(sessions_by_market):
    """Write exactly the supplied O/C range, without automatic historical padding."""
    frames = []
    for market, sessions in sessions_by_market.items():
        sessions = pd.DatetimeIndex(sessions)
        days = pd.date_range(sessions.min(), sessions.max()).as_unit("ns")
        opened = days.isin(sessions)
        frames.append(
            pd.DataFrame(
                dict(
                    timestamp=days,
                    calendar_date=days,
                    source_market_type=market,
                    is_trading_day=opened,
                    source_day_of_week=days.dayofweek,
                    source_state=np.where(opened, "O", "C"),
                )
            )
        )
    return pd.concat(frames, ignore_index=True)


def _prepare(root, panel, calendar, target, start=None, end=None, **options):
    spec = processed_catalog().get("market_panel_1d")
    spec = replace(
        spec,
        partition_columns=(),
        column_roles={
            column: spec.column_roles.get(column, "context") for column in panel
        },
    )
    source, clock = root / "panel", root / "calendar"
    write_dataset(source, panel, spec=spec)
    write_dataset(clock, calendar, spec=interim_catalog().get("trading_calendar"))
    output = prepare(
        dict(
            market_id="a_share",
            data_id=target,
            inputs={"market_panel_1d": str(source), "trading_calendar": str(clock)},
            time_window={
                "start": str(
                    pd.Timestamp(
                        start if start is not None else panel.timestamp.min()
                    ).date()
                ),
                "end": str(
                    pd.Timestamp(
                        end if end is not None else panel.timestamp.max()
                    ).date()
                ),
            },
            output_root=str(root / "output"),
            **options,
        )
    )
    return read_dataset(output)


@pytest.mark.parametrize("unknown_column", ("listing_date", "market_type"))
def test_unknown_membership_execution_is_local_and_cannot_skip_session(
    tmp_path, unknown_column
):
    dates = pd.bdate_range("2024-01-02", periods=5)
    a, b = _panel(dates), _panel(dates, "b", 4)
    clock = _calendar({1: dates, 4: dates})
    control = _prepare(tmp_path / "control", pd.concat([a, b]), clock, "execution_mask")
    a.loc[2, unknown_column] = pd.NaT if unknown_column == "listing_date" else np.nan
    result = _prepare(tmp_path / "unknown", pd.concat([a, b]), clock, "execution_mask")
    pd.testing.assert_frame_equal(
        result.loc[result.asset_id.eq("b")].reset_index(drop=True),
        control.loc[control.asset_id.eq("b")].reset_index(drop=True),
        check_exact=True,
    )
    current = result.loc[result.asset_id.eq("a")].reset_index(drop=True)
    for column in ("can_enter", "can_exit"):
        assert current[column].iloc[[0, 3]].all()
        assert current[column].iloc[[1, 2, 4]].isna().all()
    direct = derive_execution_mask(pd.concat([a, b]))
    keys = ["market_id", "asset_id", "timestamp"]
    pd.testing.assert_frame_equal(
        result.sort_values(keys).reset_index(drop=True), direct, check_exact=True
    )


@pytest.mark.parametrize("disclosed,effective", ((4, 4), (6, 4), (2, 5)))
def test_known_exit_ends_only_its_own_axis_without_rewriting_prefix(
    tmp_path, disclosed, effective
):
    dates = pd.bdate_range("2024-01-02", periods=9)
    terminal = max(disclosed, effective)
    a = _panel(dates[: terminal + 1]).assign(delisting_date=pd.NaT)
    a.loc[disclosed:, "delisting_date"] = dates[effective]
    b = _panel(dates, "b", 4).assign(delisting_date=pd.NaT)
    panel = pd.concat([a, b], ignore_index=True)
    clock = _calendar({1: dates[: terminal + 1], 4: dates})
    axis = _calendar_panel_axis(panel, clock)
    assert (
        axis.loc[axis.asset_id.eq("a"), "timestamp"].tolist()
        == dates[: terminal + 1].tolist()
    )
    assert axis.loc[axis.asset_id.eq("b"), "timestamp"].tolist() == dates.tolist()
    before = panel.loc[panel.timestamp.lt(dates[disclosed])]
    prefix = _calendar_panel_axis(before, clock)
    pd.testing.assert_frame_equal(
        axis.loc[axis.timestamp.lt(dates[disclosed])].reset_index(drop=True),
        prefix.reset_index(drop=True),
        check_exact=True,
    )
    labels = _prepare(
        tmp_path / "labels",
        panel,
        clock,
        "forward_returns",
        label_columns=["forward_return_1d"],
    )
    exited = labels.loc[labels.asset_id.eq("a")]
    assert exited.forward_return_1d.iloc[-2:].isna().all()
    assert exited.forward_return_1d.iloc[:-2].notna().all()
    control = _prepare(
        tmp_path / "control",
        b,
        _calendar({4: dates}),
        "forward_returns",
        label_columns=["forward_return_1d"],
    )
    independent = labels.loc[labels.asset_id.eq("b")].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        independent.drop(columns="forward_return_1d"),
        control.drop(columns="forward_return_1d"),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        independent.forward_return_1d,
        control.forward_return_1d,
        check_exact=False,
        rtol=1e-13,
        atol=0.0,
    )


def test_announced_exit_on_closed_day_does_not_create_session():
    dates = pd.bdate_range("2024-01-01", periods=10)
    a = _panel(dates[:5]).assign(delisting_date=pd.NaT)
    a.loc[2:, "delisting_date"] = pd.Timestamp("2024-01-06")
    b = _panel(dates, "b", 4).assign(delisting_date=pd.NaT)
    clock = _calendar({1: dates[:6], 4: dates})
    axis = _calendar_panel_axis(pd.concat([a, b]), clock)
    assert axis.loc[axis.asset_id.eq("a"), "timestamp"].tolist() == dates[:5].tolist()


@pytest.mark.parametrize("missing", (0, 5, 6))
@pytest.mark.parametrize("calendar_sessions", (20, 21))
def test_single_day_factor_uses_only_required_history_but_keeps_warmup_and_fresh(
    tmp_path, missing, calendar_sessions
):
    dates = pd.bdate_range("2024-01-02", periods=60)
    panel = _panel(dates, market=4).assign(amount=10.0)
    panel.loc[panel.index[-20 : -20 + missing] if missing else [], "amount"] = np.nan
    options = dict(
        target="price_volume_facts",
        start=dates[-1],
        end=dates[-1],
        factor_columns=["avg_amount_20d"],
    )
    result = _prepare(
        tmp_path / "short-calendar",
        panel,
        _calendar({4: dates[-calendar_sessions:]}),
        **options,
    )
    control = _prepare(
        tmp_path / "full-calendar", panel, _calendar({4: dates}), **options
    )
    pd.testing.assert_frame_equal(
        result.drop(columns="avg_amount_20d"),
        control.drop(columns="avg_amount_20d"),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        result.avg_amount_20d,
        control.avg_amount_20d,
        check_exact=False,
        rtol=1e-13,
        atol=0.0,
    )
    assert pd.notna(result.avg_amount_20d.iloc[0]) == (missing <= 5)
    if missing <= 5:
        assert result.avg_amount_20d.iloc[0] == 10.0
    short_panel = panel.tail(19)
    immature = _prepare(
        tmp_path / "warmup", short_panel, _calendar({4: dates[-21:]}), **options
    )
    assert immature.avg_amount_20d.isna().all()


@pytest.mark.parametrize("missing_offset", (-10, -11))
def test_different_market_windows_do_not_demand_each_others_history(
    tmp_path, missing_offset
):
    dates = pd.bdate_range("2024-01-02", periods=81)
    slow = dates[::2]
    panel = pd.concat(
        [
            _panel(slow).assign(amount=10.0),
            _panel(dates, "b", 4).assign(amount=20.0),
        ]
    )
    clock = _calendar({1: slow[-21:], 4: dates[-21:]})
    options = dict(
        target="price_volume_facts",
        start=dates[-1],
        end=dates[-1],
        factor_columns=["avg_amount_20d"],
    )
    result = _prepare(tmp_path / "different", panel, clock, **options)
    assert result.avg_amount_20d.tolist() == [10.0, 20.0]
    missing = clock.loc[
        ~(
            clock.source_market_type.eq(1)
            & clock.calendar_date.eq(dates[missing_offset])
        )
    ]
    with pytest.raises(ValueError, match="calendar_coverage_missing"):
        _prepare(tmp_path / "missing-required", panel, missing, **options)


@pytest.mark.parametrize(
    "unused_amount", (1e8, np.nextafter(0.0, 1.0), np.finfo(float).max)
)
@pytest.mark.parametrize("valid_return,expected", ((0.01, 1e-10), (0.0, 0.0)))
def test_invalid_amihud_pair_cannot_set_valid_reduction_scale(
    tmp_path, unused_amount, valid_return, expected
):
    dates = pd.bdate_range("2024-01-02", periods=20)
    panel = _panel(dates).assign(period_return=valid_return)
    panel.loc[19, ["period_return", "amount"]] = [np.nan, unused_amount]
    direct = calculate_price_volume_factors(panel, factor_columns=("amihud_20d",))
    native = _prepare(
        tmp_path,
        panel,
        _calendar({1: dates}),
        "price_volume_facts",
        factor_columns=["amihud_20d"],
    )
    assert direct.amihud_20d.iloc[-1] == expected
    assert native.amihud_20d.iloc[-1] == expected


def test_requested_observation_after_known_exit_remains_invalid(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=8)
    panel = _panel(dates).assign(delisting_date=pd.NaT)
    panel.loc[4:, "delisting_date"] = dates[4]
    with pytest.raises(ValueError, match="a_share_input_key_outside_panel"):
        _prepare(
            tmp_path,
            panel,
            _calendar({1: dates}),
            "forward_returns",
            label_columns=["forward_return_1d"],
        )


@pytest.mark.parametrize("factor", ("momentum_20d", "amount_shock_20d"))
def test_endpoint_and_shifted_windows_still_require_prior_twenty_sessions(
    tmp_path, factor
):
    dates = pd.bdate_range("2024-01-02", periods=60)
    panel = _panel(dates, market=4)
    options = dict(
        target="price_volume_facts",
        start=dates[-1],
        end=dates[-1],
        factor_columns=[factor],
    )
    result = _prepare(
        tmp_path / "complete", panel, _calendar({4: dates[-21:]}), **options
    )
    assert result[factor].notna().all()
    with pytest.raises(ValueError, match="calendar_coverage_missing"):
        _prepare(
            tmp_path / "missing-endpoint", panel, _calendar({4: dates[-20:]}), **options
        )


@pytest.mark.parametrize("factor", (*PRICE_VOLUME_LIBRARY, *PRICE_VOLUME_PATH_FACTORS))
def test_every_price_volume_definition_uses_its_actual_minimum_window(tmp_path, factor):
    dates = pd.bdate_range("2023-01-02", periods=320)
    x = np.arange(len(dates), dtype=float)
    panel = _panel(dates, market=4).assign(
        period_return=0.015 * np.sin(x / 3),
        amount=1e8 * (1 + 0.2 * np.cos(x / 7)),
        volume=1e6 + 5 * x,
        turnover=0.02 + 0.001 * np.sin(x / 5),
        float_market_cap=(100 + x) * 1e7,
        total_market_cap=(100 + x) * 2e7,
    )
    definitions = price_volume_factor_definitions(
        market_id=MarketId("a_share"),
        required_input_data_id="market_panel_1d",
        factor_columns=(factor,),
    )
    lookback = factor_lookback_bars(definitions)
    full = calculate_price_volume_factors(panel, factor_columns=(factor,))[factor].iloc[
        -1
    ]
    minimum = calculate_price_volume_factors(
        panel.tail(lookback + 1), factor_columns=(factor,)
    )[factor].iloc[-1]
    assert np.isfinite(full)
    assert minimum == pytest.approx(full, rel=1e-12, abs=1e-15)
    if lookback:
        short = calculate_price_volume_factors(
            panel.tail(lookback), factor_columns=(factor,)
        )
        assert short[factor].isna().all()
    native = _prepare(
        tmp_path,
        panel,
        _calendar({4: dates[-lookback - 1 :]}),
        "price_volume_facts",
        start=dates[-1],
        end=dates[-1],
        factor_columns=[factor],
    )
    assert native[factor].iloc[0] == pytest.approx(full, rel=1e-12, abs=1e-15)


def test_narrow_factor_keeps_market_boundary_state_without_older_calendar(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=60)
    panel = _panel(dates).assign(amount=10.0)
    panel.loc[50:, "market_type"] = 4
    result = _prepare(
        tmp_path,
        panel,
        _calendar({1: dates[40:50], 4: dates[50:]}),
        "price_volume_facts",
        start=dates[-1],
        end=dates[-1],
        factor_columns=["avg_amount_20d"],
    )
    assert result.avg_amount_20d.tolist() == [10.0]


def test_narrow_factor_unknown_barrier_never_borrows_earlier_warmup(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=60)
    a = _panel(dates).assign(amount=10.0)
    a.loc[40, "listing_date"] = pd.NaT
    b = _panel(dates, "b", 4).assign(amount=20.0)
    result = _prepare(
        tmp_path,
        pd.concat([a, b]),
        _calendar({1: dates[41:], 4: dates[-20:]}),
        "price_volume_facts",
        start=dates[-1],
        end=dates[-1],
        factor_columns=["avg_amount_20d"],
    )
    assert pd.isna(result.set_index("asset_id").loc["a", "avg_amount_20d"])
    assert result.set_index("asset_id").loc["b", "avg_amount_20d"] == 20.0


@pytest.mark.parametrize("unknown_column", ("listing_date", "market_type"))
@pytest.mark.parametrize(
    "disclosed,effective,relisted", ((2, 4, False), (2, 4, True), (6, 4, False))
)
def test_exit_knowledge_survives_unknown_clock_barrier_only_in_same_lifecycle(
    tmp_path, unknown_column, disclosed, effective, relisted
):
    dates = pd.bdate_range("2024-01-02", periods=8)
    terminal = max(disclosed, effective)
    a = _panel(dates[: terminal + 1]).assign(delisting_date=pd.NaT)
    a.loc[disclosed, "delisting_date"] = dates[effective]
    a.loc[3, unknown_column] = pd.NaT if unknown_column == "listing_date" else np.nan
    if relisted:
        reopened = _panel(dates[terminal + 1 :], market=4).assign(
            listing_date=dates[terminal + 1], delisting_date=pd.NaT
        )
        a = pd.concat([a, reopened], ignore_index=True)
    b = _panel(dates, "b", 4).assign(delisting_date=pd.NaT)
    panel = pd.concat([a, b], ignore_index=True)
    clock = _calendar({1: dates[: terminal + 1], 4: dates})
    axis = _calendar_panel_axis(panel, clock)
    own = axis.loc[axis.asset_id.eq("a")].set_index("timestamp")
    expected = dates if relisted else dates[: terminal + 1]
    assert own.index.sort_values().tolist() == expected.tolist()
    assert (
        own.loc[dates[2], "_calendar_segment"] != own.loc[dates[4], "_calendar_segment"]
    )
    if relisted:
        assert own.loc[dates[terminal + 1], "listing_date"] == dates[terminal + 1]
    for cutoff in (dates[disclosed], dates[4]):
        prefix = _calendar_panel_axis(panel.loc[panel.timestamp.lt(cutoff)], clock)
        pd.testing.assert_frame_equal(
            axis.loc[axis.timestamp.lt(cutoff)].reset_index(drop=True),
            prefix.reset_index(drop=True),
            check_exact=True,
        )
    result = _prepare(
        tmp_path / "original",
        panel,
        clock,
        "forward_returns",
        label_columns=["forward_return_1d"],
    )
    assert len(result) == len(panel)
    repeated = panel.copy()
    repeat = (
        repeated.asset_id.eq("a")
        & repeated.listing_date.eq(pd.Timestamp("2000-01-01"))
        & repeated.timestamp.ge(dates[disclosed])
    )
    repeated.loc[repeat, "delisting_date"] = dates[effective]
    control = _prepare(
        tmp_path / "repeated",
        repeated,
        clock,
        "forward_returns",
        label_columns=["forward_return_1d"],
    )
    pd.testing.assert_frame_equal(
        result.drop(columns="forward_return_1d"),
        control.drop(columns="forward_return_1d"),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        result.forward_return_1d,
        control.forward_return_1d,
        check_exact=False,
        rtol=1e-13,
        atol=0.0,
    )
    labels = result.loc[result.asset_id.eq("a")].set_index("timestamp")
    assert pd.isna(labels.loc[dates[2], "forward_return_1d"])
    assert pd.isna(labels.loc[dates[3], "forward_return_1d"])
    assert pd.isna(labels.loc[dates[terminal], "forward_return_1d"])
    if relisted:
        assert pd.notna(labels.loc[dates[terminal + 1], "forward_return_1d"])
