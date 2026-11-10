"""Observed history and full historical cohorts survive bounded output windows."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog,
    producer,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_interim_frames,
    _write_inputs,
)
from tests.markets.a_share.test_calendar_dependency_scope import _calendar, _panel


def test_observed_unknown_exit_halo():
    dates = pd.bdate_range("2024-01-02", periods=80).as_unit("ns")
    panel = _panel(dates).assign(delisting_date=pd.NaT, delisting_date_unresolved=True)
    axis = producer._calendar_panel_axis(
        panel,
        _calendar({1: dates}),
        time_window=TimeWindow(dates[70], dates[72]),
        lookback=60,
        lookahead=2,
    )
    assert axis.timestamp.tolist() == dates[10:75].tolist()
    assert axis._calendar_segment.nunique() == 1


@pytest.mark.parametrize("known_late", [False, True])
def test_halo_crosses_exit_knowledge_boundary(known_late):
    dates = pd.bdate_range("2024-01-02", periods=80).as_unit("ns")
    unknown = (np.arange(80) >= 30) & ((np.arange(80) < 70) if known_late else True)
    panel = _panel(dates).assign(
        delisting_date=pd.NaT, delisting_date_unresolved=unknown
    )
    axis = producer._calendar_panel_axis(
        panel,
        _calendar({1: dates}),
        time_window=TimeWindow(dates[70], dates[72]),
        lookback=60,
        lookahead=2,
    ).sort_values("timestamp")
    assert axis.timestamp.tolist() == dates[10:75].tolist()
    assert axis._calendar_segment.nunique() == 1


def _native_inputs(root):
    dates = pd.bdate_range("2020-01-02", periods=90).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    frames["stock_master"]["listing_date"] = dates[0]
    frames["stock_master"]["timestamp"] = dates[-1]
    frames["stock_master"]["status_change_date"] = dates[-1]
    inputs = _write_inputs(root / "source", frames)
    inputs["market_panel_1d"] = _prepare(
        root / "panel", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    return dates, inputs, read_dataset(inputs["market_panel_1d"])


@pytest.mark.parametrize(
    "target,options",
    [
        ("market_structure_factors", {}),
        ("price_volume_facts", {"factor_columns": ["reversal_5d"]}),
        ("forward_returns", {"label_columns": ["forward_return_1d"]}),
    ],
)
def test_native_subwindow_equals_full_history_slice(tmp_path, target, options):
    dates, inputs, panel = _native_inputs(tmp_path)
    spec = catalog.processed_catalog().get(target)
    dependencies = (*spec.source_input_data_ids, *spec.processed_input_data_ids)

    def run(name, first, last):
        return read_dataset(
            prepare(
                {
                    "market_id": "a_share",
                    "data_id": target,
                    "inputs": {k: str(inputs[k]) for k in dependencies if k in inputs},
                    "time_window": {
                        "start": str(first.date()),
                        "end": str(last.date()),
                    },
                    "output_root": str(tmp_path / name),
                    **options,
                }
            )
        )

    full = run("full", dates[0], dates[-1])
    narrow = run("narrow", dates[70], dates[72])
    expected = full.loc[full.timestamp.between(dates[70], dates[72])].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(narrow, expected, rtol=1e-10, atol=1e-12)
    if target == "market_structure_factors":
        assert narrow.market_above_ma60_fraction.eq(1).all()
    elif target == "price_volume_facts":
        a = panel.loc[panel.asset_id.eq("000001")].set_index("timestamp")
        assert narrow.loc[narrow.asset_id.eq("000001"), "reversal_5d"].iloc[
            0
        ] == pytest.approx(
            -(a.adj_close.loc[dates[70]] / a.adj_close.loc[dates[65]] - 1)
        )
    else:
        assert narrow.forward_return_1d.notna().all()


def test_structure_retains_members_that_left_inside_halo(tmp_path):
    from tests.markets.a_share.test_calendar_dependency_scope import _prepare as native

    dates = pd.bdate_range("2024-01-02", periods=15).as_unit("ns")
    current = _panel(dates).assign(is_a_share_scope=True, period_return=0.02)
    exited = _panel(dates[:8], asset="b").assign(
        is_a_share_scope=True,
        period_return=-0.01,
        delisting_date=dates[7],
        delisting_date_unresolved=False,
    )
    frame = pd.concat([current, exited], ignore_index=True)
    options = {"factor_columns": ["market_breadth_change_5d"]}
    full = native(
        tmp_path / "full",
        frame,
        _calendar({1: dates}),
        "market_structure_factors",
        **options,
    )
    narrow = native(
        tmp_path / "narrow",
        frame,
        _calendar({1: dates[5:11]}),
        "market_structure_factors",
        start=dates[10],
        end=dates[10],
        **options,
    )
    expected = full.loc[full.timestamp.eq(dates[10])].reset_index(drop=True)
    pd.testing.assert_frame_equal(narrow, expected, rtol=1e-12, atol=1e-14)
    assert narrow.market_breadth_change_5d.tolist() == [0.5]


@pytest.mark.parametrize(
    "barrier", ["observation", "calendar", "listing", "market", "relisting"]
)
def test_unknown_exit_real_breaks_survive_window_crop(barrier):
    dates = pd.bdate_range("2024-01-02", periods=80).as_unit("ns")
    panel = _panel(dates).assign(delisting_date=pd.NaT, delisting_date_unresolved=True)
    clock = _calendar({1: dates, 4: dates})
    if barrier in {"observation", "calendar"}:
        panel = panel.loc[panel.timestamp.ne(dates[68])]
        if barrier == "calendar":
            clock = clock.loc[clock.calendar_date.ne(dates[68])]
    elif barrier == "listing":
        panel.loc[68, "listing_date"] = pd.NaT
    elif barrier == "market":
        panel.loc[68, "market_type"] = np.nan
    else:
        panel.loc[68:, "listing_date"] = dates[68]
    axis = producer._calendar_panel_axis(
        panel, clock, time_window=TimeWindow(dates[70], dates[72]), lookback=60
    )
    own = axis.sort_values("timestamp").set_index("timestamp")
    if dates[67] in own.index:
        history = own[["listing_date", "_calendar_segment"]]
        assert not history.loc[dates[67]].equals(history.loc[dates[70]])
    assert own._calendar_segment.loc[dates[70] : dates[72]].nunique() == 1


def test_market_switch_preserves_evidenced_history_without_crossing_missing_open():
    dates = pd.bdate_range("2024-01-02", periods=8).as_unit("ns")
    panel = _panel(dates).assign(delisting_date=pd.NaT, delisting_date_unresolved=True)
    panel.loc[4:, "market_type"] = 4
    clock = _calendar({1: dates, 4: dates})
    axis = producer._calendar_panel_axis(panel, clock)
    assert axis._calendar_segment.nunique() == 1
    gap = producer._calendar_panel_axis(
        panel.loc[panel.timestamp.ne(dates[3])], clock
    ).set_index("timestamp")
    assert gap._calendar_segment.loc[dates[2]] != gap._calendar_segment.loc[dates[4]]


def test_structure_own_price_windows_do_not_expand_other_market_calendar(tmp_path):
    from tests.markets.a_share.test_calendar_dependency_scope import _prepare as native

    dates = pd.bdate_range("2024-01-02", periods=81).as_unit("ns")
    frame = pd.concat([_panel(dates[::2]), _panel(dates, "b", 4)]).assign(
        is_a_share_scope=True
    )
    result = native(
        tmp_path,
        frame,
        _calendar({1: dates[::2][-20:], 4: dates[-20:]}),
        "market_structure_factors",
        start=dates[-1],
        end=dates[-1],
        factor_columns=["market_above_ma20_fraction"],
    )
    assert result.market_above_ma20_fraction.tolist() == [1.0, 1.0]


def test_breadth_counts_market_sessions_not_each_stocks_sessions(tmp_path):
    from tests.markets.a_share.test_calendar_dependency_scope import _prepare as native

    dates = pd.bdate_range("2024-01-02", periods=81).as_unit("ns")
    frame = pd.concat([_panel(dates[::2]), _panel(dates, "b", 4)]).assign(
        is_a_share_scope=True
    )
    # Both markets' O/C states cover the five preceding market sessions.
    clock = _calendar({1: dates[74::2], 4: dates[74:]})
    result = native(
        tmp_path,
        frame,
        clock,
        "market_structure_factors",
        start=dates[-1],
        end=dates[-1],
        factor_columns=["market_breadth_change_5d"],
    )
    assert result.market_breadth_change_5d.tolist() == [0.0, 0.0]


def test_all_structure_factors_keep_own_60_and_peer_19_windows_separate(tmp_path):
    from tests.markets.a_share.test_calendar_dependency_scope import _prepare as native

    # Keep the original 15 owners explicit: new multi-year states have wider halos.
    factors = [
        "market_up_fraction",
        "market_above_ma20_fraction",
        "market_above_ma60_fraction",
        "market_high_low_balance_60d",
        "market_breadth_change_5d",
        "industry_return_dispersion_20d",
        "industry_positive_return_hhi_20d",
        "industry_return_relative_20d",
        "industry_turnover_relative",
        "industry_amount_share",
        "industry_amount_hhi",
        "industry_activity_pressure",
        "industry_comovement_20d",
        "leader_breadth_fragility",
        "industry_isolation_20d",
    ]

    dates = pd.bdate_range("2023-01-02", periods=241).as_unit("ns")
    frame = pd.concat([_panel(dates[::2]), _panel(dates, "b", 4)]).assign(
        is_a_share_scope=True,
        industry_standard_id="fixture",
        industry_level1_code="one",
        turnover=0.05,
        float_market_cap=1e9,
    )
    phase = (frame.timestamp - dates[0]).dt.days.to_numpy()
    frame["period_return"] = 0.01 * np.sin(phase / 7 + frame.asset_id.eq("b"))
    full = native(
        tmp_path / "full",
        frame,
        _calendar({1: dates[::2], 4: dates}),
        "market_structure_factors",
        factor_columns=factors,
    )
    # Slow-market prices need 60 own sessions; common peer returns need only 19.
    # Thus the fast market's complete 60-session price calendar is sufficient.
    narrow = native(
        tmp_path / "narrow",
        frame,
        _calendar({1: dates[::2][-61:], 4: dates[-61:]}),
        "market_structure_factors",
        start=dates[-1],
        end=dates[-1],
        factor_columns=factors,
    )
    expected = full.loc[full.timestamp.eq(dates[-1])].reset_index(drop=True)
    pd.testing.assert_frame_equal(narrow, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "target,options",
    [
        ("price_volume_facts", {"factor_columns": ["reversal_1d"]}),
        ("forward_returns", {"label_columns": ["forward_return_1d"]}),
        ("execution_mask", {}),
    ],
)
def test_contiguous_relisting_still_resets_numerical_owners(tmp_path, target, options):
    from tests.markets.a_share.test_calendar_dependency_scope import _prepare as native

    dates = pd.bdate_range("2024-01-02", periods=10).as_unit("ns")
    panel = _panel(dates)
    panel.loc[5:, "listing_date"] = dates[5]
    clock = _calendar({1: dates})
    axis = producer._calendar_panel_axis(panel, clock)
    assert axis._calendar_segment.nunique() == 1
    result = native(tmp_path, panel, clock, target, **options).set_index("timestamp")
    if target == "price_volume_facts":
        assert pd.isna(result.loc[dates[5], "reversal_1d"])
        assert result.loc[dates[6], "reversal_1d"] == pytest.approx(-(106 / 105 - 1))
    elif target == "forward_returns":
        assert pd.isna(result.loc[dates[3], "forward_return_1d"])
        assert result.loc[dates[5], "forward_return_1d"] == pytest.approx(107 / 106 - 1)
    else:
        assert result.loc[dates[4], ["can_enter", "can_exit"]].isna().all()
        assert result.loc[dates[5], ["can_enter", "can_exit"]].all()


def test_exit_knowledge_isolated_by_asset_listing_and_unknown_reset():
    dates = pd.bdate_range("2024-01-02", periods=8).as_unit("ns")
    reset = _panel(dates[:7], "a").assign(
        delisting_date=pd.NaT, delisting_date_unresolved=False
    )
    reset.loc[:1, "delisting_date"] = dates[1]
    reset.loc[2, "delisting_date_unresolved"] = True
    exited = _panel(dates[:2], "b").assign(
        delisting_date=dates[2], delisting_date_unresolved=False
    )
    active = _panel(dates[:1], "c").assign(
        delisting_date=pd.NaT, delisting_date_unresolved=False
    )
    relisted = pd.concat(
        [
            _panel(dates[:2], "d").assign(delisting_date=dates[2]),
            _panel(dates[5:6], "d").assign(
                listing_date=dates[5], delisting_date=pd.NaT
            ),
        ],
        ignore_index=True,
    ).assign(delisting_date_unresolved=False)
    panel = pd.concat([reset, exited, active, relisted], ignore_index=True)
    axis = producer._calendar_panel_axis(
        panel, _calendar({1: dates}), panel_end=dates[-1]
    )
    expected = {
        "a": dates.tolist(),
        "b": dates[:3].tolist(),
        "c": dates.tolist(),
        "d": dates[:3].tolist() + dates[5:].tolist(),
    }
    assert {
        asset: sorted(frame.timestamp.tolist())
        for asset, frame in axis.groupby("asset_id")
    } == expected
