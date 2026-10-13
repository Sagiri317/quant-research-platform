"""A missing calendar date cannot disappear together with all its quotes."""

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import producer
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.markets.a_share.test_target_dag_chain import _synthetic_interim_frames


@pytest.mark.parametrize("missing_position", [1, 3, 6])
@pytest.mark.parametrize("date_bounds", [False, True])
def test_joint_calendar_quote_gap_blocks_native_publication_and_preserves_output(
    tmp_path, monkeypatch, missing_position, date_bounds
):
    dates = pd.bdate_range("2020-01-02", periods=7).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    # Make gaps fall between candidate blocks as well as within them.
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", 6)
    window = TimeWindow(
        dates[1].date() if date_bounds else dates[1],
        dates[-1].date() if date_bounds else dates[-1],
    )

    def persist_source():
        for data_id in spec.source_input_data_ids:
            native = profile.interim.get(data_id)
            write_dataset(
                layout.interim_data_root(native.source_id, data_id),
                source[data_id],
                spec=native,
            )

    def publish():
        with producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=window,
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        ) as (frames, metadata, verify):
            write_dataset(
                root,
                frames,
                spec=spec,
                instance_metadata=metadata,
                staged_validator=verify,
            )

    persist_source()
    publish()
    old_root = root
    before = read_dataset(old_root)
    layout = StorageLayout(tmp_path / "candidate")
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    missing = dates[missing_position]
    source["market_bars_1d"] = source["market_bars_1d"].loc[
        source["market_bars_1d"].timestamp.ne(missing)
    ]
    source["trading_calendar"] = source["trading_calendar"].loc[
        source["trading_calendar"].calendar_date.ne(missing)
    ]
    persist_source()
    with pytest.raises(ValueError, match="a_share_trading_calendar_unknown_session"):
        publish()
    assert not root.exists()
    pd.testing.assert_frame_equal(read_dataset(old_root), before)


@pytest.mark.parametrize("state", ["O", "C"])
def test_joint_missing_quotes_obeys_explicit_calendar_state(tmp_path, state):
    dates = pd.bdate_range("2020-01-02", periods=7).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    missing = dates[3]
    source["market_bars_1d"] = source["market_bars_1d"].loc[
        source["market_bars_1d"].timestamp.ne(missing)
    ]
    calendar = source["trading_calendar"]
    calendar.loc[
        calendar.calendar_date.eq(missing), ["is_trading_day", "source_state"]
    ] = [state == "O", state]
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        native = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(native.source_id, data_id),
            source[data_id],
            spec=native,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        layout=layout,
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
    panel = read_dataset(root)
    gap = panel.loc[panel.timestamp.eq(missing)]
    if state == "O":
        assert len(gap) == 3
        assert gap.close.isna().all()
    else:
        assert gap.empty
    assert not panel.timestamp.dt.dayofweek.ge(5).any()


@pytest.mark.parametrize("sessions", [3, 260])
def test_joint_calendar_gap_precludes_exact_age_but_preserves_proven_lower_bound(
    sessions,
):
    from quant_research_platform.markets.implementations.a_share.calendar import (
        compute_listing_session_facts,
    )

    opened = pd.bdate_range("2020-01-02", periods=sessions).as_unit("ns")
    dates = pd.date_range(opened[0], opened[-1]).as_unit("ns")
    calendar = pd.DataFrame(
        {
            "calendar_date": dates,
            "source_market_type": 1,
            "is_trading_day": dates.isin(opened),
        }
    )
    calendar = calendar.loc[calendar.calendar_date.ne(opened[1])]
    observed = pd.DataFrame(
        {
            "asset_id": ["000001"],
            "timestamp": opened[-1],
            "listing_date": opened[0],
        }
    )
    history = pd.DataFrame(
        {
            "asset_id": ["000001"],
            "timestamp": opened[0],
            "market_type": 1,
        }
    )
    facts = compute_listing_session_facts(observed, calendar, history)
    assert facts.listing_age_trading_days.isna().all()
    if sessions >= 252:
        assert facts.has_252_trading_sessions.all()
    else:
        assert facts.has_252_trading_sessions.isna().all()


@pytest.mark.parametrize("reset_quote", [False, True])
@pytest.mark.parametrize("batch_rows", [3, 100])
def test_historical_unknown_is_a_carry_barrier_until_a_real_quote_resets_it(
    tmp_path, monkeypatch, reset_quote, batch_rows
):
    dates = pd.bdate_range("2020-01-06", periods=5).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    gap = dates[1]
    source["trading_calendar"] = source["trading_calendar"].loc[
        source["trading_calendar"].calendar_date.ne(gap)
    ]
    bars = source["market_bars_1d"]
    suspended = bars.timestamp.ge(dates[3] if reset_quote else dates[2])
    bars.loc[suspended, "filling_flag"] = 1
    prices = ["open", "high", "low", "close", "adj_open", "adj_close"]
    bars.loc[suspended, prices] = float("nan")
    expected = bars.loc[bars.timestamp.eq(dates[2])].set_index("source_asset_id").close
    source["market_bars_1d"] = bars.loc[bars.timestamp.ne(gap)]
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        native = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(native.source_id, data_id),
            source[data_id],
            spec=native,
        )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        layout=layout,
        time_window=TimeWindow(dates[3], dates[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
    actual = read_dataset(root)
    assert len(actual) == 6
    assert actual.timestamp.ge(dates[3]).all()
    assert not actual.is_trading.any()
    if reset_quote:
        assert actual.close.tolist() == actual.asset_id.map(expected).tolist()
    else:
        assert actual[prices].isna().all().all()


def test_historical_present_closed_calendar_still_rejects_actual_trade(tmp_path):
    dates = pd.bdate_range("2020-01-06", periods=5).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    source["trading_calendar"].loc[
        source["trading_calendar"].calendar_date.eq(dates[1]), "is_trading_day"
    ] = False
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        native = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(native.source_id, data_id),
            source[data_id],
            spec=native,
        )
    with pytest.raises(ValueError, match="market_bar_session_missing_from_calendar"):
        with producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=TimeWindow(dates[3], dates[-1]),
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        ) as (frames, _metadata, _verify):
            list(frames)
