"""Historical stock identity, publication time and quote-independent membership."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.markets.implementations.a_share.stock_pool import (
    _ipo_no_limit,
    load_stock_pool,
    load_stock_status,
    stock_pool_rows,
    stock_status_rows,
)
from quant_research_platform.research.datasets.high_frequency import sample_hft
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.research.test_hft_sampling import START, config, quotes, save_quotes


def fact(path, frame, keys, timestamp="timestamp"):
    roles = {
        name: "timestamp" if name == timestamp else "key" if name in keys else "context"
        for name in frame
    }
    write_dataset(
        path,
        frame.sort_values(timestamp, kind="stable"),
        spec=ProcessedDataSpec(
            data_id=path.name,
            market_id="a_share",
            frequency="1d",
            grain="research_row",
            kind="annotation",
            column_roles=roles,
            primary_key=keys,
            partition_columns=(),
        ),
    )
    return str(path)


def pool_config(tmp_path, *, status=None, stocks=None, no_limit=None):
    listing = pd.Timestamp("2024-12-02")
    if stocks is None:
        stocks = pd.DataFrame(
            {
                "source_asset_id": ["A", "B", "C"],
                "listing_date": listing,
                "timestamp": listing,
            }
        )
    if status is None:
        status = pd.DataFrame(
            {
                "asset_id": ["A", "B", "C", "A", "A"],
                "listing_date": listing,
                "timestamp": [listing] * 3
                + [START + pd.Timedelta(seconds=6), START + pd.Timedelta(seconds=1)],
                "published_at": [listing] * 3
                + [START + pd.Timedelta(seconds=7), START + pd.Timedelta(seconds=10)],
                "market_type": 1,
                "is_st": pd.array([False, False, None, True, False], dtype="boolean"),
                "is_pt": False,
                "is_delisting": False,
            }
        )
    status = status.assign(event_id=np.arange(len(status)))
    calendar = pd.date_range("2024-12-01", "2025-01-03")
    calendar = pd.DataFrame(
        {
            "calendar_date": calendar,
            "source_market_type": 1,
            "is_trading_day": calendar.day_of_week < 5,
        }
    )
    result = {
        "stock_master": fact(
            tmp_path / "stocks", stocks, ("source_asset_id", "timestamp")
        ),
        "status": fact(
            tmp_path / "status", status, ("asset_id", "timestamp", "event_id")
        ),
        "available_at_column": "published_at",
        "trading_calendar": fact(
            tmp_path / "calendar",
            calendar,
            ("source_market_type", "calendar_date"),
            "calendar_date",
        ),
    }
    if no_limit is not None:
        result["trading_no_limit"] = {
            "inputs": fact(
                tmp_path / "no_limit", no_limit, ("source_asset_id", "timestamp")
            ),
            "available_at_column": "published_at",
            "ipo_reasons": ["IPO"],
        }
    return result


@pytest.mark.parametrize("delay", [0, 1])
def test_native_stock_pool_keeps_unknown_raw_and_unquoted_qualified_rows(
    tmp_path, monkeypatch, delay
):
    source = save_quotes(
        tmp_path,
        pd.concat(
            [quotes([0, 3, 6, 9, 12, 15], asset=asset) for asset in ("A", "C", "D")]
        ),
    )
    settings = config(
        tmp_path,
        source,
        stock_pool=pool_config(tmp_path),
        quote_visibility_delay_ms=delay,
        max_quote_age_seconds=4,
    )
    path = sample_hft(settings)
    result = read_dataset(path).set_index(["asset_id", "timestamp"])
    a = result.loc["A"]
    assert a.eligible.tolist() == [True, True, True, False, False]
    np.testing.assert_allclose(
        a.f,
        [0, 6, 12, 18, 24] if delay == 0 else [np.nan, 0, 6, 12, 18],
        equal_nan=True,
    )
    assert result.loc["C"].eligible.isna().all()
    assert result.loc["D"].eligible.isna().all()
    assert result.loc["D"].listing_date.isna().all()
    assert result.loc[["C", "D"], "f"].notna().all()
    b = result.loc["B"]
    assert len(b) == 5 and b.eligible.all()
    assert (
        b[["f", "mid", "quote_timestamp", "future_mid_return_3s"]].isna().all(axis=None)
    )
    assert b.listing_date.eq("2024-12-02").all()
    assert (
        open_dataset_session(path).manifest["producer_diagnostics"]["sampling"][
            "asset_universe"
        ]
        == "point_in_time_stock_pool_with_raw_observations"
    )
    assert not any("missing" in name for name in result)
    from quant_research_platform.research.datasets import high_frequency as module

    monkeypatch.setattr(module, "DEFAULT_BATCH_ROWS", 3)
    settings["output"] = str(tmp_path / "blocked")
    blocked = read_dataset(sample_hft(settings))
    assert blocked.timestamp.is_monotonic_increasing
    pd.testing.assert_frame_equal(blocked, read_dataset(path))


def test_relisted_asset_never_uses_old_life_status(tmp_path):
    old, new = pd.Timestamp("2024-12-02"), START.normalize()
    stocks = pd.DataFrame(
        {
            "source_asset_id": ["A", "A"],
            "listing_date": [old, new],
            "timestamp": [old, new],
            "delisting_date": [pd.Timestamp("2024-12-31"), pd.NaT],
        }
    )
    status = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "listing_date": [old, new],
            "timestamp": [old, new],
            "published_at": [old, START + pd.Timedelta(seconds=6)],
            "market_type": 1,
            "is_st": False,
            "is_pt": False,
            "is_delisting": False,
        }
    )
    source = save_quotes(tmp_path, quotes([0, 3, 6, 9, 12, 15]))
    result = read_dataset(
        sample_hft(
            config(
                tmp_path,
                source,
                stock_pool=pool_config(tmp_path, status=status, stocks=stocks),
            )
        )
    )
    assert result.listing_date.eq(new).all()
    assert (
        result.loc[result.timestamp.lt(START + pd.Timedelta(seconds=6)), "eligible"]
        .isna()
        .all()
    )
    assert (
        result.loc[result.timestamp.ge(START + pd.Timedelta(seconds=6)), "eligible"]
        .isna()
        .all()
    )
    assert result.f.notna().all()


@pytest.mark.parametrize("reverse", [False, True])
def test_status_projection_keeps_rows_and_field_local_conflicts(tmp_path, reverse):
    listing = pd.Timestamp("2024-12-02")
    status = pd.DataFrame(
        {
            "asset_id": ["A"] * 4,
            "listing_date": listing,
            "timestamp": [START, START, START + pd.Timedelta(seconds=6), START],
            "published_at": START + pd.to_timedelta([0, 3, 5, 9], unit="s"),
            "market_type": 1,
            "is_st": [True, False, False, True],
            "is_pt": False,
            "is_delisting": [False, False, True, False],
        }
    )
    settings = pool_config(tmp_path, status=status.iloc[::-1] if reverse else status)
    state = load_stock_status(
        {
            name: settings[name]
            for name in ("stock_master", "status", "available_at_column")
        }
    )
    decisions = pd.DataFrame(
        {
            "asset_id": ["A", "A", "D", "A", "A"],
            "timestamp": START + pd.to_timedelta([9, 3, 3, 0, 5], unit="s"),
        }
    )
    result = stock_status_rows(decisions, **state)
    assert len(result) == len(decisions)
    assert result.asset_id.tolist() == decisions.asset_id.tolist()
    pd.testing.assert_series_equal(
        result.timestamp, decisions.timestamp.dt.as_unit("ns")
    )
    assert not result.is_st.iloc[0]
    assert pd.isna(result.is_st.iloc[1])
    assert result.is_st.iloc[3]
    assert pd.isna(result.is_st.iloc[4])
    assert result.is_pt.dropna().eq(False).all()
    assert result.is_delisting.iloc[0]
    assert not result.is_delisting.iloc[1]
    assert not result.is_delisting.iloc[4]  # published, but not yet effective
    assert (
        result.loc[2, ["listing_date", "is_st", "is_pt", "is_delisting"]].isna().all()
    )
    assert result._known_at.iloc[0] == START + pd.Timedelta(seconds=6)
    assert result._available_at.iloc[0] == START + pd.Timedelta(seconds=5)


def test_legacy_status_without_consolidation_fact_preserves_raw_unknown_membership(
    tmp_path,
):
    listing = pd.Timestamp("2024-12-02")
    status = pd.DataFrame(
        {
            "asset_id": ["A"],
            "listing_date": [listing],
            "timestamp": [listing],
            "published_at": [listing],
            "market_type": [1],
            "is_st": [False],
            "is_pt": [False],
        }
    )
    source = save_quotes(tmp_path, quotes([0, 3, 6, 9, 12, 15]))
    result = read_dataset(
        sample_hft(
            config(tmp_path, source, stock_pool=pool_config(tmp_path, status=status))
        )
    )
    assert result.eligible.isna().all()
    assert result.f.notna().all()
    assert result.mid.notna().all()
    assert result.future_mid_return_3s.notna().all()
    assert "stock_pool_available_at" not in result


@pytest.mark.parametrize("initial", [None, True, False])
def test_first_provided_life_does_not_prove_an_ipo(tmp_path, initial):
    listing = START.normalize()
    stocks = pd.DataFrame(
        {"source_asset_id": ["A"], "timestamp": [listing], "listing_date": [listing]}
    )
    if initial is not None:
        stocks["is_initial_listing"] = pd.array([initial], dtype="boolean")
    status = pd.DataFrame(
        {
            "asset_id": ["A"],
            "timestamp": [listing],
            "listing_date": [listing],
            "published_at": [listing],
            "market_type": [1],
            "is_st": [False],
            "is_pt": [False],
            "is_delisting": [False],
        }
    )
    state = load_stock_pool(pool_config(tmp_path, stocks=stocks, status=status))
    result = stock_pool_rows(pd.DatetimeIndex([START]), **state)
    if initial is True:
        assert not result.eligible.iloc[0]  # proven registered IPO, first session
    else:
        assert pd.isna(result.eligible.iloc[0])  # no inference from source ordering


@pytest.mark.parametrize("timestamp", [False, True])
def test_undated_master_keeps_life_but_does_not_date_its_mutable_state(
    tmp_path, timestamp
):
    settings = pool_config(tmp_path)
    listing = pd.Timestamp("2024-12-02")
    master = pd.DataFrame(
        {
            "source_asset_id": ["A"],
            "listing_date": [listing],
            "delisting_date": [pd.Timestamp("2025-01-01")],
            "stock_name": ["later"],
            "is_initial_listing": [True],
        }
    )
    if timestamp:
        master["timestamp"] = pd.Series(
            pd.NaT, index=master.index, dtype="datetime64[ns]"
        )
    settings["stock_master"] = fact(
        tmp_path / "undated", master, ("source_asset_id",), "listing_date"
    )
    state = load_stock_status(
        {
            name: settings[name]
            for name in ("stock_master", "status", "available_at_column")
        }
    )
    actual = stock_status_rows(
        pd.DataFrame({"asset_id": ["A"], "timestamp": [START]}), **state
    )
    assert actual.listing_date.eq(listing).all()
    assert (
        actual[["delisting_date", "stock_name", "is_initial_listing"]]
        .isna()
        .all(axis=None)
    )
    assert actual.delisting_date_unresolved.all()
    assert not actual.is_st.iloc[0]  # independent dated status is still usable


@pytest.mark.parametrize(
    ("market", "listing", "expected"),
    [
        (1, "2023-04-07", [False] * 6),
        (1, "2023-04-10", [True] * 5 + [False]),
        (4, "2023-04-10", [True] * 5 + [False]),
        (16, "2020-08-21", [False] * 6),
        (16, "2020-08-24", [True] * 5 + [False]),
        (32, "2019-07-22", [True] * 5 + [False]),
    ],
)
def test_ipo_rule_depends_on_listing_regime_and_real_sessions(
    market, listing, expected
):
    listing = pd.Timestamp(listing)
    sessions = pd.bdate_range(listing, periods=6)
    dates = pd.date_range(listing, sessions[-1])
    calendar = pd.Series(dates.day_of_week < 5, index=dates, dtype="boolean")
    frame = pd.DataFrame(
        {
            "timestamp": sessions + pd.Timedelta(hours=1, minutes=30),
            "listing_date": listing,
            "market_type": market,
            "_listing_market_type": market,
            "_initial_life": True,
        }
    )
    actual = _ipo_no_limit(frame, {market: calendar})
    assert actual.tolist() == expected
    pd.testing.assert_series_equal(
        _ipo_no_limit(frame, {market: calendar.iloc[::-1]}), actual
    )
    if expected[0]:
        # A missing exchange day is not silently compressed out of IPO age.
        missing = calendar.drop(sessions[1])
        assert pd.isna(_ipo_no_limit(frame, {market: missing}).iloc[-1])
        pd.testing.assert_series_equal(
            _ipo_no_limit(frame, {market: missing.iloc[::-1]}),
            _ipo_no_limit(frame, {market: missing}),
        )


def test_unknown_old_regime_and_unavailable_no_limit_fact_do_not_become_false(tmp_path):
    listing = pd.Timestamp("2013-01-02")
    frame = pd.DataFrame(
        {
            "timestamp": [listing + pd.Timedelta(hours=1, minutes=30)],
            "listing_date": listing,
            "market_type": 1,
            "_listing_market_type": 1,
            "_initial_life": True,
        }
    )
    assert (
        _ipo_no_limit(frame, {1: pd.Series([True], index=[listing], dtype="boolean")})
        .isna()
        .all()
    )
    events = pd.DataFrame(
        {
            "source_asset_id": ["A", "B"],
            "timestamp": START.normalize(),
            "published_at": START + pd.Timedelta(seconds=6),
            "reason": ["IPO", "resumption"],
        }
    )
    source = save_quotes(tmp_path, quotes([0, 3, 6, 9, 12, 15]))
    result = read_dataset(
        sample_hft(
            config(tmp_path, source, stock_pool=pool_config(tmp_path, no_limit=events))
        )
    ).set_index(["asset_id", "timestamp"])
    assert result.loc["A"].eligible.tolist() == [True, True, False, False, False]
    assert result.loc["B"].eligible.all()
