"""Conflicting ST events stay local after both facts are actually known."""

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed import panels
from quant_research_platform.sources.csmar.interim import materialize_interim_data
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_raw import _write_csmar
from tests.helpers.a_share_sources import _synthetic_interim_frames


def _materialize(tmp_path, rows, chunksize):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_csmar(raw, "special_treatment_changes", rows)
    path = next(raw.glob("*.csv"))
    before = path.read_bytes()
    output = tmp_path / "events"
    materialize_interim_data(
        raw_root=raw,
        output_roots={"special_treatment_changes": output},
        data_ids=("special_treatment_changes",),
        chunksize=chunksize,
    )
    assert path.read_bytes() == before
    return read_dataset(output)


def _panel(history, *, daily_known=False, end=None):
    dates = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    inputs = _synthetic_interim_frames(dates)
    bars = inputs["market_bars_1d"].copy()
    bars.loc[bars.source_asset_id.eq("000001"), "trading_status_code"] = (
        0 if daily_known else pd.NA
    )
    stocks = inputs["stock_master"].copy()
    stocks["timestamp"] = stocks["listing_date"]
    return panels.compute_market_panel(
        bars,
        **panels._prepare_market_panel_inputs(
            stocks=panels._canonical_stock_master(stocks),
            market_cap=inputs["market_cap"],
            trading_calendar=inputs["trading_calendar"],
            special_treatment_history=history,
        ),
        session_window=dates if end is None else dates[dates <= pd.Timestamp(end)],
    )


@pytest.mark.parametrize("second_code", ["BA", "AC"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("daily_known", [False, True])
@pytest.mark.parametrize("chunksize", [1, 100])
def test_raw_conflict_preserves_prefix_prices_and_recovers(
    tmp_path, reverse, daily_known, chunksize, second_code
):
    rows = [
        dict(Symbol="000001", Chgtype="AB", Annoudt="2020-01-02", Execudt="2020-01-02"),
        dict(
            Symbol="000001",
            Chgtype=second_code,
            Annoudt="2020-01-06",
            Execudt="2020-01-02",
        ),
        dict(Symbol="000001", Chgtype="BA", Annoudt="2020-01-07", Execudt="2020-01-07"),
    ]
    history = _materialize(tmp_path, rows[::-1] if reverse else rows, chunksize)
    assert len(history) == 3
    assert history.is_st.notna().all()  # source facts are not rewritten
    before = history.copy(deep=True)
    baseline = _panel(None, daily_known=daily_known)
    actual = _panel(history, daily_known=daily_known)
    assert len(actual) == 15
    pd.testing.assert_frame_equal(
        actual.drop(columns=["is_st", "is_pt"]),
        baseline.drop(columns=["is_st", "is_pt"]),
    )
    independent = actual.asset_id.ne("000001")
    pd.testing.assert_frame_equal(actual.loc[independent], baseline.loc[independent])
    affected = actual.loc[~independent].set_index("timestamp")
    assert affected.loc[:"2020-01-03", "is_st"].all()
    if daily_known:
        assert not affected.loc["2020-01-06", "is_st"]
    else:
        assert pd.isna(affected.loc["2020-01-06", "is_st"])
    if second_code == "BA" or daily_known:
        assert not affected.is_pt.any()  # common event fact/current daily evidence
    else:
        assert pd.isna(affected.loc["2020-01-06", "is_pt"])
    assert not affected.loc["2020-01-07":, "is_st"].any()
    prefix = _panel(
        history,
        daily_known=daily_known,
        end="2020-01-03",
    )
    pd.testing.assert_frame_equal(
        actual.loc[actual.timestamp.le(pd.Timestamp("2020-01-03"))].reset_index(
            drop=True
        ),
        prefix.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(history, before)


@pytest.mark.parametrize("daily_known", [False, True])
def test_unclassified_event_uses_only_independent_current_daily_fact(
    tmp_path, daily_known
):
    history = _materialize(
        tmp_path,
        [
            dict(
                Symbol="000001",
                Chgtype="ZZ",
                Annoudt="2020-01-02",
                Execudt="2020-01-02",
            )
        ],
        1,
    )
    actual = _panel(history, daily_known=daily_known)
    baseline = _panel(None, daily_known=daily_known)
    pd.testing.assert_frame_equal(actual, baseline)
