"""Proportional futures states use actual expiry; unavailable feeds stay local."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.linked_markets import (
    calculate,
    futures_states,
)


def test_futures_basis_oi_and_calendar_day_annualization_by_hand():
    date = pd.Timestamp("2024-01-02")
    f = pd.DataFrame(
        dict(
            asset_id=["IF2401", "IF2402"],
            timestamp=date,
            product="IF",
            close=[102.0, 104.0],
            volume=[40.0, 60.0],
            open_interest=[80.0, 20.0],
            available_at=date + pd.Timedelta(hours=16),
        )
    )
    c = pd.DataFrame(
        dict(
            asset_id=["IF2401", "IF2402"],
            timestamp=date,
            product="IF",
            expiry_date=[date + pd.Timedelta(days=10), date + pd.Timedelta(days=40)],
            available_at=date + pd.Timedelta(hours=16),
        )
    )
    i = pd.DataFrame(dict(source_asset_id=["000300"], timestamp=[date], close=[100.0]))
    state = futures_states(f, c, i).iloc[0]
    assert state.index_future_near_basis == pytest.approx(0.02)
    assert state.index_future_calendar_spread == pytest.approx(104 / 102 - 1)
    assert state.index_future_annualized_basis == pytest.approx(0.02 * 365 / 10)
    assert state.index_future_roll_concentration == pytest.approx(0.8**2 + 0.2**2)
    assert state.index_future_volume_to_oi == pytest.approx(1)
    decision = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["a", "a"],
            timestamp=[date + pd.Timedelta(hours=15), date + pd.Timedelta(days=1)],
            signal_session=date,
        )
    )
    result = calculate(
        decision,
        dict(public_futures_daily=f, public_futures_contracts=c, index_market=i),
        columns=["index_future_near_basis", "us_short_sale_activity_fraction"],
    )
    assert np.isnan(result.index_future_near_basis.iloc[0])
    assert result.index_future_near_basis.iloc[1] == pytest.approx(0.02)
    assert result.us_short_sale_activity_fraction.isna().all()
    assert result.attrs["missing_source_dependencies"] == {
        "us_short_sale_activity_fraction": ["public_finra_short_volume"]
    }
    partial = calculate(
        decision,
        dict(
            public_futures_daily=f,
            public_futures_contracts=c,
            index_market=i,
            public_etf_prices=pd.DataFrame({"close": [100.0]}),
        ),
        columns=["index_future_near_basis", "us_sector_etf_return_shock_1d"],
    )
    assert partial.index_future_near_basis.iloc[-1] == pytest.approx(0.02)
    assert partial.us_sector_etf_return_shock_1d.isna().all()


def test_hedging_exposure_uses_the_actual_futures_benchmark():
    dates = pd.bdate_range("2024-01-02", periods=132)
    changes = 0.01 * np.sin(np.arange(len(dates)) / 3)
    spot = 100 * np.cumprod(1 + changes)
    benchmark = pd.DataFrame(
        dict(source_asset_id="000300", timestamp=dates, close=spot)
    )
    f = pd.DataFrame(
        dict(
            asset_id="IF",
            product="IF",
            timestamp=dates,
            available_at=dates + pd.Timedelta(hours=16),
            close=spot * (1 + np.arange(len(dates)) * 0.001),
            volume=100.0,
            open_interest=100.0,
        )
    )
    c = pd.DataFrame(
        dict(
            asset_id=["IF"],
            product="IF",
            timestamp=dates[0],
            available_at=dates[0] + pd.Timedelta(hours=16),
            expiry_date=pd.Timestamp("2026-01-01"),
        )
    )
    decision = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=dates + pd.Timedelta(days=1),
            signal_session=dates,
            period_return=2 * changes,
        )
    )
    result = calculate(
        decision,
        dict(
            public_futures_daily=f, public_futures_contracts=c, index_market=benchmark
        ),
        columns=["basis_shock_x_hedging_exposure"],
    )
    assert result.basis_shock_x_hedging_exposure.iloc[-1] == pytest.approx(
        0.002, rel=1e-10
    )
