"""Proportional futures states use actual expiry; unavailable feeds stay local."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.linked_markets import (
    calculate,
    futures_states,
)


@pytest.mark.parametrize(
    "case", ["known", "late_weights", "unknown_return", "member_change"]
)
def test_index_breadth_and_concentration_use_a_visible_full_member_baseline(case):
    dates = pd.bdate_range("2024-01-02", periods=7)
    decision = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="index",
            signal_session=dates[-1],
            timestamp=[dates[-1] + pd.Timedelta(hours=9)],
        )
    )
    prices = pd.concat(
        [
            pd.DataFrame(dict(asset_id=asset, timestamp=dates, adj_close=values))
            for asset, values in [
                ("a", [100.0] * 6 + [110.0]),
                ("b", [100.0] * 5 + [75.0, 90.0]),
                ("c", [100.0] * 6 + [110.0]),
            ]
        ],
        ignore_index=True,
    )
    if case == "unknown_return":
        prices.loc[
            prices.asset_id.eq("b") & prices.timestamp.eq(dates[-1]), "adj_close"
        ] = np.nan
    events = pd.DataFrame(
        dict(
            source_asset_id=["000300"],
            constituent_stock_code=["a"],
            security_type=[1],
            effective_date=[dates[1]],
            announcement_date=[dates[0]],
            change_type=[1],
        )
    )
    if case == "member_change":
        events = pd.concat(
            [
                events,
                pd.DataFrame(
                    dict(
                        source_asset_id="000300",
                        constituent_stock_code=["a", "c"],
                        security_type=1,
                        effective_date=dates[-1],
                        announcement_date=dates[-2],
                        change_type=[2, 1],
                    )
                ),
            ],
            ignore_index=True,
        )
    weights = pd.DataFrame(
        dict(
            source_asset_id="000300",
            constituent_asset_id=["a", "b"],
            timestamp=dates[0],
            available_at=dates[-1] + pd.Timedelta(hours=10)
            if case == "late_weights"
            else dates[0] + pd.Timedelta(hours=18),
            weight=[0.6, 0.4],
        )
    )
    columns = ["index_positive_participation_5d", "index_positive_return_concentration"]
    result = calculate(
        decision,
        dict(market_panel_1d=prices, index_membership=events, index_weights=weights),
        columns=columns,
    )
    if case in {"late_weights", "unknown_return"}:
        assert result[columns].isna().all().all()
    else:
        # Unchanged member b stays in the denominator without a change event.
        assert result[columns[0]].iloc[0] == pytest.approx(0.5)
        if case == "member_change":
            assert result[columns[1]].isna().all()
        else:
            assert result[columns[1]].iloc[0] == pytest.approx(
                (0.06 / 0.14) ** 2 + (0.08 / 0.14) ** 2
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
