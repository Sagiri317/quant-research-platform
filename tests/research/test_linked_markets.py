"""Proportional futures states use actual expiry; unavailable feeds stay local."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.linked_markets import (
    calculate,
    futures_states,
)


@pytest.mark.parametrize("case", ("known", "missing_member", "not_published", "early"))
def test_fixed_index_group_does_not_shrink_or_use_prepublication_backcasts(case):
    days = pd.bdate_range("2024-01-02", periods=7)
    raw = pd.concat(
        [
            pd.DataFrame(
                dict(source_asset_id=asset, timestamp=days, close=[100.0] * 6 + [last])
            )
            for asset, last in (("a", 110), ("b", 90), ("c", 100))
        ],
        ignore_index=True,
    )
    if case == "missing_member":
        raw = raw.loc[~(raw.source_asset_id.eq("c") & raw.timestamp.eq(days[-1]))]
    group = {asset: "2024-01-01" for asset in ("a", "b", "c")}
    if case == "not_published":
        group["c"] = days[-1].date().isoformat()
    clock = days[-1] + pd.Timedelta(hours=6 if case == "early" else 8)
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["index"],
            signal_session=days[-1],
            timestamp=clock,
        )
    )
    calendar = pd.DataFrame(
        dict(calendar_date=days, source_market_type=1, is_trading_day=True)
    )
    names = [f"index_group_{kind}_5d" for kind in ("positive_fraction", "leader_gap")]
    result = calculate(
        decisions,
        {"index_market": raw, "trading_calendar": calendar},
        columns=names,
        parameters={"index_group": group},
    )
    if case == "known":
        assert result[names[0]].iloc[0] == pytest.approx(1 / 3)
        assert result[names[1]].iloc[0] == pytest.approx(0.15)
    else:
        assert result[names].isna().all(axis=None)


@pytest.mark.parametrize(
    "case",
    [
        "known",
        "late_weights",
        "unknown_return",
        "member_change",
        "relisted",
        "unknown_lifecycle",
    ],
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
    prices["listing_date"] = pd.Timestamp("2000-01-01")
    if case in {"relisted", "unknown_lifecycle"}:
        prices.loc[
            prices.asset_id.eq("b") & prices.timestamp.eq(dates[-1]), "listing_date"
        ] = dates[-1] if case == "relisted" else pd.NaT
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
    if case in {"late_weights", "unknown_return", "relisted", "unknown_lifecycle"}:
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


@pytest.mark.parametrize("lifecycle", ["known", "relisted", "unknown"])
def test_hedging_exposure_uses_the_actual_futures_benchmark(lifecycle):
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
            listing_date=pd.Timestamp("2000-01-01"),
        )
    )
    if lifecycle != "known":
        decision.loc[decision.index[-1], "listing_date"] = (
            dates[-1] if lifecycle == "relisted" else pd.NaT
        )
    decision["asset_id"] = decision.asset_id.astype("string")
    result = calculate(
        decision,
        dict(
            public_futures_daily=f, public_futures_contracts=c, index_market=benchmark
        ),
        columns=["basis_shock_x_hedging_exposure"],
    )
    if lifecycle == "known":
        assert result.basis_shock_x_hedging_exposure.iloc[-1] == pytest.approx(
            0.002, rel=1e-10
        )
    else:
        assert pd.isna(result.basis_shock_x_hedging_exposure.iloc[-1])


@pytest.mark.parametrize("changed_asset", ["a", "b", None])
def test_pair_history_restarts_when_either_listing_changes(changed_asset):
    dates = pd.bdate_range("2024-01-02", periods=90)
    boundary = 44 if changed_asset is not None else 0
    n = np.arange(len(dates))
    x = np.log(20.0) + np.cumsum(0.01 + 0.003 * np.sin(n))
    y = np.log(30.0) + np.cumsum(0.006 + 0.002 * np.cos(n / 3))
    series = {"a": y, "b": x}
    if changed_asset is not None:
        series[changed_asset][boundary:] += np.log(10)
    panel = pd.concat(
        [
            pd.DataFrame(
                dict(
                    asset_id=asset,
                    timestamp=dates,
                    adj_close=np.exp(logged),
                    listing_date=[
                        dates[boundary]
                        if asset == changed_asset and i >= boundary
                        else pd.Timestamp("2000-01-01")
                        for i in n
                    ],
                )
            )
            for asset, logged in series.items()
        ],
        ignore_index=True,
    )
    if changed_asset is None:
        # Generic/ETF histories without native stock lifecycle fields stay supported.
        panel = panel.drop(columns="listing_date").astype({"asset_id": "string"})
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            signal_session=dates,
            timestamp=dates + pd.Timedelta(days=1),
        )
    )
    names = [
        "linked_pair_relative_return_20d",
        "linked_pair_beta_adjusted_spread_20d",
        "linked_pair_spread_zscore_20d",
    ]
    values = calculate(
        decisions,
        {"market_panel_1d": panel},
        columns=names,
        parameters={"pair_assets": {"a": "b"}},
    )
    for name, warmup in zip(names, (20, 21, 41), strict=True):
        assert values[name].iloc[boundary : boundary + warmup].isna().all()
        assert np.isfinite(values[name].iloc[boundary + warmup])
    end = boundary + 20
    assert values[names[0]].iloc[end] == pytest.approx(
        np.expm1(y[end] - y[boundary]) - np.expm1(x[end] - x[boundary])
    )
    end = boundary + 21
    xr, yr = np.diff(x[boundary : end + 1]), np.diff(y[boundary : end + 1])
    slope = np.dot(xr[:-1] - xr[:-1].mean(), yr[:-1] - yr[:-1].mean()) / np.sum(
        (xr[:-1] - xr[:-1].mean()) ** 2
    )
    assert values[names[1]].iloc[end] == pytest.approx(yr[-1] - slope * xr[-1])


def test_native_pair_publication_preserves_listing_history(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    days = pd.bdate_range("2024-01-02", periods=25)
    inputs = {}
    panel = pd.concat(
        [
            pd.DataFrame(
                dict(
                    market_id="a_share",
                    asset_id=asset,
                    timestamp=days,
                    adj_close=[10.0] * 24 + [100.0 if asset == "a" else 10.0],
                    listing_date=[pd.Timestamp("2000-01-01")] * 24
                    + [days[-1] if asset == "a" else pd.Timestamp("2000-01-01")],
                )
            )
            for asset in ("a", "b")
        ],
        ignore_index=True,
    )
    decision = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=days + pd.Timedelta(days=1),
            signal_session=days,
            execution_session=days + pd.Timedelta(days=1),
            return_available_at=days + pd.Timedelta(hours=7),
        )
    )
    for name, frame in (("market_panel_1d", panel), ("decision_panel", decision)):
        path = tmp_path / name
        key = ("market_id", "asset_id", "timestamp")
        write_dataset(
            path,
            frame,
            spec=DataSpec(
                data_id=name,
                frequency="1d",
                grain="asset_timestamp",
                primary_key=key,
                partition_columns=(),
                column_roles={
                    column: "timestamp"
                    if column == "timestamp"
                    else "key"
                    if column in key
                    else "context"
                    for column in frame
                },
            ),
        )
        inputs[name] = str(path)
    name = "linked_pair_relative_return_20d"
    result = read_dataset(
        prepare(
            dict(
                market_id="a_share",
                data_id="linked_market_factors",
                inputs=inputs,
                factor_columns=[name],
                parameters={"pair_assets": {"a": "b"}},
                time_window=dict(
                    start=str(decision.timestamp.min().date()),
                    end=str(decision.timestamp.max().date()),
                ),
                output_root=str(tmp_path / "factors"),
            )
        )
    )
    assert result[name].iloc[20:24].to_numpy(float) == pytest.approx([0.0] * 4)
    assert pd.isna(result[name].iloc[-1])


@pytest.mark.parametrize("gap", ["missing_quote", "unknown_lifecycle"])
def test_pair_endpoint_return_only_requires_its_prices_and_known_lifecycle(gap):
    dates = pd.bdate_range("2024-01-02", periods=21)
    panel = pd.concat(
        [
            pd.DataFrame(
                dict(
                    asset_id=asset,
                    timestamp=dates,
                    adj_close=100.0 + change * np.arange(len(dates)),
                    listing_date=pd.Timestamp("2000-01-01"),
                )
            )
            for asset, change in (("a", 2), ("b", 1))
        ],
        ignore_index=True,
    )
    interior = panel.asset_id.eq("a") & panel.timestamp.eq(dates[10])
    if gap == "missing_quote":
        panel = panel.loc[~interior]
    else:
        panel.loc[interior, "listing_date"] = pd.NaT
    name = "linked_pair_relative_return_20d"
    result = calculate(
        pd.DataFrame(
            dict(
                market_id="a_share",
                asset_id=["a"],
                signal_session=dates[-1],
                timestamp=dates[-1] + pd.Timedelta(days=1),
            )
        ),
        {"market_panel_1d": panel},
        columns=[name],
        parameters={"pair_assets": {"a": "b"}},
    )
    if gap == "missing_quote":
        assert result[name].iloc[0] == pytest.approx(0.4 - 0.2)
    else:
        assert pd.isna(result[name].iloc[0])


def test_native_foreign_beta_keeps_listing_boundaries_local_to_response(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.research.test_prepare_selection import _input

    days = pd.bdate_range("2023-01-02", periods=145)
    n = np.arange(len(days))
    shocks = 0.01 * np.sin(n / 3) + 0.002 * np.cos(n / 7)
    market = 0.01 * np.cos(n / 5)
    # The stock's completed return responds to the shock known before its open.
    response = 2 * np.r_[np.nan, shocks[:-1]] + 0.3 * market
    response[130] = np.nan
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=days + pd.Timedelta(days=1),
            signal_session=days,
            execution_session=days + pd.Timedelta(days=1),
            return_available_at=days + pd.Timedelta(hours=7),
            period_return=response,
            market_return=market,
            listing_date=pd.Timestamp("2000-01-01"),
        )
    )
    decisions.loc[130:, "listing_date"] = days[130]
    prices = pd.DataFrame(
        dict(
            asset_id="SPY",
            timestamp=days,
            available_at=days + pd.Timedelta(hours=22),
            adjusted_close=100 * np.cumprod(1 + shocks),
        )
    )
    prices_path = tmp_path / "us_prices"
    write_dataset(
        prices_path,
        prices,
        spec=DataSpec(
            data_id="public_etf_prices",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("asset_id", "timestamp"),
            partition_columns=(),
            column_roles={
                "asset_id": "key",
                "timestamp": "timestamp",
                "available_at": "context",
                "adjusted_close": "context",
            },
        ),
    )
    result = read_dataset(
        prepare(
            dict(
                market_id="a_share",
                data_id="linked_market_factors",
                inputs=dict(
                    decision_panel=_input(
                        tmp_path, "decision_panel", decisions, native=True
                    ),
                    public_etf_prices=str(prices_path),
                ),
                factor_columns=[
                    "us_sector_own_beta",
                    "us_sector_etf_return_shock_1d",
                ],
                parameters={"asset_us_tickers": {"a": "SPY"}},
                time_window=dict(
                    start=str(decisions.timestamp.min().date()),
                    end=str(decisions.timestamp.max().date()),
                ),
                output_root=str(tmp_path / "foreign_beta"),
            )
        )
    )
    assert result.us_sector_own_beta.iloc[129] == pytest.approx(2, rel=1e-10)
    assert result.us_sector_own_beta.iloc[130:].isna().all()
    assert result.us_sector_etf_return_shock_1d.iloc[-1] == pytest.approx(shocks[-1])
