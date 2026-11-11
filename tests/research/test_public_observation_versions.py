"""Source revisions alter decisions only after publication, without double counting."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors import fund_flows, linked_markets


def _calendar(days):
    dates = pd.date_range(days.min(), days.max())
    return pd.DataFrame(
        dict(calendar_date=dates, source_market_type=1, is_trading_day=dates.isin(days))
    )


def _shares(days):
    return pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=days,
            available_at=days + pd.Timedelta(hours=8),
            shares_outstanding=100.0 + 10 * np.arange(len(days)),
            source="sse_etf_shares",
            source_sha256="first",
        )
    )


def test_native_share_revisions_preserve_past_and_recompute_actual_dependencies(
    tmp_path,
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    days = pd.bdate_range("2024-01-02", periods=23)
    release = days[-1] + pd.Timedelta(days=1)
    original = _shares(days)
    revised = original.iloc[[2]].assign(
        shares_outstanding=150.0, available_at=release, source_sha256="second"
    )
    shares = pd.concat([original, revised], ignore_index=True)
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="f",
            timestamp=[release - pd.Timedelta(hours=1), release],
            signal_session=days[-1],
            execution_session=days[-1],
            return_available_at=pd.NaT,
        )
    )
    inputs = {}
    for name, frame, key, clock in (
        ("decision_panel", decisions, ("asset_id", "timestamp"), "timestamp"),
        (
            "public_fund_shares",
            shares,
            ("asset_id", "timestamp", "source", "source_sha256"),
            "timestamp",
        ),
        (
            "trading_calendar",
            _calendar(days),
            ("source_market_type", "calendar_date"),
            "calendar_date",
        ),
    ):
        path = tmp_path / name
        write_dataset(
            path,
            frame,
            spec=DataSpec(
                data_id=name,
                frequency="1d",
                grain="research_row",
                primary_key=key,
                partition_columns=(),
                column_roles={
                    c: "timestamp" if c == clock else "key" if c in key else "context"
                    for c in frame
                },
            ),
        )
        inputs[name] = str(path)
    names = [
        "etf_share_change_1d",
        "etf_share_change_20d",
        "etf_share_change_surprise_20d",
    ]
    path = prepare(
        dict(
            market_id="a_share",
            data_id="fund_flow_factors",
            inputs=inputs,
            factor_columns=names,
            time_window=dict(
                start=decisions.timestamp.min().isoformat(), end=release.isoformat()
            ),
            output_root=str(tmp_path / "factors"),
        )
    )
    result = read_dataset(path)
    np.testing.assert_allclose(result[names[0]].to_numpy(float), [320 / 310 - 1] * 2)
    np.testing.assert_allclose(
        result[names[1]].to_numpy(float), [320 / 120 - 1, 320 / 150 - 1]
    )
    expected = []
    for value in (120.0, 150.0):
        units = original.shares_outstanding.to_numpy(copy=True)
        units[2] = value
        changes = np.diff(units) / units[:-1]
        history = changes[-21:-1]
        expected.append((changes[-1] - history.mean()) / history.std(ddof=1))
    np.testing.assert_allclose(result[names[2]].to_numpy(float), expected)


@pytest.mark.parametrize("kind", ["same_value", "unknown_clock", "conflicting_release"])
def test_later_share_copies_and_conflicts_have_local_publication_effects(kind):
    days = pd.bdate_range("2024-01-02", periods=2)
    original = _shares(days)
    release = days[-1] + pd.Timedelta(days=1)
    version = original.iloc[[-1]].assign(source_sha256="second", available_at=release)
    if kind == "unknown_clock":
        version = version.assign(available_at=pd.NaT, shares_outstanding=999.0)
    elif kind == "conflicting_release":
        version = pd.concat(
            [
                version.assign(shares_outstanding=120.0),
                version.assign(shares_outstanding=130.0, source_sha256="third"),
            ]
        )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="f",
            timestamp=[release - pd.Timedelta(hours=1), release],
        )
    )
    name = "etf_share_change_1d"
    result = fund_flows.calculate(
        decisions,
        dict(
            public_fund_shares=pd.concat([original, version], ignore_index=True),
            trading_calendar=_calendar(days),
        ),
        columns=[name],
    )
    assert result[name].iloc[0] == pytest.approx(0.1)
    if kind == "conflicting_release":
        assert pd.isna(result[name].iloc[1])
    else:
        assert result[name].iloc[1] == pytest.approx(0.1)
    empty = fund_flows.share_states(
        pd.concat([original, version]), session_days=days[:0]
    )
    assert empty.empty


def test_pcf_replaces_whole_basket_and_uses_one_identical_complete_copy():
    days = pd.bdate_range("2024-01-02", periods=2)
    parts = pd.DataFrame(
        dict(
            fund_code="f",
            timestamp=days[-1],
            available_at=days[-1] + pd.Timedelta(hours=8),
            component_code=["a"],
            component_kind="equity_security",
            component_quantity_unit="share",
            component_quantity=10.0,
            creation_unit_fund_shares=100.0,
            source_sha256="first",
        )
    )
    newer = parts.assign(
        component_code="b",
        component_quantity=20.0,
        source_sha256="second",
        available_at=days[-1] + pd.Timedelta(hours=9),
    )
    headers = pd.concat([parts, newer])[
        ["fund_code", "timestamp", "available_at", "source_sha256"]
    ].assign(declared_component_count=1)
    # This identical extraction lacks a header; it cannot erase the first proof.
    duplicate = parts.assign(source_sha256="same-content")
    decisions = pd.DataFrame(
        [
            (asset, at)
            for at in (
                days[-1] + pd.Timedelta(hours=8, minutes=30),
                days[-1] + pd.Timedelta(hours=9, minutes=30),
            )
            for asset in ("a", "b")
        ],
        columns=["asset_id", "timestamp"],
    ).assign(market_id="a_share", avg_volume_20d=100.0)
    name = "stock_passive_flow_pressure_1d"
    result = fund_flows.calculate(
        decisions,
        dict(
            public_fund_shares=_shares(days),
            trading_calendar=_calendar(days),
            public_etf_pcf_components=pd.concat(
                [parts, duplicate, newer], ignore_index=True
            ),
            public_etf_pcf_baskets=headers,
        ),
        columns=[name],
    )
    np.testing.assert_allclose(result[name].to_numpy(float), [0.01, 0.0, 0.0, 0.02])
    future = parts.assign(
        fund_code="future",
        component_code="c",
        available_at=days[-1] + pd.Timedelta(days=10),
        source_sha256="future",
    )
    wider = fund_flows.calculate(
        decisions,
        dict(
            public_fund_shares=pd.concat(
                [_shares(days), _shares(days).assign(asset_id="future")]
            ),
            trading_calendar=_calendar(days),
            public_etf_pcf_components=pd.concat(
                [parts, duplicate, newer, future], ignore_index=True
            ),
            public_etf_pcf_baskets=headers,
        ),
        columns=[name],
    )
    np.testing.assert_allclose(
        wider[name].to_numpy(float), result[name].to_numpy(float)
    )


def test_five_session_pressure_revises_both_changed_daily_deltas():
    days = pd.bdate_range("2024-01-02", periods=7)
    release = days[-1] + pd.Timedelta(days=1)
    original = _shares(days)
    revised = original.iloc[[3]].assign(
        shares_outstanding=150.0, available_at=release, source_sha256="second"
    )
    parts = pd.DataFrame(
        dict(
            fund_code="f",
            timestamp=days,
            available_at=days + pd.Timedelta(hours=8),
            component_code="s",
            component_kind="equity_security",
            component_quantity_unit="share",
            component_quantity=10.0 * np.arange(1, 8),
            creation_unit_fund_shares=100.0,
        )
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="s",
            timestamp=[release - pd.Timedelta(hours=1), release],
            avg_volume_20d=100.0,
        )
    )
    name = "stock_passive_flow_pressure_5d"
    result = fund_flows.calculate(
        decisions,
        dict(
            public_fund_shares=pd.concat([original, revised]),
            trading_calendar=_calendar(days),
            public_etf_pcf_components=parts,
        ),
        columns=[name],
    )
    # Five deltas initially sum 10*(30+40+50+60+70); correction adds 20*40-20*50.
    np.testing.assert_allclose(result[name].to_numpy(float), [0.25, 0.23])


def test_futures_and_foreign_price_revisions_keep_prior_decisions():
    day = pd.Timestamp("2024-01-02")
    release = day + pd.Timedelta(days=2)
    futures = pd.DataFrame(
        dict(
            asset_id=["IF"],
            timestamp=day,
            product="IF",
            close=102.0,
            volume=40.0,
            open_interest=80.0,
            available_at=day + pd.Timedelta(hours=8),
        )
    )
    contracts = futures[["asset_id", "timestamp", "product", "available_at"]].assign(
        expiry_date=day + pd.Timedelta(days=10)
    )
    index = pd.DataFrame(dict(source_asset_id=["000300"], timestamp=day, close=100.0))
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[release - pd.Timedelta(hours=1), release],
        )
    )
    quotes = pd.concat([futures, futures.assign(close=104.0, available_at=release)])
    result = linked_markets.calculate(
        decisions,
        dict(
            public_futures_daily=quotes,
            public_futures_contracts=contracts,
            index_market=index,
        ),
        columns=["index_future_near_basis"],
    )
    np.testing.assert_allclose(result.index_future_near_basis, [0.02, 0.04])
    conflicts = pd.concat(
        [
            contracts,
            contracts.assign(available_at=release),
            contracts.assign(
                available_at=release, expiry_date=day + pd.Timedelta(days=20)
            ),
        ]
    )
    ambiguous = linked_markets.calculate(
        decisions,
        dict(
            public_futures_daily=futures,
            public_futures_contracts=conflicts,
            index_market=index,
        ),
        columns=["index_future_near_basis", "index_future_volume_to_oi"],
    )
    assert ambiguous.index_future_near_basis.iloc[0] == pytest.approx(0.02)
    assert pd.isna(ambiguous.index_future_near_basis.iloc[1])
    np.testing.assert_allclose(ambiguous.index_future_volume_to_oi, [0.5, 0.5])
    prices = pd.DataFrame(
        dict(
            asset_id="XLK",
            timestamp=[day, day + pd.Timedelta(days=1)],
            available_at=[day, day + pd.Timedelta(days=1)],
            adjusted_close=[100.0, 110.0],
        )
    )
    prices = pd.concat(
        [prices, prices.iloc[[-1]].assign(adjusted_close=120.0, available_at=release)]
    )
    result = linked_markets.calculate(
        decisions,
        dict(public_etf_prices=prices),
        columns=["us_sector_etf_return_shock_1d"],
        parameters={"asset_us_tickers": {"a": "XLK"}},
    )
    np.testing.assert_allclose(result.us_sector_etf_return_shock_1d, [0.1, 0.2])


def test_domestic_nav_and_venue_volume_revisions_do_not_mix_versions():
    day = pd.Timestamp("2024-01-02")
    release = day + pd.Timedelta(days=1)
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="f",
            timestamp=[release - pd.Timedelta(hours=1), release],
        )
    )
    nav = pd.DataFrame(
        dict(asset_id=["f"], timestamp=day, available_at=day, nav_cny=2.0)
    )
    prices = nav.rename(columns={"nav_cny": "close"}).assign(close=2.2)
    result = fund_flows.calculate(
        decisions,
        dict(
            public_fund_nav=pd.concat(
                [nav, nav.assign(nav_cny=2.1, available_at=release)]
            ),
            public_domestic_etf_prices=prices,
        ),
        columns=["domestic_etf_premium_discount"],
    )
    np.testing.assert_allclose(
        result.domestic_etf_premium_discount, [0.1, 2.2 / 2.1 - 1]
    )
    volume = pd.DataFrame(
        dict(
            asset_id=["SPY:N"],
            timestamp=day,
            available_at=day,
            short_volume=20.0,
            total_volume=100.0,
        )
    )
    result = linked_markets.calculate(
        decisions,
        dict(
            public_finra_short_volume=pd.concat(
                [volume, volume.assign(short_volume=30.0, available_at=release)]
            )
        ),
        columns=["us_short_sale_activity_fraction"],
    )
    np.testing.assert_allclose(result.us_short_sale_activity_fraction, [0.2, 0.3])
