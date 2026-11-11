"""Published hypotheses cannot consume not-yet-visible dependencies."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors import fund_flows, linked_markets


def calendar(days):
    dates = pd.date_range(days.min(), days.max())
    return pd.DataFrame(
        dict(calendar_date=dates, source_market_type=1, is_trading_day=dates.isin(days))
    )


@pytest.mark.parametrize("clock", ["late", "unknown"])
@pytest.mark.parametrize("width", [1, 5, 20])
def test_etf_growth_waits_for_actual_endpoint_dependencies(clock, width):
    days = pd.bdate_range("2024-01-02", periods=width + 1)
    release = days[-1] + pd.Timedelta(days=3)
    shares = pd.DataFrame(
        dict(
            asset_id="510300",
            timestamp=days,
            available_at=days + pd.Timedelta(hours=16),
            shares_outstanding=100 + 10 * np.arange(len(days)),
        )
    )
    shares.loc[0, "available_at"] = release if clock == "late" else pd.NaT
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="510300",
            timestamp=[days[-1] + pd.Timedelta(days=1), release],
        )
    )
    name = f"etf_share_change_{width}d"
    result = fund_flows.calculate(
        decisions,
        {"public_fund_shares": shares, "trading_calendar": calendar(days)},
        columns=[name],
    )
    assert pd.isna(result[name].iloc[0])
    if clock == "late":
        assert result[name].iloc[1] == pytest.approx(width / 10)
    else:
        assert result[name].isna().all()


def test_etf_missing_session_is_not_a_one_session_change_without_calendar():
    days = pd.bdate_range("2024-01-02", periods=3)
    shares = pd.DataFrame(
        dict(
            asset_id="510300",
            timestamp=days[[0, 2]],
            available_at=days[[0, 2]] + pd.Timedelta(hours=16),
            shares_outstanding=[100.0, 110.0],
        )
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="510300",
            timestamp=[days[-1] + pd.Timedelta(days=1)],
        )
    )
    name = "etf_share_change_1d"
    result = fund_flows.calculate(
        decisions, {"public_fund_shares": shares}, columns=[name]
    )
    assert result[name].isna().all()
    assert result.attrs["missing_source_dependencies"] == {name: ["trading_calendar"]}


@pytest.mark.parametrize("dependency", ["past_sector", "current_spy"])
def test_foreign_returns_and_relative_shocks_respect_dependency_clocks(dependency):
    prices = pd.DataFrame(
        dict(
            asset_id=["XLK", "XLK", "SPY", "SPY"],
            timestamp=pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
            available_at=pd.to_datetime(["2024-01-02 22:00", "2024-01-03 22:00"] * 2),
            adjusted_close=[100.0, 110.0, 100.0, 120.0],
        )
    )
    prices.loc[0 if dependency == "past_sector" else 3, "available_at"] = pd.Timestamp(
        "2024-01-05 22:00"
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=pd.to_datetime(["2024-01-04", "2024-01-06"]),
        )
    )
    names = ["us_sector_etf_return_shock_1d", "us_relative_sector_shock"]
    result = linked_markets.calculate(
        decisions,
        {"public_etf_prices": prices},
        columns=names,
        parameters={"asset_us_tickers": {"a": "XLK"}},
    )
    assert pd.isna(result.us_relative_sector_shock.iloc[0])
    if dependency == "past_sector":
        assert pd.isna(result.us_sector_etf_return_shock_1d.iloc[0])
    else:
        assert result.us_sector_etf_return_shock_1d.iloc[0] == pytest.approx(0.1)
    assert result.us_relative_sector_shock.iloc[1] == pytest.approx(-0.1)


def test_finra_change_does_not_use_an_unreleased_past_fraction():
    days = pd.bdate_range("2024-01-02", periods=6)
    frame = pd.DataFrame(
        dict(
            asset_id="SPY:market",
            timestamp=days,
            available_at=days + pd.Timedelta(hours=22),
            short_volume=[10, 20, 30, 40, 50, 60],
            total_volume=100,
        )
    )
    frame.loc[0, "available_at"] = pd.NaT
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[days[-1] + pd.Timedelta(days=1)],
        )
    )
    result = linked_markets.calculate(
        decisions,
        {"public_finra_short_volume": frame},
        columns=list(linked_markets.US[:2]),
    )
    assert result.us_short_sale_activity_fraction.iloc[0] == pytest.approx(0.6)
    assert pd.isna(result.us_short_sale_activity_change.iloc[0])


def test_etf_unknown_calendar_breaks_history_without_blocking_later_recovery():
    days = pd.bdate_range("2024-01-02", periods=4)
    shares = pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=days,
            available_at=days,
            shares_outstanding=[100.0, 110.0, 120.0, 130.0],
        )
    )
    trading = calendar(days).drop(index=1)
    decisions = pd.DataFrame(
        dict(market_id="a_share", asset_id="f", timestamp=days + pd.Timedelta(hours=16))
    )
    result = fund_flows.calculate(
        decisions,
        {"public_fund_shares": shares, "trading_calendar": trading},
        columns=["etf_share_change_1d"],
    )
    assert pd.isna(result.etf_share_change_1d.iloc[2])
    assert result.etf_share_change_1d.iloc[3] == pytest.approx(130 / 120 - 1)


def test_etf_surprise_waits_for_history_but_one_day_change_does_not():
    days = pd.bdate_range("2024-01-02", periods=22)
    release = days[-1] + pd.Timedelta(days=5)
    shares = pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=days,
            available_at=days,
            shares_outstanding=100 + np.arange(len(days)) ** 2,
        )
    )
    shares.loc[0, "available_at"] = release
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="f",
            timestamp=[days[-1] + pd.Timedelta(hours=16), release],
        )
    )
    result = fund_flows.calculate(
        decisions,
        {"public_fund_shares": shares, "trading_calendar": calendar(days)},
        columns=["etf_share_change_1d", "etf_share_change_surprise_20d"],
    )
    assert result.etf_share_change_1d.iloc[0] == pytest.approx(541 / 500 - 1)
    assert pd.isna(result.etf_share_change_surprise_20d.iloc[0])
    changes = (
        shares.shares_outstanding.to_numpy()[1:]
        / shares.shares_outstanding.to_numpy()[:-1]
        - 1
    )
    assert result.etf_share_change_surprise_20d.iloc[1] == pytest.approx(
        (changes[-1] - changes[:-1].mean()) / changes[:-1].std(ddof=1)
    )


def test_futures_change_waits_for_past_data_without_blocking_current_basis():
    days = pd.bdate_range("2024-01-02", periods=2)
    futures = pd.DataFrame(
        dict(
            asset_id="IF",
            product="IF",
            timestamp=days,
            available_at=[pd.NaT, days[1] + pd.Timedelta(hours=16)],
            close=[101.0, 102.0],
            open_interest=[100.0, 110.0],
            volume=20.0,
        )
    )
    contracts = pd.DataFrame(
        dict(
            asset_id=["IF"],
            product="IF",
            timestamp=[days[0]],
            available_at=[days[0]],
            expiry_date=pd.Timestamp("2024-06-30"),
        )
    )
    index = pd.DataFrame(dict(source_asset_id="000300", timestamp=days, close=100.0))
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[days[-1] + pd.Timedelta(days=1)],
        )
    )
    names = [
        "index_future_near_basis",
        "index_future_basis_change_1d",
        "index_future_oi_change_1d",
    ]
    result = linked_markets.calculate(
        decisions,
        dict(
            public_futures_daily=futures,
            public_futures_contracts=contracts,
            index_market=index,
        ),
        columns=names,
    )
    assert result.index_future_near_basis.iloc[0] == pytest.approx(0.02)
    assert result[names[1:]].isna().all(axis=None)


def test_native_prepare_persists_dependency_safe_values(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    days = pd.bdate_range("2024-01-02", periods=2)
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="f",
            timestamp=pd.to_datetime(["2024-01-04", "2024-01-06"]),
            signal_session=days[-1],
            execution_session=days[-1],
            return_available_at=pd.NaT,
        )
    )
    shares = pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=days,
            available_at=pd.to_datetime(["2024-01-05", "2024-01-03"]),
            shares_outstanding=[100.0, 110.0],
        )
    )
    inputs = {}
    for name, frame, key, clock in (
        ("decision_panel", decisions, ("asset_id", "timestamp"), "timestamp"),
        ("public_fund_shares", shares, ("asset_id", "timestamp"), "timestamp"),
        (
            "trading_calendar",
            calendar(days),
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
    name = "etf_share_change_1d"
    path = prepare(
        dict(
            market_id="a_share",
            data_id="fund_flow_factors",
            inputs=inputs,
            factor_columns=[name],
            time_window=dict(start="2024-01-04", end="2024-01-06"),
            output_root=str(tmp_path / "factor"),
        )
    )
    result = read_dataset(path)
    assert pd.isna(result[name].iloc[0])
    assert result[name].iloc[1] == pytest.approx(0.1)


def test_passive_five_day_pressure_waits_for_every_historical_flow():
    days = pd.bdate_range("2024-01-02", periods=7)
    release = days[-1] + pd.Timedelta(days=3)
    shares = pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=days,
            available_at=days,
            shares_outstanding=100 + 10 * np.arange(7),
        )
    )
    shares.loc[2, "available_at"] = release
    parts = pd.DataFrame(
        dict(
            fund_code="f",
            timestamp=days,
            available_at=days,
            component_code="s",
            component_kind="equity_security",
            component_quantity_unit="share",
            component_quantity=10.0,
            creation_unit_fund_shares=100.0,
        )
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="s",
            timestamp=[days[-1] + pd.Timedelta(days=1), release],
            avg_volume_20d=100.0,
        )
    )
    result = fund_flows.calculate(
        decisions,
        dict(
            public_fund_shares=shares,
            public_etf_pcf_components=parts,
            trading_calendar=calendar(days),
        ),
        columns=["stock_passive_flow_pressure_1d", "stock_passive_flow_pressure_5d"],
    )
    assert result.stock_passive_flow_pressure_1d.iloc[0] == pytest.approx(0.01)
    assert pd.isna(result.stock_passive_flow_pressure_5d.iloc[0])
    assert result.stock_passive_flow_pressure_5d.iloc[1] == pytest.approx(0.05)


def test_redemption_stress_waits_for_history_not_required_by_allocation():
    from tests.research.test_fund_flows import cells

    reports = []
    for i, day in enumerate(pd.date_range("2019-12-31", periods=5, freq="YE")):
        report = cells().assign(
            timestamp=day,
            report_period_end=day,
            value_observation_date=day,
            available_at=day + pd.Timedelta(days=30),
        )
        report.loc[report.field.eq("period_redemption_fund_shares"), "value"] = 10.0 * (
            i + 1
        )
        report.loc[report.field.eq("period_subscription_fund_shares"), "value"] = (
            10.0 * (i + 2)
        )
        reports.append(report)
    release = pd.Timestamp("2024-02-20")
    reports[0]["available_at"] = release
    frame = pd.concat(reports, ignore_index=True)
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[pd.Timestamp("2024-02-01"), release],
        )
    )
    result = fund_flows.calculate(
        decisions,
        dict(public_fund_financial_cells=frame),
        columns=["cn_fund_redemption_stress", "cn_fund_equity_allocation"],
    )
    assert result.cn_fund_equity_allocation.iloc[0] == pytest.approx(0.5)
    assert pd.isna(result.cn_fund_redemption_stress.iloc[0])
    assert result.cn_fund_redemption_stress.iloc[1] == pytest.approx(
        2.5 / np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    )


def test_finra_percentile_recovers_after_unknown_clock_leaves_window():
    days = pd.bdate_range("2020-01-02", periods=757)
    frame = pd.DataFrame(
        dict(
            asset_id="SPY:market",
            timestamp=days,
            available_at=days,
            short_volume=np.arange(len(days)) + 1.0,
            total_volume=1000.0,
        )
    )
    frame.loc[0, "available_at"] = pd.NaT
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=days[-2:] + pd.Timedelta(hours=16),
        )
    )
    result = linked_markets.calculate(
        decisions,
        dict(public_finra_short_volume=frame),
        columns=["us_short_sale_activity_percentile"],
    )
    assert pd.isna(result.us_short_sale_activity_percentile.iloc[0])
    assert result.us_short_sale_activity_percentile.iloc[1] == pytest.approx(1.0)


def test_etf_unrelated_market_does_not_fill_calendar_gap():
    days = pd.bdate_range("2024-01-02", periods=3)
    shares = pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=days[[0, 2]],
            available_at=days[[0, 2]],
            shares_outstanding=[100.0, 110.0],
        )
    )
    trading = pd.concat(
        [calendar(days).drop(index=1), calendar(days).assign(source_market_type=2)],
        ignore_index=True,
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="f",
            timestamp=[days[-1] + pd.Timedelta(hours=16)],
        )
    )
    result = fund_flows.calculate(
        decisions,
        dict(public_fund_shares=shares, trading_calendar=trading),
        columns=["etf_share_change_1d"],
    )
    assert result.etf_share_change_1d.isna().all()


@pytest.mark.parametrize(
    "metadata", ["unknown", "late", "partial_late", "far_quote_unknown"]
)
def test_futures_metadata_does_not_remove_quotes_or_add_unrelated_clocks(metadata):
    date = pd.Timestamp("2024-01-02")
    release = date + pd.Timedelta(days=2)
    quotes = pd.DataFrame(
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
    contracts = pd.DataFrame(
        dict(
            asset_id=quotes.asset_id,
            timestamp=date,
            product="IF",
            expiry_date=[date + pd.Timedelta(days=10), date + pd.Timedelta(days=40)],
            available_at=date,
        )
    )
    if metadata == "unknown":
        contracts["available_at"] = pd.NaT
    elif metadata == "late":
        contracts["available_at"] = release
    elif metadata == "partial_late":
        contracts.loc[0, "available_at"] = release
    else:
        quotes.loc[1, "available_at"] = pd.NaT
    index = pd.DataFrame(
        dict(source_asset_id=["000300"], timestamp=[date], close=[100.0])
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[date + pd.Timedelta(days=1), release + pd.Timedelta(days=1)],
        )
    )
    names = [
        "index_future_near_basis",
        "index_future_calendar_spread",
        "index_future_roll_concentration",
        "index_future_volume_to_oi",
    ]
    result = linked_markets.calculate(
        decisions,
        dict(
            public_futures_daily=quotes,
            public_futures_contracts=contracts,
            index_market=index,
        ),
        columns=names,
    )
    if metadata == "far_quote_unknown":
        assert result.index_future_near_basis.tolist() == pytest.approx([0.02, 0.02])
        assert result[names[1:]].isna().all(axis=None)
    else:
        assert result.index_future_roll_concentration.tolist() == pytest.approx(
            [0.68, 0.68]
        )
        assert result.index_future_volume_to_oi.tolist() == pytest.approx([1.0, 1.0])
        assert pd.isna(result.index_future_near_basis.iloc[0])
        if metadata == "unknown":
            assert result[names[:2]].isna().all(axis=None)
        else:
            assert result.index_future_near_basis.iloc[1] == pytest.approx(0.02)
            assert result.index_future_calendar_spread.iloc[1] == pytest.approx(
                104 / 102 - 1
            )


@pytest.mark.parametrize("history", ["unknown_clock", "reverse_input"])
def test_fund_stress_uses_only_known_periods_in_chronological_order(history):
    from tests.research.test_fund_flows import cells

    reports = []
    for i, date in enumerate(pd.date_range("2018-12-31", periods=6, freq="YE")):
        report = cells().assign(
            timestamp=date,
            report_period_end=date,
            value_observation_date=date,
            available_at=date + pd.Timedelta(days=30),
        )
        report.loc[report.field.eq("period_redemption_fund_shares"), "value"] = 10.0 * (
            i + 1
        )
        report.loc[report.field.eq("period_subscription_fund_shares"), "value"] = (
            10.0 * (i + 2)
        )
        reports.append(report)
    if history == "unknown_clock":
        reports[2]["available_at"] = pd.NaT
    else:
        reports.reverse()
    decisions = pd.DataFrame(
        dict(market_id="a_share", asset_id="a", timestamp=[pd.Timestamp("2024-02-01")])
    )
    result = fund_flows.calculate(
        decisions,
        dict(public_fund_financial_cells=pd.concat(reports, ignore_index=True)),
        columns=["cn_fund_redemption_stress", "cn_fund_equity_allocation"],
    )
    assert result.cn_fund_equity_allocation.iloc[0] == pytest.approx(0.5)
    past = [1.0, 2.0, 4.0, 5.0] if history == "unknown_clock" else [1, 2, 3, 4, 5]
    assert result.cn_fund_redemption_stress.iloc[0] == pytest.approx(
        (6 - np.mean(past)) / np.std(past, ddof=1)
    )


@pytest.mark.parametrize("width", [1, 5])
def test_foreign_returns_keep_missing_endpoint_on_common_observed_sessions(width):
    days = pd.bdate_range("2024-01-02", periods=width + 1)
    prices = pd.concat(
        [
            pd.DataFrame(
                dict(
                    asset_id="SPY",
                    timestamp=days,
                    available_at=days + pd.Timedelta(hours=22),
                    adjusted_close=100.0,
                )
            ),
            pd.DataFrame(
                dict(
                    asset_id="XLK",
                    timestamp=days.delete(width - 1),
                    available_at=days.delete(width - 1) + pd.Timedelta(hours=22),
                    adjusted_close=100.0 + np.arange(width),
                )
            ),
        ],
        ignore_index=True,
    )
    if width == 1:
        # A prior session outside the return endpoint must not be substituted.
        prices = pd.concat(
            [
                prices,
                pd.DataFrame(
                    dict(
                        asset_id=["XLK"],
                        timestamp=[days[0] - pd.Timedelta(days=1)],
                        available_at=[days[0] - pd.Timedelta(hours=2)],
                        adjusted_close=[90.0],
                    )
                ),
            ],
            ignore_index=True,
        )
    else:
        # Five-session endpoints exist despite a missing interior quote.
        prices.loc[
            prices.asset_id.eq("XLK") & prices.timestamp.eq(days[-1]), "adjusted_close"
        ] = 110.0
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[days[-1] + pd.Timedelta(days=1)],
        )
    )
    result = linked_markets.calculate(
        decisions,
        dict(public_etf_prices=prices),
        columns=[
            "us_sector_etf_return_shock_1d",
            "us_sector_etf_return_shock_5d",
            "us_relative_sector_shock",
        ],
        parameters={"asset_us_tickers": {"a": "XLK"}},
    )
    if width == 1:
        assert pd.isna(result.us_sector_etf_return_shock_1d.iloc[0])
        assert pd.isna(result.us_relative_sector_shock.iloc[0])
    else:
        # The as-of owner can retain the last mature one-session return,
        # but cannot replace it with a compressed two-session return.
        assert result.us_sector_etf_return_shock_1d.iloc[0] == pytest.approx(
            103 / 102 - 1
        )
        assert result.us_relative_sector_shock.iloc[0] == pytest.approx(103 / 102 - 1)
        assert result.us_sector_etf_return_shock_5d.iloc[0] == pytest.approx(0.1)


def test_futures_same_contract_changes_do_not_skip_a_missing_quote():
    days = pd.bdate_range("2024-01-02", periods=3)
    quotes = pd.DataFrame(
        dict(
            asset_id="IF2401",
            product="IF",
            timestamp=days[[0, 2]],
            available_at=days[[0, 2]] + pd.Timedelta(hours=16),
            close=[101.0, 102.0],
            open_interest=[100.0, 110.0],
            volume=20.0,
        )
    )
    contracts = pd.DataFrame(
        dict(
            asset_id=["IF2401"],
            product="IF",
            timestamp=[days[0]],
            available_at=[days[0]],
            expiry_date=[days[-1] + pd.Timedelta(days=10)],
        )
    )
    index = pd.DataFrame(dict(source_asset_id="000300", timestamp=days, close=100.0))
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[days[-1] + pd.Timedelta(days=1)],
        )
    )
    names = [
        "index_future_near_basis",
        "index_future_basis_change_1d",
        "index_future_oi_change_1d",
    ]
    result = linked_markets.calculate(
        decisions,
        dict(
            public_futures_daily=quotes,
            public_futures_contracts=contracts,
            index_market=index,
        ),
        columns=names,
    )
    assert result.index_future_near_basis.iloc[0] == pytest.approx(0.02)
    assert result[names[1:]].isna().all(axis=None)


def test_futures_volume_and_oi_do_not_require_expiry_or_spot_sources():
    days = pd.bdate_range("2024-01-02", periods=2)
    quotes = pd.DataFrame(
        dict(
            asset_id="IF2401",
            product="IF",
            timestamp=days,
            available_at=days + pd.Timedelta(hours=16),
            close=102.0,
            open_interest=[100.0, 110.0],
            volume=[10.0, 11.0],
        )
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=[days[-1] + pd.Timedelta(days=1)],
        )
    )
    names = [
        "index_future_oi_change_1d",
        "index_future_volume_to_oi",
        "index_future_roll_concentration",
        "index_future_near_basis",
    ]
    result = linked_markets.calculate(
        decisions, dict(public_futures_daily=quotes), columns=names
    )
    assert result[names[:3]].iloc[0].to_numpy(float) == pytest.approx([0.1, 0.1, 1.0])
    assert pd.isna(result.index_future_near_basis.iloc[0])
    assert result.attrs["missing_source_dependencies"] == {
        "index_future_near_basis": ["public_futures_contracts", "index_market"]
    }


@pytest.mark.parametrize(
    "metadata",
    ["late", "future_observation", "absent", "edge_oi_only", "no_spot_spread"],
)
def test_native_futures_prepare_preserves_metadata_and_quote_dependency_domains(
    tmp_path, metadata
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    date = pd.Timestamp("2024-01-02")
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=pd.to_datetime(["2024-01-03", "2024-01-06"]),
            signal_session=date,
            execution_session=date,
            return_available_at=pd.NaT,
        )
    )
    quotes = pd.DataFrame(
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
    contracts = pd.DataFrame(
        dict(
            asset_id=quotes.asset_id,
            timestamp=date,
            product="IF",
            expiry_date=[date + pd.Timedelta(days=10), date + pd.Timedelta(days=40)],
            available_at=date + pd.Timedelta(days=2) if metadata == "late" else date,
        )
    )
    if metadata == "future_observation":
        contracts = pd.concat(
            [
                contracts,
                contracts.iloc[:1].assign(
                    timestamp=date + pd.Timedelta(days=1),
                    available_at=date + pd.Timedelta(days=1),
                    expiry_date=date + pd.Timedelta(days=60),
                ),
            ],
            ignore_index=True,
        )
    if metadata == "edge_oi_only":
        quotes = pd.concat(
            [
                quotes,
                quotes.iloc[1:].assign(
                    timestamp=date + pd.Timedelta(days=1),
                    available_at=date + pd.Timedelta(days=1, hours=16),
                    close=105.0,
                ),
            ],
            ignore_index=True,
        )
        contracts["open_date"] = date
    sources = dict(decision_panel=decisions, public_futures_daily=quotes)
    if metadata != "absent":
        sources["public_futures_contracts"] = contracts
        if metadata != "no_spot_spread":
            sources["index_market"] = pd.DataFrame(
                dict(source_asset_id=["000300"], timestamp=[date], close=[100.0])
            )
    inputs = {}
    for name, frame in sources.items():
        key = ("source_asset_id" if name == "index_market" else "asset_id", "timestamp")
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
    names = [
        "index_future_near_basis",
        "index_future_roll_concentration",
        "index_future_volume_to_oi",
    ]
    if metadata == "edge_oi_only":
        names[0] = "index_future_oi_change_1d"
    elif metadata == "no_spot_spread":
        names[0] = "index_future_calendar_spread"
    path = prepare(
        dict(
            market_id="a_share",
            data_id="linked_market_factors",
            inputs=inputs,
            factor_columns=names,
            time_window=dict(start="2024-01-03", end="2024-01-06"),
            output_root=str(tmp_path / "factor"),
        )
    )
    result = read_dataset(path)
    assert result.index_future_roll_concentration.to_numpy(float) == pytest.approx(
        [0.68, 0.68]
    )
    assert result.index_future_volume_to_oi.to_numpy(float) == pytest.approx([1.0, 1.0])
    if metadata == "edge_oi_only":
        assert result.index_future_oi_change_1d.isna().all()
    elif metadata == "no_spot_spread":
        assert result.index_future_calendar_spread.to_numpy(float) == pytest.approx(
            [104 / 102 - 1, 104 / 102 - 1]
        )
    elif metadata == "absent":
        assert result.index_future_near_basis.isna().all()
    elif metadata == "late":
        assert pd.isna(result.index_future_near_basis.iloc[0])
        assert result.index_future_near_basis.iloc[1] == pytest.approx(0.02)
    else:
        assert result.index_future_near_basis.to_numpy(float) == pytest.approx(
            [0.02, 0.02]
        )


@pytest.mark.parametrize(
    "missing",
    ["opening_quote", "closing_quote", "all_quotes", "closing_without_open_date"],
)
def test_futures_declared_listing_window_does_not_drop_missing_contract(missing):
    days = pd.bdate_range("2024-01-02", periods=2)
    near_days = (
        days[:1]
        if missing.startswith("closing")
        else days[1:]
        if missing == "opening_quote"
        else days[:0]
    )
    quotes = pd.concat(
        [
            pd.DataFrame(
                dict(
                    asset_id="IF2401",
                    product="IF",
                    timestamp=near_days,
                    available_at=near_days + pd.Timedelta(hours=16),
                    close=102.0,
                    open_interest=80.0,
                    volume=40.0,
                )
            ),
            pd.DataFrame(
                dict(
                    asset_id="IF2402",
                    product="IF",
                    timestamp=days,
                    available_at=days + pd.Timedelta(hours=16),
                    close=[104.0, 105.0],
                    open_interest=20.0,
                    volume=60.0,
                )
            ),
        ],
        ignore_index=True,
    )
    contracts = pd.DataFrame(
        dict(
            asset_id=["IF2401", "IF2402"],
            product="IF",
            timestamp=days[0],
            available_at=days[0],
            open_date=days[0],
            expiry_date=[
                days[-1] + pd.Timedelta(days=10),
                days[-1] + pd.Timedelta(days=40),
            ],
        )
    )
    if missing == "closing_without_open_date":
        contracts = contracts.drop(columns="open_date")
    index = pd.DataFrame(dict(source_asset_id="000300", timestamp=days, close=100.0))
    decisions = pd.DataFrame(
        dict(market_id="a_share", asset_id="a", timestamp=days + pd.Timedelta(hours=20))
    )
    names = [
        "index_future_near_basis",
        "index_future_oi_change_1d",
        "index_future_roll_concentration",
        "index_future_volume_to_oi",
    ]
    result = linked_markets.calculate(
        decisions,
        dict(
            public_futures_daily=quotes,
            public_futures_contracts=contracts,
            index_market=index,
        ),
        columns=names,
    )
    assert result.index_future_oi_change_1d.isna().all()
    if missing == "all_quotes":
        assert result[names].isna().all(axis=None)
    elif missing == "opening_quote":
        assert result[names].iloc[0].isna().all()
        assert result.index_future_near_basis.iloc[1] == pytest.approx(0.02)
        assert result.index_future_roll_concentration.iloc[1] == pytest.approx(0.68)
    else:
        # The latest fully observed state remains Jan 2, not a Jan 3 far-only curve.
        assert result.index_future_near_basis.to_numpy(float) == pytest.approx(
            [0.02, 0.02]
        )
        assert result.index_future_roll_concentration.to_numpy(float) == pytest.approx(
            [0.68, 0.68]
        )
        assert result.index_future_volume_to_oi.to_numpy(float) == pytest.approx(
            [1.0, 1.0]
        )


@pytest.mark.parametrize("update", ["revision", "late_first_period"])
def test_redemption_stress_counts_only_visible_periods_and_versions(update):
    from tests.research.test_fund_flows import cells

    reports = []
    for i, date in enumerate(pd.date_range("2019-12-31", periods=5, freq="YE")):
        report = cells().assign(
            timestamp=date,
            report_period_end=date,
            value_observation_date=date,
            available_at=date + pd.Timedelta(days=30),
        )
        report.loc[report.field.eq("period_redemption_fund_shares"), "value"] = 10 * (
            i + 1
        )
        report.loc[report.field.eq("period_subscription_fund_shares"), "value"] = 10 * (
            i + 2
        )
        reports.append(report)
    revision = reports[0].assign(available_at=pd.Timestamp("2024-03-01"))
    revision.loc[revision.field.eq("period_redemption_fund_shares"), "value"] = 30
    revision.loc[revision.field.eq("period_subscription_fund_shares"), "value"] = 40
    after = [3, 2, 3, 4]
    if update == "late_first_period":
        period = pd.Timestamp("2021-06-30")
        revision = reports[0].assign(
            timestamp=period,
            report_period_end=period,
            value_observation_date=period,
            available_at=pd.Timestamp("2024-03-01"),
        )
        # Its half-year flow rate equals twice the same nominal annual rate.
        after = [1, 2, 2, 3, 4]
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=pd.to_datetime(["2024-02-01", "2024-03-02"]),
        )
    )
    result = fund_flows.calculate(
        decisions,
        dict(
            public_fund_financial_cells=pd.concat(
                [*reports, revision], ignore_index=True
            )
        ),
        columns=["cn_fund_redemption_stress", "cn_fund_equity_allocation"],
    )
    # Scaling every monthly rate by 120 does not change either sample z-score.
    assert result.cn_fund_redemption_stress.to_numpy(float) == pytest.approx(
        [
            2.5 / np.std([1, 2, 3, 4], ddof=1),
            (5 - np.mean(after)) / np.std(after, ddof=1),
        ]
    )
    assert result.cn_fund_equity_allocation.to_numpy(float) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("missing", ["short_volume", "total_volume"])
def test_finra_activity_keeps_unknown_cells_in_the_declared_venue_group(missing):
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame(
        dict(
            asset_id=["SPY:Q", "SPY:N"],
            timestamp=date,
            available_at=date + pd.Timedelta(hours=22),
            short_volume=[40.0, 20.0],
            total_volume=[100.0, 100.0],
        )
    )
    decisions = pd.DataFrame(
        dict(market_id="a_share", asset_id="a", timestamp=[date + pd.Timedelta(days=1)])
    )
    name = "us_short_sale_activity_fraction"
    known = linked_markets.calculate(
        decisions, dict(public_finra_short_volume=frame), columns=[name]
    )
    assert known[name].iloc[0] == pytest.approx(0.3)
    frame.loc[1, missing] = np.nan
    unknown = linked_markets.calculate(
        decisions, dict(public_finra_short_volume=frame), columns=[name]
    )
    assert pd.isna(unknown[name].iloc[0])
