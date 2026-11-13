"""Disclosure clocks, report units and ETF observation dates are independent."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.fund_flows import (
    calculate,
    financial_states,
    nport_flow_states,
    share_states,
    us_ownership_states,
)


def _calendar(days):
    dates = pd.date_range(days.min(), days.max())
    return pd.DataFrame(
        dict(calendar_date=dates, source_market_type=1, is_trading_day=dates.isin(days))
    )


def cells():
    values = {
        ("balance_sheet", "total_assets"): 1000.0,
        ("balance_sheet", "net_assets"): 900.0,
        ("portfolio_asset_composition", "stock_assets"): 500.0,
        ("portfolio_asset_composition", "bank_deposits_and_settlement_reserves"): 100.0,
        ("fund_share_movement", "period_begin_fund_shares"): 100.0,
        ("fund_share_movement", "period_end_fund_shares"): 110.0,
        ("fund_share_movement", "period_subscription_fund_shares"): 20.0,
        ("fund_share_movement", "period_redemption_fund_shares"): 10.0,
        ("fund_share_movement", "period_split_change_fund_shares"): 0.0,
    }
    return pd.DataFrame(
        [
            dict(
                economic_portfolio_id="f",
                fund_code="001001",
                timestamp=pd.Timestamp("2024-06-30"),
                report_period_end=pd.Timestamp("2024-06-30"),
                value_observation_date=pd.Timestamp("2024-06-30"),
                available_at=pd.Timestamp("2024-08-20"),
                table_kind=kind,
                share_class_header=np.nan,
                field=field,
                value=value,
            )
            for (kind, field), value in values.items()
        ]
    )


def test_report_unit_flows_cash_and_equity_do_not_need_nav_or_holdings():
    frame = cells()
    latest = financial_states(frame).iloc[0]
    assert latest.cn_fund_net_flow_rate == pytest.approx(0.1 / 6)
    assert latest.cn_fund_cash_buffer == pytest.approx(0.1)
    assert latest.cn_fund_equity_allocation == pytest.approx(0.5)
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=pd.to_datetime(["2024-08-19", "2024-08-21"]),
        )
    )
    selected = (
        "cn_fund_equity_allocation",
        "cn_fund_cash_buffer",
        "cn_fund_net_flow_rate",
        "cn_fund_holder_concentration",
    )
    result = calculate(
        decisions, dict(public_fund_financial_cells=frame), columns=selected
    )
    assert result.loc[0, list(selected)].isna().all()
    np.testing.assert_allclose(
        result.loc[1, list(selected[:3])].to_numpy(float), [0.5, 0.1, 0.1 / 6]
    )
    assert result.cn_fund_holder_concentration.isna().all()
    invalid = frame.copy()
    invalid.loc[invalid.field.eq("period_end_fund_shares"), "value"] = 999
    broken = financial_states(invalid).iloc[0]
    assert pd.isna(broken.cn_fund_net_flow_rate)
    assert broken.cn_fund_equity_allocation == 0.5
    unknown = frame.copy()
    unknown["value"] = unknown.value.astype("Float64")
    unknown.loc[unknown.field.eq("period_redemption_fund_shares"), "value"] = pd.NA
    partial = financial_states(unknown).iloc[0]
    assert pd.isna(partial.cn_fund_net_flow_rate)
    assert partial.cn_fund_equity_allocation == 0.5


def test_etf_share_growth_and_nav_premium_require_same_economic_day():
    dates = pd.bdate_range("2024-01-02", periods=24)
    shares = pd.DataFrame(
        dict(
            asset_id="510300",
            timestamp=dates,
            available_at=dates + pd.Timedelta(days=1),
            shares_outstanding=100 + np.arange(24) * 10,
        )
    )
    states = share_states(shares, session_days=dates)
    assert states.etf_share_change_5d.iloc[-1] == pytest.approx(330 / 280 - 1)
    d = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="510300",
            timestamp=[dates[-1] + pd.Timedelta(days=1)],
            signal_session=[dates[-1]],
            close=[2.2],
        )
    )
    nav = pd.DataFrame(
        dict(
            asset_id="510300",
            timestamp=[dates[-1]],
            available_at=[dates[-1] + pd.Timedelta(hours=16)],
            nav_cny=[2.0],
        )
    )
    prices = nav.rename(columns={"nav_cny": "close"}).assign(close=2.2)
    result = calculate(
        d,
        dict(public_fund_nav=nav, public_domestic_etf_prices=prices),
        columns=["domestic_etf_premium_discount"],
    )
    assert result.domestic_etf_premium_discount.iloc[0] == pytest.approx(0.1)
    unknown = nav.assign(available_at=pd.NaT)
    assert (
        calculate(
            d,
            dict(public_fund_nav=unknown, public_domestic_etf_prices=prices),
            columns=["domestic_etf_premium_discount"],
        )
        .domestic_etf_premium_discount.isna()
        .all()
    )
    nav["timestamp"] -= pd.Timedelta(days=1)
    assert (
        calculate(
            d,
            dict(public_fund_nav=nav, public_domestic_etf_prices=prices),
            columns=["domestic_etf_premium_discount"],
        )
        .domestic_etf_premium_discount.isna()
        .all()
    )


@pytest.mark.parametrize("unknown", [None, "shares_or_principal", "value_usd"])
def test_us_complete_reports_supplements_restatements_and_empty_positions(unknown):
    submissions = [
        ("a0", "a", "2024-01-31", "2024-02-01", "N", 0, ""),
        ("b0", "b", "2024-01-31", "2024-02-01", "N", 0, ""),
        ("a1", "a", "2024-01-31", "2024-02-02", "Y", 1, "NEW HOLDINGS"),
        ("a2", "a", "2024-01-31", "2024-02-03", "Y", 2, "RESTATEMENT"),
        ("a3", "a", "2024-04-30", "2024-05-01", "N", 0, ""),
        ("a4", "a", "2024-01-31", "2024-05-02", "Y", 3, "RESTATEMENT"),
    ]
    filings = pd.DataFrame(
        submissions,
        columns=[
            "accession",
            "manager_cik",
            "timestamp",
            "available_at",
            "is_amendment",
            "amendment_number",
            "amendment_type",
        ],
    )
    filings["timestamp"] = pd.to_datetime(filings.timestamp)
    filings["available_at"] = pd.to_datetime(filings.available_at)
    filings["filing_type"] = np.where(
        filings.is_amendment.eq("Y"), "13F-HR/A", "13F-HR"
    )
    h = pd.DataFrame(
        [
            ("a0", "s", 10.0, 100.0),
            ("a0", "x", 20.0, 200.0),
            ("b0", "s", 10.0, 100.0),
            ("a1", "s", 5.0, 50.0),
            ("a2", "s", 2.0, 20.0),
            ("a2", "x", 3.0, 30.0),
            ("a4", "s", 100.0, 1000.0),
        ],
        columns=["accession", "cusip", "shares_or_principal", "value_usd"],
    )
    if unknown is not None:
        extra = h.iloc[[0]].assign(shares_or_principal=0.0, value_usd=0.0)
        extra[unknown] = np.nan
        h = pd.concat([h, extra], ignore_index=True)
    h["share_type"] = "SH"
    h["put_call"] = None
    states = us_ownership_states(h, filings, assets={"SPY": "s", "XLF": "x"})
    spy = states.loc[states.asset_id.eq("SPY")].set_index("timestamp")
    field = "us_institution_ownership_concentration"
    if unknown == "shares_or_principal":
        assert spy.loc[:"2024-02-02", field].isna().all()
    else:
        assert spy.loc["2024-02-01", field] == pytest.approx(0.5)
        assert spy.loc["2024-02-02", field] == pytest.approx((15**2 + 10**2) / 25**2)
    allocation = "us_institution_sector_allocation"
    if unknown == "value_usd":
        assert states.loc[states.timestamp.le("2024-02-02"), allocation].isna().all()
    else:
        assert spy.loc["2024-02-01", allocation] == pytest.approx(0.5)
    assert spy.loc["2024-02-03", field] == pytest.approx((2**2 + 10**2) / 12**2)
    assert spy.loc["2024-05-01", field] == 1
    assert spy.loc["2024-05-02", field] == 1


@pytest.mark.parametrize("unknown", [False, True])
def test_nport_flows_use_prior_net_assets_and_empty_reports_replace_positions(unknown):
    funds = pd.DataFrame(
        [
            ("a0", "a", "2023-12-31", "2024-02-01", 100.0),
            ("a1", "a", "2024-03-31", "2024-05-01", 200.0),
            ("a2", "a", "2024-06-30", "2024-08-01", 300.0),
            ("b0", "b", "2023-12-31", "2024-02-01", 100.0),
            ("b1", "b", "2024-03-31", "2024-05-01", 200.0),
            ("b2", "b", "2024-06-30", "2024-08-01", 300.0),
        ],
        columns=[
            "accession",
            "series_id",
            "timestamp",
            "available_at",
            "net_assets_usd",
        ],
    )
    funds["timestamp"] = pd.to_datetime(funds.timestamp)
    funds["available_at"] = pd.to_datetime(funds.available_at)
    funds["filing_type"] = "NPORT-P"
    for kind in ("sales", "reinvestment", "redemption"):
        for month in (1, 2, 3):
            funds[f"{kind}_month_{month}_usd"] = 0.0
    funds["sales_month_1_usd"] = 10.0
    holdings = pd.DataFrame(
        [
            ("a0", "a", "s", 20.0, "Long"),
            ("a0", "a", "s", 999.0, "Short"),
            ("b0", "b", "s", 40.0, "Long"),
            ("b1", "b", "s", 50.0, "Long"),
        ],
        columns=["accession", "series_id", "cusip", "value", "payoff_profile"],
    )
    known = holdings.loc[holdings.payoff_profile.eq("Long")].assign(cusip="x")
    holdings = pd.concat([holdings, known], ignore_index=True)
    if unknown:
        holdings = pd.concat([holdings, holdings.iloc[[0]].assign(value=np.nan)])
    holdings["balance_unit"], holdings["asset_category"] = "NS", "EC"
    states = nport_flow_states(funds, holdings).set_index(["timestamp", "asset_id"])
    name = "us_crowded_sector_flow_pressure"
    expected = 0.1 * (20**2 + 40**2) / 60**2
    if unknown:
        assert pd.isna(states.loc[("2024-05-01", "s"), name])
    else:
        assert states.loc[("2024-05-01", "s"), name] == pytest.approx(expected)
    assert states.loc[("2024-05-01", "x"), name] == pytest.approx(expected)
    np.testing.assert_allclose(states.loc["2024-08-01", name], 0.05)


def test_share_history_keeps_missing_declared_sessions():
    dates = pd.bdate_range("2024-01-02", periods=7)
    shares = pd.DataFrame(
        dict(
            asset_id="a",
            timestamp=dates.delete(3),
            available_at=dates.delete(3),
            shares_outstanding=[100.0, 110.0, 120.0, 140.0, 150.0, 160.0],
        )
    )
    result = share_states(shares, session_days=dates)
    assert pd.isna(result.etf_share_change_1d.iloc[4])
    assert result.etf_share_change_5d.iloc[5] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "zero_scope", ("global", "mainland", "all_equity", "restricted")
)
@pytest.mark.parametrize(
    "full_status",
    (
        "full_stock_table_reconciled",
        "full_stock_table_as_reported_unreconciled",
        "complete_all_equity_table_mainland_stock_domain",
    ),
)
def test_cn_complete_zero_report_clears_positions_and_older_amendment_stays_older(
    zero_scope,
    full_status,
):
    headers = pd.DataFrame(
        [
            (
                "h0",
                "2023-06-30",
                "2023-08-01",
                full_status,
                "false",
                3,
            ),
            (
                "h1",
                "2023-12-31",
                "2024-03-01",
                "explicit_no_stock_statement",
                "true",
                0,
            ),
            (
                "h2",
                "2023-06-30",
                "2024-04-01",
                full_status,
                "false",
                1,
            ),
        ],
        columns=[
            "report_version_identity",
            "timestamp",
            "available_at",
            "stock_disclosure_status",
            "explicit_no_stock",
            "stock_rows",
        ],
    ).assign(
        economic_portfolio_id="portfolio",
        fund_code="f",
        disclosure_scope="full_stock_holdings",
    )
    for field in ("timestamp", "available_at"):
        headers[field] = pd.to_datetime(headers[field])
    holdings = pd.DataFrame(
        [
            ("h0", "CN_MAINLAND", 10.0),
            ("h0", "CN_MAINLAND", 5.0),
            ("h0", "HK", 999.0),
            ("h2", "CN_MAINLAND", 99.0),
        ],
        columns=["report_version_identity", "stock_market", "shares_held"],
    ).assign(stock_code="s", economic_portfolio_id="portfolio")
    if full_status == "complete_all_equity_table_mainland_stock_domain":
        headers["disclosure_scope"] = "mainland_stock_holdings"
        headers.loc[headers.report_version_identity.eq("h0"), "stock_rows"] = 2
    if zero_scope != "global":
        zero = headers.report_version_identity.eq("h1")
        headers.loc[zero, "disclosure_scope"] = (
            "mainland_stock_holdings"
            if zero_scope in {"mainland", "all_equity"}
            else "restricted_stock_holdings"
        )
        headers.loc[zero, "stock_disclosure_status"] = (
            (
                "complete_all_equity_table_no_mainland_security_reported"
                if zero_scope == "all_equity"
                else "explicit_no_mainland_stock_reported"
            )
            if zero_scope in {"mainland", "all_equity"}
            else "no_restricted_stock_reported"
        )
        headers.loc[zero, "explicit_no_stock"] = pd.NA
        # A mainland-only zero leaves the fund's foreign positions intact.
        holdings.loc[len(holdings)] = ["h1", "HK", 777.0, "s", "portfolio"]
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="s",
            a_float_shares=100.0,
            timestamp=pd.to_datetime(
                ["2023-07-31", "2023-08-02", "2024-03-02", "2024-04-02"]
            ),
        )
    )
    fields = [
        "cn_fund_ownership_float_share",
        "cn_fund_holder_concentration",
        "cn_disclosed_ownership_change",
    ]
    result = calculate(
        decisions,
        dict(public_fund_holdings=holdings, public_fund_holdings_reports=headers),
        columns=fields,
    )
    assert result.loc[0, fields].isna().all()
    assert result.loc[1, fields[0]] == pytest.approx(0.15)
    assert result.loc[1, fields[1]] == 1
    if zero_scope == "restricted":
        np.testing.assert_allclose(result.loc[2:, fields[0]], [0.15, 0.99])
    else:
        assert result.loc[2, fields[0]] == 0
        assert result.loc[2, fields[2]] == pytest.approx(-0.15)
        assert result.loc[3, fields[0]] == 0
        assert pd.isna(result.loc[3, fields[1]])


@pytest.mark.parametrize("zero", [False, True])
def test_cn_redemption_uses_only_holdings_visible_before_flow_period(zero):
    headers = pd.DataFrame(
        dict(
            economic_portfolio_id="f",
            fund_code="001001",
            report_version_identity=["old", "late"],
            timestamp=pd.to_datetime(["2023-06-30", "2023-12-31"]),
            available_at=pd.to_datetime(["2023-08-20", "2024-03-20"]),
            disclosure_scope="full_stock_holdings",
            stock_disclosure_status=[
                "explicit_no_stock_reported" if zero else "full_stock_table_reconciled",
                "full_stock_table_reconciled",
            ],
            explicit_no_stock=["true" if zero else "false", "false"],
            stock_rows=[0 if zero else 1, 1],
        )
    )
    h = pd.DataFrame(
        dict(
            economic_portfolio_id="f",
            report_version_identity=["old", "late"],
            stock_market="CN_MAINLAND",
            stock_code="s",
            shares_held=[10.0, 999.0],
        )
    )
    if zero:
        h = h.loc[h.report_version_identity.ne("old")]
    d = pd.DataFrame(
        dict(
            market_id=["a_share"],
            asset_id=["s"],
            timestamp=pd.Timestamp("2024-08-21"),
            avg_volume_20d=100.0,
        )
    )
    result = calculate(
        d,
        dict(
            public_fund_holdings=h,
            public_fund_holdings_reports=headers,
            public_fund_financial_cells=cells(),
        ),
        columns=["cn_redemption_stock_pressure"],
    )
    assert result.cn_redemption_stock_pressure.iloc[0] == pytest.approx(
        0.0 if zero else 0.01
    )


def test_nullable_common_ownership_keeps_unknown_denominators_local():
    headers = pd.DataFrame(
        dict(
            economic_portfolio_id=["f1", "f2"],
            report_version_identity=["r1", "r2"],
            timestamp=pd.Timestamp("2024-06-30"),
            available_at=pd.Timestamp("2024-08-20"),
            disclosure_scope="full_stock_holdings",
            stock_disclosure_status="full_stock_table_reconciled",
            explicit_no_stock="false",
            stock_rows=2,
        )
    )
    h = pd.DataFrame(
        dict(
            economic_portfolio_id=["f1", "f1", "f2", "f2"],
            report_version_identity=["r1", "r1", "r2", "r2"],
            stock_market="CN_MAINLAND",
            stock_code=["a", "b", "d", "e"],
            shares_held=pd.Series([10.0, 20.0, 10.0, 20.0], dtype="Float64"),
        )
    )
    d = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["a", "b", "c", "d", "e"],
            timestamp=pd.Timestamp("2024-08-21"),
            a_float_shares=pd.Series(
                [100.0, pd.NA, 100.0, 100.0, 200.0], dtype="Float64"
            ),
        )
    )
    name = "cn_common_fund_ownership_exposure"
    result = calculate(
        d,
        dict(public_fund_holdings=h, public_fund_holdings_reports=headers),
        columns=[name],
    )
    assert result[name].iloc[:2].isna().all()
    np.testing.assert_allclose(result[name].iloc[2:].to_numpy(float), [0.0, 0.01, 0.01])


def test_passive_pressure_uses_each_historical_basket_and_ignores_unused_row_id():
    dates = pd.bdate_range("2024-01-02", periods=7)
    shares = pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=dates,
            available_at=dates + pd.Timedelta(hours=16),
            shares_outstanding=100 + 10 * np.arange(7),
        )
    )
    baskets = pd.DataFrame(
        dict(
            asset_id="f:s",
            fund_code="f",
            timestamp=dates,
            available_at=dates + pd.Timedelta(hours=16),
            component_code="s",
            component_kind="equity_security",
            component_quantity_unit="share",
            component_quantity=10.0,
            creation_unit_fund_shares=100.0,
        )
    )
    decision = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["s"],
            timestamp=dates[-1] + pd.Timedelta(days=1),
            avg_volume_20d=100.0,
            industry_standard_id=pd.NA,
            industry_level1_code=pd.NA,
        )
    )
    fields = ["stock_passive_flow_pressure_1d", "stock_passive_flow_pressure_5d"]
    result = calculate(
        decision,
        dict(
            public_fund_shares=shares,
            public_etf_pcf_components=baskets,
            trading_calendar=_calendar(dates),
        ),
        columns=fields,
    )
    np.testing.assert_allclose(result.loc[0, fields].to_numpy(float), [0.01, 0.05])
    industry = [name.replace("stock_", "industry_") for name in fields]
    unknown = calculate(
        decision.astype({"industry_standard_id": "str", "industry_level1_code": "str"}),
        dict(
            public_fund_shares=shares.astype({"shares_outstanding": "Float64"}),
            trading_calendar=_calendar(dates),
            public_etf_pcf_components=baskets.astype(
                {
                    "component_quantity": "Float64",
                    "creation_unit_fund_shares": "Float64",
                }
            ),
        ),
        columns=fields + industry,
    )
    np.testing.assert_allclose(unknown.loc[0, fields].to_numpy(float), [0.01, 0.05])
    assert unknown[industry].isna().all().all()


@pytest.mark.parametrize(
    "case", ["complete", "partial", "unpublished", "unknown_quantity", "different_file"]
)
def test_passive_pressure_distinguishes_absent_from_unknown_component(case):
    dates = pd.bdate_range("2024-01-02", periods=7)
    shares = pd.DataFrame(
        dict(
            asset_id="f",
            timestamp=dates,
            available_at=dates + pd.Timedelta(hours=16),
            shares_outstanding=100 + 10 * np.arange(7),
        )
    )
    parts = pd.DataFrame(
        dict(
            fund_code="f",
            timestamp=dates,
            available_at=dates + pd.Timedelta(hours=16),
            component_code="s",
            component_kind="equity_security",
            component_quantity_unit="share",
            component_quantity=np.nan if case == "unknown_quantity" else 10.0,
            creation_unit_fund_shares=100.0,
            source="etf_pcf_components_extracted",
            source_sha256="raw-pcf",
        )
    )
    headers = pd.DataFrame(
        dict(
            fund_code="f",
            timestamp=dates,
            available_at=dates
            + pd.Timedelta(days=20 if case == "unpublished" else 0, hours=16),
            declared_component_count=2 if case == "partial" else 1,
            source="etf_pcf_baskets_extracted",
            source_sha256="another-pcf" if case == "different_file" else "raw-pcf",
        )
    )
    decision = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["s", "outside_basket"],
            timestamp=dates[-1] + pd.Timedelta(days=1),
            avg_volume_20d=100.0,
            industry_standard_id="standard",
            industry_level1_code="industry",
        )
    )
    columns = [
        f"{scope}_passive_flow_pressure_{width}d"
        for scope in ("stock", "industry")
        for width in (1, 5)
    ]
    result = calculate(
        decision,
        dict(
            public_fund_shares=shares,
            public_etf_pcf_components=parts,
            public_etf_pcf_baskets=headers,
            trading_calendar=_calendar(dates),
        ),
        columns=columns,
    ).set_index("asset_id")
    if case == "unknown_quantity":
        assert result.loc["s", columns[:2]].isna().all()
    else:
        # Ten new fund shares / 100 per creation unit * ten stock shares / volume 100.
        np.testing.assert_allclose(
            result.loc["s", columns[:2]].to_numpy(float), [0.01, 0.05]
        )
    if case in {"complete", "unknown_quantity"}:
        np.testing.assert_array_equal(
            result.loc["outside_basket", columns[:2]].to_numpy(float), [0, 0]
        )
    else:
        assert result.loc["outside_basket", columns[:2]].isna().all()
    if case == "complete":
        np.testing.assert_allclose(
            result[columns[2:]].to_numpy(float), [[0.01, 0.05], [0.01, 0.05]]
        )
    else:
        assert result[columns[2:]].isna().all().all()


def test_passive_pressure_aggregates_each_funds_visible_historical_window():
    days = pd.bdate_range("2024-01-02", periods=7)
    release = days[-1] + pd.Timedelta(days=1)
    shares, parts = [], []
    for fund, change, scale in (("f1", 10, 1), ("f2", 20, 2)):
        units = pd.DataFrame(
            dict(
                asset_id=fund,
                timestamp=days,
                available_at=days + pd.Timedelta(hours=8),
                shares_outstanding=100 + change * np.arange(len(days)),
            )
        )
        if fund == "f2":
            units.loc[units.index[-1], "available_at"] = release
        shares.append(units)
        parts.append(
            pd.DataFrame(
                dict(
                    fund_code=fund,
                    timestamp=days,
                    available_at=days + pd.Timedelta(hours=8),
                    component_code="s",
                    component_kind="equity_security",
                    component_quantity_unit="share",
                    component_quantity=10.0 * scale * np.arange(1, len(days) + 1),
                    creation_unit_fund_shares=100.0,
                )
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
    names = [f"stock_passive_flow_pressure_{width}d" for width in (1, 5)]
    result = calculate(
        decisions,
        dict(
            public_fund_shares=pd.concat(shares, ignore_index=True),
            public_etf_pcf_components=pd.concat(parts, ignore_index=True),
            trading_calendar=_calendar(days),
        ),
        columns=names,
    )
    # f1 uses days 3..7 throughout; f2 uses 2..6 until its seventh report is public.
    np.testing.assert_allclose(
        result[names].to_numpy(float),
        [[0.07 + 0.24, 0.25 + 0.8], [0.07 + 0.28, 0.25 + 1.0]],
    )


@pytest.mark.parametrize("unknown_period", ["old", "new"])
def test_native_ownership_keeps_unknown_quantities_distinct_from_absence(
    tmp_path, unknown_period
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    headers = pd.DataFrame(
        [
            (fund, period + fund, pd.Timestamp(day), pd.Timestamp(released))
            for period, day, released in (
                ("old", "2023-12-31", "2024-03-20"),
                ("new", "2024-06-30", "2024-08-20"),
            )
            for fund in ("f1", "f2")
        ],
        columns=[
            "economic_portfolio_id",
            "report_version_identity",
            "timestamp",
            "available_at",
        ],
    ).assign(
        disclosure_scope="full_stock_holdings",
        stock_disclosure_status="full_stock_table_as_reported_unreconciled",
        explicit_no_stock="false",
        stock_rows=6,
    )
    holdings = pd.DataFrame(
        [
            (period + fund, fund, stock, quantity)
            for period in ("old", "new")
            for fund, rows in (
                (
                    "f1",
                    [
                        ("a", 10.0),
                        ("a", 20.0),
                        ("b", 30.0),
                        ("c", 40.0),
                        ("e", 15.0),
                        ("f", 25.0),
                    ],
                ),
                ("f2", [("a", 5.0), ("b", 5.0), ("c", 5.0)]),
            )
            for stock, quantity in rows
            if not (period == "new" and stock == "e")
        ],
        columns=[
            "report_version_identity",
            "economic_portfolio_id",
            "stock_code",
            "shares_held",
        ],
    ).assign(stock_market="CN_MAINLAND")
    unknown = holdings.report_version_identity.eq(unknown_period + "f1") & (
        holdings.stock_code.isin(("b", "f"))
        | (holdings.stock_code.eq("a") & holdings.shares_held.eq(20))
    )
    holdings.loc[unknown, "shares_held"] = np.nan
    headers["stock_rows"] = headers.report_version_identity.map(
        holdings.groupby("report_version_identity").size()
    )
    holdings["asset_id"] = np.arange(len(holdings)).astype(str)
    holdings["timestamp"] = holdings.report_version_identity.map(
        headers.set_index("report_version_identity").timestamp
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=list("abcdef"),
            timestamp=pd.Timestamp("2024-08-21"),
            signal_session=pd.Timestamp("2024-08-20"),
            execution_session=pd.Timestamp("2024-08-21"),
            return_available_at=pd.NaT,
            a_float_shares=100.0,
        )
    )
    inputs = {}
    for name, frame, keys in (
        ("decision_panel", decisions, ("asset_id", "timestamp")),
        ("public_fund_holdings", holdings, ("asset_id", "timestamp")),
        (
            "public_fund_holdings_reports",
            headers,
            ("report_version_identity", "timestamp"),
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
                primary_key=keys,
                partition_columns=(),
                column_roles={
                    c: "timestamp"
                    if c == "timestamp"
                    else "key"
                    if c in keys
                    else "context"
                    for c in frame
                },
            ),
        )
        inputs[name] = str(path)
    fields = [
        "cn_fund_ownership_float_share",
        "cn_fund_holder_concentration",
        "cn_disclosed_ownership_change",
    ]
    path = prepare(
        dict(
            market_id="a_share",
            data_id="fund_flow_factors",
            inputs=inputs,
            factor_columns=fields,
            time_window=dict(start="2024-08-21", end="2024-08-21"),
            output_root=str(tmp_path / "factors"),
        )
    )
    result = read_dataset(path).set_index("asset_id")
    assert result.loc[["a", "b", "f"], fields[2]].isna().all()
    if unknown_period == "new":
        assert result.loc[["a", "b", "f"], fields[:2]].isna().all(axis=None)
    else:
        np.testing.assert_allclose(
            result.loc[["a", "b", "f"], fields[0]], [0.35, 0.35, 0.25]
        )
    np.testing.assert_allclose(
        result.loc["c", fields].to_numpy(float), [0.45, (40**2 + 5**2) / 45**2, 0]
    )
    np.testing.assert_allclose(result.loc[["d", "e"], fields[0]], [0, 0])
    np.testing.assert_allclose(result.loc[["d", "e"], fields[2]], [0, -0.15])


def test_ownership_snapshot_reuse_keeps_revisions_unknowns_and_daily_context():
    dates = pd.date_range("2024-01-01", periods=9)
    headers = pd.DataFrame(
        dict(
            economic_portfolio_id=["f", "g", "f", "f", "f", "f"],
            report_version_identity=["f1", "g1", "f2", "f3", "f4", "f5"],
            timestamp=pd.to_datetime(
                ["2023-12-31"] * 3 + ["2024-01-05", "2023-12-31", "2024-01-07"]
            ),
            available_at=dates[[1, 1, 3, 5, 6, 7]],
            disclosure_scope="full_stock_holdings",
            stock_disclosure_status="full_stock_table_reconciled",
            explicit_no_stock=["false"] * 3 + ["true", "false", "false"],
            stock_rows=[2, 1, 2, 0, 2, 2],
        )
    )
    holdings = pd.DataFrame(
        dict(
            economic_portfolio_id=["f"] * 8 + ["g"],
            report_version_identity=np.repeat(["f1", "f2", "f4", "f5"], 2).tolist()
            + ["g1"],
            stock_market="CN_MAINLAND",
            stock_code=["a", "b"] * 4 + ["a"],
            shares_held=pd.Series(
                [10, 20, 30, pd.NA, 90, 90, 10, 40, 10], dtype="Float64"
            ),
        )
    )
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["a", "b"] * len(dates),
            timestamp=dates.repeat(2),
            a_float_shares=np.repeat(np.arange(1, 10) * 100.0, 2),
            peak_drawdown_20d=np.repeat(np.arange(1, 10) / 100.0, 2),
            amihud_20d=np.repeat(np.arange(1, 10), 2),
            amihud_60d=1.0,
        )
    )
    names = [
        "cn_fund_holder_concentration",
        "cn_fund_ownership_float_share",
        "ownership_crowding_x_drawdown",
        "ownership_crowding_x_liquidity_stress",
    ]
    result = calculate(
        decisions,
        dict(public_fund_holdings=holdings, public_fund_holdings_reports=headers),
        columns=names,
    )
    # The late f4 amendment cannot replace f3's newer zero-holdings period.
    hhi = np.array(
        [
            [np.nan, np.nan],
            [0.5, 1],
            [0.5, 1],
            [0.625, np.nan],
            [0.625, np.nan],
            [1, np.nan],
            [1, np.nan],
            [0.5, 1],
            [0.5, 1],
        ]
    ).ravel()
    quantity = np.array(
        [
            [np.nan, np.nan],
            [20, 20],
            [20, 20],
            [40, np.nan],
            [40, np.nan],
            [10, 0],
            [10, 0],
            [20, 40],
            [20, 40],
        ]
    ).ravel()
    for name, expected in zip(
        names,
        [
            hhi,
            quantity / decisions.a_float_shares,
            hhi * decisions.peak_drawdown_20d,
            hhi * (decisions.amihud_20d - 1),
        ],
        strict=True,
    ):
        np.testing.assert_allclose(
            result[name].to_numpy(float), expected, equal_nan=True
        )


@pytest.mark.parametrize(
    "case",
    ["complete", "missing_basket", "partial_basket", "unknown_delta", "zero_delta"],
)
def test_native_passive_pressure_preserves_each_funds_unknown_contribution(
    tmp_path, case
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    days = pd.bdate_range("2024-01-02", periods=7)
    shares = pd.concat(
        [
            pd.DataFrame(
                dict(
                    asset_id=fund,
                    timestamp=days,
                    available_at=days + pd.Timedelta(hours=16),
                    shares_outstanding=100.0 + 10 * np.arange(7),
                )
            )
            for fund in ("f1", "f2")
        ],
        ignore_index=True,
    )
    parts = pd.concat(
        [
            pd.DataFrame(
                dict(
                    fund_code=fund,
                    timestamp=days,
                    available_at=days + pd.Timedelta(hours=16),
                    component_code="s",
                    component_kind="equity_security",
                    component_quantity_unit="share",
                    component_quantity=10.0,
                    creation_unit_fund_shares=100.0,
                )
            )
            for fund in ("f1", "f2")
        ],
        ignore_index=True,
    )
    headers = parts[["fund_code", "timestamp", "available_at"]].assign(
        declared_component_count=1
    )
    last = parts.fund_code.eq("f2") & parts.timestamp.eq(days[-1])
    if case in {"missing_basket", "zero_delta"}:
        parts = parts.loc[~last]
    elif case == "partial_basket":
        parts.loc[last, "component_code"] = "other"
        headers.loc[last, "declared_component_count"] = 2
    if case in {"unknown_delta", "zero_delta"}:
        shares.loc[
            shares.asset_id.eq("f2") & shares.timestamp.eq(days[-1]),
            "shares_outstanding",
        ] = np.nan if case == "unknown_delta" else 150.0
    decision_time = days[-1] + pd.Timedelta(days=1)
    decisions = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=["s", "other", "outside"],
            timestamp=decision_time,
            signal_session=days[-1],
            execution_session=decision_time,
            return_available_at=pd.NaT,
            avg_volume_20d=100.0,
            industry_standard_id="standard",
            industry_level1_code=["s_industry", "other_industry", "outside_industry"],
        )
    )
    calendar = _calendar(days)
    calendar["timestamp"] = calendar.calendar_date
    inputs = {}
    for name, frame, keys in (
        ("decision_panel", decisions, ("asset_id", "timestamp")),
        ("public_fund_shares", shares, ("asset_id", "timestamp")),
        (
            "public_etf_pcf_components",
            parts,
            ("fund_code", "timestamp", "component_code"),
        ),
        ("public_etf_pcf_baskets", headers, ("fund_code", "timestamp")),
        ("trading_calendar", calendar, ("source_market_type", "timestamp")),
    ):
        path = tmp_path / name
        write_dataset(
            path,
            frame,
            spec=DataSpec(
                data_id=name,
                frequency="1d",
                grain="research_row",
                primary_key=keys,
                partition_columns=(),
                column_roles={
                    c: "timestamp"
                    if c == "timestamp"
                    else "key"
                    if c in keys
                    else "context"
                    for c in frame
                },
            ),
        )
        inputs[name] = str(path)
    fields = [
        f"{scope}_passive_flow_pressure_{width}d"
        for scope in ("stock", "industry")
        for width in (1, 5)
    ]
    output = prepare(
        dict(
            market_id="a_share",
            data_id="fund_flow_factors",
            inputs=inputs,
            factor_columns=fields,
            time_window=dict(
                start=str(decision_time.date()), end=str(decision_time.date())
            ),
            output_root=str(tmp_path / "pressure"),
        )
    )
    result = read_dataset(output).set_index("asset_id")
    if case in {"missing_basket", "partial_basket", "unknown_delta"}:
        assert result.loc["s", fields].isna().all()
    else:
        expected = [0.01, 0.09] if case == "zero_delta" else [0.02, 0.10]
        np.testing.assert_allclose(
            result.loc["s", fields].to_numpy(float), expected * 2
        )
    if case in {"missing_basket", "partial_basket"}:
        assert result.loc["outside", fields].isna().all()
    else:
        np.testing.assert_allclose(result.loc["outside", fields].to_numpy(float), 0)
    if case == "partial_basket":
        # A disclosed component is known even when another component is omitted.
        np.testing.assert_allclose(result.loc["other", fields].to_numpy(float), 0.01)


@pytest.mark.parametrize(
    "missing",
    [
        "flow",
        "prior_nav",
        "prior_nav_conflict",
        "ambiguous",
        "ambiguous_reversed",
        "ambiguous_same_flow",
    ],
)
def test_native_nport_unknown_flow_dependencies_stay_local_and_recover(
    tmp_path, missing
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    funds = pd.DataFrame(
        [
            (f"{series}{i}", series, day, released, nav)
            for series in ("a", "b")
            for i, (day, released, nav) in enumerate(
                (
                    ("2023-12-31", "2024-02-01", 100.0),
                    ("2024-03-31", "2024-05-01", 200.0),
                    ("2024-06-30", "2024-08-01", 300.0),
                )
            )
        ],
        columns=[
            "accession",
            "series_id",
            "timestamp",
            "available_at",
            "net_assets_usd",
        ],
    ).assign(filing_type="NPORT-P")
    for column in ("timestamp", "available_at"):
        funds[column] = pd.to_datetime(funds[column])
    for kind in ("sales", "reinvestment", "redemption"):
        for month in (1, 2, 3):
            funds[f"{kind}_month_{month}_usd"] = 0.0
    funds["sales_month_1_usd"] = 10.0
    accession, field = (
        ("a1", "sales_month_1_usd") if missing == "flow" else ("a0", "net_assets_usd")
    )
    if missing in {"flow", "prior_nav"}:
        funds.loc[funds.accession.eq(accession), field] = np.nan
    elif missing == "prior_nav_conflict":
        funds = pd.concat(
            [funds, funds.loc[funds.accession.eq("a0")].assign(net_assets_usd=200.0)],
            ignore_index=True,
        )
    else:
        funds.loc[funds.accession.eq("a1"), "filing_type"] = "NPORT-P/A"
        amendment = funds.loc[funds.accession.eq("a1")]
        funds = pd.concat(
            [
                funds,
                amendment.assign(
                    accession="conflict",
                    sales_month_1_usd=10.0
                    if missing == "ambiguous_same_flow"
                    else 30.0,
                ),
                amendment.assign(
                    accession="resolved", available_at=pd.Timestamp("2024-05-03")
                ),
            ],
            ignore_index=True,
        )
        if missing == "ambiguous_reversed":
            funds = funds.iloc[::-1].reset_index(drop=True)
    holdings = pd.DataFrame(
        [("a0", "a", "s", 20.0), ("a1", "a", "s", 30.0)]
        + [
            (f"b{i}", "b", stock, value)
            for i in (0, 1)
            for stock, value in (("s", 40.0), ("x", 50.0))
        ],
        columns=["accession", "series_id", "cusip", "value"],
    ).assign(balance_unit="NS", asset_category="EC", payoff_profile="Long")
    if missing.startswith("ambiguous"):
        holdings = pd.concat(
            [
                holdings,
                holdings.loc[holdings.accession.eq("a1")].assign(accession="resolved"),
            ],
            ignore_index=True,
        )
    holdings["timestamp"] = holdings.accession.map(
        funds.drop_duplicates("accession").set_index("accession").timestamp
    )
    funds["source_sha256"] = np.arange(len(funds)).astype(str)
    decisions = pd.DataFrame(
        [
            (asset, pd.Timestamp(date))
            for date in ("2024-05-02", "2024-08-02")
            for asset in ("SPY", "XLK")
        ],
        columns=["asset_id", "timestamp"],
    ).assign(market_id="a_share", return_available_at=pd.NaT)
    decisions["signal_session"] = decisions.timestamp - pd.Timedelta(days=1)
    decisions["execution_session"] = decisions.timestamp
    inputs = {}
    for name, frame, keys in (
        ("decision_panel", decisions, ("asset_id", "timestamp")),
        ("public_nport_funds", funds, ("accession", "timestamp", "source_sha256")),
        ("public_nport_holdings", holdings, ("accession", "cusip", "timestamp")),
    ):
        path = tmp_path / name
        write_dataset(
            path,
            frame,
            spec=DataSpec(
                data_id=name,
                frequency="1d",
                grain="research_row",
                primary_key=keys,
                partition_columns=(),
                column_roles={
                    c: "timestamp"
                    if c == "timestamp"
                    else "key"
                    if c in keys
                    else "context"
                    for c in frame
                },
            ),
        )
        inputs[name] = str(path)
    name = "us_crowded_sector_flow_pressure"
    output = prepare(
        dict(
            market_id="a_share",
            data_id="fund_flow_factors",
            inputs=inputs,
            factor_columns=[name],
            parameters={"us_assets": {"SPY": "s", "XLK": "x"}},
            time_window=dict(start="2024-05-02", end="2024-08-02"),
            output_root=str(tmp_path / "factors"),
        )
    )
    result = read_dataset(output).set_index(["timestamp", "asset_id"])[name]
    if missing == "ambiguous_same_flow":
        assert result.loc[("2024-05-02", "SPY")] == pytest.approx(
            0.1 * (20**2 + 40**2) / 60**2
        )
    else:
        assert pd.isna(result.loc[("2024-05-02", "SPY")])
    assert result.loc[("2024-05-02", "XLK")] == pytest.approx(0.1)
    assert result.loc[("2024-08-02", "SPY")] == pytest.approx(
        0.05 * (30**2 + 40**2) / 70**2
    )
    assert result.loc[("2024-08-02", "XLK")] == pytest.approx(0.05)
