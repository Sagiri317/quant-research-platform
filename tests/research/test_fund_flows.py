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
    states = share_states(shares)
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


def test_us_complete_reports_supplements_restatements_and_empty_positions():
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
    h["share_type"] = "SH"
    h["put_call"] = None
    states = us_ownership_states(h, filings, assets={"SPY": "s", "XLF": "x"})
    spy = states.loc[states.asset_id.eq("SPY")].set_index("timestamp")
    field = "us_institution_ownership_concentration"
    assert spy.loc["2024-02-01", field] == pytest.approx(0.5)
    assert spy.loc["2024-02-02", field] == pytest.approx((15**2 + 10**2) / 25**2)
    assert spy.loc["2024-02-03", field] == pytest.approx((2**2 + 10**2) / 12**2)
    assert spy.loc["2024-05-01", field] == 1
    assert spy.loc["2024-05-02", field] == 1


def test_nport_flows_use_prior_net_assets_and_empty_reports_replace_positions():
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
    holdings["balance_unit"], holdings["asset_category"] = "NS", "EC"
    states = nport_flow_states(funds, holdings).set_index("timestamp")
    name = "us_crowded_sector_flow_pressure"
    assert states.loc["2024-05-01", name] == pytest.approx(
        0.1 * (20**2 + 40**2) / 60**2
    )
    assert states.loc["2024-08-01", name] == pytest.approx(0.05)


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


def test_cn_complete_zero_report_clears_positions_and_older_amendment_stays_older():
    headers = pd.DataFrame(
        [
            (
                "h0",
                "2023-06-30",
                "2023-08-01",
                "full_stock_table_reconciled",
                "false",
                2,
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
                "full_stock_table_reconciled",
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
        [("h0", "CN_MAINLAND", 10.0), ("h0", "HK", 999.0), ("h2", "CN_MAINLAND", 99.0)],
        columns=["report_version_identity", "stock_market", "shares_held"],
    ).assign(stock_code="s", economic_portfolio_id="portfolio")
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
    assert result.loc[1, fields[0]] == pytest.approx(0.1)
    assert result.loc[1, fields[1]] == 1
    assert result.loc[2, fields[0]] == 0
    assert result.loc[2, fields[2]] == pytest.approx(-0.1)
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
        dict(public_fund_shares=shares, public_etf_pcf_components=baskets),
        columns=fields,
    )
    np.testing.assert_allclose(result.loc[0, fields].to_numpy(float), [0.01, 0.05])
    industry = [name.replace("stock_", "industry_") for name in fields]
    unknown = calculate(
        decision.astype({"industry_standard_id": "str", "industry_level1_code": "str"}),
        dict(
            public_fund_shares=shares.astype({"shares_outstanding": "Float64"}),
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
