"""Source-specific schemas; NAV, market prices and trading activity stay distinct."""

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_real_numeric_series

DATASETS = {
    "public_index_weights": {
        "source_asset_id": "identifier",
        "constituent_asset_id": "identifier",
        "weight": "ratio",
    },
    "public_option_risk": {
        "underlying_id": "identifier",
        "contract_id": "identifier",
        "option_type": "identifier",
        "implied_volatility": "annualized_decimal_volatility",
        "delta": "dimensionless",
        "gamma": "per_CNY",
        "vega": "reported_units",
    },
    "public_option_contracts": {
        "underlying_id": "identifier",
        "security_id": "identifier",
        "contract_id": "identifier",
        "option_type": "identifier",
        "expiry_date": "date",
        "start_date": "date",
        "exercise_price": "CNY_per_fund_unit",
        "contract_unit": "fund_units",
        "original_contract_id": "identifier",
        "original_exercise_price": "CNY_per_fund_unit",
        "original_contract_unit": "fund_units",
    },
    "public_domestic_etf_prices": {
        "open": "CNY_per_fund_unit",
        "close": "CNY_per_fund_unit",
        "high": "CNY_per_fund_unit",
        "low": "CNY_per_fund_unit",
        "volume": "fund_units",
        "amount": "CNY",
        "return_percent": "percent",
        "turnover_rate_percent": "percent",
        "price_basis": "text",
    },
    "public_13f_filings": {
        "accession": "identifier",
        "manager_cik": "identifier",
        "report_date": "date",
        "filing_type": "identifier",
        "is_amendment": "identifier",
        "amendment_number": "count",
        "amendment_type": "text",
        "report_type": "text",
        "reported_table_entry_total": "count",
        "source_infotable_row_count": "count",
        "target_holding_row_count": "count",
    },
    "public_fund_financial_cells": {
        "fund_code": "identifier",
        "economic_portfolio_id": "identifier",
        "share_class_header": "text",
        "report_period_end": "date",
        "value_observation_date": "date",
        "table_kind": "identifier",
        "field": "identifier",
        "unit": "identifier",
        "denominator": "identifier",
        "value": "reported_units",
        "report_version_identity": "identifier",
    },
    "public_futures_contracts": {
        "product": "identifier",
        "open_date": "date",
        "expiry_date": "date",
    },
    "public_futures_member_positions": {
        "product": "identifier",
        "contract_id": "identifier",
        "member_id": "identifier",
        "position_side": "identifier",
        "rank": "rank",
        "quantity": "contracts",
        "quantity_change": "contracts",
    },
    "public_refinancing_balances": {
        "beginning_quantity": "shares",
        "lent_quantity": "shares",
        "ending_quantity": "shares",
        "ending_balance": "CNY",
        "scope": "identifier",
    },
    "public_finra_margin": {
        "margin_debit": "million_USD",
        "cash_free_credit": "million_USD",
        "margin_free_credit": "million_USD",
    },
    "public_option_put_call": {
        "put_call_ratio": "ratio",
        "ratio_name": "text",
    },
    "public_fund_holdings": {
        "fund_code": "identifier",
        "economic_portfolio_id": "identifier",
        "report_version_identity": "identifier",
        "stock_code": "identifier",
        "stock_market": "identifier",
        "shares_held": "shares",
        "fair_value_cny": "CNY",
        "nav_weight_percent": "percent",
        "disclosure_scope": "text",
        "report_path": "text",
    },
    "public_fund_holdings_reports": {
        "fund_code": "identifier",
        "economic_portfolio_id": "identifier",
        "report_version_identity": "identifier",
        "disclosure_scope": "text",
        "stock_disclosure_status": "text",
        "explicit_no_stock": "identifier",
        "stock_rows": "count",
    },
    "public_13f_holdings": {
        "accession": "identifier",
        "manager_cik": "identifier",
        "cusip": "identifier",
        "report_date": "date",
        "filing_type": "identifier",
        "is_amendment": "identifier",
        "amendment_type": "text",
        "value_usd": "USD",
        "shares_or_principal": "reported_units",
        "share_type": "identifier",
        "put_call": "identifier",
        "investment_discretion": "identifier",
    },
    "public_nport_holdings": {
        "accession": "identifier",
        "series_id": "identifier",
        "cusip": "identifier",
        "report_date": "date",
        "filing_type": "identifier",
        "value": "USD",
        "currency": "identifier",
        "balance": "reported_units",
        "balance_unit": "identifier",
        "asset_category": "identifier",
        "payoff_profile": "identifier",
        "derivative_category": "identifier",
        "portfolio_weight_percent": "percent",
    },
    "public_margin_balances": {
        "financing_balance": "CNY",
        "financing_buy_amount": "CNY",
        "financing_repayment_amount": "CNY",
        "lending_balance": "CNY",
        "lending_quantity": "shares_or_fund_units",
        "lending_sell_quantity": "shares_or_fund_units",
        "lending_repayment_quantity": "shares_or_fund_units",
    },
    "public_fund_shares": {"shares_outstanding": "fund_units"},
    "public_fund_nav": {
        "fund_code": "identifier",
        "nav_cny": "CNY_per_fund_unit",
        "accumulated_nav_cny": "CNY_per_fund_unit",
        "reported_return_percent": "percent",
    },
    "public_nport_funds": {
        "accession": "identifier",
        "series_id": "identifier",
        "report_date": "date",
        "filing_type": "identifier",
        "total_assets_usd": "USD",
        "net_assets_usd": "USD",
        "liabilities_usd": "USD",
        "cash_outside_holdings_usd": "USD",
        **{
            f"{flow}_month_{month}_usd": "USD"
            for flow in ("sales", "reinvestment", "redemption")
            for month in (1, 2, 3)
        },
    },
    "public_security_identifiers": {
        "accession": "identifier",
        "holding_id": "identifier",
        "series_id": "identifier",
        "cusip": "identifier",
        "isin": "identifier",
        "ticker": "identifier",
        "report_date": "date",
        "filing_type": "identifier",
    },
    "public_etf_pcf_baskets": {
        "fund_code": "identifier",
        "declared_component_count": "count",
        "valuation_date": "date",
        "creation_unit_fund_shares": "fund_units",
        "nav_cny": "CNY_per_fund_unit",
        "nav_per_creation_unit_cny": "CNY",
        "cash_component_cny": "CNY",
        "estimated_cash_component_cny": "CNY",
    },
    "public_etf_pcf_components": {
        "fund_code": "identifier",
        "component_code": "identifier",
        "component_kind": "text",
        "component_quantity": "reported_units",
        "component_quantity_unit": "text",
        "creation_unit_fund_shares": "fund_units",
        "cash_substitution_flag_as_provided": "text",
    },
    "public_futures_daily": {
        "product": "identifier",
        "open": "index_points",
        "high": "index_points",
        "low": "index_points",
        "close": "index_points",
        "settlement": "index_points",
        "previous_settlement": "index_points",
        "volume": "contracts",
        "open_interest": "contracts",
        "open_interest_change": "contracts",
    },
    "public_cftc_positions": {
        "report_type": "identifier",
        "leveraged_long": "contracts",
        "leveraged_short": "contracts",
        "leveraged_spread": "contracts",
        "asset_manager_long": "contracts",
        "asset_manager_short": "contracts",
        "dealer_long": "contracts",
        "dealer_short": "contracts",
        "open_interest": "contracts",
    },
    "public_finra_short_volume": {
        "short_volume": "shares",
        "short_exempt_volume": "shares",
        "total_volume": "shares",
        "market": "identifier",
    },
    "public_etf_prices": {
        "open": "USD",
        "high": "USD",
        "low": "USD",
        "close": "USD",
        "adjusted_close": "USD",
        "volume": "shares",
        "adjustment_basis": "text",
    },
    "public_domestic_etf_actions": {
        "event_type": "identifier",
        "amount": "CNY_per_fund_unit",
        "numerator": "ratio",
        "denominator": "ratio",
        "split_ratio": "ratio",
    },
    "public_etf_actions": {
        "event_type": "identifier",
        "amount": "USD_per_share",
        "numerator": "ratio",
        "denominator": "ratio",
        "split_ratio": "text",
    },
    "public_etf_nav": {
        "nav": "USD_per_fund_unit",
        "shares_outstanding": "fund_units",
        "total_net_assets": "USD",
    },
}
SOURCES = {
    "public_index_weights": {"public_index_weight_history_extracted"},
    "public_option_risk": {"sse_option_risk_indicators"},
    "public_option_contracts": {"sse_option_contract_updates"},
    "public_domestic_etf_actions": {"eastmoney_cn_etf_corporate_actions_extracted"},
    "public_domestic_etf_prices": {"eastmoney_cn_etf_daily"},
    "public_13f_filings": {"sec_13f_submission_metadata_extracted"},
    "public_fund_financial_cells": {"csrc_official_fund_financial_cells_extracted"},
    "public_futures_contracts": {"cffex_contract_parameters"},
    "public_futures_member_positions": {"cffex_member_rankings"},
    "public_refinancing_balances": {
        "csf_securities_refinancing_summary",
        "csf_market_making_borrowing_summary",
    },
    "public_finra_margin": {"finra_margin"},
    "public_option_put_call": {"cboe_put_call_extracted"},
    "public_fund_holdings": {
        "chinaamc_extracted_holdings",
        "cn_fund_report_holdings_extracted",
        "cn_fund_report_full_holdings_csv_evidence",
    },
    "public_fund_holdings_reports": {"cn_fund_report_descriptors_extracted"},
    "public_13f_holdings": {"sec_13f", "sec_13f_target_holdings_extracted"},
    "public_nport_holdings": {"sec_nport", "sec_nport_target_holdings_extracted"},
    "public_margin_balances": {"sse_margin", "szse_margin"},
    "public_fund_shares": {"sse_etf_shares", "szse_etf_shares"},
    "public_fund_nav": {"eastmoney_fund_history_extracted"},
    "public_nport_funds": {"sec_nport"},
    "public_security_identifiers": {"sec_nport"},
    "public_etf_pcf_baskets": {"etf_pcf_baskets_extracted"},
    "public_etf_pcf_components": {"etf_pcf_components_extracted"},
    "public_futures_daily": {"cffex_daily"},
    "public_cftc_positions": {"cftc_tff"},
    "public_finra_short_volume": {"finra_short_volume"},
    "public_etf_prices": {"nasdaq_etf_ohlcv", "yahoo_chart_extracted"},
    "public_etf_actions": {"yahoo_corporate_actions_extracted"},
    "public_etf_nav": {"ssga_nav_shares"},
}


def accepts(data_id, record):
    if record["source"] not in SOURCES[data_id]:
        return False
    path = record["path"]
    if data_id == "public_index_weights":
        return path.endswith((".csv", ".csv.gz"))
    if data_id in {"public_option_risk", "public_option_contracts"}:
        return path.endswith(".json") and record.get("artifact_role", "").startswith(
            "official_"
        )
    if record["source"] == "sec_nport_target_holdings_extracted":
        return path.endswith(".csv")
    if data_id == "public_domestic_etf_prices":
        return path.endswith(".json")
    if record["source"] in {
        "sec_13f_submission_metadata_extracted",
        "sec_13f_target_holdings_extracted",
    }:
        return path.endswith(".csv")
    if data_id == "public_fund_financial_cells":
        return path.endswith((".csv", ".csv.gz"))
    if data_id in {"public_futures_contracts", "public_futures_member_positions"}:
        return path.endswith(".xml")
    if data_id == "public_refinancing_balances":
        return path.endswith(".json")
    if record["source"].startswith("etf_pcf_"):
        return path.endswith(".csv")
    if (
        record["source"].startswith("yahoo_")
        or data_id == "public_domestic_etf_actions"
    ):
        return path.endswith(".csv")
    if data_id in {
        "public_13f_holdings",
        "public_nport_holdings",
        "public_nport_funds",
        "public_security_identifiers",
    }:
        return path.endswith(".zip")
    if data_id == "public_fund_nav":
        return path.endswith(".json")
    if data_id == "public_option_put_call":
        return path.endswith(".csv")
    if data_id == "public_fund_holdings":
        return (
            path.endswith((".csv", ".csv.gz"))
            if record["source"] == "cn_fund_report_full_holdings_csv_evidence"
            else path.endswith(".json")
        )
    if data_id == "public_fund_holdings_reports":
        return path.endswith(".json")
    if data_id == "public_cftc_positions":
        return path.endswith(".zip")
    if data_id == "public_futures_daily":
        return path.endswith((".zip", ".csv"))
    if data_id == "public_finra_short_volume":
        return path.endswith(".txt")
    if data_id in {"public_fund_shares", "public_margin_balances"}:
        return path.endswith(
            ".json" if record["source"].startswith("sse_") else ".xlsx"
        )
    return path.endswith(".json" if data_id == "public_etf_prices" else ".xlsx")


def _numbers(frame, columns):
    counts = frame.attrs.setdefault("invalid_numeric_cells", {})
    for column in columns:
        if column in counts:
            continue
        values = (
            frame[column]
            .astype("str")
            .str.replace(",", "", regex=False)
            .replace({"": None, "--": None, "-": None, "N/A": None})
        )
        numeric = strict_real_numeric_series(
            values, error="public_numeric_invalid", errors="coerce"
        )
        finite = np.isfinite(numeric)
        counts[column] = counts.get(column, 0) + int((values.notna() & ~finite).sum())
        frame[column] = numeric.where(finite)
    return frame


def _rename(frame, mapping):
    missing = set(mapping) - set(frame)
    if missing:
        raise ValueError("public_source_columns_missing: " + ",".join(sorted(missing)))
    return frame.loc[:, list(mapping)].rename(columns=mapping).copy()


def normalize(data_id, path, record):
    source = record["source"]
    if source in {"sse_option_risk_indicators", "sse_option_contract_updates"}:
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["result"])
        if frame.empty:
            return
        if data_id == "public_option_risk":
            result = _rename(
                frame,
                {
                    "SECURITY_ID": "asset_id",
                    "CONTRACT_ID": "contract_id",
                    "TRADE_DATE": "timestamp",
                    "CONTRACT_TYPE": "option_type",
                    "IMPLC_VOLATLTY": "implied_volatility",
                    "DELTA_VALUE": "delta",
                    "GAMMA_VALUE": "gamma",
                    "VEGA_VALUE": "vega",
                },
            )
            result["underlying_id"] = result.contract_id.str[:6]
        else:
            mapping = {
                "UNDERLYING_SECURITY_ID": "underlying_id",
                "SECURITY_ID": "security_id",
                "CONTRACT_ID": "contract_id",
                "CALL_OR_PUT": "option_type",
                "EXPIRE_DATE": "expiry_date",
                "START_DATE": "start_date",
                "EXERCISE_PRICE": "exercise_price",
                "CONTRACT_UNIT": "contract_unit",
                "ORG_CONTRACT_ID": "original_contract_id",
                "ORG_EXERCISE_PRICE": "original_exercise_price",
                "ORG_CONTRACT_UNIT": "original_contract_unit",
                "ADJUST_DATE": "timestamp",
            }
            result = _rename(frame, mapping)
            result["asset_id"] = (
                result.security_id
                + ":"
                + result.contract_id
                + ":"
                + frame.index.astype("str")
            )
        yield result
        return
    if source == "sec_nport_target_holdings_extracted":
        for frame in pd.read_csv(path, dtype="str", chunksize=50000):
            result = _rename(
                frame,
                {
                    "accession": "accession",
                    "series_id": "series_id",
                    "issuer_cusip": "cusip",
                    "report_date": "report_date",
                    "sub_type": "filing_type",
                    "currency_value_as_provided": "value",
                    "currency_code_as_provided": "currency",
                    "balance_as_provided": "balance",
                    "balance_units_as_provided": "balance_unit",
                    "asset_category": "asset_category",
                    "percentage_as_provided": "portfolio_weight_percent",
                    "payoff_profile": "payoff_profile",
                    "derivative_category": "derivative_category",
                    "filing_date": "publication_date",
                    "published_at": "published_at",
                    "source_zip_sha256": "source_sha256",
                    "source_retrieved_at": "retrieved_at",
                    "evidence_url": "source_url",
                },
            )
            result["asset_id"] = result.accession + ":" + frame.holding_id
            result["timestamp"] = result.report_date
            yield result
        return
    if source == "eastmoney_cn_etf_daily":
        payload = json.loads(path.read_bytes())["data"]
        frame = pd.DataFrame(
            [line.split(",") for line in payload["klines"]],
            columns=[
                "timestamp",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude_percent",
                "return_percent",
                "price_change",
                "turnover_rate_percent",
            ],
        )
        if frame.empty:
            return
        _numbers(frame, ("volume",))
        frame["volume"] *= 100
        frame["asset_id"] = str(payload["code"]).zfill(6)
        frame["price_basis"] = "unadjusted_raw"
        yield frame
    elif source in {
        "sec_13f_submission_metadata_extracted",
        "sec_13f_target_holdings_extracted",
    }:
        for frame in pd.read_csv(path, dtype="str", chunksize=50000):
            common = {
                "accession": "accession",
                "manager_cik": "manager_cik",
                "report_period": "report_date",
                "filing_type": "filing_type",
                "is_amendment": "is_amendment",
                "amendment_type": "amendment_type",
                "filing_date": "publication_date",
                "published_at": "published_at",
                "source_zip_sha256": "source_sha256",
            }
            mapping = {
                **common,
                **(
                    {
                        "amendment_number": "amendment_number",
                        "report_type": "report_type",
                        "reported_table_entry_total": "reported_table_entry_total",
                        "source_infotable_row_count": "source_infotable_row_count",
                        "target_holding_row_count": "target_holding_row_count",
                        "source_url": "source_url",
                    }
                    if data_id == "public_13f_filings"
                    else {
                        "cusip": "cusip",
                        "value_usd": "value_usd",
                        "shares_or_principal_as_provided": "shares_or_principal",
                        "share_type": "share_type",
                        "put_call": "put_call",
                        "investment_discretion": "investment_discretion",
                        "evidence_url": "source_url",
                    }
                ),
            }
            result = _rename(frame, mapping)
            if "source_retrieved_at" in frame:
                result["retrieved_at"] = frame.source_retrieved_at
            result["timestamp"] = result.report_date
            result["asset_id"] = (
                result.accession
                if data_id == "public_13f_filings"
                else result.accession + ":" + frame.infotable_sk
            )
            result["_clock_source"] = "sec_13f:" + result.accession
            yield result
    elif source == "csrc_official_fund_financial_cells_extracted":
        for frame in pd.read_csv(path, dtype="str", chunksize=50000):
            result = _rename(
                frame,
                {
                    "primary_fund_code": "fund_code",
                    "economic_portfolio_id": "economic_portfolio_id",
                    "share_class_header_as_provided": "share_class_header",
                    "report_period_end": "report_period_end",
                    "value_observation_date": "value_observation_date",
                    "table_kind": "table_kind",
                    "field": "field",
                    "unit": "unit",
                    "denominator": "denominator",
                    "value_decimal": "value",
                    "report_version_identity": "report_version_identity",
                    "source_retrieved_at": "retrieved_at",
                    "source_url": "source_url",
                    "source_sha256": "source_sha256",
                    "published_at": "published_at",
                },
            )
            result["timestamp"] = result.report_period_end
            result["_clock_source"] = "cn_fund_report:" + result.report_version_identity
            result["publication_date"] = pd.concat(
                [
                    pd.to_datetime(frame.publication_date),
                    pd.to_datetime(frame.report_declared_send_date),
                ],
                axis=1,
            ).max(axis=1)
            result["asset_id"] = (
                result.report_version_identity
                + ":"
                + result.table_kind
                + ":"
                + frame.leaf_table_index
                + ":"
                + frame.source_row_index
                + ":"
                + frame.expanded_column_index
            )
            yield result
    elif source == "cffex_contract_parameters":
        frame = pd.DataFrame(
            [
                {child.tag: child.text for child in node}
                for node in ET.parse(path).getroot()
            ]
        )
        result = _rename(
            frame,
            {
                "TRADING_DAY": "timestamp",
                "PRODUCT_ID": "product",
                "INSTRUMENT_ID": "asset_id",
                "OPEN_DATE": "open_date",
                "END_TRADING_DAY": "expiry_date",
            },
        )
        result = result.loc[result["product"].isin(("IF", "IC", "IH", "IM"))]
        result["publication_date"] = result.timestamp
        yield result
    elif source == "cffex_member_rankings":
        frame = pd.DataFrame(
            [
                {child.tag.lower(): child.text for child in node}
                for node in ET.parse(path).getroot()
                if node.tag.lower() == "data"
            ]
        )
        if frame.empty:
            return
        result = _rename(
            frame,
            {
                "instrumentid": "contract_id",
                "tradingday": "timestamp",
                "datatypeid": "position_side",
                "rank": "rank",
                "partyid": "member_id",
                "productid": "product",
                "volume": "quantity",
                "varvolume": "quantity_change",
            },
        )
        result = result.loc[result["product"].isin(("IF", "IC", "IH", "IM"))]
        result["asset_id"] = (
            result.contract_id + ":" + result.position_side + ":" + result["rank"]
        )
        result["publication_date"] = result.timestamp
        yield result
    elif source in SOURCES["public_refinancing_balances"]:
        payload = json.loads(path.read_bytes())
        rows = payload["data"]["list"]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return
        result = _rename(
            frame,
            {
                "date": "timestamp",
                "b2": "asset_id",
                "c2": "beginning_quantity",
                "i2": "lent_quantity",
                "l2": "ending_quantity",
                "m2": "ending_balance",
            },
        )
        result = result.loc[
            result.asset_id.astype("str").str.fullmatch(r"\d{6}")
        ].copy()
        _numbers(
            result,
            (
                "beginning_quantity",
                "lent_quantity",
                "ending_quantity",
                "ending_balance",
            ),
        )
        result[
            ["beginning_quantity", "lent_quantity", "ending_quantity", "ending_balance"]
        ] *= 10000
        result["scope"] = (
            "market_maker"
            if source == "csf_market_making_borrowing_summary"
            else "securities_refinancing"
        )
        yield result
    elif source == "eastmoney_fund_history_extracted":
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = payload["fields"]
        frame = pd.DataFrame(fields["Data_netWorthTrend"])
        if frame.empty:
            return
        result = pd.DataFrame(
            {
                "fund_code": str(payload["fund_code"]).zfill(6),
                "timestamp": pd.to_datetime(frame.x, unit="ms", utc=True)
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
                .dt.normalize(),
                "nav_cny": frame.y,
                "reported_return_percent": frame.get(
                    "equityReturn", pd.Series(pd.NA, index=frame.index)
                ),
            }
        )
        accumulated = pd.DataFrame(
            fields.get("Data_ACWorthTrend", []),
            columns=["epoch_ms", "accumulated_nav_cny"],
        )
        accumulated["timestamp"] = (
            pd.to_datetime(accumulated.epoch_ms, unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        result = result.merge(
            accumulated[["timestamp", "accumulated_nav_cny"]],
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
        result["asset_id"] = result.fund_code
        yield result
    elif source == "public_index_weight_history_extracted":
        frame = pd.read_csv(path, dtype="str")
        result = _rename(
            frame,
            {
                "index_code": "source_asset_id",
                "constituent_code": "constituent_asset_id",
                "observation_date": "timestamp",
                "weight_percent": "weight",
                **{
                    name: name
                    for name in (
                        "source_url",
                        "source_sha256",
                        "retrieved_at",
                        "publication_date",
                        "published_at",
                    )
                },
            },
        )
        _numbers(result, ("weight",))
        if not result.weight.between(0, 100).all():
            raise ValueError("public_index_weight_percent_invalid")
        result["weight"] /= 100
        result["asset_id"] = result.source_asset_id + ":" + result.constituent_asset_id
        yield result
    elif source.startswith("etf_pcf_"):
        frame = pd.read_csv(path, dtype="str")
        mapping = {
            "trading_date": "timestamp",
            **{name: name for name in DATASETS[data_id]},
            **{
                name: name
                for name in (
                    "source_url",
                    "source_sha256",
                    "retrieved_at",
                    "publication_date",
                    "published_at",
                )
            },
        }
        if data_id == "public_etf_pcf_baskets":
            mapping.pop("declared_component_count")
            mapping["original_component_count"] = "declared_component_count"
        result = _rename(frame, mapping)
        result["asset_id"] = result.fund_code
        if data_id == "public_etf_pcf_components":
            result["asset_id"] += ":" + result.component_code
        result["_clock_source"] = result.fund_code.map(
            {
                "510300": "sse_huatai_pcf",
                "510500": "sse_southern_pcf",
                "159915": "szse_efunds_pcf",
                "518880": "sse_huaan_pcf",
            }
        )
        yield result
    elif source == "finra_margin":
        frame = pd.read_excel(path, dtype="str")
        result = _rename(
            frame,
            {
                "Year-Month": "timestamp",
                (
                    "Debit Balances in Customers' Securities Margin Accounts"
                ): "margin_debit",
                "Free Credit Balances in Customers' Cash Accounts": "cash_free_credit",
                (
                    "Free Credit Balances in Customers' Securities Margin Accounts"
                ): "margin_free_credit",
            },
        )
        result["timestamp"] = pd.to_datetime(result.timestamp, format="%Y-%m")
        result["asset_id"] = "US_MARGIN_ACCOUNTS"
        _numbers(result, DATASETS[data_id])
        yield result
    elif source == "cboe_put_call_extracted":
        table = pd.read_csv(path, dtype="str")
        result = _rename(
            table,
            {
                "observation_date": "timestamp",
                "ratio_name": "ratio_name",
                "put_call_ratio": "put_call_ratio",
                "source_url": "source_url",
                "source_sha256": "source_sha256",
                "publication_date": "publication_date",
                "published_at": "published_at",
            },
        )
        _numbers(result, ("put_call_ratio",))
        result["asset_id"] = result.ratio_name
        yield result
    elif source in {
        "cn_fund_report_full_holdings_csv_evidence",
        "cn_fund_report_descriptors_extracted",
    }:
        if data_id == "public_fund_holdings":
            frames = pd.read_csv(path, dtype="str", chunksize=50000)
        else:
            frames = (
                pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["records"]),
            )
        for frame in frames:
            if frame.empty:
                continue
            mapping = {
                "primary_fund_code": "fund_code",
                "economic_portfolio_id": "economic_portfolio_id",
                "report_version_identity": "report_version_identity",
                "report_period_end": "timestamp",
                "disclosure_scope": "disclosure_scope",
                "published_at": "published_at",
                "source_url": "source_url",
                "source_sha256": "source_sha256",
                "source_retrieved_at": "retrieved_at",
            }
            if data_id == "public_fund_holdings":
                mapping.update(
                    {
                        field: field
                        for field in (
                            "stock_code",
                            "stock_market",
                            "shares_held",
                            "fair_value_cny",
                            "nav_weight_percent",
                        )
                    }
                )
                mapping["source_path"] = "report_path"
            else:
                mapping.update(
                    {
                        "stock_disclosure_status": "stock_disclosure_status",
                        "stock_rows": "stock_rows",
                    }
                )
            result = _rename(frame, mapping)
            result["publication_date"] = pd.concat(
                [
                    pd.to_datetime(
                        frame[
                            "published_date"
                            if data_id == "public_fund_holdings"
                            else "publication_date"
                        ]
                    ),
                    pd.to_datetime(frame.report_declared_send_date),
                ],
                axis=1,
            ).max(axis=1)
            result["asset_id"] = result.report_version_identity
            result["_clock_source"] = "cn_fund_report:" + result.report_version_identity
            if data_id == "public_fund_holdings":
                leaf = frame.get(
                    "source_leaf_table_index", pd.Series("", index=frame.index)
                ).fillna("")
                result["asset_id"] += (
                    ":"
                    + leaf
                    + ":"
                    + frame.html_section_anchor.fillna("")
                    + ":"
                    + frame.source_row_index
                    + ":"
                    + result.stock_market
                    + ":"
                    + result.stock_code
                )
            else:
                flags = frame.explicit_no_stock_statement_in_stock_section
                if not flags.dropna().map(lambda value: type(value) is bool).all():
                    raise ValueError("public_fund_no_stock_flag_invalid")
                result["explicit_no_stock"] = flags.map({True: "true", False: "false"})
            yield result
    elif source in {"chinaamc_extracted_holdings", "cn_fund_report_holdings_extracted"}:
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["records"])
        result = _rename(
            frame,
            {
                "fund_code": "fund_code",
                "stock_code": "stock_code",
                "shares_held": "shares_held",
                "fair_value_cny": "fair_value_cny",
                "nav_weight_percent": "nav_weight_percent",
                "disclosure_scope": "disclosure_scope",
                "source_path": "report_path",
                "source_sha256": "source_sha256",
                "source_url": "source_url",
                "report_period_end": "timestamp",
                "published_date": "publication_date",
                "published_at": "published_at",
            },
        )
        result["asset_id"] = result.fund_code + ":" + result.stock_code
        for field in (
            "economic_portfolio_id",
            "report_version_identity",
            "stock_market",
        ):
            result[field] = frame[field] if field in frame else pd.NA
        yield result
    elif source in {"sec_13f", "sec_nport"}:
        yield from _sec(path, source, data_id=data_id)
    elif source.startswith("sse_"):
        payload = json.loads(path.read_bytes())
        frame = pd.DataFrame(payload["result"])
        if frame.empty:
            return
        if data_id == "public_fund_shares":
            result = _rename(
                frame,
                {
                    "STAT_DATE": "timestamp",
                    "SEC_CODE": "asset_id",
                    "TOT_VOL": "shares_outstanding",
                },
            )
            _numbers(result, ["shares_outstanding"])
            result["shares_outstanding"] *= 10000
        else:
            result = _rename(
                frame,
                {
                    "opDate": "timestamp",
                    "stockCode": "asset_id",
                    "rzye": "financing_balance",
                    "rzmre": "financing_buy_amount",
                    "rzche": "financing_repayment_amount",
                    "rqylje": "lending_balance",
                    "rqyl": "lending_quantity",
                    "rqmcl": "lending_sell_quantity",
                },
            )
            result["timestamp"] = pd.to_datetime(result.timestamp, format="%Y%m%d")
            result["lending_repayment_quantity"] = (
                frame["rqchl"] if "rqchl" in frame else pd.NA
            )
            _numbers(result, DATASETS[data_id])
        result["asset_id"] = result.asset_id.astype("str").str.zfill(6)
        yield result
    elif source.startswith("szse_"):
        frame = pd.read_excel(path, dtype="str")
        if frame.empty:
            return
        if data_id == "public_fund_shares":
            result = _rename(
                frame,
                {
                    "日期": "timestamp",
                    "基金代码": "asset_id",
                    "基金规模(份)": "shares_outstanding",
                },
            )
        else:
            result = _rename(
                frame,
                {
                    "证券代码": "asset_id",
                    "融资买入额(元)": "financing_buy_amount",
                    "融资余额(元)": "financing_balance",
                    "融券卖出量(股/份)": "lending_sell_quantity",
                    "融券余量(股/份)": "lending_quantity",
                    "融券余额(元)": "lending_balance",
                },
            )
            result["timestamp"] = (
                pd.Timestamp(record["requested_observation_date"])
                if record.get("requested_observation_date")
                else pd.to_datetime(path.stem, format="%Y%m%d")
            )
            result["financing_repayment_amount"] = pd.NA
            result["lending_repayment_quantity"] = (
                frame["融券偿还量(股/份)"] if "融券偿还量(股/份)" in frame else pd.NA
            )
        _numbers(result, DATASETS[data_id])
        result["asset_id"] = result.asset_id.str.zfill(6)
        yield result
    elif source == "cffex_daily":
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if re.search(r"\d{8}_1\.csv$", name):
                        yield _cffex(
                            archive.read(name), re.search(r"(\d{8})_1\.csv$", name)[1]
                        )
        else:
            yield _cffex(path.read_bytes(), re.search(r"\d{8}", path.stem)[0])
    elif source == "cftc_tff":
        with zipfile.ZipFile(path) as archive:
            members = [
                name for name in archive.namelist() if name.lower().endswith(".txt")
            ]
            if len(members) != 1:
                raise ValueError("cftc_archive_members_invalid")
            for chunk in pd.read_csv(
                archive.open(members[0]), dtype="str", chunksize=100000
            ):
                result = _rename(
                    chunk,
                    {
                        "Report_Date_as_YYYY-MM-DD": "timestamp",
                        "CFTC_Contract_Market_Code": "asset_id",
                        "Lev_Money_Positions_Long_All": "leveraged_long",
                        "Lev_Money_Positions_Short_All": "leveraged_short",
                        "Lev_Money_Positions_Spread_All": "leveraged_spread",
                        "Asset_Mgr_Positions_Long_All": "asset_manager_long",
                        "Asset_Mgr_Positions_Short_All": "asset_manager_short",
                        "Dealer_Positions_Long_All": "dealer_long",
                        "Dealer_Positions_Short_All": "dealer_short",
                        "Open_Interest_All": "open_interest",
                    },
                )
                result["report_type"] = (
                    "futures_only"
                    if path.name.startswith(("fut_fin", "fin_fut"))
                    else "futures_and_options"
                )
                result["asset_id"] = result.asset_id + ":" + result.report_type
                yield result
    elif source == "finra_short_volume":
        frame = pd.read_csv(
            path, sep="|", dtype="str", keep_default_na=False, na_values=[""]
        )
        frame = frame.loc[frame.Date.str.fullmatch(r"\d{8}", na=False)]
        result = _rename(
            frame,
            {
                "Date": "timestamp",
                "Symbol": "asset_id",
                "ShortVolume": "short_volume",
                "ShortExemptVolume": "short_exempt_volume",
                "TotalVolume": "total_volume",
                "Market": "market",
            },
        )
        result["timestamp"] = pd.to_datetime(result.timestamp, format="%Y%m%d")
        result["asset_id"] = result.asset_id + ":" + result.market
        yield result
    elif source == "nasdaq_etf_ohlcv":
        payload = json.loads(path.read_bytes())
        result = _rename(
            pd.DataFrame(payload["data"]["tradesTable"]["rows"]),
            {
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
        )
        for name in ("open", "high", "low", "close", "volume"):
            result[name] = result[name].str.replace(r"[$,]", "", regex=True)
        _numbers(result, ("open", "high", "low", "close", "volume"))
        result["adjusted_close"] = float("nan")
        result["asset_id"] = path.stem.split("-")[0]
        result["adjustment_basis"] = "vendor_unspecified_not_total_return"
        yield result
    elif source in {
        "yahoo_chart_extracted",
        "yahoo_corporate_actions_extracted",
        "eastmoney_cn_etf_corporate_actions_extracted",
    }:
        frame = pd.read_csv(path, dtype="str", keep_default_na=False, na_values=[""])
        mapping = {
            "symbol": "asset_id",
            "observation_date": "timestamp",
            "source_url": "source_url",
            "source_sha256": "source_sha256",
            "publication_date": "publication_date",
            "published_at": "published_at",
        }
        fields = DATASETS[data_id]
        mapping.update({name: name for name in fields if name != "adjustment_basis"})
        result = _rename(frame, mapping)
        if source == "yahoo_chart_extracted":
            result["adjustment_basis"] = (
                "close_split_adjusted;adjusted_close_splits_dividends_capital_gains;"
                "current_vintage_not_historical_pit"
            )
            if not frame.currency.eq("USD").all():
                raise ValueError("yahoo_price_currency_not_usd")
        else:
            result["asset_id"] += ":" + result.event_type
        _numbers(
            result,
            [
                name
                for name, unit in fields.items()
                if unit not in {"text", "identifier"}
            ],
        )
        yield result
    elif source == "ssga_nav_shares":
        frame = pd.read_excel(path, header=3)
        result = _rename(
            frame,
            {
                "Date": "timestamp",
                "NAV": "nav",
                "Shares Outstanding": "shares_outstanding",
                "Total Net Assets": "total_net_assets",
            },
        )
        result = result.loc[
            pd.to_datetime(result.timestamp, errors="coerce").notna()
        ].copy()
        result["asset_id"] = path.stem.split("-")[0]
        _numbers(result, DATASETS[data_id])
        yield result


def _cffex(payload, date):
    frame = pd.read_csv(io.BytesIO(payload), encoding="gb18030", dtype="str")
    frame.columns = frame.columns.str.strip()
    frame = frame.loc[
        frame["合约代码"].str.strip().str.fullmatch(r"(?:IF|IH|IC|IM)\d{4}", na=False)
    ]
    result = _rename(
        frame,
        {
            "合约代码": "asset_id",
            "今开盘": "open",
            "最高价": "high",
            "最低价": "low",
            "今收盘": "close",
            "今结算": "settlement",
            "前结算": "previous_settlement",
            "成交量": "volume",
            "持仓量": "open_interest",
            "持仓变化": "open_interest_change",
        },
    )
    result["asset_id"] = result.asset_id.str.strip()
    result["product"] = result.asset_id.str[:2]
    result["timestamp"] = pd.to_datetime(date, format="%Y%m%d")
    _numbers(
        result, [name for name in DATASETS["public_futures_daily"] if name != "product"]
    )
    return result


def _sec(path, source, *, data_id):
    """Keep filings separate; amendments are not independent aggregate positions."""
    with zipfile.ZipFile(path) as archive:
        submission = pd.read_csv(archive.open("SUBMISSION.tsv"), sep="\t", dtype="str")
        if submission.ACCESSION_NUMBER.duplicated().any():
            raise ValueError("sec_submission_accession_duplicate")
        if source == "sec_13f":
            covers = pd.read_csv(
                archive.open("COVERPAGE.tsv"),
                sep="\t",
                dtype="str",
                usecols=["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTTYPE"],
            )
            submission = submission.merge(
                covers, on="ACCESSION_NUMBER", how="left", validate="one_to_one"
            )
            member = "INFOTABLE.tsv"
        else:
            if not submission.SUB_TYPE.isin(("NPORT-P", "NPORT-P/A")).all():
                raise ValueError("nport_public_filing_type_required")
            fund_columns = ["ACCESSION_NUMBER", "SERIES_ID"]
            if data_id == "public_nport_funds":
                fund_columns += [
                    "TOTAL_ASSETS",
                    "NET_ASSETS",
                    "TOTAL_LIABILITIES",
                    "CASH_NOT_RPTD_IN_C_OR_D",
                    *(
                        f"{prefix}_FLOW_MON{month}"
                        for prefix in ("SALES", "REINVESTMENT", "REDEMPTION")
                        for month in (1, 2, 3)
                    ),
                ]
            funds = pd.read_csv(
                archive.open("FUND_REPORTED_INFO.tsv"),
                sep="\t",
                dtype="str",
                usecols=fund_columns,
            )
            submission = submission.merge(
                funds, on="ACCESSION_NUMBER", how="left", validate="one_to_one"
            )
            member = "FUND_REPORTED_HOLDING.tsv"
        submission["FILING_DATE"] = pd.to_datetime(
            submission.FILING_DATE, format="mixed"
        )
        if data_id == "public_nport_funds":
            result = _rename(
                submission,
                {
                    "ACCESSION_NUMBER": "accession",
                    "SERIES_ID": "series_id",
                    "REPORT_DATE": "report_date",
                    "SUB_TYPE": "filing_type",
                    "FILING_DATE": "publication_date",
                    "TOTAL_ASSETS": "total_assets_usd",
                    "NET_ASSETS": "net_assets_usd",
                    "TOTAL_LIABILITIES": "liabilities_usd",
                    "CASH_NOT_RPTD_IN_C_OR_D": "cash_outside_holdings_usd",
                    **{
                        f"{prefix}_FLOW_MON{month}": f"{flow}_month_{month}_usd"
                        for prefix, flow in (
                            ("SALES", "sales"),
                            ("REINVESTMENT", "reinvestment"),
                            ("REDEMPTION", "redemption"),
                        )
                        for month in (1, 2, 3)
                    },
                },
            )
            result["asset_id"] = result.accession
            result["timestamp"] = pd.to_datetime(result.report_date, format="mixed")
            result["report_date"] = result.timestamp
            yield result
            return
        identifiers = None
        if data_id == "public_security_identifiers":
            identifiers = pd.read_csv(
                archive.open("IDENTIFIERS.tsv"),
                sep="\t",
                dtype="str",
                usecols=[
                    "HOLDING_ID",
                    "IDENTIFIERS_ID",
                    "IDENTIFIER_ISIN",
                    "IDENTIFIER_TICKER",
                ],
            )
        for chunk in pd.read_csv(
            archive.open(member),
            sep="\t",
            dtype="str",
            chunksize=100000,
            usecols=["ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_CUSIP"]
            if data_id == "public_security_identifiers"
            else None,
        ):
            joined = chunk.merge(
                submission,
                on="ACCESSION_NUMBER",
                how="left",
                validate="many_to_one",
                indicator=True,
            )
            if not joined._merge.eq("both").all():
                raise ValueError("sec_holding_submission_missing")
            if data_id == "public_security_identifiers":
                joined = joined.merge(
                    identifiers, on="HOLDING_ID", how="inner", validate="one_to_many"
                )
                result = _rename(
                    joined,
                    {
                        "ACCESSION_NUMBER": "accession",
                        "HOLDING_ID": "holding_id",
                        "SERIES_ID": "series_id",
                        "ISSUER_CUSIP": "cusip",
                        "IDENTIFIER_ISIN": "isin",
                        "IDENTIFIER_TICKER": "ticker",
                        "REPORT_DATE": "report_date",
                        "SUB_TYPE": "filing_type",
                        "FILING_DATE": "publication_date",
                        "IDENTIFIERS_ID": "_holding_id",
                    },
                )
            elif source == "sec_13f":
                result = _rename(
                    joined,
                    {
                        "ACCESSION_NUMBER": "accession",
                        "CIK": "manager_cik",
                        "CUSIP": "cusip",
                        "PERIODOFREPORT": "report_date",
                        "SUBMISSIONTYPE": "filing_type",
                        "ISAMENDMENT": "is_amendment",
                        "AMENDMENTTYPE": "amendment_type",
                        "VALUE": "value_usd",
                        "SSHPRNAMT": "shares_or_principal",
                        "SSHPRNAMTTYPE": "share_type",
                        "PUTCALL": "put_call",
                        "INVESTMENTDISCRETION": "investment_discretion",
                        "FILING_DATE": "publication_date",
                        "INFOTABLE_SK": "_holding_id",
                    },
                )
                _numbers(result, ("value_usd",))
                result["value_usd"] *= result.publication_date.lt("2023-01-03").map(
                    {True: 1000, False: 1}
                )
            else:
                result = _rename(
                    joined,
                    {
                        "ACCESSION_NUMBER": "accession",
                        "SERIES_ID": "series_id",
                        "ISSUER_CUSIP": "cusip",
                        "REPORT_DATE": "report_date",
                        "SUB_TYPE": "filing_type",
                        "CURRENCY_VALUE": "value",
                        "CURRENCY_CODE": "currency",
                        "BALANCE": "balance",
                        "UNIT": "balance_unit",
                        "ASSET_CAT": "asset_category",
                        "PERCENTAGE": "portfolio_weight_percent",
                        "FILING_DATE": "publication_date",
                        "HOLDING_ID": "_holding_id",
                    },
                )
                result["payoff_profile"] = (
                    joined["PAYOFF_PROFILE"] if "PAYOFF_PROFILE" in joined else pd.NA
                )
                result["derivative_category"] = (
                    joined["DERIVATIVE_CAT"] if "DERIVATIVE_CAT" in joined else pd.NA
                )
            result["asset_id"] = result.accession + ":" + result._holding_id
            if source == "sec_13f":
                result["_clock_source"] = "sec_13f:" + result.accession
            result["timestamp"] = pd.to_datetime(result.report_date, format="mixed")
            result["report_date"] = result.timestamp
            yield result
