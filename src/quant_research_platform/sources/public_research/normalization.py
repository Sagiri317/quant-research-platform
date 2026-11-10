"""Source-specific schemas; NAV, market prices and trading activity stay distinct."""

import io
import json
import re
import zipfile

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_real_numeric_series

DATASETS = {
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
        "stock_code": "identifier",
        "shares_held": "shares",
        "fair_value_cny": "CNY",
        "nav_weight_percent": "percent",
        "disclosure_scope": "text",
        "report_path": "text",
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
        "value": "reported_currency",
        "currency": "identifier",
        "balance": "reported_units",
        "balance_unit": "identifier",
        "asset_category": "identifier",
        "portfolio_weight_percent": "percent",
    },
    "public_margin_balances": {
        "financing_balance": "CNY",
        "financing_buy_amount": "CNY",
        "financing_repayment_amount": "CNY",
        "lending_balance": "CNY",
        "lending_quantity": "shares_or_fund_units",
        "lending_sell_quantity": "shares_or_fund_units",
    },
    "public_fund_shares": {"shares_outstanding": "fund_units"},
    "public_etf_pcf_baskets": {
        "fund_code": "identifier",
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
    "public_finra_margin": {"finra_margin"},
    "public_option_put_call": {"cboe_put_call_extracted"},
    "public_fund_holdings": {
        "chinaamc_extracted_holdings",
        "cn_fund_report_holdings_extracted",
    },
    "public_13f_holdings": {"sec_13f"},
    "public_nport_holdings": {"sec_nport"},
    "public_margin_balances": {"sse_margin", "szse_margin"},
    "public_fund_shares": {"sse_etf_shares", "szse_etf_shares"},
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
    if record["source"].startswith("etf_pcf_"):
        return path.endswith(".csv")
    if record["source"].startswith("yahoo_"):
        return path.endswith(".csv")
    if data_id in {"public_13f_holdings", "public_nport_holdings"}:
        return path.endswith(".zip")
    if data_id == "public_option_put_call":
        return path.endswith(".csv")
    if data_id == "public_fund_holdings":
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
    if source.startswith("etf_pcf_"):
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
        yield result
    elif source in {"sec_13f", "sec_nport"}:
        yield from _sec(path, source)
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
    elif source in {"yahoo_chart_extracted", "yahoo_corporate_actions_extracted"}:
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


def _sec(path, source):
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
            funds = pd.read_csv(
                archive.open("FUND_REPORTED_INFO.tsv"),
                sep="\t",
                dtype="str",
                usecols=["ACCESSION_NUMBER", "SERIES_ID"],
            )
            submission = submission.merge(
                funds, on="ACCESSION_NUMBER", how="left", validate="one_to_one"
            )
            member = "FUND_REPORTED_HOLDING.tsv"
        submission["FILING_DATE"] = pd.to_datetime(
            submission.FILING_DATE, format="mixed"
        )
        for chunk in pd.read_csv(
            archive.open(member), sep="\t", dtype="str", chunksize=100000
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
            if source == "sec_13f":
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
            result["asset_id"] = result.accession + ":" + result._holding_id
            result["timestamp"] = pd.to_datetime(result.report_date, format="mixed")
            result["report_date"] = result.timestamp
            yield result
