"""Internal provider field mapping and normalization helpers."""

from __future__ import annotations

import re
from numbers import Integral, Real

import pandas as pd

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.sources.csmar.ids import market_bars_data_id
from quant_research_platform.sources.csmar.raw_schemas import RAW_TABLE_CONTRACTS

MARKET_BARS_DATA_ID = market_bars_data_id(DataFrequency.DAILY)

COLUMN_MAPPINGS = {
    "stock_master": {
        "Stkcd": "stock_code",
        "Stknme": "stock_name",
        "Listdt": "listing_date",
        "DelistedDate": "delisting_date",
        "Markettype": "market_type",
        "Nnindcd": "csrc_2012_industry_code",
        "Nnindnme": "csrc_2012_industry_name",
        "Statco": "company_status",
        "Statdt": "status_change_date",
    },
    MARKET_BARS_DATA_ID: {
        "Stkcd": "stock_code",
        "Trddt": "trade_date",
        "Markettype": "market_type",
        "Trdsta": "trading_status_code",
        "Filling": "filling_flag",
        "Opnprc": "open",
        "Hiprc": "high",
        "Loprc": "low",
        "Clsprc": "close",
        "Dnshrtrd": "volume",
        "Dnvaltrd": "amount",
        "Dsmvosd": "float_market_cap_thousand",
        "Dsmvtll": "total_market_cap_thousand",
        "Dretwd": "return_with_dividend",
        "Dretnd": "return_without_dividend",
        "Adjprcwd": "comparable_close_with_dividend",
        "Adjprcnd": "comparable_close_without_dividend",
        "PreClosePrice": "prev_close",
        "ChangeRatio": "change_ratio",
        "LimitDown": "limit_down",
        "LimitUp": "limit_up",
        "LimitStatus": "limit_status",
    },
    "market_cap": {
        "Stkcd": "stock_code",
        "Shrchgdt": "share_change_date",
        "Nshrttl": "total_shares",
        "Nshra": "a_float_shares",
        "Nshrb": "b_float_shares",
        "Nshrh": "h_float_shares",
    },
    "adjustment": {
        "Symbol": "stock_code",
        "TradingDate": "trade_date",
        "Stkcd": "stock_code",
        "Trddt": "trade_date",
        "FwardFactor": "forward_adjust_factor",
        "BwardFactor": "backward_adjust_factor",
        "CumulateFwardFactor": "cum_forward_adjust_factor",
        "CumulateBwardFactor": "cum_backward_adjust_factor",
    },
    "index_market": {
        "Indexcd": "index_code",
        "Idxtrd01": "trade_date",
        "Idxtrd02": "open",
        "Idxtrd03": "high",
        "Idxtrd04": "low",
        "Idxtrd05": "close",
        "Idxtrd06": "volume_10k_shares",
        "Idxtrd07": "amount_10k_cny",
        "Idxtrd08": "return_1d_pct",
        "Idxtrd09": "index_name",
    },
    "index_info": {
        "Indexcd": "index_code",
        "Idxinfo01": "index_name",
        "Idxinfo11": "start_date",
        "Idxinfo02": "base_date",
        "Idxinfo03": "base_point",
        "Idxinfo04": "sample_scope",
        "Idxinfo05": "weighting_method",
        "Idxinfo06": "publisher",
        "Idxinfo07": "index_category",
        "Idxinfo08": "index_type",
        "Idxinfo09": "market_belonging",
        "Idxinfo10": "calculation_formula",
    },
    "trading_calendar": {
        "Markettype": "source_market_type",
        "Clddt": "calendar_date",
        "Daywk": "source_day_of_week",
        "State": "source_state",
    },
}

COLUMN_ALIASES = {
    "adjustment": {
        "Symbol": ("Stkcd",),
        "TradingDate": ("Trddt",),
    },
}

FIELD_DESCRIPTIONS = {
    "stock_master": {
        "Stkcd": "证券代码",
        "Stknme": "证券简称",
        "Listdt": "上市日期",
        "DelistedDate": "退市日期",
        "Markettype": "市场类型",
        "Nnindcd": "行业代码 C，2012 版证监会行业分类代码",
        "Nnindnme": "行业名称 C，2012 版证监会行业分类名称",
        "Statco": "公司活动情况",
        "Statdt": "情况变动日",
    },
    MARKET_BARS_DATA_ID: {
        "Stkcd": "证券代码",
        "Trddt": "交易日期",
        "Markettype": "市场类型",
        "Trdsta": "交易状态",
        "Filling": "填充标识",
        "Opnprc": "日开盘价",
        "Hiprc": "日最高价",
        "Loprc": "日最低价",
        "Clsprc": "日收盘价",
        "Dnshrtrd": "日个股交易股数",
        "Dnvaltrd": "日个股交易金额",
        "Dsmvosd": "日个股流通市值",
        "Dsmvtll": "日个股总市值 1",
        "Dretwd": "考虑现金红利再投资的日个股回报率",
        "Dretnd": "不考虑现金红利的日个股回报率",
        "Adjprcwd": "考虑现金红利再投资的收盘价的可比价格",
        "Adjprcnd": "不考虑现金红利的收盘价的可比价格",
        "PreClosePrice": "昨收盘(交易所)",
        "ChangeRatio": "涨跌幅",
        "LimitDown": "跌停价",
        "LimitUp": "涨停价",
        "LimitStatus": "涨跌停状态",
    },
    "market_cap": {
        "Stkcd": "证券代码",
        "Shrchgdt": "股本变动日期",
        "Nshrttl": "总股数",
        "Nshra": "A 股流通股数",
        "Nshrb": "B 股流通股数",
        "Nshrh": "H 股流通股数",
    },
    "adjustment": {
        "Symbol": "证券代码",
        "TradingDate": "交易日期",
        "FwardFactor": "前复权因子",
        "BwardFactor": "后复权因子",
        "CumulateFwardFactor": "前累计复权因子",
        "CumulateBwardFactor": "后累计复权因子",
    },
    "index_market": {
        "Indexcd": "交易所指数代码",
        "Idxtrd01": "交易日期",
        "Idxtrd02": "开盘指数",
        "Idxtrd03": "最高指数",
        "Idxtrd04": "最低指数",
        "Idxtrd05": "收盘指数",
        "Idxtrd06": "成份证券成交量",
        "Idxtrd07": "成份证券成交金额",
        "Idxtrd08": "指数回报率",
        "Idxtrd09": "指数中文简称",
    },
    "index_info": {
        "Indexcd": "交易所指数代码",
        "Idxinfo01": "指数名称",
        "Idxinfo11": "开始日期",
        "Idxinfo02": "基准日期",
        "Idxinfo03": "基准点数",
        "Idxinfo04": "样本范围",
        "Idxinfo05": "加权方式",
        "Idxinfo06": "发布机构",
        "Idxinfo07": "指数类别",
        "Idxinfo08": "指数类型",
        "Idxinfo09": "所属市场",
        "Idxinfo10": "计算公式",
    },
    "trading_calendar": {
        "Markettype": "市场类型",
        "Clddt": "日历日期",
        "Daywk": "星期",
        "State": "开市情况",
    },
}


def missing_raw_columns(frame: pd.DataFrame, dataset: str) -> list[str]:
    """Return provider columns missing from a mappable raw export frame."""

    columns = tuple(str(column) for column in frame.columns)
    return missing_raw_column_names(columns, dataset)


def missing_raw_column_names(columns: tuple[str, ...], dataset: str) -> list[str]:
    """Return missing raw columns, accepting approved provider aliases."""

    contract = RAW_TABLE_CONTRACTS[dataset]
    available = set(columns)
    aliases = COLUMN_ALIASES.get(dataset, {})
    missing = []
    for column in contract.required_columns:
        candidates = (column, *aliases.get(column, ()))
        if not any(candidate in available for candidate in candidates):
            missing.append(column)
    return missing


def column_mapping(dataset: str) -> dict[str, str]:
    """Return raw-provider to source-local normalized column mappings."""

    return COLUMN_MAPPINGS[dataset]


def field_descriptions(dataset: str) -> dict[str, str]:
    """Return provider field descriptions for private metadata sidecars."""

    return FIELD_DESCRIPTIONS.get(dataset, {})


def map_columns(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rename raw provider columns to source-local normalized aliases."""

    mapping = column_mapping(dataset)
    missing = missing_raw_columns(frame, dataset)
    if missing:
        raise ValueError(f"{dataset} raw export is missing CSMAR columns: {missing}")
    aliases = COLUMN_ALIASES.get(dataset, {})
    alias_mapping = {
        alias: mapping[column]
        for column, alias_columns in aliases.items()
        for alias in alias_columns
    }
    return frame.rename(columns={**alias_mapping, **mapping}).copy()


_STOCK_CODE_PATTERN = re.compile(r"^[0-9]+(?:\.0)?$")


def normalize_provider_stock_code_series(
    values: pd.Series,
    *,
    error: str = "invalid_csmar_stock_code",
) -> pd.Series:
    """Normalize provider stock codes to six-character strings."""

    result: list[str] = []
    for value in pd.Series(values).tolist():
        result.append(_normalize_provider_stock_code_value(value, error=error))
    return pd.Series(result, index=pd.Series(values).index, dtype="object")


def normalize_stock_code(series: pd.Series) -> pd.Series:
    """Normalize provider stock codes to six-character strings."""

    return normalize_provider_stock_code_series(series)


def _normalize_provider_stock_code_value(value: object, *, error: str) -> str:
    if isinstance(value, bool):
        raise ValueError(error)
    if isinstance(value, Integral):
        number = int(value)
        if number < 0:
            raise ValueError(error)
        return _format_stock_code(number, error=error)
    if isinstance(value, Real):
        if pd.isna(value) or not float(value).is_integer() or value < 0:
            raise ValueError(error)
        return _format_stock_code(int(value), error=error)
    if not isinstance(value, str):
        raise ValueError(error)
    text = value.strip()
    if text in {"", "nan", "None"} or _STOCK_CODE_PATTERN.fullmatch(text) is None:
        raise ValueError(error)
    normalized = text[:-2] if text.endswith(".0") else text
    return _format_stock_code(int(normalized), error=error)


def _format_stock_code(value: int, *, error: str) -> str:
    text = str(value)
    if len(text) > 6:
        raise ValueError(error)
    return text.zfill(6)


__all__ = [
    "COLUMN_ALIASES",
    "COLUMN_MAPPINGS",
    "FIELD_DESCRIPTIONS",
    "column_mapping",
    "field_descriptions",
    "map_columns",
    "missing_raw_column_names",
    "missing_raw_columns",
    "normalize_provider_stock_code_series",
    "normalize_stock_code",
]
