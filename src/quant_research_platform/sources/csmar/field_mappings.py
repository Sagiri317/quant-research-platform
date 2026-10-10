"""Internal CSMAR source contracts and provider field mapping helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import pandas as pd

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.sources.csmar.ids import market_bars_data_id

MARKET_BARS_DATA_ID = market_bars_data_id(DataFrequency.DAILY)


@dataclass(frozen=True)
class _RawTableContract:
    """Raw CSMAR export contract before internal standardization."""

    table_code: str
    required_columns: tuple[str, ...]


RAW_TABLE_CONTRACTS = MappingProxyType(
    {
        "stock_master": _RawTableContract(
            table_code="TRDNEW_Co",
            required_columns=(
                "Stkcd",
                "Stknme",
                "Listdt",
                "DelistedDate",
                "Markettype",
                "Nnindcd",
                "Nnindnme",
                "Statco",
                "Statdt",
            ),
        ),
        MARKET_BARS_DATA_ID: _RawTableContract(
            table_code="TRDNEW_Dalyr",
            required_columns=(
                "Stkcd",
                "Trddt",
                "Markettype",
                "Trdsta",
                "Filling",
                "Opnprc",
                "Hiprc",
                "Loprc",
                "Clsprc",
                "Dnshrtrd",
                "Dnvaltrd",
                "Ahshrtrd_D",
                "Ahvaltrd_D",
                "Dsmvosd",
                "Dsmvtl3",
                "Dretwd",
                "Adjprcwd",
                "PreClosePrice",
                "LimitStatus",
            ),
        ),
        "market_cap": _RawTableContract(
            table_code="TRDNEW_Capchg",
            required_columns=("Stkcd", "Shrchgdt", "Nshrttl", "Nshra"),
        ),
        "adjustment": _RawTableContract(
            table_code="TRD_AdjustFactor",
            required_columns=(
                "Symbol",
                "TradingDate",
                "FwardFactor",
                "BwardFactor",
                "CumulateFwardFactor",
                "CumulateBwardFactor",
            ),
        ),
        "index_market": _RawTableContract(
            table_code="IDX_Idxtrd",
            required_columns=(
                "Indexcd",
                "Idxtrd01",
                "Idxtrd02",
                "Idxtrd03",
                "Idxtrd04",
                "Idxtrd05",
                "Idxtrd06",
                "Idxtrd07",
                "Idxtrd08",
                "Idxtrd09",
            ),
        ),
        "index_info": _RawTableContract(
            table_code="IDX_Idxinfo",
            required_columns=("Indexcd", "Idxinfo01", "Idxinfo11", "Idxinfo02"),
        ),
        "index_membership": _RawTableContract(
            table_code="IDX_Chgsmp",
            required_columns=(
                "Indexcd",
                "Chgsmp01",
                "Chgsmp02",
                "Chgsmp03",
                "Chgsmp04",
                "Chgsmp05",
                "Chgsmp06",
                "Chgsmp07",
            ),
        ),
        "index_weights": _RawTableContract(
            table_code="IDX_Smprat",
            required_columns=("Indexcd", "Enddt", "Stkcd", "Constdnme", "Weight"),
        ),
        "financial_balance": _RawTableContract(
            table_code="FS_Combas",
            required_columns=(
                "Stkcd",
                "ShortName",
                "IfCorrect",
                "DeclareDate",
                "Accper",
                "Typrep",
                "A001000000",
                "A002000000",
                "A003100000",
                "A001101000",
                "A002101000",
                "A002125000",
                "A002201000",
                "A002203000",
            ),
        ),
        "financial_income": _RawTableContract(
            table_code="FS_Comins",
            required_columns=(
                "Stkcd",
                "ShortName",
                "IfCorrect",
                "DeclareDate",
                "Accper",
                "Typrep",
                "B001101000",
                "B001201000",
                "B002000000",
                "B002000101",
            ),
        ),
        "financial_cashflow": _RawTableContract(
            table_code="FS_Comscfd",
            required_columns=(
                "Stkcd",
                "ShortName",
                "IfCorrect",
                "DeclareDate",
                "Accper",
                "Typrep",
                "C001000000",
                "C002006000",
            ),
        ),
        "industry_classification": _RawTableContract(
            table_code="STK_INDUSTRYCLASS",
            required_columns=(
                "Symbol",
                "IndustryClassificationID",
                "ImplementDate",
                "IndustryCode",
                "IndustryName",
            ),
        ),
        "industry_classification_annual": _RawTableContract(
            table_code="STK_IndustryClassAnl",
            required_columns=(
                "Symbol",
                "EndDate",
                "IndustryCode1",
                "IndustryName1",
            ),
        ),
        "industry_classification_sets": _RawTableContract(
            table_code="STK_INDCLASSIFYSETS",
            required_columns=(
                "INDUSTRYCODE",
                "INDCLASSIFYSYSTEMCODE",
                "INDUSTRYNAME",
                "P_INDUSTRYCODE",
                "INDUSTRYID",
                "RANK",
            ),
        ),
        "industry_classification_versions": _RawTableContract(
            table_code="STK_INDCLASSIFYVERSION",
            required_columns=(
                "IndClassifySystemCode",
                "IndClassifySystem",
                "ImplementDate",
                "AbandonDate",
                "InstitutionName",
            ),
        ),
        "listing_status_changes": _RawTableContract(
            table_code="STK_ITEMCHANGE",
            required_columns=(
                "Symbol",
                "DeclareDate",
                "ChangeDate",
                "ChangedItem",
                "ValueBefore",
                "ValueAfter",
            ),
        ),
        "stock_info_changes": _RawTableContract(
            table_code="STK_LISTEDCOINFOCHG",
            required_columns=(
                "Symbol",
                "AnnouncementDate",
                "ImplementDate",
                "ChangedItem",
                "Value_Before",
                "Value_After",
            ),
        ),
        "stock_info_annual": _RawTableContract(
            table_code="STK_LISTEDCOINFOANL",
            required_columns=(
                "Symbol",
                "EndDate",
                "IndustryCode",
                "IndustryName",
                "LISTINGDATE",
                "LISTINGSTATE",
            ),
        ),
        "trading_no_limit": _RawTableContract(
            table_code="TRD_NoLimit",
            required_columns=("TradingDate", "Symbol", "Reason"),
        ),
        "special_treatment_changes": _RawTableContract(
            table_code="BDT_Trdchg",
            required_columns=("Symbol", "Chgtype", "Annoudt", "Execudt"),
        ),
        "equity_changes": _RawTableContract(
            table_code="FN_FN046",
            required_columns=("Stkcd", "Accper", "Typrep"),
        ),
        "analyst_forecast": _RawTableContract(
            table_code="AF_Forecast",
            required_columns=(
                "Stkcd",
                "ReportID",
                "AnanmID",
                "DeclareDate",
                "Fenddt",
                "InstitutionID",
                "Feps",
                "Fpe",
            ),
        ),
        "trading_calendar": _RawTableContract(
            table_code="TRDNEW_Cale",
            required_columns=("Markettype", "Clddt", "Daywk", "State"),
        ),
    }
)

COLUMN_MAPPINGS = {
    "stock_master": {
        "Stkcd": "stock_code",
        "Stknme": "stock_name",
        "Listdt": "listing_date",
        "DelistedDate": "delisting_date",
        "Markettype": "market_type",
        "Nnindcd": "industry_level1_code",
        "Nnindnme": "industry_level1_name",
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
        "Ahshrtrd_D": "after_hours_volume",
        "Ahvaltrd_D": "after_hours_amount",
        "Dsmvosd": "float_market_cap_thousand",
        "Dsmvtl3": "total_market_cap_thousand",
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
    "index_membership": {
        "Indexcd": "index_code",
        "Chgsmp01": "effective_date",
        "Chgsmp02": "constituent_stock_code",
        "Chgsmp03": "constituent_name",
        "Chgsmp04": "change_type",
        "Chgsmp05": "security_type",
        "Chgsmp06": "announcement_date",
        "Chgsmp07": "market_code",
    },
    "index_weights": {
        "Indexcd": "index_code",
        "Enddt": "effective_date",
        "Stkcd": "constituent_stock_code",
        "Constdnme": "constituent_name",
        "Weight": "weight_pct",
    },
    "financial_balance": {
        "Stkcd": "stock_code",
        "ShortName": "stock_name",
        "IfCorrect": "is_corrected",
        "DeclareDate": "correction_disclosure_date",
        "Accper": "report_period",
        "Typrep": "report_type",
        "A001000000": "total_assets",
        "A002000000": "total_liabilities",
        "A003100000": "parent_equity",
        "A001101000": "cash",
        "A002101000": "short_term_borrowings",
        "A002125000": "current_portion_noncurrent_liabilities",
        "A002201000": "long_term_borrowings",
        "A002203000": "bonds_payable",
    },
    "financial_income": {
        "Stkcd": "stock_code",
        "ShortName": "stock_name",
        "IfCorrect": "is_corrected",
        "DeclareDate": "correction_disclosure_date",
        "Accper": "report_period",
        "Typrep": "report_type",
        "B001101000": "operating_revenue",
        "B001201000": "operating_cost",
        "B002000000": "consolidated_net_income",
        "B002000101": "parent_net_income",
    },
    "financial_cashflow": {
        "Stkcd": "stock_code",
        "ShortName": "stock_name",
        "IfCorrect": "is_corrected",
        "DeclareDate": "correction_disclosure_date",
        "Accper": "report_period",
        "Typrep": "report_type",
        "C001000000": "operating_cash_flow",
        "C002006000": "capex_cash_paid",
    },
    "industry_classification": {
        "Symbol": "stock_code",
        "IndustryClassificationID": "industry_standard_id",
        "ImplementDate": "implement_date",
        "IndustryCode": "industry_level1_code",
        "IndustryName": "industry_level1_name",
    },
    "industry_classification_annual": {
        "Symbol": "stock_code",
        "EndDate": "effective_date",
        "IndustryCode1": "industry_level1_code",
        "IndustryName1": "industry_level1_name",
    },
    "industry_classification_sets": {
        "INDUSTRYCODE": "industry_code",
        "INDCLASSIFYSYSTEMCODE": "industry_standard_id",
        "INDUSTRYNAME": "industry_name",
        "P_INDUSTRYCODE": "parent_industry_code",
        "INDUSTRYID": "industry_id",
        "RANK": "industry_level",
    },
    "industry_classification_versions": {
        "IndClassifySystemCode": "industry_standard_id",
        "IndClassifySystem": "industry_standard_name",
        "ImplementDate": "implement_date",
        "AbandonDate": "abandon_date",
        "InstitutionName": "institution_name",
    },
    "listing_status_changes": {
        "Symbol": "stock_code",
        "DeclareDate": "announcement_date",
        "ChangeDate": "effective_date",
        "ChangedItem": "changed_item",
        "ValueBefore": "value_before",
        "ValueAfter": "value_after",
    },
    "stock_info_changes": {
        "Symbol": "stock_code",
        "AnnouncementDate": "announcement_date",
        "ImplementDate": "effective_date",
        "ChangedItem": "changed_item",
        "Value_Before": "value_before",
        "Value_After": "value_after",
    },
    "stock_info_annual": {
        "Symbol": "stock_code",
        "EndDate": "effective_date",
        "IndustryCode": "industry_level1_code",
        "IndustryName": "industry_level1_name",
        "LISTINGDATE": "listing_date",
        "LISTINGSTATE": "listing_state",
    },
    "trading_no_limit": {
        "TradingDate": "trade_date",
        "Symbol": "stock_code",
        "Reason": "reason",
    },
    "special_treatment_changes": {
        "Symbol": "stock_code",
        "Chgtype": "change_type",
        "Annoudt": "announcement_date",
        "Execudt": "effective_date",
    },
    "equity_changes": {
        "Stkcd": "stock_code",
        "Accper": "report_period",
        "Typrep": "report_type",
    },
    "analyst_forecast": {
        "Stkcd": "stock_code",
        "ReportID": "report_id",
        "AnanmID": "analyst_ids",
        "DeclareDate": "declare_date",
        "Fenddt": "forecast_end_date",
        "InstitutionID": "institution_id",
        "Feps": "forecast_eps",
        "Fpe": "forecast_pe",
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


def _validate_mapping_facts(
    mappings: Mapping[str, Mapping[str, str]],
    aliases_by_dataset: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> None:
    for dataset, mapping in mappings.items():
        aliases = aliases_by_dataset.get(dataset, {})
        if len(mapping.values()) != len(set(mapping.values())):
            raise ValueError("invalid_csmar_column_mapping_facts")
        alias_owners: dict[str, str] = {}
        for primary, provider_aliases in aliases.items():
            if primary not in mapping:
                raise ValueError("invalid_csmar_column_mapping_facts")
            for alias in provider_aliases:
                if alias in mapping or alias in alias_owners:
                    raise ValueError("invalid_csmar_column_mapping_facts")
                alias_owners[alias] = primary
    if set(aliases_by_dataset).difference(mappings):
        raise ValueError("invalid_csmar_column_mapping_facts")


_validate_mapping_facts(COLUMN_MAPPINGS, COLUMN_ALIASES)
COLUMN_MAPPINGS = MappingProxyType(
    {name: MappingProxyType(dict(values)) for name, values in COLUMN_MAPPINGS.items()}
)
COLUMN_ALIASES = MappingProxyType(
    {
        name: MappingProxyType(
            {column: tuple(aliases) for column, aliases in values.items()}
        )
        for name, values in COLUMN_ALIASES.items()
    }
)


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


def validate_raw_column_names(columns: tuple[str, ...], dataset: str) -> None:
    """Reject ambiguous aliases while allowing bounded provider-superset fields."""

    aliases = COLUMN_ALIASES.get(dataset, {})
    available = set(columns)
    for primary, provider_aliases in aliases.items():
        if sum(column in available for column in (primary, *provider_aliases)) > 1:
            raise ValueError("csmar_raw_export_schema_invalid")


def map_columns(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rename raw provider columns to source-local normalized aliases."""

    columns = tuple(str(column) for column in frame.columns)
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate_csmar_provider_column")
    mapping = COLUMN_MAPPINGS[dataset]
    aliases = COLUMN_ALIASES.get(dataset, {})
    available = set(columns)
    for primary, alias_columns in aliases.items():
        present = tuple(
            column for column in (primary, *alias_columns) if column in available
        )
        if len(present) > 1:
            raise ValueError("ambiguous_csmar_column_mapping")
    missing = missing_raw_columns(frame, dataset)
    if missing:
        raise ValueError(f"{dataset} raw export is missing CSMAR columns: {missing}")
    alias_mapping = {
        alias: mapping[column]
        for column, alias_columns in aliases.items()
        for alias in alias_columns
    }
    return frame.rename(columns={**alias_mapping, **mapping})


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
