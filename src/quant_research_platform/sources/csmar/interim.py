"""CSMAR raw export to source-normalized interim datasets."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.validation import (
    strict_integer_series,
    strict_real_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.sources.csmar.field_mappings import (
    COLUMN_MAPPINGS,
    map_columns,
    normalize_provider_stock_code_series,
)
from quant_research_platform.sources.csmar.fields import field_role
from quant_research_platform.sources.csmar.ids import SOURCE_ID, market_bars_data_id
from quant_research_platform.sources.csmar.raw_exports import (
    INDEX_INFO_SOURCE_CODE_COLUMNS,
    INDEX_INFO_SOURCE_CODE_ENUMS,
    STRING_CODE_DTYPES,
    dataset_files_with_identity_by_id,
    iter_private_export_chunks,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    filesystem_stat,
    require_unaliased_path,
)

MARKET_BARS_DATA_ID = market_bars_data_id(DataFrequency.DAILY)
DEFAULT_INTERIM_DATA_IDS: tuple[str, ...] = (
    MARKET_BARS_DATA_ID,
    "stock_master",
    "adjustment",
    "market_cap",
    "index_info",
    "index_market",
    "trading_calendar",
    "index_membership",
    "index_weights",
    "financial_balance",
    "financial_income",
    "financial_cashflow",
    "industry_classification",
    "industry_classification_annual",
    "industry_classification_sets",
    "industry_classification_versions",
    "listing_status_changes",
    "stock_info_changes",
    "stock_info_annual",
    "trading_no_limit",
    "special_treatment_changes",
    "equity_changes",
    "corporate_actions",
    "analyst_forecast",
)
SOURCE_ASSET_TIME_COLUMNS: tuple[str, ...] = (
    "source_asset_id",
    "timestamp",
)
MARKET_BARS_INTERIM_COLUMNS: tuple[str, ...] = (
    *SOURCE_ASSET_TIME_COLUMNS,
    "trade_date",
    "market_type",
    "trading_status_code",
    "filling_flag",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "after_hours_volume",
    "after_hours_amount",
    "prev_close",
    "limit_status",
    "period_return",
    "adj_close",
    "adj_open",
    "float_market_cap",
    "total_market_cap",
    "turnover",
)
STOCK_MASTER_INTERIM_COLUMNS: tuple[str, ...] = (
    *SOURCE_ASSET_TIME_COLUMNS,
    "stock_name",
    "listing_date",
    "delisting_date",
    "delisting_date_unresolved",
    "company_status",
    "industry_level1_code",
    "industry_level1_name",
    "status_change_date",
)
MARKET_CAP_INTERIM_COLUMNS: tuple[str, ...] = (
    *SOURCE_ASSET_TIME_COLUMNS,
    "share_change_date",
    "total_shares",
    "a_float_shares",
)
ADJUSTMENT_INTERIM_COLUMNS: tuple[str, ...] = (
    *SOURCE_ASSET_TIME_COLUMNS,
    "trade_date",
    "forward_adjust_factor",
    "backward_adjust_factor",
    "cum_forward_adjust_factor",
    "cum_backward_adjust_factor",
)
INDEX_INFO_INTERIM_COLUMNS: tuple[str, ...] = (
    "source_asset_id",
    "timestamp",
    "index_name",
    "base_point",
    "sample_scope",
    "weighting_method",
    "publisher",
    "index_category",
    "index_type",
    "market_belonging",
    "calculation_formula",
    "index_class",
    "start_date",
    "base_date",
)
INDEX_RETURN_INTERIM_COLUMNS: tuple[str, ...] = (
    "source_asset_id",
    "timestamp",
    "trade_date",
    "index_name",
    "open",
    "close",
    "period_return",
)
TRADING_CALENDAR_INTERIM_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "calendar_date",
    "source_market_type",
    "is_trading_day",
    "source_day_of_week",
    "source_state",
)
INDEX_MEMBERSHIP_INTERIM_COLUMNS: tuple[str, ...] = (
    "source_asset_id",
    "constituent_asset_id",
    "constituent_stock_code",
    "timestamp",
    "effective_date",
    "announcement_date",
    "constituent_name",
    "change_type",
    "security_type",
    "market_code",
)
INDEX_WEIGHTS_INTERIM_COLUMNS: tuple[str, ...] = (
    "source_asset_id",
    "constituent_asset_id",
    "timestamp",
    "effective_date",
    "constituent_name",
    "weight",
)
FINANCIAL_BASE_COLUMNS: tuple[str, ...] = (
    "source_asset_id",
    "timestamp",
    "stock_name",
    "report_period",
    "report_type",
    "is_corrected",
    "correction_disclosure_date",
    "version_identity",
)
FINANCIAL_COLUMNS_BY_DATA_ID: dict[str, tuple[str, ...]] = {
    "financial_balance": (
        *FINANCIAL_BASE_COLUMNS,
        "total_assets",
        "total_liabilities",
        "parent_equity",
        "cash",
        "short_term_borrowings",
        "current_portion_noncurrent_liabilities",
        "long_term_borrowings",
        "bonds_payable",
    ),
    "financial_income": (
        *FINANCIAL_BASE_COLUMNS,
        "operating_revenue",
        "operating_cost",
        "consolidated_net_income",
        "parent_net_income",
    ),
    "financial_cashflow": (
        *FINANCIAL_BASE_COLUMNS,
        "operating_cash_flow",
        "capex_cash_paid",
    ),
}
INDUSTRY_CLASSIFICATION_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "industry_standard_id",
    "implement_date",
    "industry_level1_code",
    "industry_level1_name",
)
INDUSTRY_CLASSIFICATION_ANNUAL_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "effective_date",
    "industry_level1_code",
    "industry_level1_name",
)
INDUSTRY_CLASSIFICATION_SETS_COLUMNS = (
    "industry_code",
    "industry_standard_id",
    "industry_name",
    "parent_industry_code",
    "industry_id",
    "industry_level",
)
INDUSTRY_CLASSIFICATION_VERSIONS_COLUMNS = (
    "industry_standard_id",
    "industry_standard_name",
    "timestamp",
    "implement_date",
    "abandon_date",
    "abandon_date_unresolved",
    "institution_name",
)
LISTING_STATUS_CHANGE_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "announcement_date",
    "effective_date",
    "changed_item",
    "value_before",
    "value_after",
)
STOCK_INFO_CHANGE_COLUMNS = LISTING_STATUS_CHANGE_COLUMNS
STOCK_INFO_ANNUAL_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "effective_date",
    "industry_level1_code",
    "industry_level1_name",
    "listing_date",
    "listing_state",
)
TRADING_NO_LIMIT_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "trade_date",
    "reason",
)
SPECIAL_TREATMENT_CHANGE_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "change_type",
    "announcement_date",
    "effective_date",
    "is_st",
    "is_pt",
)
CORPORATE_ACTION_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "kind",
    "market_type",
    "announcement_date",
    "settlement_date",
    "settlement_date_unresolved",
    "amount",
    "subscription_price",
)
EQUITY_CHANGE_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "report_period",
    "report_type",
    "data_source",
    "subject_code",
)
ANALYST_FORECAST_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "report_id",
    "analyst_ids",
    "declare_date",
    "forecast_end_date",
    "institution_id",
    "forecast_eps",
    "forecast_pe",
)
INTERIM_COLUMNS_BY_DATA_ID: dict[str, tuple[str, ...]] = {
    MARKET_BARS_DATA_ID: MARKET_BARS_INTERIM_COLUMNS,
    "stock_master": STOCK_MASTER_INTERIM_COLUMNS,
    "market_cap": MARKET_CAP_INTERIM_COLUMNS,
    "adjustment": ADJUSTMENT_INTERIM_COLUMNS,
    "index_info": INDEX_INFO_INTERIM_COLUMNS,
    "index_market": INDEX_RETURN_INTERIM_COLUMNS,
    "trading_calendar": TRADING_CALENDAR_INTERIM_COLUMNS,
    "index_membership": INDEX_MEMBERSHIP_INTERIM_COLUMNS,
    "index_weights": INDEX_WEIGHTS_INTERIM_COLUMNS,
    **FINANCIAL_COLUMNS_BY_DATA_ID,
    "industry_classification": INDUSTRY_CLASSIFICATION_COLUMNS,
    "industry_classification_annual": INDUSTRY_CLASSIFICATION_ANNUAL_COLUMNS,
    "industry_classification_sets": INDUSTRY_CLASSIFICATION_SETS_COLUMNS,
    "industry_classification_versions": INDUSTRY_CLASSIFICATION_VERSIONS_COLUMNS,
    "listing_status_changes": LISTING_STATUS_CHANGE_COLUMNS,
    "stock_info_changes": STOCK_INFO_CHANGE_COLUMNS,
    "stock_info_annual": STOCK_INFO_ANNUAL_COLUMNS,
    "trading_no_limit": TRADING_NO_LIMIT_COLUMNS,
    "special_treatment_changes": SPECIAL_TREATMENT_CHANGE_COLUMNS,
    "equity_changes": EQUITY_CHANGE_COLUMNS,
    "corporate_actions": CORPORATE_ACTION_COLUMNS,
    "analyst_forecast": ANALYST_FORECAST_COLUMNS,
}


@dataclass(frozen=True)
class InterimImportResult:
    """Summary of a CSMAR interim materialization run."""

    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]


def interim_specs() -> tuple[InterimDataSpec, ...]:
    return tuple(
        _interim_spec(data_id, columns)
        for data_id, columns in sorted(INTERIM_COLUMNS_BY_DATA_ID.items())
    )


def interim_catalog() -> InterimDataCatalog:
    return InterimDataCatalog(interim_specs())


def normalize_raw_frame(
    frame: pd.DataFrame,
    data_id: str,
) -> pd.DataFrame:
    mapped = map_columns(frame, data_id)
    counts = {"raw_rows": len(mapped)}
    # Numeric interpretation belongs here once. Invalid optional facts remain
    # unavailable; they do not erase independent observations or measures.
    if data_id != "corporate_actions":
        for source, column in COLUMN_MAPPINGS[data_id].items():
            if (
                source in STRING_CODE_DTYPES
                or source in INDEX_INFO_SOURCE_CODE_COLUMNS
                or column not in mapped
            ):
                continue
            values = mapped[column].replace(r"^\s*$", pd.NA, regex=True)
            numeric = strict_real_numeric_series(
                values, error="invalid_source_numeric", errors="coerce"
            )
            valid = np.isfinite(numeric)
            if column in {
                "market_type",
                "filling_flag",
                "source_market_type",
                "source_day_of_week",
                "security_type",
            }:
                valid &= (
                    numeric.eq(np.floor(numeric))
                    & numeric.ge(-(2**63))
                    & numeric.lt(2**63)
                )
            invalid_count = int((values.notna() & ~valid).sum())
            if invalid_count:
                counts[f"invalid_{column}_rows"] = invalid_count
            mapped[column] = numeric.where(valid)
            if data_id == MARKET_BARS_DATA_ID and column in {
                "after_hours_volume",
                "after_hours_amount",
            }:
                mapped[f"_invalid_{column}"] = values.notna() & ~valid
    valid_rows = pd.Series(True, index=mapped.index)
    stock_code_column = (
        "constituent_stock_code" if data_id == "index_weights" else "stock_code"
    )
    if stock_code_column in mapped:
        text = mapped[stock_code_column].astype("string").str.strip()
        valid_rows &= text.str.fullmatch(r"[0-9]{1,6}(?:\.0)?").fillna(False)
    if "index_code" in mapped:
        valid_rows &= (
            mapped["index_code"]
            .astype("string")
            .str.fullmatch(r"[0-9A-Za-z_.-]{1,32}")
            .fillna(False)
        )
    identity_columns = {
        "industry_classification": ("industry_standard_id",),
        "industry_classification_sets": ("industry_standard_id", "industry_code"),
        "industry_classification_versions": ("industry_standard_id",),
        "listing_status_changes": ("changed_item",),
        "stock_info_changes": ("changed_item",),
        "equity_changes": ("report_type", "data_source", "subject_code"),
        "financial_balance": ("report_type",),
        "financial_income": ("report_type",),
        "financial_cashflow": ("report_type",),
        "index_membership": ("constituent_stock_code",),
        "index_weights": ("constituent_stock_code",),
    }.get(data_id, ())
    for column in identity_columns:
        valid_rows &= mapped[column].astype("string").str.strip().ne("").fillna(False)
    for column in (
        "listing_date",
        "delisting_date",
        "status_change_date",
        "announcement_date",
        "abandon_date",
        "declare_date",
    ):
        if column not in mapped or data_id == "corporate_actions":
            continue
        values = mapped[column]
        parsed = _parse_optional_csmar_date(values)
        invalid = (
            values.notna() & values.astype("string").str.strip().ne("") & parsed.isna()
        )
        if invalid.any():
            counts[f"invalid_{column}_rows"] = int(invalid.sum())
        mapped[column] = parsed
        if data_id == "industry_classification_versions" and column == "abandon_date":
            mapped["abandon_date_unresolved"] = invalid
        if data_id == "stock_master" and column == "delisting_date":
            mapped["delisting_date_unresolved"] = invalid
        if data_id == "stock_master" and column == "status_change_date":
            mapped["_invalid_status_change_date"] = invalid
    date_column = {
        "market_bars_1d": "trade_date",
        "adjustment": "trade_date",
        "market_cap": "share_change_date",
        "index_market": "trade_date",
        "trading_calendar": "calendar_date",
        "corporate_actions": "timestamp",
        "financial_balance": "report_period",
        "financial_income": "report_period",
        "financial_cashflow": "report_period",
        "industry_classification": "implement_date",
        "industry_classification_versions": "implement_date",
        "industry_classification_annual": "effective_date",
        "listing_status_changes": "effective_date",
        "stock_info_changes": "effective_date",
        "stock_info_annual": "effective_date",
        "trading_no_limit": "trade_date",
        "special_treatment_changes": "effective_date",
        "equity_changes": "report_period",
        "analyst_forecast": "forecast_end_date",
        "index_weights": "effective_date",
        "index_membership": "effective_date",
    }.get(data_id)
    if date_column is not None:
        mapped[date_column] = _parse_optional_csmar_date(mapped[date_column])
        valid_rows &= mapped[date_column].notna()
    if data_id == "trading_calendar":
        valid_rows &= mapped["source_market_type"].notna()
        valid_rows &= (
            mapped["source_state"]
            .astype("string")
            .str.strip()
            .str.upper()
            .isin(("O", "C"))
        )
    if data_id == "special_treatment_changes":
        valid_rows &= (
            mapped["change_type"].astype("string").str.strip().ne("").fillna(False)
        )
    if data_id == "analyst_forecast":
        valid_rows &= (
            mapped["report_id"].astype("string").str.strip().ne("").fillna(False)
        )
    if data_id == "corporate_actions":
        valid_rows &= mapped["kind"].astype("string").str.strip().ne("").fillna(False)
    excluded = int((~valid_rows).sum())
    if excluded:
        counts["excluded_missing_required_rows"] = excluded
    mapped = mapped.loc[valid_rows].copy()
    result = _normalize_mapped_frame(mapped, data_id, frame.loc[valid_rows])
    # Derived arithmetic can overflow despite finite inputs. Keep only the
    # affected derived cell unavailable, without invalidating its neighbors.
    for column in result.select_dtypes(include="number"):
        invalid = result[column].notna() & ~np.isfinite(result[column])
        if invalid.any():
            counts[f"invalid_{column}_rows"] = counts.get(
                f"invalid_{column}_rows", 0
            ) + int(invalid.sum())
            result.loc[invalid, column] = np.nan
    if result.empty:
        for column in result.select_dtypes(include="object"):
            result[column] = result[column].astype(pd.ArrowDtype(pa.string()))
    previous = result.attrs.get("source_row_counts", {})
    counts.update(previous)
    counts["raw_rows"] = len(frame)
    if excluded:
        counts["excluded_missing_required_rows"] = (
            previous.get("excluded_missing_required_rows", 0) + excluded
        )
    counts["retained_before_date_filter_rows"] = len(result)
    result.attrs["source_row_counts"] = counts
    return result


def _normalize_mapped_frame(
    mapped: pd.DataFrame, data_id: str, frame: pd.DataFrame
) -> pd.DataFrame:
    if "stock_code" in mapped.columns:
        mapped["stock_code"] = normalize_provider_stock_code_series(
            mapped["stock_code"]
        )
    if "index_code" in mapped.columns:
        mapped["index_code"] = strict_string_series(
            mapped["index_code"],
            error="invalid_csmar_index_code",
        )

    if data_id == "corporate_actions":
        return _normalize_corporate_actions(mapped)
    if data_id == "trading_calendar":
        return _normalize_trading_calendar_frame(mapped)
    if data_id == "index_membership":
        result = mapped
        result["constituent_stock_code"] = strict_string_series(
            result["constituent_stock_code"], error="invalid_index_membership_code"
        )
        result["market_code"] = strict_string_series(
            result["market_code"],
            error="invalid_index_membership_market_code",
            allow_null=True,
        ).astype("string")
        result["security_type"] = strict_integer_series(
            result["security_type"],
            error="invalid_index_membership_security_type",
            allow_null=True,
        )
        result["constituent_asset_id"] = [
            canonical_json_hash(
                {
                    "code": code,
                    "market_code": None if pd.isna(market) else market,
                    "security_type": None if pd.isna(kind) else int(kind),
                }
            )
            for code, market, kind in result[
                ["constituent_stock_code", "market_code", "security_type"]
            ].itertuples(index=False, name=None)
        ]
        for column in ("effective_date", "announcement_date"):
            result[column] = strict_timestamp_series(
                result[column],
                error=f"invalid_index_membership_{column}",
                allow_null=column == "announcement_date",
            ).dt.normalize()
        result["timestamp"] = result["effective_date"]
        result["source_asset_id"] = result["index_code"]
        result.sort_values(
            ["source_asset_id", "constituent_asset_id", "timestamp"], inplace=True
        )
        return result.reset_index(drop=True)
    if data_id == "index_weights":
        result = mapped
        result["constituent_asset_id"] = normalize_provider_stock_code_series(
            result["constituent_stock_code"]
        )
        result["effective_date"] = strict_timestamp_series(
            result["effective_date"], error="invalid_index_weight_date"
        ).dt.normalize()
        result["weight"] = result["weight_pct"].astype("float64") / 100.0
        negative = result["weight"].lt(0.0)
        result.attrs["source_row_counts"] = {"invalid_weight_rows": int(negative.sum())}
        result.loc[negative, "weight"] = np.nan
        result["timestamp"] = result["effective_date"]
        result["source_asset_id"] = result["index_code"]
        result.sort_values(
            ["source_asset_id", "constituent_asset_id", "timestamp"], inplace=True
        )
        return result.reset_index(drop=True)
    if data_id in FINANCIAL_COLUMNS_BY_DATA_ID:
        result = mapped
        result["report_period"] = strict_timestamp_series(
            result["report_period"], error="invalid_financial_report_period"
        ).dt.normalize()
        result["correction_disclosure_date"] = _parse_financial_correction_date(
            result["correction_disclosure_date"]
        )
        result["is_corrected"] = _parse_correction_flag(result["is_corrected"])
        usable = result["is_corrected"].notna() & (
            result["is_corrected"].eq(result["correction_disclosure_date"].notna())
        )
        excluded = int((~usable).sum())
        result = result.loc[usable].copy()
        result["is_corrected"] = result["is_corrected"].astype(bool)
        result.attrs["source_row_counts"] = {"excluded_missing_required_rows": excluded}
        result["report_type"] = strict_string_series(
            result["report_type"], error="invalid_financial_report_type"
        )
        result["version_identity"] = "original"
        corrected = result["is_corrected"]
        result.loc[corrected, "version_identity"] = "correction@" + result.loc[
            corrected, "correction_disclosure_date"
        ].dt.strftime("%Y-%m-%d").astype("string")
        result["timestamp"] = result["report_period"]
        result["source_asset_id"] = result["stock_code"]
        result.sort_values(
            [
                "source_asset_id",
                "timestamp",
                "report_type",
                "version_identity",
            ],
            inplace=True,
        )
        return result.reset_index(drop=True)
    if data_id == MARKET_BARS_DATA_ID:
        result = mapped
        result["trade_date"] = strict_timestamp_series(
            result["trade_date"],
            error="invalid_csmar_trade_date",
        )
        result["timestamp"] = result["trade_date"]
        result["source_asset_id"] = result["stock_code"]
        result["period_return"] = result["return_with_dividend"]
        result["adj_close"] = result["comparable_close_with_dividend"]
        close = result["close"]
        # Keep the adjustment ratio bounded before restoring the price exponent.
        opening, opening_exponent = np.frexp(
            result["open"].to_numpy(dtype="float64", na_value=np.nan)
        )
        adjusted, adjusted_exponent = np.frexp(result["adj_close"])
        closing, closing_exponent = np.frexp(close.where(close.ne(0.0)))
        result["adj_open"] = np.ldexp(
            opening * (adjusted / closing),
            opening_exponent + adjusted_exponent - closing_exponent,
        )
        del opening, adjusted, closing
        del opening_exponent, adjusted_exponent, closing_exponent
        result["float_market_cap"] = result["float_market_cap_thousand"] * 1000.0
        result["total_market_cap"] = result["total_market_cap_thousand"] * 1000.0
        market_type = strict_integer_series(
            result["market_type"], error="invalid_csmar_market_type", allow_null=True
        )
        filling = strict_integer_series(
            result["filling_flag"], error="invalid_csmar_filling_flag", allow_null=True
        )
        filling = filling.where(filling.isin((0, 1, 2)))
        result["filling_flag"] = filling
        dates = result["trade_date"].dt.normalize()
        active_board = (market_type.eq(32) & dates.ge(pd.Timestamp("2019-07-22"))) | (
            market_type.eq(16) & dates.ge(pd.Timestamp("2020-08-24"))
        )
        not_applicable = (
            dates.le(pd.Timestamp("2025-12-31"))
            & market_type.isin((1, 2, 4, 8, 16, 32, 64))
            & (~active_board | filling.isin((1, 2)))
        )
        for column in ("after_hours_volume", "after_hours_amount"):
            values = result[column]
            result[column] = values.mask(
                values.isna()
                & not_applicable.fillna(False)
                & ~result.pop(f"_invalid_{column}"),
                0.0,
            )
        result["volume"] = result["volume"] + result["after_hours_volume"]
        result["amount"] = result["amount"] + result["after_hours_amount"]
        result["turnover"] = pd.Series(
            float("nan"), index=result.index, dtype="float64"
        )
        result = result.loc[:, list(MARKET_BARS_INTERIM_COLUMNS)]
        result = result.sort_values(["source_asset_id", "timestamp"], kind="mergesort")
        result.reset_index(drop=True, inplace=True)
        return result
    if data_id == "adjustment":
        result = mapped
        result["trade_date"] = strict_timestamp_series(
            result["trade_date"],
            error="invalid_csmar_adjustment_date",
        )
        result["timestamp"] = result["trade_date"]
        result["source_asset_id"] = result["stock_code"]
        result.sort_values(["source_asset_id", "timestamp"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
    if data_id == "market_cap":
        result = mapped
        result["share_change_date"] = strict_timestamp_series(
            result["share_change_date"],
            error="invalid_csmar_market_cap_date",
        )
        result["timestamp"] = result["share_change_date"]
        result["source_asset_id"] = result["stock_code"]
        result.sort_values(["source_asset_id", "timestamp"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
    if data_id == "stock_master":
        result = mapped
        for column in ("listing_date", "delisting_date", "status_change_date"):
            if column in result.columns:
                result[column] = _parse_optional_csmar_date(result[column])
        result["timestamp"] = result["status_change_date"].fillna(
            result["listing_date"].where(~result.pop("_invalid_status_change_date"))
        )
        result.attrs["source_row_counts"] = {
            "excluded_missing_timestamp_rows": int(result["timestamp"].isna().sum())
        }
        result = result.loc[result["timestamp"].notna()].copy()
        result["source_asset_id"] = result["stock_code"]
        result.sort_values(["source_asset_id", "timestamp"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
    if data_id == "index_info":
        result = mapped
        for column in INDEX_INFO_INTERIM_COLUMNS:
            if column not in result and column not in {
                "source_asset_id",
                "timestamp",
                "index_class",
            }:
                result[column] = pd.Series(
                    np.nan if column == "base_point" else pd.NA,
                    index=result.index,
                    dtype="float64" if column == "base_point" else "string",
                )
        if "index_class" not in result.columns and "index_category" in result.columns:
            result["index_class"] = result["index_category"]
        if "base_point" in result.columns:
            base_point_text = result["base_point"].astype("string").str.strip()
            base_point = result["base_point"].mask(base_point_text.eq(""))
            result["base_point"] = base_point.mask(base_point.eq(-9666))
        counts = {"raw_rows": len(frame)}
        for column in ("base_date", "start_date"):
            values = result[column].astype("string").str.strip()
            parsed = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
            # CSV dates outside the existing nanosecond domain stay unknown.
            parsed = (
                parsed.mask(
                    (parsed < pd.Timestamp.min.tz_localize("UTC"))
                    | (parsed > pd.Timestamp.max.tz_localize("UTC"))
                )
                .dt.tz_localize(None)
                .astype("datetime64[ns]")
            )
            nonempty = values.notna() & values.ne("")
            counts[f"invalid_{column}_rows"] = int((nonempty & parsed.isna()).sum())
            result[column] = parsed
        result["timestamp"] = result["start_date"].combine_first(result["base_date"])
        result["source_asset_id"] = result["index_code"]
        affected = pd.Series(False, index=frame.index)
        for column, allowed_values in zip(
            INDEX_INFO_SOURCE_CODE_COLUMNS, INDEX_INFO_SOURCE_CODE_ENUMS, strict=True
        ):
            unsupported = pd.Series(False, index=frame.index)
            if column in frame:
                values = frame[column].astype("string")
                unsupported = (
                    values.notna() & values.ne("") & ~values.isin(allowed_values)
                )
            counts[f"unsupported_{column}_rows"] = int(unsupported.sum())
            affected |= unsupported
        counts["unsupported_source_code_rows"] = int(affected.sum())
        missing_timestamp = result["timestamp"].isna()
        counts["excluded_missing_timestamp_rows"] = int(missing_timestamp.sum())
        result = result.loc[~missing_timestamp].copy()
        if result.empty:
            for column in result.select_dtypes(include="object"):
                result[column] = result[column].astype(pd.ArrowDtype(pa.string()))
        counts["retained_before_date_filter_rows"] = len(result)
        result.sort_values(["source_asset_id", "timestamp"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        result.attrs["source_row_counts"] = counts
        return result
    if data_id == "index_market":
        result = mapped
        result["trade_date"] = strict_timestamp_series(
            result["trade_date"], error="invalid_index_return_date"
        ).dt.normalize()
        result["period_return"] = result["return_1d_pct"].astype("float64") / 100.0
        result["timestamp"] = result["trade_date"]
        result["source_asset_id"] = result["index_code"]
        result.sort_values(["source_asset_id", "timestamp"], inplace=True)
        return result.reset_index(drop=True)
    if data_id == "industry_classification":
        result = mapped
        result["implement_date"] = strict_timestamp_series(
            result["implement_date"], error="invalid_industry_implement_date"
        ).dt.normalize()
        result["timestamp"] = result["implement_date"]
        result["source_asset_id"] = result["stock_code"]
        return (
            result.loc[:, INDUSTRY_CLASSIFICATION_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id == "industry_classification_annual":
        result = mapped
        result["effective_date"] = strict_timestamp_series(
            result["effective_date"], error="invalid_industry_annual_date"
        ).dt.normalize()
        result["timestamp"] = result["effective_date"]
        result["source_asset_id"] = result["stock_code"]
        return (
            result.loc[:, INDUSTRY_CLASSIFICATION_ANNUAL_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id == "industry_classification_sets":
        result = mapped
        for column in INDUSTRY_CLASSIFICATION_SETS_COLUMNS:
            result[column] = strict_string_series(
                result[column],
                error="invalid_industry_dictionary_value",
                allow_empty=column == "parent_industry_code",
                allow_null=column not in {"industry_code", "industry_standard_id"},
            ).astype("string")
        result = result.loc[:, INDUSTRY_CLASSIFICATION_SETS_COLUMNS]
        return result.reset_index(drop=True)
    if data_id == "industry_classification_versions":
        result = mapped
        result["implement_date"] = strict_timestamp_series(
            result["implement_date"],
            error="invalid_industry_version_date",
            allow_null=True,
        ).dt.normalize()
        result["abandon_date"] = _parse_optional_csmar_date(result["abandon_date"])
        result["timestamp"] = result["implement_date"]
        result = result.loc[result["timestamp"].notna()].copy()
        return (
            result.loc[:, INDUSTRY_CLASSIFICATION_VERSIONS_COLUMNS]
            .sort_values(["timestamp", "industry_standard_id"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id in {
        "listing_status_changes",
        "stock_info_changes",
        "stock_info_annual",
        "trading_no_limit",
        "special_treatment_changes",
        "equity_changes",
    }:
        return _normalize_state_change_frame(mapped, data_id)
    if data_id == "analyst_forecast":
        result = mapped
        result["source_asset_id"] = normalize_provider_stock_code_series(
            result["stock_code"]
        ).astype("string")
        result["declare_date"] = _parse_optional_csmar_date(
            result["declare_date"]
        ).dt.normalize()
        result["forecast_end_date"] = strict_timestamp_series(
            result["forecast_end_date"], error="invalid_analyst_forecast_end_date"
        ).dt.normalize()
        result["report_id"] = strict_string_series(
            result["report_id"], error="invalid_analyst_report_id"
        ).astype("string")
        for column in ("institution_id", "analyst_ids"):
            values = result[column]
            blank = values.map(
                lambda value: isinstance(value, str) and not value.strip()
            )
            result[column] = strict_string_series(
                values.mask(blank),
                error=f"invalid_analyst_{column}",
                allow_null=True,
                strip=column != "analyst_ids",
            ).astype("string")
        result["timestamp"] = result["declare_date"]
        missing_date = result["declare_date"].isna()
        missing_institution = result["institution_id"].isna()
        excluded = missing_date
        counts = {
            "raw_rows": len(result),
            "missing_declare_date_rows": int(missing_date.sum()),
            "missing_institution_id_rows": int(missing_institution.sum()),
            "excluded_missing_required_rows": int(excluded.sum()),
            "retained_before_date_filter_rows": int((~excluded).sum()),
        }
        output = (
            result.loc[~excluded, ANALYST_FORECAST_COLUMNS]
            .sort_values(
                ["source_asset_id", "timestamp", "report_id", "forecast_end_date"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        output.attrs["source_row_counts"] = counts
        return output
    raise ValueError(f"unsupported_csmar_data_id: {data_id}")


def _normalize_corporate_actions(mapped: pd.DataFrame) -> pd.DataFrame:
    """Preserve distribution facts without inferring holder entitlements."""
    result = mapped.copy()
    result["source_asset_id"] = result["stock_code"].astype("string")
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="invalid_corporate_action_ex_date"
    ).dt.normalize()
    result["kind"] = strict_string_series(
        result["kind"], error="invalid_corporate_action_kind"
    ).astype("string")
    counts = {
        "raw_rows": len(result),
        "retained_before_date_filter_rows": len(result),
    }
    counts["unsupported_kind_rows"] = int(
        (~result["kind"].isin({"CA", "SD", "RO", "DS", "GQ", "SN"})).sum()
    )
    # Optional facts can be unknown without erasing an otherwise identified event.
    for column in ("announcement_date", "settlement_date"):
        values = result[column].replace(r"^\s*$", pd.NA, regex=True)
        dates = strict_timestamp_series(
            values,
            error="invalid_corporate_action_date",
            allow_null=True,
            errors="coerce",
        )
        counts[f"invalid_{column}_rows"] = int((values.notna() & dates.isna()).sum())
        result[column] = dates.astype("datetime64[ns]").dt.normalize()
        if column == "settlement_date":
            result["settlement_date_unresolved"] = values.notna() & dates.isna()
    for column in ("amount", "subscription_price", "market_type"):
        values = result[column].replace(r"^\s*$", pd.NA, regex=True)
        numeric = strict_real_numeric_series(
            values, error="invalid_source_numeric", errors="coerce"
        )
        valid = np.isfinite(numeric)
        if column == "market_type":
            valid &= (
                numeric.eq(np.floor(numeric)) & numeric.ge(-(2**63)) & numeric.lt(2**63)
            )
        counts[f"invalid_{column}_rows"] = int((values.notna() & ~valid).sum())
        result[column] = numeric.where(valid)
        if column == "market_type":
            result[column] = result[column].astype("Int64")
    result = (
        result.loc[:, CORPORATE_ACTION_COLUMNS]
        .sort_values(["source_asset_id", "timestamp", "kind"], kind="mergesort")
        .reset_index(drop=True)
    )
    result.attrs["source_row_counts"] = counts
    return result


def _normalize_state_change_frame(frame: pd.DataFrame, data_id: str) -> pd.DataFrame:
    result = frame
    result["source_asset_id"] = result["stock_code"]
    if data_id == "listing_status_changes":
        result["announcement_date"] = _parse_optional_csmar_date(
            result["announcement_date"]
        )
        result["effective_date"] = strict_timestamp_series(
            result["effective_date"], error="invalid_listing_status_date"
        ).dt.normalize()
        result["timestamp"] = result["effective_date"]
        return (
            result.loc[:, LISTING_STATUS_CHANGE_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id == "stock_info_changes":
        result["announcement_date"] = _parse_optional_csmar_date(
            result["announcement_date"]
        )
        result["effective_date"] = strict_timestamp_series(
            result["effective_date"], error="invalid_stock_info_change_date"
        ).dt.normalize()
        result["timestamp"] = result["effective_date"]
        return (
            result.loc[:, STOCK_INFO_CHANGE_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id == "stock_info_annual":
        for column in ("effective_date", "listing_date"):
            result[column] = strict_timestamp_series(
                result[column], error="invalid_stock_info_annual_date", allow_null=True
            ).dt.normalize()
        result["timestamp"] = result["effective_date"]
        return (
            result.loc[:, STOCK_INFO_ANNUAL_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id == "trading_no_limit":
        result["trade_date"] = strict_timestamp_series(
            result["trade_date"], error="invalid_trading_no_limit_date"
        ).dt.normalize()
        result["timestamp"] = result["trade_date"]
        result["reason"] = (
            result["reason"].astype("string").str.strip().replace("", pd.NA)
        )
        return (
            result.loc[:, TRADING_NO_LIMIT_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id == "special_treatment_changes":
        result["announcement_date"] = strict_timestamp_series(
            result["announcement_date"],
            error="invalid_special_treatment_date",
            allow_null=True,
        ).dt.normalize()
        result["effective_date"] = strict_timestamp_series(
            result["effective_date"], error="invalid_special_treatment_date"
        ).dt.normalize()
        result["timestamp"] = result["effective_date"]
        result["change_type"] = strict_string_series(
            result["change_type"], error="invalid_special_treatment_change_type"
        )
        supported = result["change_type"].str.fullmatch(r"[ABCDSTX]{2}")
        result.attrs["source_row_counts"] = {
            "unsupported_transition_rows": int((~supported).sum())
        }
        destination = result["change_type"].str[1]
        known = supported & destination.isin(("A", "B", "C", "D"))
        result["is_st"] = destination.isin(("B", "D")).astype("boolean").where(known)
        result["is_pt"] = destination.eq("C").astype("boolean").where(known)
        return (
            result.loc[:, SPECIAL_TREATMENT_CHANGE_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    result["report_period"] = strict_timestamp_series(
        result["report_period"], error="invalid_equity_change_period"
    ).dt.normalize()
    result["timestamp"] = result["report_period"]
    result["report_type"] = strict_string_series(
        result["report_type"], error="invalid_equity_change_report_type"
    )
    for column in ("data_source", "subject_code"):
        result[column] = strict_string_series(
            result[column], error=f"invalid_equity_change_{column}"
        )
    return (
        result.loc[:, EQUITY_CHANGE_COLUMNS]
        .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )


def _parse_optional_csmar_date(series: pd.Series) -> pd.Series:
    return strict_timestamp_series(
        series, error="invalid_source_date", allow_null=True, errors="coerce"
    )


def _parse_financial_correction_date(series: pd.Series) -> pd.Series:
    """Normalize the provider's comma-separated correction disclosure dates."""

    latest: list[object] = []
    for value in series:
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            latest.append(pd.NaT)
            continue
        tokens = [token.strip() for token in str(value).split(",")]
        parsed = _parse_optional_csmar_date(pd.Series(tokens))
        if (
            parsed.isna().any()
            or bool(parsed.duplicated().any())
            or not (parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing)
        ):
            latest.append(pd.NaT)
            continue
        latest.append(parsed.max())
    return pd.Series(latest, index=series.index, dtype="datetime64[ns]")


def _parse_correction_flag(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin((0, 1))).astype("boolean")


def materialize_interim_data(
    *,
    raw_root: Path,
    output_roots: Mapping[str, Path],
    data_ids: tuple[str, ...] = DEFAULT_INTERIM_DATA_IDS,
    start_date: str | None = None,
    end_date: str | None = None,
    chunksize: int = 500000,
    metadata: Mapping[str, object] | None = None,
) -> InterimImportResult:
    if type(chunksize) is not int or chunksize <= 0:
        raise ValueError("invalid_csmar_chunksize")
    source_raw_root = Path(raw_root)
    written: dict[str, Path] = {}
    skipped: set[str] = set()
    pending = set(data_ids)
    require_unaliased_path(
        source_raw_root,
        error="csmar_raw_export_identity_changed",
    )
    indexed_files = dataset_files_with_identity_by_id(
        source_raw_root, requested_datasets=data_ids
    )

    with closing_iterator(indexed_files):
        for data_id, group in groupby(indexed_files, key=lambda item: item[0]):
            input_files = ((item[1], item[2]) for item in group)
            source_evidence = {"source_id": str(SOURCE_ID)}
            pending.discard(data_id)
            spec = interim_catalog().get(data_id)
            artifact_root = Path(output_roots[spec.data_id])
            target_metadata = dict(metadata or {})
            diagnostics = dict(target_metadata.get("producer_diagnostics", {}))
            diagnostics["exact_duplicate_count"] = 0
            target_metadata["producer_diagnostics"] = diagnostics
            source_row_counts = {}
            diagnostics["source_row_counts"] = source_row_counts
            frames = _iter_normalized_interim_frames(
                data_id=data_id,
                input_files=input_files,
                start_date=start_date,
                end_date=end_date,
                chunksize=chunksize,
                source_row_counts=source_row_counts,
                producer_diagnostics=diagnostics,
                source_evidence=source_evidence,
            )
            with closing_iterator(frames):
                try:
                    write_dataset(
                        artifact_root,
                        frames,
                        spec=spec,
                        metadata={
                            **target_metadata,
                            "source_evidence": source_evidence,
                        },
                    )
                except ValueError as exc:
                    if str(exc) != "empty_dataset_schema_missing":
                        raise
                    skipped.add(data_id)
                    continue
            written[data_id] = artifact_root
    skipped.update(pending)

    return InterimImportResult(
        artifact_roots=written,
        skipped=tuple(data_id for data_id in data_ids if data_id in skipped),
    )


def _iter_normalized_interim_frames(
    *,
    data_id: str,
    input_files: Iterable[tuple[Path, DatasetFileIdentity]],
    start_date: str | None,
    end_date: str | None,
    chunksize: int,
    producer_diagnostics: dict[str, object] | None,
    source_row_counts: dict[str, int] | None = None,
    source_evidence: dict[str, object] | None = None,
) -> Iterator[object]:
    primary_key = interim_catalog().get(data_id).primary_key
    duplicate_index = ExactPrimaryKeyIndex(primary_key=primary_key)
    if source_row_counts is None:
        source_row_counts = {}
    exact_duplicate_count = 0
    content_hashes = []
    empty_index_frame = None
    emitted_rows = 0
    # Event/reference facts must be reconciled before emission: a later file or
    # chunk can make one earlier cell unknown without erasing its event key.
    coalesce = data_id in {
        "index_info",
        "index_weights",
        "trading_no_limit",
        "index_market",
        "stock_master",
        "corporate_actions",
        "financial_balance",
        "financial_income",
        "financial_cashflow",
        "industry_classification",
        "industry_classification_annual",
        "industry_classification_sets",
        "industry_classification_versions",
        "market_cap",
        "equity_changes",
        "analyst_forecast",
    }
    pending = []
    for input_file, expected_identity in input_files:
        parent = require_unaliased_path(
            input_file.parent, error="csmar_interim_source_parent_invalid"
        )
        try:
            parent_stat = filesystem_stat(parent, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("csmar_interim_source_parent_invalid") from exc
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("csmar_interim_source_parent_invalid")
        completed_before = len(content_hashes)
        chunks = iter_private_export_chunks(
            input_file,
            dataset=data_id,
            chunksize=chunksize,
            expected_identity=expected_identity,
            content_hashes=content_hashes,
        )
        with closing_iterator(chunks):
            for chunk in chunks:
                pruned, counts = _transform_interim_chunk(
                    chunk,
                    data_id=data_id,
                )
                if data_id in {MARKET_BARS_DATA_ID, "adjustment", "market_cap"}:
                    mask = pd.Series(True, index=pruned.index)
                    if start_date is not None and data_id != "market_cap":
                        mask &= pruned["timestamp"].ge(pd.Timestamp(start_date))
                    if end_date is not None:
                        mask &= pruned["timestamp"].le(pd.Timestamp(end_date))
                    if counts is not None:
                        counts["excluded_date_range_rows"] = int((~mask).sum())
                        counts["retained_after_date_filter_rows"] = int(mask.sum())
                    pruned = pruned.loc[mask].copy()
                if coalesce:
                    pending.append(pruned)
                    if counts is not None:
                        for key, count in counts.items():
                            source_row_counts[key] = (
                                source_row_counts.get(key, 0) + count
                            )
                    continue
                skip_duplicate = False
                if not pruned.empty:
                    table = pa.Table.from_pandas(pruned, preserve_index=False)
                    try:
                        positions, duplicate_count = (
                            duplicate_index.classify_comparison_table(
                                table,
                                namespace=data_id,
                                value_columns=tuple(
                                    sorted(
                                        column
                                        for column in pruned
                                        if column not in primary_key
                                    )
                                ),
                            )
                        )
                    except ValueError as error:
                        if str(error) == "conflicting_exact_primary_key_payload":
                            raise ValueError(
                                f"conflicting_csmar_primary_key_duplicates: {data_id}"
                            ) from error
                        raise
                    exact_duplicate_count += duplicate_count
                    if producer_diagnostics is not None:
                        producer_diagnostics["exact_duplicate_count"] = (
                            exact_duplicate_count
                        )
                    if not positions:
                        skip_duplicate = True
                    elif len(positions) != len(pruned):
                        pruned = pruned.iloc[positions].reset_index(drop=True)
                    del positions
                if counts is not None:
                    for key, count in counts.items():
                        source_row_counts[key] = source_row_counts.get(key, 0) + count
                if pruned.empty:
                    empty_index_frame = pruned
                    continue
                if skip_duplicate:
                    continue
                emitted_rows += len(pruned)
                try:
                    yield pruned
                except GeneratorExit:
                    return
        if len(content_hashes) != completed_before + 1:
            raise ValueError("csmar_raw_export_incomplete_read")
    if pending:
        combined = pd.concat(pending, ignore_index=True)
        pending.clear()
        exact_duplicate = combined.duplicated()
        exact_duplicates = int(exact_duplicate.sum())
        if exact_duplicates:
            combined = combined.loc[~exact_duplicate].copy()
        del exact_duplicate
        duplicate = combined.duplicated(list(primary_key), keep=False)
        if duplicate.any():
            if data_id == "stock_master":
                combined["delisting_date_unresolved"] = combined.groupby(
                    list(primary_key), dropna=False
                )["delisting_date_unresolved"].transform("any")
            if data_id == "industry_classification_versions":
                combined["abandon_date_unresolved"] = combined.groupby(
                    list(primary_key), dropna=False
                )["abandon_date_unresolved"].transform("any")
            if data_id == "corporate_actions":
                combined["settlement_date_unresolved"] = combined.groupby(
                    list(primary_key), dropna=False
                )["settlement_date_unresolved"].transform("any")
            for column in combined.columns.difference(primary_key):
                distinct = (
                    combined.loc[duplicate]
                    .groupby(list(primary_key), dropna=False, sort=False)[column]
                    .transform("nunique", dropna=False)
                )
                conflict = distinct.gt(1).reindex(combined.index, fill_value=False)
                if conflict.any():
                    source_row_counts[f"conflicting_{column}_rows"] = int(
                        conflict.sum()
                    )
                    if (
                        data_id == "industry_classification_versions"
                        and column == "abandon_date"
                    ):
                        earliest = combined.groupby(list(primary_key), dropna=False)[
                            column
                        ].transform("min")
                        combined.loc[conflict, column] = earliest.loc[conflict].where(
                            ~combined.loc[conflict, "abandon_date_unresolved"]
                        )
                        combined.loc[conflict, "abandon_date_unresolved"] = True
                        continue
                    if data_id == "stock_master" and column == "delisting_date":
                        combined.loc[conflict, "delisting_date_unresolved"] = True
                    if data_id == "corporate_actions" and column == "settlement_date":
                        combined.loc[conflict, "settlement_date_unresolved"] = True
                    if combined[column].dtype == bool:
                        combined[column] = combined[column].astype("boolean")
                    combined.loc[conflict, column] = pd.NA
            # All retained cells agree now; no source observation is preferred.
            combined = combined.drop_duplicates(list(primary_key))
        if producer_diagnostics is not None:
            producer_diagnostics["exact_duplicate_count"] = exact_duplicates
        for start in range(0, max(1, len(combined)), chunksize):
            yield combined.iloc[start : start + chunksize].reset_index(drop=True)
    if empty_index_frame is not None and emitted_rows == 0:
        yield empty_index_frame
    if source_evidence is not None:
        digest = hashlib.sha256()
        for member_hash in sorted(content_hashes):
            for value in (data_id, member_hash):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        source_evidence["content_hash"] = digest.hexdigest()


def _transform_interim_chunk(
    chunk: pd.DataFrame,
    *,
    data_id: str,
) -> tuple[pd.DataFrame, dict[str, int] | None]:
    normalized = normalize_raw_frame(chunk, data_id)
    source_row_counts = normalized.attrs.pop("source_row_counts", None)
    if source_row_counts is not None:
        source_row_counts["retained_after_date_filter_rows"] = len(normalized)
        source_row_counts["excluded_date_range_rows"] = source_row_counts[
            "retained_before_date_filter_rows"
        ] - len(normalized)
    return normalized.loc[
        :, list(INTERIM_COLUMNS_BY_DATA_ID[data_id])
    ].copy(), source_row_counts


def _interim_spec(data_id: str, columns: tuple[str, ...]) -> InterimDataSpec:
    grain = DataGrain.ASSET_REFERENCE_TIMESTAMP
    partition_columns = ("year",)
    if data_id == "analyst_forecast":
        primary_key = ("source_asset_id", "timestamp", "report_id", "forecast_end_date")
    elif data_id == "industry_classification":
        primary_key = ("source_asset_id", "timestamp", "industry_standard_id")
    elif data_id == "industry_classification_sets":
        primary_key = ("industry_standard_id", "industry_code")
        grain = DataGrain.ASSET_REFERENCE
        partition_columns = ()
    elif data_id == "industry_classification_versions":
        primary_key = ("industry_standard_id", "timestamp")
    elif data_id in {"listing_status_changes", "stock_info_changes"}:
        primary_key = ("source_asset_id", "timestamp", "changed_item")
    elif data_id == "special_treatment_changes":
        primary_key = ("source_asset_id", "timestamp", "change_type")
    elif data_id == "corporate_actions":
        primary_key = ("source_asset_id", "timestamp", "kind")
    elif data_id == "equity_changes":
        primary_key = (
            "source_asset_id",
            "timestamp",
            "report_type",
            "data_source",
            "subject_code",
        )
    elif data_id in {"index_membership", "index_weights"}:
        primary_key = ("source_asset_id", "constituent_asset_id", "timestamp")
    elif data_id in FINANCIAL_COLUMNS_BY_DATA_ID:
        primary_key = (
            "source_asset_id",
            "timestamp",
            "report_type",
            "version_identity",
        )
    elif data_id == "trading_calendar":
        primary_key = ("timestamp", "source_market_type")
        grain = DataGrain.SOURCE_MARKET_CALENDAR_DATE
    else:
        primary_key = ("source_asset_id", "timestamp")
        if data_id not in {"index_info", "stock_master"}:
            grain = DataGrain.ASSET_TIMESTAMP
    return InterimDataSpec(
        data_id=data_id,
        source_id=SourceId(SOURCE_ID),
        frequency=DataFrequency.DAILY,
        grain=grain,
        column_roles=_column_roles(columns, primary_key=primary_key),
        primary_key=primary_key,
        partition_columns=partition_columns,
    )


def _column_roles(
    columns: tuple[str, ...],
    *,
    primary_key: tuple[str, ...],
) -> dict[str, ColumnRole]:
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column == "timestamp":
            roles[column] = ColumnRole.TIMESTAMP
        elif column in primary_key:
            roles[column] = ColumnRole.KEY
        else:
            roles[column] = field_role(column)
    return roles


def _normalize_trading_calendar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    result["calendar_date"] = strict_timestamp_series(
        result["calendar_date"],
        error="invalid_trading_calendar_date",
    )
    result["source_market_type"] = strict_integer_series(
        result["source_market_type"],
        error="invalid_trading_calendar_market_type",
    )
    weekday = result["source_day_of_week"]
    result["source_day_of_week"] = weekday.where(weekday.isin(range(1, 8))).astype(
        "Int64"
    )
    result["source_state"] = result["source_state"].astype("string").str.strip()
    result["source_state"] = result["source_state"].str.upper()
    if not result["source_state"].isin(("O", "C")).all():
        raise ValueError("invalid_trading_calendar_state")
    result["is_trading_day"] = result["source_state"].eq("O").astype("bool")
    result["timestamp"] = result["calendar_date"]
    return (
        result[list(TRADING_CALENDAR_INTERIM_COLUMNS)]
        .sort_values(["calendar_date", "source_market_type"])
        .reset_index(drop=True)
    )
