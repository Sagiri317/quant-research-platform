"""CSMAR raw export to source-normalized interim datasets."""

from __future__ import annotations

import stat
import sys
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain, groupby, islice
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.business_manifest import (
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
    strict_string_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.field_mappings import (
    map_columns,
    normalize_provider_stock_code_series,
)
from quant_research_platform.sources.csmar.fields import field_role
from quant_research_platform.sources.csmar.ids import SOURCE_ID, market_bars_data_id
from quant_research_platform.sources.csmar.raw_exports import (
    _verify_csmar_member_set,
    dataset_files_with_identity_by_id,
    iter_private_export_chunks_with_policy,
)
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    content_hash_for_files,
    require_artifact_root,
    require_unaliased_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_bounded_qrp_json


class _InterimCapacityExceeded(DatasetReadCapacityExceeded):
    def __init__(self, required_bytes: int, budget_bytes: int) -> None:
        super().__init__("csmar_interim_working_set_exceeds_task_memory_budget")
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes


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
    "analyst_forecast",
)
TIME_RANGE_FILTER_DATA_IDS = {MARKET_BARS_DATA_ID}
END_ONLY_FILTER_DATA_IDS = {"market_cap"}
START_END_FILTER_DATA_IDS = {"adjustment"}
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
EQUITY_CHANGE_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "report_period",
    "report_type",
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
    "analyst_forecast": ANALYST_FORECAST_COLUMNS,
}
AUTO_DROPPED_COLUMNS: dict[str, dict[str, str]] = {
    MARKET_BARS_DATA_ID: {
        "return_with_dividend": "derived as period_return",
        "comparable_close_with_dividend": "derived as adj_close",
        "float_market_cap_thousand": "converted to float_market_cap",
        "total_market_cap_thousand": "converted to total_market_cap",
        "after_hours_volume": "included in canonical volume",
        "after_hours_amount": "included in canonical amount",
    }
}
OPTIONAL_COLUMNS: dict[str, dict[str, str]] = {
    MARKET_BARS_DATA_ID: {
        "return_without_dividend": "not retained in the source interim contract",
        "comparable_close_without_dividend": (
            "not retained in the source interim contract"
        ),
        "change_ratio": "period_return and price fields are used instead",
        "limit_down": "limit_status is used instead",
        "limit_up": "limit_status is used instead",
    },
    "stock_master": {
        "market_type": "not retained in source reference output",
        "status_change_date": "retained when present for reference timing",
    },
    "market_cap": {
        "b_float_shares": "not retained in source market-cap output",
        "h_float_shares": "not retained in source market-cap output",
    },
}


@dataclass(frozen=True)
class InterimImportResult:
    """Summary of a CSMAR interim materialization run."""

    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]
    pruning_report_text: str


def default_override_payload() -> dict[str, dict[str, object]]:
    data_ids = sorted(set(INTERIM_COLUMNS_BY_DATA_ID) | set(OPTIONAL_COLUMNS))
    return {
        data_id: {
            "keep": [],
            "drop": [],
            "_available_optional_columns": OPTIONAL_COLUMNS.get(data_id, {}),
        }
        for data_id in data_ids
    }


def load_column_overrides(path: Path) -> Mapping[str, Mapping[str, object]]:
    candidate = require_unaliased_path(
        Path(path), error="csmar_column_overrides_identity_changed"
    )
    if not candidate.exists():
        return _validated_column_overrides(default_override_payload())
    try:
        _raw, payload = read_bounded_qrp_json(candidate)
    except ValueError as exc:
        if str(exc) == "dataset_metadata_file_too_large":
            raise ValueError("csmar_column_overrides_exceed_max_bytes") from exc
        raise
    return _validated_column_overrides(payload)


def _validated_column_overrides(
    payload: Mapping[str, object],
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(payload, Mapping):
        raise ValueError("invalid_csmar_column_override")
    payload_snapshot = _bounded_mapping_snapshot(
        payload,
        max_items=len(INTERIM_COLUMNS_BY_DATA_ID),
    )
    unknown = set(payload_snapshot).difference(INTERIM_COLUMNS_BY_DATA_ID)
    if unknown:
        raise ValueError("unknown_csmar_column_override_data_id")
    result: dict[str, dict[str, object]] = {}
    for data_id, raw_override in payload_snapshot.items():
        if not isinstance(raw_override, Mapping):
            raise ValueError("invalid_csmar_column_override")
        override = _bounded_mapping_snapshot(raw_override, max_items=3)
        if set(override).difference({"keep", "drop", "_available_optional_columns"}):
            raise ValueError("invalid_csmar_column_override")
        allowed = set(INTERIM_COLUMNS_BY_DATA_ID[data_id])
        allowed.update(OPTIONAL_COLUMNS.get(data_id, {}))
        allowed.update(AUTO_DROPPED_COLUMNS.get(data_id, {}))
        normalized: dict[str, object] = {}
        for key in ("keep", "drop"):
            values = override.get(key, [])
            if type(values) not in (list, tuple):
                raise ValueError("invalid_csmar_column_override")
            if len(values) > len(allowed):
                raise ValueError("invalid_csmar_column_override")
            value_snapshot = tuple(values)
            if any(type(value) is not str for value in value_snapshot):
                raise ValueError("invalid_csmar_column_override")
            if len(value_snapshot) != len(set(value_snapshot)) or set(
                value_snapshot
            ).difference(allowed):
                raise ValueError("invalid_csmar_column_override")
            normalized[key] = value_snapshot
        if set(normalized["keep"]).intersection(normalized["drop"]):
            raise ValueError("conflicting_csmar_column_override")
        if "_available_optional_columns" in override:
            available_optional = override["_available_optional_columns"]
            expected_optional = OPTIONAL_COLUMNS.get(data_id, {})
            if (
                not isinstance(available_optional, Mapping)
                or _bounded_mapping_snapshot(
                    available_optional,
                    max_items=len(expected_optional),
                )
                != expected_optional
            ):
                raise ValueError("invalid_csmar_column_override")
        normalized["_available_optional_columns"] = MappingProxyType(
            dict(OPTIONAL_COLUMNS.get(data_id, {}))
        )
        result[data_id] = MappingProxyType(normalized)
    return MappingProxyType(result)


def _bounded_mapping_snapshot(
    mapping: Mapping[str, object],
    *,
    max_items: int,
) -> dict[str, object]:
    items = tuple(islice(mapping.items(), max_items + 1))
    if len(items) > max_items:
        raise ValueError("invalid_csmar_column_override")
    snapshot: dict[str, object] = {}
    for key, value in items:
        if type(key) is not str or key in snapshot:
            raise ValueError("invalid_csmar_column_override")
        snapshot[key] = value
    return snapshot


def _apply_output_column_rules_validated(
    frame: pd.DataFrame,
    data_id: str,
    *,
    overrides: Mapping[str, Mapping[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_keep = list(INTERIM_COLUMNS_BY_DATA_ID.get(data_id, tuple(frame.columns)))
    override = overrides.get(data_id, {})
    manual_keep = list(override.get("keep", []))
    manual_drop = list(override.get("drop", []))

    keep_order = list(dict.fromkeys([*base_keep, *manual_keep]))
    available_keep = [column for column in keep_order if column in frame.columns]
    final_columns = [column for column in available_keep if column not in manual_drop]

    dropped = [column for column in frame.columns if column not in final_columns]
    auto_dropped = {
        column: reason
        for column, reason in AUTO_DROPPED_COLUMNS.get(data_id, {}).items()
        if column in dropped
    }
    optional_available = {
        column: reason
        for column, reason in OPTIONAL_COLUMNS.get(data_id, {}).items()
        if column in frame.columns
    }
    report = {
        "data_id": data_id,
        "input_columns": list(frame.columns),
        "kept_columns": final_columns,
        "dropped_columns": dropped,
        "auto_dropped_columns": auto_dropped,
        "optional_available_columns": optional_available,
        "manual_keep": manual_keep,
        "manual_drop": manual_drop,
    }
    return frame[final_columns].copy(), report


def render_column_pruning_report(reports: list[dict[str, object]]) -> str:
    lines = [
        "# CSMAR interim column pruning report",
        "",
        "This private audit report summarizes aggregate column pruning metadata.",
        "",
    ]
    for report in reports:
        if report.get("report_type") == "source_duplicate_summary":
            lines.extend(
                [
                    f"## {report['data_id']} source duplicate summary",
                    "",
                    "- exact duplicate primary-key rows deduped: "
                    f"{report['exact_duplicate_count']}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"## {report['data_id']}",
                "",
                f"- input columns: {len(report['input_columns'])}",
                f"- kept columns: {len(report['kept_columns'])}",
                f"- dropped columns: {len(report['dropped_columns'])}",
                f"- manual keep: {', '.join(report['manual_keep']) or 'none'}",
                f"- manual drop: {', '.join(report['manual_drop']) or 'none'}",
                "",
                "### Kept columns",
                "",
                ", ".join(report["kept_columns"]),
                "",
            ]
        )
        auto_dropped = report["auto_dropped_columns"]
        if auto_dropped:
            lines.extend(["### Auto-dropped columns", ""])
            for column, reason in auto_dropped.items():
                lines.append(f"- `{column}`: {reason}")
            lines.append("")
        optional = report["optional_available_columns"]
        if optional:
            lines.extend(["### Available optional columns", ""])
            for column, reason in optional.items():
                lines.append(f"- `{column}`: {reason}")
            lines.append("")
    return "\n".join(lines) + "\n"


def interim_specs() -> tuple[InterimDataSpec, ...]:
    return tuple(
        _interim_spec(data_id, columns)
        for data_id, columns in sorted(INTERIM_COLUMNS_BY_DATA_ID.items())
    )


def interim_catalog() -> InterimDataCatalog:
    return InterimDataCatalog(interim_specs())


def source_content_identity(raw_root: Path, data_id: str) -> str | None:
    """Return the provider-validated raw content identity for one target."""

    root = require_unaliased_path(
        Path(raw_root),
        error="csmar_raw_export_identity_changed",
    )
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    if not root.exists():
        return None
    index = ExactPrimaryKeyIndex(
        primary_key=("dataset_path",),
    )
    with (
        policy.lease(
            "source.csmar.raw.identity.index",
            index.retained_memory_bytes,
        ),
        index,
    ):
        records_iterator = dataset_files_with_identity_by_id(
            root,
            resource_policy=policy,
            identity_index=index,
            requested_datasets=(data_id,),
        )
        with closing_iterator(records_iterator):
            records = tuple(records_iterator)
        _verify_csmar_member_set(
            root,
            index,
            requested_datasets=(data_id,),
        )
        selected = [
            (dataset, path, identity)
            for dataset, path, identity in records
            if dataset == data_id
        ]
        if not selected:
            return None
        identity = content_hash_for_files(
            selected,
            error="csmar_raw_export_identity_changed",
            resource_policy=policy,
        )
        _verify_csmar_member_set(
            root,
            index,
            requested_datasets=(data_id,),
        )
        return identity


def normalize_raw_frame(
    frame: pd.DataFrame,
    data_id: str,
) -> pd.DataFrame:
    mapped = map_columns(frame, data_id)
    if "stock_code" in mapped.columns:
        mapped["stock_code"] = normalize_provider_stock_code_series(
            mapped["stock_code"]
        )
    if "index_code" in mapped.columns:
        mapped["index_code"] = strict_string_series(
            mapped["index_code"],
            error="invalid_csmar_index_code",
        )

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
                result[column], error=f"invalid_index_membership_{column}"
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
        result["weight"] = (
            strict_finite_numeric_series(
                result["weight_pct"], error="invalid_index_weight"
            )
            / 100.0
        )
        if bool((result["weight"] < 0.0).any()):
            raise ValueError("invalid_index_weight")
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
        if bool(
            (result["is_corrected"] & result["correction_disclosure_date"].isna()).any()
        ):
            raise ValueError("financial_correction_disclosure_missing")
        if bool(
            (
                ~result["is_corrected"] & result["correction_disclosure_date"].notna()
            ).any()
        ):
            raise ValueError("invalid_financial_correction_date")
        result["report_type"] = strict_string_series(
            result["report_type"], error="invalid_financial_report_type"
        )
        result["version_identity"] = "original"
        corrected = result["is_corrected"]
        result.loc[corrected, "version_identity"] = "correction@" + result.loc[
            corrected, "correction_disclosure_date"
        ].dt.strftime("%Y-%m-%d").astype("string")
        for column in set(FINANCIAL_COLUMNS_BY_DATA_ID[data_id]).difference(
            FINANCIAL_BASE_COLUMNS
        ):
            result[column] = strict_finite_numeric_series(
                result[column],
                error=f"invalid_financial_value: {column}",
                allow_null=True,
            )
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
        result["limit_status"] = strict_finite_numeric_series(
            result["limit_status"], error="invalid_csmar_limit_status", allow_null=True
        )
        result["trade_date"] = strict_timestamp_series(
            result["trade_date"],
            error="invalid_csmar_trade_date",
        )
        result["timestamp"] = result["trade_date"]
        result["source_asset_id"] = result["stock_code"]
        result["period_return"] = strict_finite_numeric_series(
            result["return_with_dividend"],
            error="invalid_csmar_period_return",
            allow_null=True,
        )
        result["adj_close"] = strict_finite_numeric_series(
            result["comparable_close_with_dividend"],
            error="invalid_csmar_adj_close",
            allow_null=True,
        )
        close = strict_finite_numeric_series(
            result["close"], error="invalid_csmar_close", allow_null=True
        )
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
            result["market_type"], error="invalid_csmar_market_type"
        )
        filling = strict_integer_series(
            result["filling_flag"], error="invalid_csmar_filling_flag"
        )
        if not filling.isin((0, 1, 2)).all():
            raise ValueError("invalid_csmar_filling_flag")
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
            values = strict_finite_numeric_series(
                result[column], error=f"invalid_csmar_{column}", allow_null=True
            )
            if (values.isna() & ~not_applicable).any():
                raise ValueError(f"invalid_csmar_{column}_missing")
            result[column] = values.fillna(0.0)
        result["volume"] = (
            strict_finite_numeric_series(
                result["volume"], error="invalid_csmar_volume", allow_null=True
            )
            + result["after_hours_volume"]
        )
        result["amount"] = (
            strict_finite_numeric_series(
                result["amount"], error="invalid_csmar_amount", allow_null=True
            )
            + result["after_hours_amount"]
        )
        result["turnover"] = pd.Series(
            float("nan"), index=result.index, dtype="float64"
        )
        result = result.loc[:, MARKET_BARS_INTERIM_COLUMNS]
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
                result[column] = _parse_optional_csmar_date(
                    result[column],
                    column,
                    error_code="invalid_stock_master_date",
                )
        result["timestamp"] = result["listing_date"]
        if result["timestamp"].isna().any():
            raise ValueError("missing_stock_master_effective_timestamp")
        result["source_asset_id"] = result["stock_code"]
        result.sort_values(["source_asset_id", "timestamp"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
    if data_id == "index_info":
        result = mapped
        if "index_class" not in result.columns and "index_category" in result.columns:
            result["index_class"] = result["index_category"]
        if "base_point" in result.columns:
            base_point_text = result["base_point"].astype("string").str.strip()
            base_point = strict_finite_numeric_series(
                result["base_point"].mask(base_point_text.eq("")),
                error="invalid_index_info_base_point",
                allow_null=True,
            )
            result["base_point"] = base_point.mask(base_point.eq(-9666))
        for column in ("base_date", "start_date"):
            if column in result.columns:
                result[column] = _parse_optional_csmar_date(
                    result[column],
                    column,
                    error_code="invalid_index_info_date",
                )
        result["timestamp"] = result["start_date"].combine_first(result["base_date"])
        if result["timestamp"].isna().any():
            raise ValueError("missing_index_info_effective_timestamp")
        result["source_asset_id"] = result["index_code"]
        result.sort_values(["source_asset_id", "timestamp"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
    if data_id == "index_market":
        result = mapped
        result["open"] = strict_finite_numeric_series(
            result["open"], error="invalid_index_open", allow_null=True
        )
        result["trade_date"] = strict_timestamp_series(
            result["trade_date"], error="invalid_index_return_date"
        ).dt.normalize()
        result["period_return"] = (
            strict_finite_numeric_series(
                result["return_1d_pct"], error="invalid_index_return"
            )
            / 100.0
        )
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
                allow_null=column
                not in {"industry_code", "industry_standard_id", "industry_level"},
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
        result["abandon_date"] = _parse_optional_csmar_date(
            result["abandon_date"],
            "abandon_date",
            error_code="invalid_industry_version_date",
        )
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
            result["declare_date"], "declare_date", error_code="invalid_analyst_date"
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
        for column in ("forecast_eps", "forecast_pe"):
            result[column] = strict_finite_numeric_series(
                result[column],
                error=f"invalid_analyst_value: {column}",
                allow_null=True,
            )
        result["timestamp"] = result["declare_date"]
        missing_date = result["declare_date"].isna()
        missing_institution = result["institution_id"].isna()
        excluded = missing_date | missing_institution
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


def _normalize_state_change_frame(frame: pd.DataFrame, data_id: str) -> pd.DataFrame:
    result = frame
    result["source_asset_id"] = result["stock_code"]
    if data_id == "listing_status_changes":
        result["announcement_date"] = _parse_optional_csmar_date(
            result["announcement_date"],
            "announcement_date",
            error_code="invalid_listing_status_date",
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
            result["announcement_date"],
            "announcement_date",
            error_code="invalid_stock_info_change_date",
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
        result["reason"] = strict_string_series(
            result["reason"], error="invalid_trading_no_limit_reason"
        )
        return (
            result.loc[:, TRADING_NO_LIMIT_COLUMNS]
            .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
            .reset_index(drop=True)
        )
    if data_id == "special_treatment_changes":
        result["announcement_date"] = strict_timestamp_series(
            result["announcement_date"], error="invalid_special_treatment_date"
        ).dt.normalize()
        result["effective_date"] = strict_timestamp_series(
            result["effective_date"], error="invalid_special_treatment_date"
        ).dt.normalize()
        result["timestamp"] = result["effective_date"]
        result["change_type"] = strict_string_series(
            result["change_type"], error="invalid_special_treatment_change_type"
        )
        if not result["change_type"].str.fullmatch(r"[ABCDSTX]{2}").all():
            raise ValueError("invalid_special_treatment_change_type")
        destination = result["change_type"].str[1]
        known = destination.isin(("A", "B", "C", "D"))
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
    return (
        result.loc[:, EQUITY_CHANGE_COLUMNS]
        .sort_values(["source_asset_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )


def _parse_optional_csmar_date(
    series: pd.Series,
    column: str,
    *,
    error_code: str,
) -> pd.Series:
    text = series.copy()
    blank = text.map(lambda value: isinstance(value, str) and not value.strip())
    return strict_timestamp_series(
        text.mask(blank),
        error=f"{error_code}: {column}",
        allow_null=True,
    )


def _parse_financial_correction_date(series: pd.Series) -> pd.Series:
    """Normalize the provider's comma-separated correction disclosure dates."""

    latest: list[object] = []
    for value in series:
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            latest.append(pd.NaT)
            continue
        tokens = [token.strip() for token in str(value).split(",")]
        if any(not token for token in tokens):
            raise ValueError("invalid_financial_correction_date")
        parsed = strict_timestamp_series(
            pd.Series(tokens), error="invalid_financial_correction_date"
        ).dt.normalize()
        if bool(parsed.duplicated().any()) or not (
            parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing
        ):
            raise ValueError("invalid_financial_correction_date")
        latest.append(parsed.max())
    return pd.Series(latest, index=series.index, dtype="datetime64[ns]")


def _parse_correction_flag(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if not numeric.isin((0, 1)).all():
        raise ValueError("invalid_financial_correction_flag")
    return numeric.astype(bool)


def filter_by_source_date_range(
    frame: pd.DataFrame,
    *,
    data_id: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if start_date is None and end_date is None:
        return frame
    column = _date_column(data_id)
    timestamps = strict_timestamp_series(
        frame[column],
        error=f"invalid_{data_id}_source_timestamp",
        allow_null=True,
    )
    mask = pd.Series(True, index=frame.index)
    if start_date is not None:
        mask &= timestamps >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= timestamps <= pd.Timestamp(end_date)
    return frame[mask].copy()


def _iter_interim_frames(
    *,
    data_id: str,
    input_files: Iterable[Path | tuple[Path, DatasetFileIdentity]],
    start_date: str | None,
    end_date: str | None,
    overrides: dict[str, dict[str, object]] | None,
    chunksize: int,
    pruning_reports: list[dict[str, object]] | None,
    verify_members: Callable[[], None] | None = None,
    source_row_counts: dict[str, int] | None = None,
    producer_diagnostics: dict[str, object] | None = None,
) -> Iterator[object]:
    validated_overrides = {} if overrides is None else overrides

    def activate() -> Iterator[object]:
        operation_reports: list[dict[str, object]] = []
        inputs = iter(input_files)
        with closing_iterator(inputs):
            try:
                first_input = next(inputs)
            except StopIteration:
                _append_duplicate_summary(
                    operation_reports,
                    data_id,
                    0,
                )
                first_input = None
            if first_input is not None:
                prepared_inputs = _iter_source_inputs(first_input, inputs)
                policy = current_runtime_resource_policy()
                if policy is None:
                    raise RuntimeError("runtime_resource_policy_not_active")
                spec = interim_catalog().get(data_id)
                index = ExactPrimaryKeyIndex(
                    primary_key=spec.primary_key,
                )
                with (
                    policy.lease(
                        "source.csmar.interim.index", index.retained_memory_bytes
                    ),
                    index,
                ):
                    frames = _iter_pruned_interim_frames(
                        data_id=data_id,
                        input_files=prepared_inputs,
                        start_date=start_date,
                        end_date=end_date,
                        overrides=validated_overrides,
                        chunksize=chunksize,
                        pruning_reports=operation_reports,
                        source_row_counts=source_row_counts,
                        duplicate_index=index,
                        producer_diagnostics=producer_diagnostics,
                    )
                    with closing_iterator(frames):
                        yield from frames
            if verify_members is not None:
                verify_members()
        if pruning_reports is not None:
            for report in operation_reports:
                _merge_pruning_report(pruning_reports, report)

    return activate()


def materialize_interim_data(
    *,
    layout: StorageLayout,
    raw_root: Path | None = None,
    data_ids: tuple[str, ...] = DEFAULT_INTERIM_DATA_IDS,
    start_date: str | None = "2014-01-01",
    end_date: str | None = "2025-12-31",
    override_path: Path | None = None,
    chunksize: int = 500_000,
    resource_policy: RuntimeResourcePolicy | None = None,
    instance_metadata: Mapping[str, object] | None = None,
) -> InterimImportResult:
    with runtime_resource_policy_scope(resource_policy):
        return _materialize_interim_data(
            layout=layout,
            raw_root=raw_root,
            data_ids=data_ids,
            start_date=start_date,
            end_date=end_date,
            override_path=override_path,
            chunksize=chunksize,
            instance_metadata=instance_metadata,
        )


def _materialize_interim_data(
    *,
    layout: StorageLayout,
    raw_root: Path | None,
    data_ids: tuple[str, ...],
    start_date: str | None,
    end_date: str | None,
    override_path: Path | None,
    chunksize: int,
    instance_metadata: Mapping[str, object] | None,
) -> InterimImportResult:
    resource_policy = current_runtime_resource_policy()
    if resource_policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    if type(chunksize) is not int or chunksize <= 0:
        raise ValueError("invalid_csmar_chunksize")
    catalog = interim_catalog()
    source_raw_root = raw_root or layout.raw_source_root(SourceId(SOURCE_ID))
    overrides = (
        load_column_overrides(override_path)
        if override_path
        else _validated_column_overrides({})
    )
    written: dict[str, Path] = {}
    skipped: set[str] = set()
    pruning_reports: list[dict[str, object]] = []
    pending = set(data_ids)
    require_unaliased_path(
        source_raw_root,
        error="csmar_raw_export_identity_changed",
    )
    identity_index = ExactPrimaryKeyIndex(
        primary_key=("dataset_path",),
    )
    with (
        resource_policy.lease(
            "source.csmar.raw.identity.index",
            identity_index.retained_memory_bytes,
        ),
        identity_index,
    ):
        indexed_files = dataset_files_with_identity_by_id(
            source_raw_root,
            resource_policy=resource_policy,
            identity_index=identity_index,
            requested_datasets=data_ids,
        )

        def verify_members() -> None:
            _verify_csmar_member_set(
                source_raw_root,
                identity_index,
                requested_datasets=data_ids,
            )

        with closing_iterator(indexed_files):
            for data_id, group in groupby(indexed_files, key=lambda item: item[0]):
                selected = tuple(group)
                identity = content_hash_for_files(
                    selected,
                    error="csmar_raw_export_identity_changed",
                    resource_policy=resource_policy,
                )
                if (
                    instance_metadata is not None
                    and COMMON_MANIFEST_METADATA_KEY in instance_metadata
                ):
                    envelope = validate_business_manifest_envelope(
                        instance_metadata[COMMON_MANIFEST_METADATA_KEY]
                    )
                    if (
                        envelope["semantic_metadata"]["raw_content_identity"]
                        != identity
                    ):
                        raise ValueError("csmar_raw_export_identity_changed")

                def verify_source(_receipt, files=selected, expected=identity):
                    verify_members()
                    if (
                        content_hash_for_files(
                            files,
                            error="csmar_raw_export_identity_changed",
                            resource_policy=resource_policy,
                        )
                        != expected
                    ):
                        raise ValueError("csmar_raw_export_identity_changed")
                    verify_members()

                input_files = ((item[1], item[2]) for item in selected)
                pending.discard(data_id)
                spec = catalog.get(data_id)
                artifact_root = require_artifact_root(
                    layout.target_root(
                        spec.data_id,
                        layout.interim_data_root(spec.source_id, spec.data_id),
                    )
                )
                metadata = dict(instance_metadata or {})
                diagnostics = dict(metadata.get("producer_diagnostics", {}))
                diagnostics["exact_duplicate_count"] = 0
                metadata["producer_diagnostics"] = diagnostics
                source_row_counts = None
                if data_id == "analyst_forecast":
                    source_row_counts = {}
                    diagnostics["source_row_counts"] = source_row_counts
                frames = _iter_interim_frames(
                    data_id=data_id,
                    input_files=input_files,
                    start_date=start_date,
                    end_date=end_date,
                    overrides=overrides,
                    chunksize=chunksize,
                    pruning_reports=pruning_reports,
                    verify_members=verify_members,
                    source_row_counts=source_row_counts,
                    producer_diagnostics=diagnostics,
                )
                with closing_iterator(frames):
                    try:
                        write_dataset(
                            artifact_root,
                            frames,
                            spec=spec,
                            resource_policy=resource_policy,
                            instance_metadata=metadata,
                            staged_validator=verify_source,
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
        pruning_report_text=render_column_pruning_report(pruning_reports),
    )


def _iter_pruned_interim_frames(
    *,
    data_id: str,
    input_files: Iterable[Path | tuple[Path, DatasetFileIdentity]],
    start_date: str | None,
    end_date: str | None,
    overrides: Mapping[str, Mapping[str, object]],
    chunksize: int,
    pruning_reports: list[dict[str, object]] | None,
    duplicate_index: ExactPrimaryKeyIndex,
    producer_diagnostics: dict[str, object] | None,
    source_row_counts: dict[str, int] | None = None,
) -> Iterator[object]:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    local_reports: list[dict[str, object]] = []
    primary_key = interim_catalog().get(data_id).primary_key
    exact_duplicate_count = 0
    lease: MemoryLease | None = None
    for input in input_files:
        if isinstance(input, tuple):
            input_file, expected_identity = input
        else:
            input_file = input
            expected_identity = None
        chunks = iter(
            iter_private_export_chunks_with_policy(
                input_file,
                dataset=data_id,
                chunksize=chunksize,
                resource_policy=policy,
                expected_identity=expected_identity,
            )
        )
        with closing_iterator(chunks):
            try:
                upstream_bound = next(chunks)
            except StopIteration:
                continue
            pending_frame: pd.DataFrame | None = None
            pending_lease: MemoryLease | None = None
            pending_bound: int | None = None
            pending_span: int | None = None
            try:
                while upstream_bound is not None or pending_frame is not None:
                    source_owned = pending_frame is None
                    if source_owned:
                        if lease is None:
                            lease = yield upstream_bound
                        elif upstream_bound > lease.reserved_bytes:
                            policy.require_working_set(
                                upstream_bound - lease.reserved_bytes,
                                error="csmar_interim_working_set_exceeds_task_memory_budget",
                            )
                            lease.grow(upstream_bound - lease.reserved_bytes)
                        try:
                            owned = chunks.send(lease)
                        except StopIteration:
                            break
                        if not isinstance(owned, OwnedValue):
                            raise TypeError("invalid_csmar_owned_frame")
                        if owned.lease is not lease:
                            raise RuntimeError("runtime_memory_lease_transfer_mismatch")
                        chunk = owned.value
                    else:
                        pending_handoff = yield (pending_bound, pending_lease)
                        if pending_handoff is not pending_lease:
                            raise RuntimeError("runtime_memory_lease_transfer_mismatch")
                        lease = pending_handoff
                        owned = None
                        chunk = pending_frame
                    if type(chunk) is not pd.DataFrame:
                        raise TypeError("invalid_dataset_chunk")
                    chunk_bytes = pandas_frame_payload_bytes(chunk)
                    start = 0
                    span = pending_span or len(chunk)
                    downsized = False
                    emitted = False
                    while start < len(chunk):
                        stop = min(len(chunk), start + span)
                        candidate = (
                            chunk
                            if start == 0 and stop == len(chunk)
                            else chunk.iloc[start:stop]
                        )
                        try:
                            pruned, report = _transform_interim_chunk(
                                candidate,
                                data_id=data_id,
                                start_date=start_date,
                                end_date=end_date,
                                overrides=overrides,
                                retained_previous_bytes=(
                                    0 if candidate is chunk else chunk_bytes
                                ),
                            )
                        except _InterimCapacityExceeded as exc:
                            del candidate
                            if downsized or stop - start == 1:
                                raise
                            available_bytes = exc.budget_bytes - chunk_bytes
                            if available_bytes <= 0:
                                raise
                            span = max(
                                1,
                                min(
                                    stop - start - 1,
                                    (stop - start)
                                    * available_bytes
                                    // max(1, exc.required_bytes),
                                ),
                            )
                            downsized = True
                            continue
                        if data_id == "analyst_forecast":
                            counts = report.pop("source_row_counts")
                        if source_row_counts is not None:
                            for key, count in counts.items():
                                source_row_counts[key] = (
                                    source_row_counts.get(key, 0) + count
                                )
                        if report:
                            _merge_pruning_report(local_reports, report)
                        if not pruned.empty:
                            frame_bytes = pandas_frame_payload_bytes(pruned)
                            dedupe_bound = (
                                chunk_bytes + _deduplication_working_set_bytes(pruned)
                            )
                            if dedupe_bound > lease.reserved_bytes:
                                policy.require_working_set(
                                    dedupe_bound - lease.reserved_bytes,
                                    error="csmar_interim_working_set_exceeds_task_memory_budget",
                                )
                                lease.grow(dedupe_bound - lease.reserved_bytes)
                            table = pa.Table.from_pandas(pruned, preserve_index=False)
                            try:
                                if table.nbytes > 2 * frame_bytes + 64 * 1024:
                                    raise ValueError("csmar_interim_bound_invalid")
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
                                        payload_target_bytes=max(
                                            1,
                                            lease.reserved_bytes
                                            - chunk_bytes
                                            - frame_bytes,
                                        ),
                                    )
                                )
                            except ValueError as error:
                                if (
                                    str(error)
                                    == "conflicting_exact_primary_key_payload"
                                ):
                                    raise ValueError(
                                        "conflicting_csmar_primary_key_duplicates: "
                                        f"{data_id}"
                                    ) from error
                                raise
                            finally:
                                del table
                            exact_duplicate_count += duplicate_count
                            if producer_diagnostics is not None:
                                producer_diagnostics["exact_duplicate_count"] = (
                                    exact_duplicate_count
                                )
                            if not positions:
                                del positions, pruned, report, candidate
                                start = stop
                                continue
                            if len(positions) != len(pruned):
                                selected = pruned.iloc[positions.view].reset_index(
                                    drop=True
                                )
                                required = (
                                    chunk_bytes
                                    + frame_bytes
                                    + pandas_frame_payload_bytes(selected)
                                    + positions.retained_memory_bytes
                                    + 64 * 1024
                                )
                                if required > lease.reserved_bytes:
                                    raise ValueError("csmar_interim_bound_invalid")
                                pruned = selected
                                del selected
                            del positions
                        if pruned.empty and data_id != "analyst_forecast":
                            del pruned, report, candidate
                            start = stop
                            continue
                        next_pending_frame: pd.DataFrame | None = None
                        next_pending_lease: MemoryLease | None = None
                        next_pending_bound: int | None = None
                        next_pending_span: int | None = None
                        if stop < len(chunk):
                            remainder_bytes = pandas_frame_payload_bytes(
                                chunk.iloc[stop:]
                            )
                            next_pending_lease = policy.lease(
                                "source.csmar.interim.pending", remainder_bytes
                            )
                            try:
                                next_pending_frame = chunk.iloc[stop:].copy()
                                next_pending_bound = lease.reserved_bytes
                                next_pending_span = span
                            except BaseException:
                                _close_lease_preserving_primary(
                                    next_pending_lease,
                                    owner="source.csmar.interim.pending",
                                )
                                raise
                        if source_owned:
                            try:
                                upstream_bound = chunks.send(None)
                            except StopIteration:
                                upstream_bound = None
                            if upstream_bound is not None and (
                                type(upstream_bound) is not int or upstream_bound <= 0
                            ):
                                raise ValueError("invalid_csmar_raw_bound")
                        del report, candidate
                        del owned, chunk
                        pending_frame = None
                        if pending_lease is not None and pending_lease is not lease:
                            pending_lease.close()
                        pending_lease = None
                        pending_frame = next_pending_frame
                        pending_lease = next_pending_lease
                        pending_bound = next_pending_bound
                        pending_span = next_pending_span
                        lease.shrink_to(pandas_frame_payload_bytes(pruned) + 64 * 1024)
                        owned_output = OwnedValue(pruned, lease)
                        del pruned
                        yield owned_output
                        del owned_output
                        lease = None
                        emitted = True
                        break
                    if not emitted:
                        del owned, chunk
                        pending_frame = None
                        if pending_lease is not None and pending_lease is not lease:
                            pending_lease.close()
                        pending_lease = None
                        pending_bound = None
                        pending_span = None
                        if source_owned:
                            try:
                                upstream_bound = chunks.send(None)
                            except StopIteration:
                                upstream_bound = None
                            if upstream_bound is not None and (
                                type(upstream_bound) is not int or upstream_bound <= 0
                            ):
                                raise ValueError("invalid_csmar_raw_bound")
            finally:
                pending_frame = None
                if pending_lease is not None:
                    _close_lease_preserving_primary(
                        pending_lease,
                        owner="source.csmar.interim.pending",
                    )
                    pending_lease = None
                pending_bound = None
                pending_span = None
    if pruning_reports is not None:
        _append_duplicate_summary(pruning_reports, data_id, exact_duplicate_count)
        for report in local_reports:
            _merge_pruning_report(pruning_reports, report)


def _transform_interim_chunk(
    chunk: pd.DataFrame,
    *,
    data_id: str,
    start_date: str | None,
    end_date: str | None,
    overrides: Mapping[str, Mapping[str, object]],
    retained_previous_bytes: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    chunk_bytes = pandas_frame_payload_bytes(chunk)
    normalization_preallocation_bytes = _normalization_preallocation_bytes(
        chunk, data_id=data_id
    )
    _require_interim_transform_headroom(
        retained_previous_bytes + chunk_bytes + normalization_preallocation_bytes,
    )
    normalized = normalize_raw_frame(chunk, data_id)
    source_row_counts = normalized.attrs.pop("source_row_counts", None)
    normalized_bytes = pandas_frame_payload_bytes(normalized)
    filter_preallocation_bytes = _filter_preallocation_bytes(
        normalized,
        data_id=data_id,
        start_date=start_date,
        end_date=end_date,
    )
    _require_interim_transform_headroom(
        retained_previous_bytes + normalized_bytes + filter_preallocation_bytes,
    )
    filtered = _filter_normalized_frame(
        normalized,
        data_id=data_id,
        start_date=start_date,
        end_date=end_date,
    )
    filtered_bytes = pandas_frame_payload_bytes(filtered)
    _require_interim_transform_headroom(
        retained_previous_bytes + normalized_bytes + filtered_bytes,
    )
    del normalized
    if source_row_counts is not None:
        source_row_counts["retained_after_date_filter_rows"] = len(filtered)
        source_row_counts["excluded_date_range_rows"] = source_row_counts[
            "retained_before_date_filter_rows"
        ] - len(filtered)
    if filtered.empty:
        return filtered, (
            {"source_row_counts": source_row_counts}
            if source_row_counts is not None
            else {}
        )
    _require_interim_transform_headroom(
        retained_previous_bytes + filtered_bytes + filtered_bytes,
    )
    pruned, report = _apply_output_column_rules_validated(
        filtered,
        data_id,
        overrides=overrides,
    )
    _require_interim_transform_headroom(
        retained_previous_bytes
        + filtered_bytes
        + _deduplication_working_set_bytes(pruned),
    )
    del filtered
    if source_row_counts is not None:
        report["source_row_counts"] = source_row_counts
    return pruned, report


def _deduplication_working_set_bytes(frame: pd.DataFrame) -> int:
    return (
        3 * pandas_frame_payload_bytes(frame)
        + len(frame) * array("Q").itemsize
        + 64 * 1024
    )


def _require_interim_transform_headroom(required_bytes: int) -> None:
    policy = current_runtime_resource_policy()
    explicit = None if policy is None else policy.explicit_memory_cap_bytes
    if explicit is not None and policy is not None:
        if policy.used_memory_bytes + required_bytes > explicit:
            raise _InterimCapacityExceeded(required_bytes, explicit)


def _normalization_preallocation_bytes(frame: pd.DataFrame, *, data_id: str) -> int:
    """Bound the actual variable-width and ordering allocations of normalization."""

    if frame.empty:
        return 0
    pointer_bytes = array("Q").itemsize
    variable_width_bytes = pandas_frame_payload_bytes(frame)
    primary_key_columns = len(interim_catalog().get(data_id).primary_key)
    ordering_bytes = len(frame) * (primary_key_columns + 2) * pointer_bytes
    if data_id == "index_membership":
        # Hash strings coexist with the construction list and assigned object array.
        variable_width_bytes += len(frame) * (
            sys.getsizeof("0" * 64) + 2 * pointer_bytes
        )
    return variable_width_bytes + ordering_bytes


def _filter_preallocation_bytes(
    frame: pd.DataFrame,
    *,
    data_id: str,
    start_date: str | None,
    end_date: str | None,
) -> int:
    if not (
        (data_id in TIME_RANGE_FILTER_DATA_IDS and (start_date or end_date))
        or (data_id in START_END_FILTER_DATA_IDS and (start_date or end_date))
        or (data_id in END_ONLY_FILTER_DATA_IDS and end_date)
    ):
        return 0
    return len(frame) * (array("Q").itemsize + 1) + pandas_frame_payload_bytes(frame)


def _merge_pruning_report(
    reports: list[dict[str, object]], report: dict[str, object]
) -> None:
    for current in reports:
        if current.get("data_id") != report.get("data_id") or current.get(
            "report_type"
        ) != report.get("report_type"):
            continue
        for key, value in report.items():
            if isinstance(value, list) and isinstance(current.get(key), list):
                current[key] = list(dict.fromkeys([*current[key], *value]))
            elif isinstance(value, dict) and isinstance(current.get(key), dict):
                current[key] = {**current[key], **value}
        return
    reports.append(report)


def _iter_source_inputs(
    first: Path | tuple[Path, DatasetFileIdentity],
    remaining: Iterator[Path | tuple[Path, DatasetFileIdentity]],
) -> Iterator[Path | tuple[Path, DatasetFileIdentity]]:
    for value in chain((first,), remaining):
        parent = require_unaliased_path(
            Path(value[0] if isinstance(value, tuple) else value).parent,
            error="csmar_interim_source_parent_invalid",
        )
        try:
            metadata = parent.lstat()
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("csmar_interim_source_parent_invalid") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("csmar_interim_source_parent_invalid")
        yield value


def _append_duplicate_summary(
    reports: list[dict[str, object]],
    data_id: str,
    exact_duplicate_count: int,
) -> None:
    reports.append(
        {
            "data_id": data_id,
            "report_type": "source_duplicate_summary",
            "exact_duplicate_count": int(exact_duplicate_count),
        }
    )


def _interim_spec(data_id: str, columns: tuple[str, ...]) -> InterimDataSpec:
    if data_id == "analyst_forecast":
        primary_key = ("source_asset_id", "timestamp", "report_id", "forecast_end_date")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id == "industry_classification":
        primary_key = ("source_asset_id", "timestamp", "industry_standard_id")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id == "industry_classification_sets":
        primary_key = ("industry_standard_id", "industry_code")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=(),
        )
    if data_id == "industry_classification_versions":
        primary_key = ("industry_standard_id", "timestamp")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id in {"listing_status_changes", "stock_info_changes"}:
        primary_key = ("source_asset_id", "timestamp", "changed_item")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id == "special_treatment_changes":
        primary_key = ("source_asset_id", "timestamp", "change_type")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id == "equity_changes":
        primary_key = ("source_asset_id", "timestamp", "report_type")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id in {"index_membership", "index_weights"}:
        primary_key = ("source_asset_id", "constituent_asset_id", "timestamp")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id in FINANCIAL_COLUMNS_BY_DATA_ID:
        primary_key = (
            "source_asset_id",
            "timestamp",
            "report_type",
            "version_identity",
        )
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id == "trading_calendar":
        primary_key = ("timestamp", "source_market_type")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.SOURCE_MARKET_CALENDAR_DATE,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id == "index_info":
        primary_key = ("source_asset_id", "timestamp")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if data_id == "stock_master":
        primary_key = ("source_asset_id", "timestamp")
        return InterimDataSpec(
            data_id=data_id,
            source_id=SourceId(SOURCE_ID),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    primary_key = ("source_asset_id", "timestamp")
    return InterimDataSpec(
        data_id=data_id,
        source_id=SourceId(SOURCE_ID),
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=_column_roles(columns, primary_key=primary_key),
        primary_key=primary_key,
        partition_columns=("year",),
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
    result["source_day_of_week"] = strict_integer_series(
        result["source_day_of_week"],
        error="invalid_trading_calendar_day_of_week",
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


def _date_column(data_id: str) -> str:
    if data_id in {
        MARKET_BARS_DATA_ID,
        "adjustment",
        "market_cap",
        "stock_master",
        "trading_calendar",
        "index_membership",
        "index_weights",
        "index_market",
        "financial_balance",
        "financial_income",
        "financial_cashflow",
        "industry_classification",
        "industry_classification_annual",
        "industry_classification_versions",
        "listing_status_changes",
        "stock_info_changes",
        "stock_info_annual",
        "trading_no_limit",
        "special_treatment_changes",
        "equity_changes",
        "analyst_forecast",
    }:
        return "timestamp"
    raise ValueError(f"data_id_has_no_filtering_date_column: {data_id}")


def _filter_normalized_frame(
    frame: pd.DataFrame,
    *,
    data_id: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if data_id in TIME_RANGE_FILTER_DATA_IDS:
        return filter_by_source_date_range(
            frame,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
        )
    if data_id in START_END_FILTER_DATA_IDS:
        return filter_by_source_date_range(
            frame,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
        )
    if data_id in END_ONLY_FILTER_DATA_IDS:
        return filter_by_source_date_range(
            frame,
            data_id=data_id,
            start_date=None,
            end_date=end_date,
        )
    return frame
