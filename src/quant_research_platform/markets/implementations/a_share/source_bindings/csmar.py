"""A-share binding for CSMAR source exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.fields import (
    platform_entity_time_key,
    platform_field_catalog,
    platform_interim_column_names,
)
from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.catalog.provider_bindings import ProviderBinding
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.markets.implementations.a_share.adapter import (
    AShareMarketAdapter,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_DAILY_BAR_DATA_PRODUCT_ID,
)
from quant_research_platform.sources.csmar.field_mappings import (
    map_csmar_columns,
    normalize_stock_code,
)
from quant_research_platform.sources.csmar.fields import csmar_field_role
from quant_research_platform.sources.csmar.ids import CSMAR_SOURCE_ID
from quant_research_platform.sources.csmar.raw_exports import (
    find_dataset_files,
    iter_private_export_chunks,
)
from quant_research_platform.sources.csmar.raw_schemas import parse_raw_filename
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.parquet import write_interim_data

INDEX_METADATA_TIME_COLUMNS: tuple[str, ...] = (
    "start_date",
    "base_date",
)
DEFAULT_INTERIM_DATASETS: tuple[str, ...] = (
    "daily_market",
    "benchmark_index_market",
    "stock_master",
    "adjustment",
    "market_cap",
    "index_info",
    "trading_calendar",
)
TIME_RANGE_FILTER_DATASETS = {"daily_market", "index_market", "benchmark_index_market"}
END_ONLY_FILTER_DATASETS = {"market_cap"}
START_END_FILTER_DATASETS = {"adjustment"}
ASSET_REFERENCE_INTERIM_DATASETS = {"stock_master", "index_info"}
DAILY_A_SHARE_MARKET_TYPES = {1, 4, 16, 32}
DAILY_ST_STATUS_CODES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
ASSET_TIME_INTERIM_KEEP_COLUMNS = platform_interim_column_names()
ASSET_REFERENCE_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "source_id",
    "source_asset_id",
)
ASSET_REFERENCE_TIMESTAMP_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_REFERENCE_INTERIM_KEEP_COLUMNS,
    "timestamp",
)
DAILY_MARKET_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_TIME_INTERIM_KEEP_COLUMNS,
    "market_type",
    "trading_status_code",
    "filling_flag",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "prev_close",
    "limit_status",
    "return_1d",
    "adj_close",
    "adj_open",
    "float_market_cap",
    "total_market_cap",
    "is_limit_up",
    "is_limit_down",
    "is_trading",
    "is_st",
    "is_pt",
    "is_a_share_scope",
    "eligible_base",
    "turnover",
)
STOCK_MASTER_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_REFERENCE_TIMESTAMP_INTERIM_KEEP_COLUMNS,
    "stock_name",
    "listing_date",
    "delisting_date",
    "company_status",
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
    "status_change_date",
)
MARKET_CAP_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_TIME_INTERIM_KEEP_COLUMNS,
    "share_change_date",
    "total_shares",
    "a_float_shares",
)
ADJUSTMENT_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_TIME_INTERIM_KEEP_COLUMNS,
    "forward_adjust_factor",
    "backward_adjust_factor",
    "cum_forward_adjust_factor",
    "cum_backward_adjust_factor",
)
INDEX_INFO_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_REFERENCE_TIMESTAMP_INTERIM_KEEP_COLUMNS,
    "index_name",
    "calculation_formula",
    "index_class",
    "start_date",
    "base_date",
)
BENCHMARK_INDEX_MARKET_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_TIME_INTERIM_KEEP_COLUMNS,
    "open",
    "high",
    "low",
    "close",
    "volume_10k_shares",
    "amount_10k_cny",
    "return_1d_pct",
    "return_1d",
    "index_name",
    "benchmark_name",
    "benchmark_tier",
    "ohlc_valid",
)
INDEX_MARKET_KEEP_COLUMNS: tuple[str, ...] = (
    *ASSET_TIME_INTERIM_KEEP_COLUMNS,
    "open",
    "high",
    "low",
    "close",
    "volume_10k_shares",
    "amount_10k_cny",
    "return_1d_pct",
    "return_1d",
    "index_name",
)
TRADING_CALENDAR_INTERIM_KEEP_COLUMNS: tuple[str, ...] = (
    "market_id",
    "source_id",
    "timestamp",
    "calendar_date",
    "source_market_type",
    "is_trading_day",
    "source_day_of_week",
    "source_state",
)
A_SHARE_CSMAR_INTERIM_KEEP_COLUMNS: dict[str, tuple[str, ...]] = {
    "daily_market": DAILY_MARKET_INTERIM_KEEP_COLUMNS,
    "stock_master": STOCK_MASTER_INTERIM_KEEP_COLUMNS,
    "market_cap": MARKET_CAP_INTERIM_KEEP_COLUMNS,
    "adjustment": ADJUSTMENT_INTERIM_KEEP_COLUMNS,
    "index_info": INDEX_INFO_INTERIM_KEEP_COLUMNS,
    "benchmark_index_market": BENCHMARK_INDEX_MARKET_KEEP_COLUMNS,
    "index_market": INDEX_MARKET_KEEP_COLUMNS,
    "trading_calendar": TRADING_CALENDAR_INTERIM_KEEP_COLUMNS,
}
A_SHARE_CSMAR_AUTO_DROPPED_COLUMNS: dict[str, dict[str, str]] = {
    "daily_market": {
        "return_with_dividend": "derived as return_1d",
        "comparable_close_with_dividend": "derived as adj_close",
        "float_market_cap_thousand": "converted to float_market_cap",
        "total_market_cap_thousand": "converted to total_market_cap",
    }
}
A_SHARE_CSMAR_OPTIONAL_COLUMNS: dict[str, dict[str, str]] = {
    "daily_market": {
        "return_without_dividend": "not retained in the 0.1 A-share interim contract",
        "comparable_close_without_dividend": (
            "not retained in the 0.1 A-share interim contract"
        ),
        "Dsmvtl2": "not retained in the 0.1 A-share interim contract",
        "Dsmvtl3": "not retained in the 0.1 A-share interim contract",
        "Capchgdt": "market_cap provides share-change timing",
        "Ahshrtrd_D": "after-hours volume is not part of the 0.1 liquidity inputs",
        "Ahvaltrd_D": "after-hours amount is not part of the 0.1 liquidity inputs",
        "change_ratio": "return_1d and price fields are used instead",
        "limit_down": "limit_status is used instead",
        "limit_up": "limit_status is used instead",
    },
    "stock_master": {
        "market_type": "daily_market retains market_type as an asset-time field",
        "status_change_date": "retained when present for asset reference timing",
    },
    "market_cap": {
        "b_float_shares": "not retained in the 0.1 A-share market-cap contract",
        "h_float_shares": "not retained in the 0.1 A-share market-cap contract",
    },
}


@dataclass(frozen=True)
class BenchmarkIndex:
    """Candidate benchmark index metadata."""

    code: str
    name: str
    tier: str


BENCHMARK_INDEX_POOL: dict[str, BenchmarkIndex] = {
    "000985": BenchmarkIndex("000985", "中证全指指数", "core"),
    "000300": BenchmarkIndex("000300", "沪深300指数", "core"),
    "000905": BenchmarkIndex("000905", "中证小盘500指数", "core"),
    "000852": BenchmarkIndex("000852", "中证1000指数", "core"),
    "000001": BenchmarkIndex("000001", "上证综合指数", "core"),
    "399001": BenchmarkIndex("399001", "深证成份指数", "core"),
    "399006": BenchmarkIndex("399006", "创业板指数(价格)", "core"),
    "000688": BenchmarkIndex("000688", "上证科创板50成份指数", "core"),
    "932000": BenchmarkIndex("932000", "中证2000指数", "extended"),
    "000906": BenchmarkIndex("000906", "中证800指数", "extended"),
    "000510": BenchmarkIndex("000510", "中证A500指数", "extended"),
    "000016": BenchmarkIndex("000016", "上证50指数", "extended"),
    "399106": BenchmarkIndex("399106", "深证综合指数", "extended"),
    "399102": BenchmarkIndex("399102", "创业板综合指数", "extended"),
    "000698": BenchmarkIndex("000698", "上证科创板100指数", "extended"),
    "000984": BenchmarkIndex("000984", "沪深300等权重指数", "equal_weight"),
    "000982": BenchmarkIndex("000982", "中证500等权重指数", "equal_weight"),
    "000842": BenchmarkIndex("000842", "中证800等权重指数", "equal_weight"),
    "932382": BenchmarkIndex("932382", "中证1000等权重指数", "equal_weight"),
    "932548": BenchmarkIndex("932548", "中证A500等权重指数", "equal_weight"),
}


@dataclass(frozen=True)
class AShareCSMARInterimImportResult:
    """Summary of an A-share CSMAR interim import run."""

    written_data_roots: dict[str, Path]
    skipped: tuple[str, ...]
    pruning_report_text: str


def a_share_csmar_default_override_payload() -> dict[str, dict[str, object]]:
    """Return a local column-override template for A-share CSMAR interim imports."""

    datasets = sorted(
        set(A_SHARE_CSMAR_INTERIM_KEEP_COLUMNS) | set(A_SHARE_CSMAR_OPTIONAL_COLUMNS)
    )
    return {
        dataset: {
            "keep": [],
            "drop": [],
            "_available_optional_columns": A_SHARE_CSMAR_OPTIONAL_COLUMNS.get(
                dataset,
                {},
            ),
        }
        for dataset in datasets
    }


def a_share_csmar_load_column_overrides(
    path: Path,
) -> dict[str, dict[str, object]]:
    """Load a local A-share CSMAR column override file when present."""

    if not path.exists():
        return a_share_csmar_default_override_payload()
    return json.loads(path.read_text(encoding="utf-8"))


def a_share_csmar_apply_interim_column_rules(
    frame: pd.DataFrame,
    interim_data_id: str,
    *,
    overrides: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply A-share CSMAR interim keep/drop rules."""

    rule_dataset = (
        "benchmark_index_market"
        if interim_data_id == "benchmark_index_market"
        else interim_data_id
    )
    base_keep = list(
        A_SHARE_CSMAR_INTERIM_KEEP_COLUMNS.get(rule_dataset, tuple(frame.columns))
    )
    override = (overrides or {}).get(rule_dataset, {})
    manual_keep = [str(column) for column in override.get("keep", [])]
    manual_drop = [str(column) for column in override.get("drop", [])]

    keep_order = list(dict.fromkeys([*base_keep, *manual_keep]))
    available_keep = [column for column in keep_order if column in frame.columns]
    final_columns = [column for column in available_keep if column not in manual_drop]

    dropped = [column for column in frame.columns if column not in final_columns]
    auto_dropped = {
        column: reason
        for column, reason in A_SHARE_CSMAR_AUTO_DROPPED_COLUMNS.get(
            rule_dataset,
            {},
        ).items()
        if column in dropped
    }
    optional_available = {
        column: reason
        for column, reason in A_SHARE_CSMAR_OPTIONAL_COLUMNS.get(
            rule_dataset,
            {},
        ).items()
        if column in frame.columns
    }
    report = {
        "dataset": rule_dataset,
        "input_columns": list(frame.columns),
        "kept_columns": final_columns,
        "dropped_columns": dropped,
        "auto_dropped_columns": auto_dropped,
        "optional_available_columns": optional_available,
        "manual_keep": manual_keep,
        "manual_drop": manual_drop,
    }
    return frame[final_columns].copy(), report


def a_share_csmar_render_column_pruning_report(
    reports: list[dict[str, object]],
) -> str:
    """Render the latest private A-share CSMAR pruning audit as Markdown."""

    lines = [
        "# A-share CSMAR interim column pruning report",
        "",
        "This private audit report summarizes aggregate column pruning metadata.",
        "",
    ]
    for report in reports:
        if report.get("report_type") == "reference_duplicate_summary":
            lines.extend(
                [
                    f"## {report['dataset']} reference duplicate summary",
                    "",
                    "- exact duplicate primary-key rows deduped: "
                    f"{report['reference_exact_duplicate_count']}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"## {report['dataset']}",
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


def a_share_csmar_interim_catalog() -> InterimDataCatalog:
    """Return A-share CSMAR persisted interim specs."""

    catalog = InterimDataCatalog()
    for interim_data_id, columns in A_SHARE_CSMAR_INTERIM_KEEP_COLUMNS.items():
        catalog.register(_interim_spec(interim_data_id, columns))
    return catalog


def a_share_csmar_reference_spec_for_frame(
    interim_data_id: str,
    frame: pd.DataFrame,
    *,
    force_time_aware: bool = False,
) -> InterimDataSpec:
    """Return the reference spec implied by one normalized reference frame.

    The static branch is kept for synthetic classification tests and future
    provider reuse. The active A-share CSMAR index_info catalog policy is
    forced time-aware in _interim_spec so imports do not silently change schema
    based on local file state.
    """

    if interim_data_id not in ASSET_REFERENCE_INTERIM_DATASETS:
        raise ValueError(f"unsupported_reference_interim_data_id: {interim_data_id}")
    columns = A_SHARE_CSMAR_INTERIM_KEEP_COLUMNS[interim_data_id]
    time_aware = force_time_aware or _has_multiple_effective_reference_records(frame)
    if time_aware:
        primary_key = (
            *ASSET_REFERENCE_INTERIM_KEEP_COLUMNS,
            "timestamp",
        )
        grain = DataGrain.ASSET_REFERENCE_TIMESTAMP
        partition_columns = ("year",)
    else:
        primary_key = ASSET_REFERENCE_INTERIM_KEEP_COLUMNS
        grain = DataGrain.ASSET_REFERENCE
        partition_columns = ()
        columns = tuple(column for column in columns if column != "timestamp")
    return InterimDataSpec(
        market_id=MarketId("a_share"),
        source_id=SourceId(CSMAR_SOURCE_ID),
        interim_data_id=interim_data_id,
        frequency=DataFrequency.DAILY,
        grain=grain,
        column_roles=_column_roles(columns, primary_key=primary_key),
        primary_key=primary_key,
        partition_columns=partition_columns,
    )


def a_share_csmar_provider_binding() -> ProviderBinding:
    """Return the A-share data product binding to CSMAR source datasets."""

    return ProviderBinding(
        data_product_id=A_SHARE_DAILY_BAR_DATA_PRODUCT_ID,
        provider_id=CSMAR_SOURCE_ID,
        source_id=CSMAR_SOURCE_ID,
        implementation_package="quant_research_platform.sources.csmar",
        required_source_dataset_ids=(
            "daily_market",
            "stock_master",
            "market_cap",
            "adjustment",
            "index_info",
            "benchmark_index_market",
            "trading_calendar",
        ),
    )


def a_share_csmar_normalize_daily_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 0.1 canonical interim keys to a normalized daily market frame."""

    return a_share_csmar_normalize_asset_time_frame(frame, time_column="trade_date")


def a_share_csmar_normalize_asset_time_frame(
    frame: pd.DataFrame,
    *,
    time_column: str,
    source_asset_column: str = "stock_code",
) -> pd.DataFrame:
    """Add canonical A-share asset/time fields to stock-level CSMAR interim data."""

    return AShareMarketAdapter().with_daily_bar_canonical_fields(
        frame,
        source_id=CSMAR_SOURCE_ID,
        source_asset_column=source_asset_column,
        session_date_column=time_column,
        asset_namespace="stock",
    )


def a_share_csmar_normalize_index_time_frame(
    frame: pd.DataFrame,
    *,
    time_column: str = "trade_date",
    source_asset_column: str = "index_code",
) -> pd.DataFrame:
    """Add canonical fields for A-share benchmark/index time-series data."""

    return AShareMarketAdapter().with_daily_bar_canonical_fields(
        frame,
        source_id=CSMAR_SOURCE_ID,
        source_asset_column=source_asset_column,
        session_date_column=time_column,
        asset_namespace="index",
    )


def a_share_csmar_normalize_index_metadata_frame(
    frame: pd.DataFrame,
    *,
    source_asset_column: str = "index_code",
    time_columns: tuple[str, ...] = INDEX_METADATA_TIME_COLUMNS,
) -> pd.DataFrame:
    """Add canonical fields for A-share index metadata without requiring dates."""

    return AShareMarketAdapter().with_metadata_canonical_fields(
        frame,
        source_id=CSMAR_SOURCE_ID,
        source_asset_column=source_asset_column,
        time_columns=time_columns,
        asset_namespace="index",
    )


def a_share_csmar_normalize_private_export_frame(
    frame: pd.DataFrame,
    dataset: str,
    *,
    fallback_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize one raw CSMAR frame to A-share canonical interim columns."""

    mapped = map_csmar_columns(frame, dataset)
    if "stock_code" in mapped.columns:
        mapped["stock_code"] = normalize_stock_code(mapped["stock_code"])
    if "index_code" in mapped.columns:
        mapped["index_code"] = mapped["index_code"].astype(str).str.strip()

    if dataset == "trading_calendar":
        return _normalize_trading_calendar_frame(mapped)

    if dataset == "daily_market":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result["return_1d"] = result["return_with_dividend"]
        result["adj_close"] = result["comparable_close_with_dividend"]
        result["adj_open"] = result["open"] * result["adj_close"] / result["close"]
        result["float_market_cap"] = result["float_market_cap_thousand"] * 1000.0
        result["total_market_cap"] = result["total_market_cap_thousand"] * 1000.0
        result["is_limit_up"] = result["limit_status"] == 1
        result["is_limit_down"] = result["limit_status"] == -1
        result["is_trading"] = result["filling_flag"].astype(str).eq("0")
        result["is_st"] = result["trading_status_code"].isin(DAILY_ST_STATUS_CODES)
        result["is_pt"] = result["trading_status_code"].eq(16)
        result["is_a_share_scope"] = result["market_type"].isin(
            DAILY_A_SHARE_MARKET_TYPES
        )
        result = result[result["is_a_share_scope"]].copy()
        result["eligible_base"] = (
            result["is_a_share_scope"]
            & result["is_trading"]
            & ~result["is_st"]
            & ~result["is_pt"]
        )
        result["turnover"] = pd.NA
        result = a_share_csmar_normalize_daily_market_frame(result)
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "index_market":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result["return_1d"] = result["return_1d_pct"]
        result = a_share_csmar_normalize_index_time_frame(result)
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "adjustment":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = a_share_csmar_normalize_asset_time_frame(
            result,
            time_column="trade_date",
        )
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "market_cap":
        result = mapped.copy()
        result["share_change_date"] = pd.to_datetime(result["share_change_date"])
        result = a_share_csmar_normalize_asset_time_frame(
            result,
            time_column="share_change_date",
        )
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "stock_master":
        result = mapped.copy()
        for column in ("listing_date", "delisting_date", "status_change_date"):
            if column in result.columns:
                result[column] = pd.to_datetime(result[column], errors="coerce")
        status_change = (
            result["status_change_date"]
            if "status_change_date" in result.columns
            else pd.Series(pd.NaT, index=result.index)
        )
        result["asset_record_date"] = status_change.combine_first(
            result["listing_date"]
        )
        if result["asset_record_date"].isna().any():
            raise ValueError("missing_stock_master_effective_timestamp")
        result = a_share_csmar_normalize_asset_time_frame(
            result,
            time_column="asset_record_date",
        )
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "index_info":
        result = mapped.copy()
        if "index_class" not in result.columns and "index_category" in result.columns:
            result["index_class"] = result["index_category"]
        for column in ("base_date", "start_date"):
            if column in result.columns:
                result[column] = pd.to_datetime(result[column], errors="coerce")
        result = a_share_csmar_normalize_index_metadata_frame(result)
        if fallback_timestamp is not None:
            result = _fill_missing_canonical_time(result, fallback_timestamp)
        return result.sort_values(["asset_id"]).reset_index(drop=True)

    raise ValueError(f"unsupported_csmar_dataset: {dataset}")


def a_share_csmar_with_benchmark_index_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter index market rows to the approved benchmark pool and add metadata."""

    selected_codes = set(BENCHMARK_INDEX_POOL)
    result = frame[frame["source_asset_id"].isin(selected_codes)].copy()
    if result.empty:
        return result
    result["benchmark_name"] = result["source_asset_id"].map(
        lambda code: BENCHMARK_INDEX_POOL[code].name
    )
    result["benchmark_tier"] = result["source_asset_id"].map(
        lambda code: BENCHMARK_INDEX_POOL[code].tier
    )
    result = a_share_csmar_with_ohlc_valid(result)
    return result.sort_values(["benchmark_tier", "asset_id", "timestamp"])


def a_share_csmar_with_ohlc_valid(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an OHLC validity flag."""

    result = frame.copy()
    prices = result[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    result["ohlc_valid"] = (
        (prices["high"] >= prices[["open", "close"]].max(axis=1))
        & (prices["low"] <= prices[["open", "close"]].min(axis=1))
        & (prices["high"] >= prices["low"])
    )
    return result


def a_share_csmar_filter_by_date_range(
    frame: pd.DataFrame,
    *,
    dataset: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Filter a normalized frame by timestamp when bounds are provided."""

    if start_date is None and end_date is None:
        return frame
    column = _date_column(dataset)
    result = frame.copy()
    timestamps = pd.to_datetime(result[column], errors="coerce")
    mask = pd.Series(True, index=result.index)
    if start_date is not None:
        mask &= timestamps >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= timestamps <= pd.Timestamp(end_date)
    return result[mask].copy()


def a_share_csmar_build_interim_frame(
    *,
    interim_data_id: str,
    input_files: list[Path],
    start_date: str | None,
    end_date: str | None,
    overrides: dict[str, dict[str, object]] | None = None,
    chunksize: int = 500_000,
    pruning_reports: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Build one canonical A-share CSMAR interim frame without persisting output."""

    raw_dataset = _raw_dataset_for_interim(interim_data_id)
    frames: list[pd.DataFrame] = []
    for input_file in sorted(input_files):
        for chunk in iter_private_export_chunks(
            input_file,
            dataset=raw_dataset,
            chunksize=chunksize,
        ):
            normalized = a_share_csmar_normalize_private_export_frame(
                chunk,
                raw_dataset,
                fallback_timestamp=_fallback_timestamp_for_file(
                    input_file,
                    raw_dataset,
                ),
            )
            normalized = _filter_normalized_frame(
                normalized,
                interim_data_id=interim_data_id,
                start_date=start_date,
                end_date=end_date,
            )
            if interim_data_id == "benchmark_index_market":
                normalized = a_share_csmar_with_benchmark_index_fields(normalized)
            if normalized.empty:
                continue
            pruned, report = a_share_csmar_apply_interim_column_rules(
                normalized,
                interim_data_id,
                overrides=overrides,
            )
            if pruning_reports is not None:
                pruning_reports.append(report)
            frames.append(pruned)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if interim_data_id in ASSET_REFERENCE_INTERIM_DATASETS:
        result = _dedupe_reference_primary_key_rows(result, interim_data_id)
        if pruning_reports is not None:
            _append_reference_duplicate_summary(
                pruning_reports,
                interim_data_id,
                int(result.attrs["reference_exact_duplicate_count"]),
            )
    return result


def a_share_csmar_import_interim_data(
    *,
    layout: StorageLayout,
    raw_root: Path | None = None,
    interim_data_ids: tuple[str, ...] = DEFAULT_INTERIM_DATASETS,
    start_date: str | None = "2014-01-01",
    end_date: str | None = "2025-12-31",
    override_path: Path | None = None,
    chunksize: int = 500_000,
) -> AShareCSMARInterimImportResult:
    """Build and persist A-share CSMAR interim parquet datasets."""

    catalog = a_share_csmar_interim_catalog()
    source_raw_root = raw_root or layout.raw_source_root(SourceId(CSMAR_SOURCE_ID))
    overrides = (
        a_share_csmar_load_column_overrides(override_path) if override_path else None
    )
    written: dict[str, Path] = {}
    skipped: list[str] = []
    pruning_reports: list[dict[str, object]] = []

    for interim_data_id in interim_data_ids:
        catalog.get(interim_data_id)
        input_files = find_dataset_files(
            source_raw_root,
            _raw_dataset_for_interim(interim_data_id),
        )
        if not input_files:
            skipped.append(interim_data_id)
            continue
        frame = a_share_csmar_build_interim_frame(
            interim_data_id=interim_data_id,
            input_files=input_files,
            start_date=start_date,
            end_date=end_date,
            overrides=overrides,
            chunksize=chunksize,
            pruning_reports=pruning_reports,
        )
        if frame.empty:
            skipped.append(interim_data_id)
            continue
        written[interim_data_id] = write_interim_data(
            frame,
            layout=layout,
            catalog=catalog,
            interim_data_id=interim_data_id,
        )

    return AShareCSMARInterimImportResult(
        written_data_roots=written,
        skipped=tuple(skipped),
        pruning_report_text=a_share_csmar_render_column_pruning_report(
            pruning_reports,
        ),
    )


def _dedupe_reference_primary_key_rows(
    frame: pd.DataFrame,
    interim_data_id: str,
) -> pd.DataFrame:
    spec = a_share_csmar_interim_catalog().get(interim_data_id)
    primary_key = list(spec.primary_key)
    missing = [column for column in primary_key if column not in frame.columns]
    if missing:
        raise ValueError("missing_reference_primary_key_columns: " + ", ".join(missing))
    duplicated = frame.duplicated(primary_key, keep=False)
    if not bool(duplicated.any()):
        result = frame.copy()
        result.attrs["reference_exact_duplicate_count"] = 0
        return result

    ordered = frame.copy()
    ordered["_reference_row_order"] = range(len(ordered))
    exact_duplicate_count = 0
    for _, group in ordered.loc[duplicated].groupby(
        primary_key,
        sort=False,
        dropna=False,
    ):
        content = group.drop(columns=["_reference_row_order"])
        if len(content.drop_duplicates()) > 1:
            raise ValueError(
                f"conflicting_reference_primary_key_duplicates: {interim_data_id}"
            )
        exact_duplicate_count += len(group) - 1

    result = (
        ordered.sort_values(
            [*primary_key, "_reference_row_order"],
            kind="mergesort",
        )
        .drop_duplicates(primary_key, keep="last")
        .drop(columns=["_reference_row_order"])
        .reset_index(drop=True)
    )
    result.attrs["reference_exact_duplicate_count"] = int(exact_duplicate_count)
    return result


def _append_reference_duplicate_summary(
    reports: list[dict[str, object]],
    interim_data_id: str,
    exact_duplicate_count: int,
) -> None:
    reports.append(
        {
            "dataset": interim_data_id,
            "report_type": "reference_duplicate_summary",
            "reference_exact_duplicate_count": int(exact_duplicate_count),
        }
    )


def _has_multiple_effective_reference_records(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    identity_columns = list(ASSET_REFERENCE_INTERIM_KEEP_COLUMNS)
    if any(column not in frame.columns for column in (*identity_columns, "timestamp")):
        return False
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    grouped = frame.assign(_effective_timestamp=timestamps).groupby(
        identity_columns,
        dropna=False,
    )["_effective_timestamp"]
    return bool((grouped.nunique(dropna=True) > 1).any())


def _interim_spec(interim_data_id: str, columns: tuple[str, ...]) -> InterimDataSpec:
    if interim_data_id == "trading_calendar":
        primary_key = ("market_id", "source_id", "timestamp", "source_market_type")
        return InterimDataSpec(
            market_id=MarketId("a_share"),
            source_id=SourceId(CSMAR_SOURCE_ID),
            interim_data_id=interim_data_id,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.SOURCE_MARKET_CALENDAR_DATE,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if interim_data_id == "stock_master":
        primary_key = (
            "market_id",
            "asset_id",
            "source_id",
            "source_asset_id",
            "timestamp",
        )
        return InterimDataSpec(
            market_id=MarketId("a_share"),
            source_id=SourceId(CSMAR_SOURCE_ID),
            interim_data_id=interim_data_id,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=("year",),
        )
    if interim_data_id == "index_info":
        return a_share_csmar_reference_spec_for_frame(
            interim_data_id,
            pd.DataFrame(columns=list(columns)),
            force_time_aware=True,
        )
    if interim_data_id in ASSET_REFERENCE_INTERIM_DATASETS:
        primary_key = ASSET_REFERENCE_INTERIM_KEEP_COLUMNS
        return InterimDataSpec(
            market_id=MarketId("a_share"),
            source_id=SourceId(CSMAR_SOURCE_ID),
            interim_data_id=interim_data_id,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_REFERENCE,
            column_roles=_column_roles(columns, primary_key=primary_key),
            primary_key=primary_key,
            partition_columns=(),
        )
    return InterimDataSpec(
        market_id=MarketId("a_share"),
        source_id=SourceId(CSMAR_SOURCE_ID),
        interim_data_id=interim_data_id,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=_column_roles(columns, primary_key=platform_entity_time_key()),
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )


def _column_roles(
    columns: tuple[str, ...],
    *,
    primary_key: tuple[str, ...],
) -> dict[str, ColumnRole]:
    platform_catalog = platform_field_catalog()
    platform_columns = set(platform_interim_column_names())
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in primary_key and column not in {"timestamp", "calendar_date"}:
            roles[column] = ColumnRole.KEY
        elif column in platform_columns:
            roles[column] = platform_catalog.get(column).role
        else:
            roles[column] = csmar_field_role(column)
    return roles


def _normalize_trading_calendar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["calendar_date"] = pd.to_datetime(result["calendar_date"], errors="raise")
    result["source_market_type"] = pd.to_numeric(
        result["source_market_type"],
        errors="raise",
    ).astype("int64")
    result["source_day_of_week"] = pd.to_numeric(
        result["source_day_of_week"],
        errors="raise",
    ).astype("int64")
    result["source_state"] = result["source_state"].astype("string").str.strip()
    result["source_state"] = result["source_state"].str.upper()
    invalid = sorted(
        state
        for state in result["source_state"].dropna().unique()
        if state not in {"O", "C"}
    )
    if invalid or result["source_state"].isna().any():
        raise ValueError(f"invalid_trading_calendar_state: {invalid}")
    duplicate = result.duplicated(["source_market_type", "calendar_date"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate_trading_calendar_key")
    result["is_trading_day"] = result["source_state"].eq("O")
    result["market_id"] = "a_share"
    result["source_id"] = CSMAR_SOURCE_ID
    result["timestamp"] = result["calendar_date"]
    columns = [
        "market_id",
        "source_id",
        "timestamp",
        "calendar_date",
        "source_market_type",
        "is_trading_day",
        "source_day_of_week",
        "source_state",
    ]
    return (
        result[columns]
        .sort_values(["source_market_type", "calendar_date"])
        .reset_index(drop=True)
    )


def _date_column(dataset: str) -> str:
    if dataset in {
        "daily_market",
        "index_market",
        "adjustment",
        "market_cap",
        "stock_master",
        "index_info",
    }:
        return "timestamp"
    raise ValueError(f"dataset_has_no_filtering_date_column: {dataset}")


def _raw_dataset_for_interim(interim_data_id: str) -> str:
    if interim_data_id == "benchmark_index_market":
        return "index_market"
    return interim_data_id


def _fallback_timestamp_for_file(
    input_file: Path,
    raw_dataset: str,
) -> pd.Timestamp | None:
    if raw_dataset != "index_info":
        return None
    try:
        metadata = parse_raw_filename(input_file)
    except ValueError:
        return None
    return pd.Timestamp(metadata.export_date)


def _filter_normalized_frame(
    frame: pd.DataFrame,
    *,
    interim_data_id: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if interim_data_id in TIME_RANGE_FILTER_DATASETS:
        return a_share_csmar_filter_by_date_range(
            frame,
            dataset=_raw_dataset_for_interim(interim_data_id),
            start_date=start_date,
            end_date=end_date,
        )
    if interim_data_id in START_END_FILTER_DATASETS:
        return a_share_csmar_filter_by_date_range(
            frame,
            dataset=interim_data_id,
            start_date=start_date,
            end_date=end_date,
        )
    if interim_data_id in END_ONLY_FILTER_DATASETS:
        return a_share_csmar_filter_by_date_range(
            frame,
            dataset=interim_data_id,
            start_date=None,
            end_date=end_date,
        )
    return frame


def _fill_missing_canonical_time(
    frame: pd.DataFrame,
    fallback_timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    result = frame.copy()
    fallback = pd.Timestamp(fallback_timestamp).normalize()
    for column in ("timestamp", "session_date", "available_at"):
        parsed = pd.to_datetime(result[column], errors="coerce")
        result[column] = parsed.fillna(fallback)
    return result


__all__ = [
    "AShareCSMARInterimImportResult",
    "BENCHMARK_INDEX_POOL",
    "BenchmarkIndex",
    "DAILY_A_SHARE_MARKET_TYPES",
    "DAILY_ST_STATUS_CODES",
    "DEFAULT_INTERIM_DATASETS",
    "INDEX_METADATA_TIME_COLUMNS",
    "a_share_csmar_apply_interim_column_rules",
    "a_share_csmar_build_interim_frame",
    "a_share_csmar_default_override_payload",
    "a_share_csmar_filter_by_date_range",
    "a_share_csmar_import_interim_data",
    "a_share_csmar_interim_catalog",
    "a_share_csmar_load_column_overrides",
    "a_share_csmar_normalize_asset_time_frame",
    "a_share_csmar_normalize_daily_market_frame",
    "a_share_csmar_normalize_index_metadata_frame",
    "a_share_csmar_normalize_index_time_frame",
    "a_share_csmar_normalize_private_export_frame",
    "a_share_csmar_provider_binding",
    "a_share_csmar_reference_spec_for_frame",
    "a_share_csmar_render_column_pruning_report",
    "a_share_csmar_with_benchmark_index_fields",
    "a_share_csmar_with_ohlc_valid",
]
