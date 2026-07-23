"""CSMAR raw export to source-normalized interim datasets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.validation import strict_string_series
from quant_research_platform.runtime.resources import (
    active_runtime_resource_policy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.field_mappings import (
    map_columns,
    normalize_stock_code,
)
from quant_research_platform.sources.csmar.fields import field_role
from quant_research_platform.sources.csmar.ids import SOURCE_ID, market_bars_data_id
from quant_research_platform.sources.csmar.raw_exports import (
    dataset_files_by_id,
    iter_private_export_chunks,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout

MARKET_BARS_DATA_ID = market_bars_data_id(DataFrequency.DAILY)
DEFAULT_INTERIM_DATA_IDS: tuple[str, ...] = (
    MARKET_BARS_DATA_ID,
    "stock_master",
    "adjustment",
    "market_cap",
    "index_info",
    "trading_calendar",
)
TIME_RANGE_FILTER_DATA_IDS = {MARKET_BARS_DATA_ID}
END_ONLY_FILTER_DATA_IDS = {"market_cap"}
START_END_FILTER_DATA_IDS = {"adjustment"}
SOURCE_REFERENCE_DATA_IDS = {"stock_master", "index_info"}

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
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
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
TRADING_CALENDAR_INTERIM_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "calendar_date",
    "source_market_type",
    "is_trading_day",
    "source_day_of_week",
    "source_state",
)
INTERIM_COLUMNS_BY_DATA_ID: dict[str, tuple[str, ...]] = {
    MARKET_BARS_DATA_ID: MARKET_BARS_INTERIM_COLUMNS,
    "stock_master": STOCK_MASTER_INTERIM_COLUMNS,
    "market_cap": MARKET_CAP_INTERIM_COLUMNS,
    "adjustment": ADJUSTMENT_INTERIM_COLUMNS,
    "index_info": INDEX_INFO_INTERIM_COLUMNS,
    "trading_calendar": TRADING_CALENDAR_INTERIM_COLUMNS,
}
AUTO_DROPPED_COLUMNS: dict[str, dict[str, str]] = {
    MARKET_BARS_DATA_ID: {
        "return_with_dividend": "derived as period_return",
        "comparable_close_with_dividend": "derived as adj_close",
        "float_market_cap_thousand": "converted to float_market_cap",
        "total_market_cap_thousand": "converted to total_market_cap",
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


def load_column_overrides(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return default_override_payload()
    return json.loads(path.read_text(encoding="utf-8"))


def apply_output_column_rules(
    frame: pd.DataFrame,
    data_id: str,
    *,
    overrides: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base_keep = list(INTERIM_COLUMNS_BY_DATA_ID.get(data_id, tuple(frame.columns)))
    override = (overrides or {}).get(data_id, {})
    manual_keep = [str(column) for column in override.get("keep", [])]
    manual_drop = [str(column) for column in override.get("drop", [])]

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
        if report.get("report_type") == "reference_duplicate_summary":
            lines.extend(
                [
                    f"## {report['data_id']} reference duplicate summary",
                    "",
                    "- exact duplicate primary-key rows deduped: "
                    f"{report['reference_exact_duplicate_count']}",
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
    catalog = InterimDataCatalog()
    for spec in interim_specs():
        catalog.register(spec)
    return catalog


def normalize_raw_frame(
    frame: pd.DataFrame,
    data_id: str,
) -> pd.DataFrame:
    mapped = map_columns(frame, _raw_dataset_for_interim(data_id))
    if "stock_code" in mapped.columns:
        mapped["stock_code"] = normalize_stock_code(mapped["stock_code"])
    if "index_code" in mapped.columns:
        mapped["index_code"] = strict_string_series(
            mapped["index_code"],
            error="invalid_csmar_index_code",
        )

    if data_id == "trading_calendar":
        return _normalize_trading_calendar_frame(mapped)
    if data_id == MARKET_BARS_DATA_ID:
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result["timestamp"] = result["trade_date"]
        result["source_asset_id"] = result["stock_code"]
        result["period_return"] = result["return_with_dividend"]
        result["adj_close"] = result["comparable_close_with_dividend"]
        result["adj_open"] = result["open"] * result["adj_close"] / result["close"]
        result["float_market_cap"] = result["float_market_cap_thousand"] * 1000.0
        result["total_market_cap"] = result["total_market_cap_thousand"] * 1000.0
        result["turnover"] = pd.Series(
            float("nan"),
            index=result.index,
            dtype="float64",
        )
        return result.sort_values(["source_asset_id", "timestamp"]).reset_index(
            drop=True
        )
    if data_id == "adjustment":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result["timestamp"] = result["trade_date"]
        result["source_asset_id"] = result["stock_code"]
        return result.sort_values(["source_asset_id", "timestamp"]).reset_index(
            drop=True
        )
    if data_id == "market_cap":
        result = mapped.copy()
        result["share_change_date"] = pd.to_datetime(result["share_change_date"])
        result["timestamp"] = result["share_change_date"]
        result["source_asset_id"] = result["stock_code"]
        return result.sort_values(["source_asset_id", "timestamp"]).reset_index(
            drop=True
        )
    if data_id == "stock_master":
        result = mapped.copy()
        for column in ("listing_date", "delisting_date", "status_change_date"):
            if column in result.columns:
                result[column] = pd.to_datetime(result[column], errors="coerce")
        status_change = (
            result["status_change_date"]
            if "status_change_date" in result.columns
            else pd.Series(pd.NaT, index=result.index)
        )
        result["timestamp"] = status_change.combine_first(result["listing_date"])
        if result["timestamp"].isna().any():
            raise ValueError("missing_stock_master_effective_timestamp")
        result["source_asset_id"] = result["stock_code"]
        return result.sort_values(["source_asset_id", "timestamp"]).reset_index(
            drop=True
        )
    if data_id == "index_info":
        result = mapped.copy()
        if "index_class" not in result.columns and "index_category" in result.columns:
            result["index_class"] = result["index_category"]
        if "base_point" in result.columns:
            base_point = pd.to_numeric(result["base_point"], errors="coerce")
            result["base_point"] = base_point.mask(base_point.eq(-9666))
        for column in ("base_date", "start_date"):
            if column in result.columns:
                result[column] = _parse_optional_index_info_date(result[column], column)
        result["timestamp"] = result["start_date"].combine_first(result["base_date"])
        if result["timestamp"].isna().any():
            raise ValueError("missing_index_info_effective_timestamp")
        result["source_asset_id"] = result["index_code"]
        return result.sort_values(["source_asset_id", "timestamp"]).reset_index(
            drop=True
        )
    raise ValueError(f"unsupported_csmar_data_id: {data_id}")


def _parse_optional_index_info_date(series: pd.Series, column: str) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text.mask(text.eq("")), errors="coerce")
    invalid = text.notna() & text.ne("") & parsed.isna()
    if bool(invalid.any()):
        raise ValueError(f"invalid_index_info_date: {column}")
    return parsed


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
    result = frame.copy()
    timestamps = pd.to_datetime(result[column], errors="coerce")
    mask = pd.Series(True, index=result.index)
    if start_date is not None:
        mask &= timestamps >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= timestamps <= pd.Timestamp(end_date)
    return result[mask].copy()


def iter_interim_frames(
    *,
    data_id: str,
    input_files: list[Path],
    start_date: str | None,
    end_date: str | None,
    overrides: dict[str, dict[str, object]] | None = None,
    chunksize: int = 500_000,
    pruning_reports: list[dict[str, object]] | None = None,
) -> Iterator[pd.DataFrame]:
    bound_policy = current_runtime_resource_policy()

    def activate() -> Iterator[pd.DataFrame]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            raw_dataset = _raw_dataset_for_interim(data_id)
            frames = _iter_pruned_interim_frames(
                data_id=data_id,
                raw_dataset=raw_dataset,
                input_files=input_files,
                start_date=start_date,
                end_date=end_date,
                overrides=overrides,
                chunksize=chunksize,
                pruning_reports=pruning_reports,
            )
            if data_id in SOURCE_REFERENCE_DATA_IDS:
                reference_frames = _iter_deduped_reference_frames(
                    frames, data_id=data_id, pruning_reports=pruning_reports
                )
                with closing_dataset_iterator(reference_frames):
                    for frame in reference_frames:
                        yield frame
                return
            with closing_dataset_iterator(frames):
                for frame in frames:
                    yield frame

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
) -> InterimImportResult:
    catalog = interim_catalog()
    source_raw_root = raw_root or layout.raw_source_root(SourceId(SOURCE_ID))
    overrides = load_column_overrides(override_path) if override_path else None
    written: dict[str, Path] = {}
    skipped: list[str] = []
    pruning_reports: list[dict[str, object]] = []
    raw_files_by_data_id = dataset_files_by_id(source_raw_root)

    for data_id in data_ids:
        spec = catalog.get(data_id)
        input_files = list(
            raw_files_by_data_id.get(_raw_dataset_for_interim(data_id), ())
        )
        if not input_files:
            skipped.append(data_id)
            continue
        artifact_root = layout.interim_data_root(spec.source_id, spec.data_id)
        frames = iter_interim_frames(
            data_id=data_id,
            input_files=input_files,
            start_date=start_date,
            end_date=end_date,
            overrides=overrides,
            chunksize=chunksize,
            pruning_reports=pruning_reports,
        )
        try:
            first_frame = next(frames)
        except StopIteration:
            skipped.append(data_id)
            continue
        write_dataset(artifact_root, _iter_with_first(first_frame, frames), spec=spec)
        written[data_id] = artifact_root

    return InterimImportResult(
        artifact_roots=written,
        skipped=tuple(skipped),
        pruning_report_text=render_column_pruning_report(pruning_reports),
    )


def _iter_pruned_interim_frames(
    *,
    data_id: str,
    raw_dataset: str,
    input_files: list[Path],
    start_date: str | None,
    end_date: str | None,
    overrides: dict[str, dict[str, object]] | None,
    chunksize: int,
    pruning_reports: list[dict[str, object]] | None,
) -> Iterator[pd.DataFrame]:
    for input_file in sorted(input_files):
        for chunk in iter_private_export_chunks(
            input_file,
            dataset=raw_dataset,
            chunksize=chunksize,
        ):
            normalized = normalize_raw_frame(chunk, data_id)
            normalized = _filter_normalized_frame(
                normalized,
                data_id=data_id,
                start_date=start_date,
                end_date=end_date,
            )
            if normalized.empty:
                continue
            pruned, report = apply_output_column_rules(
                normalized,
                data_id,
                overrides=overrides,
            )
            if pruning_reports is not None:
                pruning_reports.append(report)
            yield pruned


def _iter_deduped_reference_frames(
    frames: Iterator[pd.DataFrame],
    *,
    data_id: str,
    pruning_reports: list[dict[str, object]] | None,
) -> Iterator[pd.DataFrame]:
    spec = interim_catalog().get(data_id)
    primary_key = list(spec.primary_key)
    seen_rows: dict[tuple[object, ...], tuple[object, ...]] = {}
    exact_duplicate_count = 0
    try:
        with closing_dataset_iterator(frames):
            for frame in frames:
                missing = [
                    column for column in primary_key if column not in frame.columns
                ]
                if missing:
                    raise ValueError(
                        "missing_reference_primary_key_columns: " + ", ".join(missing)
                    )
                keep_positions: list[int] = []
                for position, row in enumerate(
                    frame.itertuples(index=False, name=None)
                ):
                    columns = tuple(str(column) for column in frame.columns)
                    values = dict(zip(columns, row, strict=True))
                    key = tuple(values[column] for column in primary_key)
                    payload = tuple(values[column] for column in columns)
                    previous = seen_rows.get(key)
                    if previous is None:
                        seen_rows[key] = payload
                        keep_positions.append(position)
                        continue
                    if previous != payload:
                        raise ValueError(
                            f"conflicting_reference_primary_key_duplicates: {data_id}"
                        )
                    exact_duplicate_count += 1
                if keep_positions:
                    yield frame.iloc[keep_positions].reset_index(drop=True)
    finally:
        if pruning_reports is not None:
            _append_reference_duplicate_summary(
                pruning_reports,
                data_id,
                exact_duplicate_count,
            )


def _append_reference_duplicate_summary(
    reports: list[dict[str, object]],
    data_id: str,
    exact_duplicate_count: int,
) -> None:
    reports.append(
        {
            "data_id": data_id,
            "report_type": "reference_duplicate_summary",
            "reference_exact_duplicate_count": int(exact_duplicate_count),
        }
    )


def _interim_spec(data_id: str, columns: tuple[str, ...]) -> InterimDataSpec:
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
    result["timestamp"] = result["calendar_date"]
    return (
        result[list(TRADING_CALENDAR_INTERIM_COLUMNS)]
        .sort_values(["source_market_type", "calendar_date"])
        .reset_index(drop=True)
    )


def _date_column(data_id: str) -> str:
    if data_id in {
        MARKET_BARS_DATA_ID,
        "adjustment",
        "market_cap",
        "stock_master",
        "trading_calendar",
    }:
        return "timestamp"
    raise ValueError(f"data_id_has_no_filtering_date_column: {data_id}")


def _raw_dataset_for_interim(data_id: str) -> str:
    return data_id


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


def _iter_with_first(
    first_frame: pd.DataFrame,
    frames: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    yield first_frame
    yield from frames


__all__ = [
    "InterimImportResult",
    "DEFAULT_INTERIM_DATA_IDS",
    "apply_output_column_rules",
    "default_override_payload",
    "filter_by_source_date_range",
    "interim_catalog",
    "interim_specs",
    "iter_interim_frames",
    "load_column_overrides",
    "materialize_interim_data",
    "normalize_raw_frame",
    "render_column_pruning_report",
]
