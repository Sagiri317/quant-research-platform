"""Aggregate-only safety helpers for A-share research batch runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from quant_research_platform.runtime.private_paths import (
    looks_like_private_or_local_path,
    text_has_private_marker,
)

KEY_COLUMNS = frozenset({"stock_code", "trade_date", "timestamp", "asset"})
RAW_PRICE_VOLUME_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_close",
        "prev_close",
        "amount",
        "volume",
        "turnover",
        "return_1d",
        "float_market_cap",
        "total_market_cap",
    }
)
TRADABILITY_COLUMNS = frozenset(
    {
        "eligible",
        "eligible_base",
        "can_enter",
        "can_exit",
        "is_trading",
        "is_st",
        "is_pt",
        "is_limit_up",
        "is_limit_down",
        "limit_status",
        "trading_status_code",
        "filling_flag",
    }
)
FILTER_COLUMNS = frozenset({"market_type", "is_a_share_scope"})
CONTEXT_COLUMNS = frozenset({"industry", "industry_code", "sector"})
LABEL_PREFIXES = ("forward_return", "target", "label")
FACTOR_PREFIXES = (
    "mom_",
    "rev_",
    "vol_",
    "downside_vol_",
    "upside_vol_",
    "max_return_",
    "min_return_",
    "drawdown_",
    "skew_",
    "kurt_",
    "avg_amount_",
    "turnover_",
    "turnover_chg_",
    "turnover_vol_",
    "amihud_",
    "zero_volume_ratio_",
    "amplitude_",
    "intraday_range_",
    "log_float_market_cap",
    "log_total_market_cap",
)


def build_schema_inventory(
    parquet_path: Path,
    *,
    expected_columns: tuple[str, ...] = (),
    key_columns: tuple[str, ...] = ("stock_code", "trade_date"),
    date_column: str = "trade_date",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Build aggregate-only schema inventory for a local Parquet table."""

    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    column_names = schema.names
    inventory: dict[str, Any] = {
        "schema_inventory_version": "a_share_schema_inventory_v1",
        "row_count": _count_rows(
            parquet_path,
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
            fallback=int(parquet_file.metadata.num_rows),
        ),
        "row_group_count": int(parquet_file.metadata.num_row_groups),
        "columns": [
            {
                "name": field.name,
                "dtype": str(field.type),
                "role": detect_column_role(field.name),
            }
            for field in schema
        ],
        "missing_expected_columns": [
            column for column in expected_columns if column not in column_names
        ],
        "unexpected_columns": [
            column
            for column in column_names
            if expected_columns and column not in expected_columns
        ],
        "duplicate_key_count": None,
        "date_range": None,
        "market_scope": None,
        "date_filter": _date_filter_payload(start_date=start_date, end_date=end_date),
    }
    aggregate_columns = [
        column
        for column in dict.fromkeys((*key_columns, date_column, "is_a_share_scope"))
        if column in column_names
    ]
    if aggregate_columns:
        frame = pd.read_parquet(
            parquet_path,
            columns=aggregate_columns,
            filters=_pandas_filters(
                date_column=date_column,
                start_date=start_date,
                end_date=end_date,
            ),
            engine="pyarrow",
        )
        present_keys = [column for column in key_columns if column in frame.columns]
        if present_keys:
            inventory["duplicate_key_count"] = int(frame.duplicated(present_keys).sum())
        if date_column in frame.columns:
            dates = pd.to_datetime(frame[date_column], errors="coerce")
            inventory["date_range"] = {
                "min": None if dates.dropna().empty else dates.min().date().isoformat(),
                "max": None if dates.dropna().empty else dates.max().date().isoformat(),
            }
        if "is_a_share_scope" in frame.columns:
            scope = frame["is_a_share_scope"]
            inventory["market_scope"] = {
                "field": "is_a_share_scope",
                "true_count": int((scope == True).sum()),  # noqa: E712
                "false_count": int((scope == False).sum()),  # noqa: E712
                "null_count": int(scope.isna().sum()),
            }
    return inventory


def run_memory_preflight(
    parquet_path: Path,
    *,
    required_columns: tuple[str, ...] = (),
    selected_columns: tuple[str, ...] | None = None,
    estimated_output_columns: int | None = None,
    date_column: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    memory_budget_gb: float = 10.0,
    safety_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Estimate memory from Parquet metadata before row-level reads."""

    parquet_file = pq.ParquetFile(parquet_path)
    columns = tuple(parquet_file.schema.names)
    if date_column is not None and (start_date or end_date):
        row_count = _count_rows(
            parquet_path,
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
            fallback=int(parquet_file.metadata.num_rows),
        )
    else:
        row_count = int(parquet_file.metadata.num_rows)
    column_count = len(columns)
    missing = [column for column in required_columns if column not in columns]
    selected_count = (
        len(tuple(dict.fromkeys(selected_columns)))
        if selected_columns is not None
        else column_count
    )
    operation_column_count = max(
        selected_count,
        int(estimated_output_columns or 0),
        1,
    )
    estimated_bytes = max(
        int(row_count * operation_column_count * 8 * safety_multiplier),
        int(parquet_file.metadata.serialized_size or 0),
    )
    budget_bytes = int(float(memory_budget_gb) * 1024**3)
    status = "passed"
    if missing or estimated_bytes > budget_bytes:
        status = "failed"
    return {
        "memory_preflight_version": "a_share_memory_preflight_v1",
        "status": status,
        "row_count": row_count,
        "column_count": column_count,
        "operation_column_count": operation_column_count,
        "required_columns": list(required_columns),
        "missing_required_columns": missing,
        "estimated_memory_bytes": estimated_bytes,
        "memory_budget_bytes": budget_bytes,
        "safety_multiplier": float(safety_multiplier),
        "date_filter": _date_filter_payload(
            start_date=start_date,
            end_date=end_date,
        ),
        "recommendation": (
            "proceed"
            if status == "passed"
            else "do not start heavy operation; use smoke fallback or reduce columns"
        ),
    }


def write_json_sanitized(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON only after rejecting private/local path leaks in payload text."""

    _validate_safe_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def detect_column_role(column: str) -> str:
    """Classify a column for aggregate schema inventory output."""

    normalized = column.casefold()
    if column in KEY_COLUMNS:
        return "key"
    if column in RAW_PRICE_VOLUME_COLUMNS:
        return "raw_price_volume"
    if any(normalized.startswith(prefix) for prefix in LABEL_PREFIXES):
        return "label_candidate"
    if normalized.startswith(FACTOR_PREFIXES):
        return "factor_candidate"
    if column in TRADABILITY_COLUMNS:
        return "tradability"
    if column in FILTER_COLUMNS:
        return "filter"
    if column in CONTEXT_COLUMNS:
        return "context"
    if normalized.startswith(("raw_", "vendor_")):
        return "excluded"
    return "unknown"


def _count_rows(
    parquet_path: Path,
    *,
    date_column: str,
    start_date: str | None,
    end_date: str | None,
    fallback: int,
) -> int:
    if not (start_date or end_date):
        return fallback
    try:
        dataset = ds.dataset(parquet_path, format="parquet")
        expression = _dataset_filter_expression(
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
        )
        return int(dataset.count_rows(filter=expression))
    except Exception:  # pragma: no cover - pyarrow filter fallback
        frame = pd.read_parquet(
            parquet_path,
            columns=(date_column,),
            filters=_pandas_filters(
                date_column=date_column,
                start_date=start_date,
                end_date=end_date,
            ),
            engine="pyarrow",
        )
        return int(len(frame))


def _dataset_filter_expression(
    *,
    date_column: str,
    start_date: str | None,
    end_date: str | None,
):
    expression = None
    if start_date:
        expression = ds.field(date_column) >= pd.Timestamp(start_date).to_pydatetime()
    if end_date:
        upper = ds.field(date_column) <= pd.Timestamp(end_date).to_pydatetime()
        expression = upper if expression is None else expression & upper
    return expression


def _pandas_filters(
    *,
    date_column: str,
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str, object]] | None:
    filters: list[tuple[str, str, object]] = []
    if start_date:
        filters.append((date_column, ">=", pd.Timestamp(start_date)))
    if end_date:
        filters.append((date_column, "<=", pd.Timestamp(end_date)))
    return filters or None


def _date_filter_payload(
    *,
    start_date: str | None,
    end_date: str | None,
) -> dict[str, str | None] | None:
    if not (start_date or end_date):
        return None
    return {"start_date": start_date, "end_date": end_date}


def _validate_safe_payload(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_safe_payload(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_safe_payload(item)
    elif isinstance(value, str):
        if text_has_private_marker(value) or looks_like_private_or_local_path(value):
            raise ValueError("unsafe private or local path in aggregate output")


__all__ = [
    "build_schema_inventory",
    "detect_column_role",
    "run_memory_preflight",
    "write_json_sanitized",
]
