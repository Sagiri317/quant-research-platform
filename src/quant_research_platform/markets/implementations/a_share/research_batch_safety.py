"""Aggregate-only safety helpers for A-share research batch runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
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
) -> dict[str, Any]:
    """Build aggregate-only schema inventory for a local Parquet table."""

    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    column_names = schema.names
    inventory: dict[str, Any] = {
        "schema_inventory_version": "a_share_schema_inventory_v1",
        "row_count": int(parquet_file.metadata.num_rows),
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
    memory_budget_gb: float = 10.0,
    safety_multiplier: float = 3.0,
) -> dict[str, Any]:
    """Estimate memory from Parquet metadata before row-level reads."""

    parquet_file = pq.ParquetFile(parquet_path)
    columns = tuple(parquet_file.schema.names)
    row_count = int(parquet_file.metadata.num_rows)
    column_count = len(columns)
    missing = [column for column in required_columns if column not in columns]
    estimated_bytes = max(
        int(row_count * max(column_count, 1) * 8 * safety_multiplier),
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
        "required_columns": list(required_columns),
        "missing_required_columns": missing,
        "estimated_memory_bytes": estimated_bytes,
        "memory_budget_bytes": budget_bytes,
        "safety_multiplier": float(safety_multiplier),
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
