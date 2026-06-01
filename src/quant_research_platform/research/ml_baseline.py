"""Local N6 ML baseline utility for prediction handoff smoke tests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.runtime.private_paths import (
    PRIVATE_PATH_MARKERS,
    looks_like_private_or_local_path,
    path_has_private_marker,
    require_private_outputs_for_private_input,
)

BOUNDARY_STATEMENT = (
    "local research baseline only; not ML platform; not model registry; "
    "not investment advice; not trading, not risk, not capacity, not execution "
    "simulation, not paper trading, and not live trading evidence"
)


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Input/output contract for the standalone local baseline trainer."""

    input_path: Path
    output_prediction_path: Path
    output_metadata_path: Path
    timestamp_column: str
    entity_column: str
    feature_columns: tuple[str, ...]
    label_column: str
    train_end: str
    test_start: str
    test_end: str | None = None
    embargo_days: int = 0
    fold_id: str | None = None
    split_name: str | None = None
    prediction_column: str = "prediction_score"
    ridge_alpha: float = 1e-6
    source_ref: str = "synthetic_or_local_feature_panel"
    model_ref: str = "numpy_linear_ridge_baseline_v1"


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Small in-memory summary of generated local baseline outputs."""

    prediction_path: Path
    metadata_path: Path
    prediction_column: str
    train_row_count: int
    test_row_count: int
    source_ref: str
    model_ref: str


@dataclass(frozen=True)
class _ResolvedSplit:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp | None


def run_baseline_training(config: BaselineTrainingConfig) -> BaselineTrainingResult:
    """Train a deterministic numpy ridge baseline and write test predictions."""

    _validate_config(config)
    input_columns = _input_columns(config.input_path)
    read_columns = _read_columns(input_columns, config)
    frame = pd.read_parquet(config.input_path, columns=read_columns)
    _validate_columns(frame, config)

    timestamps = pd.to_datetime(frame[config.timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{config.timestamp_column} must contain valid datetimes.")
    split = _resolve_split(config)

    train_mask = timestamps <= split.train_end
    test_mask = timestamps >= split.test_start
    if split.test_end is not None:
        test_mask &= timestamps <= split.test_end
    train = frame.loc[train_mask].copy()
    test = frame.loc[test_mask].copy()
    if train.empty:
        raise ValueError("train split must contain at least one row.")
    if test.empty:
        raise ValueError("test split must contain at least one row.")

    _validate_non_nulls(train, (*config.feature_columns, config.label_column), "train")
    _validate_non_nulls(test, config.feature_columns, "test")

    coefficients, intercept = _fit_ridge(
        train,
        feature_columns=config.feature_columns,
        label_column=config.label_column,
        ridge_alpha=float(config.ridge_alpha),
    )
    prediction_values = _predict(
        test,
        feature_columns=config.feature_columns,
        coefficients=coefficients,
        intercept=intercept,
    )
    prediction_frame = _prediction_frame(
        test,
        config=config,
        prediction_values=prediction_values,
    )

    config.output_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(
        config.output_prediction_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    _write_metadata(
        config,
        coefficients=coefficients,
        intercept=intercept,
        train_row_count=len(train),
        test_row_count=len(test),
    )
    return BaselineTrainingResult(
        prediction_path=config.output_prediction_path,
        metadata_path=config.output_metadata_path,
        prediction_column=config.prediction_column,
        train_row_count=len(train),
        test_row_count=len(test),
        source_ref=config.source_ref,
        model_ref=config.model_ref,
    )


def _validate_config(config: BaselineTrainingConfig) -> None:
    if not config.input_path.is_file():
        raise FileNotFoundError(config.input_path)
    _validate_column_names((config.timestamp_column,), "timestamp_column")
    _validate_column_names((config.entity_column,), "entity_column")
    _validate_column_names(config.feature_columns, "feature_columns")
    _validate_column_names((config.label_column,), "label_column")
    _validate_column_names((config.prediction_column,), "prediction_column")
    if config.label_column in config.feature_columns:
        raise ValueError("label_column must not be included in feature_columns.")
    leakage_columns = [
        column for column in config.feature_columns if _looks_like_label_leakage(column)
    ]
    if leakage_columns:
        raise ValueError(
            "feature_columns must not include leakage-prone label or "
            f"forward-return columns: {leakage_columns}."
        )
    if config.prediction_column in {
        config.timestamp_column,
        config.entity_column,
        config.label_column,
        *config.feature_columns,
    }:
        raise ValueError("prediction_column must be distinct from input columns.")
    if float(config.ridge_alpha) < 0:
        raise ValueError("ridge_alpha must be nonnegative.")
    if not isinstance(config.embargo_days, int) or isinstance(
        config.embargo_days,
        bool,
    ):
        raise TypeError("embargo_days must be an integer.")
    if config.embargo_days < 0:
        raise ValueError("embargo_days must be nonnegative.")
    _validate_optional_identifier(config.fold_id, "fold_id")
    _validate_optional_identifier(config.split_name, "split_name")
    _validate_portable_ref(config.source_ref, "source_ref")
    _validate_portable_ref(config.model_ref, "model_ref")
    _validate_output_policy(config)
    if config.output_prediction_path == config.output_metadata_path:
        raise ValueError("prediction and metadata output paths must differ.")
    _resolve_split(config)


def _validate_output_policy(config: BaselineTrainingConfig) -> None:
    validate_baseline_output_policy(
        input_path=config.input_path,
        output_prediction_path=config.output_prediction_path,
        output_metadata_path=config.output_metadata_path,
    )


def validate_baseline_output_policy(
    *,
    input_path: Path,
    output_prediction_path: Path,
    output_metadata_path: Path,
) -> None:
    """Validate baseline private input/output policy without reading input data."""

    require_private_outputs_for_private_input(
        input_path,
        (output_prediction_path, output_metadata_path),
        error_message=(
            "private input policy requires prediction and metadata outputs to "
            "use approved private output paths."
        ),
    )


def _resolve_split(config: BaselineTrainingConfig) -> _ResolvedSplit:
    train_end = _parse_timestamp(config.train_end, "train_end")
    test_start = _parse_timestamp(config.test_start, "test_start")
    test_end = (
        _parse_timestamp(config.test_end, "test_end")
        if config.test_end is not None
        else None
    )
    if train_end >= test_start:
        raise ValueError("train_end must be before test_start.")
    if test_end is not None and test_end < test_start:
        raise ValueError("test_end must be on or after test_start.")
    embargo_end = train_end + pd.Timedelta(days=config.embargo_days)
    if embargo_end > test_start:
        raise ValueError(
            "train_end plus embargo_days must be on or before test_start."
        )
    return _ResolvedSplit(train_end=train_end, test_start=test_start, test_end=test_end)


def _parse_timestamp(value: str, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date or timestamp.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid date or timestamp.")
    return timestamp


def _input_columns(path: Path) -> tuple[str, ...]:
    return tuple(pq.read_schema(path).names)


def _read_columns(
    input_columns: tuple[str, ...],
    config: BaselineTrainingConfig,
) -> list[str]:
    required_columns = (
        config.timestamp_column,
        config.entity_column,
        *config.feature_columns,
        config.label_column,
    )
    missing = sorted(set(required_columns).difference(input_columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}.")
    columns = list(dict.fromkeys(required_columns))
    if (
        "forward_return_1d" in input_columns
        and config.label_column != "forward_return_1d"
    ):
        columns.append("forward_return_1d")
    return columns


def _validate_columns(frame: pd.DataFrame, config: BaselineTrainingConfig) -> None:
    required_columns = (
        config.timestamp_column,
        config.entity_column,
        *config.feature_columns,
        config.label_column,
    )
    missing = sorted(set(required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}.")
    for column in (*config.feature_columns, config.label_column):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(f"{column} must be numeric.")


def _validate_column_names(values: tuple[str, ...], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one column.")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


def _looks_like_label_leakage(column: str) -> bool:
    normalized = column.casefold()
    return (
        normalized == "label"
        or normalized == "target"
        or normalized.startswith("label_")
        or normalized.startswith("target_")
        or normalized.startswith("forward_return")
    )


def _validate_non_nulls(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    split_name: str,
) -> None:
    missing_values = [
        column for column in columns if frame[column].isna().any()
    ]
    if missing_values:
        raise ValueError(
            f"{split_name} split contains nulls in required columns: "
            f"{missing_values}."
        )


def _fit_ridge(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    label_column: str,
    ridge_alpha: float,
) -> tuple[np.ndarray, float]:
    x = frame.loc[:, feature_columns].to_numpy(dtype=float)
    y = frame.loc[:, label_column].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    penalty = np.eye(design.shape[1], dtype=float) * ridge_alpha
    penalty[0, 0] = 0.0
    params = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return params[1:], float(params[0])


def _predict(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    coefficients: np.ndarray,
    intercept: float,
) -> np.ndarray:
    x = frame.loc[:, feature_columns].to_numpy(dtype=float)
    return x @ coefficients + intercept


def _prediction_frame(
    frame: pd.DataFrame,
    *,
    config: BaselineTrainingConfig,
    prediction_values: np.ndarray,
) -> pd.DataFrame:
    output_columns = [
        config.timestamp_column,
        config.entity_column,
        config.label_column,
    ]
    if (
        "forward_return_1d" in frame.columns
        and config.label_column != "forward_return_1d"
    ):
        output_columns.append("forward_return_1d")
    result = frame.loc[:, output_columns].copy()
    result.insert(2, config.prediction_column, prediction_values)
    if config.split_name is not None:
        result["split_name"] = config.split_name
    if config.fold_id is not None:
        result["fold_id"] = config.fold_id
    return result


def _write_metadata(
    config: BaselineTrainingConfig,
    *,
    coefficients: np.ndarray,
    intercept: float,
    train_row_count: int,
    test_row_count: int,
) -> None:
    metadata: dict[str, Any] = {
        "model_type": "numpy_linear_ridge_baseline",
        "model_ref": config.model_ref,
        "source_ref": config.source_ref,
        "feature_columns": list(config.feature_columns),
        "feature_summary": {
            "columns": list(config.feature_columns),
            "count": len(config.feature_columns),
        },
        "label_column": config.label_column,
        "target_column": config.label_column,
        "timestamp_column": config.timestamp_column,
        "entity_column": config.entity_column,
        "prediction_column": config.prediction_column,
        "train_end": str(pd.Timestamp(config.train_end).date()),
        "test_start": str(pd.Timestamp(config.test_start).date()),
        "test_end": (
            str(pd.Timestamp(config.test_end).date())
            if config.test_end is not None
            else None
        ),
        "embargo_days": config.embargo_days,
        "fold_id": config.fold_id,
        "split_name": config.split_name,
        "training_time": {
            "train_end": str(pd.Timestamp(config.train_end).date()),
            "embargo_days": config.embargo_days,
        },
        "prediction_time": {
            "test_start": str(pd.Timestamp(config.test_start).date()),
            "test_end": (
                str(pd.Timestamp(config.test_end).date())
                if config.test_end is not None
                else None
            ),
        },
        "train_row_count": train_row_count,
        "test_row_count": test_row_count,
        "coefficients": [float(value) for value in coefficients],
        "intercept": float(intercept),
        "ridge_alpha": float(config.ridge_alpha),
        "boundary": BOUNDARY_STATEMENT,
    }
    _validate_metadata_private_safe(metadata)
    config.output_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_portable_ref(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if _looks_like_private_or_local_path(value):
        raise ValueError(f"{name} must be a portable alias, not a path.")


def _validate_optional_identifier(value: str | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a portable identifier.")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", value):
        raise ValueError(f"{name} must be a portable identifier.")
    if _looks_like_private_or_local_path(value):
        raise ValueError(f"{name} must be a portable identifier.")


def _validate_metadata_private_safe(metadata: dict[str, Any]) -> None:
    text = json.dumps(metadata, ensure_ascii=False)
    lowered = text.casefold()
    forbidden = (
        *PRIVATE_PATH_MARKERS,
        "stock_sample",
        "row_sample",
        "sample_stocks",
        "sample_rows",
        "sample_assets",
    )
    leaked = [marker for marker in forbidden if marker in lowered]
    if leaked:
        raise ValueError(f"metadata contains private or row-level fields: {leaked}.")
    _validate_no_private_or_local_payload(metadata, "metadata")


def _validate_no_private_or_local_payload(value: Any, path: str) -> None:
    if isinstance(value, str):
        if _looks_like_private_or_local_path(value):
            raise ValueError(f"{path} must not contain private or machine-local paths.")
        return
    if isinstance(value, dict):
        for key, nested_value in value.items():
            normalized_key = str(key).casefold()
            if "sample" in normalized_key:
                raise ValueError(f"{path} must not include row-level sample fields.")
            _validate_no_private_or_local_payload(nested_value, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_no_private_or_local_payload(item, f"{path}[{index}]")


def _looks_like_private_or_local_path(value: str) -> bool:
    return looks_like_private_or_local_path(value)


def is_private_baseline_path(path: Path) -> bool:
    """Return whether a baseline input/output path is inside a private boundary."""

    return path_has_private_marker(path)


__all__ = [
    "BaselineTrainingConfig",
    "BaselineTrainingResult",
    "is_private_baseline_path",
    "run_baseline_training",
    "validate_baseline_output_policy",
]
