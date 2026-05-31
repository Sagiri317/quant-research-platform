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

PRIVATE_PATH_MARKERS = (
    "raw_private",
    "interim_private",
    "processed_private",
    "private_real",
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
    train_end = pd.Timestamp(config.train_end)
    test_start = pd.Timestamp(config.test_start)
    if train_end >= test_start:
        raise ValueError("train_end must be before test_start.")

    train_mask = timestamps <= train_end
    test_mask = timestamps >= test_start
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
    if config.prediction_column in {
        config.timestamp_column,
        config.entity_column,
        config.label_column,
        *config.feature_columns,
    }:
        raise ValueError("prediction_column must be distinct from input columns.")
    if float(config.ridge_alpha) < 0:
        raise ValueError("ridge_alpha must be nonnegative.")
    _validate_portable_ref(config.source_ref, "source_ref")
    _validate_portable_ref(config.model_ref, "model_ref")
    _validate_output_path(config.output_prediction_path, "output_prediction_path")
    _validate_output_path(config.output_metadata_path, "output_metadata_path")
    if config.output_prediction_path == config.output_metadata_path:
        raise ValueError("prediction and metadata output paths must differ.")


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
        "label_column": config.label_column,
        "timestamp_column": config.timestamp_column,
        "entity_column": config.entity_column,
        "prediction_column": config.prediction_column,
        "train_end": str(pd.Timestamp(config.train_end).date()),
        "test_start": str(pd.Timestamp(config.test_start).date()),
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


def _validate_output_path(path: Path, name: str) -> None:
    normalized = path.as_posix().casefold()
    if any(marker in normalized for marker in PRIVATE_PATH_MARKERS):
        raise ValueError(f"{name} must not target private generated artifact paths.")


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


def _looks_like_private_or_local_path(value: str) -> bool:
    normalized = value.replace("\\", "/").casefold()
    if "/" in normalized:
        return True
    if re.match(r"^[a-z]:", normalized):
        return True
    return any(marker in normalized for marker in PRIVATE_PATH_MARKERS)


__all__ = [
    "BaselineTrainingConfig",
    "BaselineTrainingResult",
    "run_baseline_training",
]
