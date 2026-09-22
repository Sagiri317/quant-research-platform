"""Training-only model matrix scaling and equal-timestamp weighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import TrainingWeighting
from quant_research_platform.contracts.validation import strict_finite_numeric_series


@dataclass(frozen=True)
class ModelMatrixScaler:
    """Weighted population scaler fitted only on the training matrix."""

    columns: tuple[str, ...]
    mean_: tuple[float, ...]
    scale_: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        sample_weight: object | None = None,
    ) -> ModelMatrixScaler:
        columns = tuple(str(column) for column in frame.columns)
        if frame.empty or not columns or frame.columns.has_duplicates:
            raise ValueError("model_scaler_columns_invalid")
        values = _finite_matrix(frame, columns)
        weights = _weights(sample_weight, len(values))
        weights = weights / weights.max()
        total = float(weights.sum())
        with np.errstate(over="ignore"):
            differences = values - values[0]
        shifted = np.isfinite(differences).all(axis=0)
        origin = np.where(shifted, values[0], 0.0)
        differences[:, ~shifted] = values[:, ~shifted]
        magnitude = np.max(np.abs(differences), axis=0)
        constant = magnitude == 0.0
        magnitude[constant] = 1.0
        normalized = differences / magnitude
        center = (weights[:, None] * normalized).sum(axis=0) / total
        mean = origin + center * magnitude
        variance = (weights[:, None] * (normalized - center) ** 2).sum(axis=0) / total
        scale = np.sqrt(variance) * magnitude
        scale[constant] = 1.0
        if np.any(scale == 0.0):
            raise ValueError("model_scaler_scale_unrepresentable")
        return cls(
            columns=columns,
            mean_=tuple(float(value) for value in mean),
            scale_=tuple(float(value) for value in scale),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(str(column) for column in frame.columns) != self.columns:
            raise ValueError("model_scaler_columns_mismatch")
        values = _finite_matrix(frame, self.columns)
        mean = np.asarray(self.mean_)
        scale = np.asarray(self.scale_)
        with np.errstate(over="ignore"):
            differences = values - mean
        scaled = differences / scale
        overflow = ~np.isfinite(differences)
        if overflow.any():
            rows, columns = np.nonzero(overflow)
            scaled[rows, columns] = (
                values[rows, columns] / scale[columns] - mean[columns] / scale[columns]
            )
        return pd.DataFrame(scaled, index=frame.index, columns=self.columns)

    def to_payload(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "weighted_mean": list(self.mean_),
            "weighted_population_scale": list(self.scale_),
        }


@dataclass
class ScaledPredictor:
    """Estimator wrapper that keeps the training-only scaler with the model."""

    estimator: object
    scaler: ModelMatrixScaler

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.estimator.predict(self.scaler.transform(frame)),
            dtype="float64",
        )

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "estimator"), name)


@dataclass(frozen=True)
class LinearScaledPredictor:
    coefficients: tuple[float, ...]
    scaler: ModelMatrixScaler

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.scaler.transform(frame).to_numpy(dtype="float64")
        coefficients = np.asarray(self.coefficients, dtype="float64")
        return coefficients[0] + values @ coefficients[1:]

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self.coefficients[1:], dtype="float64")


def equal_timestamp_sample_weights(
    timestamps: object, weighting: TrainingWeighting | None = None
) -> np.ndarray:
    """Assign equal timestamp mass, then decay and normalize to training rows."""

    weighting = TrainingWeighting() if weighting is None else weighting
    values = pd.Series(pd.to_datetime(timestamps, errors="raise"))
    if values.empty or values.isna().any():
        raise ValueError("sample_weight_timestamps_invalid")
    counts = values.groupby(values, sort=False).transform("size").to_numpy()
    timestamp_count = int(values.nunique())
    weights = len(values) / (timestamp_count * counts.astype("float64"))
    if weighting.half_life is not None:
        periods = pd.Index(values.unique()).sort_values()
        age = len(periods) - 1 - periods.get_indexer(values)
        weights *= np.exp2(-age / weighting.half_life)
        weights *= len(values) / weights.sum()
    return weights


def _finite_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = np.column_stack(
        [
            strict_finite_numeric_series(
                frame[column],
                error=f"invalid_model_scaler_column: {column}",
            ).to_numpy(dtype="float64", copy=False)
            for column in columns
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("model_scaler_values_invalid")
    return values


def _weights(sample_weight: object | None, row_count: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(row_count, dtype="float64")
    values = np.asarray(sample_weight, dtype="float64")
    if values.ndim != 1 or len(values) != row_count or not np.isfinite(values).all():
        raise ValueError("sample_weights_invalid")
    if (values <= 0.0).any():
        raise ValueError("sample_weights_invalid")
    return values
