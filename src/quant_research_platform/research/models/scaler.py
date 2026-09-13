"""Training-only model matrix scaling and equal-date weighting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
        total = float(weights.sum())
        mean = (weights[:, None] * values).sum(axis=0) / total
        constant = np.ptp(values, axis=0) == 0.0
        mean[constant] = values[0, constant]
        variance = (weights[:, None] * (values - mean) ** 2).sum(axis=0) / total
        scale = np.sqrt(variance)
        scale[scale == 0.0] = 1.0
        return cls(
            columns=columns,
            mean_=tuple(float(value) for value in mean),
            scale_=tuple(float(value) for value in scale),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(str(column) for column in frame.columns) != self.columns:
            raise ValueError("model_scaler_columns_mismatch")
        values = _finite_matrix(frame, self.columns)
        scaled = (values - np.asarray(self.mean_)) / np.asarray(self.scale_)
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


def equal_date_sample_weights(timestamps: object) -> np.ndarray:
    """Return weights whose date totals are equal and whose sum is row count."""

    values = pd.Series(pd.to_datetime(timestamps, errors="raise"))
    if values.empty or values.isna().any():
        raise ValueError("sample_weight_timestamps_invalid")
    counts = values.groupby(values, sort=False).transform("size").to_numpy()
    date_count = int(values.nunique())
    return len(values) / (date_count * counts.astype("float64"))


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
    if (values <= 0.0).any() or float(values.sum()) <= 0.0:
        raise ValueError("sample_weights_invalid")
    return values
