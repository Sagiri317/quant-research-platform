"""Training-only model matrix scaling and equal-timestamp weighting."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.linalg import norm

from quant_research_platform.contracts.models import TrainingWeighting
from quant_research_platform.contracts.validation import strict_finite_numeric_series


@dataclass(frozen=True)
class ModelMatrixScaler:
    """Weighted population scaler fitted only on the training matrix."""

    columns: tuple[str, ...]
    origin_: tuple[float, ...]
    offset_: tuple[Fraction, ...]
    scale_: tuple[float, ...]

    @property
    def mean_(self) -> tuple[float, ...]:
        return tuple(
            float(Fraction(origin) + offset)
            for origin, offset in zip(self.origin_, self.offset_, strict=True)
        )

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
        original_weights = weights
        anchor = values[int(np.argmax(weights))]
        root_weights = np.sqrt(weights)
        root_weights /= root_weights.max()
        weights = weights / weights.max()
        total = float(weights.sum())
        with np.errstate(over="ignore"):
            differences = values - anchor
        shifted = np.isfinite(differences).all(axis=0)
        origin = np.where(shifted, anchor, 0.0)
        differences[:, ~shifted] = values[:, ~shifted]
        magnitude = np.max(np.abs(differences), axis=0)
        constant = magnitude == 0.0
        magnitude[constant] = 1.0
        normalized = differences / magnitude
        weighted = weights[:, None] * normalized
        center = weighted.sum(axis=0) / total
        offset = center * magnitude
        normalized_scale = np.array(
            [
                norm(
                    root_weights * (normalized[:, column] - center[column]),
                    check_finite=False,
                )
                for column in range(len(columns))
            ]
        ) / np.sqrt(total)
        # Values are in [-1, 1], so the population deviation cannot exceed one.
        scale = np.minimum(normalized_scale, 1.0) * magnitude
        # Preserve original-unit centers across underflow and anchor cancellation.
        exact_columns = ~constant & (
            ((normalized != 0) & (np.abs(weighted) < np.finfo(float).tiny)).any(axis=0)
            | (np.abs(origin + offset) <= np.finfo(float).eps * np.abs(origin))
        )
        exact_offsets = [Fraction(float(value)) for value in offset]
        if exact_columns.any():
            exact_weights = tuple(
                Fraction(float(weight)) for weight in original_weights
            )
            exact_total = sum(exact_weights)
            for column in np.flatnonzero(exact_columns):
                exact_mean = (
                    sum(
                        weight * Fraction(float(value))
                        for weight, value in zip(
                            exact_weights, values[:, column], strict=True
                        )
                    )
                    / exact_total
                )
                exact_offset = exact_mean - Fraction(float(origin[column]))
                if abs(exact_mean) < abs(exact_offset):
                    origin[column] = 0.0
                    exact_offset = exact_mean
                exact_offsets[column] = exact_offset
        scale[constant] = 1.0
        if np.any(scale == 0.0):
            raise ValueError("model_scaler_scale_unrepresentable")
        return cls(
            columns=columns,
            origin_=tuple(float(value) for value in origin),
            offset_=tuple(exact_offsets),
            scale_=tuple(float(value) for value in scale),
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(str(column) for column in frame.columns) != self.columns:
            raise ValueError("model_scaler_columns_mismatch")
        values = _finite_matrix(frame, self.columns)
        origin = np.asarray(self.origin_)
        offset = np.array([float(value) for value in self.offset_])
        scale = np.asarray(self.scale_)
        with np.errstate(over="ignore"):
            differences = (values - origin) - offset
            overflow = ~np.isfinite(differences)
            differences /= scale
        if overflow.any():
            rows, columns = np.nonzero(overflow)
            differences[rows, columns] = (
                values[rows, columns] / scale[columns]
                - origin[columns] / scale[columns]
                - offset[columns] / scale[columns]
            )
        for column, exact_offset in enumerate(self.offset_):
            if exact_offset != Fraction(float(offset[column])):
                exact_origin = Fraction(self.origin_[column])
                exact_scale = Fraction(self.scale_[column])
                for row, value in enumerate(values[:, column]):
                    standardized = (
                        Fraction(float(value)) - exact_origin - exact_offset
                    ) / exact_scale
                    try:
                        differences[row, column] = float(standardized)
                    except OverflowError:
                        differences[row, column] = (
                            np.inf if standardized > 0 else -np.inf
                        )
        return pd.DataFrame(differences, index=frame.index, columns=self.columns)

    def to_payload(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "weighted_mean_origin": list(self.origin_),
            "weighted_mean_offset": [
                [value.numerator, value.denominator] for value in self.offset_
            ],
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
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = coefficients[0] + values @ coefficients[1:]
        # A subnormal standardized value can lose a representable final affine term.
        underflow = (
            (np.abs(values) < np.finfo(float).tiny)
            & (coefficients[1:] != 0)
            & (
                (frame.to_numpy(dtype="float64") != np.asarray(self.scaler.origin_))
                | np.array([offset != 0 for offset in self.scaler.offset_])
            )
        ).any(axis=1)
        for row in np.flatnonzero(~np.isfinite(predictions) | underflow):
            exact = Fraction(self.coefficients[0]) + sum(
                Fraction(coefficient)
                * (Fraction(float(value)) - Fraction(origin) - offset)
                / Fraction(scale)
                for coefficient, value, origin, offset, scale in zip(
                    self.coefficients[1:],
                    frame.iloc[row],
                    self.scaler.origin_,
                    self.scaler.offset_,
                    self.scaler.scale_,
                    strict=True,
                )
            )
            try:
                predictions[row] = float(exact)
            except OverflowError:
                predictions[row] = np.inf if exact > 0 else -np.inf
        return predictions

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
    if weighting.half_life_periods is None:
        return len(values) / (timestamp_count * counts.astype("float64"))
    periods = pd.Index(values.unique()).sort_values()
    with np.errstate(over="ignore", under="ignore"):
        age = (
            len(periods) - 1 - periods.get_indexer(values)
        ) / weighting.half_life_periods
        total_mass = float((np.exp2(-age) / counts).sum())
        weights = np.exp2(np.log2(len(values) / total_mass) - np.log2(counts) - age)
    if (weights <= 0.0).any():
        raise ValueError("sample_weights_unrepresentable")
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
