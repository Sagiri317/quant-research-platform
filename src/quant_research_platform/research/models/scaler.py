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
    input_columns: tuple[str, ...] = ()
    medians: tuple[float, ...] = ()
    native_missing: bool = False
    scaling: bool = True

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
        missing: str | None = None,
        scaling: bool | None = None,
    ) -> ModelMatrixScaler:
        if missing not in {None, "native", "impute"}:
            raise ValueError("model_missing_strategy_invalid")
        if scaling is None:
            scaling = missing != "native"
        if type(scaling) is not bool:
            raise ValueError("model_training_scaling_invalid")
        columns = tuple(str(column) for column in frame.columns)
        input_columns, medians = (), ()
        if frame.empty or not columns or frame.columns.has_duplicates:
            raise ValueError("model_scaler_columns_invalid")
        if missing is not None:
            values = np.column_stack(
                [
                    strict_finite_numeric_series(
                        frame[column],
                        error=f"invalid_model_scaler_column: {column}",
                        allow_null=True,
                    ).to_numpy(dtype="float64")
                    for column in columns
                ]
            )
            if missing == "native":
                origins = [0.0] * len(columns)
                offsets = [Fraction()] * len(columns)
                scales = [1.0] * len(columns)
                weights = _weights(sample_weight, len(frame))
                if scaling:
                    for index, column in enumerate(columns):
                        observed = ~np.isnan(values[:, index])
                        # An all-missing native column has no fitted moments.
                        if observed.any():
                            fitted = cls.fit(
                                frame.loc[observed, [column]],
                                sample_weight=weights[observed],
                            )
                            origins[index] = fitted.origin_[0]
                            offsets[index] = fitted.offset_[0]
                            scales[index] = fitted.scale_[0]
                return cls(
                    columns,
                    tuple(origins),
                    tuple(offsets),
                    tuple(scales),
                    native_missing=True,
                    scaling=scaling,
                )
            if np.isnan(values).all(axis=0).any():
                raise ValueError("model_training_feature_all_missing")
            medians = []
            for index in range(len(columns)):
                observed = np.sort(values[~np.isnan(values[:, index]), index])
                middle = len(observed) // 2
                medians.append(
                    float(observed[middle])
                    if len(observed) % 2
                    else float(
                        (
                            Fraction(float(observed[middle - 1]))
                            + Fraction(float(observed[middle]))
                        )
                        / 2
                    )
                )
            medians = tuple(medians)
            if set(columns) & {f"{column}__missing" for column in columns}:
                raise ValueError("model_missing_indicator_name_collision")
            expanded = frame.fillna(dict(zip(columns, medians, strict=True)))
            for column in columns:
                expanded[f"{column}__missing"] = frame[column].isna().astype(float)
            input_columns = columns
            frame = expanded
            columns = tuple(expanded.columns)
        values = _finite_matrix(frame, columns)
        weights = _weights(sample_weight, len(values))
        if not scaling:
            return cls(
                columns,
                (0.0,) * len(columns),
                (Fraction(),) * len(columns),
                (1.0,) * len(columns),
                input_columns=input_columns,
                medians=medians,
                scaling=False,
            )
        original_weights = weights
        anchor = values[int(np.argmax(weights))]
        root_weights = np.sqrt(weights)
        root_weights /= root_weights.max()
        weights = weights / weights.max()
        total = float(weights.sum())
        with np.errstate(over="ignore"):
            normalized = values - anchor
        shifted = np.isfinite(normalized).all(axis=0)
        origin = np.where(shifted, anchor, 0.0)
        for column in np.flatnonzero(~shifted):
            normalized[:, column] = values[:, column]
        magnitude = np.max(np.abs(normalized), axis=0)
        constant = magnitude == 0.0
        magnitude[constant] = 1.0
        normalized /= magnitude
        weighted = weights[:, None] * normalized
        center = weighted.sum(axis=0) / total
        offset = center * magnitude
        # Preserve original-unit centers across underflow and anchor cancellation.
        exact_columns = ~constant & (
            np.array(
                [
                    (
                        (normalized[:, column] != 0)
                        & (np.abs(weighted[:, column]) < np.finfo(float).tiny)
                    ).any()
                    for column in range(len(columns))
                ]
            )
            | (np.abs(origin + offset) <= np.finfo(float).eps * np.abs(origin))
        )
        del weighted
        normalized_scale = np.array(
            [
                norm(
                    root_weights * (normalized[:, column] - center[column]),
                    check_finite=False,
                )
                for column in range(len(columns))
            ]
        ) / np.sqrt(total)
        del normalized
        # Values are in [-1, 1], so the population deviation cannot exceed one.
        scale = np.minimum(normalized_scale, 1.0) * magnitude
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
            input_columns=input_columns,
            medians=medians,
        )

    def input_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        if tuple(str(column) for column in frame.columns) != (
            self.input_columns or self.columns
        ):
            raise ValueError("model_scaler_columns_mismatch")
        if not self.medians:
            return frame
        result = frame.fillna(dict(zip(self.input_columns, self.medians, strict=True)))
        for column in self.input_columns:
            result[f"{column}__missing"] = frame[column].isna().astype(float)
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = self.input_matrix(frame)
        if self.native_missing:
            frame = pd.DataFrame(
                {
                    column: strict_finite_numeric_series(
                        frame[column],
                        error=f"invalid_model_scaler_column: {column}",
                        allow_null=True,
                    )
                    for column in self.columns
                },
                index=frame.index,
            ).astype("float64")
            if not self.scaling:
                return frame
        if tuple(str(column) for column in frame.columns) != self.columns:
            raise ValueError("model_scaler_columns_mismatch")
        values = (
            frame.to_numpy(dtype="float64")
            if self.native_missing
            else _finite_matrix(frame, self.columns)
        )
        origin = np.asarray(self.origin_)
        offset = np.array([float(value) for value in self.offset_])
        scale = np.asarray(self.scale_)
        with np.errstate(over="ignore"):
            differences = (values - origin) - offset
            overflow = ~np.isfinite(differences) & ~np.isnan(values)
            differences /= scale
        if overflow.any():
            rows, columns = np.nonzero(overflow)
            differences[rows, columns] = (
                values[rows, columns] / scale[columns]
                - origin[columns] / scale[columns]
                - offset[columns] / scale[columns]
            )
        for column, exact_offset in enumerate(self.offset_):
            residual = exact_offset - Fraction(float(offset[column]))
            if not residual:
                continue
            # One exact correction per column; ordinary rows remain vectorized.
            exact_scale = Fraction(self.scale_[column])
            try:
                correction = float(residual / exact_scale)
            except OverflowError:
                correction = np.inf if residual > 0 else -np.inf
            with np.errstate(over="ignore", invalid="ignore"):
                differences[:, column] -= correction
            for row in np.flatnonzero(
                ~np.isfinite(differences[:, column]) & ~np.isnan(values[:, column])
            ):
                standardized = (
                    Fraction(float(values[row, column]))
                    - Fraction(self.origin_[column])
                    - exact_offset
                ) / exact_scale
                try:
                    differences[row, column] = float(standardized)
                except OverflowError:
                    differences[row, column] = np.inf if standardized > 0 else -np.inf
        return pd.DataFrame(differences, index=frame.index, columns=self.columns)

    def to_payload(self) -> dict[str, object]:
        return {
            **(
                {
                    "input_columns": list(self.input_columns),
                    "medians": list(self.medians),
                }
                if self.medians
                else {}
            ),
            **({"native_missing": True} if self.native_missing else {}),
            "columns": list(self.columns),
            "scaling": self.scaling,
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
        predictions = np.asarray(
            self.estimator.predict(self.scaler.transform(frame)),
            dtype="float64",
        )
        predictions[frame.isna().all(axis=1)] = np.nan
        return predictions

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "estimator"), name)


@dataclass(frozen=True)
class LinearScaledPredictor:
    coefficients: tuple[float, ...]
    scaler: ModelMatrixScaler

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing_rows = frame.isna().all(axis=1).to_numpy()
        values = self.scaler.transform(frame).to_numpy(dtype="float64")
        frame = self.scaler.input_matrix(frame)
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
        predictions[missing_rows] = np.nan
        return predictions

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self.coefficients[1:], dtype="float64")


def equal_timestamp_sample_weights(
    timestamps: object,
    weighting: TrainingWeighting | None = None,
    *,
    trading_sessions=None,
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
    if trading_sessions is None:
        raise ValueError("training_decay_calendar_required")
    periods = pd.DatetimeIndex(pd.to_datetime(trading_sessions, errors="raise"))
    if (
        periods.has_duplicates
        or periods.isna().any()
        or not periods.is_monotonic_increasing
    ):
        raise ValueError("training_decay_calendar_invalid")
    indices = periods.get_indexer(values)
    if (indices < 0).any():
        raise ValueError("training_decay_calendar_coverage_missing")
    with np.errstate(over="ignore", under="ignore"):
        age = (indices.max() - indices) / weighting.half_life_periods
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
