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
    input_columns: tuple[str, ...] = ()
    medians: tuple[float, ...] = ()
    native_missing: bool = False
    scaling: bool = True

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
            values = []
            for column in columns:
                parsed = strict_finite_numeric_series(
                    frame[column],
                    error=f"invalid_model_scaler_column: {column}",
                    allow_null=True,
                )
                if missing != "native" or scaling:
                    values.append(parsed.to_numpy(dtype="float64"))
            if missing == "native":
                means = [0.0] * len(columns)
                scales = [1.0] * len(columns)
                weights = _weights(sample_weight, len(frame))
                if scaling:
                    for index, column_values in enumerate(values):
                        column = columns[index]
                        observed = ~np.isnan(column_values)
                        # An all-missing native column has no fitted moments.
                        if observed.any():
                            fitted = cls.fit(
                                frame.loc[observed, [column]],
                                sample_weight=weights[observed],
                            )
                            means[index] = fitted.mean_[0]
                            scales[index] = fitted.scale_[0]
                return cls(
                    columns,
                    tuple(means),
                    tuple(scales),
                    native_missing=True,
                    scaling=scaling,
                )
            values = np.column_stack(values)
            if np.isnan(values).all(axis=0).any():
                raise ValueError("model_training_feature_all_missing")
            medians = tuple(float(value) for value in np.nanmedian(values, axis=0))
            if set(columns) & {f"{column}__missing" for column in columns}:
                raise ValueError("model_missing_indicator_name_collision")
            expanded = pd.DataFrame(values, index=frame.index, columns=columns).fillna(
                dict(zip(columns, medians, strict=True))
            )
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
                (1.0,) * len(columns),
                input_columns=input_columns,
                medians=medians,
                scaling=False,
            )
        weights = weights / weights.sum()
        anchor = values[0].copy()
        values -= anchor
        center = weights @ values
        mean = anchor + center
        values -= center
        np.square(values, out=values)
        scale = np.sqrt(weights @ values)
        scale[scale == 0.0] = 1.0
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("model_scaler_moments_nonfinite")
        return cls(
            columns=columns,
            mean_=tuple(float(value) for value in mean),
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
        result = pd.DataFrame(
            {
                column: strict_finite_numeric_series(
                    frame[column],
                    error=f"invalid_model_scaler_column: {column}",
                    allow_null=True,
                ).to_numpy(dtype="float64")
                for column in self.input_columns
            },
            index=frame.index,
        ).fillna(dict(zip(self.input_columns, self.medians, strict=True)))
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
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = (values - np.asarray(self.mean_)) / np.asarray(self.scale_)
        return pd.DataFrame(transformed, index=frame.index, columns=self.columns)

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
            "weighted_mean": list(self.mean_),
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
        coefficients = np.asarray(self.coefficients, dtype="float64")
        with np.errstate(over="ignore", invalid="ignore"):
            predictions = coefficients[0] + values @ coefficients[1:]
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
