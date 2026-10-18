"""Rank-ensemble model helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _average_ranks,
    _pearson_correlation_result,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)


@dataclass(frozen=True)
class RankEnsemblePredictor:
    columns: tuple[str, ...]
    directions: tuple[float, ...] = ()
    label_calibration: tuple[float, float] | None = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        directions = self.directions or (1.0,) * len(self.columns)
        scores = (
            _rank_frame_columns(frame, self.columns)
            @ np.asarray(directions, dtype="float64")
            / len(self.columns)
        )
        scores[frame.loc[:, self.columns].isna().all(axis=1)] = np.nan
        return scores


@dataclass(frozen=True)
class EqualWeightRankEnsembleFactory:
    columns: tuple[str, ...]
    label_column: str

    def fit(
        self, frame: pd.DataFrame, _target: np.ndarray, *, sample_weight=None
    ) -> RankEnsemblePredictor:
        directions = _rank_directions(
            frame,
            input_columns=self.columns,
            label_column=self.label_column,
            sample_weight=sample_weight,
        )
        return RankEnsemblePredictor(
            columns=self.columns,
            directions=directions,
        )


@dataclass(frozen=True)
class ICWeightedRankEnsembleFactory:
    columns: tuple[str, ...]
    label_column: str

    def fit(
        self, frame: pd.DataFrame, _target: np.ndarray, *, sample_weight=None
    ) -> ICWeightedRankEnsemble:
        return ICWeightedRankEnsemble.fit(
            frame,
            input_columns=self.columns,
            label_column=self.label_column,
            sample_weight=sample_weight,
        )


@dataclass(frozen=True)
class ICWeightedRankEnsemble:
    columns: tuple[str, ...]
    weights: tuple[float, ...]
    label_calibration: tuple[float, float] | None = None

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        input_columns: tuple[str, ...],
        label_column: str,
        sample_weight=None,
    ) -> ICWeightedRankEnsemble:
        weights = []
        defined_count = 0
        for _column, correlation in _training_rank_correlations(
            frame, input_columns, label_column, sample_weight
        ):
            weights.append(0.0 if correlation is None else correlation)
            defined_count += correlation is not None
        if not defined_count:
            raise ValueError("all_ensemble_correlations_undefined")
        normalizer = sum(map(abs, weights))
        if normalizer == 0.0:
            raise ValueError("all_ensemble_correlations_zero")
        columns = input_columns
        normalized = np.asarray(
            [float(value / normalizer) for value in weights], dtype="float64"
        )
        return cls(
            columns=columns,
            weights=tuple(float(value) for value in normalized),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ranks = _rank_frame_columns(frame, self.columns)
        scores = ranks @ np.asarray(self.weights, dtype="float64")
        scores[frame.loc[:, self.columns].isna().all(axis=1)] = np.nan
        return scores


def _label_predictions(scores, calibration):
    """Return label-scale values; callers retain raw scores for ordering."""
    return scores if calibration is None else calibration[0] + calibration[1] * scores


def rank_model_parameters(
    predictor: RankEnsemblePredictor | ICWeightedRankEnsemble,
) -> dict[str, list]:
    parameters = {"columns": list(predictor.columns)}
    if predictor.label_calibration is not None:
        parameters["label_calibration"] = list(predictor.label_calibration)
    key = "directions" if isinstance(predictor, RankEnsemblePredictor) else "weights"
    parameters[key] = list(getattr(predictor, key))
    return parameters


def _label_calibration(scores, labels, sample_weight) -> tuple[float, float]:
    from quant_research_platform.research.models.scaler import _weights

    weights = _weights(sample_weight, len(scores))
    labels = np.asarray(labels, dtype="float64")
    scores = np.asarray(scores, dtype="float64")
    mass = weights / weights.sum()
    dx, dy = scores - scores[0], labels - labels[0]
    xbar, ybar = np.dot(mass, dx), np.dot(mass, dy)
    x, y = dx - xbar, dy - ybar
    variance = np.dot(mass * x, x)
    slope = float(np.dot(mass * x, y) / variance) if variance > 0 else 0.0
    intercept = float(labels[0] + ybar - slope * (scores[0] + xbar))
    if not np.isfinite((intercept, slope)).all():
        raise ValueError("rank_label_calibration_nonfinite")
    return intercept, slope


def _rank_frame_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if "timestamp" in frame:
        timestamps = strict_timestamp_series(
            frame["timestamp"],
            error="invalid_rank_ensemble_timestamp",
        )
        ranked = _centered_rank_frame(frame, timestamps, columns)
    else:
        ranked = _centered_rank_frame(frame, None, columns)
    return ranked.to_numpy(dtype="float64", copy=False)


def _centered_rank_frame(
    frame: pd.DataFrame,
    timestamps: pd.Series | None,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=frame.index, columns=columns)
    for column in columns:
        values = strict_finite_numeric_series(
            frame[column],
            error=f"invalid_rank_ensemble_column: {column}",
            allow_null=True,
        )
        if timestamps is None:
            ranks = values.rank(method="average")
            count = values.count()
        else:
            grouped = values.groupby(timestamps, sort=False)
            ranks = grouped.rank(method="average")
            count = grouped.transform("count")
        centered = (ranks - (count + 1.0) / 2.0) / count
        # Preserve the original matrix layout and final dot-product reduction.
        result.loc[:, column] = centered.fillna(0.0)
    return result


def _rank_directions(
    frame: pd.DataFrame,
    *,
    input_columns: tuple[str, ...],
    label_column: str,
    sample_weight=None,
) -> tuple[float, ...]:
    directions = []
    for column, correlation in _training_rank_correlations(
        frame, input_columns, label_column, sample_weight
    ):
        if correlation is None:
            raise ValueError(
                f"ensemble_direction_undefined:{column}:no_valid_training_rankic"
            )
        directions.append(1.0 if correlation >= 0.0 else -1.0)
    return tuple(directions)


def _training_rank_correlations(frame, columns, label_column, sample_weight):
    from quant_research_platform.research.models.scaler import _weights

    weights = _weights(sample_weight, len(frame))
    timestamps = (
        strict_timestamp_series(
            frame["timestamp"], error="invalid_rank_ensemble_timestamp"
        )
        if "timestamp" in frame
        else None
    )
    groups = (
        frame.groupby(timestamps, sort=True).indices
        if timestamps is not None
        else {None: np.arange(len(frame))}
    )
    masses = (
        {
            timestamp: float(weights[indices].sum())
            for timestamp, indices in groups.items()
        }
        if timestamps is not None and sample_weight is not None
        else None
    )
    label_domains = {}
    for timestamp, indices in groups.items():
        labels = (
            frame[label_column]
            .array.take(indices)
            .to_numpy(dtype="float64", copy=False)
        )
        label_domains[timestamp] = labels, np.isfinite(labels)
    for column in columns:
        correlations, period_weights = [], []
        for timestamp, indices in groups.items():
            x = frame[column].array.take(indices).to_numpy(dtype="float64", copy=False)
            y, label_finite = label_domains[timestamp]
            valid = np.isfinite(x) & label_finite
            if int(valid.sum()) < 2:
                continue
            corr, _reason = _pearson_correlation_result(
                _average_ranks(x[valid]), _average_ranks(y[valid])
            )
            if corr is not None:
                correlations.append(corr)
                period_weights.append(
                    1.0
                    if sample_weight is None or masses is None
                    else masses[pd.Timestamp(timestamp)]
                )
        yield (
            column,
            float(np.average(correlations, weights=period_weights))
            if correlations
            else None,
        )
