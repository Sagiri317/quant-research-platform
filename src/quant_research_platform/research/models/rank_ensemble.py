"""Rank-ensemble model helpers."""

from __future__ import annotations

from collections.abc import Iterator
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

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        directions = self.directions or (1.0,) * len(self.columns)
        return (
            _rank_frame_columns(frame, self.columns)
            @ np.asarray(directions, dtype="float64")
            / len(self.columns)
        )


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
        defined_columns: list[str] = []
        for column, correlation in _training_rank_correlations(
            frame, input_columns, label_column, sample_weight
        ):
            if correlation is not None:
                defined_columns.append(column)
                weights.append(correlation)
        if not weights:
            raise ValueError("all_ensemble_correlations_undefined")
        values = np.asarray(weights, dtype="float64")
        normalizer = float(np.sum(np.abs(values)))
        if normalizer == 0.0:
            raise ValueError("all_ensemble_correlations_zero")
        return cls(
            columns=tuple(defined_columns),
            weights=tuple(float(value) for value in values / normalizer),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ranks = _rank_frame_columns(frame, self.columns)
        return ranks @ np.asarray(self.weights, dtype="float64")


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
    groups = (
        frame.groupby(timestamps, sort=False)
        if timestamps is not None
        else ((None, frame),)
    )
    for _timestamp, group in groups:
        for column in columns:
            values = strict_finite_numeric_series(
                group[column],
                error=f"invalid_rank_ensemble_column: {column}",
                allow_null=True,
            )
            valid = values.notna()
            count = int(valid.sum())
            if count == 0:
                continue
            ranks = values.loc[valid].rank(method="average")
            centered = (ranks - (count + 1.0) / 2.0) / count
            result.loc[centered.index, column] = centered
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
    masses = (
        pd.Series(weights).groupby(pd.to_datetime(frame["timestamp"]).to_numpy()).sum()
        if "timestamp" in frame
        else None
    )
    for column in columns:
        correlations, period_weights = [], []
        for group in _cross_section_groups(frame):
            x = group[column].to_numpy(dtype="float64", copy=False)
            y = group[label_column].to_numpy(dtype="float64", copy=False)
            valid = np.isfinite(x) & np.isfinite(y)
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
                    else masses.loc[pd.Timestamp(group["timestamp"].iloc[0])]
                )
        yield (
            column,
            (
                float(np.average(correlations, weights=period_weights))
                if correlations
                else None
            ),
        )


def _cross_section_groups(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    if "timestamp" not in frame:
        yield frame
        return
    timestamps = strict_timestamp_series(
        frame["timestamp"],
        error="invalid_rank_ensemble_timestamp",
    )
    for _, group in frame.groupby(timestamps, sort=True):
        yield group
