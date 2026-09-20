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

    def fit(self, frame: pd.DataFrame, _target: np.ndarray) -> RankEnsemblePredictor:
        directions = _rank_directions(
            frame,
            input_columns=self.columns,
            label_column=self.label_column,
        )
        return RankEnsemblePredictor(
            columns=self.columns,
            directions=directions,
        )


@dataclass(frozen=True)
class ICWeightedRankEnsembleFactory:
    columns: tuple[str, ...]
    label_column: str

    def fit(self, frame: pd.DataFrame, _target: np.ndarray) -> ICWeightedRankEnsemble:
        return ICWeightedRankEnsemble.fit(
            frame,
            input_columns=self.columns,
            label_column=self.label_column,
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
    ) -> ICWeightedRankEnsemble:
        weights = []
        defined_columns: list[str] = []
        for column in input_columns:
            timestamp_correlations: list[float] = []
            for timestamp_frame in _cross_section_groups(frame):
                input_values = timestamp_frame[column].to_numpy(
                    dtype="float64",
                    copy=False,
                )
                label_values = timestamp_frame[label_column].to_numpy(
                    dtype="float64",
                    copy=False,
                )
                valid = np.isfinite(input_values) & np.isfinite(label_values)
                if int(valid.sum()) < 2:
                    continue
                corr, _reason = _pearson_correlation_result(
                    _average_ranks(input_values[valid]),
                    _average_ranks(label_values[valid]),
                )
                if corr is not None:
                    timestamp_correlations.append(corr)
            if not timestamp_correlations:
                continue
            defined_columns.append(column)
            weights.append(float(np.mean(timestamp_correlations)))
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
) -> tuple[float, ...]:
    directions: list[float] = []
    for column in input_columns:
        correlations: list[float] = []
        for timestamp_frame in _cross_section_groups(frame):
            input_values = timestamp_frame[column].to_numpy(dtype="float64", copy=False)
            label_values = timestamp_frame[label_column].to_numpy(
                dtype="float64", copy=False
            )
            valid = np.isfinite(input_values) & np.isfinite(label_values)
            if int(valid.sum()) < 2:
                continue
            correlation, _reason = _pearson_correlation_result(
                _average_ranks(input_values[valid]),
                _average_ranks(label_values[valid]),
            )
            if correlation is not None:
                correlations.append(correlation)
        if not correlations:
            raise ValueError(
                f"ensemble_direction_undefined:{column}:no_valid_training_rankic"
            )
        directions.append(1.0 if float(np.mean(correlations)) >= 0.0 else -1.0)
    return tuple(directions)


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
