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
from quant_research_platform.contracts.validation import strict_timestamp_series


@dataclass(frozen=True)
class RankEnsemblePredictor:
    columns: tuple[str, ...]
    coefficient_count: int = 1

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return _rank_frame_columns(frame, self.columns).mean(axis=1)


@dataclass(frozen=True)
class ICWeightedRankEnsembleFactory:
    columns: tuple[str, ...]
    label_column: str

    def fit_training_frame(self, frame: pd.DataFrame) -> ICWeightedRankEnsemble:
        return ICWeightedRankEnsemble.fit(
            frame,
            input_columns=self.columns,
            label_column=self.label_column,
        )


@dataclass(frozen=True)
class ICWeightedRankEnsemble:
    columns: tuple[str, ...]
    weights: tuple[float, ...]
    coefficient_count: int

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
            coefficient_count=len(defined_columns),
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
        ranked = frame.groupby(timestamps, sort=False)[list(columns)].rank(
            method="average",
            pct=True,
        )
    else:
        ranked = frame.loc[:, columns].rank(method="average", pct=True)
    return ranked.to_numpy(dtype="float64", copy=False)


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
