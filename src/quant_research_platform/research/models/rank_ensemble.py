"""Rank-ensemble model helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _pearson_correlation_result,
)


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
    weights: np.ndarray
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
        label = _rank_series(frame, label_column)
        label_values = label.to_numpy(dtype="float64", copy=False)
        for column in input_columns:
            input_values = _rank_series(frame, column).to_numpy(
                dtype="float64",
                copy=False,
            )
            valid = np.isfinite(input_values) & np.isfinite(label_values)
            if int(valid.sum()) < 2:
                corr = None
            else:
                corr, _reason = _pearson_correlation_result(
                    input_values[valid],
                    label_values[valid],
                )
            if corr is None:
                continue
            defined_columns.append(column)
            weights.append(float(corr))
        if not weights:
            raise ValueError("all_ensemble_correlations_undefined")
        values = np.asarray(weights, dtype="float64")
        normalizer = float(np.sum(np.abs(values)))
        if normalizer == 0.0:
            raise ValueError("all_ensemble_correlations_zero")
        return cls(
            columns=tuple(defined_columns),
            weights=values / normalizer,
            coefficient_count=len(defined_columns),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ranks = _rank_frame_columns(frame, self.columns)
        return ranks @ self.weights


def _rank_frame_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if "timestamp" in frame:
        ranked = frame.groupby("timestamp", sort=False)[list(columns)].rank(
            method="average",
            pct=True,
        )
    else:
        ranked = frame.loc[:, columns].rank(method="average", pct=True)
    return ranked.to_numpy(dtype="float64", copy=False)


def _rank_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if "timestamp" in frame:
        return frame.groupby("timestamp", sort=False)[column].rank(
            method="average",
            pct=True,
        )
    return frame[column].rank(method="average", pct=True)
