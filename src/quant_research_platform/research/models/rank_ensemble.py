"""Rank-ensemble model helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
        label = _rank_series(frame, label_column)
        for column in input_columns:
            corr = _rank_series(frame, column).corr(label, method="pearson")
            weights.append(0.0 if pd.isna(corr) else float(corr))
        values = np.asarray(weights, dtype="float64")
        if np.allclose(values, 0.0):
            values = np.ones(len(input_columns), dtype="float64")
        return cls(
            columns=input_columns,
            weights=values / np.sum(np.abs(values)),
            coefficient_count=len(input_columns),
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
