"""Rank-ensemble model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _pearson_correlation_result,
)

_RANK_ZERO_ROUNDOFF_LIMIT = np.finfo(np.float64).eps


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
                if (
                    corr is not None
                    # Rank values are exact rationals; only near-roundoff
                    # results pay for the exact covariance check below.
                    and abs(corr)
                    <= _RANK_ZERO_ROUNDOFF_LIMIT * max(1, int(valid.sum()))
                    and _exact_rank_covariance_is_zero(
                        input_values,
                        label_values,
                        valid,
                        denominator_bound=2 * len(frame),
                    )
                ):
                    corr = 0.0
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


def _exact_rank_covariance_is_zero(
    left_values: np.ndarray,
    right_values: np.ndarray,
    valid: np.ndarray,
    *,
    denominator_bound: int,
) -> bool:
    count = 0
    left_sum = Fraction(0)
    right_sum = Fraction(0)
    product_sum = Fraction(0)
    # pct average rank = (integer or half-integer) / group_count, and every
    # group_count is at most len(frame), so 2 * len(frame) bounds its reduced
    # denominator. The loop retains only four sufficient statistics.
    for left_value, right_value, is_valid in zip(
        left_values,
        right_values,
        valid,
        strict=True,
    ):
        if not is_valid:
            continue
        left = Fraction(float(left_value)).limit_denominator(denominator_bound)
        right = Fraction(float(right_value)).limit_denominator(denominator_bound)
        count += 1
        left_sum += left
        right_sum += right
        product_sum += left * right
    return count >= 2 and count * product_sum == left_sum * right_sum
