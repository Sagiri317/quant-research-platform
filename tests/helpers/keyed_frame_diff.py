from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tests.helpers.price_volume_reference import KEY_COLUMNS

FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12


@dataclass(frozen=True)
class KeyDiffResult:
    status: str
    baseline_only: int
    candidate_only: int
    intersection: int
    mismatch_columns: dict[str, int]
    baseline_only_classification: str
    duplicate_keys: int = 0


@dataclass(frozen=True)
class NumericDiffReport:
    mismatch_count: int
    max_abs_diff: float
    mean_abs_diff: float
    quantiles: dict[str, float]


def numeric_series_diff_report(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float = FLOAT_ATOL,
    rtol: float = FLOAT_RTOL,
) -> NumericDiffReport:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    diffs = (left_num - right_num).abs()
    mismatch = ~numeric_values_equal(left_num, right_num, atol=atol, rtol=rtol)
    finite = diffs[mismatch & np.isfinite(diffs)]
    if finite.empty:
        return NumericDiffReport(
            mismatch_count=int(mismatch.sum()),
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            quantiles={"0.5": 0.0, "0.9": 0.0, "0.99": 0.0, "1.0": 0.0},
        )
    quantiles = finite.quantile([0.5, 0.9, 0.99, 1.0])
    return NumericDiffReport(
        mismatch_count=int(mismatch.sum()),
        max_abs_diff=float(finite.max()),
        mean_abs_diff=float(finite.mean()),
        quantiles={str(index): float(value) for index, value in quantiles.items()},
    )


def numeric_values_equal(
    left: pd.Series,
    right: pd.Series,
    *,
    atol: float = FLOAT_ATOL,
    rtol: float = FLOAT_RTOL,
) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    left_na = left_num.isna()
    right_na = right_num.isna()
    tolerance = atol + rtol * right_num.abs()
    return (left_na & right_na) | (
        left_na.eq(False)
        & right_na.eq(False)
        & (left_num - right_num).abs().le(tolerance)
    )


def compare_synthetic_keyed_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    trading_sessions: set[str],
    keys: tuple[str, ...] = KEY_COLUMNS,
) -> KeyDiffResult:
    left = baseline.copy()
    right = candidate.copy()
    left["__key__"] = key_series(left, keys)
    right["__key__"] = key_series(right, keys)
    duplicate_keys = int(left["__key__"].duplicated().sum()) + int(
        right["__key__"].duplicated().sum()
    )
    if duplicate_keys:
        return KeyDiffResult(
            status="DUPLICATE_KEYS",
            baseline_only=0,
            candidate_only=0,
            intersection=0,
            mismatch_columns={},
            baseline_only_classification="not_evaluated",
            duplicate_keys=duplicate_keys,
        )
    left_keys = set(left["__key__"])
    right_keys = set(right["__key__"])
    baseline_only_keys = left_keys - right_keys
    candidate_only_keys = right_keys - left_keys
    common_keys = left_keys & right_keys
    common_columns = [
        column
        for column in baseline.columns
        if column in candidate.columns and column not in keys
    ]
    left_common = left[left["__key__"].isin(common_keys)].sort_values("__key__")
    right_common = right[right["__key__"].isin(common_keys)].sort_values("__key__")
    mismatch_columns: dict[str, int] = {}
    for column in common_columns:
        equal = value_equal(
            left_common[column].reset_index(drop=True),
            right_common[column].reset_index(drop=True),
        )
        mismatch = int((~equal).sum())
        if mismatch:
            mismatch_columns[column] = mismatch
    baseline_only_classification = classify_baseline_only_dates(
        left,
        baseline_only_keys,
        trading_sessions=trading_sessions,
    )
    status = "PASS"
    if candidate_only_keys:
        status = "CANDIDATE_ONLY_KEYS"
    elif baseline_only_classification == "required_session_missing":
        status = "BASELINE_ONLY_REQUIRED_SESSION"
    elif mismatch_columns:
        status = "VALUE_MISMATCH"
    elif baseline_only_keys:
        status = "EXPECTED_BASELINE_ONLY"
    return KeyDiffResult(
        status=status,
        baseline_only=len(baseline_only_keys),
        candidate_only=len(candidate_only_keys),
        intersection=len(common_keys),
        mismatch_columns=mismatch_columns,
        baseline_only_classification=baseline_only_classification,
    )


def key_series(frame: pd.DataFrame, keys: tuple[str, ...]) -> pd.Series:
    parts = []
    for column in keys:
        if column == "timestamp":
            parts.append(pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d"))
        else:
            parts.append(frame[column].astype("string").fillna("<NA>"))
    return pd.Series(
        ["|".join(values) for values in zip(*parts, strict=True)],
        index=frame.index,
    )


def value_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    left_na = left.isna()
    right_na = right.isna()
    if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
        return numeric_values_equal(left, right)
    return (left_na & right_na) | (
        left.astype("string").fillna("<NA>") == right.astype("string").fillna("<NA>")
    )


def classify_baseline_only_dates(
    frame: pd.DataFrame,
    baseline_only_keys: set[str],
    *,
    trading_sessions: set[str],
) -> str:
    if not baseline_only_keys:
        return "none"
    only = frame[frame["__key__"].isin(baseline_only_keys)]
    dates = set(pd.to_datetime(only["timestamp"]).dt.strftime("%Y-%m-%d"))
    if dates.isdisjoint(trading_sessions):
        return "non_required_session"
    return "required_session_missing"
