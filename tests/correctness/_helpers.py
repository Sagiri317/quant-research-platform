from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY_COLUMNS = ("market_id", "asset_id", "timestamp")


def synthetic_price_volume_panel(
    *,
    sessions: int = 160,
    assets: tuple[str, ...] = ("asset_a", "asset_b"),
) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=sessions)
    rows: list[dict[str, object]] = []
    for asset_index, asset_id in enumerate(assets, start=1):
        for date_index, timestamp in enumerate(dates, start=1):
            close = 10.0 + asset_index + date_index * 0.07
            wave = ((date_index % 11) - 5) * 0.0007
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "return_1d": 0.001 * asset_index + wave,
                    "adj_close": close,
                    "adj_open": close * 0.997,
                    "open": close * 0.997,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "prev_close": close / (1.0 + 0.001 * asset_index + wave),
                    "amount": 100_000.0 + date_index * 10 + asset_index,
                    "turnover": 0.01 + asset_index * 0.001 + date_index / 1_000_000,
                    "float_market_cap": 1_000_000.0 + date_index,
                    "total_market_cap": 2_000_000.0 + date_index,
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                    "limit_status": 0,
                }
            )
    return pd.DataFrame(rows)


def reference_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True).copy()
    group_keys = result["asset_id"]
    returns = pd.to_numeric(result["return_1d"], errors="coerce")
    turnover = pd.to_numeric(result["turnover"], errors="coerce")
    for column in factor_columns:
        window = factor_window(column)
        if column.startswith("vol_"):
            source = returns
            function = reference_std_window
        elif column.startswith("skew_"):
            source = returns
            function = reference_skew_window
        elif column.startswith("kurt_"):
            source = returns
            function = reference_kurt_window
        elif column.startswith("turnover_vol_"):
            source = turnover
            function = reference_std_window
        else:
            raise ValueError(f"unsupported_reference_price_volume_column: {column}")
        result[column] = source.groupby(group_keys, sort=False).transform(
            lambda series, w=window, fn=function: series.rolling(
                w,
                min_periods=w,
            ).apply(fn, raw=True)
        )
    return result.loc[:, [*KEY_COLUMNS, *factor_columns]]


def factor_window(column: str) -> int:
    for token in column.split("_"):
        if token.endswith("d") and token[:-1].isdigit():
            return int(token[:-1])
    raise ValueError(f"factor_window_missing: {column}")


def reference_std_window(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    centered = values - values.mean()
    return float(np.sqrt(np.sum(centered * centered) / (len(values) - 1)))


def reference_skew_window(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return np.nan
    centered = values - values.mean()
    moment_2 = float(np.mean(centered * centered))
    moment_3 = float(np.mean(centered * centered * centered))
    denominator = moment_2**1.5
    if denominator == 0.0:
        return np.nan
    count = len(values)
    value = np.sqrt(count * (count - 1)) / (count - 2) * moment_3 / denominator
    return float(value) if np.isfinite(value) else np.nan


def reference_kurt_window(values: np.ndarray) -> float:
    if len(values) < 4 or not np.isfinite(values).all():
        return np.nan
    centered = values - values.mean()
    moment_2 = float(np.mean(centered * centered))
    if moment_2 == 0.0:
        return np.nan
    moment_4 = float(np.mean(centered**4))
    count = len(values)
    value = (
        (count - 1)
        / ((count - 2) * (count - 3))
        * ((count + 1) * (moment_4 / (moment_2 * moment_2) - 3.0) + 6.0)
    )
    return float(value) if np.isfinite(value) else np.nan


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
    atol: float = 1e-12,
    rtol: float = 0.0,
) -> NumericDiffReport:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_nan = left_num.isna() & right_num.isna()
    diffs = (left_num - right_num).abs()
    tolerance = atol + rtol * right_num.abs()
    mismatch = ~(both_nan | diffs.le(tolerance))
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
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        return (left_na & right_na) | ((left_num - right_num).abs() <= 1e-12)
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


def hashed_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
