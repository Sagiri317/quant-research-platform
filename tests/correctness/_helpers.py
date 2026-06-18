from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

KEY_COLUMNS = ("market_id", "asset_id", "timestamp")
FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12
REFERENCE_ROLLING_MOMENT_NEAR_ZERO_VARIANCE = 1e-12


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


def _reference_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
) -> pd.DataFrame:
    result = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True).copy()
    group_keys = result["asset_id"]
    grouped = result.groupby("asset_id", sort=False)
    price = pd.to_numeric(result["adj_close"], errors="coerce")
    returns = pd.to_numeric(result["return_1d"], errors="coerce")
    amount = pd.to_numeric(result["amount"], errors="coerce")
    turnover = pd.to_numeric(result["turnover"], errors="coerce")
    requested = tuple(dict.fromkeys(factor_columns))
    requested_set = set(requested)
    required_windows = reference_windows_from_requested_factors(requested_set)

    if "rev_1d" in requested_set:
        result["rev_1d"] = -returns
    for window in required_windows:
        if window == 1:
            continue
        needs_momentum = f"mom_{window}d" in requested_set
        needs_reversal = f"rev_{window}d" in requested_set
        if needs_momentum or needs_reversal:
            shifted_price = price.groupby(group_keys, sort=False).shift(window)
            trailing_return = price / reference_valid_denominator(shifted_price) - 1.0
            trailing_return = reference_finite_or_nan(trailing_return)
        if needs_momentum:
            result[f"mom_{window}d"] = trailing_return
        if needs_reversal:
            result[f"rev_{window}d"] = -trailing_return

    for window in required_windows:
        if f"vol_{window}d" in requested_set:
            result[f"vol_{window}d"] = reference_qrp_window_moment_by_asset(
                returns,
                group_keys,
                window,
                "std",
            )
        if f"downside_vol_{window}d" in requested_set:
            result[f"downside_vol_{window}d"] = reference_rolling_rms(
                returns.clip(upper=0.0),
                group_keys,
                window,
            )
        if f"upside_vol_{window}d" in requested_set:
            result[f"upside_vol_{window}d"] = reference_rolling_rms(
                returns.clip(lower=0.0),
                group_keys,
                window,
            )
        if f"max_return_{window}d" in requested_set:
            result[f"max_return_{window}d"] = grouped["return_1d"].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).max()
            )
        if f"min_return_{window}d" in requested_set:
            result[f"min_return_{window}d"] = grouped["return_1d"].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).min()
            )
        if f"drawdown_{window}d" in requested_set:
            result[f"drawdown_{window}d"] = reference_rolling_drawdown(
                price,
                group_keys,
                window,
            )
        if f"skew_{window}d" in requested_set:
            result[f"skew_{window}d"] = reference_qrp_window_moment_by_asset(
                returns,
                group_keys,
                window,
                "skew",
            )
        if f"kurt_{window}d" in requested_set:
            result[f"kurt_{window}d"] = reference_qrp_window_moment_by_asset(
                returns,
                group_keys,
                window,
                "kurt",
            )

    for column in factor_columns:
        try:
            window = factor_window(column)
        except ValueError:
            continue
        if f"avg_amount_{window}d" == column:
            result[column] = amount.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        elif f"turnover_{window}d" == column:
            result[column] = turnover.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        elif f"turnover_chg_{window}d" == column:
            shifted_turnover = turnover.groupby(group_keys, sort=False).shift(window)
            result[column] = (
                turnover / reference_valid_denominator(shifted_turnover) - 1.0
            )
        elif f"turnover_vol_{window}d" == column:
            result[column] = reference_qrp_window_moment_by_asset(
                turnover,
                group_keys,
                window,
                "std",
            )
        elif f"zero_volume_ratio_{window}d" == column:
            zero_volume = (amount <= 0.0).astype(float)
            result[column] = zero_volume.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )

    denominator = reference_valid_denominator(amount)
    illiquidity = (returns.abs() / denominator).replace([np.inf, -np.inf], np.nan)
    amplitude = (
        pd.to_numeric(result["high"], errors="coerce")
        - pd.to_numeric(result["low"], errors="coerce")
    ) / reference_valid_denominator(
        pd.to_numeric(result["prev_close"], errors="coerce")
    )
    amplitude = reference_finite_or_nan(amplitude)
    intraday_range = (
        pd.to_numeric(result["high"], errors="coerce")
        - pd.to_numeric(result["low"], errors="coerce")
    ) / reference_valid_denominator(pd.to_numeric(result["adj_close"], errors="coerce"))
    intraday_range = reference_finite_or_nan(intraday_range)
    for window in required_windows:
        if f"amihud_{window}d" in requested_set:
            result[f"amihud_{window}d"] = illiquidity.groupby(
                group_keys,
                sort=False,
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"amplitude_{window}d" in requested_set:
            result[f"amplitude_{window}d"] = amplitude.groupby(
                group_keys,
                sort=False,
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"intraday_range_{window}d" in requested_set:
            result[f"intraday_range_{window}d"] = intraday_range.groupby(
                group_keys,
                sort=False,
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )

    if "log_float_market_cap" in requested_set:
        result["log_float_market_cap"] = reference_safe_log(result["float_market_cap"])
    if "log_total_market_cap" in requested_set:
        result["log_total_market_cap"] = reference_safe_log(result["total_market_cap"])
    output_columns = [column for column in requested if column in result.columns]
    if output_columns:
        result.loc[:, output_columns] = result.loc[:, output_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )
    return result.loc[:, [*KEY_COLUMNS, *requested]]


def pandas_rolling_by_asset(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
    method: str,
) -> pd.Series:
    return values.groupby(group_keys, sort=False).transform(
        lambda series, w=window, name=method: getattr(
            series.rolling(w, min_periods=w),
            name,
        )()
    )


def pandas_window_moment_by_asset(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
    method: str,
) -> pd.Series:
    return values.groupby(group_keys, sort=False).transform(
        lambda series, w=window, name=method: series.rolling(
            w,
            min_periods=w,
        ).apply(
            lambda window_values: pandas_window_moment(window_values, name),
            raw=True,
        )
    )


def pandas_window_moment(values: np.ndarray, method: str) -> float:
    series = pd.Series(values, dtype="float64")
    rolling = series.rolling(len(series), min_periods=len(series))
    if method == "std":
        return float(rolling.std(ddof=1).iloc[-1])
    if method == "skew":
        return float(rolling.skew().iloc[-1])
    if method == "kurt":
        return float(rolling.kurt().iloc[-1])
    raise ValueError(f"unsupported_pandas_window_moment: {method}")


def reference_qrp_window_moment_by_asset(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
    method: str,
) -> pd.Series:
    return values.groupby(group_keys, sort=False).transform(
        lambda series, w=window, name=method: series.rolling(
            w,
            min_periods=w,
        ).apply(
            lambda window_values: reference_qrp_window_moment(window_values, name),
            raw=True,
        )
    )


def reference_qrp_window_moment(values: np.ndarray, method: str) -> float:
    numeric = np.asarray(values, dtype="float64")
    if not np.isfinite(numeric).all():
        return float("nan")
    counts = len(numeric)
    centered = numeric - numeric.mean()
    if method == "std":
        if counts < 2:
            return float("nan")
        variance = np.sum(centered * centered) / (counts - 1)
        return float(np.sqrt(max(variance, 0.0)))
    moment_2 = float(np.mean(centered * centered))
    if moment_2 <= REFERENCE_ROLLING_MOMENT_NEAR_ZERO_VARIANCE:
        return float("nan")
    if method == "skew":
        if counts < 3:
            return float("nan")
        moment_3 = float(np.mean(centered * centered * centered))
        value = (
            np.sqrt(counts * (counts - 1))
            / (counts - 2)
            * (moment_3 / np.power(moment_2, 1.5))
        )
    elif method == "kurt":
        if counts < 4:
            return float("nan")
        moment_4 = float(np.mean(centered**4))
        value = (
            (counts - 1)
            / ((counts - 2) * (counts - 3))
            * ((counts + 1) * (moment_4 / (moment_2 * moment_2) - 3.0) + 6.0)
        )
    else:
        raise ValueError(f"unsupported_qrp_window_moment: {method}")
    if not np.isfinite(value):
        return float("nan")
    return float(value)


def factor_window(column: str) -> int:
    for token in column.split("_"):
        if token.endswith("d") and token[:-1].isdigit():
            return int(token[:-1])
    raise ValueError(f"factor_window_missing: {column}")


def reference_rolling_rms(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    squared = values.pow(2)
    mean_squared = squared.groupby(group_keys, sort=False).transform(
        lambda series, w=window: series.rolling(w, min_periods=w).mean()
    )
    return np.sqrt(mean_squared)


def reference_rolling_drawdown(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    def calculate(series: pd.Series) -> float:
        peak = series.cummax()
        drawdown = series / peak.replace(0.0, np.nan) - 1.0
        return float(reference_finite_or_nan(drawdown).min())

    return values.groupby(group_keys, sort=False).transform(
        lambda series, w=window: series.rolling(w, min_periods=w).apply(
            calculate,
            raw=False,
        )
    )


def reference_safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric) & numeric.gt(0.0))
    return reference_finite_or_nan(np.log(numeric))


def reference_finite_or_nan(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)


def reference_valid_denominator(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric) & numeric.ne(0.0))


def reference_windows_from_requested_factors(requested: set[str]) -> tuple[int, ...]:
    values = {1} if "rev_1d" in requested else set()
    for factor in requested:
        for token in factor.split("_"):
            if token.endswith("d") and token[:-1].isdigit():
                values.add(int(token[:-1]))
    return tuple(sorted(values))


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


def assert_frame_matches_reference(
    actual: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    rtol: float = FLOAT_RTOL,
    atol: float = FLOAT_ATOL,
) -> None:
    pd.testing.assert_frame_equal(
        actual,
        reference,
        check_exact=False,
        rtol=rtol,
        atol=atol,
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


def hashed_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def assert_forced_tiny_processed_chunks_match_default(
    tmp_path,
    *,
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
) -> None:
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    write_minimal_interim_inputs(default_layout, end_date=end_date)
    write_minimal_interim_inputs(tiny_layout, end_date=end_date)

    materializer.materialize_processed_data(
        layout=default_layout,
        source_id=SourceId("csmar"),
        time_window=time_window,
        data_ids=(data_id,),
    )
    materializer.materialize_processed_data(
        layout=tiny_layout,
        source_id=SourceId("csmar"),
        time_window=time_window,
        data_ids=(data_id,),
        resource_policy=RuntimeResourcePolicy(
            available_memory_bytes=1_000,
            memory_budget_fraction=1.0,
            chunk_target_bytes=250,
            memory_safety_factor=1.0,
        ),
    )

    default_root = processed_root(tmp_path / "default", kind, data_id)
    tiny_root = processed_root(tmp_path / "tiny", kind, data_id)
    default = read_dataset(default_root).sort_values(list(KEY_COLUMNS))
    tiny = read_dataset(tiny_root).sort_values(list(KEY_COLUMNS))

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
    assert (
        inspect_dataset(tiny_root).file_count
        > inspect_dataset(default_root).file_count
    )


def processed_root(project_root, kind: str, data_id: str):
    return project_root / "data" / "processed" / "a_share" / kind / data_id


def write_minimal_interim_inputs(
    layout: StorageLayout,
    *,
    start_date: str = "2014-01-02",
    end_date: str = "2014-01-03",
    closed_dates: tuple[str, ...] = (),
) -> None:
    catalog = interim_catalog()
    source_id = SourceId("csmar")
    frames = {
        "daily_market": minimal_daily_market_frame(
            start_date=start_date,
            end_date=end_date,
        ),
        "stock_master": minimal_stock_master_frame(),
        "market_cap": minimal_market_cap_frame(),
        "trading_calendar": minimal_trading_calendar_frame(
            end_date=end_date,
            closed_dates=closed_dates,
        ),
    }
    for data_id, frame in frames.items():
        spec = catalog.get(data_id)
        write_dataset(
            layout.interim_data_root(source_id, data_id),
            frame.loc[:, list(spec.required_columns)],
            spec=spec,
        )


def minimal_daily_market_frame(
    *,
    start_date: str = "2014-01-02",
    end_date: str = "2014-01-03",
) -> pd.DataFrame:
    rows = []
    for offset, timestamp in enumerate(
        pd.bdate_range(start_date, end_date),
        start=1,
    ):
        rows.append(
            {
                "source_asset_id": "000001",
                "timestamp": timestamp,
                "trade_date": timestamp,
                "market_type": 4,
                "trading_status_code": 1,
                "filling_flag": "0",
                "open": 10.0 + offset,
                "high": 10.5 + offset,
                "low": 9.8 + offset,
                "close": 10.2 + offset,
                "volume": 1000.0 + offset,
                "amount": 10200.0 + offset,
                "prev_close": 9.9 + offset,
                "limit_status": 0,
                "return_1d": 0.01,
                "adj_close": 10.2 + offset,
                "adj_open": 10.0 + offset,
                "float_market_cap": 100000.0 + offset,
                "total_market_cap": 200000.0 + offset,
                "turnover": 0.01,
            }
        )
    return pd.DataFrame(rows)


def minimal_stock_master_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_asset_id": "000001",
                "timestamp": pd.Timestamp("2010-01-01"),
                "stock_name": "demo",
                "listing_date": pd.Timestamp("2010-01-01"),
                "delisting_date": pd.NaT,
                "company_status": "normal",
                "csrc_2012_industry_code": "I",
                "csrc_2012_industry_name": "industry",
                "status_change_date": pd.Timestamp("2010-01-01"),
            }
        ]
    )


def minimal_market_cap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_asset_id": "000001",
                "timestamp": pd.Timestamp("2010-01-01"),
                "share_change_date": pd.Timestamp("2010-01-01"),
                "total_shares": 1000.0,
                "a_float_shares": 500.0,
            }
        ]
    )


def minimal_trading_calendar_frame(
    *,
    end_date: str = "2014-01-03",
    closed_dates: tuple[str, ...] = (),
) -> pd.DataFrame:
    closed = {pd.Timestamp(date).date() for date in closed_dates}
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(date),
                "calendar_date": pd.Timestamp(date),
                "source_market_type": 4,
                "is_trading_day": pd.Timestamp(date).weekday() < 5
                and pd.Timestamp(date).date() not in closed,
                "source_day_of_week": pd.Timestamp(date).dayofweek + 1,
                "source_state": (
                    "O"
                    if pd.Timestamp(date).weekday() < 5
                    and pd.Timestamp(date).date() not in closed
                    else "C"
                ),
            }
            for date in pd.date_range("2014-01-01", end_date)
        ]
    )


def research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "mom_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)


def correctness_processed_catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_factors",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_labels",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.LABEL,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_annotations",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.ANNOTATION,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
                "market_type": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def correctness_research_dataset_catalog(
    *,
    include_feature_transform: bool = False,
) -> ResearchDatasetCatalog:
    feature_columns = (
        ("mom_20d_zscore",) if include_feature_transform else ()
    )
    feature_transforms = (
        (
                FeatureTransformSpec(
                    transform_id="mom_20d_zscore",
                    source_columns=("mom_20d",),
                    output_columns=("mom_20d_zscore",),
                    transform="cross_sectional_zscore",
                ),
        )
        if include_feature_transform
        else ()
    )
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            data_id="demo",
            market_id=MarketId("a_share"),
            input_data_ids=("demo_factors", "demo_labels", "demo_annotations"),
            base_input_data_id="demo_annotations",
            factor_columns=("mom_20d",),
            feature_columns=feature_columns,
            feature_transforms=feature_transforms,
            label_columns=("forward_return_20d",),
            filter_columns=("eligible", "can_enter", "can_exit"),
            context_columns=("market_type",),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
                **(
                    {"mom_20d_zscore": ColumnRole.FEATURE}
                    if include_feature_transform
                    else {}
                ),
                "forward_return_20d": ColumnRole.LABEL,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
                "market_type": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def write_correctness_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    frame: pd.DataFrame,
) -> None:
    columns_by_id = {
        "demo_factors": ["market_id", "asset_id", "timestamp", "mom_20d"],
        "demo_annotations": [
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            "can_enter",
            "can_exit",
            "market_type",
        ],
        "demo_labels": [
            "market_id",
            "asset_id",
            "timestamp",
            "forward_return_20d",
        ],
    }
    for data_id, columns in columns_by_id.items():
        spec = processed_catalog.get(data_id)
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame.loc[:, columns],
            spec=spec,
        )
