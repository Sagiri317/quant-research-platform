"""Literature-prior A-share price-volume factor pack helpers."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID = (
    "a_share_literature_price_volume_rewrite_01"
)
LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID = (
    "a_share_literature_price_volume_preset_01"
)

DEFAULT_LITERATURE_WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 60, 120, 252)
LITERATURE_FACTOR_SETS: tuple[str, ...] = (
    "full",
    "core",
    "smoke",
    "experimental",
    "deprecated",
)
CORE_FACTORS: tuple[str, ...] = (
    "rev_1d",
    "rev_5d",
    "rev_10d",
    "rev_20d",
    "mom_20d",
    "mom_60d",
    "mom_120d",
    "vol_20d",
    "vol_60d",
    "downside_vol_20d",
    "downside_vol_60d",
    "avg_amount_20d",
    "avg_amount_60d",
    "turnover_20d",
    "turnover_60d",
    "amihud_20d",
    "amihud_60d",
    "amplitude_20d",
    "amplitude_60d",
    "log_float_market_cap",
    "log_total_market_cap",
)
SMOKE_FACTORS: tuple[str, ...] = (
    "rev_1d",
    "mom_20d",
    "vol_20d",
    "avg_amount_20d",
    "turnover_20d",
    "amihud_20d",
    "amplitude_20d",
    "log_float_market_cap",
    "log_total_market_cap",
)
LITERATURE_FEATURE_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "forward_return",
    "target",
    "label",
)
LITERATURE_FACTOR_PREFIXES: tuple[str, ...] = (
    "mom_",
    "rev_",
    "vol_",
    "downside_vol_",
    "upside_vol_",
    "max_return_",
    "min_return_",
    "drawdown_",
    "skew_",
    "kurt_",
    "avg_amount_",
    "turnover_",
    "turnover_chg_",
    "turnover_vol_",
    "amihud_",
    "zero_volume_ratio_",
    "amplitude_",
    "intraday_range_",
    "log_float_market_cap",
    "log_total_market_cap",
)
LITERATURE_FEATURE_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {
        "stock_code",
        "trade_date",
        "timestamp",
        "asset",
        "market_type",
        "eligible",
        "eligible_base",
        "can_enter",
        "can_exit",
        "is_a_share_scope",
        "is_trading",
        "is_st",
        "is_pt",
        "is_limit_up",
        "is_limit_down",
        "limit_status",
        "trading_status_code",
        "filling_flag",
    }
)
BASE_FACTOR_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "rev_1d": ("return_1d",),
    "mom": ("adj_close",),
    "rev": ("adj_close",),
    "vol": ("return_1d",),
    "downside_vol": ("return_1d",),
    "upside_vol": ("return_1d",),
    "max_return": ("return_1d",),
    "min_return": ("return_1d",),
    "drawdown": ("adj_close",),
    "skew": ("return_1d",),
    "kurt": ("return_1d",),
    "avg_amount": ("amount",),
    "turnover": ("turnover",),
    "turnover_chg": ("turnover",),
    "turnover_vol": ("turnover",),
    "amihud": ("return_1d", "amount"),
    "zero_volume_ratio": ("amount",),
    "amplitude": ("high", "low", "prev_close"),
    "intraday_range": ("high", "low", "adj_close"),
    "log_float_market_cap": ("float_market_cap",),
    "log_total_market_cap": ("total_market_cap",),
}


def expected_literature_price_volume_factor_columns(
    *,
    windows: tuple[int, ...] = DEFAULT_LITERATURE_WINDOWS,
    factor_set: str = "full",
) -> tuple[str, ...]:
    """Return the explicit 0.1 literature-prior factor column contract."""

    if factor_set == "core":
        return CORE_FACTORS
    if factor_set == "smoke":
        return SMOKE_FACTORS
    if factor_set in {"experimental", "deprecated"}:
        return ()
    if factor_set != "full":
        raise ValueError("invalid_factor_set")

    columns: list[str] = ["rev_1d"]
    for window in windows:
        if window != 1:
            columns.extend((f"rev_{window}d", f"mom_{window}d"))
    for window in (5, 20, 60, 120):
        columns.append(f"vol_{window}d")
    for window in (20, 60):
        columns.extend(
            (
                f"downside_vol_{window}d",
                f"upside_vol_{window}d",
                f"max_return_{window}d",
                f"min_return_{window}d",
                f"drawdown_{window}d",
                f"skew_{window}d",
                f"kurt_{window}d",
            )
        )
    for window in (5, 20, 60, 120):
        columns.extend((f"avg_amount_{window}d", f"turnover_{window}d"))
    for window in (20, 60):
        columns.extend((f"turnover_chg_{window}d", f"turnover_vol_{window}d"))
    for window in (20, 60, 120):
        columns.append(f"amihud_{window}d")
    for window in (20, 60):
        columns.append(f"zero_volume_ratio_{window}d")
    for window in (5, 20, 60, 120):
        columns.append(f"amplitude_{window}d")
    for window in (20, 60):
        columns.append(f"intraday_range_{window}d")
    columns.extend(("log_float_market_cap", "log_total_market_cap"))
    return tuple(dict.fromkeys(columns))


def exact_literature_price_volume_factor_input_columns(factor: str) -> tuple[str, ...]:
    """Return exact base input columns required by one registered factor."""

    required = _required_fields_for_factor(factor)
    if not required:
        raise ValueError(f"unknown_literature_price_volume_factor: {factor}")
    return required


def unsupported_literature_price_volume_factors(
    available_columns: tuple[str, ...] | list[str] | set[str],
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], ...]:
    """Return unsupported requested factors with missing field reasons."""

    available = set(available_columns)
    expected = expected_columns or expected_literature_price_volume_factor_columns()
    unsupported: list[dict[str, object]] = []
    for factor in expected:
        required = _required_fields_for_factor(factor)
        missing = [column for column in required if column not in available]
        if missing:
            unsupported.append(
                {
                    "factor": factor,
                    "required_fields": list(required),
                    "reason": "missing_required_fields",
                    "missing_fields": missing,
                }
            )
    return tuple(unsupported)


def calculate_literature_price_volume_factors(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_LITERATURE_WINDOWS,
    expected_columns: tuple[str, ...] | None = None,
    asset_column: str = "stock_code",
    date_column: str = "trade_date",
    price_column: str = "adj_close",
    return_column: str = "return_1d",
    amount_column: str = "amount",
    turnover_column: str = "turnover",
    high_column: str = "high",
    low_column: str = "low",
    prev_close_column: str = "prev_close",
    float_market_cap_column: str = "float_market_cap",
    total_market_cap_column: str = "total_market_cap",
) -> pd.DataFrame:
    """Add 0.1 literature-prior price-volume factors without future data."""

    requested = set(expected_columns or _all_literature_factor_columns(windows))
    required_windows = _windows_from_requested_factors(requested, windows=windows)
    result = frame.sort_values([asset_column, date_column]).copy()
    grouped = result.groupby(asset_column, sort=False)
    price = pd.to_numeric(result[price_column], errors="coerce")
    returns = pd.to_numeric(result[return_column], errors="coerce")
    amount = pd.to_numeric(result[amount_column], errors="coerce")
    turnover = pd.to_numeric(result[turnover_column], errors="coerce")

    if "rev_1d" in requested:
        result["rev_1d"] = -returns
    for window in required_windows:
        if window == 1:
            continue
        needs_momentum = f"mom_{window}d" in requested
        needs_reversal = f"rev_{window}d" in requested
        if needs_momentum or needs_reversal:
            shifted_price = price.groupby(result[asset_column], sort=False).shift(
                window
            )
            trailing_return = price / shifted_price - 1.0
        if needs_momentum:
            result[f"mom_{window}d"] = trailing_return
        if needs_reversal:
            result[f"rev_{window}d"] = -trailing_return

    for window in required_windows:
        if f"vol_{window}d" in requested:
            result[f"vol_{window}d"] = returns.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).std()
            )
        if f"downside_vol_{window}d" in requested:
            result[f"downside_vol_{window}d"] = _rolling_rms(
                returns.clip(upper=0.0),
                result[asset_column],
                window,
            )
        if f"upside_vol_{window}d" in requested:
            result[f"upside_vol_{window}d"] = _rolling_rms(
                returns.clip(lower=0.0),
                result[asset_column],
                window,
            )
        if f"max_return_{window}d" in requested:
            result[f"max_return_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).max()
            )
        if f"min_return_{window}d" in requested:
            result[f"min_return_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).min()
            )
        if f"drawdown_{window}d" in requested:
            result[f"drawdown_{window}d"] = _rolling_drawdown(
                pd.to_numeric(result[price_column], errors="coerce"),
                result[asset_column],
                window,
            )
        if f"skew_{window}d" in requested:
            result[f"skew_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).skew()
            )
        if f"kurt_{window}d" in requested:
            result[f"kurt_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).kurt()
            )

    for window in required_windows:
        if f"avg_amount_{window}d" in requested:
            result[f"avg_amount_{window}d"] = amount.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"turnover_{window}d" in requested:
            result[f"turnover_{window}d"] = turnover.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"turnover_chg_{window}d" in requested:
            shifted_turnover = turnover.groupby(result[asset_column], sort=False).shift(
                window
            )
            denominator = shifted_turnover.where(shifted_turnover != 0.0)
            result[f"turnover_chg_{window}d"] = turnover / denominator - 1.0
        if f"turnover_vol_{window}d" in requested:
            result[f"turnover_vol_{window}d"] = turnover.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).std()
            )
        if f"zero_volume_ratio_{window}d" in requested:
            result[f"zero_volume_ratio_{window}d"] = (
                (pd.to_numeric(result[amount_column], errors="coerce") <= 0.0)
                .astype(float)
                .groupby(result[asset_column], sort=False)
                .transform(
                    lambda series, w=window: series.rolling(w, min_periods=w).mean()
                )
            )

    denominator = amount.where(amount != 0.0)
    illiquidity = (returns.abs() / denominator).replace([np.inf, -np.inf], np.nan)
    amplitude = (
        pd.to_numeric(result[high_column], errors="coerce")
        - pd.to_numeric(result[low_column], errors="coerce")
    ) / pd.to_numeric(result[prev_close_column], errors="coerce")
    for window in required_windows:
        if f"amihud_{window}d" in requested:
            result[f"amihud_{window}d"] = illiquidity.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"amplitude_{window}d" in requested:
            result[f"amplitude_{window}d"] = amplitude.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )

    intraday_range = (
        pd.to_numeric(result[high_column], errors="coerce")
        - pd.to_numeric(result[low_column], errors="coerce")
    ) / pd.to_numeric(result[price_column], errors="coerce").replace(0.0, np.nan)
    for window in required_windows:
        if f"intraday_range_{window}d" in requested:
            result[f"intraday_range_{window}d"] = intraday_range.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )

    if (
        "log_float_market_cap" in requested
        and float_market_cap_column in result.columns
    ):
        result["log_float_market_cap"] = _safe_log(result[float_market_cap_column])
    if (
        "log_total_market_cap" in requested
        and total_market_cap_column in result.columns
    ):
        result["log_total_market_cap"] = _safe_log(result[total_market_cap_column])
    return result


def select_literature_price_volume_features(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select preset 0.1 feature columns and exclude labels/filters/leakage."""

    return select_literature_price_volume_feature_columns_from_schema(
        tuple(frame.columns),
        expected_columns=expected_columns,
        include_actual_factor_columns=expected_columns is None,
    )


def select_literature_price_volume_feature_columns_from_schema(
    columns: tuple[str, ...] | list[str] | set[str],
    *,
    expected_columns: tuple[str, ...] | None = None,
    include_actual_factor_columns: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve preset 0.1 feature columns from schema names only."""

    schema_columns = tuple(columns)
    expected = expected_columns or expected_literature_price_volume_factor_columns()
    actual_factor_columns = (
        ()
        if not include_actual_factor_columns
        else tuple(
            column
            for column in schema_columns
            if column.casefold().startswith(LITERATURE_FACTOR_PREFIXES)
        )
    )
    columns = tuple(
        dict.fromkeys(
            (
                *actual_factor_columns,
                *(column for column in expected if column in schema_columns),
            )
        )
    )
    warnings_out: list[str] = [
        f"missing expected feature column: {column}"
        for column in expected
        if column not in schema_columns
    ]
    selected = tuple(column for column in columns if not _is_excluded_feature(column))
    if not selected:
        raise ValueError("invalid_feature_preset: no usable feature columns remain")
    dropped = [column for column in columns if column not in selected]
    warnings_out.extend(
        f"excluded leakage/control column: {column}" for column in dropped
    )
    return selected, tuple(warnings_out)


def largest_literature_price_volume_lookback_days(
    expected_columns: tuple[str, ...] | list[str] | set[str],
) -> int:
    """Return the largest rolling/trailing lookback encoded in factor names."""

    windows = []
    for column in expected_columns:
        if column == "rev_1d":
            windows.append(1)
        windows.extend(int(value) for value in re.findall(r"_(\d+)d(?:_|$)", column))
    return max(windows, default=0)


def warn_for_missing_literature_features(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Emit Python warnings for missing preset columns and return messages."""

    _, messages = select_literature_price_volume_features(
        frame,
        expected_columns=expected_columns,
    )
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return messages


def _rolling_rms(values: pd.Series, group_keys: pd.Series, window: int) -> pd.Series:
    squared = values.pow(2)
    return squared.groupby(group_keys, sort=False).transform(
        lambda series, w=window: np.sqrt(series.rolling(w, min_periods=w).mean())
    )


def _rolling_drawdown(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    def drawdown(series: pd.Series) -> float:
        peak = series.cummax()
        return float((series / peak - 1.0).min())

    return values.groupby(group_keys, sort=False).transform(
        lambda series, w=window: series.rolling(w, min_periods=w).apply(
            drawdown,
            raw=False,
        )
    )


def _safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log(numeric.where(numeric > 0.0))


def _is_excluded_feature(column: str) -> bool:
    normalized = column.casefold()
    if column in LITERATURE_FEATURE_EXCLUDE_COLUMNS:
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in LITERATURE_FEATURE_EXCLUDE_PREFIXES
    )


def _required_fields_for_factor(factor: str) -> tuple[str, ...]:
    if factor in BASE_FACTOR_REQUIREMENTS:
        return BASE_FACTOR_REQUIREMENTS[factor]
    for prefix, required in BASE_FACTOR_REQUIREMENTS.items():
        if factor.startswith(f"{prefix}_"):
            return required
    return ()


def _windows_from_requested_factors(
    requested: set[str],
    *,
    windows: tuple[int, ...],
) -> tuple[int, ...]:
    values = {1} if "rev_1d" in requested else set()
    for factor in requested:
        for token in factor.split("_"):
            if token.endswith("d") and token[:-1].isdigit():
                values.add(int(token[:-1]))
    return tuple(window for window in windows if window in values)


def _all_literature_factor_columns(windows: tuple[int, ...]) -> tuple[str, ...]:
    columns: list[str] = ["rev_1d"]
    for window in windows:
        if window != 1:
            columns.extend((f"rev_{window}d", f"mom_{window}d"))
        columns.extend(
            (
                f"vol_{window}d",
                f"downside_vol_{window}d",
                f"upside_vol_{window}d",
                f"max_return_{window}d",
                f"min_return_{window}d",
                f"drawdown_{window}d",
                f"skew_{window}d",
                f"kurt_{window}d",
                f"avg_amount_{window}d",
                f"turnover_{window}d",
                f"turnover_chg_{window}d",
                f"turnover_vol_{window}d",
                f"amihud_{window}d",
                f"zero_volume_ratio_{window}d",
                f"amplitude_{window}d",
                f"intraday_range_{window}d",
            )
        )
    columns.extend(("log_float_market_cap", "log_total_market_cap"))
    return tuple(dict.fromkeys(columns))


__all__ = [
    "CORE_FACTORS",
    "DEFAULT_LITERATURE_WINDOWS",
    "LITERATURE_FACTOR_SETS",
    "LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID",
    "LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID",
    "SMOKE_FACTORS",
    "calculate_literature_price_volume_factors",
    "expected_literature_price_volume_factor_columns",
    "exact_literature_price_volume_factor_input_columns",
    "largest_literature_price_volume_lookback_days",
    "select_literature_price_volume_feature_columns_from_schema",
    "select_literature_price_volume_features",
    "unsupported_literature_price_volume_factors",
    "warn_for_missing_literature_features",
]
