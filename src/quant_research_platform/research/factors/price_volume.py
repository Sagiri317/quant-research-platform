"""Reusable price-volume factor definitions and calculations."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

PRICE_VOLUME_FACTOR_FAMILY = "price_volume"

PRICE_VOLUME_STANDARD_PRESET = "standard"
PRICE_VOLUME_CORE_PRESET = "core"
PRICE_VOLUME_SMOKE_PRESET = "smoke"

_PRICE_VOLUME_STANDARD_WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 60, 120, 252)

CORE_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
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
SMOKE_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
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

PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "forward_return",
    "target",
    "label",
)
PRICE_VOLUME_FACTOR_PREFIXES: tuple[str, ...] = (
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
PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {
        "timestamp",
        "asset",
    }
)
BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS: dict[str, tuple[str, ...]] = {
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


def price_volume_factor_columns_for_windows(
    windows: Iterable[int],
) -> tuple[str, ...]:
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


STANDARD_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    price_volume_factor_columns_for_windows(_PRICE_VOLUME_STANDARD_WINDOWS)
)

PRICE_VOLUME_FACTOR_PRESETS: dict[str, tuple[str, ...]] = {
    PRICE_VOLUME_STANDARD_PRESET: STANDARD_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_CORE_PRESET: CORE_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_SMOKE_PRESET: SMOKE_PRICE_VOLUME_FACTORS,
}


def price_volume_factor_columns(*, preset: str = "standard") -> tuple[str, ...]:
    """Return the official price-volume factor columns for a named preset."""

    try:
        return PRICE_VOLUME_FACTOR_PRESETS[preset]
    except KeyError as exc:
        raise ValueError("invalid_price_volume_preset") from exc


def price_volume_factor_definitions(
    *,
    market_id: MarketId,
    required_processed_data_id: str,
    preset: str = "standard",
) -> tuple[FactorDefinition, ...]:
    """Return factor definitions for one market binding."""

    definitions: list[FactorDefinition] = []
    for factor_id in price_volume_factor_columns(preset=preset):
        base_inputs = price_volume_factor_input_columns(factor_id)
        lookback_window = _lookback_window(factor_id)
        definitions.append(
            FactorDefinition(
                factor_id=factor_id,
                market_id=market_id,
                factor_set=_factor_set_for_column(factor_id),
                input_columns=base_inputs,
                base_input_columns=base_inputs,
                derived_input_columns=(),
                output_column=factor_id,
                display_name=factor_id,
                description=(
                    "Daily price-volume factor registered by the QRP 0.1 "
                    "factor catalog."
                ),
                family=PRICE_VOLUME_FACTOR_FAMILY,
                required_windows=_required_windows(factor_id),
                lookback_window=lookback_window,
                grain=DataGrain.ASSET_TIMESTAMP,
                timing_semantics="same_session_close_available_at_next_session",
                null_policy="preserve_nulls_after_min_periods",
                source_layer="processed",
                required_processed_data_id=required_processed_data_id,
                window=lookback_window,
            )
        )
    return tuple(definitions)


def price_volume_factor_input_columns(factor_id: str) -> tuple[str, ...]:
    """Return exact base input columns required by one price-volume factor."""

    required = _required_fields_for_factor(factor_id)
    if not required:
        raise ValueError(f"unknown_price_volume_factor: {factor_id}")
    return required


def unsupported_price_volume_factors(
    factor_columns: Iterable[str],
) -> tuple[str, ...]:
    """Return requested factor names that the price-volume engine cannot compute."""

    unsupported: list[str] = []
    for factor in tuple(dict.fromkeys(factor_columns)):
        if not _required_fields_for_factor(factor):
            unsupported.append(factor)
    return tuple(unsupported)


def calculate_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
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
    """Add requested price-volume factors without future data."""

    requested = tuple(
        dict.fromkeys(factor_columns or price_volume_factor_columns())
    )
    unsupported = unsupported_price_volume_factors(requested)
    if unsupported:
        raise ValueError(
            "unsupported_price_volume_factors: " + ", ".join(unsupported)
        )

    required_inputs = tuple(
        dict.fromkeys(
            (
                asset_column,
                date_column,
                *(
                    column
                    for factor in requested
                    for column in price_volume_factor_input_columns(factor)
                ),
            )
        )
    )
    missing_inputs = [
        column for column in required_inputs if column not in panel.columns
    ]
    if missing_inputs:
        raise ValueError(
            "missing_price_volume_input_columns: " + ", ".join(missing_inputs)
        )

    requested_set = set(requested)
    required_windows = _windows_from_requested_factors(requested_set)
    result = panel.sort_values([asset_column, date_column]).copy()
    grouped = result.groupby(asset_column, sort=False)
    price = pd.to_numeric(result[price_column], errors="coerce")
    returns = pd.to_numeric(result[return_column], errors="coerce")
    amount = pd.to_numeric(result[amount_column], errors="coerce")
    turnover = pd.to_numeric(result[turnover_column], errors="coerce")

    if "rev_1d" in requested_set:
        result["rev_1d"] = -returns
    for window in required_windows:
        if window == 1:
            continue
        needs_momentum = f"mom_{window}d" in requested_set
        needs_reversal = f"rev_{window}d" in requested_set
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
        if f"vol_{window}d" in requested_set:
            result[f"vol_{window}d"] = returns.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).std()
            )
        if f"downside_vol_{window}d" in requested_set:
            result[f"downside_vol_{window}d"] = _rolling_rms(
                returns.clip(upper=0.0),
                result[asset_column],
                window,
            )
        if f"upside_vol_{window}d" in requested_set:
            result[f"upside_vol_{window}d"] = _rolling_rms(
                returns.clip(lower=0.0),
                result[asset_column],
                window,
            )
        if f"max_return_{window}d" in requested_set:
            result[f"max_return_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).max()
            )
        if f"min_return_{window}d" in requested_set:
            result[f"min_return_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).min()
            )
        if f"drawdown_{window}d" in requested_set:
            result[f"drawdown_{window}d"] = _rolling_drawdown(
                pd.to_numeric(result[price_column], errors="coerce"),
                result[asset_column],
                window,
            )
        if f"skew_{window}d" in requested_set:
            result[f"skew_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).skew()
            )
        if f"kurt_{window}d" in requested_set:
            result[f"kurt_{window}d"] = grouped[return_column].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).kurt()
            )

    for window in required_windows:
        if f"avg_amount_{window}d" in requested_set:
            result[f"avg_amount_{window}d"] = amount.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"turnover_{window}d" in requested_set:
            result[f"turnover_{window}d"] = turnover.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"turnover_chg_{window}d" in requested_set:
            shifted_turnover = turnover.groupby(result[asset_column], sort=False).shift(
                window
            )
            denominator = shifted_turnover.where(shifted_turnover != 0.0)
            result[f"turnover_chg_{window}d"] = turnover / denominator - 1.0
        if f"turnover_vol_{window}d" in requested_set:
            result[f"turnover_vol_{window}d"] = turnover.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).std()
            )
        if f"zero_volume_ratio_{window}d" in requested_set:
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
        if f"amihud_{window}d" in requested_set:
            result[f"amihud_{window}d"] = illiquidity.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
        if f"amplitude_{window}d" in requested_set:
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
        if f"intraday_range_{window}d" in requested_set:
            result[f"intraday_range_{window}d"] = intraday_range.groupby(
                result[asset_column], sort=False
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )

    if "log_float_market_cap" in requested_set:
        result["log_float_market_cap"] = _safe_log(result[float_market_cap_column])
    if "log_total_market_cap" in requested_set:
        result["log_total_market_cap"] = _safe_log(result[total_market_cap_column])
    return result


def select_price_volume_features(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select preset 0.1 feature columns and exclude labels/filters/leakage."""

    return select_price_volume_feature_columns_from_schema(
        tuple(frame.columns),
        expected_columns=expected_columns,
        include_actual_factor_columns=expected_columns is None,
    )


def select_price_volume_feature_columns_from_schema(
    columns: Iterable[str],
    *,
    expected_columns: tuple[str, ...] | None = None,
    include_actual_factor_columns: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve preset 0.1 feature columns from schema names only."""

    schema_columns = tuple(columns)
    expected = expected_columns or price_volume_factor_columns()
    actual_factor_columns = (
        ()
        if not include_actual_factor_columns
        else tuple(
            column
            for column in schema_columns
            if column.casefold().startswith(PRICE_VOLUME_FACTOR_PREFIXES)
        )
    )
    candidates = tuple(
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
    selected = tuple(
        column for column in candidates if not _is_excluded_feature(column)
    )
    if not selected:
        raise ValueError("invalid_feature_preset: no usable feature columns remain")
    dropped = [column for column in candidates if column not in selected]
    warnings_out.extend(
        f"excluded leakage/control column: {column}" for column in dropped
    )
    return selected, tuple(warnings_out)


def largest_price_volume_lookback_days(
    factor_columns: Iterable[str],
) -> int:
    """Return the largest rolling/trailing lookback encoded in factor names."""

    windows = []
    for column in factor_columns:
        if column == "rev_1d":
            windows.append(1)
        windows.extend(int(value) for value in re.findall(r"_(\d+)d(?:_|$)", column))
    return max(windows, default=0)


def warn_for_missing_price_volume_features(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Emit Python warnings for missing preset columns and return messages."""

    _, messages = select_price_volume_features(
        frame,
        expected_columns=expected_columns,
    )
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return messages


def _factor_set_for_column(factor_id: str) -> FactorSet:
    if factor_id in SMOKE_PRICE_VOLUME_FACTORS:
        return FactorSet.SMOKE
    if factor_id in CORE_PRICE_VOLUME_FACTORS:
        return FactorSet.CORE
    return FactorSet.STANDARD


def _required_windows(factor_id: str) -> tuple[int, ...]:
    windows = tuple(int(match) for match in re.findall(r"_(\d+)d\b", factor_id))
    return windows


def _lookback_window(factor_id: str) -> int | None:
    windows = _required_windows(factor_id)
    if not windows:
        return None
    return max(windows)


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
    if column in PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS:
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES
    )


def _required_fields_for_factor(factor: str) -> tuple[str, ...]:
    if factor in BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS:
        return BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS[factor]
    for prefix, required in BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS.items():
        if factor.startswith(f"{prefix}_"):
            return required
    return ()


def _windows_from_requested_factors(requested: set[str]) -> tuple[int, ...]:
    values = {1} if "rev_1d" in requested else set()
    for factor in requested:
        for token in factor.split("_"):
            if token.endswith("d") and token[:-1].isdigit():
                values.add(int(token[:-1]))
    return tuple(sorted(values))


__all__ = [
    "BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS",
    "CORE_PRICE_VOLUME_FACTORS",
    "PRICE_VOLUME_CORE_PRESET",
    "PRICE_VOLUME_FACTOR_FAMILY",
    "PRICE_VOLUME_FACTOR_PRESETS",
    "PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS",
    "PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES",
    "PRICE_VOLUME_FACTOR_PREFIXES",
    "PRICE_VOLUME_SMOKE_PRESET",
    "PRICE_VOLUME_STANDARD_PRESET",
    "SMOKE_PRICE_VOLUME_FACTORS",
    "STANDARD_PRICE_VOLUME_FACTORS",
    "calculate_price_volume_factors",
    "largest_price_volume_lookback_days",
    "price_volume_factor_columns",
    "price_volume_factor_columns_for_windows",
    "price_volume_factor_definitions",
    "price_volume_factor_input_columns",
    "select_price_volume_feature_columns_from_schema",
    "select_price_volume_features",
    "unsupported_price_volume_factors",
    "warn_for_missing_price_volume_features",
]
