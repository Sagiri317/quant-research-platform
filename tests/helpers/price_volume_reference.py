from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.research.factors.price_volume.columns import (
    price_volume_window_column,
)
from tests.helpers.panel_keys import DEFAULT_PANEL_KEY_COLUMNS

KEY_COLUMNS = DEFAULT_PANEL_KEY_COLUMNS
FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12
REFERENCE_ROLLING_MOMENT_NEAR_ZERO_VARIANCE = 1e-12
MOMENTUM_20D_COLUMN = price_volume_window_column("mom", 20)
REVERSAL_1D_COLUMN = price_volume_window_column("rev", 1)


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
                    "period_return": 0.001 * asset_index + wave,
                    "adj_close": close,
                    "adj_open": close * 0.997,
                    "open": close * 0.997,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "prev_close": close / (1.0 + 0.001 * asset_index + wave),
                    "volume": 8_000.0 + date_index * 3 + asset_index,
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
    grouped = result.groupby("asset_id", sort=False)
    price = pd.to_numeric(result["adj_close"], errors="coerce")
    returns = pd.to_numeric(result["period_return"], errors="coerce")
    amount = pd.to_numeric(result["amount"], errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce")
    turnover = pd.to_numeric(result["turnover"], errors="coerce")
    requested = tuple(dict.fromkeys(factor_columns))
    requested_set = set(requested)
    required_windows = reference_windows_from_requested_factors(requested_set)

    if REVERSAL_1D_COLUMN in requested_set:
        result[REVERSAL_1D_COLUMN] = -returns
    for window in required_windows:
        if window == 1:
            continue
        needs_momentum = f"mom_{window}d" in requested_set
        needs_reversal = f"rev_{window}d" in requested_set
        if needs_momentum or needs_reversal:
            shifted_price = price.groupby(group_keys, sort=False).shift(window)
            trailing_return = price / referencevalid_denominator(shifted_price) - 1.0
            trailing_return = referencefinite_or_nan(trailing_return)
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
            result[f"downside_vol_{window}d"] = referencerolling_rms(
                returns.clip(upper=0.0),
                group_keys,
                window,
            )
        if f"upside_vol_{window}d" in requested_set:
            result[f"upside_vol_{window}d"] = referencerolling_rms(
                returns.clip(lower=0.0),
                group_keys,
                window,
            )
        if f"max_return_{window}d" in requested_set:
            result[f"max_return_{window}d"] = grouped["period_return"].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).max()
            )
        if f"min_return_{window}d" in requested_set:
            result[f"min_return_{window}d"] = grouped["period_return"].transform(
                lambda series, w=window: series.rolling(w, min_periods=w).min()
            )
        if f"drawdown_{window}d" in requested_set:
            result[f"drawdown_{window}d"] = referencerolling_drawdown(
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
                turnover / referencevalid_denominator(shifted_turnover) - 1.0
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
        elif f"amount_chg_{window}d" == column:
            shifted_amount = amount.groupby(group_keys, sort=False).shift(window)
            result[column] = amount / referencevalid_denominator(shifted_amount) - 1.0
        elif f"amount_shock_{window}d" == column:
            mean_amount = amount.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            result[column] = amount / referencevalid_denominator(mean_amount) - 1.0
        elif f"amount_vol_{window}d" == column:
            result[column] = reference_qrp_window_moment_by_asset(
                amount,
                group_keys,
                window,
                "std",
            )
        elif f"volume_chg_{window}d" == column:
            shifted_volume = volume.groupby(group_keys, sort=False).shift(window)
            result[column] = volume / referencevalid_denominator(shifted_volume) - 1.0
        elif f"volume_shock_{window}d" == column:
            mean_volume = volume.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            result[column] = volume / referencevalid_denominator(mean_volume) - 1.0
        elif f"ret_amount_corr_{window}d" == column:
            result[column] = reference_rolling_corr(
                returns,
                amount,
                group_keys,
                window,
            )
        elif f"ret_volume_corr_{window}d" == column:
            result[column] = reference_rolling_corr(
                returns,
                volume,
                group_keys,
                window,
            )
        elif f"ret_turnover_corr_{window}d" == column:
            result[column] = reference_rolling_corr(
                returns,
                turnover,
                group_keys,
                window,
            )
        elif f"trend_efficiency_{window}d" == column:
            return_mean = returns.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            abs_return_mean = (
                returns.abs()
                .groupby(group_keys, sort=False)
                .transform(
                    lambda series, w=window: series.rolling(w, min_periods=w).mean()
                )
            )
            result[column] = return_mean.abs() / referencevalid_denominator(
                abs_return_mean
            )
        elif f"signed_amount_flow_{window}d" == column:
            signed_amount = np.sign(returns) * amount
            numerator = signed_amount.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            denominator = amount.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            result[column] = numerator / referencevalid_denominator(denominator)
        elif f"ret_amount_imbalance_{window}d" == column:
            weighted_return = returns * amount
            weighted_abs_return = returns.abs() * amount
            numerator = weighted_return.groupby(group_keys, sort=False).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            denominator = weighted_abs_return.groupby(
                group_keys,
                sort=False,
            ).transform(
                lambda series, w=window: series.rolling(w, min_periods=w).mean()
            )
            result[column] = numerator / referencevalid_denominator(denominator)

    denominator = referencevalid_denominator(amount)
    illiquidity = (returns.abs() / denominator).replace([np.inf, -np.inf], np.nan)
    amplitude = (
        pd.to_numeric(result["high"], errors="coerce")
        - pd.to_numeric(result["low"], errors="coerce")
    ) / referencevalid_denominator(pd.to_numeric(result["prev_close"], errors="coerce"))
    amplitude = referencefinite_or_nan(amplitude)
    intraday_range = (
        pd.to_numeric(result["high"], errors="coerce")
        - pd.to_numeric(result["low"], errors="coerce")
    ) / referencevalid_denominator(pd.to_numeric(result["adj_close"], errors="coerce"))
    intraday_range = referencefinite_or_nan(intraday_range)
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
        result["log_float_market_cap"] = referencesafe_log(result["float_market_cap"])
    if "log_total_market_cap" in requested_set:
        result["log_total_market_cap"] = referencesafe_log(result["total_market_cap"])
    output_columns = [column for column in requested if column in result.columns]
    if output_columns:
        result.loc[:, output_columns] = result.loc[:, output_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )
    return result.loc[:, [*KEY_COLUMNS, *requested]]


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


def referencerolling_rms(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    squared = values.pow(2)
    mean_squared = squared.groupby(group_keys, sort=False).transform(
        lambda series, w=window: series.rolling(w, min_periods=w).mean()
    )
    return np.sqrt(mean_squared)


def referencerolling_drawdown(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    def calculate(series: pd.Series) -> float:
        peak = series.cummax()
        drawdown = series / peak.replace(0.0, np.nan) - 1.0
        return float(referencefinite_or_nan(drawdown).min())

    return values.groupby(group_keys, sort=False).transform(
        lambda series, w=window: series.rolling(w, min_periods=w).apply(
            calculate,
            raw=False,
        )
    )


def reference_rolling_corr(
    left: pd.Series,
    right: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    pairs = pd.DataFrame({"left": left, "right": right, "asset_id": group_keys})
    parts = []
    for _, group in pairs.groupby("asset_id", sort=False):
        parts.append(
            group["left"].rolling(window, min_periods=window).corr(group["right"])
        )
    return pd.concat(parts).sort_index()


def referencesafe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric) & numeric.gt(0.0))
    return referencefinite_or_nan(np.log(numeric))


def referencefinite_or_nan(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)


def referencevalid_denominator(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric) & numeric.ne(0.0))


def reference_windows_from_requested_factors(requested: set[str]) -> tuple[int, ...]:
    values = {1} if REVERSAL_1D_COLUMN in requested else set()
    for factor in requested:
        for token in factor.split("_"):
            if token.endswith("d") and token[:-1].isdigit():
                values.add(int(token[:-1]))
    return tuple(sorted(values))


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
