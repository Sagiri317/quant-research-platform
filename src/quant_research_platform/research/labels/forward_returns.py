"""A-share return labels on an explicit session axis and real exchange clocks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_hashable_key_series,
    strict_timestamp_series,
)


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    execution_price_column: str = "adj_open",
    sessions: Sequence[object],
) -> pd.DataFrame:
    """Add fixed-session real-open returns independently of execution permissions.

    Endpoints require real observed quotes; suspension valuation carries are not
    price observations. Interior missing quotes never shift either endpoint.
    The wd-adjusted opening price depends on that session's closing inputs.
    Availability is therefore the exit session's 15:00 Asia/Shanghai close,
    represented as a UTC-naive instant, without changing the opening endpoints.
    """

    horizons = _validate_horizons(horizons)
    validated = _validate_label_input_frame(
        frame,
        asset_column=asset_column,
        date_column=date_column,
        execution_price_column=execution_price_column,
    )
    axis = _session_axis(sessions)
    return _forward_return_labels(
        validated,
        axis,
        horizons,
        asset_column=asset_column,
        date_column=date_column,
        execution_price_column=execution_price_column,
    )


def _forward_return_labels(
    validated,
    axis,
    horizons,
    *,
    asset_column="asset_id",
    date_column="timestamp",
    execution_price_column="adj_open",
):
    """Calculate endpoints after the caller validates keys and the session axis."""
    result = validated.sort_values([asset_column, date_column], kind="mergesort")
    price = strict_finite_numeric_series(
        result[execution_price_column],
        error="invalid_forward_return_price",
        allow_null=True,
    ).astype("float64")
    keys = pd.MultiIndex.from_arrays([result[asset_column], result[date_column]])
    positions = axis.get_indexer(result[date_column])
    prices = price.to_numpy(dtype="float64", copy=False)
    real_quote, _ = _quote_observation_masks(result)
    assets = result[asset_column].to_numpy(copy=False)
    entry_positions = np.flatnonzero((positions >= 0) & (positions + 1 < len(axis)))
    entry_rows = keys.get_indexer(
        pd.MultiIndex.from_arrays(
            [assets[entry_positions], axis.take(positions[entry_positions] + 1)]
        )
    )
    present = entry_rows >= 0
    entry_positions, entry_rows = entry_positions[present], entry_rows[present]
    entry_prices = prices[entry_rows]
    positive = np.isfinite(entry_prices) & (entry_prices > 0) & real_quote[entry_rows]
    entry_positions, entry_prices = entry_positions[positive], entry_prices[positive]
    for horizon in horizons:
        labels = np.full(len(result), np.nan, dtype="float64")
        available = np.full(len(result), np.datetime64("NaT"), dtype="datetime64[ns]")
        mature = positions[entry_positions] + 1 + horizon < len(axis)
        valid_positions = entry_positions[mature]
        exit_dates = axis.take(positions[valid_positions] + 1 + horizon)
        exit_rows = keys.get_indexer(
            pd.MultiIndex.from_arrays([assets[valid_positions], exit_dates])
        )
        present = exit_rows >= 0
        valid_positions = valid_positions[present]
        exit_dates = exit_dates[present]
        entries, exit_prices = entry_prices[mature][present], prices[exit_rows[present]]
        positive = (
            np.isfinite(exit_prices)
            & (exit_prices > 0)
            & real_quote[exit_rows[present]]
        )
        valid_positions, exit_dates = valid_positions[positive], exit_dates[positive]
        with np.errstate(over="ignore"):
            values = (exit_prices[positive] - entries[positive]) / entries[positive]
        finite = np.isfinite(values)
        labels[valid_positions[finite]] = values[finite]
        available[valid_positions[finite]] = _a_share_session_event(
            exit_dates[finite], hour=15
        )
        label_column = forward_return_label_column(horizon)
        result[label_column] = labels
        result[f"{label_column}_available_at"] = available
    return result.reset_index(drop=True)


def _quote_observation_masks(frame):
    """Validate quote facts locally without changing their values or dtypes."""
    missing = {"filling_flag", "is_trading"}.difference(frame.columns)
    if missing:
        raise ValueError(
            "missing_forward_return_columns: " + ", ".join(sorted(missing))
        )
    filling = strict_finite_numeric_series(
        frame.filling_flag, error="invalid_forward_return_filling_flag", allow_null=True
    )
    trading = strict_bool_series(
        frame.is_trading, error="invalid_forward_return_is_trading", allow_null=True
    )
    return (
        (filling.eq(0) & trading.fillna(False)).fillna(False).to_numpy(dtype=bool),
        (filling.eq(1) & trading.eq(False)).fillna(False).to_numpy(dtype=bool),
    )


def forward_return_label_column(horizon: int) -> str:
    _validate_horizon(horizon, error="invalid_forward_return_horizon")
    return f"forward_return_{horizon}d"


def forward_return_available_at_column(horizon: int) -> str:
    return f"{forward_return_label_column(horizon)}_available_at"


def _a_share_session_event(sessions, *, hour, minute=0):
    """Convert exchange-local session events to canonical UTC-naive instants."""
    return (
        (sessions + pd.Timedelta(hours=hour, minutes=minute))
        .tz_localize("Asia/Shanghai")
        .tz_convert("UTC")
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )


def _session_axis(sessions: Sequence[object]) -> pd.DatetimeIndex:
    values = strict_timestamp_series(
        pd.Series(tuple(sessions)), error="invalid_forward_return_sessions"
    ).dt.normalize()
    axis = pd.DatetimeIndex(values).drop_duplicates()
    if axis.empty or not axis.is_monotonic_increasing:
        raise ValueError("invalid_forward_return_sessions")
    return axis


def _validate_horizons(horizons: object) -> tuple[int, ...]:
    if not isinstance(horizons, tuple) or not horizons:
        raise ValueError("invalid_forward_return_horizons")
    values = tuple(
        _validate_horizon(value, error="invalid_forward_return_horizons")
        for value in horizons
    )
    if len(set(values)) != len(values):
        raise ValueError("invalid_forward_return_horizons")
    return values


def _validate_horizon(value: object, *, error: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(error)
    return value


def _validate_label_input_frame(
    frame: pd.DataFrame,
    *,
    asset_column: str,
    date_column: str,
    execution_price_column: str,
) -> pd.DataFrame:
    required_columns = (asset_column, date_column, execution_price_column)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing_forward_return_columns: {', '.join(missing)}")
    result = frame.copy()
    if any(isinstance(value, (list, dict, set)) for value in result[asset_column]):
        raise ValueError("invalid_forward_return_asset_key")
    strict_hashable_key_series(
        result[asset_column], error="invalid_forward_return_asset_key"
    )
    result[date_column] = strict_timestamp_series(
        result[date_column], error="invalid_forward_return_timestamp"
    )
    if result.duplicated([asset_column, date_column]).any():
        raise ValueError("duplicate_forward_return_asset_timestamp")
    return result


def add_on_id_decomposition(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    sessions: Sequence[object],
) -> pd.DataFrame:
    """Decompose adjusted prices on an explicit session and listing-lifecycle axis.

    Each component has its own price dependencies. Availability includes the
    wd adjustment's closing inputs, represented as UTC-naive Shanghai exchange
    events; this does not certify vendor publication times.
    """
    from quant_research_platform.contracts.validation import strict_real_numeric_series

    horizons = _validate_horizons(horizons)
    result = _validate_label_input_frame(
        frame,
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
    )
    required = {"adj_close", "listing_date", "filling_flag", "is_trading"}
    if required - set(result):
        raise ValueError("missing_on_id_columns")
    raw_axis = strict_timestamp_series(
        pd.Series(tuple(sessions)), error="invalid_on_id_sessions"
    )
    axis = pd.DatetimeIndex(raw_axis)
    if (
        axis.empty
        or axis.has_duplicates
        or not axis.is_monotonic_increasing
        or not axis.equals(axis.normalize())
        or not result.timestamp.isin(axis).all()
    ):
        raise ValueError("invalid_on_id_sessions")
    result["listing_date"] = strict_timestamp_series(
        result.listing_date, error="invalid_on_id_listing_date", allow_null=True
    )
    for column in ("adj_open", "adj_close"):
        result[column] = strict_real_numeric_series(
            result[column], error="invalid_on_id_price_type"
        )
    real_quote, suspended = _quote_observation_masks(result)
    result.loc[~(real_quote | suspended), ["adj_open", "adj_close"]] = np.nan
    pieces = [
        _on_id_asset_decomposition(group, axis, horizons)
        for _, group in result.groupby("asset_id", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else result


def _on_id_asset_decomposition(group, axis, horizons):
    # Preserve every intervening session and the future dependency context,
    # including missing rows whose lifecycle reasons precede price failures.
    # One preceding session preserves reindex's missing-value dtype promotion.
    start = max(0, axis.searchsorted(group.timestamp.min()) - 1)
    stop = axis.searchsorted(group.timestamp.max(), side="right") + max(horizons) + 1
    axis = axis[start:stop]
    block = group.sort_values("timestamp").set_index("timestamp").reindex(axis)
    n = len(block)
    prices = block[["adj_open", "adj_close"]].to_numpy(dtype="float64")
    missing = np.isnan(prices)
    real_quote, _ = _quote_observation_masks(block)
    invalid = ~np.isfinite(prices) & ~missing | (prices <= 0)
    day = _on_id_log_ratio(prices[:, 1], prices[:, 0])
    night = np.full(n, np.nan)
    night[:-1] = _on_id_log_ratio(prices[1:, 0], prices[:-1, 1])
    day_missing, day_invalid = missing.any(axis=1), invalid.any(axis=1)
    night_missing = np.r_[missing[1:, 0] | missing[:-1, 1], False]
    night_invalid = np.r_[invalid[1:, 0] | invalid[:-1, 1], False]
    life = block.listing_date.to_numpy(dtype="datetime64[ns]")
    life_reasons = {}

    def lifecycle_reason(offset):
        if offset not in life_reasons:
            reasons = np.full(n, "insufficient_future_sessions", dtype=object)
            count = max(0, n - offset)
            if count:
                dependencies = np.lib.stride_tricks.sliding_window_view(
                    life, offset + 1
                )
                unknown = np.isnat(dependencies).any(axis=1)
                crossed = (
                    (dependencies != life[:count, None])
                    & ~np.isnat(dependencies)
                    & ~np.isnat(life[:count, None])
                ).any(axis=1)
                reasons[:count] = None
                reasons[:count][unknown] = "unknown_listing_lifecycle"
                reasons[:count][crossed] = "lifecycle_boundary"
            reasons[np.isnat(life)] = "unknown_listing_lifecycle"
            life_reasons[offset] = reasons
        return life_reasons[offset].copy()

    def shifted(values, offset, fill):
        output = np.full(n, fill, dtype=values.dtype)
        if offset < n:
            output[: n - offset] = values[offset:]
        return output

    def window(values, start, width, *, total=False):
        output = np.full(n, np.nan if total else False, dtype=float if total else bool)
        count = max(0, n - start - width + 1)
        if count:
            windows = np.lib.stride_tricks.sliding_window_view(values[start:], width)
            output[:count] = windows.sum(axis=1) if total else windows.any(axis=1)
        return output

    def publish(
        name,
        values,
        price_missing,
        price_invalid,
        offset,
        hour,
        minute=0,
    ):
        reasons = lifecycle_reason(offset)
        reasons[pd.isna(reasons) & price_missing] = "missing_price"
        reasons[(pd.isna(reasons) | (reasons == "missing_price")) & price_invalid] = (
            "invalid_price"
        )
        legal = pd.isna(reasons)
        reasons[legal & ~np.isfinite(values)] = "nonfinite_result"
        legal = pd.isna(reasons)
        values = np.where(legal, values, np.nan).astype("float64")
        maturity = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        count = max(0, n - offset)
        if count:
            event = _a_share_session_event(axis[offset:], hour=hour, minute=minute)
            maturity[:count] = np.where(legal[:count], event, np.datetime64("NaT"))
        block[name] = values
        block[f"{name}_available_at"] = maturity
        block[f"{name}_reason"] = pd.array(reasons, dtype="string")

    publish("on_id_pre_entry_log_return", night, night_missing, night_invalid, 1, 15)
    endpoint_parts = []
    for _, lifecycle in group.groupby("listing_date", sort=False, dropna=False):
        clean = lifecycle.copy()
        clean["adj_open"] = clean.adj_open.where(
            np.isfinite(clean.adj_open) & clean.adj_open.gt(0)
        )
        endpoint_parts.append(_forward_return_labels(clean, axis, horizons))
    endpoints = pd.concat(endpoint_parts).set_index("timestamp").reindex(axis)
    for horizon in horizons:
        intraday = f"on_id_holding_intraday_log_return_{horizon}d"
        overnight = f"on_id_holding_overnight_log_return_{horizon}d"
        endpoint = f"on_id_endpoint_return_{horizon}d"
        publish(
            intraday,
            window(day, 1, horizon, total=True),
            window(day_missing, 1, horizon),
            window(day_invalid, 1, horizon),
            horizon,
            15,
        )
        publish(
            overnight,
            window(night, 1, horizon, total=True),
            window(night_missing, 1, horizon),
            window(night_invalid, 1, horizon),
            horizon + 1,
            15,
        )
        publish(
            endpoint,
            endpoints[forward_return_label_column(horizon)].to_numpy(dtype=float),
            shifted(missing[:, 0] | ~real_quote, 1, False)
            | shifted(missing[:, 0] | ~real_quote, horizon + 1, False),
            shifted(invalid[:, 0], 1, False)
            | shifted(invalid[:, 0], horizon + 1, False),
            horizon + 1,
            15,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            reconstructed = np.expm1(block[intraday] + block[overnight])
        comparable = np.isfinite(block[[intraday, overnight, endpoint]]).all(
            axis=1
        ) & np.isfinite(reconstructed)
        # Bound roundoff in cancelling log legs as well as the endpoint scale.
        magnitude = window(np.abs(day), 1, horizon, total=True) + window(
            np.abs(night), 1, horizon, total=True
        )
        tolerance = (
            1e-12
            + 1e-10 * np.abs(block[endpoint])
            + 8 * np.finfo(float).eps * magnitude * (1 + np.abs(block[endpoint]))
        )
        if ((np.abs(reconstructed - block[endpoint]) > tolerance) & comparable).any():
            raise ValueError("on_id_endpoint_mismatch")
        block[f"on_id_decomposition_qualified_{horizon}d"] = comparable.to_numpy(
            dtype=bool
        )
        reasons = (
            block[f"{endpoint}_reason"]
            .fillna(block[f"{intraday}_reason"])
            .fillna(block[f"{overnight}_reason"])
        )
        reasons = reasons.mask(
            ~np.isfinite(reconstructed) & reasons.isna(), "nonfinite_result"
        )
        block[f"on_id_decomposition_reason_{horizon}d"] = reasons.astype("string")
    block.index.name = "timestamp"
    return block.loc[group.timestamp].reset_index()


def _on_id_log_ratio(numerator, denominator):
    """Preserve tiny relative moves without overflowing wide finite ratios."""
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > 0)
        & (denominator > 0)
    )
    result = np.full(len(numerator), np.nan)
    near = valid & (numerator * 0.5 <= denominator) & (denominator * 0.5 <= numerator)
    result[near] = np.log1p((numerator[near] - denominator[near]) / denominator[near])
    far = valid & ~near
    result[far] = np.log(numerator[far]) - np.log(denominator[far])
    return result
