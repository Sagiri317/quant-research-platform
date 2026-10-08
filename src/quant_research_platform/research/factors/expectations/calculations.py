"""PIT analyst forecast aggregation without provider-specific assumptions."""

from __future__ import annotations

from fractions import Fraction
from functools import partial
from statistics import median

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)

from .definitions import EXPECTATIONS_FACTORS

FORECAST_FRESHNESS_DAYS = 90


def _prepare_expectations(
    records,
    *,
    trading_sessions,
    freshness_days=FORECAST_FRESHNESS_DAYS,
    minimum_institutions=3,
):
    """Prepare one immutable request's source, then query observation slabs."""
    required_records = {
        "asset_id",
        "declare_date",
        "forecast_end_date",
        "institution_id",
        "forecast_eps",
    }
    if not required_records.issubset(records.columns):
        raise ValueError("expectations_input_columns_invalid")
    if type(freshness_days) is not int or freshness_days <= 0:
        raise ValueError("expectations_freshness_invalid")
    if type(minimum_institutions) is not int or minimum_institutions < 1:
        raise ValueError("expectations_minimum_institutions_invalid")
    sessions = pd.DatetimeIndex(
        strict_timestamp_series(
            pd.Series(tuple(trading_sessions)),
            error="expectations_trading_sessions_invalid",
        ).dt.normalize()
    ).drop_duplicates()
    if sessions.empty or not sessions.is_monotonic_increasing:
        raise ValueError("expectations_trading_sessions_invalid")

    forecast = records.loc[:, list(required_records)].copy()
    forecast["asset_id"] = strict_string_series(
        forecast["asset_id"], error="expectations_asset_id_invalid"
    )
    forecast["institution_id"] = strict_string_series(
        forecast["institution_id"], error="expectations_institution_id_invalid"
    )
    for column in ("declare_date", "forecast_end_date"):
        forecast[column] = strict_timestamp_series(
            forecast[column], error="expectations_date_invalid"
        ).dt.normalize()
    forecast["forecast_eps"] = strict_finite_numeric_series(
        forecast["forecast_eps"], error="expectations_eps_invalid", allow_null=True
    )
    forecast = forecast.loc[
        forecast["forecast_eps"].notna()
        & forecast["forecast_end_date"].dt.month.eq(12)
        & forecast["forecast_end_date"].dt.day.eq(31)
    ].copy()
    forecast["available_at"] = forecast["declare_date"].map(
        lambda value: _next_session(sessions, value)
    )
    forecast = forecast.dropna(subset=["available_at"])
    forecast = forecast.groupby(
        [
            "asset_id",
            "forecast_end_date",
            "institution_id",
            "declare_date",
            "available_at",
        ],
        as_index=False,
        sort=False,
    )["forecast_eps"].agg(_finite_median)
    forecast = forecast.sort_values(
        ["asset_id", "available_at", "declare_date"], kind="mergesort"
    )

    return partial(
        _expectations_for_observations,
        forecast,
        freshness_days=freshness_days,
        minimum_institutions=minimum_institutions,
    )


def _expectations_for_observations(
    forecast,
    observations,
    *,
    freshness_days,
    minimum_institutions,
):
    if not {"asset_id", "timestamp", "close"}.issubset(observations.columns):
        raise ValueError("expectations_input_columns_invalid")
    output = observations.loc[:, ["asset_id", "timestamp", "close"]].copy()
    output["asset_id"] = strict_string_series(
        output["asset_id"], error="expectations_asset_id_invalid"
    )
    output["timestamp"] = strict_timestamp_series(
        output["timestamp"], error="expectations_timestamp_invalid"
    ).dt.normalize()
    output["close"] = strict_finite_numeric_series(
        output["close"], error="expectations_close_invalid", allow_null=True
    )
    if output.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("expectations_observation_key_duplicate")
    rows: list[dict[str, object] | None] = [None] * len(output)
    forecast_groups = forecast.groupby("asset_id", sort=False).indices
    observation_times = output["timestamp"].array
    observation_closes = output["close"].array
    for asset_id, positions in output.groupby("asset_id", sort=False).indices.items():
        asset_forecast = forecast.iloc[forecast_groups.get(asset_id, [])]
        events = iter(asset_forecast.itertuples(index=False))
        event = next(events, None)
        active = {}
        for position in sorted(positions, key=observation_times.__getitem__):
            timestamp = observation_times[position]
            while event is not None and event.available_at <= timestamp:
                active[(event.forecast_end_date, event.institution_id)] = event
                event = next(events, None)
            oldest = timestamp - pd.Timedelta(days=freshness_days)
            active = {
                key: value
                for key, value in active.items()
                if value.declare_date >= oldest and value.forecast_end_date >= timestamp
            }
            rows[position] = _features_for_observation(
                asset_id,
                timestamp,
                observation_closes[position],
                active.values(),
                minimum_institutions=minimum_institutions,
            )
    return pd.DataFrame.from_records(
        rows,
        columns=("asset_id", "timestamp", *EXPECTATIONS_FACTORS),
    )


def _features_for_observation(
    asset_id: str,
    timestamp: pd.Timestamp,
    close: float,
    forecast,
    *,
    minimum_institutions: int,
) -> dict[str, object]:
    base: dict[str, object] = {
        "asset_id": asset_id,
        "timestamp": timestamp,
        **{factor: float("nan") for factor in EXPECTATIONS_FACTORS},
    }
    if pd.isna(close) or close <= 0.0:
        return base
    by_period = {}
    for event in forecast:
        by_period.setdefault(event.forecast_end_date, []).append(event.forecast_eps)
    if not by_period:
        return base
    end_dates = sorted(by_period)
    selected = [by_period[end_dates[0]]]
    if len(end_dates) > 1 and end_dates[1].year == end_dates[0].year + 1:
        selected.append(by_period[end_dates[1]])
    eps_values = selected[0]
    if len(eps_values) >= minimum_institutions:
        exact = eps_values
        center = median(exact)
        mad = float(median(abs(value - center) for value in exact))
        base["forward_earnings_yield_fy1"] = float(center / Fraction(float(close)))
        base["forecast_dispersion_fy1"] = mad
        base["log_analyst_coverage"] = float(np.log(len(eps_values)))
    if len(selected) > 1:
        fy2 = selected[1]
        if len(fy2) >= minimum_institutions:
            base["forward_earnings_yield_fy2"] = float(
                _finite_median(fy2) / Fraction(float(close))
            )
    return base


def _finite_median(values) -> Fraction:
    # Exact dyadic values preserve even centers without overflow or underflow.
    return median(Fraction(value) for value in values)


def _next_session(
    sessions: pd.DatetimeIndex, date_value: pd.Timestamp
) -> pd.Timestamp | pd.NaT:
    position = int(sessions.searchsorted(date_value, side="right"))
    return sessions[position] if position < len(sessions) else pd.NaT
