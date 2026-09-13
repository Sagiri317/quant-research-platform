"""PIT analyst forecast aggregation without provider-specific assumptions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)

from .definitions import EXPECTATIONS_FACTORS


def build_expectations_features(
    records: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    trading_sessions: Iterable[object],
    freshness_days: int = 90,
    minimum_institutions: int = 3,
) -> pd.DataFrame:
    """Aggregate the latest PIT annual forecasts for each asset/session."""

    required_records = {
        "asset_id",
        "declare_date",
        "forecast_end_date",
        "institution_id",
        "forecast_eps",
    }
    required_observations = {"asset_id", "timestamp", "close"}
    if not required_records.issubset(
        records.columns
    ) or not required_observations.issubset(observations.columns):
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
    forecast = forecast.sort_values(
        ["asset_id", "forecast_end_date", "institution_id", "declare_date"],
        kind="mergesort",
    )

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
    for asset_id, positions in output.groupby("asset_id", sort=False).indices.items():
        asset_forecast = forecast.iloc[forecast_groups.get(asset_id, [])]
        for position, observation in zip(
            positions, output.iloc[positions].itertuples(index=False), strict=True
        ):
            rows[position] = _features_for_observation(
                observation.asset_id,
                observation.timestamp,
                observation.close,
                asset_forecast,
                freshness_days=freshness_days,
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
    forecast: pd.DataFrame,
    *,
    freshness_days: int,
    minimum_institutions: int,
) -> dict[str, object]:
    base: dict[str, object] = {
        "asset_id": asset_id,
        "timestamp": timestamp,
        **{factor: float("nan") for factor in EXPECTATIONS_FACTORS},
    }
    if not np.isfinite(close) or close <= 0.0:
        return base
    candidates = forecast.loc[
        forecast["forecast_end_date"].ge(timestamp)
        & forecast["available_at"].le(timestamp)
        & forecast["declare_date"].ge(timestamp - pd.Timedelta(days=freshness_days))
    ]
    if candidates.empty:
        return base
    end_dates = tuple(sorted(candidates["forecast_end_date"].unique()))
    selected: list[pd.DataFrame] = []
    for end_date in end_dates[:2]:
        if pd.Timestamp(end_date).year > pd.Timestamp(end_dates[0]).year + 1:
            break
        current = candidates.loc[candidates["forecast_end_date"].eq(end_date)]
        latest = current.groupby("institution_id", sort=False)["declare_date"].max()
        current = current.merge(
            latest.rename("latest_declare_date"),
            left_on="institution_id",
            right_index=True,
            how="inner",
        )
        current = current.loc[
            current["declare_date"].eq(current["latest_declare_date"])
        ]
        current = current.groupby("institution_id", as_index=False)[
            "forecast_eps"
        ].median()
        selected.append(current)
    eps_values = selected[0]["forecast_eps"].to_numpy(dtype="float64")
    if len(eps_values) >= minimum_institutions:
        median_eps = float(np.median(eps_values))
        mad = float(np.median(np.abs(eps_values - median_eps)))
        base["forward_earnings_yield_fy1"] = median_eps / close
        base["forecast_dispersion_fy1"] = mad
        base["log_analyst_coverage"] = float(np.log(len(eps_values)))
    if len(selected) > 1:
        fy2 = selected[1]["forecast_eps"].to_numpy(dtype="float64")
        if len(fy2) >= minimum_institutions:
            base["forward_earnings_yield_fy2"] = float(np.median(fy2)) / close
    return base


def _next_session(
    sessions: pd.DatetimeIndex, date_value: pd.Timestamp
) -> pd.Timestamp | pd.NaT:
    position = int(sessions.searchsorted(date_value, side="right"))
    return sessions[position] if position < len(sessions) else pd.NaT
