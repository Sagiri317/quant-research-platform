"""PIT analyst forecast aggregation without provider-specific assumptions."""

from __future__ import annotations

from functools import partial
from statistics import median

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_real_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)

from .definitions import ALL_EXPECTATIONS_FACTORS, PRICE_SCALED_EXPECTATIONS

FORECAST_FRESHNESS_DAYS = 90


def _prepare_expectations(
    records,
    *,
    trading_sessions,
    freshness_days=FORECAST_FRESHNESS_DAYS,
    minimum_institutions=3,
    selected_columns=ALL_EXPECTATIONS_FACTORS,
):
    """Prepare one immutable request's source, then query observation slabs."""
    if not selected_columns or set(selected_columns).difference(
        ALL_EXPECTATIONS_FACTORS
    ):
        raise ValueError("expectations_selected_columns_invalid")
    selected_columns = tuple(selected_columns)
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
    forecast["report_id"] = records.get(
        "report_id", pd.Series(pd.NA, index=records.index)
    )
    # Locate the annual, potentially visible source rows before interpreting
    # optional values. Unlocatable records cannot supply an institutional event.
    for column in ("declare_date", "forecast_end_date"):
        forecast[column] = strict_timestamp_series(
            forecast[column],
            error="invalid_forecast_date",
            allow_null=True,
            errors="coerce",
        ).dt.normalize()
    forecast = forecast.loc[
        forecast["forecast_end_date"].dt.month.eq(12)
        & forecast["forecast_end_date"].dt.day.eq(31)
        & forecast["forecast_end_date"].ge(sessions[0])
        & forecast["declare_date"].ge(sessions[0] - pd.Timedelta(days=freshness_days))
        & forecast["declare_date"].lt(sessions[-1])
    ].copy()
    for column in ("asset_id", "institution_id", "report_id"):
        values = forecast[column]
        usable = values.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).astype(bool)
        forecast[column] = values.where(usable).astype("string").str.strip()
    forecast = forecast.loc[
        forecast["asset_id"].notna()
        & (forecast["institution_id"].notna() | forecast["report_id"].notna())
    ].copy()
    # An identified later report with unknown EPS must replace its predecessor.
    # Dropping it here would resurrect an older institutional forecast.
    forecast["forecast_eps"] = _local_finite_numeric(forecast["forecast_eps"])
    forecast["available_at"] = forecast["declare_date"].map(
        lambda value: _next_session(sessions, value)
    )
    forecast = forecast.dropna(subset=["available_at"])
    # Preserve each report until its latest visible revision is known. Sources
    # without report IDs still contribute independent rows to institution medians;
    # their positional keys are internal only and never infer report identity.
    forecast["report_key"] = (
        forecast["report_id"]
        .astype(object)
        .where(forecast["report_id"].notna(), np.arange(len(forecast)))
    )
    forecast = forecast.sort_values(
        ["asset_id", "available_at", "declare_date"], kind="mergesort"
    )

    return partial(
        _expectations_for_observations,
        {
            asset_id: (
                pd.DatetimeIndex(group["declare_date"]),
                pd.DatetimeIndex(group["available_at"]),
                group,
            )
            for asset_id, group in forecast.groupby("asset_id", sort=False)
        },
        freshness_days=freshness_days,
        minimum_institutions=minimum_institutions,
        selected_columns=selected_columns,
    )


def _expectations_for_observations(
    forecast,
    observations,
    *,
    freshness_days,
    minimum_institutions,
    selected_columns,
):
    needs_price = bool(set(selected_columns) & PRICE_SCALED_EXPECTATIONS)
    columns = ["asset_id", "timestamp", *(("close",) if needs_price else ())]
    if not set(columns).issubset(observations.columns):
        raise ValueError("expectations_input_columns_invalid")
    output = observations.loc[:, columns].copy()
    output["asset_id"] = strict_string_series(
        output["asset_id"], error="expectations_asset_id_invalid"
    )
    output["timestamp"] = strict_timestamp_series(
        output["timestamp"], error="expectations_timestamp_invalid"
    ).dt.normalize()
    output["close"] = _local_finite_numeric(output["close"]) if needs_price else np.nan
    if output.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("expectations_observation_key_duplicate")
    rows: list[dict[str, object] | None] = [None] * len(output)
    observation_times = output["timestamp"].array
    observation_closes = output["close"].array
    for asset_id, positions in output.groupby("asset_id", sort=False).indices.items():
        positions = sorted(positions, key=observation_times.__getitem__)
        source = forecast.get(asset_id)
        if source is None:
            events = iter(())
        else:
            declarations, available, asset_forecast = source
            first = declarations.searchsorted(
                observation_times[positions[0]] - pd.Timedelta(days=freshness_days),
                side="left",
            )
            last = available.searchsorted(
                observation_times[positions[-1]], side="right"
            )
            events = iter(asset_forecast.iloc[first:last].itertuples(index=False))
        event = next(events, None)
        active = {}
        for position in positions:
            timestamp = observation_times[position]
            while event is not None and event.available_at <= timestamp:
                key = (event.forecast_end_date, event.report_key)
                active[key] = event
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
                selected_columns=selected_columns,
            )
    return pd.DataFrame.from_records(
        rows,
        columns=("asset_id", "timestamp", *selected_columns),
    )


def _features_for_observation(
    asset_id: str,
    timestamp: pd.Timestamp,
    close: float,
    forecast,
    *,
    minimum_institutions: int,
    selected_columns,
) -> dict[str, object]:
    base: dict[str, object] = {
        "asset_id": asset_id,
        "timestamp": timestamp,
        **{factor: float("nan") for factor in selected_columns},
    }
    valid_price = pd.notna(close) and close > 0.0
    # Report attribution can change independently of other reports from its old
    # institution. Reduce only after applying report revisions and freshness.
    latest = {}
    for event in forecast:
        institution = None if pd.isna(event.institution_id) else event.institution_id
        key = (event.forecast_end_date, institution)
        reports = latest.get(key)
        if reports is None or event.declare_date > reports[0].declare_date:
            latest[key] = [event]
        elif event.declare_date == reports[0].declare_date:
            reports.append(event)
    by_period = {}
    for reports in latest.values():
        event = reports[0]._replace(
            forecast_eps=_finite_median([report.forecast_eps for report in reports])
        )
        by_period.setdefault(event.forecast_end_date, []).append(event)
    if not by_period:
        return base
    end_dates = sorted(by_period)
    selected = [by_period[end_dates[0]]]
    if len(end_dates) > 1 and end_dates[1].year == end_dates[0].year + 1:
        selected.append(by_period[end_dates[1]])
    fy1_events = (
        []
        if any(pd.isna(event.institution_id) for event in selected[0])
        else [event for event in selected[0] if pd.notna(event.forecast_eps)]
    )
    eps_values = [event.forecast_eps for event in fy1_events]
    if len(eps_values) >= minimum_institutions:
        if any(
            column in selected_columns
            for column in (
                "forward_earnings_yield_fy1",
                "forecast_dispersion_fy1",
                "forecast_dispersion_price_scaled_fy1",
            )
        ):
            center = median(eps_values)
            if "forward_earnings_yield_fy1" in selected_columns and valid_price:
                base["forward_earnings_yield_fy1"] = _finite_float(
                    center / float(close)
                )
            if any(
                column in selected_columns
                for column in (
                    "forecast_dispersion_fy1",
                    "forecast_dispersion_price_scaled_fy1",
                )
            ):
                mad = median(abs(value - center) for value in eps_values)
                if "forecast_dispersion_fy1" in selected_columns:
                    base["forecast_dispersion_fy1"] = _finite_float(mad)
                if (
                    "forecast_dispersion_price_scaled_fy1" in selected_columns
                    and valid_price
                ):
                    base["forecast_dispersion_price_scaled_fy1"] = _finite_float(
                        mad / float(close)
                    )
        if "log_analyst_coverage" in selected_columns:
            base["log_analyst_coverage"] = float(np.log(len(eps_values)))
        if "median_forecast_age_days_fy1" in selected_columns:
            base["median_forecast_age_days_fy1"] = float(
                median((timestamp - event.declare_date).days for event in fy1_events)
            )
        if "newly_updated_institution_fraction_fy1" in selected_columns:
            base["newly_updated_institution_fraction_fy1"] = sum(
                event.available_at == timestamp for event in fy1_events
            ) / len(fy1_events)
    if (
        "forward_earnings_yield_fy2" in selected_columns
        and len(selected) > 1
        and valid_price
    ):
        fy2 = (
            []
            if any(pd.isna(event.institution_id) for event in selected[1])
            else [
                event.forecast_eps
                for event in selected[1]
                if pd.notna(event.forecast_eps)
            ]
        )
        if len(fy2) >= minimum_institutions:
            base["forward_earnings_yield_fy2"] = _finite_float(
                _finite_median(fy2) / float(close)
            )
    return base


def _finite_float(value):
    """Keep an unrepresentable derived value local to its own factor."""
    value = float(value)
    return value if np.isfinite(value) else np.nan


def _local_finite_numeric(values):
    numeric = strict_real_numeric_series(
        values, error="expectations_numeric_invalid", errors="coerce"
    )
    return numeric.where(np.isfinite(numeric))


def _finite_median(values) -> float:
    # Unknown reports on the same institutional declaration day make its median
    # unavailable; later known declarations can resolve that local barrier.
    if any(pd.isna(value) for value in values):
        return np.nan
    return float(np.median(values))


def _next_session(
    sessions: pd.DatetimeIndex, date_value: pd.Timestamp
) -> pd.Timestamp | pd.NaT:
    position = int(sessions.searchsorted(date_value, side="right"))
    return sessions[position] if position < len(sessions) else pd.NaT
