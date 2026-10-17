"""A-share PIT financial facts; factor mathematics lives in ``research``."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from functools import partial

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    finite_exact_float,
    strict_bool_series,
    strict_finite_numeric_series,
    strict_real_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)

FINANCIAL_VALUE_COLUMNS = {
    "financial_balance": (
        "total_assets",
        "total_liabilities",
        "parent_equity",
        "cash",
        "short_term_borrowings",
        "current_portion_noncurrent_liabilities",
        "long_term_borrowings",
        "bonds_payable",
    ),
    "financial_income": (
        "operating_revenue",
        "operating_cost",
        "consolidated_net_income",
        "parent_net_income",
    ),
    "financial_cashflow": ("operating_cash_flow", "capex_cash_paid"),
}
FLOW_METRIC_IDS = frozenset(
    {
        "operating_revenue",
        "operating_cost",
        "consolidated_net_income",
        "parent_net_income",
        "operating_cash_flow",
        "capex_cash_paid",
    }
)
DISCLOSURE_COLUMNS = ("asset_id", "report_period", "first_disclosure_date")


def financial_events_from_interim(
    interim_by_data_id: Mapping[str, pd.DataFrame],
    disclosures: pd.DataFrame,
    *,
    trading_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build PIT events from report facts and the separate Choice disclosure owner."""

    if not interim_by_data_id or not set(interim_by_data_id).issubset(
        FINANCIAL_VALUE_COLUMNS
    ):
        raise ValueError("financial_interim_set_invalid")
    if tuple(disclosures.columns) != DISCLOSURE_COLUMNS:
        raise ValueError("financial_disclosure_schema_invalid")
    disclosure = disclosures.loc[:, DISCLOSURE_COLUMNS].copy()
    counts = {}
    disclosure["asset_id"] = (
        disclosure["asset_id"].astype("string").str.strip().replace("", pd.NA)
    )
    for column in DISCLOSURE_COLUMNS[1:]:
        disclosure[column] = strict_timestamp_series(
            disclosure[column],
            error="financial_disclosure_date_invalid",
            allow_null=True,
            errors="coerce",
        ).dt.normalize()
    valid = disclosure.notna().all(axis=1) & disclosure["first_disclosure_date"].ge(
        disclosure["report_period"]
    )
    counts["excluded_invalid_disclosure_rows"] = int((~valid).sum())
    disclosure = disclosure.loc[valid].drop_duplicates()
    conflicts = disclosure.duplicated(["asset_id", "report_period"], keep=False)
    counts["excluded_conflicting_disclosure_rows"] = int(conflicts.sum())
    disclosure = disclosure.loc[~conflicts]
    sessions = pd.DatetimeIndex(trading_sessions).normalize().drop_duplicates()
    if sessions.empty or not sessions.is_monotonic_increasing:
        raise ValueError("financial_sessions_invalid")

    records: list[pd.DataFrame] = []
    for data_id, original in interim_by_data_id.items():
        value_columns = FINANCIAL_VALUE_COLUMNS[data_id]
        frame = original.copy()
        required = {
            "source_asset_id",
            "report_period",
            "report_type",
            "is_corrected",
            "correction_disclosure_date",
            "version_identity",
            *value_columns,
        }
        if not required.issubset(frame.columns):
            raise ValueError("financial_interim_schema_invalid")
        # Only consolidated reports are consumed by these metrics.
        frame = frame.loc[frame["report_type"].eq("A")].copy()
        frame["asset_id"] = (
            frame["source_asset_id"].astype("string").str.strip().replace("", pd.NA)
        )
        frame["report_period"] = strict_timestamp_series(
            frame["report_period"],
            error="financial_report_period_invalid",
            allow_null=True,
            errors="coerce",
        ).dt.normalize()
        frame["correction_disclosure_date"] = strict_timestamp_series(
            frame["correction_disclosure_date"],
            error="financial_correction_date_invalid",
            allow_null=True,
            errors="coerce",
        ).dt.normalize()
        flags = (
            frame["is_corrected"]
            .map(lambda value: value if isinstance(value, (bool, np.bool_)) else pd.NA)
            .astype("boolean")
        )
        valid = (
            frame["asset_id"].notna() & frame["report_period"].notna() & flags.notna()
        )
        valid &= flags.eq(frame["correction_disclosure_date"].notna()).fillna(False)
        expected_versions = pd.Series("original", index=frame.index, dtype="string")
        corrected = flags.fillna(False)
        expected_versions.loc[corrected] = "correction@" + frame.loc[
            corrected, "correction_disclosure_date"
        ].dt.strftime("%Y-%m-%d").astype("string")
        valid &= frame["version_identity"].eq(expected_versions).fillna(False)
        counts["excluded_invalid_version_rows"] = counts.get(
            "excluded_invalid_version_rows", 0
        ) + int((~valid).sum())
        frame = frame.loc[valid].copy()
        frame["is_corrected"] = flags.loc[valid].astype(bool)
        for column in value_columns:
            values = strict_real_numeric_series(
                frame[column], error="invalid_financial_value", errors="coerce"
            )
            frame[column] = values.where(np.isfinite(values))
        version_key = [
            "asset_id",
            "report_period",
            "report_type",
            "version_identity",
        ]
        frame = frame.drop_duplicates()
        duplicate = frame.duplicated(version_key, keep=False)
        if duplicate.any():
            for column in value_columns:
                conflict = (
                    frame.loc[duplicate]
                    .groupby(version_key)[column]
                    .transform(lambda values: values.nunique(dropna=False))
                    .gt(1)
                )
                frame.loc[conflict.index[conflict], column] = np.nan
            frame = frame.drop_duplicates(version_key)
        frame = frame.merge(
            disclosure,
            on=("asset_id", "report_period"),
            how="left",
            validate="many_to_one",
        )
        valid = frame["first_disclosure_date"].notna() & (
            ~frame["is_corrected"]
            | frame["correction_disclosure_date"].gt(frame["first_disclosure_date"])
        )
        counts["excluded_unlocated_disclosure_rows"] = counts.get(
            "excluded_unlocated_disclosure_rows", 0
        ) + int((~valid).sum())
        frame = frame.loc[valid].copy()
        frame["disclosure_date"] = frame["first_disclosure_date"].where(
            ~frame["is_corrected"], frame["correction_disclosure_date"]
        )
        frame = frame.loc[frame["disclosure_date"].lt(sessions[-1])].copy()
        if frame.empty:
            continue
        positions = sessions.searchsorted(
            frame["disclosure_date"].to_numpy(dtype="datetime64[ns]"),
            side="right",
        )
        frame["available_at"] = sessions.take(positions)
        # Different public dates can share the same next trading session.
        frame = frame.sort_values("disclosure_date", kind="mergesort").drop_duplicates(
            ["asset_id", "report_period", "available_at"], keep="last"
        )
        melted = frame.melt(
            id_vars=(
                "asset_id",
                "report_period",
                "available_at",
                "is_corrected",
                "version_identity",
            ),
            value_vars=value_columns,
            var_name="metric_id",
            value_name="value",
        )
        records.append(melted)
    if not records:
        result = pd.DataFrame(
            {
                "asset_id": pd.Series(dtype="string"),
                "report_period": pd.Series(dtype="datetime64[ns]"),
                "available_at": pd.Series(dtype="datetime64[ns]"),
                "is_corrected": pd.Series(dtype=bool),
                "version_identity": pd.Series(dtype="string"),
                "metric_id": pd.Series(dtype="string"),
                "value": pd.Series(dtype="float64"),
            }
        )
        result.attrs["source_row_counts"] = counts
        return result
    result = pd.concat(records, ignore_index=True)
    result.attrs["source_row_counts"] = counts
    key = [
        "asset_id",
        "metric_id",
        "report_period",
        "available_at",
        "version_identity",
    ]
    if result.duplicated(key, keep=False).any():
        raise ValueError("financial_event_key_duplicate")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def construct_ttm_events(
    events: pd.DataFrame,
    *,
    value_column: str = "value",
) -> pd.DataFrame:
    """Compute PIT TTM facts and append correction-triggered derived events."""

    required = {
        "asset_id",
        "metric_id",
        "report_period",
        "available_at",
        "is_corrected",
        "version_identity",
        value_column,
    }
    if not required.issubset(events.columns):
        raise ValueError("ttm_event_columns_invalid")
    result = events.reset_index(drop=True).copy()
    for column in ("report_period", "available_at"):
        result[column] = strict_timestamp_series(
            result[column], error="ttm_event_date_invalid"
        ).dt.normalize()
    result["is_corrected"] = strict_bool_series(
        result["is_corrected"], error="ttm_correction_flag_invalid"
    )
    result["version_identity"] = strict_string_series(
        result["version_identity"], error="ttm_version_identity_invalid"
    )
    result[value_column] = strict_finite_numeric_series(
        result[value_column], error="invalid_ttm_value", allow_null=True
    )
    event_key = [
        "asset_id",
        "metric_id",
        "report_period",
        "available_at",
        "version_identity",
    ]
    if result.duplicated(event_key, keep=False).any():
        raise ValueError("ttm_event_key_duplicate")
    revision_key = [
        "asset_id",
        "metric_id",
        "report_period",
        "available_at",
    ]
    if result.duplicated(revision_key, keep=False).any():
        raise ValueError("ttm_revision_order_ambiguous")
    result["ttm_value"] = np.nan
    derived_records: list[dict[str, object]] = []
    for (_asset_id, metric_id), positions in result.groupby(
        ["asset_id", "metric_id"], sort=False
    ).groups.items():
        if metric_id not in FLOW_METRIC_IDS:
            continue
        group = result.loc[positions].sort_values(
            ["available_at", "report_period", "version_identity"],
            kind="mergesort",
        )
        known = {}
        dependents = {}
        derived = []
        for cutoff, available in group.groupby("available_at", sort=False):
            changed = set(available["report_period"])
            # All same-session disclosures are known before evaluating that session.
            for _position, row in available.iterrows():
                period = row["report_period"]
                if period not in known and period.month != 12:
                    for dependency in (
                        _prior_fy_report_period(period),
                        _prior_year_report_period(period),
                    ):
                        if dependency is not None:
                            dependents.setdefault(dependency, set()).add(period)
                known[period] = (
                    None
                    if pd.isna(row[value_column])
                    or (metric_id == "capex_cash_paid" and row[value_column] < 0)
                    else float(row[value_column])
                )
            for position, row in available.iterrows():
                result.loc[position, "ttm_value"] = _ttm_value(
                    known, row["report_period"]
                )
            affected = set().union(
                *(dependents.get(period, set()) for period in changed)
            )
            for period in affected - changed:
                derived.append(
                    {
                        "asset_id": _asset_id,
                        "metric_id": metric_id,
                        "report_period": period,
                        "available_at": cutoff,
                        "version_identity": f"derived@{cutoff:%Y-%m-%d}",
                        "is_corrected": True,
                        value_column: known[period],
                        "ttm_value": _ttm_value(known, period),
                    }
                )
        derived_records.extend(
            sorted(derived, key=lambda row: (row["report_period"], row["available_at"]))
        )
    if derived_records:
        result = pd.concat(
            (result, pd.DataFrame.from_records(derived_records)),
            ignore_index=True,
        )
    if result.duplicated(event_key, keep=False).any():
        raise ValueError("ttm_event_key_duplicate")
    return result


def _ttm_value(known, report_period: pd.Timestamp) -> float | None:
    current = known.get(report_period)
    if current is None:
        return None
    if report_period.month == 12:
        return current
    prior_fy = known.get(_prior_fy_report_period(report_period))
    prior_ytd = known.get(_prior_year_report_period(report_period))
    if prior_fy is None or prior_ytd is None:
        return None
    return finite_exact_float(
        Fraction(current) + Fraction(prior_fy) - Fraction(prior_ytd)
    )


_PIT_EVENT_COLUMNS = {
    "asset_id",
    "metric_id",
    "report_period",
    "available_at",
    "is_corrected",
    "version_identity",
    "value",
}


def _pit_observations(observations):
    if not {"asset_id", "timestamp"}.issubset(observations.columns):
        raise ValueError("pit_financial_schema_invalid")
    left = observations.loc[:, ("asset_id", "timestamp")].copy()
    left["asset_id"] = strict_string_series(left["asset_id"], error="pit_asset_invalid")
    left["timestamp"] = strict_timestamp_series(
        left["timestamp"], error="pit_timestamp_invalid"
    ).dt.normalize()
    if left.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("pit_observation_key_duplicate")
    return left


def _prepare_pit_fundamentals(events):
    """Validate all source events and order each stream once per request."""
    if not _PIT_EVENT_COLUMNS.issubset(events.columns):
        raise ValueError("pit_financial_schema_invalid")
    right = events.loc[:, list(_PIT_EVENT_COLUMNS)].copy()
    right["asset_id"] = strict_string_series(
        right["asset_id"], error="pit_asset_invalid"
    )
    right["metric_id"] = strict_string_series(
        right["metric_id"], error="pit_metric_invalid"
    )
    for column in ("report_period", "available_at"):
        right[column] = strict_timestamp_series(
            right[column], error="pit_event_date_invalid"
        ).dt.normalize()
    values = strict_real_numeric_series(
        right["value"], error="invalid_pit_value", errors="coerce"
    )
    right["value"] = values.where(np.isfinite(values))
    right["version_identity"] = strict_string_series(
        right["version_identity"], error="pit_version_identity_invalid"
    )
    if right.duplicated(
        [
            "asset_id",
            "metric_id",
            "report_period",
            "available_at",
            "version_identity",
        ],
        keep=False,
    ).any():
        raise ValueError("pit_event_key_duplicate")
    if right.duplicated(
        ["asset_id", "metric_id", "report_period", "available_at"], keep=False
    ).any():
        raise ValueError("pit_revision_order_ambiguous")
    streams = tuple(
        (
            key,
            stream.sort_values(
                ["available_at", "report_period", "version_identity"], kind="mergesort"
            ).reset_index(drop=True),
        )
        for key, stream in right.groupby(["asset_id", "metric_id"], sort=False)
    )
    return partial(_query_pit_fundamentals, streams=streams, right=right.iloc[:0])


def _query_pit_fundamentals(left, *, streams, right):
    if not streams:
        result = left.iloc[:0].copy()
        for column in _PIT_EVENT_COLUMNS.difference({"asset_id"}):
            result[column] = right[column]
        for prefix in ("beginning", "prior_year", "prior_fy"):
            result[f"{prefix}_report_period"] = pd.Series(dtype="datetime64[ns]")
            result[f"{prefix}_value"] = pd.Series(dtype="float64")
        return result
    frames: list[pd.DataFrame] = []
    observation_positions = left.groupby("asset_id", sort=False).indices
    for (asset_id, metric_id), stream in streams:
        asset_observations = left.iloc[
            observation_positions.get(asset_id, np.empty(0, dtype=np.intp))
        ].sort_values("timestamp", kind="mergesort")
        timestamps = asset_observations["timestamp"].to_numpy()
        availability = stream["available_at"].to_numpy()
        periods = stream["report_period"].to_numpy()
        # A late revision of an older report must not roll back the current period.
        current_at_event = np.maximum.accumulate(
            np.where(
                periods == np.maximum.accumulate(periods),
                np.arange(len(stream)),
                -1,
            )
        )
        last_legal = np.searchsorted(availability, timestamps, side="right") - 1
        current_positions = np.where(
            last_legal >= 0,
            current_at_event[np.maximum(last_legal, 0)],
            -1,
        )
        current = stream.reindex(current_positions).reset_index(drop=True)
        current_values = current["value"].to_numpy(dtype=object, copy=True)
        current_values[current_positions < 0] = np.nan
        corrected = current["is_corrected"].to_numpy(dtype=object, copy=True)
        corrected[current_positions < 0] = pd.NA
        frame = pd.DataFrame(
            {
                "asset_id": asset_id,
                "timestamp": timestamps,
                "metric_id": metric_id,
                "report_period": current["report_period"].to_numpy(),
                "available_at": current["available_at"].to_numpy(),
                "version_identity": current["version_identity"].to_numpy(),
                "is_corrected": corrected,
                "value": current_values,
            }
        )
        period_positions = stream.groupby("report_period", sort=False).indices
        current_periods = current["report_period"].to_numpy()
        legal_periods = current["report_period"].dropna().unique()
        for prefix, preceding_period in (
            ("prior_year", _prior_year_report_period),
            ("prior_fy", _prior_fy_report_period),
        ):
            dependency_positions = np.full(len(timestamps), -1, dtype=np.intp)
            for period in legal_periods:
                positions = period_positions.get(preceding_period(pd.Timestamp(period)))
                if positions is None:
                    continue
                selected = current_periods == period
                legal = (
                    np.searchsorted(
                        availability[positions], timestamps[selected], side="right"
                    )
                    - 1
                )
                dependency_positions[selected] = np.where(
                    legal >= 0, positions[np.maximum(legal, 0)], -1
                )
            dependency = stream.reindex(dependency_positions)
            values = dependency["value"].to_numpy(dtype=object, copy=True)
            values[dependency_positions < 0] = np.nan
            frame[f"{prefix}_report_period"] = dependency["report_period"].to_numpy()
            frame[f"{prefix}_value"] = values
        frame.insert(8, "beginning_report_period", frame["prior_year_report_period"])
        frame.insert(9, "beginning_value", frame["prior_year_value"])
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .infer_objects()
        .sort_values(["asset_id", "timestamp", "metric_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _prior_year_report_period(report_period: pd.Timestamp) -> pd.Timestamp | None:
    try:
        return report_period.replace(year=report_period.year - 1)
    except ValueError:
        return None


def _prior_fy_report_period(report_period: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=report_period.year - 1, month=12, day=31)
