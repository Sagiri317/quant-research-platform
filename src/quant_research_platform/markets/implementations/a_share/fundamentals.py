"""A-share PIT financial facts; factor mathematics lives in ``research``."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_string_series,
    strict_timestamp_series,
    validate_financial_disclosure_chronology,
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

    if set(interim_by_data_id) != set(FINANCIAL_VALUE_COLUMNS):
        raise ValueError("financial_interim_set_invalid")
    if tuple(disclosures.columns) != DISCLOSURE_COLUMNS:
        raise ValueError("financial_disclosure_schema_invalid")
    disclosure = disclosures.loc[:, DISCLOSURE_COLUMNS].copy()
    disclosure["asset_id"] = strict_string_series(
        disclosure["asset_id"], error="financial_disclosure_asset_invalid"
    )
    for column in DISCLOSURE_COLUMNS[1:]:
        disclosure[column] = strict_timestamp_series(
            disclosure[column], error="financial_disclosure_date_invalid"
        ).dt.normalize()
    validate_financial_disclosure_chronology(
        disclosure["report_period"], disclosure["first_disclosure_date"]
    )
    if disclosure.duplicated(["asset_id", "report_period"], keep=False).any():
        raise ValueError("financial_disclosure_key_duplicate")
    sessions = pd.DatetimeIndex(trading_sessions).normalize().drop_duplicates()
    if sessions.empty or not sessions.is_monotonic_increasing:
        raise ValueError("financial_sessions_invalid")

    records: list[pd.DataFrame] = []
    for data_id, value_columns in FINANCIAL_VALUE_COLUMNS.items():
        frame = interim_by_data_id[data_id].copy()
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
        frame["asset_id"] = strict_string_series(
            frame["source_asset_id"], error="financial_asset_invalid"
        )
        frame["report_period"] = strict_timestamp_series(
            frame["report_period"], error="financial_report_period_invalid"
        ).dt.normalize()
        frame["report_type"] = strict_string_series(
            frame["report_type"], error="financial_report_type_invalid"
        )
        frame["is_corrected"] = strict_bool_series(
            frame["is_corrected"], error="financial_correction_flag_invalid"
        )
        frame["correction_disclosure_date"] = strict_timestamp_series(
            frame["correction_disclosure_date"],
            error="financial_correction_date_invalid",
            allow_null=True,
        ).dt.normalize()
        if bool(
            (frame["is_corrected"] & frame["correction_disclosure_date"].isna()).any()
        ):
            raise ValueError("financial_correction_disclosure_missing")
        frame["version_identity"] = strict_string_series(
            frame["version_identity"], error="financial_version_identity_invalid"
        )
        expected_versions = pd.Series("original", index=frame.index, dtype="string")
        corrected = frame["is_corrected"]
        expected_versions.loc[corrected] = "correction@" + frame.loc[
            corrected, "correction_disclosure_date"
        ].dt.strftime("%Y-%m-%d").astype("string")
        if bool(frame["version_identity"].ne(expected_versions).any()):
            raise ValueError("financial_version_identity_invalid")
        for column in value_columns:
            frame[column] = strict_finite_numeric_series(
                frame[column],
                error=f"financial_value_invalid: {column}",
                allow_null=True,
            )
        version_key = [
            "asset_id",
            "report_period",
            "report_type",
            "version_identity",
        ]
        if bool(frame.duplicated(version_key, keep=False).any()):
            raise ValueError("financial_version_key_duplicate")
        frame = frame.loc[frame["report_type"].eq("A")]
        frame = frame.merge(
            disclosure,
            on=("asset_id", "report_period"),
            how="left",
            validate="many_to_one",
        )
        if bool(frame["first_disclosure_date"].isna().any()):
            raise ValueError("financial_first_disclosure_missing")
        if bool(
            frame.loc[frame["is_corrected"], "correction_disclosure_date"].isna().any()
        ):
            raise ValueError("financial_correction_disclosure_missing")
        invalid_original_correction_date = frame.loc[
            ~frame["is_corrected"], "correction_disclosure_date"
        ].notna()
        if bool(invalid_original_correction_date.any()):
            raise ValueError("financial_correction_date_invalid")
        if bool(
            (
                frame.loc[frame["is_corrected"], "correction_disclosure_date"]
                <= frame.loc[frame["is_corrected"], "first_disclosure_date"]
            ).any()
        ):
            raise ValueError("financial_correction_before_first_disclosure")
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
        return pd.DataFrame(
            columns=[
                "asset_id",
                "report_period",
                "available_at",
                "is_corrected",
                "version_identity",
                "metric_id",
                "value",
            ]
        )
    result = pd.concat(records, ignore_index=True)
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
    try:
        return float(Fraction(current) + Fraction(prior_fy) - Fraction(prior_ytd))
    except OverflowError as exc:
        raise ValueError("invalid_ttm_value") from exc


def materialize_pit_fundamentals(
    observations: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """As-of select the latest legal report period and its latest available revision."""

    required_observations = {"asset_id", "timestamp"}
    required_events = {
        "asset_id",
        "metric_id",
        "report_period",
        "available_at",
        "is_corrected",
        "version_identity",
        "value",
    }
    if not required_observations.issubset(
        observations.columns
    ) or not required_events.issubset(events.columns):
        raise ValueError("pit_financial_schema_invalid")
    left = observations.loc[:, ("asset_id", "timestamp")].copy()
    left["asset_id"] = strict_string_series(left["asset_id"], error="pit_asset_invalid")
    left["timestamp"] = strict_timestamp_series(
        left["timestamp"], error="pit_timestamp_invalid"
    ).dt.normalize()
    if left.duplicated(list(required_observations)).any():
        raise ValueError("pit_observation_key_duplicate")
    right = events.loc[:, list(required_events)].copy()
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
    right["value"] = strict_finite_numeric_series(
        right["value"], error="pit_value_invalid", allow_null=True
    )
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
    if right.empty:
        result = left.iloc[:0].copy()
        for column in required_events.difference({"asset_id"}):
            result[column] = right[column]
        for prefix in ("beginning", "prior_year", "prior_fy"):
            result[f"{prefix}_report_period"] = pd.Series(dtype="datetime64[ns]")
            result[f"{prefix}_value"] = pd.Series(dtype="float64")
        return result
    frames: list[pd.DataFrame] = []
    observation_positions = left.groupby("asset_id", sort=False).indices
    for (asset_id, metric_id), stream in right.groupby(
        ["asset_id", "metric_id"], sort=False
    ):
        stream = stream.sort_values(
            ["available_at", "report_period", "version_identity"],
            kind="mergesort",
        ).reset_index(drop=True)
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
