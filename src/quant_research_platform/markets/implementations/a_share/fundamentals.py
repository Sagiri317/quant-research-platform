"""A-share PIT financial facts; factor mathematics lives in ``research``."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
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
FINANCIAL_TABLE_CODE_BY_DATA_ID = {
    "financial_balance": "FS_Combas",
    "financial_income": "FS_Comins",
    "financial_cashflow": "FS_Comscfd",
}
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
    if disclosure.duplicated(list(DISCLOSURE_COLUMNS)).any():
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
        for column in value_columns:
            frame[column] = strict_finite_numeric_series(
                frame[column],
                error=f"financial_value_invalid: {column}",
                allow_null=True,
            )
        frame = frame.loc[frame["report_type"].eq("A")]
        frame = frame.merge(
            disclosure,
            on=("asset_id", "report_period"),
            how="left",
            validate="many_to_one",
        )
        if bool(
            frame.loc[~frame["is_corrected"], "first_disclosure_date"].isna().any()
        ):
            raise ValueError("financial_first_disclosure_missing")
        if bool(
            frame.loc[frame["is_corrected"], "correction_disclosure_date"].isna().any()
        ):
            raise ValueError("financial_correction_disclosure_missing")
        frame["disclosure_date"] = frame["first_disclosure_date"].where(
            ~frame["is_corrected"], frame["correction_disclosure_date"]
        )
        positions = sessions.searchsorted(
            frame["disclosure_date"].to_numpy(dtype="datetime64[ns]"),
            side="right",
        )
        if bool((positions >= len(sessions)).any()):
            raise ValueError("financial_availability_outside_calendar")
        frame["available_at"] = sessions.take(positions)
        melted = frame.melt(
            id_vars=("asset_id", "report_period", "available_at", "is_corrected"),
            value_vars=value_columns,
            var_name="metric_id",
            value_name="value",
        )
        records.append(melted)
    result = pd.concat(records, ignore_index=True)
    key = ["asset_id", "metric_id", "report_period", "available_at"]
    if result.duplicated(key, keep=False).any():
        raise ValueError("financial_event_key_duplicate")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


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
    if right.duplicated(
        ["asset_id", "metric_id", "report_period", "available_at"], keep=False
    ).any():
        raise ValueError("pit_event_key_duplicate")
    records: list[dict[str, object]] = []
    for (asset_id, metric_id), stream in right.groupby(
        ["asset_id", "metric_id"], sort=False
    ):
        stream = stream.sort_values(["available_at", "report_period"], kind="mergesort")
        asset_observations = left.loc[left["asset_id"].eq(asset_id)].sort_values(
            "timestamp", kind="mergesort"
        )
        for observation in asset_observations.itertuples(index=False):
            legal = stream.loc[stream["available_at"].le(observation.timestamp)]
            if legal.empty:
                current = previous = None
            else:
                latest_period = legal["report_period"].max()
                current_rows = legal.loc[legal["report_period"].eq(latest_period)]
                current = current_rows.iloc[-1]
                prior_periods = legal.loc[
                    legal["report_period"].lt(latest_period), "report_period"
                ]
                if prior_periods.empty:
                    previous = None
                else:
                    prior_period = prior_periods.max()
                    previous = legal.loc[legal["report_period"].eq(prior_period)].iloc[
                        -1
                    ]
            records.append(
                {
                    "asset_id": asset_id,
                    "timestamp": observation.timestamp,
                    "metric_id": metric_id,
                    "report_period": pd.NaT
                    if current is None
                    else current["report_period"],
                    "available_at": pd.NaT
                    if current is None
                    else current["available_at"],
                    "is_corrected": pd.NA
                    if current is None
                    else current["is_corrected"],
                    "value": np.nan if current is None else current["value"],
                    "previous_report_period": pd.NaT
                    if previous is None
                    else previous["report_period"],
                    "previous_value": np.nan if previous is None else previous["value"],
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["asset_id", "timestamp", "metric_id"], kind="mergesort")
        .reset_index(drop=True)
    )
