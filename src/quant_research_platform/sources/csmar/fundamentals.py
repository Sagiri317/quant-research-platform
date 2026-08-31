"""PIT financial events derived from the strict CSMAR interim contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)

FINANCIAL_TABLE_CODE_BY_DATA_ID = {
    "financial_balance": "FS_Combas",
    "financial_income": "FS_Comins",
    "financial_cashflow": "FS_Comscfd",
}
FINANCIAL_VALUE_COLUMNS = {
    "financial_balance": ("total_assets", "total_liabilities", "total_equity"),
    "financial_income": ("operating_revenue", "net_income"),
    "financial_cashflow": ("operating_cash_flow",),
}
FINANCIAL_METRICS_BY_TABLE_ID = {
    FINANCIAL_TABLE_CODE_BY_DATA_ID[data_id]: metrics
    for data_id, metrics in FINANCIAL_VALUE_COLUMNS.items()
}
DISCLOSURE_COLUMNS = (
    "asset_id",
    "table_id",
    "report_period",
    "first_disclosure_date",
)


def financial_events_from_interim(
    interim_by_data_id: Mapping[str, pd.DataFrame],
    disclosures: pd.DataFrame,
    *,
    trading_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create long PIT events from exact interim tables plus first disclosures."""

    if set(interim_by_data_id) != set(FINANCIAL_TABLE_CODE_BY_DATA_ID):
        raise ValueError("csmar_financial_interim_set_invalid")
    if not isinstance(disclosures, pd.DataFrame) or set(disclosures.columns) != set(
        DISCLOSURE_COLUMNS
    ):
        raise ValueError("csmar_financial_disclosure_schema_invalid")
    disclosure = disclosures.loc[:, DISCLOSURE_COLUMNS].copy()
    disclosure["asset_id"] = strict_string_series(
        disclosure["asset_id"], error="csmar_financial_disclosure_asset_invalid"
    )
    disclosure["table_id"] = strict_string_series(
        disclosure["table_id"], error="csmar_financial_disclosure_table_invalid"
    )
    if not set(disclosure["table_id"]).issubset(
        set(FINANCIAL_TABLE_CODE_BY_DATA_ID.values())
    ):
        raise ValueError("csmar_financial_disclosure_table_invalid")
    for column in ("report_period", "first_disclosure_date"):
        disclosure[column] = strict_timestamp_series(
            disclosure[column], error=f"csmar_financial_disclosure_{column}_invalid"
        ).dt.normalize()
    if bool(
        disclosure.duplicated(
            ["asset_id", "table_id", "report_period"], keep=False
        ).any()
    ):
        raise ValueError("csmar_financial_disclosure_key_duplicate")
    sessions = pd.DatetimeIndex(trading_sessions).astype("datetime64[ns]")
    if (
        sessions.empty
        or sessions.has_duplicates
        or not sessions.is_monotonic_increasing
    ):
        raise ValueError("csmar_financial_sessions_invalid")
    frames: list[pd.DataFrame] = []
    for data_id, table_code in FINANCIAL_TABLE_CODE_BY_DATA_ID.items():
        interim = interim_by_data_id[data_id].copy()
        required = {
            "source_asset_id",
            "report_period",
            "report_type",
            "is_corrected",
            "correction_disclosure_date",
            *FINANCIAL_VALUE_COLUMNS[data_id],
        }
        if not required.issubset(interim.columns):
            raise ValueError("csmar_financial_interim_schema_invalid")
        interim = interim.loc[interim["report_type"].eq("A")].copy()
        interim["table_id"] = table_code
        interim.rename(columns={"source_asset_id": "asset_id"}, inplace=True)
        interim = interim.merge(
            disclosure,
            on=("asset_id", "table_id", "report_period"),
            how="left",
            validate="one_to_one",
        )
        ordinary = ~interim["is_corrected"]
        if bool(interim.loc[ordinary, "first_disclosure_date"].isna().any()):
            raise ValueError("csmar_financial_first_disclosure_missing")
        if bool(interim.loc[~ordinary, "correction_disclosure_date"].isna().any()):
            raise ValueError("csmar_financial_correction_disclosure_missing")
        interim["disclosure_date"] = interim["first_disclosure_date"].where(
            ordinary, interim["correction_disclosure_date"]
        )
        positions = sessions.searchsorted(
            interim["disclosure_date"].to_numpy(dtype="datetime64[ns]"), side="right"
        )
        if bool((positions >= len(sessions)).any()):
            raise ValueError("csmar_financial_availability_outside_calendar")
        interim["available_at"] = pd.Series(
            sessions.take(positions), index=interim.index, dtype="datetime64[ns]"
        )
        frames.append(
            interim.melt(
                id_vars=(
                    "asset_id",
                    "table_id",
                    "report_period",
                    "available_at",
                    "is_corrected",
                ),
                value_vars=FINANCIAL_VALUE_COLUMNS[data_id],
                var_name="metric_id",
                value_name="value",
            )
        )
    result = pd.concat(frames, ignore_index=True)
    key = [
        "asset_id",
        "table_id",
        "metric_id",
        "report_period",
        "available_at",
    ]
    if bool(result.duplicated(key, keep=False).any()):
        raise ValueError("csmar_financial_event_key_duplicate")
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def materialize_pit_fundamentals(
    observations: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Select the latest available revision of the latest available report period."""

    if not {"asset_id", "timestamp"}.issubset(observations.columns):
        raise ValueError("csmar_pit_columns_invalid")
    required_events = {
        "asset_id",
        "table_id",
        "metric_id",
        "report_period",
        "available_at",
        "is_corrected",
        "value",
    }
    if not required_events.issubset(events.columns):
        raise ValueError("csmar_pit_columns_invalid")
    left = observations.loc[:, ("asset_id", "timestamp")].copy()
    left["asset_id"] = strict_string_series(
        left["asset_id"], error="csmar_pit_asset_invalid"
    )
    left["timestamp"] = strict_timestamp_series(
        left["timestamp"], error="csmar_pit_timestamp_invalid"
    )
    if bool(left.duplicated(["asset_id", "timestamp"], keep=False).any()):
        raise ValueError("csmar_pit_observation_key_duplicate")
    event_data = events.loc[
        :,
        (
            "asset_id",
            "table_id",
            "metric_id",
            "report_period",
            "available_at",
            "is_corrected",
            "value",
        ),
    ].copy()
    for column in ("asset_id", "table_id", "metric_id"):
        event_data[column] = strict_string_series(
            event_data[column], error=f"csmar_pit_{column}_invalid"
        )
    for column in ("report_period", "available_at"):
        event_data[column] = strict_timestamp_series(
            event_data[column], error=f"csmar_pit_{column}_invalid"
        )
    event_data["value"] = strict_finite_numeric_series(
        event_data["value"], error="csmar_pit_value_invalid", allow_null=True
    )
    event_key = [
        "asset_id",
        "table_id",
        "metric_id",
        "report_period",
        "available_at",
    ]
    if bool(event_data.duplicated(event_key, keep=False).any()):
        raise ValueError("csmar_financial_event_key_duplicate")
    expected_pairs = {
        (table_id, metric_id)
        for table_id, metric_ids in FINANCIAL_METRICS_BY_TABLE_ID.items()
        for metric_id in metric_ids
    }
    if not set(
        zip(event_data["table_id"], event_data["metric_id"], strict=True)
    ).issubset(expected_pairs):
        raise ValueError("csmar_pit_metric_invalid")
    observations_by_asset = {
        asset_id: group.sort_values("timestamp", kind="mergesort")
        for asset_id, group in left.groupby("asset_id", sort=False)
    }
    event_group_indices = event_data.groupby(
        ["asset_id", "table_id", "metric_id"], sort=False
    ).groups
    records: list[dict[str, object]] = []
    for asset_id, observed in observations_by_asset.items():
        for table_id, metric_ids in FINANCIAL_METRICS_BY_TABLE_ID.items():
            for metric_id in metric_ids:
                key = (asset_id, table_id, metric_id)
                indices = event_group_indices.get(key)
                ordered_events = (
                    event_data.loc[
                        indices,
                        ("report_period", "available_at", "is_corrected", "value"),
                    ].sort_values(["available_at", "report_period"], kind="mergesort")
                    if indices is not None
                    else pd.DataFrame(
                        columns=(
                            "report_period",
                            "available_at",
                            "is_corrected",
                            "value",
                        )
                    )
                )
                event_stream = iter(ordered_events.itertuples(index=False, name=None))
                next_event = next(event_stream, None)
                current = None
                previous = None
                for observation in observed.itertuples(index=False):
                    while (
                        next_event is not None
                        and next_event[1] <= observation.timestamp
                    ):
                        report_period = next_event[0]
                        if current is None or report_period > current[0]:
                            previous = current
                            current = next_event
                        elif report_period == current[0]:
                            current = next_event
                        elif previous is None or report_period > previous[0]:
                            previous = next_event
                        elif report_period == previous[0]:
                            previous = next_event
                        next_event = next(event_stream, None)
                    records.append(
                        {
                            "asset_id": asset_id,
                            "timestamp": observation.timestamp,
                            "table_id": table_id,
                            "metric_id": metric_id,
                            "report_period": pd.NaT if current is None else current[0],
                            "available_at": pd.NaT if current is None else current[1],
                            "is_corrected": pd.NA if current is None else current[2],
                            "value": float("nan") if current is None else current[3],
                            "previous_report_period": (
                                pd.NaT if previous is None else previous[0]
                            ),
                            "previous_value": (
                                float("nan") if previous is None else previous[3]
                            ),
                        }
                    )
    output_columns = (
        "asset_id",
        "timestamp",
        "table_id",
        "metric_id",
        "report_period",
        "available_at",
        "is_corrected",
        "value",
        "previous_report_period",
        "previous_value",
    )
    return pd.DataFrame.from_records(records, columns=output_columns).sort_values(
        ["asset_id", "timestamp", "table_id", "metric_id"],
        kind="mergesort",
        ignore_index=True,
    )
