"""PIT financial events derived from the strict CSMAR interim contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.validation import (
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
DISCLOSURE_COLUMNS = (
    "asset_id",
    "table_id",
    "report_period",
    "first_disclosure_date",
)


def financial_disclosure_interim_spec() -> InterimDataSpec:
    """Canonical contract for the user-provided first-disclosure source."""

    columns = (
        "source_asset_id",
        "timestamp",
        "table_id",
        "report_period",
        "first_disclosure_date",
    )
    return InterimDataSpec(
        data_id="financial_disclosures",
        source_id=SourceId("csmar"),
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
        column_roles={
            column: (
                ColumnRole.KEY
                if column in {"source_asset_id", "table_id", "report_period"}
                else ColumnRole.TIMESTAMP
            )
            for column in columns
        },
        primary_key=(
            "source_asset_id",
            "table_id",
            "report_period",
            "timestamp",
        ),
        partition_columns=("year",),
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
    left["_observation_id"] = range(len(left))
    candidates = left.merge(events, on="asset_id", how="left", validate="many_to_many")
    candidates = candidates.loc[
        candidates["available_at"].le(candidates["timestamp"])
    ].copy()
    candidates.sort_values(
        [
            "_observation_id",
            "table_id",
            "metric_id",
            "report_period",
            "available_at",
        ],
        kind="mergesort",
        inplace=True,
    )
    revision_keys = [
        "_observation_id",
        "table_id",
        "metric_id",
        "report_period",
    ]
    revisions = candidates.groupby(revision_keys, sort=False, as_index=False).tail(1)
    selection_keys = ["_observation_id", "table_id", "metric_id"]
    selected = revisions.groupby(selection_keys, sort=False, as_index=False).tail(1)
    last_two = revisions.groupby(selection_keys, sort=False, group_keys=False).tail(2)
    previous = last_two.loc[
        last_two.duplicated(selection_keys, keep="last"),
        (*selection_keys, "report_period", "value"),
    ].rename(
        columns={
            "report_period": "previous_report_period",
            "value": "previous_value",
        }
    )
    selected = selected.merge(
        previous,
        on=selection_keys,
        how="left",
        validate="one_to_one",
    )
    metric_universe = events.loc[
        :, ("asset_id", "table_id", "metric_id")
    ].drop_duplicates()
    expected = left.merge(
        metric_universe, on="asset_id", how="left", validate="many_to_many"
    )
    output = expected.merge(
        selected.drop(columns=["asset_id", "timestamp"]),
        on=("_observation_id", "table_id", "metric_id"),
        how="left",
        validate="one_to_one",
    )
    return (
        output.drop(columns="_observation_id")
        .sort_values(
            ["asset_id", "timestamp", "table_id", "metric_id"],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(drop=True)
    )
