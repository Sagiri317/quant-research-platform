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
        frame = frame.loc[frame["disclosure_date"].le(sessions[-1])].copy()
        if frame.empty:
            continue
        positions = sessions.searchsorted(
            frame["disclosure_date"].to_numpy(dtype="datetime64[ns]"),
            side="right",
        )
        if bool((positions >= len(sessions)).any()):
            raise ValueError("financial_availability_next_session_missing")
        frame["available_at"] = sessions.take(positions)
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
        group = result.loc[positions].sort_values(
            ["available_at", "report_period", "version_identity"],
            kind="mergesort",
        )
        for position, row in group.iterrows():
            period = pd.Timestamp(row["report_period"])
            result.loc[position, "ttm_value"] = _ttm_value_at_cutoff(
                group,
                report_period=period,
                cutoff=pd.Timestamp(row["available_at"]),
                value_column=value_column,
            )
        if metric_id not in FLOW_METRIC_IDS:
            continue
        correction_rows = group.loc[group["is_corrected"]]
        for period in sorted(
            pd.Timestamp(value) for value in group["report_period"].drop_duplicates()
        ):
            if period.month == 12:
                continue
            prior_fy = _prior_fy_report_period(period)
            prior_ytd = _prior_year_report_period(period)
            dependencies = {prior_fy, prior_ytd}
            if None in dependencies:
                dependencies.discard(None)
            target_rows = group.loc[group["report_period"].eq(period)]
            target_first_available = target_rows["available_at"].min()
            cutoffs = correction_rows.loc[
                correction_rows["report_period"].isin(dependencies)
                & correction_rows["available_at"].gt(target_first_available),
                "available_at",
            ].drop_duplicates()
            for cutoff in sorted(pd.Timestamp(value) for value in cutoffs):
                if bool(target_rows["available_at"].eq(cutoff).any()):
                    continue
                derived_records.append(
                    {
                        "asset_id": target_rows["asset_id"].iloc[0],
                        "metric_id": metric_id,
                        "report_period": period,
                        "available_at": cutoff,
                        "version_identity": f"derived@{cutoff:%Y-%m-%d}",
                        "is_corrected": True,
                        value_column: _latest_legal_event_value(
                            group,
                            report_period=period,
                            cutoff=cutoff,
                            value_column=value_column,
                        ),
                        "ttm_value": _ttm_value_at_cutoff(
                            group,
                            report_period=period,
                            cutoff=cutoff,
                            value_column=value_column,
                        ),
                    }
                )
    if derived_records:
        result = pd.concat(
            (result, pd.DataFrame.from_records(derived_records)),
            ignore_index=True,
        )
    if result.duplicated(event_key, keep=False).any():
        raise ValueError("ttm_event_key_duplicate")
    return result


def _ttm_value_at_cutoff(
    events: pd.DataFrame,
    *,
    report_period: pd.Timestamp,
    cutoff: pd.Timestamp,
    value_column: str,
) -> float | None:
    current = _latest_legal_event_value(
        events,
        report_period=report_period,
        cutoff=cutoff,
        value_column=value_column,
    )
    if current is None:
        return None
    if report_period.month == 12:
        return current
    prior_fy = _latest_legal_event_value(
        events,
        report_period=_prior_fy_report_period(report_period),
        cutoff=cutoff,
        value_column=value_column,
    )
    prior_ytd = _latest_legal_event_value(
        events,
        report_period=_prior_year_report_period(report_period),
        cutoff=cutoff,
        value_column=value_column,
    )
    if prior_fy is None or prior_ytd is None:
        return None
    return current + prior_fy - prior_ytd


def _latest_legal_event_value(
    events: pd.DataFrame,
    *,
    report_period: pd.Timestamp | None,
    cutoff: pd.Timestamp,
    value_column: str,
) -> float | None:
    if report_period is None:
        return None
    candidates = events.loc[
        events["report_period"].eq(report_period) & events["available_at"].le(cutoff)
    ]
    if candidates.empty:
        return None
    row = candidates.sort_values(
        ["available_at", "version_identity"], kind="mergesort"
    ).iloc[-1]
    return None if pd.isna(row[value_column]) else float(row[value_column])


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
    records: list[dict[str, object]] = []
    for (asset_id, metric_id), stream in right.groupby(
        ["asset_id", "metric_id"], sort=False
    ):
        stream = stream.sort_values(
            ["available_at", "report_period", "version_identity"],
            kind="mergesort",
        )
        asset_observations = left.loc[left["asset_id"].eq(asset_id)].sort_values(
            "timestamp", kind="mergesort"
        )
        for observation in asset_observations.itertuples(index=False):
            legal = stream.loc[stream["available_at"].le(observation.timestamp)]
            if legal.empty:
                current = beginning = prior_year = prior_fy = None
            else:
                latest_period = legal["report_period"].max()
                current = _latest_legal_event_row(
                    legal, report_period=latest_period
                ).iloc[-1]
                beginning_period = _beginning_report_period(latest_period)
                beginning = (
                    _latest_legal_event_row(legal, report_period=beginning_period).iloc[
                        -1
                    ]
                    if beginning_period is not None
                    and bool(legal["report_period"].eq(beginning_period).any())
                    else None
                )
                prior_year_period = _prior_year_report_period(latest_period)
                prior_year = (
                    _latest_legal_event_row(
                        legal, report_period=prior_year_period
                    ).iloc[-1]
                    if prior_year_period is not None
                    and bool(legal["report_period"].eq(prior_year_period).any())
                    else None
                )
                prior_fy_period = _prior_fy_report_period(latest_period)
                prior_fy = (
                    _latest_legal_event_row(legal, report_period=prior_fy_period).iloc[
                        -1
                    ]
                    if bool(legal["report_period"].eq(prior_fy_period).any())
                    else None
                )
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
                    "version_identity": pd.NA
                    if current is None
                    else current["version_identity"],
                    "is_corrected": pd.NA
                    if current is None
                    else current["is_corrected"],
                    "value": np.nan if current is None else current["value"],
                    "beginning_report_period": pd.NaT
                    if beginning is None
                    else beginning["report_period"],
                    "beginning_value": (
                        np.nan if beginning is None else beginning["value"]
                    ),
                    "prior_year_report_period": pd.NaT
                    if prior_year is None
                    else prior_year["report_period"],
                    "prior_year_value": (
                        np.nan if prior_year is None else prior_year["value"]
                    ),
                    "prior_fy_report_period": pd.NaT
                    if prior_fy is None
                    else prior_fy["report_period"],
                    "prior_fy_value": (
                        np.nan if prior_fy is None else prior_fy["value"]
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["asset_id", "timestamp", "metric_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _latest_legal_event_row(
    events: pd.DataFrame,
    *,
    report_period: pd.Timestamp,
) -> pd.DataFrame:
    return events.loc[events["report_period"].eq(report_period)].sort_values(
        ["available_at", "version_identity"], kind="mergesort"
    )


def _beginning_report_period(report_period: pd.Timestamp) -> pd.Timestamp | None:
    return _prior_year_report_period(report_period)


def _prior_year_report_period(report_period: pd.Timestamp) -> pd.Timestamp | None:
    try:
        return report_period.replace(year=report_period.year - 1)
    except ValueError:
        return None


def _prior_fy_report_period(report_period: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=report_period.year - 1, month=12, day=31)
