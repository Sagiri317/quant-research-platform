"""A-share market-session facts and dense panel construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
    strict_string_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
    compute_listing_session_facts,
    market_type_calendar_state,
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.ids import MARKET_ID

ST_STATUS_CODES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
MARKET_PANEL_KEY_COLUMNS = ("market_id", "asset_id", "timestamp")
INDUSTRY_STANDARD_BY_PROVIDER_ID = {
    "P0207": "CSRC2012",
    "P0221": "CAPCO",
}
FINANCIAL_INDUSTRY_LEVEL1_CODES = frozenset({"J"})


def _prepare_market_panel_inputs(
    *, stocks, market_cap, industry_history=None, **inputs
):
    """Prepare request-wide facts after the caller has normalized stock lives."""
    caps = _canonical_market_cap(market_cap)
    industry = _prepare_industry_inputs(
        industry_history,
        industry_sets=inputs.pop("industry_sets", None),
        industry_versions=inputs.pop("industry_versions", None),
        industry_changes=inputs.pop("industry_changes", None),
        industry_snapshots=inputs.pop("industry_snapshots", None),
        stock_master=stocks,
    )
    industry["industry_history"] = industry.pop("history")
    return {**inputs, **industry, "market_cap": caps}


def compute_market_panel(
    market_bars: pd.DataFrame,
    *,
    stock_master: pd.DataFrame,
    market_cap: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    industry_history: pd.DataFrame | None = None,
    industry_sets: pd.DataFrame | None = None,
    industry_versions: pd.DataFrame | None = None,
    industry_snapshots: pd.DataFrame | None = None,
    no_limit_history: pd.DataFrame | None = None,
    special_treatment_history: pd.DataFrame | None = None,
    session_window: pd.DatetimeIndex | None = None,
    required_calendar_start: pd.Timestamp | None = None,
    asset_market_types: pd.Series | None = None,
    suspension_history: pd.DataFrame | None = None,
    market_history: pd.DataFrame | None = None,
    observed_market_sessions: dict[int, pd.DatetimeIndex] | None = None,
) -> pd.DataFrame:
    """Build one slab from prepared facts, retaining suspension rows."""

    bars = _canonical_bars(market_bars)
    if market_history is None:
        market_history = bars.loc[
            :, ["asset_id", "timestamp", "market_type"]
        ].sort_values(["asset_id", "timestamp"], kind="mergesort")
    if observed_market_sessions is None:
        observed_market_sessions = {
            market: pd.DatetimeIndex(group.timestamp.unique()).sort_values()
            for market, group in bars.loc[bars.filling_flag.isin((0, 1))].groupby(
                "market_type", sort=False
            )
        }
    if bars.empty and asset_market_types is None:
        return bars
    sessions = unified_trading_sessions(trading_calendar)
    calendar_bars = bars
    if session_window is not None and not session_window.empty:
        calendar_bars = bars.loc[bars["timestamp"].isin(session_window)]
    validate_market_bar_calendar(
        calendar_bars, trading_calendar, required_calendar_start=required_calendar_start
    )
    if session_window is None:
        session_window = pd.date_range(
            bars["timestamp"].min(), bars["timestamp"].max()
        ).as_unit("ns")
    if session_window.empty:
        return bars.iloc[0:0].copy()
    dense = _dense_lifecycle_panel(
        bars,
        stock_master,
        session_window,
        asset_market_types=asset_market_types,
        market_history=market_history,
    )
    open_session = dense["market_type"].isna() & dense["timestamp"].isin(sessions)
    for market_type, group in dense.groupby("market_type", sort=False):
        state = market_type_calendar_state(trading_calendar, market_type).reindex(
            group["timestamp"]
        )
        required = (
            np.ones(len(group), dtype=bool)
            if required_calendar_start is None
            else group["timestamp"].ge(required_calendar_start).to_numpy()
        )
        if (state.isna().to_numpy() & required).any():
            raise ValueError("a_share_trading_calendar_unknown_session")
        # Earlier unknown dates are state barriers, never published open sessions.
        open_session.loc[group.index] = state.fillna(True).to_numpy(dtype=bool)
    dense = dense.loc[open_session].copy()
    if dense.empty:
        return dense
    dense = _asof_market_cap(dense, market_cap)
    # Earlier rows initialize price/suspension state only. Industry has no
    # recursive state, so validate its frontier only where it is consumed.
    industry_rows = (
        dense["timestamp"].ge(required_calendar_start)
        if required_calendar_start is not None
        else pd.Series(True, index=dense.index)
    )
    classified = _asof_industry(
        dense.loc[industry_rows],
        industry_history,
        industry_sets=industry_sets,
        industry_versions=industry_versions,
        industry_snapshots=industry_snapshots,
        stock_master=stock_master,
    )
    dense = pd.concat((dense.loc[~industry_rows], classified), ignore_index=True)
    dense = _apply_no_limit_exceptions(dense, no_limit_history)
    dense = _apply_special_treatment(dense, special_treatment_history)
    if suspension_history is not None and not suspension_history.empty:
        dense = pd.concat([suspension_history, dense], ignore_index=True)
    dense = _apply_suspension_semantics(dense)
    dense = dense.loc[dense["timestamp"].isin(session_window)].copy()
    delisting_sentinel = pd.Timestamp.max.normalize()
    dense["delisting_date"] = dense["delisting_date"].where(
        dense["delisting_date"] != delisting_sentinel
    )
    facts = compute_listing_session_facts(
        dense,
        trading_calendar,
        market_history,
        observed_market_sessions=observed_market_sessions,
    )
    facts.loc[dense["delisting_date_unresolved"].fillna(True), :] = pd.NA
    dense[facts.columns] = facts
    return dense.sort_values(
        ["timestamp", "market_id", "asset_id"], kind="mergesort"
    ).reset_index(drop=True)


def validate_market_bar_calendar(
    bars: pd.DataFrame, calendar: pd.DataFrame, *, required_calendar_start=None
) -> None:
    if unified_trading_sessions(calendar).empty:
        raise ValueError("a_share_trading_calendar_empty")
    for market_type, group in bars.groupby("market_type", sort=False):
        state = market_type_calendar_state(calendar, market_type)
        if not state.any() and (
            required_calendar_start is None
            or group.timestamp.ge(required_calendar_start).any()
        ):
            raise ValueError("a_share_trading_calendar_empty")
        holiday_fill_dates = group.loc[group["filling_flag"].eq(2), "timestamp"]
        if state.reindex(holiday_fill_dates).fillna(False).any():
            raise ValueError("holiday_fill_conflicts_with_open_session")
        observed_sessions = group.loc[group["filling_flag"].isin((0, 1)), "timestamp"]
        observed_state = state.reindex(observed_sessions)
        unknown_required = observed_state.isna().to_numpy() & (
            np.ones(len(observed_sessions), dtype=bool)
            if required_calendar_start is None
            else observed_sessions.ge(required_calendar_start).to_numpy()
        )
        if observed_state.eq(False).any() or unknown_required.any():
            raise ValueError("market_bar_session_missing_from_calendar")


def _canonical_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_asset_id",
        "timestamp",
        "market_type",
        "trading_status_code",
        "filling_flag",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "period_return",
        "adj_close",
        "adj_open",
        "float_market_cap",
        "total_market_cap",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("market_bars_missing_columns: " + ", ".join(missing))
    result = frame.copy()
    result["asset_id"] = strict_string_series(
        result["source_asset_id"], error="invalid_source_asset_id"
    )
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="invalid_source_timestamp"
    ).dt.normalize()
    for column in (
        "market_type",
        "trading_status_code",
        "filling_flag",
    ):
        result[column] = strict_integer_series(
            result[column], error=f"invalid_{column}", allow_null=True
        )
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "period_return",
        "adj_close",
        "adj_open",
        "float_market_cap",
        "total_market_cap",
        "limit_status",
    ):
        if column in result:
            result[column] = strict_finite_numeric_series(
                result[column],
                error=f"invalid_market_bar_value: {column}",
                allow_null=True,
            )
        else:
            result[column] = np.nan
    result = result.loc[result["market_type"].isin(CURRENT_MARKET_TYPES)]
    if result.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("duplicate_market_panel_key")
    result["market_id"] = str(MARKET_ID)
    result["session_date"] = result["timestamp"]
    result["available_at"] = result["timestamp"]
    return result


def _canonical_stock_master(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"source_asset_id", "listing_date"}
    if not required.issubset(frame.columns):
        raise ValueError("stock_master_missing_columns")
    result = frame.copy()
    result["asset_id"] = strict_string_series(
        result["source_asset_id"], error="invalid_stock_master_asset_id"
    )
    for column in ("listing_date", "delisting_date", "timestamp"):
        if column in result:
            result[column] = strict_timestamp_series(
                result[column], error="invalid_stock_master_date", allow_null=True
            ).dt.normalize()
    result["listing_date"] = strict_timestamp_series(
        result["listing_date"], error="invalid_stock_master_date", allow_null=True
    ).dt.normalize()
    if "delisting_date" not in result:
        result["delisting_date"] = pd.NaT
    result["delisting_date_unresolved"] = (
        result.get("delisting_date_unresolved", pd.Series(False, index=result.index))
        .astype("boolean")
        .fillna(True)
    )
    result["delisting_date"] = result["delisting_date"].where(
        result["delisting_date"].notna() | result["delisting_date_unresolved"],
        pd.Timestamp.max.normalize(),
    )
    for column in (
        "stock_name",
        "company_status",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        if column not in result:
            result[column] = pd.NA
    for column in (
        "stock_name",
        "company_status",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        result[column] = result[column].astype("string")
    result["stock_effective_date"] = result.get(
        "timestamp", result["listing_date"]
    ).fillna(result["listing_date"])
    key = ["asset_id", "stock_effective_date"]
    if result.duplicated(key, keep=False).any():
        duplicates = result.loc[result.duplicated(key, keep=False)]
        if duplicates.drop_duplicates().duplicated(key, keep=False).any():
            raise ValueError("duplicate_stock_master_key")
        result = result.drop_duplicates(key, keep="last")
    result = result.sort_values(key, kind="mergesort")
    if result["delisting_date"].lt(result["listing_date"]).any():
        raise ValueError("invalid_stock_master_lifecycle")
    previous = result.groupby("asset_id", sort=False)[
        ["listing_date", "delisting_date"]
    ].shift()
    changed = previous["listing_date"].notna() & result["listing_date"].ne(
        previous["listing_date"]
    )
    if (
        changed
        & (
            result["listing_date"].le(previous["listing_date"])
            | result["listing_date"].le(previous["delisting_date"])
        )
    ).any():
        raise ValueError("conflicting_stock_master_lifecycles")
    return result


def _canonical_market_cap(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_asset_id" not in frame or "timestamp" not in frame:
        raise ValueError("market_cap_missing_columns")
    result = frame.copy()
    result["asset_id"] = strict_string_series(
        result["source_asset_id"], error="invalid_market_cap_asset_id"
    )
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="invalid_market_cap_date"
    ).dt.normalize()
    for column in ("share_change_date", "total_shares", "a_float_shares"):
        if column not in result:
            result[column] = pd.NaT if column == "share_change_date" else np.nan
    result["share_change_date"] = strict_timestamp_series(
        result["share_change_date"], error="invalid_market_cap_date", allow_null=True
    ).dt.normalize()
    for column in ("total_shares", "a_float_shares"):
        result[column] = strict_finite_numeric_series(
            result[column], error=f"invalid_market_cap_value: {column}", allow_null=True
        )
    if result.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("duplicate_market_cap_key")
    return result.sort_values(["asset_id", "timestamp"], kind="mergesort")


def _dense_lifecycle_panel(
    bars: pd.DataFrame,
    stocks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    asset_market_types: pd.Series | None = None,
    market_history: pd.DataFrame,
) -> pd.DataFrame:
    if asset_market_types is None:
        asset_market_types = bars.drop_duplicates("asset_id").set_index("asset_id")[
            "market_type"
        ]
    assets = tuple(sorted(asset_market_types.index))
    base = pd.DataFrame(
        {"asset_id": pd.Series(assets, dtype=stocks.asset_id.dtype)}
    ).merge(pd.DataFrame({"timestamp": sessions}), how="cross")
    first_listing = stocks.groupby("asset_id").listing_date.min().reindex(base.asset_id)
    first_listing.index = base.index
    base = base.loc[first_listing.isna() | base.timestamp.ge(first_listing)]
    if base.empty:
        return bars.iloc[0:0].copy()
    base = base.merge(
        bars.drop(
            columns=["market_id", "session_date", "available_at"], errors="ignore"
        ).assign(_source_bar_observed=True),
        on=["asset_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    base = _asof_join_stock_master(base, stocks)
    membership = market_history.loc[:, ["asset_id", "timestamp", "market_type"]].rename(
        columns={
            "timestamp": "_market_observed_at",
            "market_type": "_known_market_type",
        }
    )
    membership["asset_id"] = membership.asset_id.astype(stocks.asset_id.dtype)
    result = pd.merge_asof(
        base.sort_values("timestamp", kind="stable"),
        membership.sort_values("_market_observed_at", kind="stable"),
        left_on="timestamp",
        right_on="_market_observed_at",
        by="asset_id",
        direction="backward",
    )
    known = result._market_observed_at.notna() & (
        result.listing_date.isna() | result._market_observed_at.ge(result.listing_date)
    )
    result["market_type"] = result._known_market_type.astype("Int64").where(known)
    result = result.drop(columns=["_market_observed_at", "_known_market_type"])
    result["market_id"] = str(MARKET_ID)
    result["session_date"] = result["timestamp"]
    result["available_at"] = result["timestamp"]
    lifecycle_known = ~result["delisting_date_unresolved"].fillna(True)
    observed = result["_source_bar_observed"].eq(True)
    return (
        result.loc[
            (
                result["timestamp"].ge(result["listing_date"])
                | (result["listing_date"].isna() & observed)
            )
            & (
                (lifecycle_known & result["timestamp"].le(result["delisting_date"]))
                | (~lifecycle_known & observed)
            )
        ]
        .drop(columns="_source_bar_observed")
        .copy()
    )


def _asof_join_stock_master(panel: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "stock_name",
        "listing_date",
        "delisting_date",
        "delisting_date_unresolved",
        "company_status",
    )
    right = stocks.loc[stocks.asset_id.isin(panel.asset_id)]
    # Listing is a life fact; a later status cannot erase or backfill it.
    lives = right.drop_duplicates(["asset_id", "listing_date"], keep="first").loc[
        :, ["asset_id", "listing_date", "delisting_date", "delisting_date_unresolved"]
    ]
    lives = lives.loc[lives.listing_date.notna()].copy()
    lives["stock_effective_date"] = lives.listing_date
    key = ["asset_id", "stock_effective_date"]
    lives = lives.loc[
        ~pd.MultiIndex.from_frame(lives[key]).isin(pd.MultiIndex.from_frame(right[key]))
    ]
    right = pd.concat([lives, right], ignore_index=True)
    result = pd.merge_asof(
        panel.sort_values("timestamp", kind="stable"),
        right.loc[:, ["asset_id", "stock_effective_date", *columns]].sort_values(
            "stock_effective_date", kind="stable"
        ),
        left_on="timestamp",
        right_on="stock_effective_date",
        by="asset_id",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_stock_master"),
    ).drop(columns="stock_effective_date")
    for column in columns:
        joined = f"{column}_stock_master"
        if joined in result:
            result[column] = result.pop(joined)
    result.loc[~result.asset_id.isin(stocks.asset_id), "delisting_date_unresolved"] = (
        True
    )
    for column in (
        "stock_name",
        "company_status",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        if column not in result:
            result[column] = pd.NA
        result[column] = result[column].astype("string")
    if "is_nonfinancial_industry" not in result:
        result["is_nonfinancial_industry"] = pd.Series(
            pd.NA, index=result.index, dtype="boolean"
        )
    else:
        result["is_nonfinancial_industry"] = result["is_nonfinancial_industry"].astype(
            "boolean"
        )
    return result


def _asof_market_cap(panel: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["asset_id"] = strict_string_series(
        panel["asset_id"], error="invalid_market_cap_asset_id"
    )
    return pd.merge_asof(
        panel.sort_values("timestamp", kind="stable"),
        caps.loc[
            :,
            [
                "asset_id",
                "timestamp",
                "share_change_date",
                "total_shares",
                "a_float_shares",
            ],
        ].sort_values("timestamp", kind="stable"),
        on="timestamp",
        by="asset_id",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_cap"),
    )


def _prepare_industry_inputs(
    history,
    *,
    industry_sets=None,
    industry_versions=None,
    industry_changes=None,
    industry_snapshots=None,
    stock_master=None,
):
    """Normalize whole source history once, retaining window-dependent checks."""
    prepared = dict(
        history=None,
        industry_sets=industry_sets,
        industry_versions=industry_versions,
        industry_snapshots=industry_snapshots,
        stock_master=stock_master,
    )
    if industry_versions is None or (
        (history is None or history.empty)
        and (industry_snapshots is None or industry_snapshots.empty)
    ):
        return prepared
    if history is None:
        history = pd.DataFrame(
            columns=(
                "source_asset_id",
                "timestamp",
                "industry_standard_id",
                "industry_level1_code",
                "industry_level1_name",
            )
        )
    if industry_changes is None:
        if not history.empty:
            return prepared
        industry_changes = pd.DataFrame(
            columns=(
                "source_asset_id",
                "changed_item",
                "effective_date",
                "announcement_date",
                "value_after",
            )
        )
    history = history.rename(columns={"source_asset_id": "asset_id"}).copy()
    required = {
        "asset_id",
        "timestamp",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    }
    if not required.issubset(history.columns):
        raise ValueError("industry_history_columns_invalid")
    history["asset_id"] = strict_string_series(
        history["asset_id"], error="invalid_industry_asset_id"
    )
    history["industry_standard_id"] = _canonical_industry_standard(
        history["industry_standard_id"]
    )
    history = history.loc[
        history["industry_standard_id"].isin(INDUSTRY_STANDARD_BY_PROVIDER_ID.values())
    ].copy()
    history["timestamp"] = strict_timestamp_series(
        history["timestamp"], error="invalid_industry_timestamp"
    ).dt.normalize()
    history["industry_level1_code"] = strict_string_series(
        history["industry_level1_code"], error="invalid_industry_code", allow_null=True
    )
    history["_industry_classification_code"] = history["industry_level1_code"]
    history["industry_level1_name"] = strict_string_series(
        history["industry_level1_name"],
        error="invalid_industry_name",
        allow_null=True,
    )
    history["is_nonfinancial_industry"] = pd.Series(
        pd.NA, index=history.index, dtype="boolean"
    )
    history = history.loc[
        _industry_version_active_mask(
            history["industry_standard_id"], history["timestamp"], industry_versions
        )
    ].copy()
    # Preserve competing classifications until the consumed as-of frontier can
    # mark only that asset's industry state unknown.
    history = history.drop_duplicates()
    change_standards = {
        f"{provider}行业代码": standard
        for provider, standard in INDUSTRY_STANDARD_BY_PROVIDER_ID.items()
    }
    changes = industry_changes.loc[
        industry_changes["changed_item"].isin(change_standards),
        [
            "source_asset_id",
            "changed_item",
            "effective_date",
            "announcement_date",
            "value_after",
        ],
    ].rename(columns={"source_asset_id": "asset_id", "effective_date": "timestamp"})
    changes["asset_id"] = strict_string_series(
        changes["asset_id"], error="invalid_industry_asset_id"
    )
    changes["industry_standard_id"] = changes.pop("changed_item").map(change_standards)
    changes["timestamp"] = strict_timestamp_series(
        changes["timestamp"], error="invalid_industry_timestamp"
    ).dt.normalize()
    changes["announcement_date"] = strict_timestamp_series(
        changes["announcement_date"],
        error="invalid_industry_announcement_date",
        allow_null=True,
    ).dt.normalize()
    changes["value_after"] = strict_string_series(
        changes["value_after"], error="invalid_industry_name", allow_null=True
    )
    history = history.merge(
        changes,
        on=["asset_id", "industry_standard_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    history["_industry_known"] = (
        history["announcement_date"].notna()
        & history["industry_level1_code"].notna()
        & (history["value_after"] == history["industry_level1_name"])
    ).fillna(False)
    history["_industry_event_timestamp"] = history["timestamp"]
    # Date-only announcements become usable after their calendar date.
    publication = (history["announcement_date"] + pd.Timedelta(days=1)).where(
        history["_industry_known"]
    )
    history["timestamp"] = history["timestamp"].where(
        publication.isna() | history["timestamp"].ge(publication), publication
    )
    if industry_sets is not None:
        known = history["_industry_known"]
        resolved = _resolve_industry_hierarchy(
            history.loc[known], _industry_dictionary(industry_sets)
        )
        history.loc[known, "_industry_known"] = resolved["industry_level1_code"].notna()
        for column in (
            "industry_level1_code",
            "industry_level1_name",
            "is_nonfinancial_industry",
        ):
            history.loc[known, column] = resolved[column]
    prepared["history"] = history
    return prepared


def _asof_industry(
    panel: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    industry_sets: pd.DataFrame | None,
    industry_versions: pd.DataFrame | None,
    industry_snapshots: pd.DataFrame | None = None,
    stock_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if history is None:
        return panel
    panel = panel.copy()
    panel["asset_id"] = strict_string_series(
        panel["asset_id"], error="invalid_industry_asset_id"
    )
    if industry_snapshots is not None:
        snapshots = industry_snapshots.loc[
            industry_snapshots["source_asset_id"].isin(panel["asset_id"])
        ]
        if not snapshots.empty:
            snapshots = _snapshot_industry_history(
                snapshots,
                industry_sets=industry_sets,
                industry_versions=industry_versions,
                stock_master=stock_master,
            )
            history = pd.concat([history, snapshots], ignore_index=True)
    history = history.loc[history["timestamp"].le(panel["timestamp"].max())].copy()
    if history.empty:
        return panel
    known = history.loc[history["_industry_known"]]
    event_key = ["asset_id", "_industry_event_timestamp", "industry_standard_id"]
    first_observations = known.sort_values(
        "timestamp", kind="mergesort"
    ).drop_duplicates([*event_key, "_industry_classification_code"])
    competing = first_observations.groupby(event_key, sort=False).cumcount().gt(0)
    conflicts = first_observations.loc[competing].groupby(event_key)["timestamp"].min()
    conflict_at = pd.Series(
        conflicts.reindex(pd.MultiIndex.from_frame(history[event_key])).to_numpy(),
        index=history.index,
    )
    history["_industry_value_conflict"] = history["timestamp"].ge(conflict_at)
    state_key = ["asset_id", "_industry_event_timestamp"]
    first_standards = known.sort_values("timestamp", kind="mergesort").drop_duplicates(
        [*state_key, "industry_standard_id"]
    )
    competing_standards = (
        first_standards.groupby(state_key, sort=False).cumcount().gt(0)
    )
    standards_at = (
        first_standards.loc[competing_standards].groupby(state_key)["timestamp"].min()
    )
    standard_conflict_at = pd.Series(
        standards_at.reindex(pd.MultiIndex.from_frame(history[state_key])).to_numpy(),
        index=history.index,
    )
    history["_industry_value_conflict"] |= history["timestamp"].ge(standard_conflict_at)
    # Unknown publication blocks unproved coverage from its effective date.
    # A late older event cannot replace a newer state or cross that barrier.
    history = history.sort_values(
        ["timestamp", "_industry_event_timestamp", "asset_id", "_industry_known"],
        kind="mergesort",
    )
    # All events becoming available together share one effective frontier. This
    # excludes superseded rows even when their older conflicts arrive that day.
    latest_effective = (
        history.groupby(["asset_id", "timestamp"], sort=False)[
            "_industry_event_timestamp"
        ]
        .transform("max")
        .groupby(history["asset_id"], sort=False)
        .cummax()
    )
    history = history.loc[history["_industry_event_timestamp"].eq(latest_effective)]
    history["_industry_standard_conflict"] = (
        history["industry_standard_id"]
        .where(history["_industry_known"])
        .groupby([history["asset_id"], history["timestamp"]], sort=False)
        .transform("nunique")
        .gt(1)
    )
    history["_industry_standard_conflict"] |= history.groupby(
        ["asset_id", "timestamp"], sort=False
    )["_industry_value_conflict"].transform("any")
    history = history.drop_duplicates(["asset_id", "timestamp"], keep="last")
    right = history.loc[
        :,
        [
            "asset_id",
            "timestamp",
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
            "is_nonfinancial_industry",
            "_industry_event_timestamp",
            "_industry_known",
            "_industry_standard_conflict",
        ],
    ]
    current = pd.merge_asof(
        panel.sort_values("timestamp", kind="mergesort"),
        right,
        on="timestamp",
        by="asset_id",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_industry"),
    )
    # Only a frontier consumed by this panel can block its output. Historical
    # computation rows are included in the panel and cannot use a future repair.
    valid_standard = current["_industry_known"].fillna(False) & ~current[
        "_industry_standard_conflict"
    ].fillna(False)
    valid_standard &= _industry_version_active_mask(
        current["industry_standard_id_industry"],
        current["timestamp"],
        industry_versions,
        event_timestamps=current["_industry_event_timestamp"],
    )
    for column in (
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
        "is_nonfinancial_industry",
    ):
        current[column] = current[f"{column}_industry"].combine_first(current[column])
        current.loc[~valid_standard, column] = pd.NA
        current.drop(columns=f"{column}_industry", inplace=True)
    current.drop(
        columns=[
            "_industry_event_timestamp",
            "_industry_known",
            "_industry_standard_conflict",
        ],
        inplace=True,
    )
    order = current["asset_id"].map(
        {asset: i for i, asset in enumerate(panel["asset_id"].unique())}
    )
    result = current.iloc[
        np.lexsort((current["timestamp"].to_numpy(), order.to_numpy()))
    ].reset_index(drop=True)
    for column in (
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        result[column] = result[column].astype("string")
    result["is_nonfinancial_industry"] = result["is_nonfinancial_industry"].astype(
        "boolean"
    )
    return result


def _snapshot_industry_history(
    snapshots, *, industry_sets, industry_versions, stock_master
):
    """Qualify public observations without inventing publication or issuer history."""
    if industry_sets is None or stock_master is None:
        raise ValueError("industry_snapshot_context_missing")
    result = snapshots.rename(columns={"source_asset_id": "asset_id"}).copy()
    result["asset_id"] = strict_string_series(
        result["asset_id"], error="invalid_industry_asset_id"
    )
    result["industry_standard_id"] = _canonical_industry_standard(
        result["industry_standard_id"]
    )
    result["_industry_event_timestamp"] = strict_timestamp_series(
        result["observation_date"], error="invalid_industry_timestamp"
    ).dt.normalize()
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="invalid_industry_timestamp"
    ).dt.normalize()
    result["_snapshot_available_at"] = result["timestamp"]
    result["timestamp"] = result["_industry_event_timestamp"]
    result = _asof_join_stock_master(result, stock_master)
    result["timestamp"] = result.pop("_snapshot_available_at")
    observed = result["_industry_event_timestamp"]
    result["_industry_known"] = (
        observed.ge(result["listing_date"])
        & observed.le(result["delisting_date"])
        & _industry_version_active_mask(
            result["industry_standard_id"], observed, industry_versions
        )
    )
    lookup = _industry_dictionary(industry_sets)
    result["industry_level1_code"] = (
        result["industry_category_code"] + result["industry_code"]
    ).astype("string")
    result["_industry_classification_code"] = result["industry_level1_code"]
    valid_paths = []
    for row in result.itertuples(index=False):
        category = (row.industry_standard_id, row.industry_category_code)
        leaf = (row.industry_standard_id, row.industry_level1_code)
        major = "" if pd.isna(row.industry_major_code) else row.industry_major_code
        category_entry = lookup.get(category)
        leaf_entry = lookup.get(leaf)
        valid = (
            category_entry is not None
            and leaf_entry is not None
            and str(category_entry.industry_level) == "1"
            and pd.notna(leaf_entry.parent_industry_code)
            and leaf_entry.parent_industry_code == (major or row.industry_category_code)
        )
        if major:
            major_entry = lookup.get((row.industry_standard_id, major))
            valid = (
                valid
                and major_entry is not None
                and pd.notna(major_entry.parent_industry_code)
                and major_entry.parent_industry_code == row.industry_category_code
            )
        valid_paths.append(valid)
    result["_industry_known"] &= pd.Series(valid_paths, index=result.index)
    result["industry_level1_name"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["is_nonfinancial_industry"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    known = result["_industry_known"]
    if known.any():
        resolved = _resolve_industry_hierarchy(result.loc[known], lookup)
        for column in (
            "industry_level1_code",
            "industry_level1_name",
            "is_nonfinancial_industry",
        ):
            result.loc[known, column] = resolved[column]
    return result.loc[
        :,
        [
            "asset_id",
            "timestamp",
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
            "is_nonfinancial_industry",
            "_industry_event_timestamp",
            "_industry_known",
            "_industry_classification_code",
        ],
    ]


def _canonical_industry_standard(values: pd.Series) -> pd.Series:
    result = strict_string_series(values, error="invalid_industry_standard_id")
    return result.replace(INDUSTRY_STANDARD_BY_PROVIDER_ID)


def _canonical_industry_versions(
    versions: pd.DataFrame,
    standards: set[str],
) -> pd.DataFrame:
    required = {
        "industry_standard_id",
        "implement_date",
        "abandon_date",
    }
    if not required.issubset(versions.columns):
        raise ValueError("industry_versions_columns_invalid")
    right = versions.loc[:, list(required)].copy()
    right["abandon_date_unresolved"] = (
        versions.get("abandon_date_unresolved", pd.Series(False, index=versions.index))
        .astype("boolean")
        .fillna(True)
    )
    right["industry_standard_id"] = _canonical_industry_standard(
        right["industry_standard_id"]
    )
    right = right.loc[right["industry_standard_id"].isin(standards)].copy()
    right["implement_date"] = strict_timestamp_series(
        right["implement_date"],
        error="industry_version_date_invalid",
        allow_null=True,
        errors="coerce",
    ).dt.normalize()
    values = right["abandon_date"]
    parsed = strict_timestamp_series(
        values, error="industry_version_date_invalid", allow_null=True, errors="coerce"
    ).dt.normalize()
    right["abandon_date_unresolved"] |= (
        values.notna() & values.astype("string").str.strip().ne("") & parsed.isna()
    )
    right["abandon_date"] = parsed
    right = right.loc[right["implement_date"].notna()].drop_duplicates()
    key = ["industry_standard_id", "implement_date"]
    duplicate = right.duplicated(key, keep=False)
    if duplicate.any():
        unknown = right.groupby(key)["abandon_date_unresolved"].transform("any")
        different = (
            right.groupby(key)["abandon_date"]
            .transform(lambda values: values.nunique(dropna=False))
            .gt(1)
        )
        earliest = right.groupby(key)["abandon_date"].transform("min")
        right.loc[duplicate, "abandon_date"] = earliest.loc[duplicate].where(
            ~unknown.loc[duplicate]
        )
        right["abandon_date_unresolved"] = unknown | different
        right = right.drop_duplicates(key)
    return right


def _industry_version_active_mask(
    standards: pd.Series,
    timestamps: pd.Series,
    versions: pd.DataFrame,
    *,
    event_timestamps: pd.Series | None = None,
) -> pd.Series:
    canonical_standards = strict_string_series(
        standards,
        error="invalid_industry_standard_id",
        allow_null=True,
    ).replace(INDUSTRY_STANDARD_BY_PROVIDER_ID)
    right = _canonical_industry_versions(versions, set(canonical_standards.dropna()))
    event_values = timestamps if event_timestamps is None else event_timestamps
    size = len(standards)
    if len(timestamps) != size or len(event_values) != size:
        raise ValueError("industry_version_input_length_mismatch")
    observation_count = np.zeros(size, dtype=np.intp)
    event_count = np.zeros(size, dtype=np.intp)
    same_version = np.zeros(size, dtype=bool)
    unresolved = np.zeros(size, dtype=bool)
    for version in right.itertuples(index=False):
        selected = canonical_standards.eq(version.industry_standard_id).to_numpy(
            dtype=bool, na_value=False
        )
        if version.abandon_date_unresolved:
            uncertain_from = (
                version.implement_date
                if pd.isna(version.abandon_date)
                else version.abandon_date
            )
            unresolved |= selected & (
                timestamps.ge(uncertain_from).to_numpy(dtype=bool, na_value=False)
                | event_values.ge(uncertain_from).to_numpy(dtype=bool, na_value=False)
            )
        observed = selected & timestamps.ge(version.implement_date).to_numpy(
            dtype=bool, na_value=False
        )
        if not pd.isna(version.abandon_date):
            observed &= timestamps.lt(version.abandon_date).to_numpy(
                dtype=bool, na_value=False
            )
        if event_timestamps is None:
            event = observed
        else:
            event = selected & event_values.ge(version.implement_date).to_numpy(
                dtype=bool, na_value=False
            )
            if not pd.isna(version.abandon_date):
                event &= event_values.lt(version.abandon_date).to_numpy(
                    dtype=bool, na_value=False
                )
        observation_count += observed
        event_count += event
        same_version |= observed & event
    return pd.Series(
        (observation_count == 1) & (event_count == 1) & same_version & ~unresolved,
        index=standards.index,
        dtype=bool,
    )


def _industry_dictionary(sets: pd.DataFrame):
    required = {
        "industry_code",
        "industry_standard_id",
        "industry_name",
        "parent_industry_code",
        "industry_level",
    }
    if not required.issubset(sets.columns):
        raise ValueError("industry_sets_columns_invalid")
    dictionary = sets.loc[:, list(required)].copy()
    dictionary["industry_standard_id"] = _canonical_industry_standard(
        dictionary["industry_standard_id"]
    )
    for column in required.difference({"industry_standard_id"}):
        dictionary[column] = strict_string_series(
            dictionary[column],
            error="invalid_industry_dictionary_value",
            allow_empty=column == "parent_industry_code",
            allow_null=True,
        )
    key = ["industry_standard_id", "industry_code"]
    if dictionary.duplicated(key, keep=False).any():
        duplicate = dictionary.loc[dictionary.duplicated(key, keep=False)]
        conflicting = duplicate.drop_duplicates().duplicated(key, keep=False)
        unavailable = pd.MultiIndex.from_frame(
            duplicate.drop_duplicates().loc[conflicting, key]
        )
        dictionary = dictionary.loc[
            ~pd.MultiIndex.from_frame(dictionary[key]).isin(unavailable)
        ]
        dictionary = dictionary.drop_duplicates()
    return {
        (row.industry_standard_id, row.industry_code): row
        for row in dictionary.itertuples(index=False)
    }


def _resolve_industry_hierarchy(
    history: pd.DataFrame,
    lookup: dict,
) -> pd.DataFrame:
    result = history.copy()
    resolved_codes: list[str | None] = []
    resolved_names: list[str | None] = []
    for row in result.loc[
        :, ["industry_standard_id", "industry_level1_code"]
    ].itertuples(index=False):
        standard = str(row.industry_standard_id)
        code = str(row.industry_level1_code)
        seen: set[str] = set()
        while True:
            if code in seen:
                resolved_codes.append(None)
                resolved_names.append(None)
                break
            seen.add(code)
            entry = lookup.get((standard, code))
            if entry is None:
                resolved_codes.append(None)
                resolved_names.append(None)
                break
            if str(entry.industry_level) == "1":
                resolved_codes.append(code)
                resolved_names.append(
                    None if pd.isna(entry.industry_name) else str(entry.industry_name)
                )
                break
            parent = entry.parent_industry_code
            if pd.isna(parent) or not str(parent).strip():
                resolved_codes.append(None)
                resolved_names.append(None)
                break
            code = str(parent).strip()
    result["industry_level1_code"] = pd.Series(
        resolved_codes, index=result.index, dtype="string"
    )
    result["industry_level1_name"] = pd.Series(
        resolved_names, index=result.index, dtype="string"
    )
    result["is_nonfinancial_industry"] = (
        (~result["industry_level1_code"].isin(FINANCIAL_INDUSTRY_LEVEL1_CODES))
        .astype("boolean")
        .where(result["industry_level1_code"].notna())
    )
    return result


def _apply_no_limit_exceptions(
    panel: pd.DataFrame,
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return panel
    result = panel.copy()
    right = history.rename(columns={"source_asset_id": "asset_id"}).copy()
    required = {"asset_id", "timestamp"}
    if not required.issubset(right.columns):
        raise ValueError("no_limit_history_columns_invalid")
    right["asset_id"] = strict_string_series(
        right["asset_id"], error="invalid_no_limit_asset_id"
    )
    right["timestamp"] = strict_timestamp_series(
        right["timestamp"], error="invalid_no_limit_date"
    ).dt.normalize()
    if right.duplicated(["asset_id", "timestamp"], keep=False).any():
        raise ValueError("no_limit_history_key_duplicate")
    event_keys = pd.MultiIndex.from_frame(right.loc[:, ["asset_id", "timestamp"]])
    panel_keys = pd.MultiIndex.from_frame(result.loc[:, ["asset_id", "timestamp"]])
    result.loc[panel_keys.isin(event_keys), "limit_status"] = 0.0
    return result


def _apply_special_treatment(
    panel: pd.DataFrame,
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    result = panel.copy()
    result["asset_id"] = strict_string_series(
        result["asset_id"], error="invalid_special_treatment_asset_id"
    )
    status = result["trading_status_code"].astype("Int64")
    known = status.between(0, 16).fillna(False)
    result["is_st"] = status.isin(ST_STATUS_CODES).astype("boolean").where(known)
    result["is_pt"] = status.eq(16).where(known)
    if history is None or history.empty:
        return result
    history = history.rename(columns={"source_asset_id": "asset_id"})
    history["asset_id"] = strict_string_series(
        history["asset_id"], error="invalid_special_treatment_asset_id"
    )
    right = history.loc[
        :, ["asset_id", "announcement_date", "effective_date", "is_st", "is_pt"]
    ].copy()
    right = right.loc[
        right["announcement_date"].notna() & right["effective_date"].notna()
    ].copy()
    right["timestamp"] = right[["announcement_date", "effective_date"]].max(axis=1)
    right = right.loc[right["timestamp"] <= result["timestamp"].max()]
    right = right.sort_values(
        ["timestamp", "effective_date", "asset_id"], kind="mergesort"
    )
    # Disagreement becomes unknown only when both facts are known. The earlier
    # prefix and any agreeing state field remain usable; no source row wins.
    event_key = [right["asset_id"], right["effective_date"]]
    for column in ("is_st", "is_pt"):
        state = right[column].astype("boolean").astype("Int8").fillna(-1)
        first = state.groupby(event_key, sort=False).transform("first")
        conflict_at = (
            right["timestamp"]
            .where(state.ne(first))
            .groupby(event_key, sort=False)
            .transform("min")
        )
        right[column] = right[column].where(
            conflict_at.isna() | right["timestamp"].lt(conflict_at)
        )
    right = right.drop_duplicates()
    # A late announcement of an older effective event must not replace newer facts.
    latest_effective = right.groupby("asset_id", sort=False)["effective_date"].cummax()
    right = right.loc[right["effective_date"].eq(latest_effective)]
    right = right.drop_duplicates(["asset_id", "timestamp"], keep="last")
    right = right.loc[:, ["asset_id", "timestamp", "is_st", "is_pt"]].sort_values(
        ["timestamp", "asset_id"], kind="mergesort"
    )
    merged = pd.merge_asof(
        result.sort_values(["timestamp", "asset_id"], kind="mergesort"),
        right,
        on="timestamp",
        by="asset_id",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_event"),
    )
    merged["is_st"] = merged["is_st_event"].combine_first(merged["is_st"])
    merged["is_pt"] = merged["is_pt_event"].combine_first(merged["is_pt"])
    return merged.drop(columns=["is_st_event", "is_pt_event"])


def _apply_suspension_semantics(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["asset_id", "timestamp"], kind="mergesort").copy()
    result["is_trading"] = (
        result["filling_flag"]
        .eq(0)
        .astype("boolean")
        .where(result["filling_flag"].isin((0, 1)))
    )
    lives = result.groupby(
        ["asset_id", "listing_date"], sort=False, dropna=False
    ).ngroup()
    trading = result["is_trading"].fillna(False)
    suspended = result["filling_flag"].eq(1).fillna(False)
    unknown = ~(trading | suspended)
    for close_column, price_columns in (
        ("close", ("close", "open", "high", "low")),
        ("adj_close", ("adj_close", "adj_open")),
    ):
        missing_baseline = unknown | (trading & result[close_column].isna())
        last_close = (
            result[close_column]
            .where(trading)
            .groupby(
                [lives, missing_baseline.groupby(lives, sort=False).cumsum()],
                sort=False,
            )
            .ffill()
        )
        for column in price_columns:
            result[column] = result[column].where(~suspended, last_close)
    for column in ("after_hours_volume", "after_hours_amount"):
        if column not in result:
            result[column] = np.nan
    for column in (
        "period_return",
        "volume",
        "amount",
        "after_hours_volume",
        "after_hours_amount",
    ):
        result[column] = result[column].where(~suspended, 0.0)
    free_shares = result["float_market_cap"].div(
        result["close"].where(result["close"].gt(0.0))
    )
    turnover = result["volume"].div(free_shares.where(free_shares.gt(0.0)))
    result["turnover"] = turnover.astype("float64").where(~suspended, 0.0)
    reopening = trading & suspended.groupby(lives, sort=False).shift(
        1, fill_value=False
    )
    prior_close = result["adj_close"].groupby(lives, sort=False).shift(1)
    reopening_return = result["adj_close"].div(prior_close) - 1.0
    result["period_return"] = result["period_return"].where(
        ~reopening | reopening_return.isna(), reopening_return
    )
    result["is_a_share_scope"] = result["market_type"].isin(CURRENT_MARKET_TYPES)
    return result
