"""A-share portfolio-only session and lifecycle evidence projections."""

from __future__ import annotations

from bisect import bisect_right
from contextlib import ExitStack
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.portfolios import (
    _EXECUTION_COLUMNS,
    EQUITY_EVENT_COLUMNS,
    _portfolio_session_frame,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import decode_exact_numeric
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
    market_type_calendar_state,
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.processed import panels
from quant_research_platform.markets.implementations.a_share.processed.producer import (
    DEFAULT_BATCH_ROWS,
    _processed_axis,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    _iter_dataset_session_frames,
    materialize_dataset_session,
    open_dataset_session,
)

_MARKET_FACT_COLUMNS = ("source_asset_id", "timestamp", "market_type", "filling_flag")
_STOCK_FACT_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "listing_date",
    "delisting_date",
    "company_status",
)


_CORPORATE_SOURCE_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "kind",
    "market_type",
    "announcement_date",
    "settlement_date",
    "amount",
)
_CORPORATE_EVENT_COLUMNS = (
    "event_id",
    "asset_id",
    "listing_date",
    "kind",
    "record_at",
    "effective_at",
    "settlement_at",
    "available_at",
    "cash_per_share",
    "additional_shares_per_share",
    "evidence_identity",
    "entitlement_proved",
)
CORPORATE_ACTION_POLICY = {
    "record_date": "previous_own_market_session_for_ordinary_CA_SD",
    "late_disclosure": "retain_actual_record_shares_confirm_only_when_available",
    "missing_cash_payment_date": "ex_date_open",
    "voluntary_subscription": "none",
    "stock_distribution_availability": "source_settlement_date_or_locked",
}


@dataclass(frozen=True, slots=True)
class _PortfolioSession:
    timestamp: pd.Timestamp
    panel: pd.DataFrame
    valuation: pd.DataFrame
    closed_assets: frozenset[str]
    execution: pd.DataFrame
    unproved_lifecycle_assets: frozenset[str]
    confirmed_delisted_lifecycles: frozenset[tuple[str, pd.Timestamp]]
    unknown_market_assets: frozenset[str]
    rights_events: pd.DataFrame
    corporate_events: pd.DataFrame = field(default_factory=pd.DataFrame)


def _portfolio_session_evidence(
    market_bars_root,
    stock_master_root,
    panel_root,
    calendar,
    *,
    panel_columns,
    include_asset_sessions,
):
    bars = open_dataset_session(
        **_dataset_input_options(market_bars_root)
    ).projected_view(_MARKET_FACT_COLUMNS)
    panel = open_dataset_session(**_dataset_input_options(panel_root))
    business_columns = set(panel_columns) & {
        "open",
        "close",
        "adj_open",
        "adj_close",
        "amount",
        "period_return",
        "float_market_cap",
        "total_market_cap",
        "industry_standard_id",
        "industry_level1_code",
    }
    if not business_columns <= set(panel.logical_columns):
        raise ValueError("portfolio_input_business_columns_excluded")
    asset_sessions = {}
    if include_asset_sessions:
        _, _, _, asset_sessions, _, _raw_axis = _processed_axis(
            panel, panel_source=False, include_asset_sessions=True
        )
    stocks = panels._canonical_stock_master(
        materialize_dataset_session(
            open_dataset_session(
                **_dataset_input_options(stock_master_root)
            ).projected_view(_STOCK_FACT_COLUMNS)
        )
    )
    _, _, membership, _, _, raw_axis = _processed_axis(
        bars, panel_source=True, calendar=calendar, stock_master=stocks
    )
    return bars, panel, membership, asset_sessions, stocks, raw_axis


def iter_portfolio_sessions(
    *,
    market_bars_root,
    stock_master_root,
    panel_root,
    panel_columns,
    calendar,
    time_window,
    execution_root=None,
    equity_events_root=None,
    corporate_actions_root=None,
    mark_price_columns=("open", "close"),
):
    """Project proved closed-market marks without changing materialized inputs."""
    if mark_price_columns not in {("open", "close"), ("adj_open", "adj_close")}:
        raise ValueError("portfolio_mark_price_columns_invalid")
    if not set(mark_price_columns) <= set(panel_columns):
        raise ValueError("portfolio_mark_price_columns_missing")
    bars, panel, membership, asset_sessions, stocks, raw_axis = (
        _portfolio_session_evidence(
            market_bars_root,
            stock_master_root,
            panel_root,
            calendar,
            panel_columns=panel_columns,
            include_asset_sessions=execution_root is not None,
        )
    )
    rights_root = equity_events_root
    rights_session = (
        open_dataset_session(**_dataset_input_options(rights_root))
        if rights_root is not None
        else None
    )
    if rights_session is not None and not {
        "value_per_old_unit",
        "remaining_old_units_per_old_unit",
        "successor_units_per_old_unit",
    } <= set(rights_session.logical_columns):
        raise ValueError("portfolio_input_business_columns_excluded")
    rights = (
        materialize_dataset_session(rights_session.projected_view(EQUITY_EVENT_COLUMNS))
        if rights_session is not None
        else pd.DataFrame(columns=EQUITY_EVENT_COLUMNS)
    )
    for column in (
        "value_per_old_unit",
        "remaining_old_units_per_old_unit",
        "successor_units_per_old_unit",
    ):
        rights[column] = rights[column].map(
            lambda value: None if pd.isna(value) else decode_exact_numeric(value)
        )
    if rights.timestamp.isna().any():
        raise ValueError("equity_event_timestamp_invalid")
    rights = rights.loc[rights.timestamp.le(pd.Timestamp(time_window.end))].copy()
    rights["entitlement_proved"] = True
    pending_rights = rights
    rights_lineage = {}
    corporate = (
        _corporate_action_events(
            materialize_dataset_session(
                open_dataset_session(**_dataset_input_options(corporate_actions_root))
                .filtered_view(time_window)
                .projected_view(_CORPORATE_SOURCE_COLUMNS)
            ),
            stocks,
            calendar,
        )
        if corporate_actions_root is not None
        else pd.DataFrame(columns=_CORPORATE_EVENT_COLUMNS)
    )
    sessions = unified_trading_sessions(calendar)
    starts = corporate.record_at.fillna(corporate.effective_at)
    ends = corporate[["effective_at", "settlement_at", "available_at"]].max(axis=1)
    positions = sessions.searchsorted(ends)
    last_session = pd.Series(pd.NaT, index=corporate.index, dtype="datetime64[ns]")
    inside = positions < len(sessions)
    last_session.loc[inside] = sessions.take(positions[inside])
    calendar = calendar.set_index(["calendar_date", "source_market_type"])[
        "is_trading_day"
    ]
    if calendar.index.has_duplicates:
        raise ValueError("duplicate_trading_calendar_key")
    market_events = iter(membership.sort_values("timestamp").groupby("timestamp"))
    pending_membership = next(market_events, None)
    market_types = pd.Series(dtype="Int64")
    unknown_membership: set[str] = set()
    blocked: set[str] = set()
    last = pd.DataFrame(columns=panel_columns).set_index("asset_id", drop=False)
    masks = pd.DataFrame(columns=(*_EXECUTION_COLUMNS, "execution_at")).set_index(
        "asset_id", drop=False
    )

    def ordered_bars():
        dates, assets = raw_axis
        block_size = max(1, DEFAULT_BATCH_ROWS // max(1, len(assets)))
        for start in range(0, len(dates), block_size):
            core = dates[start : start + block_size]
            frame = materialize_dataset_session(
                bars.filtered_view(TimeWindow(core[0], core[-1]))
            ).sort_values("timestamp", kind="mergesort")
            for _, day in frame.groupby("timestamp", sort=False):
                yield day

    with closing_iterator(ExitStack()) as channels:
        bar_frames = channels.enter_context(closing_iterator(ordered_bars()))
        panel_frames = channels.enter_context(
            closing_iterator(_iter_dataset_session_frames(panel, panel_columns))
        )
        execution_frames = (
            channels.enter_context(
                closing_iterator(
                    _iter_dataset_session_frames(
                        open_dataset_session(**_dataset_input_options(execution_root)),
                        _EXECUTION_COLUMNS,
                    )
                )
            )
            if execution_root is not None
            else iter(())
        )
        pending_bar = next(bar_frames, None)
        pending_panel = next(panel_frames, None)
        pending_execution = next(execution_frames, None)
        dates = pd.date_range(
            calendar.index.get_level_values("calendar_date").min(),
            pd.Timestamp(time_window.end),
            freq="D",
        ).as_unit("ns")
        for timestamp in dates:
            day_bars = pd.DataFrame(columns=_MARKET_FACT_COLUMNS)
            while (
                pending_bar is not None
                and pd.Timestamp(pending_bar.timestamp.iloc[0]) <= timestamp
            ):
                current = pending_bar
                unknown = ~current.market_type.isin(CURRENT_MARKET_TYPES)
                unknown_membership.update(
                    current.loc[unknown, "source_asset_id"].astype(str)
                )
                unknown_membership.difference_update(
                    current.loc[~unknown, "source_asset_id"].astype(str)
                )
                blocked.update(
                    current.loc[
                        unknown | ~current.filling_flag.isin((0, 1, 2)),
                        "source_asset_id",
                    ].astype(str)
                )
                if pd.Timestamp(current.timestamp.iloc[0]) == timestamp:
                    day_bars = current.set_index("source_asset_id", drop=False)
                pending_bar = next(bar_frames, None)
            while pending_membership is not None and pending_membership[0] <= timestamp:
                event = pending_membership[1].set_index("asset_id").market_type
                market_types = pd.concat(
                    (market_types.drop(event.index, errors="ignore"), event)
                )
                pending_membership = next(market_events, None)
            status = calendar.reindex(
                pd.MultiIndex.from_arrays(
                    ([timestamp] * len(market_types), market_types),
                    names=("calendar_date", "source_market_type"),
                )
            )
            status.index = market_types.index
            blocked.update(status.index[status.isna()].astype(str))
            closed = frozenset(
                str(asset)
                for asset in status.index[status.eq(False).fillna(False)]
                if str(asset) not in unknown_membership
            )
            raw = pd.DataFrame(columns=panel_columns)
            while (
                pending_panel is not None
                and pd.Timestamp(pending_panel.timestamp.iloc[0]) <= timestamp
            ):
                if pd.Timestamp(pending_panel.timestamp.iloc[0]) == timestamp:
                    raw = _portfolio_session_frame(pending_panel, panel_columns)
                pending_panel = next(panel_frames, None)
            current_trading_assets = frozenset()
            if not raw.empty:
                indexed = raw.set_index("asset_id", drop=False)
                if set(indexed.index.astype(str)) & closed:
                    raise ValueError("portfolio_market_panel_calendar_conflict")
                prices = indexed.loc[:, list(mark_price_columns)]
                valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
                flags = day_bars.filling_flag.reindex(indexed.index)
                known = flags.isin((0, 1)) & ~indexed.index.isin(unknown_membership)
                current_trading_assets = frozenset(
                    indexed.index[known & flags.eq(0)].astype(str)
                )
                blocked.update(indexed.index[~valid | ~known].astype(str))
                real = valid & known & flags.eq(0)
                blocked.difference_update(indexed.index[real].astype(str))
                previous = indexed.loc[real]
                if not previous.empty:
                    last = (
                        previous.copy()
                        if last.empty
                        else pd.concat(
                            (last.drop(previous.index, errors="ignore"), previous)
                        )
                    )
            blocked.update(
                status.index[
                    status.eq(True).fillna(False) & ~status.index.isin(raw.asset_id)
                ].astype(str)
            )
            execution = pd.DataFrame(columns=_EXECUTION_COLUMNS)
            while (
                pending_execution is not None
                and pd.Timestamp(pending_execution.timestamp.iloc[0]) <= timestamp
            ):
                if pd.Timestamp(pending_execution.timestamp.iloc[0]) == timestamp:
                    execution = _portfolio_session_frame(
                        pending_execution, _EXECUTION_COLUMNS
                    )
                pending_execution = next(execution_frames, None)
            if not raw.empty:
                masks = masks.drop(raw.asset_id, errors="ignore")
            if not execution.empty:
                execution = execution.copy()
                endpoints = []
                for asset in execution.asset_id.astype(str):
                    axis = asset_sessions.get(asset, ())
                    position = bisect_right(axis, timestamp)
                    endpoints.append(axis[position] if position < len(axis) else pd.NaT)
                execution["execution_at"] = endpoints
                masks = pd.concat((masks, execution.set_index("asset_id", drop=False)))
                masks[["can_enter", "can_exit"]] = masks[
                    ["can_enter", "can_exit"]
                ].astype("boolean")
            if timestamp not in sessions or timestamp < pd.Timestamp(time_window.start):
                continue
            life_marks = last.loc[
                last.index.isin(closed)
                | last.index.isin(blocked)
                | ~last.index.isin(raw.asset_id)
            ]
            unproved_lifecycle = frozenset()
            carried = last.loc[last.index.isin(closed)].copy()
            if not life_marks.empty:
                facts = panels._asof_join_stock_master(
                    pd.DataFrame(
                        {
                            "asset_id": life_marks.index.astype(stocks.asset_id.dtype),
                            "timestamp": timestamp,
                        }
                    ),
                    stocks,
                ).set_index("asset_id")
                same_life = facts.listing_date.notna() & facts.listing_date.eq(
                    life_marks.listing_date
                )
                unproved_lifecycle = frozenset(same_life.index[~same_life].astype(str))
                alive = (
                    same_life
                    & facts.listing_date.le(timestamp)
                    & facts.delisting_date.ge(timestamp)
                    & facts.listing_date.le(life_marks.timestamp)
                )
                carried = carried.loc[alive.reindex(carried.index).fillna(False)]
                carried["timestamp"] = timestamp
                carried["market_type"] = market_types.reindex(carried.index)
                for opening, closing in (("open", "close"), ("adj_open", "adj_close")):
                    if opening in carried and closing in carried:
                        carried[opening] = carried[closing]
                        carried.loc[carried.index.isin(blocked), [opening, closing]] = (
                            np.nan
                        )
            valuation = raw.copy()
            # A missing side does not invalidate the other observed trading phase.
            valuation.loc[
                valuation.asset_id.isin(blocked - current_trading_assets),
                [
                    name
                    for name in ("open", "close", "adj_open", "adj_close")
                    if name in valuation
                ],
            ] = np.nan
            if not carried.empty:
                valuation = (
                    carried.reset_index(drop=True)
                    if valuation.empty
                    else pd.concat(
                        (valuation, carried.reset_index(drop=True)), ignore_index=True
                    )
                )
            # Use actual as-of source rows, not the synthetic life-start rows used
            # to construct the historical panel. A later status must not become
            # early delisting knowledge or link a reused code to old equity.
            known_lives = stocks.loc[stocks.stock_effective_date.le(timestamp)]
            known_lives = known_lives.drop_duplicates(
                ["asset_id", "listing_date"], keep="last"
            )
            terminated = known_lives.loc[known_lives.delisting_date.le(timestamp)]
            confirmed_delisted_lifecycles = frozenset(
                (str(asset), pd.Timestamp(listing))
                for asset, listing in terminated[
                    ["asset_id", "listing_date"]
                ].itertuples(index=False, name=None)
            )
            # Events are recognized on the first market session when both their
            # economic date and conservative public availability have arrived.
            if pending_rights.empty:
                current_rights = pending_rights.copy()
            else:
                ready = pending_rights.timestamp.le(timestamp)
                current_rights = pending_rights.loc[ready].copy()
                pending_rights = pending_rights.loc[~ready]
                recognition = current_rights[["asset_id", "listing_date"]].assign(
                    recognition_at=sessions.take(
                        sessions.searchsorted(current_rights.timestamp)
                    )
                )
                if recognition.duplicated(keep=False).any():
                    raise ValueError("equity_event_session_order_unproved")
                current_rights = _qualify_equity_events(
                    current_rights, recognition.recognition_at, stocks, rights_lineage
                )
            # Retain record-date bookkeeping even for later disclosures. The
            # accounting owner cannot disclose/apply it until available_at.
            # Closed-day settlement uses the next unified account session.
            current_corporate = corporate.loc[
                starts.reindex(corporate.index).le(timestamp)
                & (
                    last_session.reindex(corporate.index).ge(timestamp)
                    | corporate.settlement_at.isna()
                    | last_session.reindex(corporate.index).isna()
                )
            ].copy()
            # An explicit ordinary-holder transition supersedes an unresolved
            # aggregate source row; it does not create a second distribution.
            proved = current_rights.loc[current_rights.entitlement_proved]
            for event in proved.itertuples(index=False):
                if event.kind not in {"share_change", "conversion", "cancel"}:
                    continue
                same = (
                    current_corporate.asset_id.eq(event.asset_id)
                    & current_corporate.listing_date.eq(event.listing_date)
                    & current_corporate.effective_at.eq(event.effective_at)
                    & ~current_corporate.entitlement_proved
                )
                corporate = corporate.loc[
                    ~corporate.index.isin(current_corporate.index[same])
                ]
                current_corporate = current_corporate.loc[~same]
            result = _PortfolioSession(
                timestamp,
                raw,
                valuation,
                closed,
                masks.reset_index(drop=True).copy(),
                unproved_lifecycle,
                confirmed_delisted_lifecycles,
                frozenset(unknown_membership)
                | frozenset(status.index[status.isna()].astype(str)),
                current_rights,
                current_corporate,
            )
            try:
                yield result
            except GeneratorExit:
                return


def _qualify_equity_events(events, recognition, stocks, lineage):
    """Qualify lifecycle-scoped recovery and explicit holder share transitions."""
    ordered = []
    events = events.copy()
    events["entitlement_proved"] = True
    for recognized, positions in recognition.groupby(
        recognition, sort=True
    ).groups.items():
        facts = (
            stocks.loc[stocks.stock_effective_date.le(recognized)]
            .drop_duplicates(["asset_id", "listing_date"], keep="last")
            .set_index(["asset_id", "listing_date"])
        )
        pending = list(events.loc[positions].itertuples(index=False))
        while pending:
            unresolved = []
            for event in pending:
                identity = (event.asset_id, event.listing_date)
                terminated = False
                active = False
                if identity in facts.index:
                    fact = facts.loc[identity]
                    terminated = (
                        fact.delisting_date <= recognized
                        and event.effective_at >= fact.delisting_date
                    )
                    active = identity[1] <= recognized < fact.delisting_date
                linked = (
                    identity in lineage
                    and event.effective_at >= lineage[identity][0]
                    and not active
                )
                holder_transition = event.kind == "share_change" and active
                if not (terminated or linked or holder_transition):
                    unresolved.append(event)
                    continue
                ordered.append(event.event_id)
                if event.kind == "conversion":
                    successor = (event.successor_asset_id, event.successor_listing_date)
                    ancestors = lineage.get(identity, (None, frozenset()))[1] | {
                        identity
                    }
                    if successor in ancestors:
                        raise ValueError("equity_event_successor_lineage_cycle")
                    prior = lineage.get(successor, (event.effective_at, frozenset()))
                    lineage[successor] = (
                        min(prior[0], event.effective_at),
                        prior[1] | ancestors,
                    )
            if len(unresolved) == len(pending):
                for event in unresolved:
                    ordered.append(event.event_id)
                    events.loc[
                        events.event_id.eq(event.event_id), "entitlement_proved"
                    ] = False
                break
            pending = unresolved
    return events.set_index("event_id", drop=False).loc[ordered].reset_index(drop=True)


def _corporate_action_events(source, stocks, calendar):
    """Project source facts once; ordinary registration follows the own-market rule."""
    if source.duplicated(["source_asset_id", "timestamp", "kind"]).any():
        raise ValueError("corporate_action_key_duplicate")
    rows = []
    calendars = {
        market: market_type_calendar_state(calendar, market)
        for market in CURRENT_MARKET_TYPES
    }
    for event in source.itertuples(index=False):
        if event.kind in {"RO", "SN"}:
            continue
        effective = pd.Timestamp(event.timestamp)
        state = calendars.get(event.market_type)
        record = pd.NaT
        calendar_proved = False
        if state is not None and bool(state.get(effective, False)):
            axis = state.index[state.fillna(False)].sort_values()
            position = axis.searchsorted(effective)
            if position:
                record = axis[position - 1]
                # Absent calendar days cannot be silently treated as closures.
                between = state.reindex(pd.date_range(record, effective))
                calendar_proved = bool(between.notna().all())
        published = event.announcement_date
        available = (
            pd.Timestamp(published) + pd.Timedelta(days=1)
            if pd.notna(published)
            else effective
        )
        settlement = event.settlement_date
        amount = event.amount
        finite = pd.notna(amount) and np.isfinite(amount)
        kind = {"CA": "cash_dividend", "SD": "stock_distribution"}.get(
            event.kind, "unproved"
        )
        proved = calendar_proved and pd.notna(published) and finite and amount >= 0
        cash, shares = None, None
        if event.kind == "CA":
            cash = Fraction(amount) if finite else None
            if pd.isna(settlement):
                settlement = effective
            proved &= settlement >= effective
        elif event.kind == "SD":
            shares = Fraction(amount) if finite and amount >= 0 else None
            # Sentinel/pre-ex dates do not prove a tradable-share release.
            if pd.notna(settlement) and settlement < effective:
                settlement = pd.NaT
        else:
            proved = False
            settlement = effective
        identity = (
            f"csmar:corporate_actions:{event.source_asset_id}:"
            f"{event.kind}:{effective.date().isoformat()}"
        )
        rows.append(
            {
                "event_id": identity,
                "asset_id": str(event.source_asset_id),
                "listing_date": pd.NaT,
                "kind": kind,
                "record_at": record,
                "effective_at": effective,
                "settlement_at": settlement,
                "available_at": available,
                "cash_per_share": cash,
                "additional_shares_per_share": shares,
                "evidence_identity": canonical_json_hash(
                    {
                        key: None if pd.isna(value) else str(value)
                        for key, value in event._asdict().items()
                    }
                ),
                "entitlement_proved": bool(proved),
            }
        )
    result = pd.DataFrame(rows, columns=_CORPORATE_EVENT_COLUMNS)
    for column in (
        "listing_date",
        "record_at",
        "effective_at",
        "settlement_at",
        "available_at",
    ):
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    result["entitlement_proved"] = result.entitlement_proved.astype(bool)
    if not result.empty:
        result["asset_id"] = result.asset_id.astype(stocks.asset_id.dtype)
        facts = panels._asof_join_stock_master(
            result[["asset_id"]].assign(
                event_position=result.index,
                timestamp=result.record_at.fillna(result.effective_at),
            ),
            stocks,
        ).sort_values("event_position")
        result["listing_date"] = facts.listing_date.to_numpy()
        result["entitlement_proved"] &= (
            result.listing_date.notna()
            & result.record_at.ge(result.listing_date)
            & result.effective_at.le(facts.delisting_date.to_numpy())
        )
    result.attrs["corporate_action_policy"] = dict(CORPORATE_ACTION_POLICY)
    return result
