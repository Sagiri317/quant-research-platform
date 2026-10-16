"""A-share portfolio-only session and lifecycle evidence projections."""

from __future__ import annotations

from bisect import bisect_right
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import (
    _EXECUTION_COLUMNS,
    EQUITY_EVENT_COLUMNS,
    _portfolio_session_frame,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import decode_exact_numeric
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
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


@dataclass(frozen=True, slots=True)
class _PortfolioSession:
    timestamp: pd.Timestamp
    panel: pd.DataFrame
    valuation: pd.DataFrame
    closed_assets: frozenset[str]
    execution: pd.DataFrame
    unproved_lifecycle_assets: frozenset[str]
    confirmed_delisted_lifecycles: frozenset[tuple[str, pd.Timestamp]]
    rights_events: pd.DataFrame
    unit_unproved_assets: frozenset[str] = frozenset()


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
):
    """Project proved closed-market marks without changing materialized inputs."""
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
        "shares_per_old_unit",
        "successor_shares_per_unit",
    ):
        rights[column] = rights[column].map(
            lambda value: None if pd.isna(value) else decode_exact_numeric(value)
        )
    if rights.timestamp.isna().any():
        raise ValueError("equity_event_timestamp_invalid")
    rights = rights.loc[rights.timestamp.le(pd.Timestamp(time_window.end))].copy()
    rights = (
        _prove_equity_units(rights, panel)
        if not rights.empty
        else rights.assign(unit_basis_proved=True)
    )
    pending_rights = rights
    rights_lineage = {}
    returning_units = {}
    unit_columns = (
        tuple(name for name in ("open", "close") if name in panel.arrow_schema.names)
        if not rights.empty
        else ()
    )
    panel_columns = tuple(dict.fromkeys((*panel_columns, *unit_columns)))
    sessions = unified_trading_sessions(calendar)
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
                prices = indexed.loc[:, ["adj_open", "adj_close"]]
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
                carried["adj_open"] = carried["adj_close"]
                carried.loc[carried.index.isin(blocked), ["adj_open", "adj_close"]] = (
                    np.nan
                )
            valuation = raw.copy()
            # A missing side does not invalidate the other observed trading phase.
            valuation.loc[
                valuation.asset_id.isin(blocked - current_trading_assets),
                ["adj_open", "adj_close"],
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
            for event in current_rights.itertuples(index=False):
                if (
                    event.kind == "conversion"
                    and event.unit_basis_proved
                    and event.entitlement_proved
                ):
                    returning_units[
                        (event.successor_asset_id, event.successor_listing_date)
                    ] = Fraction(event.successor_shares_per_unit)
            unit_unproved = set()
            for row in raw.itertuples(index=False):
                key = (row.asset_id, row.listing_date)
                if key not in returning_units:
                    continue
                opening = getattr(row, "open", np.nan)
                if (
                    pd.isna(opening)
                    or pd.isna(row.adj_open)
                    or opening <= 0
                    or row.adj_open <= 0
                ):
                    unit_unproved.add(str(row.asset_id))
                elif Fraction(row.adj_open) / Fraction(opening) != returning_units[key]:
                    unit_unproved.add(str(row.asset_id))
                else:
                    del returning_units[key]
            result = _PortfolioSession(
                timestamp,
                raw,
                valuation,
                closed,
                masks.reset_index(drop=True).copy(),
                unproved_lifecycle,
                confirmed_delisted_lifecycles,
                current_rights,
                frozenset(unit_unproved),
            )
            try:
                yield result
            except GeneratorExit:
                return


def _prove_equity_units(rights, panel):
    """Bind old units to actual quotes or previously proved off-market rights."""
    rights = rights.copy()
    rights["unit_basis_proved"] = False
    anchors, declared = {}, {}

    def panel_basis(asset, listing, at, recognized):
        raw_column, adjusted_column = (
            ("open", "adj_open") if at == recognized else ("close", "adj_close")
        )
        if raw_column not in panel.arrow_schema.names:
            return None
        key = (asset, listing, at, raw_column)
        if key not in anchors:
            frame = materialize_dataset_session(
                panel.filtered_view(TimeWindow(at, at)).projected_view(
                    (
                        "asset_id",
                        "timestamp",
                        "listing_date",
                        raw_column,
                        adjusted_column,
                    )
                )
            )
            frame = frame.loc[frame.asset_id.eq(asset) & frame.listing_date.eq(listing)]
            value = None
            if len(frame) == 1 and frame[[raw_column, adjusted_column]].notna().all(
                axis=None
            ):
                raw, adjusted = frame.iloc[0][[raw_column, adjusted_column]]
                if raw > 0 and adjusted > 0 and np.isfinite([raw, adjusted]).all():
                    value = Fraction(adjusted) / Fraction(raw)
            anchors[key] = value
        return anchors[key]

    for _, group in rights.groupby("timestamp", sort=True):
        pending = list(group.itertuples(index=False))
        while pending:
            unresolved = []
            for event in pending:
                identity = (event.asset_id, event.listing_date)
                at = event.unit_basis_date
                if pd.isna(at) or not event.listing_date <= at <= event.timestamp:
                    continue
                actual = panel_basis(*identity, at, event.timestamp)
                basis = actual if actual is not None else declared.get(identity)
                if basis is None:
                    unresolved.append(event)
                    continue
                if basis != Fraction(event.shares_per_old_unit):
                    continue
                if identity in declared and declared[identity] != basis:
                    continue
                if event.kind == "conversion":
                    successor = (event.successor_asset_id, event.successor_listing_date)
                    successor_at = event.successor_unit_basis_date
                    if (
                        pd.isna(successor_at)
                        or not successor[1] <= successor_at <= event.timestamp
                    ):
                        continue
                    successor_basis = Fraction(event.successor_shares_per_unit)
                    if successor_basis <= 0:
                        continue
                    actual_successor = panel_basis(
                        *successor, successor_at, event.timestamp
                    )
                    if (
                        actual_successor is not None
                        and actual_successor != successor_basis
                    ):
                        continue
                    if successor in declared and declared[successor] != successor_basis:
                        continue
                    # The accepted official source defines a new off-market unit.
                    # It cannot be joined to a future tradable quote until checked.
                    declared[successor] = successor_basis
                rights.loc[rights.event_id.eq(event.event_id), "unit_basis_proved"] = (
                    True
                )
            if len(unresolved) == len(pending):
                break
            pending = unresolved
    return rights


def _qualify_equity_events(events, recognition, stocks, lineage):
    """Limit reference cash and rights to terminated equities and proved successors."""
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
                if not (terminated or linked):
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
