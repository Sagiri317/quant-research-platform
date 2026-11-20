"""Continuous A-share inventory accounts on observed subsequent quotes."""

from collections import deque
from dataclasses import replace
from datetime import time
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    canonical_json_hash,
)
from quant_research_platform.contracts.policies import (
    PolicyDecision,
    PolicyExecution,
    policy_target,
)
from quant_research_platform.contracts.portfolios import (
    A_SHARE_INSTRUMENT_TYPES,
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutedTrade,
    ExecutionAssumptions,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    prepare_trade_cost,
    round_stock_orders,
    stock_order_limits,
)
from quant_research_platform.markets.implementations.a_share.portfolio_sessions import (
    CORPORATE_ACTION_POLICY,
    iter_portfolio_sessions,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.accounting import (
    EquityEntitlements,
    ResearchValuationMarks,
    execute_stock_target,
    quoted_execution_depth,
)
from quant_research_platform.research.portfolios.inventory_analysis import (
    inventory_daily_economics,
)
from quant_research_platform.research.portfolios.passive import (
    PassiveOrders,
    passive_parameters,
    trade_events,
)
from quant_research_platform.research.portfolios.performance import performance_rows
from quant_research_platform.research.predictions.outputs import (
    read_rolling_predictions,
)
from quant_research_platform.research.signals.processing import (
    process_prediction_signals,
    signal_processing_parameters,
)
from quant_research_platform.storage.datasets.inspection import inspect_dataset_session
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
    read_dataset,
)


def _parameters(value, *, policy=False):
    params = _strict_config(
        value,
        {
            "initial_capital",
            "maximum_weights",
            "buy_threshold",
            "sell_threshold",
            "participation_rate",
            "minimum_commission",
            "restore_at",
        }
        - ({"buy_threshold", "sell_threshold"} if policy else set()),
        {
            "initial_inventory",
            "initial_weights",
            "passive",
            "taker_depth_levels",
            "markout_ms",
            "latency_ms",
        }
        | (set() if policy else {"signal_processing"}),
    )
    params.setdefault("latency_ms", 10.0)
    if ("initial_inventory" in params) == ("initial_weights" in params):
        raise ValueError("hft_initial_holdings_require_quantities_or_weights")
    initial = params.get("initial_inventory", {})
    if not isinstance(initial, dict) or any(
        type(asset) is not str
        or not asset
        or asset == PORTFOLIO_CASH_ASSET_ID
        or type(quantity) is not int
        or quantity < 0
        for asset, quantity in initial.items()
    ):
        raise ValueError("hft_inventory_invalid")
    maximum = params["maximum_weights"]
    if (
        not isinstance(maximum, dict)
        or not maximum
        or any(
            type(asset) is not str or not asset or asset == PORTFOLIO_CASH_ASSET_ID
            for asset in maximum
        )
    ):
        raise ValueError("hft_maximum_weights_invalid")
    maximum = {
        asset: strict_finite_float(weight, error="hft_maximum_weights_invalid")
        for asset, weight in maximum.items()
    }
    if any(not 0 <= weight <= 1 for weight in maximum.values()):
        raise ValueError("hft_maximum_weights_invalid")
    if not initial.keys() <= maximum.keys():
        raise ValueError("hft_inventory_invalid")
    params["maximum_weights"] = maximum
    if "initial_weights" in params:
        weights = params["initial_weights"]
        if not isinstance(weights, dict) or not weights.keys() <= maximum.keys():
            raise ValueError("hft_initial_weights_invalid")
        weights = {
            asset: strict_finite_float(weight, error="hft_initial_weights_invalid")
            for asset, weight in weights.items()
        }
        if any(weight < 0 for weight in weights.values()) or fsum(weights.values()) > 1:
            raise ValueError("hft_initial_weights_invalid")
        params["initial_weights"] = weights
    if "signal_processing" in params:
        params["signal_processing"] = signal_processing_parameters(
            params["signal_processing"]
        )
    if "passive" in params:
        params["passive"] = passive_parameters(params["passive"])
    depth_levels = params.get("taker_depth_levels", 1)
    if type(depth_levels) is not int or not 1 <= depth_levels <= 10:
        raise ValueError("hft_taker_depth_levels_invalid")
    for name in set(params) - {
        "initial_inventory",
        "initial_weights",
        "maximum_weights",
        "restore_at",
        "signal_processing",
        "passive",
        "taker_depth_levels",
    }:
        params[name] = strict_finite_float(params[name], error="hft_parameter_invalid")
    if (
        params["initial_capital"] <= 0
        or (not policy and params["buy_threshold"] <= params["sell_threshold"])
        or params["latency_ms"] < 0
        or not 0 < params["participation_rate"] <= 1
        or params["minimum_commission"] < 0
        or params.get("markout_ms", 1.0) <= 0
    ):
        raise ValueError("hft_parameter_invalid")
    restore = params["restore_at"]
    if restore is not None:
        if not isinstance(restore, str):
            raise ValueError("hft_restore_time_invalid")
        clock = time.fromisoformat(restore)
        if clock.tzinfo is not None or not time(13) <= clock < time(15):
            raise ValueError("hft_restore_time_invalid")
        params["restore_at"] = clock.isoformat()
    return params


def _stock_status_events(state, assets, start, end, *, closed):
    """Project opening holdings and published changes without a quote-grid product."""
    from quant_research_platform.markets.implementations.a_share.stock_pool import (
        stock_status_rows,
    )

    if not len(assets):
        return {}
    source = state["status"]
    selected = source.loc[
        source.asset_id.isin(assets)
        & source._known_at.gt(start)
        & (source._known_at.le(end) if closed else source._known_at.lt(end)),
        ["asset_id", "_known_at"],
    ].rename(columns={"_known_at": "timestamp"})
    points = pd.concat(
        (pd.DataFrame({"asset_id": assets, "timestamp": start}), selected),
        ignore_index=True,
    ).drop_duplicates(["asset_id", "timestamp"])
    rows = stock_status_rows(points, **state)
    return {
        int(at.value): frame.set_index("asset_id")
        for at, frame in rows.groupby("timestamp", sort=False)
    }


def inventory_batches(
    sessions,
    schedule,
    params,
    window,
    *,
    decision_columns=None,
    prediction_decisions=False,
    prediction_task="inventory",
    quote_visibility_delay_ms=0,
    stock_status=None,
    execution_order=None,
    trade_session=None,
    max_quote_gap_seconds=None,
):
    """Yield quote-account transitions with shared cash and next-session sellability.

    Interactive callers receive PolicyDecision events after current-quote accounting
    and send share targets for strictly subsequent eligible quotes. Execution latency,
    lot sizes, liquidity, fees, and T+1 constraints remain active."""
    if "passive" in params and max_quote_gap_seconds is None:
        raise ValueError("passive_quote_gap_required")
    dynamic_execution = prediction_decisions and prediction_task == "dynamic_execution"
    if dynamic_execution and "passive" not in params:
        raise ValueError("dynamic_execution_order_input_required")
    status_state = None
    if stock_status is not None:
        from quant_research_platform.markets.implementations.a_share.stock_pool import (
            load_stock_status,
        )

        status_state = load_stock_status(stock_status)
    lower, upper, closed = time_window_timestamp_interval(window)
    lower, upper = pd.Timestamp(lower), pd.Timestamp(upper)
    assets = pd.Index(sorted(params["maximum_weights"]), name="asset_id")
    maximum_weights = pd.Series(params["maximum_weights"], dtype="float64")
    units = pd.Series(params.get("initial_inventory", {}), dtype="float64").reindex(
        assets, fill_value=0.0
    )
    instrument_types = pd.Series("stock", index=assets)
    lives = pd.Series(dtype="datetime64[ns]")
    marks = pd.Series(dtype="float64")
    sources = {}
    cash = None
    previous = {}
    sequence = 0
    observed_quotes = 0
    pending_markouts = {}
    markout_ns = (
        max(1, int(params["markout_ms"] * 1_000_000))
        if "markout_ms" in params
        else None
    )
    entitlements = EquityEntitlements()
    valuation_marks = ResearchValuationMarks()
    latency_ns = int(params["latency_ms"] * 1_000_000)
    remaining_order = None if execution_order is None else execution_order["quantity"]
    assumptions = ExecutionAssumptions(
        params["initial_capital"], 0.0, 0.0, 1.0, (1.0,), params["minimum_commission"]
    )

    def cash_balance():
        return cash

    def account_nav(prices):
        if len(prices) < len(units):
            # The policy book omits nontradable shares received in a conversion.
            prices = pd.concat((prices, marks.drop(prices.index)))
        quantities = units.reindex(prices.index)
        return (
            cash
            + fsum((quantities * prices).where(quantities.ne(0), 0.0))
            + fsum(entitlements.receivables.values())
        )

    def maximum_quantities(prices, nav=None):
        return (
            maximum_weights
            * max(0.0, account_nav(prices) if nav is None else nav)
            / prices.where(prices.gt(0))
        )

    def values(at):
        rows = {}
        for asset in units.index:
            quantity = float(units[asset])
            mark = marks.get(asset, np.nan)
            if quantity and (
                not np.isfinite(mark) or mark < 0 or pd.isna(lives.get(asset))
            ):
                raise ValueError("hft_held_valuation_missing")
            if not quantity and pd.isna(mark):
                mark = 0.0
            receivable = entitlements.receivables.get(asset, 0.0)
            listing = lives.get(asset)
            rows[asset] = PortfolioValuation(
                int(at.value),
                asset,
                "end",
                quantity,
                quantity * mark + receivable,
                float(mark),
                lifecycle_id=None
                if pd.isna(listing)
                else pd.Timestamp(listing).date().isoformat(),
                receivable=receivable,
                locked_quantity=entitlements.locked_quantities.get(asset, 0.0),
            )
        rows[PORTFOLIO_CASH_ASSET_ID] = PortfolioValuation(
            int(at.value), PORTFOLIO_CASH_ASSET_ID, "end", cash, cash
        )
        return rows

    def emit(at, *, trades=(), flows=(), adjustments=(), diagnostics=()):
        nonlocal previous, sequence
        current = values(at)
        axis = sorted(previous.keys() | current.keys())
        before, after, end = [], [], []
        for asset in axis:
            prior, present = previous.get(asset), current.get(asset)
            if present is None:
                present = replace(
                    prior,
                    timestamp_ns=int(at.value),
                    quantity=0.0,
                    value=0.0,
                    receivable=0.0,
                    locked_quantity=0.0,
                )
            if prior is None:
                prior = replace(
                    present,
                    quantity=0.0,
                    value=0.0,
                    receivable=0.0,
                    locked_quantity=0.0,
                )
            before.append(replace(prior, phase="before"))
            after.append(replace(present, phase="after"))
            end.append(present)
        batch = PortfolioLedgerBatch(
            sequence,
            tuple(trades),
            tuple(flows),
            (*before, *after, *end),
            tuple(diagnostics),
            tuple(adjustments),
        )
        previous = {row.asset_id: row for row in end}
        sequence += 1
        return batch

    def equity_at(records, at):
        adjustments = tuple(
            replace(
                row,
                timestamp_ns=int(at.value),
                phase="after",
                entitlement_record_at_ns=None
                if row.entitlement_record_at_ns is None
                else int(
                    (
                        pd.Timestamp(row.entitlement_record_at_ns).normalize()
                        + pd.Timedelta(hours=7)
                    ).value
                ),
            )
            for row in records
        )
        flows = tuple(
            PortfolioCashFlow(
                ("dividend:" if row.kind == "dividend" else "recovery:") + row.event_id,
                int(at.value),
                row.asset_id,
                None,
                "dividend" if row.kind == "dividend" else "recovery",
                row.cash_amount,
            )
            for row in adjustments
            if row.cash_amount
        )
        return adjustments, flows

    def settle(at, asset, order_id, signal_at, quantity, notional, costs):
        nonlocal cash, remaining_order
        fill_id = f"{sequence}:{asset}"
        cash = fsum((cash, -notional, *(-costs)))
        units[asset] += quantity
        if quantity > 0 and instrument_types[asset] != "gold_etf":
            bought_today[asset] += quantity
        if execution_order is not None:
            remaining_order -= quantity
        if markout_ns is not None:
            pending_markouts.setdefault(asset, deque()).append(
                (
                    fill_id,
                    notional / quantity,
                    np.sign(quantity),
                    int(at.value) + markout_ns,
                    segments.get(asset),
                )
            )
        trade = ExecutedTrade(
            fill_id, order_id, int(at.value), asset, quantity, notional
        )
        flows = [
            PortfolioCashFlow(
                f"{fill_id}:principal",
                int(at.value),
                asset,
                fill_id,
                "principal",
                -notional,
            )
        ]
        flows.extend(
            PortfolioCashFlow(
                f"{fill_id}:{name}", int(at.value), asset, fill_id, name, -float(value)
            )
            for name, value in zip(PORTFOLIO_LEDGER_COST_COLUMNS, costs, strict=True)
            if value
        )
        diagnostic = PortfolioDiagnostic(
            asset, fill_id, "decision_timestamp_ns", text=str(signal_at)
        )
        return trade, flows, diagnostic

    for view, quotes, predictions in sessions:
        day = pd.Timestamp(view.timestamp).normalize()
        opening, closing = (
            day + pd.Timedelta(hours=1, minutes=30),
            day + pd.Timedelta(hours=7),
        )
        projected = valuation_marks.project(view).set_index("asset_id")
        if closing < lower:
            continue
        if opening > upper or (not closed and opening == upper):
            break
        start = max(opening, lower)
        quotes = quotes.loc[quotes.asset_id.isin(assets)].sort_values(
            ["timestamp", "asset_id"]
        )
        if quotes.duplicated(["timestamp", "asset_id"]).any():
            raise ValueError("hft_quote_key_duplicate")
        if "instrument_type" in quotes:
            kinds = quotes[["asset_id", "instrument_type"]].drop_duplicates()
            if (
                kinds.asset_id.duplicated().any()
                or not kinds.instrument_type.isin(A_SHARE_INSTRUMENT_TYPES).all()
            ):
                raise ValueError("hft_instrument_type_invalid")
            instrument_types.update(kinds.set_index("asset_id").instrument_type)
        if prediction_decisions and not predictions.empty and not quotes.empty:
            permitted = (
                {"stock"}
                if prediction_task == "portfolio"
                else {"equity_etf", "gold_etf"}
            )
            if not instrument_types.loc[quotes.asset_id.unique()].isin(permitted).all():
                raise ValueError("policy_prediction_instrument_mismatch")
        opening_marks = projected.open.reindex(units.index).astype("float64")
        observed = quotes.loc[
            quotes.timestamp.le(start) & np.isfinite(quotes.mid) & quotes.mid.gt(0)
        ].drop_duplicates("asset_id", keep="last")
        opening_marks.loc[observed.asset_id] = observed.mid.to_numpy()
        if cash is None:
            lives = projected.listing_date.reindex(assets)
            marks = opening_marks
            if "initial_weights" in params:
                weights = pd.Series(params["initial_weights"], dtype="float64")
                selected = weights.index[weights.gt(0)]
                prices = marks.reindex(selected)
                if not (np.isfinite(prices) & prices.gt(0)).all():
                    raise ValueError("hft_initial_weight_price_missing")
                _, minimum, increment, _ = stock_order_limits(
                    projected.market_type.reindex(selected),
                    day.date(),
                    instrument_types,
                )
                units.loc[selected] = round_stock_orders(
                    params["initial_capital"]
                    * weights.loc[selected].to_numpy()
                    / prices.to_numpy(),
                    minimum,
                    increment,
                )
            if units.gt(maximum_quantities(marks, params["initial_capital"])).any():
                raise ValueError("hft_initial_inventory_exceeds_maximum")
            initial_inventory = units.copy()
            for record_at, events in view.corporate_events.groupby("record_at"):
                if record_at < day:
                    if events.asset_id.isin(units.index[units.ne(0)]).any():
                        raise ValueError("hft_initial_inventory_registration_missing")
                    entitlements.register(
                        events,
                        pd.Series(dtype="float64"),
                        lives,
                        pd.Timestamp(record_at),
                    )
            # Opening inventory and capital are supplied before this session's actions.
            cash = params["initial_capital"] - fsum(
                float(units[asset]) * float(marks[asset])
                for asset in assets
                if units[asset]
            )
            if not np.isfinite(cash) or cash < 0:
                raise ValueError("hft_initial_inventory_exceeds_capital")
            previous = values(start)
        units, lives, paid, rights = entitlements.process(
            view.corporate_events, view.rights_events, units, lives, day
        )
        cash += paid
        units = units.reindex(units.index.union(assets, sort=False), fill_value=0.0)
        active = units.ne(0)
        if (
            not projected.listing_date.reindex(units.index[active])
            .eq(lives.reindex(units.index[active]))
            .all()
        ):
            raise ValueError("hft_held_lifecycle_unproved")
        lives.loc[~active] = projected.listing_date.reindex(units.index[~active])
        marks = opening_marks.reindex(units.index)
        successors = units.index.difference(opening_marks.index)
        marks.loc[successors] = projected.open.reindex(successors)
        sources = {
            asset: "daily_open"
            if pd.isna(projected.research_open_mark_at.get(asset))
            else str(projected.research_open_mark_kind.get(asset))
            for asset in units.index
        }
        opening_inventory = units.copy()
        reference_inventory = (
            initial_inventory
            if prediction_decisions
            and prediction_task in {"inventory", "dynamic_execution"}
            else opening_inventory
        )
        bought_today = pd.Series(0.0, index=units.index)
        adjustments, flows = equity_at(rights, start)
        yield emit(
            start,
            flows=flows,
            adjustments=adjustments,
            diagnostics=(
                PortfolioDiagnostic(
                    PORTFOLIO_CASH_ASSET_ID, None, "session_phase", text="open"
                ),
            ),
        )
        if (decision_columns is None or prediction_decisions) and not predictions.empty:
            if "signal_processing" in params:
                predictions = process_prediction_signals(
                    predictions,
                    quotes,
                    params["signal_processing"],
                    quote_visibility_delay_ms=quote_visibility_delay_ms,
                )
            predictions = predictions.loc[
                predictions.timestamp.ge(lower)
                & (
                    predictions.timestamp.le(upper)
                    if closed
                    else predictions.timestamp.lt(upper)
                )
            ]
        interactive = decision_columns is not None
        taker_intents = {asset: deque() for asset in assets}
        # A truncated execution window can start between two observed quotes.
        segments = (
            quotes.loc[quotes.timestamp.lt(start), ["asset_id", "segment_id"]]
            .drop_duplicates("asset_id", keep="last")
            .set_index("asset_id")
            .segment_id.to_dict()
        )
        if params["restore_at"] is None:
            restore_at = None
        else:
            restore_at = (
                day + pd.Timedelta(params["restore_at"]) - pd.Timedelta(hours=8)
            )
        endpoint = min(closing, upper)
        if not closed and endpoint == upper:
            endpoint -= pd.Timedelta(1, "ns")
        quotes = quotes.loc[quotes.timestamp.le(endpoint)]
        visible_books = (
            {asset: group for asset, group in quotes.groupby("asset_id", sort=False)}
            if prediction_decisions
            else {}
        )
        quotes = quotes.loc[quotes.timestamp.ge(start)]
        observed_quotes += len(quotes)
        passive = None
        event_iterator = iter(())
        next_trade = None
        if "passive" in params:
            if trade_session is None:
                raise ValueError("passive_trade_input_required")
            events = materialize_dataset_session(
                trade_session.filtered_view(TimeWindow(day.date(), day.date()))
            )
            events = events.loc[
                events.timestamp.ge(start) & events.timestamp.le(endpoint)
            ]
            event_iterator = trade_events(events)
            next_trade = next(event_iterator, None)
            passive = PassiveOrders(
                params["passive"],
                projected.market_type.reindex(
                    visible_books.keys()
                    if dynamic_execution
                    else quotes.loc[
                        quotes.bid_px1.gt(0) | quotes.ask_px1.gt(0), "asset_id"
                    ].unique()
                ),
                day.date(),
                schedule,
                assumptions,
                params["participation_rate"],
                max_quote_gap_seconds=max_quote_gap_seconds,
                depth_levels=params.get("taker_depth_levels", 1),
                face_value_per_share=projected.get("face_value_per_share"),
                instrument_types=instrument_types,
                cash_balance=cash_balance if dynamic_execution else None,
            )
            passive.segments.update(segments)
        requests = {}
        if (not interactive or prediction_decisions) and not predictions.empty:
            signals = predictions.loc[predictions.asset_id.isin(assets)]
            if not prediction_decisions:
                signals = signals.dropna(subset=["prediction_score"])
            if restore_at is not None:
                signals = signals.loc[signals.timestamp.lt(restore_at)]
            for row in signals.itertuples(index=False):
                requests.setdefault(row.timestamp.value, {})[row.asset_id] = (
                    row.prediction_score
                )
        if (
            passive is not None
            and restore_at is not None
            and start <= restore_at <= endpoint
        ):
            requests[restore_at.value] = dict.fromkeys(assets)
        status_events = (
            _stock_status_events(
                status_state,
                instrument_types.index[instrument_types.eq("stock")],
                start,
                min(closing, upper),
                closed=closed or closing < upper,
            )
            if status_state is not None
            else {}
        )
        for stamp in status_events:
            requests.setdefault(stamp, {})
        status_can_enter = pd.Series(True, index=units.index)
        status_exit_at = {}

        def status_target(
            asset,
            target,
            *,
            units=units,
            status_can_enter=status_can_enter,
            status_exit_at=status_exit_at,
        ):
            if asset in status_exit_at:
                return 0.0
            return (
                min(target, units[asset])
                if not status_can_enter.get(asset, True)
                else target
            )

        request_iterator = iter(sorted(requests.items()))
        next_request = next(request_iterator, None)
        policy_costs, policy_limits = {}, {}
        if prediction_decisions and requests:
            quoted_assets = visible_books.keys()
            policy_markets, policy_minimum, _, policy_maximum = stock_order_limits(
                projected.market_type.reindex(quoted_assets),
                day.date(),
                instrument_types,
            )
            policy_limits = dict(
                zip(
                    quoted_assets,
                    zip(policy_minimum, policy_maximum, strict=True),
                    strict=True,
                )
            )
            policy_costs = {
                (asset, side): prepare_trade_cost(
                    day.date(), market, side, schedule, assumptions
                )
                for asset, market in zip(quoted_assets, policy_markets, strict=True)
                for side in ("buy", "sell")
            }
        prediction_rows = (
            predictions.set_index(["timestamp", "asset_id"])
            if prediction_decisions and not predictions.empty
            else None
        )

        def request_targets(
            stamp,
            requests_at,
            *,
            reference_inventory=reference_inventory,
            units=units,
            passive=passive,
            taker_intents=taker_intents,
            status_can_enter=status_can_enter,
            status_exit_at=status_exit_at,
            status_target=status_target,
            marks=marks,
        ):
            targets, modes = {}, {}
            buy_only = set()
            limits = None
            for asset, score in requests_at.items():
                buy = score is not None and score > params["buy_threshold"]
                if buy or score is None:
                    if limits is None:
                        limits = maximum_quantities(marks)
                    target = max(units[asset], limits[asset])
                    if score is None:
                        target = min(reference_inventory[asset], target)
                else:
                    target = 0.0 if score < params["sell_threshold"] else None
                if passive is None:
                    if asset not in status_exit_at and (
                        status_can_enter[asset] or not buy
                    ):
                        taker_intents[asset].append((stamp, target, buy))
                else:
                    targets[asset] = status_target(
                        asset, units[asset] if target is None else target
                    )
                    if buy and asset not in status_exit_at:
                        buy_only.add(asset)
                    elif score is None:
                        modes[asset] = "taker"
                    elif target is None and asset not in status_exit_at:
                        modes[asset] = "cancel"
            if passive is None:
                return []
            diagnostics = passive.request(
                pd.Timestamp(stamp),
                targets,
                dict.fromkeys(targets, stamp + latency_ns),
                modes=modes,
                segment_bound=any(score is not None for score in requests_at.values()),
                buy_only=buy_only,
            )
            diagnostics.extend(
                PortfolioDiagnostic(
                    asset,
                    None,
                    "inventory_target",
                    number=float(status_target(asset, reference_inventory[asset])),
                )
                for asset, score in requests_at.items()
                if score is None
            )
            return diagnostics

        def apply_request(
            stamp,
            requested,
            cash,
            *,
            restore_at=restore_at,
            visible_books=visible_books,
            policy_costs=policy_costs,
            policy_limits=policy_limits,
            projected=projected,
            prediction_rows=prediction_rows,
            passive=passive,
            taker_intents=taker_intents,
            segments=segments,
            units=units,
            bought_today=bought_today,
            opening_inventory=opening_inventory,
            reference_inventory=reference_inventory,
            request_targets=request_targets,
            status_events=status_events,
            status_can_enter=status_can_enter,
            status_exit_at=status_exit_at,
            status_target=status_target,
        ):
            at = pd.Timestamp(stamp)
            if passive is not None:
                cancellations = passive.advance(at)
                if cancellations:
                    yield emit(at, diagnostics=(*cancellations, *passive.diagnostics()))
            status_diagnostics = []
            for asset, state in (
                status_events[stamp].iterrows() if stamp in status_events else ()
            ):
                known = state[["is_st", "is_pt", "is_delisting"]].notna().all()
                exits = any(
                    pd.notna(value) and bool(value)
                    for value in (state.is_st, state.is_delisting)
                )
                can_enter = bool(
                    known
                    and not state.is_st
                    and not state.is_pt
                    and not state.is_delisting
                )
                status_can_enter[asset] = can_enter
                if exits:
                    status_exit_at.setdefault(asset, stamp)
                else:
                    status_exit_at.pop(asset, None)
                reason = (
                    "delisting_exit"
                    if pd.notna(state.is_delisting) and state.is_delisting
                    else "special_treatment_exit"
                    if pd.notna(state.is_st) and state.is_st
                    else "status_unknown"
                    if not known
                    else "pt_entry_blocked"
                    if state.is_pt
                    else "entry_allowed"
                )
                status_diagnostics.append(
                    PortfolioDiagnostic(
                        asset, None, "stock_status_control", text=reason
                    )
                )
                if can_enter:
                    continue
                taker_intents[asset] = deque(
                    (signal_at, target, buy_only)
                    for signal_at, target, buy_only in taker_intents[asset]
                    if not exits
                    and not buy_only
                    and (target is None or target <= units[asset])
                )
                if passive is not None:
                    intent = passive.intents.get(asset)
                    order = passive.orders.get(asset)
                    target = (
                        0.0
                        if exits
                        else min(
                            units[asset],
                            intent[0]
                            if intent is not None
                            else order.target
                            if order is not None
                            else units[asset],
                        )
                    )
                    status_diagnostics.extend(
                        passive.request(
                            at,
                            {asset: target},
                            {asset: stamp + latency_ns},
                            segment_bound=False,
                        )
                    )
            if status_diagnostics:
                yield emit(at, diagnostics=status_diagnostics)
            if not dynamic_execution:
                requested = {
                    asset: score
                    for asset, score in requested.items()
                    if asset in segments or score is None
                }
            if not requested:
                return
            if not prediction_decisions or (
                restore_at is not None and at >= restore_at
            ):
                diagnostics = request_targets(stamp, requested)
                if diagnostics:
                    yield emit(at, diagnostics=diagnostics)
                return
            visible_at = at - pd.Timedelta(milliseconds=quote_visibility_delay_ms)
            visible_marks = pd.Series(np.nan, index=assets)
            estimated_costs = pd.Series(np.nan, index=assets)
            decision_prices = {}
            visible_quotes = {}
            visible_features = pd.DataFrame(
                np.nan, index=assets, columns=decision_columns
            )
            for asset, book in visible_books.items():
                position = book.timestamp.searchsorted(visible_at, side="right") - 1
                if position >= 0:
                    row = book.iloc[position]
                    visible_quotes[asset] = row
                    visible_marks[asset] = row.mid
                    for column in decision_columns:
                        if column not in {"prediction_score", "prediction_error_scale"}:
                            visible_features.loc[asset, column] = row[column]
                    if not np.isfinite(row.mid) or row.mid <= 0:
                        continue
                    costs = []
                    for side in ("buy", "sell"):
                        book_side = (
                            ("bid" if side == "buy" else "ask")
                            if passive is not None and not dynamic_execution
                            else ("ask" if side == "buy" else "bid")
                        )
                        level = (
                            params["passive"].get("quote_level", 1)
                            if passive is not None and not dynamic_execution
                            else 1
                        )
                        price = row.get(f"{book_side}_px{level}", np.nan)
                        if not np.isfinite(price) or price <= 0:
                            costs.append(np.nan)
                            continue
                        decision_prices[asset, side] = price
                        fee = fsum(
                            policy_costs[asset, side].order_components(
                                price,
                                0.0,
                                price=price,
                                face_value_per_share=projected.get(
                                    "face_value_per_share", pd.Series(dtype=float)
                                ).get(asset),
                            )
                        )
                        spread = max(
                            0, (price - row.mid) * (1 if side == "buy" else -1)
                        )
                        costs.append((fee + spread) / row.mid * 10000)
                    estimated_costs[asset] = (
                        max(costs) if np.isfinite(costs).all() else np.nan
                    )

            def estimate_trade_cost(asset, quantity):
                if quantity == 0:
                    return 0.0
                side = "buy" if quantity > 0 else "sell"
                price = decision_prices.get((asset, side), np.nan)
                if not np.isfinite(price):
                    return np.inf
                count = abs(quantity)
                minimum, maximum = policy_limits[asset]
                fee = fsum(
                    policy_costs[asset, side].stock_components(
                        np.asarray(count),
                        np.asarray(price),
                        np.asarray(np.nan),
                        np.asarray(minimum),
                        np.asarray(maximum),
                        params["minimum_commission"],
                        face_value_per_share=projected.get(
                            "face_value_per_share", pd.Series(dtype=float)
                        ).get(asset),
                    )
                )
                spread = max(
                    0, (price - visible_marks[asset]) * (1 if quantity > 0 else -1)
                )
                return fee + count * spread

            def estimate_trade_cost_gradient(asset, notional, side):
                price = decision_prices.get((asset, side), np.nan)
                mid = visible_marks[asset]
                if not np.isfinite(price) or not np.isfinite(mid) or mid <= 0:
                    raise ValueError("policy_decision_cost_unavailable")
                ratio = price / mid
                fee, gradient = policy_costs[asset, side].totals_and_gradients(
                    np.asarray(notional * ratio),
                    np.asarray(np.nan),
                    prices=np.asarray(price),
                    face_value_per_share=projected.get(
                        "face_value_per_share", pd.Series(dtype=float)
                    ).get(asset),
                )
                spread = max(0.0, (ratio - 1.0) * (1 if side == "buy" else -1))
                return float(fee + notional * spread), float(gradient * ratio + spread)

            if (units.reindex(assets).gt(0) & ~np.isfinite(visible_marks)).any():
                yield emit(
                    at,
                    diagnostics=(
                        PortfolioDiagnostic(
                            PORTFOLIO_CASH_ASSET_ID,
                            None,
                            "policy_decision_unavailable",
                            text="held_visible_mark_missing",
                        ),
                    ),
                )
                return
            features = visible_features
            for asset in requested:
                if prediction_rows is not None and (at, asset) in prediction_rows.index:
                    for column in prediction_rows.columns:
                        features.loc[asset, column] = prediction_rows.loc[
                            (at, asset), column
                        ]
            pending = pd.Series(0.0, index=assets)
            pending_targets = pd.Series(np.nan, index=assets)
            if passive is not None:
                for asset, order in passive.orders.items():
                    pending[asset] = order.side * order.remaining
                for asset, intent in passive.intents.items():
                    pending_targets[asset] = intent[0]
                    if (
                        asset not in passive.orders
                        and asset in passive.pending_reservations
                    ):
                        pending[asset] = intent[0] - units[asset]
                for asset, (intent, _, _) in passive.replacements.items():
                    pending_targets[asset] = intent[0]
            else:
                for asset, queue in taker_intents.items():
                    if queue:
                        target = queue[-1][1]
                        pending_targets[asset] = target
                        pending[asset] = target - units[asset]
            locked = pd.Series(entitlements.locked_quantities, dtype="float64").reindex(
                units.index, fill_value=0.0
            )
            sellable = (units - locked - bought_today).clip(lower=0.0).reindex(assets)
            available_sellable = sellable.copy()
            if passive is not None:
                sellable -= pd.Series(
                    {asset: passive.frozen_shares(asset) for asset in assets}
                )
            reserved_cash = passive.frozen_cash() if passive is not None else 0.0
            nav = account_nav(visible_marks)
            limits = maximum_quantities(visible_marks, nav)
            target = yield PolicyDecision(
                at,
                features,
                units.reindex(assets).copy(),
                sellable,
                visible_marks,
                cash - reserved_cash,
                nav,
                reference_inventory.reindex(assets).copy(),
                pending,
                pending_targets,
                estimated_costs,
                estimate_trade_cost,
                reserved_cash=reserved_cash,
                receivable_value=fsum(entitlements.receivables.values()),
                estimate_trade_cost_gradient=estimate_trade_cost_gradient,
                order_state=passive.state(at, assets) if dynamic_execution else None,
                physical_sellable=available_sellable,
                maximum_quantities=limits,
            )
            modes = None
            strategy_targets = None
            target_limits = limits.fillna(units).clip(lower=units)
            if dynamic_execution:
                if not isinstance(target, PolicyExecution):
                    raise ValueError("policy_execution_request_required")
                modes = target.modes
                strategy_targets = policy_target(
                    target.strategy_targets,
                    assets,
                    maximum=target_limits,
                    allow_missing=True,
                )
                if (
                    not isinstance(modes, pd.Series)
                    or not modes.index.is_unique
                    or set(modes.index) != set(assets)
                    or not modes.isin(("wait", "cancel", "maker", "taker")).all()
                ):
                    raise ValueError("policy_execution_modes_invalid")
                target = target.targets
            target = (
                policy_target(
                    target,
                    assets,
                    maximum=target_limits,
                    allow_missing=True,
                )
                .where(features.prediction_score.notna() | dynamic_execution)
                .dropna()
            )
            for asset in target.index:
                target[asset] = status_target(asset, target[asset])
            diagnostics = []
            for asset, quantity in target.items():
                for name in ("raw_prediction_score", "prediction_error_scale"):
                    if name in features and pd.notna(features.loc[asset, name]):
                        diagnostics.append(
                            PortfolioDiagnostic(
                                asset,
                                None,
                                name,
                                number=float(features.loc[asset, name]),
                            )
                        )
                diagnostics.extend(
                    (
                        *(
                            [
                                PortfolioDiagnostic(
                                    asset,
                                    None,
                                    "prediction_score",
                                    number=float(
                                        features.loc[asset, "prediction_score"]
                                    ),
                                )
                            ]
                            if pd.notna(features.loc[asset, "prediction_score"])
                            else []
                        ),
                        *(
                            [
                                PortfolioDiagnostic(
                                    asset,
                                    None,
                                    "inventory_target",
                                    number=float(
                                        quantity
                                        if strategy_targets is None
                                        else strategy_targets[asset]
                                    ),
                                )
                            ]
                            if strategy_targets is None
                            or pd.notna(strategy_targets[asset])
                            else []
                        ),
                        PortfolioDiagnostic(
                            asset,
                            None,
                            "inventory_pending_quantity",
                            number=float(pending[asset]),
                        ),
                        PortfolioDiagnostic(
                            asset,
                            None,
                            "inventory_opening_quantity",
                            number=float(opening_inventory[asset]),
                        ),
                        *(
                            [
                                PortfolioDiagnostic(
                                    asset,
                                    None,
                                    "inventory_estimated_trade_cost_bps",
                                    number=float(estimated_costs[asset]),
                                )
                            ]
                            if np.isfinite(estimated_costs[asset])
                            else []
                        ),
                        *(
                            [
                                PortfolioDiagnostic(
                                    asset, None, "execution_mode", text=modes[asset]
                                ),
                                PortfolioDiagnostic(
                                    asset,
                                    None,
                                    "execution_destination",
                                    number=float(quantity),
                                ),
                            ]
                            if modes is not None
                            else []
                        ),
                    )
                )
                if passive is None and asset in requested:
                    taker_intents[asset].append((stamp, float(quantity), False))
            if passive is not None:
                diagnostics.extend(
                    passive.request(
                        at,
                        target,
                        {
                            asset: stamp + latency_ns if asset in requested else None
                            for asset in target.index
                        },
                        segment_bound=not (restore_at is not None and at >= restore_at),
                        modes=modes,
                        decision_book=pd.DataFrame(visible_quotes).T.reindex(assets)
                        if dynamic_execution
                        else None,
                        quantities=units,
                        sellable=available_sellable,
                        cash=cash,
                    )
                )
                if dynamic_execution:
                    diagnostics.extend(passive.diagnostics())
            if diagnostics:
                yield emit(at, diagnostics=diagnostics)

        def advance_events(
            until,
            *,
            include_requests=False,
            passive=passive,
            event_iterator=event_iterator,
            request_iterator=request_iterator,
            apply_request=apply_request,
        ):
            nonlocal next_trade, next_request, cash
            # Tied prints precede cancel acknowledgement, not observation expiry.
            while (
                passive is not None
                and next_trade is not None
                and next_trade.timestamp <= until
            ) or (
                next_request is not None
                and (
                    next_request[0] <= until.value
                    if include_requests
                    else next_request[0] < until.value
                )
            ):
                if (
                    passive is not None
                    and next_trade is not None
                    and next_trade.timestamp <= until
                    and (
                        next_request is None
                        or next_trade.timestamp.value <= next_request[0]
                    )
                ):
                    event = next_trade
                    next_trade = next(event_iterator, None)
                    cancellations = passive.advance(event.timestamp, inclusive=False)
                    fill = passive.trade(event)
                    if fill is not None:
                        trade, cashflows, diagnostic = settle(
                            event.timestamp, event.asset_id, *fill
                        )
                        yield emit(
                            event.timestamp,
                            trades=(trade,),
                            flows=cashflows,
                            diagnostics=(
                                diagnostic,
                                *cancellations,
                                *passive.diagnostics(),
                            ),
                        )
                    elif cancellations:
                        yield emit(
                            event.timestamp,
                            diagnostics=(*cancellations, *passive.diagnostics()),
                        )
                else:
                    request_at, requested = next_request
                    next_request = next(request_iterator, None)
                    # Fills update cash; do not freeze the old balance in an argument.
                    yield from apply_request(request_at, requested, cash)  # noqa: B023

        for stamp, group in quotes.groupby("timestamp", sort=False):
            at = pd.Timestamp(stamp)
            if passive is not None or next_request is not None:
                yield from advance_events(at)
            current = group.set_index("asset_id")
            current["can_enter"] = status_can_enter.reindex(current.index).fillna(False)
            valid = np.isfinite(current.mid) & current.mid.gt(0)
            marks.loc[current.index[valid]] = current.loc[valid, "mid"]
            sources.update({asset: "observed_mid" for asset in current.index[valid]})
            if markout_ns is not None:
                for asset, row in current.iterrows():
                    queue = pending_markouts.get(asset, ())
                    while queue:
                        fill_id, price, side, due, segment = queue[0]
                        same_segment = row.segment_id == segment
                        observed = same_segment and np.isfinite(row.mid) and row.mid > 0
                        if same_segment and (not observed or at.value < due):
                            break
                        queue.popleft()
                        if observed:
                            diagnostics = (
                                PortfolioDiagnostic(
                                    asset,
                                    fill_id,
                                    "markout_bps",
                                    number=float(side * (row.mid / price - 1) * 10000),
                                ),
                                PortfolioDiagnostic(
                                    asset,
                                    fill_id,
                                    "markout_observed_at_ns",
                                    text=str(at.value),
                                ),
                            )
                        else:
                            diagnostics = (
                                PortfolioDiagnostic(
                                    asset,
                                    fill_id,
                                    "markout_reason",
                                    text="unobserved_segment",
                                ),
                            )
                        # The ledger allows one diagnostic name per asset and batch.
                        yield emit(at, diagnostics=diagnostics)
            targets = units.copy()
            signal_times = {}
            restoring = (
                restore_at is not None
                and at > restore_at
                and at.value >= restore_at.value + latency_ns
            )
            limits = maximum_quantities(marks) if restoring else None
            for asset, segment in current.segment_id.items():
                queue = taker_intents[asset]
                if segments.get(asset) != segment:
                    queue.clear()
                segments[asset] = segment
                decision = None
                while (
                    queue
                    and queue[0][0] < at.value
                    and queue[0][0] + latency_ns <= at.value
                ):
                    decision = queue.popleft()
                if restoring:
                    targets[asset] = min(
                        reference_inventory[asset], max(units[asset], limits[asset])
                    )
                    signal_times[asset] = int(restore_at.value)
                elif decision is not None:
                    signal_at, target, buy_only = decision
                    if target is not None:
                        targets[asset] = (
                            max(units[asset], target) if buy_only else target
                        )
                        if execution_order is not None:
                            # Delayed targets cannot reverse or overfill a mother order.
                            targets[asset] = np.clip(
                                target,
                                units[asset] + min(0, remaining_order),
                                units[asset] + max(0, remaining_order),
                            )
                    signal_times[asset] = signal_at
                if asset in status_exit_at:
                    requested_at = status_exit_at[asset]
                    if (
                        at.value > requested_at
                        and at.value >= requested_at + latency_ns
                    ):
                        targets[asset] = 0.0
                        signal_times[asset] = requested_at
                elif not status_can_enter.get(asset, True):
                    targets[asset] = min(targets[asset], units[asset])
            locked = pd.Series(entitlements.locked_quantities, dtype="float64").reindex(
                units.index, fill_value=0.0
            )
            sellable = (units - locked - bought_today).clip(lower=0.0)
            diagnostics = [
                PortfolioDiagnostic(
                    PORTFOLIO_CASH_ASSET_ID, None, "session_phase", text="quote"
                )
            ]
            if restoring:
                diagnostics.extend(
                    PortfolioDiagnostic(
                        asset,
                        None,
                        "inventory_target",
                        number=float(status_target(asset, reference_inventory[asset])),
                    )
                    for asset in current.index
                )
            trades, flows = [], []
            taker_requests = {}
            if passive is not None:
                if execution_order is not None:
                    asset = execution_order["asset_id"]
                    if asset in passive.intents:
                        target, *arrival = passive.intents[asset]
                        passive.intents[asset] = (
                            float(
                                np.clip(
                                    target,
                                    units[asset] + min(0, remaining_order),
                                    units[asset] + max(0, remaining_order),
                                )
                            ),
                            *arrival,
                        )
                diagnostics.extend(passive.quote(at, current, units, sellable, cash))
                if dynamic_execution or restoring:
                    taker_requests = passive.taker_requests(at, current.index)
                    for asset, (signal_at, target) in taker_requests.items():
                        targets[asset] = status_target(asset, target)
                        signal_times[asset] = signal_at
                    sellable -= pd.Series(
                        {
                            asset: passive.frozen_shares(asset)
                            if asset not in taker_requests
                            else 0
                            for asset in sellable.index
                        }
                    )
                if restoring:
                    for asset in current.index:
                        if (
                            asset in passive.orders
                            or asset in passive.intent_cancellations
                            or asset in passive.replacements
                        ):
                            if asset not in taker_requests:
                                targets[asset] = units[asset]
                        else:
                            targets[asset] = status_target(
                                asset,
                                min(
                                    reference_inventory[asset],
                                    max(units[asset], limits[asset]),
                                ),
                            )
                            signal_times[asset] = int(restore_at.value)
            if (passive is None or dynamic_execution or restoring) and targets.ne(
                units
            ).any():
                # One-sided quotes can execute against the observed side while
                # valuation retains its last known mid; crossed books cannot.
                crossed = (
                    current.bid_px1.gt(0)
                    & current.ask_px1.gt(0)
                    & current.bid_px1.ge(current.ask_px1)
                )
                permissions = (
                    (~crossed.fillna(False)).reindex(units.index, fill_value=False)
                    & np.isfinite(marks)
                    & marks.gt(0)
                )
                orders = pd.DataFrame(
                    {
                        "quantity": units,
                        "target_quantity": targets,
                        "sellable_quantity": sellable,
                        "market_type": projected.market_type.reindex(units.index),
                        "instrument_type": instrument_types.reindex(units.index),
                        "mark_price": marks.fillna(0.0),
                        "buy_price": current.ask_px1.reindex(units.index),
                        "sell_price": current.bid_px1.reindex(units.index),
                        "buy_capacity": 0.0,
                        "sell_capacity": 0.0,
                        "adv": np.nan,
                        "can_enter": permissions
                        & status_can_enter.reindex(units.index).fillna(False),
                        "can_exit": permissions,
                    }
                )
                if "face_value_per_share" in projected:
                    orders["face_value_per_share"] = (
                        projected.face_value_per_share.reindex(units.index)
                    )
                depth = {
                    asset: quoted_execution_depth(
                        current.loc[asset],
                        1 if targets[asset] > units[asset] else -1,
                        params.get("taker_depth_levels", 1),
                        params["participation_rate"],
                    )
                    for asset in current.index[
                        targets.reindex(current.index).ne(units.reindex(current.index))
                    ]
                }
                # Successor reservations cannot block the IOC that precedes them.
                reserved_cash = (
                    passive.frozen_cash(exclude_assets=taker_requests)
                    if passive is not None
                    else 0.0
                )
                account = execute_stock_target(
                    orders,
                    cash=cash - reserved_cash,
                    execution_date=day.date(),
                    fee_schedule=schedule,
                    assumptions=assumptions,
                    receivable=fsum(entitlements.receivables.values()),
                    depth=depth,
                )
                executed = account.asset_ledger
                requests = executed.target_notional - executed.pre_trade_notional
                diagnostics.append(
                    PortfolioDiagnostic(
                        PORTFOLIO_CASH_ASSET_ID,
                        None,
                        "requested_turnover_notional",
                        number=(fsum(abs(requests)) + abs(fsum(requests))) / 2,
                    )
                )
                for row in executed.itertuples(index=False):
                    quantity = int(row.executed_quantity)
                    requested = float(targets[row.asset_id] - units[row.asset_id])
                    if requested:
                        diagnostics.extend(
                            (
                                PortfolioDiagnostic(
                                    row.asset_id,
                                    None,
                                    "requested_quantity",
                                    number=requested,
                                ),
                                PortfolioDiagnostic(
                                    row.asset_id,
                                    None,
                                    "unfilled_quantity",
                                    number=requested - quantity,
                                ),
                            )
                        )
                    if pd.notna(row.reason) and row.reason:
                        diagnostics.append(
                            PortfolioDiagnostic(
                                row.asset_id,
                                None,
                                "order_block_reason",
                                text=row.reason,
                            )
                        )
                    if not quantity:
                        continue
                    trade, cashflows, diagnostic = settle(
                        at,
                        row.asset_id,
                        str(signal_times[row.asset_id]),
                        signal_times[row.asset_id],
                        quantity,
                        float(row.executed_delta_notional),
                        np.array(
                            [
                                getattr(row, name)
                                for name in PORTFOLIO_LEDGER_COST_COLUMNS
                            ]
                        ),
                    )
                    trades.append(trade)
                    flows.extend(cashflows)
                    diagnostics.append(diagnostic)
                cash = account.post_trade_cash + reserved_cash
            if passive is not None:
                diagnostics.extend(passive.diagnostics())
            for asset in units.index:
                diagnostics.extend(
                    (
                        PortfolioDiagnostic(
                            asset,
                            None,
                            "sellable_quantity",
                            number=float(
                                max(
                                    0.0,
                                    units[asset] - locked[asset] - bought_today[asset],
                                )
                            ),
                        ),
                        PortfolioDiagnostic(
                            asset, None, "mark_source", text=sources[asset]
                        ),
                    )
                )
            yield emit(at, trades=trades, flows=flows, diagnostics=diagnostics)
            if next_request is not None and next_request[0] == at.value:
                request_at, requested = next_request
                next_request = next(request_iterator, None)
                yield from apply_request(request_at, requested, cash)
            if (
                interactive
                and not prediction_decisions
                and (restore_at is None or at < restore_at)
                and at < endpoint
                and at.value + latency_ns <= endpoint.value
            ):
                limits = maximum_quantities(marks)
                target = yield PolicyDecision(
                    at,
                    current.reindex(assets).loc[:, list(decision_columns)].copy(),
                    units.reindex(assets).copy(),
                    (units - locked - bought_today).clip(lower=0.0).reindex(assets)
                    - (
                        pd.Series(
                            {asset: passive.frozen_shares(asset) for asset in assets}
                        )
                        if passive is not None
                        else 0
                    ),
                    marks.reindex(assets).copy(),
                    cash - (passive.frozen_cash() if passive is not None else 0),
                    fsum(row.value for row in values(at).values()),
                    maximum_quantities=limits,
                )
                target = policy_target(
                    target, assets, maximum=limits.fillna(units).clip(lower=units)
                )
                for asset in target.index:
                    target[asset] = status_target(asset, target[asset])
                if execution_order is not None:
                    other_assets = assets.difference([execution_order["asset_id"]])
                    if (
                        not target.reindex(other_assets)
                        .eq(units.reindex(other_assets))
                        .all()
                    ):
                        raise ValueError("policy_execution_off_mandate")
                if passive is None:
                    for asset in current.index:
                        taker_intents[asset].append(
                            (at.value, float(target[asset]), False)
                        )
                else:
                    diagnostics = passive.request(
                        at,
                        target.reindex(current.index),
                        dict.fromkeys(current.index, at.value + latency_ns),
                    )
                    if diagnostics:
                        yield emit(at, diagnostics=diagnostics)
        if passive is not None or next_request is not None:
            yield from advance_events(endpoint, include_requests=True)
        if passive is not None:
            diagnostics = passive.advance(endpoint)
            if diagnostics:
                yield emit(endpoint, diagnostics=(*diagnostics, *passive.diagnostics()))
        if pending_markouts:
            for asset, queue in pending_markouts.items():
                while queue:
                    fill_id, _, _, _, _ = queue.popleft()
                    yield emit(
                        endpoint,
                        diagnostics=(
                            PortfolioDiagnostic(
                                asset,
                                fill_id,
                                "markout_reason",
                                text="no_observed_markout",
                            ),
                        ),
                    )
        if closing <= upper if closed else closing < upper:
            marks = projected.close.reindex(units.index).astype("float64")
            rights = entitlements.register(
                view.corporate_events, units, lives, day, phase="after"
            )
            adjustments, flows = equity_at(rights, closing)
            diagnostics = [
                PortfolioDiagnostic(
                    PORTFOLIO_CASH_ASSET_ID, None, "session_phase", text="close"
                )
            ]
            if passive is not None:
                diagnostics.extend(passive.expire("day_expired"))
                if dynamic_execution:
                    diagnostics.extend(passive.diagnostics())
            for asset in status_exit_at:
                if units[asset] > 0:
                    diagnostics.extend(
                        (
                            PortfolioDiagnostic(
                                asset,
                                None,
                                "stock_status_exit_pending_quantity",
                                number=float(units[asset]),
                            ),
                            PortfolioDiagnostic(
                                asset,
                                None,
                                "stock_status_exit_unfinished",
                                text="same_day_or_corporate_lock"
                                if bought_today[asset] > 0
                                or entitlements.locked_quantities.get(asset, 0) > 0
                                else "liquidity_or_execution_window",
                            ),
                        )
                    )
            diagnostics.extend(
                PortfolioDiagnostic(
                    asset,
                    None,
                    "inventory_change",
                    number=float(units[asset] - opening_inventory.get(asset, 0.0)),
                )
                for asset in units.index
            )
            if restore_at is not None:
                diagnostics.append(
                    PortfolioDiagnostic(
                        PORTFOLIO_CASH_ASSET_ID,
                        None,
                        "inventory_restoration_status",
                        text="COMPLETE"
                        if np.allclose(
                            units,
                            reference_inventory.reindex(units.index, fill_value=0.0),
                            rtol=1e-12,
                            atol=1e-9,
                        )
                        else "INCOMPLETE",
                    )
                )
            yield emit(
                closing, flows=flows, adjustments=adjustments, diagnostics=diagnostics
            )
    if not observed_quotes:
        raise ValueError("hft_backtest_empty")


def statistics_tables(ledgers, *, window, config, metadata=None):
    """Reduce quote transitions to trading-day endpoints before annualization."""
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )

    analyses = config.get("analyses", ["performance"])
    if set(analyses) - {"performance", "attribution"}:
        raise ValueError("hft_statistics_analysis_invalid")
    attribution = "attribution" in analyses
    risk_inputs = _inventory_risk_inputs(config, window) if attribution else None
    inputs = config.get("inputs", {})
    calendar_path = inputs.get("calendar", inputs.get("daily", {}).get("calendar"))
    calendar = (
        pd.DatetimeIndex(unified_trading_sessions(read_dataset(Path(calendar_path))))
        if calendar_path is not None
        else None
    )
    daily, summaries, attribution_tables = [], [], {}
    risk_cache = {}
    for candidate_id, ledger in ledgers.items():
        if attribution:
            from quant_research_platform.research.portfolios.attribution import (
                inventory_risk_attribution,
            )
            from quant_research_platform.storage.datasets.reading import (
                iter_dataset_time_chunks,
            )

            observations, model, facts, available_at = risk_inputs
            selected, economic, parts = inventory_risk_attribution(
                ledger,
                model=model,
                observations=iter_dataset_time_chunks(observations, one_timestamp=True),
                risk_facts=facts,
                available_at_column=available_at,
                daily_views=_daily_market_views(config["inputs"]["daily"], window)
                if "daily" in config["inputs"]
                else (),
                risk_cache=risk_cache,
            )
            for name, frame in parts.items():
                attribution_tables.setdefault(name, []).append(frame)
        else:
            selected, economic = inventory_daily_economics(ledger)
        selected["candidate_id"] = candidate_id
        if selected.empty or not selected.entry_at.lt(selected.end_at).all():
            raise ValueError("hft_statistics_empty_interval")
        denominator = selected.nav_before.where(selected.nav_before > 0)
        selected["nav_post"] = selected.nav_end
        selected["cash_weight_post"] = selected.cash_post / selected.nav_post.where(
            selected.nav_post > 0
        )
        selected["net_return"] = (selected.nav_end - selected.nav_before) / denominator
        selected["gross_return"] = (
            selected.nav_end - selected.nav_before + selected.cost
        ) / denominator
        selected["cost_drag"] = selected.cost / denominator
        selected["target_turnover"] = selected["requested_turnover"] = (
            selected.requested / denominator
        )
        selected["realized_turnover"] = selected.traded / denominator
        selected["return_status"] = np.where(
            selected.nav_before > 0, "DEFINED", "UNDEFINED"
        )
        selected["return_reason"] = np.where(
            selected.nav_before > 0, "", "nonpositive_start_nav"
        )
        selected["sample_role"] = config["sample_role"]
        dates = pd.DatetimeIndex(selected.end_at.dt.normalize())
        complete = (
            calendar is not None
            and dates.equals(
                calendar[
                    (calendar >= pd.Timestamp(window.start).normalize())
                    & (calendar <= pd.Timestamp(window.end).normalize())
                ]
            )
            and selected.end_at.eq(
                selected.end_at.dt.normalize() + pd.Timedelta(hours=7)
            ).all()
            and selected.entry_at.iloc[0]
            == dates[0] + pd.Timedelta(hours=1, minutes=30)
            and np.array_equal(
                selected.entry_at.iloc[1:].to_numpy(),
                selected.end_at.iloc[:-1].to_numpy(),
            )
        )
        if "performance" in analyses:
            summaries.extend(
                performance_rows(
                    selected,
                    capital=ledger.capital,
                    economic=economic,
                    daily_complete=complete,
                    sample_role=config["sample_role"],
                )
            )
        daily.append(
            selected.reindex(
                columns=STATISTICS_TABLE_COLUMNS["portfolio_daily.parquet"]
            )
        )
    tables = {
        "portfolio_daily": pd.concat(daily, ignore_index=True),
        "portfolio_performance": pd.DataFrame(summaries).reindex(
            columns=STATISTICS_TABLE_COLUMNS["portfolio_performance.parquet"]
        ),
    }
    for name, frames in attribution_tables.items():
        tables[name] = pd.concat(frames, ignore_index=True).reindex(
            columns=STATISTICS_TABLE_COLUMNS[name + ".parquet"]
        )
    if attribution:
        tables["holding_factor_returns"] = tables[
            "holding_factor_returns"
        ].drop_duplicates(["attribution_group", "entry_at", "end_at", "factor_id"])
        if metadata is not None:
            metadata.update(
                risk_model=model.to_dict(),
                attribution_clock="inventory_intervals",
                attribution_scope="absolute_inventory_price_intervals_and_native_account_economics",
            )
    return tables


def _inventory_risk_inputs(config, window):
    """Read one explicit full-market source, never a holdings-filtered risk fit."""
    from quant_research_platform.contracts.risk import FactorRiskModel
    from quant_research_platform.research.datasets.risk import (
        read_asof_risk_facts,
        risk_exposure_columns,
    )

    inputs = config["inputs"]
    declaration = _strict_config(
        config.get("risk_exposure_input", {}), {"data_id", "available_at_column"}, set()
    )
    data_id, available_at = declaration["data_id"], declaration["available_at_column"]
    if (
        type(data_id) is not str
        or type(available_at) is not str
        or not available_at
        or data_id in {"observations", "daily", "calendar"}
        or data_id not in inputs
        or available_at in {"timestamp", "asset_id", "market_id", "listing_date"}
        or "observations" not in inputs
    ):
        raise ValueError("hft_risk_inputs_invalid")
    observations = open_dataset_session(
        **_dataset_input_options(inputs["observations"], time_window=window)
    )
    declared = observations.spec.get("risk_model")
    model = FactorRiskModel.from_mapping(config.get("risk_model", declared or {}))
    if declared is not None and model != FactorRiskModel.from_mapping(declared):
        raise ValueError("portfolio_attribution_risk_model_conflict")
    facts = open_dataset_session(**_dataset_input_options(inputs[data_id]))
    if (
        observations.spec.get("market_id") != "a_share"
        or observations.spec["frequency"] != "tick"
        or facts.spec.get("market_id") != "a_share"
        or facts.spec["frequency"] != "1d"
    ):
        raise ValueError("hft_risk_inputs_invalid")
    observation_columns = {
        "market_id",
        "asset_id",
        "timestamp",
        "listing_date",
        "eligible",
        "mid",
        "quote_timestamp",
    }
    if observations.spec["column_roles"].get("eligible") not in {"universe", "mask"}:
        # A raw book-completeness flag does not define the research universe.
        observation_columns.remove("eligible")
    fact_columns = {
        "market_id",
        "asset_id",
        "timestamp",
        "listing_date",
        available_at,
        *risk_exposure_columns(model),
    }
    observations = observations.projected_view(
        tuple(
            column
            for column in observations.logical_columns
            if column in observation_columns
        )
    )
    facts = facts.projected_view(
        tuple(column for column in facts.logical_columns if column in fact_columns)
    )
    return (
        observations,
        model,
        read_asof_risk_facts(
            facts,
            model=model,
            available_at_column=available_at,
            time_window=window,
        ),
        available_at,
    )


def _daily_market_views(daily, window):
    """Reuse native daily identity, corporate actions and valuation projections."""
    daily = _strict_config(
        daily,
        {"panel", "calendar", "market_bars", "stock_master", "corporate_actions"},
        {"equity_events"},
    )
    calendar = read_dataset(Path(daily["calendar"]))
    # Native bounds retain all mark/event warmup; no fact can affect earlier views.
    first_facts = [pd.Timestamp(window.start)]
    for name in ("panel", "market_bars", "corporate_actions", "equity_events"):
        if name in daily:
            source = open_dataset_session(**_dataset_input_options(daily[name]))
            first = inspect_dataset_session(source).date_min
            if first is not None:
                first_facts.append(pd.Timestamp(first))
    daily_window = TimeWindow(
        max(pd.Timestamp(calendar.calendar_date.min()), min(first_facts)).date(),
        pd.Timestamp(window.end).date(),
    )
    return iter_portfolio_sessions(
        market_bars_root=daily["market_bars"],
        stock_master_root=daily["stock_master"],
        panel_root=daily["panel"],
        panel_columns=(
            "asset_id",
            "listing_date",
            "timestamp",
            "open",
            "close",
            "amount",
            "market_type",
            "is_trading",
        ),
        calendar=calendar,
        time_window=daily_window,
        corporate_actions_root=daily["corporate_actions"],
        equity_events_root=daily.get("equity_events"),
    )


def inventory_inputs(
    config,
    window,
    *,
    prediction_paths=(),
    prediction_calibration=None,
    decision_columns=None,
    prediction_validator=None,
):
    """Open the same explicit market inputs for training and native backtests."""

    from quant_research_platform.research.protocols.a_share_hft import PANEL_COLUMNS

    daily = config["daily_inputs"]
    start, end = pd.Timestamp(window.start).date(), pd.Timestamp(window.end).date()
    asset_filter = (("asset_id", "in", tuple(config["parameters"]["maximum_weights"])),)
    panel = open_dataset_session(
        **_dataset_input_options(
            config["panel"],
            time_window=TimeWindow(start, end),
            row_filters=asset_filter,
        )
    )
    if panel.spec.get("market_id") != "a_share" or panel.spec["frequency"] != "tick":
        raise ValueError("hft_panel_required")
    passive = "passive" in config["parameters"]
    if passive != ("trades" in config):
        raise ValueError("passive_trade_input_required_or_unused")
    trade_session = None
    max_quote_gap_seconds = None
    if passive:
        max_quote_gap_seconds = panel.spec.get("parameters", {}).get("max_gap_seconds")
        if max_quote_gap_seconds is None:
            raise ValueError("passive_quote_gap_required")
        trade_session = open_dataset_session(
            **{
                **_dataset_input_options(
                    config["trades"],
                    time_window=TimeWindow(start, end),
                    row_filters=asset_filter,
                ),
                "columns": (
                    "timestamp",
                    "asset_id",
                    "source_row",
                    "event_kind",
                    "side",
                    "price",
                    "volume",
                ),
            }
        )
    if decision_columns is not None and any(
        panel.spec["column_roles"].get(column) not in {"feature", "factor"}
        and not (
            panel.spec["column_roles"].get(column) == "context"
            and column in PANEL_COLUMNS
        )
        for column in decision_columns
    ):
        raise ValueError("policy_observation_feature_role_required")
    fee_frame = read_dataset(Path(config["fee_schedule"]))
    schedule = fee_schedule_from_frame(
        fee_frame,
        content_identity=canonical_json_hash(fee_frame.to_json(date_format="iso")),
    )
    daily_sessions = _daily_market_views(daily, window)
    predictions, metadata = read_rolling_predictions(
        prediction_paths,
        window=TimeWindow(start, end),
        asset_ids=tuple(config["parameters"]["maximum_weights"]),
    )
    for facts in metadata:
        if facts["model_facts"]["market_id"] != "a_share":
            raise ValueError("hft_prediction_market_mismatch")
        if prediction_validator is not None:
            prediction_validator(facts)
    if prediction_calibration is not None:
        from quant_research_platform.research.predictions.amplitude import (
            apply_amplitude,
        )

        predictions = apply_amplitude(predictions, metadata, prediction_calibration)
    prediction_days = {}
    if not predictions.empty:
        prediction_days = {
            day: frame
            for day, frame in predictions.groupby(
                predictions.timestamp.dt.normalize(), sort=False
            )
        }

    def sessions():
        with closing_iterator(daily_sessions) as views:
            for view in views:
                day_window = TimeWindow(view.timestamp.date(), view.timestamp.date())
                if view.timestamp.date() < start:
                    quotes = predictions = pd.DataFrame()
                else:
                    quotes = materialize_dataset_session(
                        panel.filtered_view(day_window)
                    )
                    predictions = prediction_days.pop(view.timestamp, pd.DataFrame())
                yield view, quotes, predictions

    return sessions(), schedule, trade_session, max_quote_gap_seconds


def backtest(config):
    """Publish one continuous multi-stock account with shared equity handling."""
    config = _strict_config(
        config,
        {
            "panel",
            "predictions",
            "fee_schedule",
            "daily_inputs",
            "window",
            "parameters",
            "output",
        },
        {"trades", "stock_status", "candidate_id"},
    )
    params = _parameters(config["parameters"])
    window = TimeWindow(**config["window"])
    if not isinstance(config["predictions"], list):
        raise ValueError("hft_prediction_paths_required")
    sessions, schedule, trade_session, max_quote_gap_seconds = inventory_inputs(
        config, window, prediction_paths=config["predictions"]
    )

    def produce(emit):
        with closing_iterator(sessions) as inputs:
            for batch in inventory_batches(
                inputs,
                schedule,
                params,
                window,
                trade_session=trade_session,
                max_quote_gap_seconds=max_quote_gap_seconds,
                stock_status=config.get("stock_status"),
            ):
                emit(batch)

    capital = PortfolioCapital(
        config.get("candidate_id", "hft_inventory"),
        params["initial_capital"],
        "CNY",
        "cash_equity",
    )
    return write_portfolio_ledger_artifact(
        Path(config["output"]),
        capital=capital,
        metadata={
            **{key: value for key, value in config.items() if key != "output"},
            "market_id": "a_share",
            "kind": "a_share_hft",
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "parameters": params,
            "account_semantics": "shared_cash_continuous_inventory_by_instrument",
            **(
                {"passive_quote_max_gap_seconds": max_quote_gap_seconds}
                if "passive" in params
                else {}
            ),
            "execution": (
                "causal_observed_depth_passive_queue_no_market_feedback_or_receive_clock"
                if "passive" in params
                else (
                    "strictly_subsequent_observed_depth_no_queue_or_receive_clock"
                    if params.get("taker_depth_levels", 1) > 1
                    else (
                        "strictly_subsequent_observed_top_of_book_no_queue_or_receive_clock"
                    )
                )
            ),
            "corporate_action_policy": CORPORATE_ACTION_POLICY,
            "fee_schedule_identity": schedule.content_identity,
            "stock_status_controls": "published_lifecycle_status"
            if "stock_status" in config
            else "not_configured",
        },
        produce=produce,
    )
