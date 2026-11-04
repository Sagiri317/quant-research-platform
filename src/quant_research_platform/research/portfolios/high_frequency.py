"""Continuous A-share inventory accounts on observed subsequent quotes."""

from dataclasses import replace
from datetime import time
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    canonical_json_hash,
)
from quant_research_platform.contracts.models import _validate_model_prediction_window
from quant_research_platform.contracts.portfolios import (
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
    PredictionWindow,
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
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
)
from quant_research_platform.research.portfolios.performance import performance_rows
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
    read_dataset,
)


def _parameters(value):
    params = _strict_config(
        value,
        {
            "initial_capital",
            "initial_inventory",
            "maximum_inventory",
            "buy_threshold",
            "sell_threshold",
            "latency_ms",
            "participation_rate",
            "minimum_commission",
            "restore_at",
        },
        set(),
    )
    for name in ("initial_inventory", "maximum_inventory"):
        values = params[name]
        if not isinstance(values, dict) or any(
            type(asset) is not str
            or not asset
            or asset == PORTFOLIO_CASH_ASSET_ID
            or type(quantity) is not int
            or quantity < 0
            for asset, quantity in values.items()
        ):
            raise ValueError("hft_inventory_invalid")
    initial, maximum = params["initial_inventory"], params["maximum_inventory"]
    if (
        not maximum
        or not initial.keys() <= maximum.keys()
        or any(quantity > maximum[asset] for asset, quantity in initial.items())
    ):
        raise ValueError("hft_inventory_invalid")
    for name in set(params) - {"initial_inventory", "maximum_inventory", "restore_at"}:
        params[name] = strict_finite_float(params[name], error="hft_parameter_invalid")
    if (
        params["initial_capital"] <= 0
        or params["buy_threshold"] <= params["sell_threshold"]
        or params["latency_ms"] < 0
        or not 0 < params["participation_rate"] <= 1
        or params["minimum_commission"] < 0
    ):
        raise ValueError("hft_parameter_invalid")
    restore = params["restore_at"]
    if restore is not None:
        if not isinstance(restore, str):
            raise ValueError("hft_restore_time_invalid")
        clock = time.fromisoformat(restore)
        if clock.tzinfo is not None or not time(13) <= clock < time(14, 57):
            raise ValueError("hft_restore_time_invalid")
        params["restore_at"] = clock.isoformat()
    return params


def _decisions(quotes, predictions, latency_ns):
    result = {}
    signals = {
        asset: group for asset, group in predictions.groupby("asset_id", sort=False)
    }
    for asset, group in quotes.groupby("asset_id", sort=False):
        if asset not in signals:
            continue
        times = group.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
        segments = group.segment_id.to_numpy()
        for row in signals[asset].sort_values("timestamp").itertuples(index=False):
            signal_at = int(row.timestamp.value)
            index = max(
                np.searchsorted(times, signal_at + latency_ns),
                np.searchsorted(times, signal_at, side="right"),
            )
            decision = np.searchsorted(times, signal_at)
            if (
                index < len(times)
                and decision < len(times)
                and times[decision] == signal_at
                and segments[index] == segments[decision]
            ):
                result[int(times[index]), asset] = (
                    signal_at,
                    float(row.prediction_score),
                )
    return result


def inventory_batches(sessions, schedule, params, window):
    """Share one cash pool and carry holdings; today's buys unlock next session."""
    lower, upper, closed = time_window_timestamp_interval(window)
    lower, upper = pd.Timestamp(lower), pd.Timestamp(upper)
    assets = pd.Index(sorted(params["maximum_inventory"]), name="asset_id")
    units = pd.Series(params["initial_inventory"], dtype="float64").reindex(
        assets, fill_value=0.0
    )
    lives = pd.Series(dtype="datetime64[ns]")
    marks = pd.Series(dtype="float64")
    sources = {}
    cash = None
    previous = {}
    sequence = 0
    observed_quotes = 0
    entitlements = EquityEntitlements()
    valuation_marks = ResearchValuationMarks()
    latency_ns = int(params["latency_ms"] * 1_000_000)
    assumptions = ExecutionAssumptions(
        params["initial_capital"], 0.0, 0.0, 1.0, (1.0,), params["minimum_commission"]
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
        before, after = [], []
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
        batch = PortfolioLedgerBatch(
            sequence,
            tuple(trades),
            tuple(flows),
            (*before, *after, *(replace(row, phase="end") for row in after)),
            tuple(diagnostics),
            tuple(adjustments),
        )
        previous = {row.asset_id: replace(row, phase="end") for row in after}
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
        opening_marks = projected.open.reindex(units.index).astype("float64")
        observed = quotes.loc[
            quotes.timestamp.le(start) & quotes.eligible & np.isfinite(quotes.mid)
        ].drop_duplicates("asset_id", keep="last")
        opening_marks.loc[observed.asset_id] = observed.mid.to_numpy()
        if cash is None:
            lives = projected.listing_date.reindex(assets)
            marks = opening_marks
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
        predictions = predictions.loc[
            predictions.timestamp.ge(lower)
            & (
                predictions.timestamp.le(upper)
                if closed
                else predictions.timestamp.lt(upper)
            )
        ]
        decisions = _decisions(quotes, predictions, latency_ns)
        if params["restore_at"] is None:
            restore_at = None
        else:
            restore_at = (
                day + pd.Timedelta(params["restore_at"]) - pd.Timedelta(hours=8)
            )
        quotes = quotes.loc[
            quotes.timestamp.ge(start)
            & (quotes.timestamp.le(upper) if closed else quotes.timestamp.lt(upper))
        ]
        observed_quotes += len(quotes)
        for stamp, group in quotes.groupby("timestamp", sort=False):
            at = pd.Timestamp(stamp)
            current = group.set_index("asset_id")
            valid = current.eligible & np.isfinite(current.mid) & current.mid.gt(0)
            marks.loc[current.index[valid]] = current.loc[valid, "mid"]
            sources.update({asset: "observed_mid" for asset in current.index[valid]})
            targets = units.copy()
            signal_times = {}
            restoring = (
                restore_at is not None
                and at > restore_at
                and at.value >= restore_at.value + latency_ns
            )
            for asset in current.index:
                if restoring:
                    targets[asset] = min(
                        opening_inventory[asset], params["maximum_inventory"][asset]
                    )
                    signal_times[asset] = int(restore_at.value)
                elif (int(at.value), asset) in decisions:
                    signal_at, score = decisions[int(at.value), asset]
                    if score > params["buy_threshold"]:
                        targets[asset] = params["maximum_inventory"][asset]
                    elif score < params["sell_threshold"]:
                        targets[asset] = 0.0
                    signal_times[asset] = signal_at
            locked = pd.Series(entitlements.locked_quantities, dtype="float64").reindex(
                units.index, fill_value=0.0
            )
            sellable = (units - locked - bought_today).clip(lower=0.0)
            diagnostics = [
                PortfolioDiagnostic(
                    PORTFOLIO_CASH_ASSET_ID, None, "session_phase", text="quote"
                )
            ]
            trades, flows = [], []
            if targets.ne(units).any():
                permissions = valid.reindex(units.index, fill_value=False)
                orders = pd.DataFrame(
                    {
                        "quantity": units,
                        "target_quantity": targets,
                        "sellable_quantity": sellable,
                        "market_type": projected.market_type.reindex(units.index),
                        "mark_price": marks.fillna(0.0),
                        "buy_price": current.ask_px1.reindex(units.index),
                        "sell_price": current.bid_px1.reindex(units.index),
                        "buy_capacity": current.ask_vol1.reindex(units.index)
                        * params["participation_rate"],
                        "sell_capacity": current.bid_vol1.reindex(units.index)
                        * params["participation_rate"],
                        "adv": np.nan,
                        "can_enter": permissions,
                        "can_exit": permissions,
                    }
                )
                account = execute_stock_target(
                    orders,
                    cash=cash,
                    execution_date=day.date(),
                    fee_schedule=schedule,
                    assumptions=assumptions,
                    receivable=fsum(entitlements.receivables.values()),
                )
                cash = account.post_trade_cash
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
                    if row.reason:
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
                    fill_id = f"{sequence}:{row.asset_id}"
                    trades.append(
                        ExecutedTrade(
                            fill_id,
                            str(signal_times[row.asset_id]),
                            int(at.value),
                            row.asset_id,
                            quantity,
                            float(row.executed_delta_notional),
                        )
                    )
                    flows.append(
                        PortfolioCashFlow(
                            f"{fill_id}:principal",
                            int(at.value),
                            row.asset_id,
                            fill_id,
                            "principal",
                            -float(row.executed_delta_notional),
                        )
                    )
                    flows.extend(
                        PortfolioCashFlow(
                            f"{fill_id}:{name}",
                            int(at.value),
                            row.asset_id,
                            fill_id,
                            name,
                            -float(getattr(row, name)),
                        )
                        for name in PORTFOLIO_LEDGER_COST_COLUMNS
                        if getattr(row, name)
                    )
                    units[row.asset_id] += quantity
                    if quantity > 0:
                        bought_today[row.asset_id] += quantity
                    diagnostics.append(
                        PortfolioDiagnostic(
                            row.asset_id,
                            fill_id,
                            "decision_timestamp_ns",
                            text=str(signal_times[row.asset_id]),
                        )
                    )
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
                            opening_inventory.reindex(units.index, fill_value=0.0),
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


def statistics_tables(ledgers, *, window, config):
    """Reduce quote transitions to trading-day endpoints before annualization."""
    from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )

    if config.get("analyses", ["performance"]) != ["performance"]:
        raise ValueError("hft_statistics_requires_performance_analysis")
    calendar_path = config.get("inputs", {}).get("calendar")
    calendar = (
        pd.DatetimeIndex(unified_trading_sessions(read_dataset(Path(calendar_path))))
        if calendar_path is not None
        else None
    )
    daily, summaries = [], []
    for candidate_id, ledger in ledgers.items():
        economic = _LedgerEconomics(ledger.capital)
        days = {}
        with closing_iterator(ledger.iter_chunks()) as batches:
            for batch in batches:
                before = [row for row in batch.valuations if row.phase == "before"]
                after = [row for row in batch.valuations if row.phase == "end"]
                at = pd.Timestamp(after[0].timestamp_ns)
                day = at.normalize()
                if day not in days:
                    days[day] = dict(
                        candidate_id=candidate_id,
                        entry_at=pd.Timestamp(before[0].timestamp_ns),
                        nav_before=fsum(row.value for row in before),
                        cost=0.0,
                        requested=0.0,
                        traded=0.0,
                    )
                row = days[day]
                increment = economic.consume(batch)
                row["cost"] += increment["fees"]
                row["requested"] += fsum(
                    item.number
                    for item in batch.diagnostics
                    if item.name == "requested_turnover_notional"
                )
                amounts = [trade.notional for trade in batch.trades]
                row["traded"] += (fsum(map(abs, amounts)) + abs(fsum(amounts))) / 2
                row.update(
                    end_at=at,
                    nav_end=fsum(item.value for item in after),
                    cash_post=next(
                        item.value
                        for item in after
                        if item.asset_id == PORTFOLIO_CASH_ASSET_ID
                    ),
                )
        selected = pd.DataFrame(days.values())
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
        summaries.extend(
            performance_rows(
                selected,
                capital=ledger.capital,
                economic=economic.summary(),
                daily_complete=complete,
                sample_role=config["sample_role"],
            )
        )
        daily.append(
            selected.reindex(
                columns=STATISTICS_TABLE_COLUMNS["portfolio_daily.parquet"]
            )
        )
    return {
        "portfolio_daily": pd.concat(daily, ignore_index=True),
        "portfolio_performance": pd.DataFrame(summaries),
    }


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
        set(),
    )
    daily = _strict_config(
        config["daily_inputs"],
        {"panel", "calendar", "market_bars", "stock_master", "corporate_actions"},
        {"equity_events"},
    )
    params = _parameters(config["parameters"])
    window = TimeWindow(**config["window"])
    start, end = pd.Timestamp(window.start).date(), pd.Timestamp(window.end).date()
    panel = open_dataset_session(
        **_dataset_input_options(config["panel"], time_window=TimeWindow(start, end))
    )
    if panel.spec.get("market_id") != "a_share" or panel.spec["data_id"] != "hft_panel":
        raise ValueError("hft_panel_required")
    fee_frame = read_dataset(Path(config["fee_schedule"]))
    schedule = fee_schedule_from_frame(
        fee_frame,
        content_identity=canonical_json_hash(fee_frame.to_json(date_format="iso")),
    )
    calendar = read_dataset(Path(daily["calendar"]))
    daily_window = TimeWindow(pd.Timestamp(calendar.calendar_date.min()).date(), end)
    daily_sessions = iter_portfolio_sessions(
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
                    predictions, facts = read_predictions(
                        Path(config["predictions"]), time_window=day_window
                    )
                    if facts["model_facts"]["market_id"] != "a_share":
                        raise ValueError("hft_prediction_market_mismatch")
                    _validate_model_prediction_window(
                        facts["model_facts"], PredictionWindow(window.start, window.end)
                    )
                yield view, quotes, predictions

    def produce(emit):
        with closing_iterator(sessions()) as inputs:
            for batch in inventory_batches(inputs, schedule, params, window):
                emit(batch)

    capital = PortfolioCapital(
        "hft_inventory", params["initial_capital"], "CNY", "cash_equity"
    )
    return write_portfolio_ledger_artifact(
        Path(config["output"]),
        capital=capital,
        metadata={
            "market_id": "a_share",
            "kind": "a_share_hft",
            "window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "parameters": params,
            "account_semantics": "shared_cash_continuous_inventory_t1",
            "execution": (
                "strictly_subsequent_observed_top_of_book_no_queue_or_receive_clock"
            ),
            "corporate_action_policy": CORPORATE_ACTION_POLICY,
            "fee_schedule_identity": schedule.content_identity,
        },
        produce=produce,
    )
