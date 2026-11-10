"""Causal statistics of recorded trades and explicitly linked order observations."""

from math import isfinite

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.level2 import (
    first_unknown_event_time,
    iter_level2_events,
    observed_segments,
    order_reference,
)
from quant_research_platform.research.protocols import a_share_hft as protocol


def parameters(values):
    defaults = {
        "event_window": 50,
        "volume_window": 10000,
        "history_window": 200,
        "large_quantile": 0.9,
        "order_link_field": None,
    }
    if set(values) - defaults.keys():
        raise ValueError("event_factor_parameter_not_consumed")
    result = {**defaults, **values}
    if (
        any(
            type(result[name]) is not int or result[name] < 2
            for name in ("event_window", "history_window")
        )
        or type(result["volume_window"]) not in (int, float)
        or not 0 < result["volume_window"] < float("inf")
        or type(result["large_quantile"]) not in (int, float)
        or not 0.5 < result["large_quantile"] < 1
        or result["order_link_field"] not in {None, "order_id", "ex_order_id"}
    ):
        raise ValueError("event_factor_parameters_invalid")
    return result


def _at_quotes(values, event_times, quote_times):
    if len(event_times) == 0:
        return pd.DataFrame(
            np.nan, index=range(len(quote_times)), columns=values.columns
        )
    index = (
        event_times.to_numpy(dtype="datetime64[ns]").searchsorted(
            quote_times.to_numpy(dtype="datetime64[ns]"), side="right"
        )
        - 1
    )
    result = values.iloc[np.maximum(index, 0)].reset_index(drop=True)
    result.loc[index < 0] = np.nan
    return result


def trade_statistics(trades, config, *, columns=None):
    """Summarize the timestamp/source-row order established by observed_segments."""
    columns = protocol.TRADE_EVENT_FACTORS if columns is None else set(columns)
    work = trades.loc[trades.event_kind.isin(("trade", "unknown"))].reset_index(
        drop=True
    )
    window, history = config["event_window"], config["history_window"]
    known = work.event_kind.eq("trade")
    result = pd.DataFrame(index=work.index)
    interval_columns = protocol.TRADE_INTERVAL_FACTORS
    if columns & interval_columns:
        interval = (
            work.timestamp.diff()
            .dt.total_seconds()
            .where(known & known.shift(fill_value=False))
        )
        if "trade_interval_seconds" in columns:
            result["trade_interval_seconds"] = interval
        if columns & (interval_columns - {"trade_interval_seconds"}):
            mean = interval.rolling(window, min_periods=window).mean()
            if "trade_interval_mean_seconds" in columns:
                result["trade_interval_mean_seconds"] = mean
            if "trade_interval_cv" in columns:
                result["trade_interval_cv"] = interval.rolling(
                    window, min_periods=window
                ).std(ddof=1) / mean.where(mean.gt(0))
            if "trade_activity_ratio" in columns:
                result["trade_activity_ratio"] = interval.shift(1).rolling(
                    history, min_periods=history
                ).mean() / mean.where(mean.gt(0))
            if "trade_interval_return_interaction" in columns:
                price = work.price.where(known)
                result["trade_interval_return_interaction"] = (
                    price / price.shift(window - 1) - 1
                ) * mean
    if "trade_intensity" in columns:
        elapsed = (work.timestamp - work.timestamp.shift(window - 1)).dt.total_seconds()
        result["trade_intensity"] = ((window - 1) / elapsed.where(elapsed.gt(0))).where(
            known.rolling(window).sum().eq(window)
        )
    flow_columns = columns - interval_columns - {"trade_intensity"}
    if flow_columns:
        side = work.side.map({"buy": 1.0, "sell": -1.0}).astype(float).where(known)
        volume = work.volume.where(work.volume.gt(0) & known)
        signed = side * volume
        if columns & {
            "trade_event_flow_imbalance",
            "large_trade_pressure",
            "small_trade_pressure",
        }:
            window_volume = volume.rolling(window, min_periods=window).sum()
        if "trade_event_flow_imbalance" in columns:
            result["trade_event_flow_imbalance"] = (
                signed.rolling(window, min_periods=window).sum() / window_volume
            )
        if columns & {"trade_signed_run", "trade_signed_run_volume"}:
            observed = side.notna() & volume.notna()
            run = (
                side.ne(side.shift()) | ~observed | ~observed.shift(fill_value=False)
            ).cumsum()
            if "trade_signed_run" in columns:
                result["trade_signed_run"] = (
                    (side.groupby(run).cumcount() + 1) * side
                ).where(observed)
            if "trade_signed_run_volume" in columns:
                result["trade_signed_run_volume"] = (
                    volume.groupby(run).cumsum() * side
                ).where(observed)
        for name, quantile, large in (
            ("large_trade_pressure", config["large_quantile"], True),
            ("small_trade_pressure", 1 - config["large_quantile"], False),
        ):
            if name in columns:
                threshold = (
                    volume.shift(1)
                    .rolling(history, min_periods=history)
                    .quantile(quantile)
                )
                selected = volume.ge(threshold) if large else volume.le(threshold)
                pressure = signed.where(selected, 0).where(
                    threshold.notna() & signed.notna()
                )
                result[name] = (
                    pressure.rolling(window, min_periods=window).sum() / window_volume
                )
        if "trade_volume_flow_imbalance" in columns:
            cumulative = np.r_[0.0, np.nancumsum(volume.to_numpy())]
            signed_sum = np.r_[0.0, np.nancumsum(signed.to_numpy())]
            bad = np.r_[0, np.cumsum(~np.isfinite(signed.to_numpy()))]
            target = config["volume_window"]
            starts = np.maximum(0, cumulative[1:] - target)
            first = np.minimum(
                np.searchsorted(cumulative, starts, side="right") - 1, len(work) - 1
            )
            if len(work):
                pressure = (
                    signed_sum[1:]
                    - signed_sum[first]
                    - (starts - cumulative[first]) * side.to_numpy()[first]
                )
                valid = (cumulative[1:] >= target) & ((bad[1:] - bad[first]) == 0)
                result["trade_volume_flow_imbalance"] = np.where(
                    valid, pressure / target, np.nan
                )
            else:
                result["trade_volume_flow_imbalance"] = pd.Series(dtype=float)
    result.loc[~known] = np.nan
    return work.timestamp, result


def lifecycle_features(quotes, orders, trades, *, link_field, window):
    """Track observed remainders; full lifetimes require an original Shenzhen order."""
    names = (
        "observed_order_age_seconds",
        "observed_order_age_side_difference",
        "observed_cancel_age_seconds",
        "observed_age_weighted_cancel_pressure",
        "observed_completed_lifetime_seconds",
        "observed_lifecycle_match_fraction",
    )
    output_index = range(len(quotes))
    if link_field is None:
        return pd.DataFrame(np.nan, index=output_index, columns=names)
    unknown_at = first_unknown_event_time(orders, trades)
    if pd.notna(unknown_at):
        quotes = quotes.loc[quotes.timestamp.lt(unknown_at)]
    if quotes.empty:
        return pd.DataFrame(np.nan, index=output_index, columns=names)
    active, quantity, birth_quantity = {}, np.zeros(2), np.zeros(2)
    cancelled, cancel_pressure, completed, matched = histories = ([], [], [], [])
    quantities = np.empty((len(quotes), 2))
    births = np.empty_like(quantities)
    positions = np.empty((len(quotes), 4), dtype=np.intp)
    events = iter(iter_level2_events(orders, trades))
    pending = next(events, None)
    origin = quotes.timestamp.iloc[0]

    def remove(key):
        born, remaining, side, original = active.pop(key)
        quantity[side] -= remaining
        birth_quantity[side] -= born * remaining
        return born, remaining, side, original

    def consume(key, amount, when, cancel):
        if key not in active:
            matched.append(0)
            if cancel:
                cancelled.append(np.nan)
                cancel_pressure.append(np.nan)
            return
        born, remaining, side, original = remove(key)
        valid = isfinite(amount) and 0 < amount <= remaining
        matched.append(int(valid))
        if not valid:
            if cancel:
                cancelled.append(np.nan)
                cancel_pressure.append(np.nan)
            return
        age = when - born if when > born else np.nan
        if cancel:
            cancelled.append(age)
            cancel_pressure.append((1 if side == 0 else -1) * amount * age)
        remaining -= amount
        if remaining:
            active[key] = born, remaining, side, original
            quantity[side] += remaining
            birth_quantity[side] += born * remaining
        elif original:
            completed.append(age)

    for index, timestamp in enumerate(quotes.timestamp):
        while pending is not None and pending[1].timestamp <= timestamp:
            stream, row = pending
            event_time = (row.timestamp - origin).total_seconds()
            if row.event_kind == "order":
                key = order_reference(getattr(row, link_field))
                original = row.exchange == "SZSE" and row.order_type in {"0", "2"}
                if key in active:
                    remove(key)
                    original = False
                if (
                    key is not None
                    and row.side in {"buy", "sell"}
                    and isfinite(row.volume)
                    and row.volume > 0
                ):
                    side = 0 if row.side == "buy" else 1
                    active[key] = event_time, row.volume, side, original
                    quantity[side] += row.volume
                    birth_quantity[side] += event_time * row.volume
            elif row.event_kind == "trade":
                for field in ("bid_order_id", "ask_order_id"):
                    consume(
                        order_reference(getattr(row, field)),
                        row.volume,
                        event_time,
                        False,
                    )
            elif row.event_kind == "cancel":
                if stream == "order":
                    key = order_reference(getattr(row, link_field))
                else:
                    references = [
                        order_reference(getattr(row, field))
                        for field in ("bid_order_id", "ask_order_id")
                    ]
                    nonzero = [key for key in references if key is not None]
                    key = nonzero[0] if len(nonzero) == 1 else None
                consume(key, row.volume, event_time, True)
            pending = next(events, None)
        quantities[index] = quantity
        births[index] = birth_quantity
        positions[index] = [len(values) - 1 for values in histories]
    when = (quotes.timestamp - origin).dt.total_seconds().to_numpy()
    ages = np.divide(
        when[:, None] * quantities - births,
        quantities,
        out=np.full_like(quantities, np.nan),
        where=quantities > 0,
    )
    total = quantities.sum(axis=1)
    result = pd.DataFrame(
        {
            names[0]: np.divide(
                when * total - births.sum(axis=1),
                total,
                out=np.full(len(quotes), np.nan),
                where=total > 0,
            ),
            names[1]: ages[:, 0] - ages[:, 1],
        }
    )
    for column, values, index in zip(names[2:], histories, positions.T, strict=True):
        if not values:
            result[column] = np.nan
            continue
        rolling = pd.Series(values).rolling(window, min_periods=1)
        size = np.minimum(np.arange(1, len(values) + 1), window)
        summed = rolling.sum().where(rolling.count().eq(size)).to_numpy()
        if column != "observed_age_weighted_cancel_pressure":
            summed = summed / size
        result[column] = np.where(index >= 0, summed[np.maximum(index, 0)], np.nan)
    return result.reindex(output_index)


def compute_event_factors(
    panel, orders, trades, *, columns, config, missing_streams=()
):
    config = parameters(config)
    selected_columns = set(columns)
    trade_columns = selected_columns & protocol.TRADE_EVENT_FACTORS
    cancel_columns = selected_columns & protocol.CANCEL_EVENT_FACTORS
    lifecycle_columns = selected_columns & protocol.LIFECYCLE_FACTORS
    linked_lifecycle = lifecycle_columns and config["order_link_field"] is not None
    output = []
    streams = {}
    needs_cancel_trades = bool(cancel_columns) and panel.exchange.eq("SZSE").any()
    if trade_columns or needs_cancel_trades or linked_lifecycle:
        streams["trades"] = trades
    if cancel_columns or linked_lifecycle:
        streams["orders"] = orders
    for work, selected in observed_segments(panel, **streams):
        aligned = pd.DataFrame(index=range(len(work)))
        if trade_columns:
            event_times, values = trade_statistics(
                selected["trades"], config, columns=trade_columns
            )
            aligned = _at_quotes(values, event_times, work.timestamp)
        if cancel_columns:
            additions = selected["orders"]
            cancellations = (
                selected["trades"].loc[
                    selected["trades"].event_kind.isin(("cancel", "unknown"))
                ]
                if work.exchange.iloc[0] == "SZSE"
                else additions.iloc[:0]
            )
            message_columns = ["timestamp", "source_row"]
            if cancel_columns != {"observed_cancel_count_fraction"}:
                message_columns.append("volume")
            messages = (
                pd.concat(
                    [
                        additions[message_columns].assign(
                            cancel=additions.event_kind.map(
                                {"order": 0.0, "cancel": 1.0}
                            ),
                            stream_priority=1,
                        ),
                        cancellations[message_columns].assign(
                            cancel=cancellations.event_kind.map({"cancel": 1.0}),
                            stream_priority=1 if work.exchange.iloc[0] == "SSE" else 0,
                        ),
                    ]
                )
                .sort_values(
                    ["timestamp", "stream_priority", "source_row"], kind="stable"
                )
                .reset_index(drop=True)
            )
            n = config["event_window"]
            cancellation_values = pd.DataFrame(index=messages.index)
            if cancel_columns & {
                "observed_cancel_order_ratio",
                "observed_cancel_pressure_change",
            }:
                cancel_volume = (
                    messages.volume.where(messages.cancel.eq(1), 0)
                    .where(messages.cancel.notna())
                    .rolling(n, min_periods=n)
                    .sum()
                )
                order_volume = (
                    messages.volume.where(messages.cancel.eq(0), 0)
                    .where(messages.cancel.notna())
                    .rolling(n, min_periods=n)
                    .sum()
                )
                ratio = cancel_volume / order_volume.where(order_volume.gt(0))
                if "observed_cancel_order_ratio" in cancel_columns:
                    cancellation_values["observed_cancel_order_ratio"] = ratio
                if "observed_cancel_pressure_change" in cancel_columns:
                    cancellation_values["observed_cancel_pressure_change"] = (
                        ratio.diff()
                    )
            if "observed_cancel_count_fraction" in cancel_columns:
                cancellation_values["observed_cancel_count_fraction"] = (
                    messages.cancel.astype(float).rolling(n, min_periods=n).mean()
                )
            aligned = pd.concat(
                [
                    aligned,
                    _at_quotes(cancellation_values, messages.timestamp, work.timestamp),
                ],
                axis=1,
            )
        if lifecycle_columns:
            aligned = pd.concat(
                [
                    aligned,
                    lifecycle_features(
                        work,
                        selected.get("orders"),
                        selected.get("trades"),
                        link_field=config["order_link_field"],
                        window=config["event_window"],
                    ).loc[:, list(lifecycle_columns)],
                ],
                axis=1,
            )
        for column in columns:
            if protocol.event_factor_streams(
                column, work.exchange.iloc[0]
            ).intersection(missing_streams):
                aligned[column] = np.nan
        output.append(
            pd.concat([work.loc[:, protocol.KEY], aligned.loc[:, columns]], axis=1)
        )
    return (
        pd.concat(output, ignore_index=True).sort_values(
            ["timestamp", "asset_id"], kind="stable"
        )
        if output
        else pd.DataFrame(columns=(*protocol.KEY, *columns))
    )
