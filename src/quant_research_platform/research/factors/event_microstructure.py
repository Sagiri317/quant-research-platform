"""Causal statistics of recorded trades and explicitly linked order observations."""

from collections import deque

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.level2 import (
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


def trade_statistics(trades, config):
    work = (
        trades.loc[trades.event_kind.eq("trade")]
        .sort_values(["timestamp", "source_row"], kind="stable")
        .reset_index(drop=True)
    )
    window, history = config["event_window"], config["history_window"]
    side = work.side.map({"buy": 1.0, "sell": -1.0})
    volume = work.volume.where(work.volume.gt(0))
    signed = side * volume
    interval = work.timestamp.diff().dt.total_seconds()
    mean = interval.rolling(window, min_periods=window).mean()
    elapsed = (work.timestamp - work.timestamp.shift(window - 1)).dt.total_seconds()
    result = pd.DataFrame(
        {
            "trade_interval_seconds": interval,
            "trade_interval_mean_seconds": mean,
            "trade_interval_cv": interval.rolling(window, min_periods=window).std(
                ddof=1
            )
            / mean.where(mean.gt(0)),
            "trade_intensity": (window - 1) / elapsed.where(elapsed.gt(0)),
            "trade_activity_ratio": interval.shift(1)
            .rolling(history, min_periods=history)
            .mean()
            / mean.where(mean.gt(0)),
            "trade_event_flow_imbalance": signed.rolling(
                window, min_periods=window
            ).sum()
            / volume.rolling(window, min_periods=window).sum(),
            "trade_interval_return_interaction": (
                work.price / work.price.shift(window - 1) - 1
            )
            * mean,
        }
    )
    observed = side.notna() & volume.notna()
    run = (
        side.ne(side.shift()) | ~observed | ~observed.shift(fill_value=False)
    ).cumsum()
    result["trade_signed_run"] = ((side.groupby(run).cumcount() + 1) * side).where(
        observed
    )
    result["trade_signed_run_volume"] = (volume.groupby(run).cumsum() * side).where(
        observed
    )
    for name, quantile, large in (
        ("large_trade_pressure", config["large_quantile"], True),
        ("small_trade_pressure", 1 - config["large_quantile"], False),
    ):
        threshold = (
            volume.shift(1).rolling(history, min_periods=history).quantile(quantile)
        )
        selected = volume.ge(threshold) if large else volume.le(threshold)
        pressure = signed.where(selected, 0).where(threshold.notna() & signed.notna())
        result[name] = (
            pressure.rolling(window, min_periods=window).sum()
            / volume.rolling(window, min_periods=window).sum()
        )
    amounts = volume.to_numpy()
    cumulative = np.r_[0.0, np.nancumsum(amounts)]
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
    output = []
    if link_field is None:
        return pd.DataFrame(np.nan, index=range(len(quotes)), columns=names)
    active, quantity, birth_quantity = {}, np.zeros(2), np.zeros(2)
    cancelled, cancel_pressure, completed, matched = (
        deque(maxlen=window) for _ in range(4)
    )
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
        valid = np.isfinite(amount) and 0 < amount <= remaining and when > born
        matched.append(int(valid))
        if not valid:
            if cancel:
                cancelled.append(np.nan)
                cancel_pressure.append(np.nan)
            return
        age = when - born
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

    for quote in quotes.itertuples(index=False):
        when = (quote.timestamp - origin).total_seconds()
        while pending is not None and pending[1].timestamp <= quote.timestamp:
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
                    and np.isfinite(row.volume)
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
        ages = np.divide(
            when * quantity - birth_quantity,
            quantity,
            out=np.full(2, np.nan),
            where=quantity > 0,
        )
        total = quantity.sum()
        output.append(
            (
                (when * total - birth_quantity.sum()) / total if total > 0 else np.nan,
                ages[0] - ages[1],
                np.mean(cancelled) if cancelled else np.nan,
                np.sum(cancel_pressure) if cancel_pressure else np.nan,
                np.mean(completed) if completed else np.nan,
                np.mean(matched) if matched else np.nan,
            )
        )
    return pd.DataFrame(output, columns=names)


def compute_event_factors(
    panel, orders, trades, *, columns, config, missing_streams=()
):
    config = parameters(config)
    output = []
    for work, selected in observed_segments(panel, orders=orders, trades=trades):
        event_times, values = trade_statistics(selected["trades"], config)
        aligned = _at_quotes(values, event_times, work.timestamp)
        additions = selected["orders"].loc[selected["orders"].event_kind.eq("order")]
        cancel_source = (
            selected["orders"] if work.exchange.iloc[0] == "SSE" else selected["trades"]
        )
        cancellations = cancel_source.loc[cancel_source.event_kind.eq("cancel")]
        messages = (
            pd.concat(
                [
                    additions[["timestamp", "source_row", "volume"]].assign(
                        cancel=False, stream_priority=1
                    ),
                    cancellations[["timestamp", "source_row", "volume"]].assign(
                        cancel=True,
                        stream_priority=1 if work.exchange.iloc[0] == "SSE" else 0,
                    ),
                ]
            )
            .sort_values(["timestamp", "stream_priority", "source_row"], kind="stable")
            .reset_index(drop=True)
        )
        n = config["event_window"]
        cancel_volume = (
            messages.volume.where(messages.cancel, 0).rolling(n, min_periods=n).sum()
        )
        order_volume = (
            messages.volume.where(~messages.cancel, 0).rolling(n, min_periods=n).sum()
        )
        ratio = cancel_volume / order_volume.where(order_volume.gt(0))
        cancellation_values = pd.DataFrame(
            {
                "observed_cancel_order_ratio": ratio,
                "observed_cancel_count_fraction": messages.cancel.astype(float)
                .rolling(n, min_periods=n)
                .mean(),
                "observed_cancel_pressure_change": ratio.diff(),
            }
        )
        aligned = pd.concat(
            [
                aligned,
                _at_quotes(cancellation_values, messages.timestamp, work.timestamp),
                lifecycle_features(
                    work,
                    selected["orders"],
                    selected["trades"],
                    link_field=config["order_link_field"],
                    window=n,
                ),
            ],
            axis=1,
        )
        for column in columns:
            required = (
                {"hft_trades"}
                if column.startswith("trade_")
                or column in {"large_trade_pressure", "small_trade_pressure"}
                else {"hft_orders"}
                if column
                in {
                    "observed_cancel_order_ratio",
                    "observed_cancel_count_fraction",
                    "observed_cancel_pressure_change",
                }
                and work.exchange.iloc[0] == "SSE"
                else {"hft_orders", "hft_trades"}
            )
            if required.intersection(missing_streams):
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
