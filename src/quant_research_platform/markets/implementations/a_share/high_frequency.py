"""Observed A-share quote panels, causal factors and within-session labels."""

import json
from contextlib import contextmanager

import numpy as np
import pandas as pd

from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.research.protocols import a_share_hft as protocol
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def continuous_session(timestamps, exchange):
    """Shanghai introduced the closing auction on 2018-08-20."""
    local = (
        pd.to_datetime(timestamps).dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")
    )
    seconds = (
        local.dt.hour * 3600
        + local.dt.minute * 60
        + local.dt.second
        + local.dt.microsecond / 1e6
    )
    closing = np.where(
        exchange.eq("SSE") & local.dt.date.lt(pd.Timestamp("2018-08-20").date()),
        15 * 3600,
        14 * 3600 + 57 * 60,
    )
    return pd.Series(
        np.select(
            [
                seconds.ge(9 * 3600 + 30 * 60) & seconds.lt(11 * 3600 + 30 * 60),
                seconds.ge(13 * 3600) & seconds.lt(closing),
            ],
            ["am", "pm"],
            default="",
        ),
        index=timestamps.index,
    )


def _flow(quotes, events, kind):
    """Recorded interval sums after the first event; no feed-completeness claim."""
    total = np.full(len(quotes), np.nan)
    signed = total.copy()
    if events.empty or len(quotes) < 2:
        return total, signed
    times = quotes.timestamp.to_numpy(dtype="datetime64[ns]")
    event_times = events.timestamp.to_numpy(dtype="datetime64[ns]")
    selected = events.event_kind.eq(kind).to_numpy()
    bins = np.searchsorted(times, event_times[selected], side="left")
    selected_events = events.loc[selected]
    within = (bins > 0) & (bins < len(quotes))
    bins = bins[within]
    quantities = selected_events.volume.to_numpy(dtype="float64")[within]
    sides = selected_events.side.map({"buy": 1.0, "sell": -1.0}).to_numpy(
        dtype="float64"
    )[within]
    counts = np.bincount(bins, weights=np.isnan(quantities), minlength=len(quotes))
    unknown_side = np.bincount(bins, weights=np.isnan(sides), minlength=len(quotes))
    amount = np.bincount(bins, weights=np.nan_to_num(quantities), minlength=len(quotes))
    direction = np.bincount(
        bins, weights=np.nan_to_num(quantities * sides), minlength=len(quotes)
    )
    covered = times[:-1] >= event_times.min()
    total[1:] = np.where(covered & (counts[1:] == 0), amount[1:], np.nan)
    signed[1:] = np.where(
        covered & (counts[1:] == 0) & (unknown_side[1:] == 0), direction[1:], np.nan
    )
    return total, signed


def build_panel(quotes, orders, trades, *, max_gap_seconds):
    quotes = (
        quotes.sort_values(["asset_id", "timestamp", "source_row"], kind="stable")
        .drop_duplicates(["asset_id", "timestamp"], keep="last")
        .copy()
    )
    quotes["session"] = continuous_session(quotes.timestamp, quotes.exchange)
    quotes = quotes.loc[quotes.session.ne("")]
    output = []
    for (asset, day, session), group in quotes.groupby(
        ["asset_id", "session_date", "session"], sort=True
    ):
        work = group.reset_index(drop=True).copy()
        work["market_id"] = "a_share"
        work["last_price"] = work.price
        valid = (
            work.bid_px1.gt(0)
            & work.ask_px1.gt(work.bid_px1)
            & work.bid_vol1.gt(0)
            & work.ask_vol1.gt(0)
        )
        work["eligible"] = valid
        work["mid"] = ((work.bid_px1 + work.ask_px1) / 2).where(valid)
        breaks = (
            work.timestamp.diff().dt.total_seconds().gt(max_gap_seconds)
            | ~valid
            | ~valid.shift(1, fill_value=False)
        )
        work["segment_id"] = (
            f"{asset}:{day:%Y%m%d}:{session}:" + breaks.cumsum().astype(str)
        )
        for kind, source in (
            ("trade", trades),
            ("order", orders),
            ("cancel", orders if work.exchange.iloc[0] == "SSE" else trades),
        ):
            selected = source.loc[
                source.asset_id.eq(asset) & source.session_date.eq(day)
            ]
            selected = selected.loc[
                continuous_session(selected.timestamp, selected.exchange).eq(session)
            ]
            total, signed = _flow(work, selected, kind)
            work[f"observed_{kind}_volume"] = total
            work[f"observed_signed_{kind}_volume"] = signed
        output.append(work.loc[:, protocol.PANEL_COLUMNS])
    if not output:
        return pd.DataFrame(columns=protocol.PANEL_COLUMNS)
    return pd.concat(output, ignore_index=True).sort_values(
        ["timestamp", "asset_id"], kind="stable"
    )


def _trailing_sum(values, timestamps, seconds):
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(timestamps))
    rolling = series.rolling(f"{seconds}s", closed="right")
    elapsed = timestamps - timestamps.iloc[0]
    complete = (
        series.isna().astype(int).rolling(f"{seconds}s", closed="right").sum().eq(0)
    )
    result = rolling.sum().where(complete).to_numpy(copy=True)
    result[elapsed.lt(pd.Timedelta(seconds=seconds))] = np.nan
    return result


def compute_factors(panel):
    output = []
    for _, group in panel.groupby(["asset_id", "segment_id"], sort=False):
        work = group.sort_values("timestamp").reset_index(drop=True)
        values = work.loc[:, protocol.KEY].copy()
        spread = work.ask_px1 - work.bid_px1
        values["spread_cny"] = spread
        values["relative_spread"] = spread / work.mid
        for level in (1, 5, 10):
            depth = {}
            for side in ("bid", "ask"):
                quantity = work[
                    [f"{side}_vol{i}" for i in range(1, level + 1)]
                ].to_numpy(dtype=float)
                price = work[[f"{side}_px{i}" for i in range(1, level + 1)]].to_numpy(
                    dtype=float
                )
                depth[side] = np.where(
                    (quantity == 0) | np.isfinite(price), quantity, np.nan
                ).sum(axis=1)
            denominator = depth["bid"] + depth["ask"]
            values[f"imbalance_{level}"] = np.divide(
                depth["bid"] - depth["ask"],
                denominator,
                out=np.full(len(work), np.nan),
                where=denominator > 0,
            )
        depth1 = work.bid_vol1 + work.ask_vol1
        values["microprice_deviation"] = (
            (work.ask_px1 * work.bid_vol1 + work.bid_px1 * work.ask_vol1) / depth1
            - work.mid
        ) / work.mid
        previous = work.shift(1)
        ofi = (
            work.bid_vol1.where(work.bid_px1.ge(previous.bid_px1), 0)
            - previous.bid_vol1.where(work.bid_px1.le(previous.bid_px1), 0)
            - work.ask_vol1.where(work.ask_px1.le(previous.ask_px1), 0)
            + previous.ask_vol1.where(work.ask_px1.ge(previous.ask_px1), 0)
        )
        ofi.iloc[0] = np.nan
        values["book_ofi"] = ofi
        values["depth_normalized_ofi"] = ofi / depth1
        times = work.timestamp.to_numpy(dtype="datetime64[ns]")
        mid = work.mid.to_numpy(dtype=float)
        for seconds in (3, 30, 60):
            previous_index = (
                np.searchsorted(
                    times, times - np.timedelta64(seconds, "s"), side="right"
                )
                - 1
            )
            values[f"return_{seconds}s"] = np.where(
                previous_index >= 0,
                mid / mid[np.maximum(previous_index, 0)] - 1,
                np.nan,
            )
        log_returns = np.log(work.mid).diff()
        values["realized_volatility_60s"] = np.sqrt(
            _trailing_sum(log_returns**2, work.timestamp, 60)
        )
        for kind in ("trade", "order", "cancel"):
            denominator = work[f"observed_{kind}_volume"]
            values[f"observed_{kind}_flow_imbalance"] = work[
                f"observed_signed_{kind}_volume"
            ] / denominator.where(denominator.gt(0))
        for column in ("observed_trade_volume", "observed_signed_trade_volume"):
            values[f"{column}_30s"] = _trailing_sum(work[column], work.timestamp, 30)
        values["book_ofi_30s"] = _trailing_sum(ofi, work.timestamp, 30)
        values.loc[~work.eligible, list(protocol.FACTOR_INPUTS)] = np.nan
        output.append(values)
    return (
        pd.concat(output, ignore_index=True)
        .loc[:, (*protocol.KEY, *protocol.FACTOR_INPUTS)]
        .sort_values(["timestamp", "asset_id"], kind="stable")
    )


def compute_labels(panel, definitions):
    output = []
    for _, group in panel.groupby(["asset_id", "segment_id"], sort=False):
        work = group.sort_values("timestamp").reset_index(drop=True)
        values = work.loc[:, protocol.KEY].copy()
        times = work.timestamp.to_numpy(dtype="datetime64[ns]")
        mid = work.mid.to_numpy(dtype=float)
        for definition in definitions:
            future = np.searchsorted(
                times, times + np.timedelta64(definition.horizon, "ms")
            )
            valid = (future < len(times)) & work.eligible.to_numpy()
            future = np.minimum(future, len(times) - 1)
            returns = mid[future] / mid - 1
            valid &= np.isfinite(returns)
            values[definition.output_column] = np.where(
                valid,
                np.sign(returns)
                if definition.family == "future_mid_direction"
                else returns,
                np.nan,
            )
            values[definition.available_at_column] = pd.to_datetime(
                np.where(valid, times[future], np.datetime64("NaT", "ns"))
            )
        output.append(values)
    return pd.concat(output, ignore_index=True).sort_values(
        ["timestamp", "asset_id"], kind="stable"
    )


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    sessions = {
        name: open_dataset_session(**_dataset_input_options(path))
        for name, path in inputs.items()
    }
    for name, session in sessions.items():
        if (
            session.spec["data_id"] != name
            or session.manifest.get("market_id", "a_share") != "a_share"
        ):
            raise ValueError("hft_input_identity_mismatch")
    metadata = {
        "producer_diagnostics": {
            "timestamp_semantics": "vendor_event_time_no_receive_clock",
        },
    }
    definitions = [
        item
        for item in protocol.label_catalog().list()
        if item.output_column in spec.label_availability_columns
    ]
    if definitions:
        metadata["producer_diagnostics"]["label_semantics"] = {
            item.output_column: item.to_manifest() for item in definitions
        }
    lower, upper, closed = time_window_timestamp_interval(time_window)

    def frames():
        for day in pd.date_range(
            pd.Timestamp(lower).normalize(), pd.Timestamp(upper).normalize(), freq="D"
        ):
            window = TimeWindow(day.date(), day.date())
            data = {
                name: materialize_dataset_session(session.filtered_view(window))
                for name, session in sessions.items()
            }
            if spec.data_id == "hft_panel":
                frame = build_panel(
                    data["hft_quotes"],
                    data["hft_orders"],
                    data["hft_trades"],
                    **json.loads(spec.parameters_json),
                )
            elif data["hft_panel"].empty:
                continue
            elif spec.data_id == "hft_factors":
                frame = compute_factors(data["hft_panel"])
            else:
                frame = compute_labels(data["hft_panel"], definitions)
            if frame.empty:
                continue
            mask = frame.timestamp.ge(pd.Timestamp(lower)) & (
                frame.timestamp.le(pd.Timestamp(upper))
                if closed
                else frame.timestamp.lt(pd.Timestamp(upper))
            )
            yield frame.loc[mask, spec.required_columns].reset_index(drop=True)

    yield frames(), metadata
