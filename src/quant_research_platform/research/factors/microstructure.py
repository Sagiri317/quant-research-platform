"""Causal microstructure factors on observed quote segments."""

import numpy as np
import pandas as pd

from quant_research_platform.research.protocols import a_share_hft as protocol


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


def compute_factors(panel, *, columns=None, ofi_halflife_seconds=30):
    columns = protocol.DEFAULT_FACTORS if columns is None else tuple(columns)
    requested = set(columns)
    needed = requested.copy()
    if "trade_activity_ratio_30_300s" in requested:
        needed.add("observed_trade_volume_30s")
    if requested & {"flow_absorption_30s", "drawdown_sell_pressure_300s"}:
        needed.update(("observed_trade_volume_30s", "observed_signed_trade_volume_30s"))
    if "flow_absorption_30s" in requested:
        needed.add("return_30s")
    if "compression_order_flow_300s" in requested:
        needed.add("book_ofi_30s")
    output = []
    for _, group in panel.groupby(["asset_id", "segment_id"], sort=False):
        work = group.sort_values("timestamp").reset_index(drop=True)
        values = work.loc[:, protocol.KEY].copy()
        if needed & {"spread_cny", "relative_spread"}:
            spread = (work.ask_px1 - work.bid_px1).where(
                work.bid_px1.gt(0) & work.ask_px1.gt(work.bid_px1)
            )
            if "spread_cny" in needed:
                values["spread_cny"] = spread
            if "relative_spread" in needed:
                values["relative_spread"] = spread / work.mid
        for level in (1, 5, 10):
            if f"imbalance_{level}" not in needed:
                continue
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
        if "microprice_deviation" in needed:
            depth1 = work.bid_vol1 + work.ask_vol1
            values["microprice_deviation"] = (
                (work.ask_px1 * work.bid_vol1 + work.bid_px1 * work.ask_vol1) / depth1
                - work.mid
            ) / work.mid
        if needed & {"book_ofi", "depth_normalized_ofi", "book_ofi_30s"}:
            previous = work[["bid_px1", "ask_px1", "bid_vol1", "ask_vol1"]].shift(1)
            ofi = (
                work.bid_vol1.where(work.bid_px1.ge(previous.bid_px1), 0)
                - previous.bid_vol1.where(work.bid_px1.le(previous.bid_px1), 0)
                - work.ask_vol1.where(work.ask_px1.le(previous.ask_px1), 0)
                + previous.ask_vol1.where(work.ask_px1.ge(previous.ask_px1), 0)
            )
            prices = ["bid_px1", "ask_px1"]
            ofi = ofi.where(
                work[prices].gt(0).all(axis=1) & previous[prices].gt(0).all(axis=1)
            )
            if "book_ofi" in needed:
                values["book_ofi"] = ofi
            if "depth_normalized_ofi" in needed:
                depth = work.bid_vol1 + work.ask_vol1
                values["depth_normalized_ofi"] = ofi / depth.where(depth.gt(0))
            if "book_ofi_30s" in needed:
                values["book_ofi_30s"] = _trailing_sum(ofi, work.timestamp, 30)
        for seconds in (3, 30, 60):
            if f"return_{seconds}s" in needed:
                times = work.timestamp.to_numpy(dtype="datetime64[ns]")
                mid = work.mid.to_numpy(dtype=float)
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
        if "realized_volatility_60s" in needed:
            values["realized_volatility_60s"] = np.sqrt(
                _trailing_sum(np.log(work.mid).diff() ** 2, work.timestamp, 60)
            )
        for kind in ("trade", "order", "cancel"):
            name = f"observed_{kind}_flow_imbalance"
            if name in needed:
                denominator = work[f"observed_{kind}_volume"]
                values[name] = work[
                    f"observed_signed_{kind}_volume"
                ] / denominator.where(denominator.gt(0))
        for column in ("observed_trade_volume", "observed_signed_trade_volume"):
            if f"{column}_30s" in needed:
                values[f"{column}_30s"] = _trailing_sum(
                    work[column], work.timestamp, 30
                )
        extended = (
            requested
            - set(protocol.DEFAULT_FACTORS)
            - protocol.INTRADAY_THEME_FACTORS.keys()
        )
        if extended:
            values = pd.concat(
                [values, _book_features(work, ofi_halflife_seconds, extended)], axis=1
            )
        themes = requested & protocol.INTRADAY_THEME_FACTORS.keys()
        if themes:
            values = pd.concat([values, _intraday_themes(work, values, themes)], axis=1)
        output.append(values.loc[:, (*protocol.KEY, *columns)])
    return (
        pd.concat(output, ignore_index=True).sort_values(
            ["timestamp", "asset_id"], kind="stable"
        )
        if output
        else pd.DataFrame(columns=(*protocol.KEY, *columns))
    )


def _intraday_themes(work, base, columns):
    """Price-path and absorption hypotheses use only their declared inputs."""
    times = pd.DatetimeIndex(work.timestamp)
    result = pd.DataFrame(index=times)
    price_columns = set(columns) - {"trade_activity_ratio_30_300s"}
    if price_columns:
        mid = pd.Series(work.mid.to_numpy(), index=times)
    extrema = {
        "price_position_300s",
        "peak_drawdown_300s",
        "support_distance_300s",
        "resistance_distance_300s",
        "drawdown_sell_pressure_300s",
        "failed_high_break_300s",
        "failed_low_break_300s",
    }
    if columns & extrema:
        previous = mid.rolling("300s", closed="left")
        complete = mid.isna().astype(int).rolling("300s", closed="left").sum().eq(0) & (
            (times - times[0]).total_seconds() >= 300
        )
        high, low = previous.max().where(complete), previous.min().where(complete)
        if "price_position_300s" in columns:
            result["price_position_300s"] = (mid - low) / (high - low).where(high > low)
        if columns & {"peak_drawdown_300s", "drawdown_sell_pressure_300s"}:
            drawdown = (1 - mid / high).clip(lower=0)
            if "peak_drawdown_300s" in columns:
                result["peak_drawdown_300s"] = drawdown
        if "support_distance_300s" in columns:
            result["support_distance_300s"] = mid / low - 1
        if "resistance_distance_300s" in columns:
            result["resistance_distance_300s"] = mid / high - 1
    if columns & {
        "volatility_compression_30_300s",
        "compression_order_flow_300s",
        "flow_absorption_30s",
        "signed_trend_efficiency_300s",
    }:
        log_returns = np.log(work.mid).diff()
    if columns & {
        "volatility_compression_30_300s",
        "compression_order_flow_300s",
        "flow_absorption_30s",
    }:
        variance = _trailing_sum(log_returns**2, work.timestamp, 300)
    if columns & {"volatility_compression_30_300s", "compression_order_flow_300s"}:
        short_variance = _trailing_sum(log_returns**2, work.timestamp, 30)
        compression = np.sqrt(
            np.divide(
                10 * short_variance,
                variance,
                out=np.full(len(work), np.nan),
                where=variance > 0,
            )
        )
        if "volatility_compression_30_300s" in columns:
            result["volatility_compression_30_300s"] = compression
    if "signed_trend_efficiency_300s" in columns:
        path = _trailing_sum(log_returns.abs(), work.timestamp, 300)
        trend = _trailing_sum(log_returns, work.timestamp, 300)
        result["signed_trend_efficiency_300s"] = np.divide(
            trend, path, out=np.full(len(work), np.nan), where=path > 0
        )
    if columns & {
        "trade_activity_ratio_30_300s",
        "flow_absorption_30s",
        "drawdown_sell_pressure_300s",
    }:
        volume = base.observed_trade_volume_30s.to_numpy()
    if "trade_activity_ratio_30_300s" in columns:
        long_volume = _trailing_sum(work.observed_trade_volume, work.timestamp, 300)
        result["trade_activity_ratio_30_300s"] = np.divide(
            10 * volume,
            long_volume,
            out=np.full(len(work), np.nan),
            where=long_volume > 0,
        )
    if columns & {"flow_absorption_30s", "drawdown_sell_pressure_300s"}:
        imbalance = np.divide(
            base.observed_signed_trade_volume_30s.to_numpy(),
            volume,
            out=np.full(len(work), np.nan),
            where=volume > 0,
        )
        if "flow_absorption_30s" in columns:
            impact = np.divide(
                np.abs(base.return_30s.to_numpy()),
                np.sqrt(variance / 10),
                out=np.full(len(work), np.nan),
                where=variance > 0,
            )
            result["flow_absorption_30s"] = imbalance * (1 - impact)
        if "drawdown_sell_pressure_300s" in columns:
            result["drawdown_sell_pressure_300s"] = drawdown * np.maximum(-imbalance, 0)
    if "compression_order_flow_300s" in columns:
        depth = (work.bid_vol1 + work.ask_vol1).to_numpy()
        result["compression_order_flow_300s"] = np.maximum(
            1 - compression, 0
        ) * np.divide(
            base.book_ofi_30s.to_numpy(),
            depth,
            out=np.full(len(work), np.nan),
            where=depth > 0,
        )
    if columns & {"failed_high_break_300s", "failed_low_break_300s"}:
        indices = (
            times.to_numpy().searchsorted(
                times.to_numpy() - np.timedelta64(30, "s"), side="right"
            )
            - 1
        )
        past = np.maximum(indices, 0)
        reference_high, reference_low = high.to_numpy()[past], low.to_numpy()[past]
        recent_high, recent_low = mid.rolling("30s").max(), mid.rolling("30s").min()
        recent_known = mid.isna().astype(int).rolling("30s").sum().eq(0)
        for name, extreme, level, returned in (
            (
                "failed_high_break_300s",
                recent_high,
                reference_high,
                mid <= reference_high,
            ),
            ("failed_low_break_300s", recent_low, reference_low, mid >= reference_low),
        ):
            if name in columns:
                excess = (
                    extreme / level - 1
                    if name.startswith("failed_high")
                    else 1 - extreme / level
                )
                result[name] = (
                    excess.clip(lower=0)
                    .where(returned, 0)
                    .where(
                        np.isfinite(level) & (indices >= 0) & recent_known & mid.notna()
                    )
                )
    return result.reset_index(drop=True)


def _book_features(work, halflife, columns):
    """Snapshot OFI proxies and depth shape at actual price distances."""
    flow, shapes = {}, {}
    flow_names = {name for name in columns if "ofi" in name}
    flow_levels = 10 if flow_names - {"book_ofi_5", "depth_normalized_ofi_5"} else 5
    for side, sign in (("bid", 1), ("ask", -1)):
        selected = {name for name in columns if name.startswith(side + "_")}
        slopes = bool(
            selected & {f"{side}_depth_slope", f"{side}_depth_slope_change"}
            or "depth_slope_difference" in columns
        )
        if not flow_names and not selected and not slopes:
            continue
        levels = 10 if selected or slopes else flow_levels
        prices = work[[f"{side}_px{i}" for i in range(1, levels + 1)]].to_numpy(float)
        quantity = work[[f"{side}_vol{i}" for i in range(1, levels + 1)]].to_numpy(
            float
        )
        prior_price = np.vstack((np.full(levels, np.nan), prices[:-1]))
        prior_quantity = np.vstack((np.full(levels, np.nan), quantity[:-1]))
        valid = (
            np.isfinite(prices)
            & np.isfinite(prior_price)
            & np.isfinite(quantity)
            & np.isfinite(prior_quantity)
        )
        if flow_names:
            flow[side] = np.where(
                valid,
                np.where(sign * prices >= sign * prior_price, quantity, 0)
                - np.where(sign * prices <= sign * prior_price, prior_quantity, 0),
                np.nan,
            )[:, :flow_levels]
        known = np.isfinite(quantity).all(axis=1) & (
            (quantity == 0) | np.isfinite(prices)
        ).all(axis=1)
        if selected or slopes:
            total = quantity.sum(axis=1)
            denominator = np.where(known & (total > 0), total, np.nan)
            fraction = quantity / denominator[:, None]
            if f"{side}_depth_concentration" in columns:
                shapes[f"{side}_depth_concentration"] = (fraction**2).sum(axis=1)
            if f"{side}_near_depth_fraction" in columns:
                shapes[f"{side}_near_depth_fraction"] = fraction[:, :3].sum(axis=1)
        if slopes:
            present = quantity > 0
            distance = (
                sign
                * (work.mid.to_numpy()[:, None] - prices)
                / work.mid.to_numpy()[:, None]
                * 10000
            )
            count = present.sum(axis=1)
            count_safe = np.maximum(count, 1)
            cumulative = fraction.cumsum(axis=1)
            x, y = np.where(present, distance, 0), np.where(present, cumulative, 0)
            variance = (x * x).sum(axis=1) - x.sum(axis=1) ** 2 / count_safe
            covariance = (x * y).sum(axis=1) - x.sum(axis=1) * y.sum(
                axis=1
            ) / count_safe
            slope = np.divide(
                covariance,
                variance,
                out=np.full(len(work), np.nan),
                where=known & (count >= 2) & (variance > 0),
            )
            shapes[f"{side}_depth_slope"] = slope
            if f"{side}_depth_slope_change" in columns:
                shapes[f"{side}_depth_slope_change"] = np.r_[np.nan, np.diff(slope)]
        if f"{side}_replenishment_30s" in columns:
            replenished = np.where(
                valid & (prices == prior_price),
                np.maximum(quantity - prior_quantity, 0),
                0,
            ).sum(axis=1)
            replenished[~valid.all(axis=1)] = np.nan
            shapes[f"{side}_replenishment_30s"] = _trailing_sum(
                pd.Series(replenished), work.timestamp, 30
            )
    if flow_names:
        level_flow = flow["bid"] - flow["ask"]
        for level in (5, 10):
            current = level_flow[:, :level].sum(axis=1)
            if f"book_ofi_{level}" in columns:
                shapes[f"book_ofi_{level}"] = current
            if f"depth_normalized_ofi_{level}" in columns:
                depth = work[
                    [
                        f"{side}_vol{i}"
                        for side in ("bid", "ask")
                        for i in range(1, level + 1)
                    ]
                ].sum(axis=1, skipna=False)
                shapes[f"depth_normalized_ofi_{level}"] = current / depth.where(
                    depth.gt(0)
                )
        if "book_ofi_near_far" in columns:
            shapes["book_ofi_near_far"] = level_flow[:, :5].sum(axis=1) - level_flow[
                :, 5:
            ].sum(axis=1)
        total = pd.Series(level_flow.sum(axis=1))
        if "book_ofi_10_30s" in columns:
            shapes["book_ofi_10_30s"] = _trailing_sum(total, work.timestamp, 30)
        if "book_ofi_10_ewma" in columns:
            shapes["book_ofi_10_ewma"] = total.ewm(
                halflife=pd.Timedelta(seconds=halflife),
                times=work.timestamp,
                min_periods=1,
            ).mean()
    if "depth_slope_difference" in columns:
        shapes["depth_slope_difference"] = (
            shapes["bid_depth_slope"] - shapes["ask_depth_slope"]
        )
    return pd.DataFrame(shapes).loc[:, list(columns)]
