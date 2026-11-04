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
        if set(columns) - set(protocol.DEFAULT_FACTORS):
            extended = _book_features(work, ofi_halflife_seconds)
            values = pd.concat([values, extended], axis=1)
        values.loc[~work.eligible, list(columns)] = np.nan
        output.append(values)
    return (
        pd.concat(output, ignore_index=True)
        .loc[:, (*protocol.KEY, *columns)]
        .sort_values(["timestamp", "asset_id"], kind="stable")
    )


def _book_features(work, halflife):
    """Snapshot OFI proxies and depth shape measured at actual price distances."""
    flow, shapes = {}, {}
    for side, sign in (("bid", 1), ("ask", -1)):
        prices = work[[f"{side}_px{i}" for i in range(1, 11)]].to_numpy(float)
        quantity = work[[f"{side}_vol{i}" for i in range(1, 11)]].to_numpy(float)
        prior_price = np.vstack((np.full(10, np.nan), prices[:-1]))
        prior_quantity = np.vstack((np.full(10, np.nan), quantity[:-1]))
        valid = (
            np.isfinite(prices)
            & np.isfinite(prior_price)
            & np.isfinite(quantity)
            & np.isfinite(prior_quantity)
        )
        flow[side] = np.where(
            valid,
            np.where(sign * prices >= sign * prior_price, quantity, 0)
            - np.where(sign * prices <= sign * prior_price, prior_quantity, 0),
            np.nan,
        )
        known = np.isfinite(quantity).all(axis=1) & (
            (quantity == 0) | np.isfinite(prices)
        ).all(axis=1)
        total = quantity.sum(axis=1)
        denominator = np.where(known & (total > 0), total, np.nan)
        fraction = quantity / denominator[:, None]
        shapes[f"{side}_depth_concentration"] = (fraction**2).sum(axis=1)
        shapes[f"{side}_near_depth_fraction"] = fraction[:, :3].sum(axis=1)
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
        x = np.where(present, distance, 0)
        y = np.where(present, cumulative, 0)
        variance = (x * x).sum(axis=1) - x.sum(axis=1) ** 2 / count_safe
        covariance = (x * y).sum(axis=1) - x.sum(axis=1) * y.sum(axis=1) / count_safe
        slope = np.divide(
            covariance,
            variance,
            out=np.full(len(work), np.nan),
            where=known & (count >= 2) & (variance > 0),
        )
        shapes[f"{side}_depth_slope"] = slope
        shapes[f"{side}_depth_slope_change"] = np.r_[np.nan, np.diff(slope)]
        # Only volume increases at unchanged displayed prices count as this proxy.
        replenished = np.where(
            valid & (prices == prior_price), np.maximum(quantity - prior_quantity, 0), 0
        ).sum(axis=1)
        replenished[~valid.all(axis=1)] = np.nan
        shapes[f"{side}_replenishment_30s"] = _trailing_sum(
            pd.Series(replenished), work.timestamp, 30
        )
    level_flow = flow["bid"] - flow["ask"]
    for level in (5, 10):
        current = level_flow[:, :level].sum(axis=1)
        shapes[f"book_ofi_{level}"] = current
        depth = work[
            [f"{side}_vol{i}" for side in ("bid", "ask") for i in range(1, level + 1)]
        ].sum(axis=1, skipna=False)
        shapes[f"depth_normalized_ofi_{level}"] = current / depth.where(depth.gt(0))
    shapes["book_ofi_near_far"] = level_flow[:, :5].sum(axis=1) - level_flow[:, 5:].sum(
        axis=1
    )
    total = pd.Series(shapes["book_ofi_10"])
    shapes["book_ofi_10_30s"] = _trailing_sum(total, work.timestamp, 30)
    shapes["book_ofi_10_ewma"] = total.ewm(
        halflife=pd.Timedelta(seconds=halflife), times=work.timestamp, min_periods=1
    ).mean()
    shapes["depth_slope_difference"] = (
        shapes["bid_depth_slope"] - shapes["ask_depth_slope"]
    )
    return pd.DataFrame(shapes)
