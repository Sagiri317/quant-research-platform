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
