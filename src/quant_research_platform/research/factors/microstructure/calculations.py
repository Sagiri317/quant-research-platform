"""Generic microstructure factor math."""

from __future__ import annotations

import pandas as pd


def calculate_microstructure_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the approved 1m microstructure factors using rolling 5-row sums."""

    result = frame.sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)
    best_bid = _numeric(result, "best_bid")
    best_ask = _numeric(result, "best_ask")
    mid_price = _numeric(result, "mid_price")
    bid_quantity = _numeric(result, "bid_quantity")
    ask_quantity = _numeric(result, "ask_quantity")
    bid_depth = _numeric(result, "bid_depth")
    ask_depth = _numeric(result, "ask_depth")

    spread = best_ask - best_bid
    result["spread"] = spread
    result["relative_spread"] = (spread / mid_price).mask(mid_price.le(0))

    depth_denominator = bid_depth + ask_depth
    result["depth_imbalance"] = (
        (bid_depth - ask_depth) / depth_denominator
    ).mask(depth_denominator.le(0))

    quote_denominator = bid_quantity + ask_quantity
    microprice = (
        (best_ask * bid_quantity + best_bid * ask_quantity) / quote_denominator
    ).mask(quote_denominator.le(0))
    result["microprice"] = microprice
    result["microprice_deviation"] = (microprice / mid_price - 1.0).mask(
        mid_price.le(0)
    )

    trade_count_sum = _rolling_group_sum(result, _numeric(result, "trade_count"))
    signed_volume_sum = _rolling_group_sum(result, _numeric(result, "signed_volume"))
    volume_sum = _rolling_group_sum(result, _numeric(result, "volume"))
    result["trade_intensity_5m"] = trade_count_sum
    result["signed_volume_5m"] = signed_volume_sum
    result["flow_imbalance_5m"] = (signed_volume_sum / volume_sum).mask(
        volume_sum.le(0)
    )
    return result


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _rolling_group_sum(frame: pd.DataFrame, series: pd.Series) -> pd.Series:
    return (
        series.groupby([frame["market_id"], frame["asset_id"]], sort=False)
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .sort_index()
    )


__all__ = ["calculate_microstructure_factors"]
