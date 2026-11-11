"""Observed opening paths, liquidity recovery and realized variation."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.history import (
    path_efficiency,
    quadratic_log_trend,
)
from quant_research_platform.markets.implementations.a_share import calendar

PRICE = ("mid",)
FLOW = ("observed_trade_volume", "observed_signed_trade_volume")
BOOK = ("bid_px1", "ask_px1", "bid_vol1", "ask_vol1")
INPUTS = {
    **{f"opening_observed_return_{m}m": PRICE for m in (5, 15, 30)},
    **{f"opening_path_efficiency_{m}m": PRICE for m in (5, 15, 30)},
    **{f"opening_cumulative_flow_imbalance_{m}m": FLOW for m in (5, 15, 30)},
    "late_session_flow_imbalance": (*FLOW, "exchange", "instrument_type"),
    "early_late_pressure_shift": (*FLOW, "exchange", "instrument_type"),
    **{f"observed_spread_recovery_{s}s": BOOK for s in (30, 60, 300)},
    **{f"observed_depth_recovery_{s}s": BOOK for s in (30, 60, 300)},
    **{f"observed_sell_pressure_decay_{s}s": FLOW for s in (30, 60, 300)},
    **{
        f"event_price_impact_efficiency_{s}s": (*PRICE, *FLOW, *BOOK)
        for s in (30, 60, 300)
    },
    **{
        f"hf_{side}_realized_variance_{w}": PRICE
        for side in ("upside", "downside")
        for w in ("30m", "1d")
    },
    **{f"hf_downside_variance_fraction_{w}": PRICE for w in ("30m", "1d")},
    "daily_continuous_variation": PRICE,
    "daily_jump_variation_fraction": PRICE,
    **{
        f"price_{side}_acceleration_{w}s": PRICE
        for side in ("up", "down")
        for w in (30, 300)
    },
    **{f"price_flow_divergence_{w}s": (*PRICE, *FLOW) for w in (30, 300)},
}
UNITS = {
    name: "squared_log_return"
    if "realized_variance" in name or name == "daily_continuous_variation"
    else "decimal_return"
    if name.startswith("opening_observed_return")
    else "dimensionless"
    for name in INPUTS
}
DESCRIPTION = (
    "Causal observed-quote hypothesis; opening windows freeze at the first "
    "observed quote at/after 5/15/30 minutes, with an opening quote within 6s "
    "of 09:30 required. No cross-segment price difference. Recovery compares "
    "the current state with a shock at t-W relative to its prior-60s mean "
    "(spread >=2x or depth <=half); zero without a known shock. Variation "
    "uses observed log returns; bipower variation is pi/2 times adjacent "
    "absolute-return products, not a claim to noise-free continuous variance."
    " Acceleration fits the same quadratic-log-price kernel as daily factors "
    "on a trailing 30/300-second grid sampled every 3 seconds by causal as-of "
    "quotes with maximum age 6 seconds, within the same continuous segment. "
    "Twice curvature / RMS grid log return is positive only when slope and "
    "curvature have the named direction; a flat path is undefined. Price-flow "
    "divergence is log(mid/current-window-start-mid) times the positive part "
    "of opposite-direction signed trade-volume imbalance over that window. "
    "These are QRP hypotheses, not formulas claimed from a published paper."
)


def calculate(panel, *, columns):
    output = []
    for _, group in panel.groupby(
        [panel.asset_id, panel.timestamp.dt.normalize()], sort=False
    ):
        work = group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        times = pd.DatetimeIndex(work.timestamp)
        result = work.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
        date = (times[0] + pd.Timedelta(hours=8)).normalize()
        seconds = (times + pd.Timedelta(hours=8) - date).total_seconds().to_numpy()
        requested = set(columns)
        price_needed = any("mid" in INPUTS[name] for name in columns)
        price = work.mid.to_numpy(dtype=float) if price_needed else None
        returns = None
        if price_needed:
            returns = pd.Series(np.log(price)).diff()
            returns.loc[work.segment_id.ne(work.segment_id.shift())] = np.nan
        flow_needed = any("observed_trade_volume" in INPUTS[name] for name in columns)
        if flow_needed:
            volume = pd.Series(work.observed_trade_volume.to_numpy(float), index=times)
            signed = pd.Series(
                work.observed_signed_trade_volume.to_numpy(float), index=times
            )
        opening = 9.5 * 3600
        complete_open = 0 <= seconds[0] - opening <= 6
        opening_flow = {}
        for minutes in (5, 15, 30):
            indices = np.flatnonzero(seconds >= opening + minutes * 60)
            end = indices[0] if len(indices) else None
            visible = np.arange(len(work)) >= (end if end is not None else len(work))
            for kind in (
                "observed_return",
                "path_efficiency",
                "cumulative_flow_imbalance",
            ):
                name = f"opening_{kind}_{minutes}m"
                if name not in requested and not (
                    kind == "cumulative_flow_imbalance"
                    and minutes == 15
                    and "early_late_pressure_shift" in requested
                ):
                    continue
                value = np.nan
                if complete_open and end is not None:
                    if kind == "cumulative_flow_imbalance":
                        v, s = volume.iloc[1 : end + 1], signed.iloc[1 : end + 1]
                        if v.notna().all() and s.notna().all() and v.sum() > 0:
                            value = s.sum() / v.sum()
                        opening_flow[minutes] = value
                    elif (
                        np.isfinite(price[: end + 1]).all()
                        and (price[: end + 1] > 0).all()
                        and work.segment_id.iloc[: end + 1].nunique() == 1
                    ):
                        value = (
                            price[end] / price[0] - 1
                            if kind == "observed_return"
                            else float(
                                path_efficiency(
                                    np.log(price[end] / price[0]),
                                    np.abs(np.diff(np.log(price[: end + 1]))).sum(),
                                )
                            )
                        )
                if name in requested:
                    result[name] = np.where(visible, value, np.nan)
        if requested & {"late_session_flow_imbalance", "early_late_pressure_shift"}:
            close = calendar.continuous_session_end(
                work.timestamp, work.exchange, work.instrument_type
            ).iloc[0]
            start = ((close + pd.Timedelta(hours=8) - date).total_seconds()) - 900
            selected = seconds > start
            prefix_known = bool((seconds <= start).any())
            known = (volume.notna() & signed.notna()).where(selected, True).cummin()
            gross = volume.where(selected, 0).cumsum()
            net = signed.where(selected, 0).cumsum()
            late = (net / gross.where(gross > 0)).where(
                selected & known.to_numpy() & prefix_known
            )
            if "late_session_flow_imbalance" in requested:
                result["late_session_flow_imbalance"] = late.to_numpy()
            if "early_late_pressure_shift" in requested:
                result["early_late_pressure_shift"] = (
                    late.to_numpy() - opening_flow.get(15, np.nan)
                )
        for _, positions in work.groupby("segment_id", sort=False).indices.items():
            at = np.asarray(positions)
            segment = work.iloc[at]
            t = pd.DatetimeIndex(segment.timestamp)
            elapsed = (t - t[0]).total_seconds().to_numpy()
            for width in (30, 300):
                names = {
                    side: f"price_{side}_acceleration_{width}s"
                    for side in ("up", "down")
                }
                if requested & set(names.values()):
                    grid = t.to_numpy()[:, None] - np.arange(width, -1, -3).astype(
                        "timedelta64[s]"
                    )
                    observed = t.to_numpy()
                    indices = observed.searchsorted(grid, side="right") - 1
                    lag = np.maximum(indices, 0)
                    prices = price[at][lag]
                    known = (
                        (indices >= 0)
                        & (grid - observed[lag] <= np.timedelta64(6, "s"))
                        & np.isfinite(prices)
                        & (prices > 0)
                    ).all(axis=1)
                    slope, curvature = quadratic_log_trend(prices[known])
                    for side, direction in (("up", 1), ("down", -1)):
                        name = names[side]
                        if name in requested:
                            values = np.full(len(at), np.nan)
                            values[known] = np.where(
                                np.isfinite(curvature),
                                np.where(
                                    direction * slope > 0,
                                    np.maximum(direction * curvature, 0),
                                    0,
                                ),
                                np.nan,
                            )
                            result.loc[at, name] = values
                name = f"price_flow_divergence_{width}s"
                if name in requested:
                    index = (
                        t.searchsorted(t - pd.Timedelta(seconds=width), side="right")
                        - 1
                    )
                    lag = np.maximum(index, 0)
                    gross = volume.iloc[at].rolling(f"{width}s").sum()
                    net = signed.iloc[at].rolling(f"{width}s").sum()
                    known = (
                        (index >= 0)
                        & (
                            (t - pd.Timedelta(seconds=width) - t[lag]).total_seconds()
                            <= 6
                        )
                        & volume.iloc[at]
                        .isna()
                        .rolling(f"{width}s")
                        .sum()
                        .eq(0)
                        .to_numpy()
                        & signed.iloc[at]
                        .isna()
                        .rolling(f"{width}s")
                        .sum()
                        .eq(0)
                        .to_numpy()
                    )
                    change = np.log(price[at] / price[at][lag])
                    imbalance = (net / gross.where(gross > 0)).to_numpy()
                    result.loc[at, name] = np.where(
                        known,
                        change * np.maximum(-np.sign(change) * imbalance, 0),
                        np.nan,
                    )
            book_needed = any(set(INPUTS[name]) & set(BOOK) for name in columns)
            if book_needed:
                spread = pd.Series(
                    (segment.ask_px1 - segment.bid_px1).to_numpy(float), index=t
                )
                depth = pd.Series(
                    (segment.bid_vol1 + segment.ask_vol1).to_numpy(float), index=t
                )
                baseline_spread = (
                    spread.rolling("60s", closed="left")
                    .mean()
                    .where(
                        (elapsed >= 60)
                        & spread.isna().rolling("60s", closed="left").sum().eq(0)
                    )
                )
                baseline_depth = (
                    depth.rolling("60s", closed="left")
                    .mean()
                    .where(
                        (elapsed >= 60)
                        & depth.isna().rolling("60s", closed="left").sum().eq(0)
                    )
                )
            if flow_needed:
                v, s = volume.iloc[at], signed.iloc[at]
                v30 = v.rolling("30s").sum()
                sell = (
                    (-s.rolling("30s").sum() / v30.where(v30 > 0))
                    .clip(lower=0)
                    .where(
                        (elapsed >= 30)
                        & v.isna().rolling("30s").sum().eq(0)
                        & s.isna().rolling("30s").sum().eq(0)
                    )
                )
            for width in (30, 60, 300):
                index = (
                    t.searchsorted(t - pd.Timedelta(seconds=width), side="right") - 1
                )
                lag = np.maximum(index, 0)
                for kind in (
                    "spread_recovery",
                    "depth_recovery",
                    "sell_pressure_decay",
                    "price_impact_efficiency",
                ):
                    name = (
                        ("event_" if kind == "price_impact_efficiency" else "observed_")
                        + kind
                        + f"_{width}s"
                    )
                    if name not in requested:
                        continue
                    if kind == "spread_recovery":
                        old, normal, now = (
                            spread.to_numpy()[lag],
                            baseline_spread.to_numpy()[lag],
                            spread.to_numpy(),
                        )
                        shock = (old >= 2 * normal) & (normal > 0)
                        score = np.divide(
                            old - now,
                            old - normal,
                            out=np.full(len(at), np.nan),
                            where=shock,
                        )
                    elif kind == "depth_recovery":
                        old, normal, now = (
                            depth.to_numpy()[lag],
                            baseline_depth.to_numpy()[lag],
                            depth.to_numpy(),
                        )
                        shock = (old <= normal / 2) & (normal > old)
                        score = np.divide(
                            now - old,
                            normal - old,
                            out=np.full(len(at), np.nan),
                            where=shock,
                        )
                    elif kind == "sell_pressure_decay":
                        old, now = sell.to_numpy()[lag], sell.to_numpy()
                        normal = np.zeros(len(at))
                        shock = old > 0
                        score = np.divide(
                            old - now, old, out=np.full(len(at), np.nan), where=shock
                        )
                    else:
                        old = price[at][lag]
                        normal = depth.to_numpy()[lag]
                        now = price[at]
                        gross = (
                            v.rolling(f"{width}s")
                            .sum()
                            .where(v.isna().rolling(f"{width}s").sum().eq(0))
                            .to_numpy()
                        )
                        scale = np.divide(
                            gross,
                            normal,
                            out=np.full(len(at), np.nan),
                            where=normal > 0,
                        )
                        score = np.divide(
                            now / old - 1,
                            scale,
                            out=np.full(len(at), np.nan),
                            where=scale > 0,
                        )
                        shock = np.ones(len(at), dtype=bool)
                    known = (
                        (index >= 0)
                        & np.isfinite(old)
                        & np.isfinite(normal)
                        & np.isfinite(now)
                    )
                    result.loc[at, name] = np.where(
                        known, np.where(shock, score, 0), np.nan
                    )
        if returns is not None:
            r = pd.Series(returns.to_numpy(), index=times)
            same = work.segment_id.eq(work.segment_id.shift()).to_numpy()
            unexpected = r.isna().to_numpy() & same
            r = r.where(same, 0)
            up, down = r.clip(lower=0) ** 2, r.clip(upper=0) ** 2
            for width in ("30m", "1d"):
                if width == "30m":
                    u, d = up.rolling("30min").sum(), down.rolling("30min").sum()
                    known = ((times - times[0]).total_seconds() >= 1800) & pd.Series(
                        unexpected, index=times
                    ).rolling("30min").sum().eq(0).to_numpy()
                else:
                    u, d = up.cumsum(), down.cumsum()
                    known = (
                        complete_open
                        & ((times - times[0]).total_seconds() > 0)
                        & ~np.maximum.accumulate(unexpected)
                    )
                for side, values in (
                    ("upside_realized_variance", u),
                    ("downside_realized_variance", d),
                    ("downside_variance_fraction", d / (u + d).where(u + d > 0)),
                ):
                    name = f"hf_{side}_{width}"
                    if name in requested:
                        result[name] = values.where(known).to_numpy()
            if requested & {
                "daily_continuous_variation",
                "daily_jump_variation_fraction",
            }:
                products = (r.abs() * r.abs().shift()).where(same, 0)
                bipower = (np.pi / 2) * products.cumsum()
                variance = (r * r).cumsum()
                paired = same & np.r_[False, same[:-1]]
                known = known & np.maximum.accumulate(paired)
                if "daily_continuous_variation" in requested:
                    result["daily_continuous_variation"] = bipower.where(
                        known
                    ).to_numpy()
                if "daily_jump_variation_fraction" in requested:
                    result["daily_jump_variation_fraction"] = (
                        (
                            (variance - bipower).clip(lower=0)
                            / variance.where(variance > 0)
                        )
                        .where(known)
                        .to_numpy()
                    )
        output.append(result.loc[:, ["market_id", "asset_id", "timestamp", *columns]])
    return (
        pd.concat(output, ignore_index=True)
        if output
        else panel.loc[:, ["market_id", "asset_id", "timestamp"]].assign(
            **dict.fromkeys(columns, np.nan)
        )
    )
