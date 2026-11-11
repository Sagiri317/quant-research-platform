"""Causal intraday peers, known daily context and observed trade profiles."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.asof import latest_observations
from quant_research_platform.compute.dataframe.cross_section import leave_one_out_mean
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.markets.implementations.a_share import calendar
from quant_research_platform.research.factors.intraday_path import (
    calculate as path_factors,
)

KEY = ("market_id", "asset_id", "timestamp")
FACTORS = (
    *(f"opening_gap_repair_fraction_{m}m" for m in (5, 15, 30)),
    *(f"opening_relative_activity_{m}m" for m in (5, 15, 30)),
    "late_session_relative_activity",
    "prior_low_position_x_depth_recovery",
    "prior_drawdown_x_sell_pressure_decay",
    "prior_drawdown_x_activity_x_depth_recovery",
    "prior_residual_drawdown_x_activity_x_depth_recovery",
    "prior_financing_x_intraday_liquidity_stress",
    "prior_ownership_crowding_x_intraday_liquidity_stress",
    "intraday_positive_return_fraction",
    "intraday_positive_flow_fraction",
    "intraday_liquidity_stress_fraction",
    *(f"industry_synchronous_sell_pressure_{w}s" for w in (30, 60)),
    *(f"industry_synchronous_liquidity_stress_{w}s" for w in (30, 60)),
    *(f"lagged_industry_return_{w}s" for w in (30, 60)),
    *(f"lagged_linked_benchmark_return_{w}s" for w in (30, 60)),
    *(
        f"industry_{kind}_{w}s"
        for kind in ("return_relative", "catchup", "catchdown")
        for w in (30, 60, 300)
    ),
    "intraday_short_squeeze_pressure",
    "intraday_long_deleveraging_pressure",
    *(
        f"trade_profile_{kind}_{w}d"
        for kind in (
            "node_distance",
            "concentration",
            "above_price_fraction",
            "value_area_width",
        )
        for w in (1, 5)
    ),
)
UNITS = dict.fromkeys(FACTORS, "dimensionless")
for _name in FACTORS:
    if "return" in _name and "fraction" not in _name:
        UNITS[_name] = "decimal_return"
DAILY_FIELDS = (
    "close",
    "industry_standard_id",
    "industry_level1_code",
    "price_position_60d",
    "peak_drawdown_20d",
    "risk_residual_drawdown_20d",
    "financing_balance_to_float_cap",
    "cn_fund_holder_concentration",
    "lending_days_to_cover_20d",
    "lending_net_cover_fraction_5d",
    "financing_balance_change_5d",
    "reversal_5d",
)


def _lag(values, times, seconds):
    positions = (
        times.searchsorted(times - pd.Timedelta(seconds=seconds), side="right") - 1
    )
    return np.where(
        positions >= 0, np.asarray(values)[np.maximum(positions, 0)], np.nan
    )


def trade_profiles(quotes, trades, *, columns, session_days=None):
    output = quotes.loc[:, list(KEY)].copy()
    for name in columns:
        output[name] = np.nan
    if trades is None or trades.empty:
        return output
    trade_groups = trades.loc[trades.event_kind.isin(("trade", "unknown"))].groupby(
        "asset_id", sort=False
    )
    for asset, at in quotes.groupby("asset_id", sort=False).indices.items():
        at = np.asarray(at)
        q = quotes.iloc[at].sort_values("timestamp")
        positions = q.index.to_numpy()
        if asset not in trade_groups.indices:
            continue
        t = trade_groups.get_group(asset).sort_values(
            ["timestamp", "source_row"], kind="stable"
        )
        price = pd.to_numeric(t.price).to_numpy(float)
        quantity = pd.to_numeric(t.volume).to_numpy(float)
        valid = (
            np.isfinite(price)
            & (price > 0)
            & np.isfinite(quantity)
            & (quantity >= 0)
            & t.event_kind.eq("trade").to_numpy()
        )
        if not valid.any():
            continue
        step = np.log(1.002)
        bins = np.floor(np.log(price[valid]) / step).astype(int)
        minimum, maximum = bins.min(), bins.max()
        size = maximum - minimum + 1
        times = pd.DatetimeIndex(q.timestamp)
        events = pd.DatetimeIndex(t.timestamp)
        observed_days = pd.DatetimeIndex(times.normalize().unique())
        days = observed_days
        if session_days is not None:
            first = max(0, session_days.searchsorted(observed_days.min()) - 4)
            days = session_days[
                first : session_days.searchsorted(observed_days.max(), side="right")
            ]
        # Retain prior-day totals without assigning their trades to today's first quote.
        grid = times.union(days - pd.Timedelta(1, "ns")).sort_values()
        current_rows = grid.get_indexer(times)
        assigned = grid.searchsorted(events, side="left")
        usable = valid & (assigned < len(grid))
        contributions = np.zeros((len(grid), size))
        np.add.at(
            contributions,
            (
                assigned[usable],
                np.floor(np.log(price[usable]) / step).astype(int) - minimum,
            ),
            quantity[usable],
        )
        cumulative = contributions.cumsum(axis=0)
        bad = np.bincount(
            assigned[(~valid) & (assigned < len(grid))], minlength=len(grid)
        ).cumsum()
        for width in (1, 5):
            selected = [name for name in columns if name.endswith(f"_{width}d")]
            if not selected:
                continue
            day_positions = days.get_indexer(times.normalize())
            start = days[np.maximum(day_positions - width + 1, 0)]
            before = grid.searchsorted(start, side="left") - 1
            history = cumulative[current_rows] - np.where(
                (before >= 0)[:, None], cumulative[np.maximum(before, 0)], 0
            )
            unknown = bad[current_rows] - np.where(
                before >= 0, bad[np.maximum(before, 0)], 0
            )
            total = history.sum(axis=1)
            known = (day_positions >= width - 1) & (unknown == 0) & (total > 0)
            if width == 5:
                trade_days = pd.DatetimeIndex(events[valid].normalize().unique())
                complete = (
                    pd.Series(days.isin(trade_days), index=days)
                    .rolling(5, min_periods=5)
                    .sum()
                    .eq(5)
                )
                known &= complete.reindex(times.normalize()).to_numpy()
            node = history.argmax(axis=1)
            center = np.exp((minimum + node + 0.5) * step)
            current = q.mid.to_numpy(float)
            for name in selected:
                if "_node_distance_" in name:
                    value = np.log(current / center)
                elif "_concentration_" in name:
                    value = history.max(axis=1) / np.where(total > 0, total, np.nan)
                elif "_above_price_fraction_" in name:
                    edges = np.exp((minimum + np.arange(size + 1)) * step)
                    above = np.clip(
                        (edges[1:] - current[:, None]) / np.diff(edges), 0, 1
                    )
                    value = (history * above).sum(axis=1) / np.where(
                        total > 0, total, np.nan
                    )
                else:
                    left = node.copy()
                    right = node.copy()
                    mass = history[np.arange(len(q)), node].copy()
                    for _ in range(size - 1):
                        active = mass < 0.7 * total
                        if not active.any():
                            break
                        lv = np.where(
                            left > 0,
                            history[np.arange(len(q)), np.maximum(left - 1, 0)],
                            -1,
                        )
                        rv = np.where(
                            right < size - 1,
                            history[np.arange(len(q)), np.minimum(right + 1, size - 1)],
                            -1,
                        )
                        choose_left = (lv >= rv) & active
                        choose_right = (~choose_left) & active
                        left -= choose_left
                        right += choose_right
                        mass += np.where(choose_left, lv, np.where(choose_right, rv, 0))
                    value = (right - left + 1) * step
                output.loc[positions, name] = np.where(known, value, np.nan)
    return output


def calculate(
    panel,
    *,
    columns,
    daily_context=None,
    trades=None,
    linked_panel=None,
    linked_assets=None,
    session_days=None,
):
    work = panel.sort_values(["asset_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    result = work.loc[:, list(KEY)].copy()
    for name in columns:
        result[name] = np.nan
    if work.empty:
        return result
    if all(name.startswith("trade_profile_") for name in columns):
        return trade_profiles(work, trades, columns=columns, session_days=session_days)
    day = work.timestamp.dt.normalize()
    daily = pd.DataFrame(np.nan, index=work.index, columns=DAILY_FIELDS)
    if daily_context is not None:
        fields = tuple(name for name in DAILY_FIELDS if name in daily_context)
        if fields:
            facts = daily_context.rename(
                columns={"timestamp": "available_at", "signal_session": "timestamp"}
            )
            query = work
            daily[list(fields)] = latest_observations(facts, query, columns=fields)[
                list(fields)
            ].to_numpy()
    returns = {w: pd.Series(np.nan, index=work.index) for w in (30, 60, 300)}
    flows = {w: pd.Series(np.nan, index=work.index) for w in (30, 60)}
    stress = pd.Series(np.nan, index=work.index)
    activities = {}
    relative_activity = pd.Series(np.nan, index=work.index)
    late_samples = []
    for (asset, date), at in work.groupby(
        [work.asset_id, day], sort=False
    ).indices.items():
        at = np.asarray(at)
        block = work.iloc[at]
        times = pd.DatetimeIndex(block.timestamp)
        opening = date + pd.Timedelta(hours=1, minutes=30)
        complete = 0 <= (times[0] - opening).total_seconds() <= 6
        volume = block.observed_trade_volume
        for minutes in (5, 15, 30):
            ends = np.flatnonzero(times >= opening + pd.Timedelta(minutes=minutes))
            if not len(ends):
                continue
            end = ends[0]
            visible = np.arange(len(block)) >= end
            gap = block.mid.iloc[0] / pd.to_numeric(daily.loc[at, "close"]).iloc[0] - 1
            name = f"opening_gap_repair_fraction_{minutes}m"
            if name in columns and complete:
                observed = block.mid.iloc[end] / block.mid.iloc[0] - 1
                result.loc[at, name] = np.where(
                    visible, -observed / gap if gap != 0 else np.nan, np.nan
                )
            window = volume.iloc[1 : end + 1]
            activities[(asset, date, minutes)] = (
                window.sum(min_count=len(window)) if complete else np.nan
            )
        if "late_session_relative_activity" in columns:
            close = calendar.continuous_session_end(
                block.timestamp, block.exchange, block.instrument_type
            ).iloc[0]
            start = close - pd.Timedelta(minutes=15)
            selected = times > start
            prefix_known = bool((times <= start).any())
            known = volume.notna().where(selected, True).cummin()
            late = (
                volume.where(selected, 0)
                .cumsum()
                .where(selected & known & prefix_known)
            )
            slots = np.floor((times - times.normalize()).total_seconds() / 3).astype(
                int
            )
            if selected.any():
                late_samples.append(
                    pd.DataFrame(
                        {
                            "asset_id": asset,
                            "day": date,
                            "slot": slots,
                            "volume": late.to_numpy(),
                            "row": at,
                        }
                    ).loc[selected]
                )
    for asset, at in work.groupby("asset_id", sort=False).indices.items():
        observed = pd.DatetimeIndex(day.iloc[at].unique())
        dates = (
            observed
            if session_days is None
            else session_days[
                (session_days >= observed.min()) & (session_days <= observed.max())
            ]
        )
        for window in (5, 15, 30):
            amounts = pd.Series(
                [activities.get((asset, date, window), np.nan) for date in dates],
                index=dates,
            )
            baseline = amounts.shift().rolling(20, min_periods=20).mean()
            name = f"opening_relative_activity_{window}m"
            if name not in columns or session_days is None:
                continue
            for date in dates:
                positions = np.asarray(at)[day.iloc[at].eq(date)]
                start = date + pd.Timedelta(hours=1, minutes=30 + window)
                if baseline.loc[date] > 0:
                    result.loc[positions, name] = np.where(
                        work.timestamp.iloc[positions].ge(start),
                        amounts.loc[date] / baseline.loc[date],
                        np.nan,
                    )
    if late_samples and session_days is not None:
        samples = pd.concat(late_samples, ignore_index=True)
        history = samples.groupby(["asset_id", "slot", "day"], sort=False).volume.last()
        matrix = history.unstack("day")
        matrix = matrix.reindex(
            columns=session_days[
                (session_days >= matrix.columns.min())
                & (session_days <= matrix.columns.max())
            ]
        )
        matrix.columns.name = "day"
        baseline = (
            matrix.T.shift()
            .rolling(20, min_periods=20)
            .mean()
            .T.stack()
            .rename("baseline")
            .reset_index()
        )
        reference = samples.merge(
            baseline, on=["asset_id", "slot", "day"], how="left", validate="many_to_one"
        )
        ratio = reference.volume / reference.baseline.where(reference.baseline > 0)
        result.loc[reference.row, "late_session_relative_activity"] = ratio.to_numpy()
    if all(
        name.startswith("opening_relative_activity_")
        or name == "late_session_relative_activity"
        for name in columns
    ):
        return result
    stress_windows = {w: pd.Series(np.nan, index=work.index) for w in (30, 60)}
    for _, at in work.groupby(
        [work.asset_id, day, work.segment_id], sort=False
    ).indices.items():
        at = np.asarray(at)
        block = work.iloc[at]
        times = pd.DatetimeIndex(block.timestamp)
        for width in (30, 60, 300):
            returns[width].loc[at] = (
                block.mid.to_numpy() / _lag(block.mid, times, width) - 1
            )
        volume = pd.Series(block.observed_trade_volume.to_numpy(), index=times)
        signed = pd.Series(block.observed_signed_trade_volume.to_numpy(), index=times)
        if any("_activity_x_depth_recovery" in name for name in columns):
            baseline = (
                volume.rolling("300s")
                .sum()
                .where(volume.isna().rolling("300s").sum().eq(0))
            )
            baseline = _lag(baseline, times, 30)
            gross = (
                volume.rolling("30s")
                .sum()
                .where(volume.isna().rolling("30s").sum().eq(0))
                .to_numpy()
            )
            relative_activity.loc[at] = np.where(
                (times - times[0]).total_seconds() >= 330,
                10 * gross / np.where(baseline > 0, baseline, np.nan),
                np.nan,
            )
        for width in (30, 60):
            gross = volume.rolling(f"{width}s").sum()
            net = signed.rolling(f"{width}s").sum()
            complete = (
                volume.isna().astype(int).rolling(f"{width}s").sum().eq(0)
                & signed.isna().astype(int).rolling(f"{width}s").sum().eq(0)
                & ((times - times[0]).total_seconds() >= width)
            )
            flows[width].loc[at] = (
                (net / gross.where(gross > 0)).where(complete).to_numpy()
            )
        spread = pd.Series((block.ask_px1 - block.bid_px1).to_numpy(), index=times)
        depth = pd.Series((block.bid_vol1 + block.ask_vol1).to_numpy(), index=times)
        normal_spread = spread.rolling("60s", closed="left").mean()
        normal_depth = depth.rolling("60s", closed="left").mean()
        spread_known = (
            (normal_spread > 0)
            & spread.notna()
            & spread.isna().rolling("60s", closed="left").sum().eq(0)
        )
        depth_known = (
            (normal_depth > 0)
            & depth.notna()
            & depth.isna().rolling("60s", closed="left").sum().eq(0)
        )
        spread_shock = spread > 2 * normal_spread
        depth_shock = depth < normal_depth / 2
        known = (
            (spread_known & depth_known)
            | (spread_known & spread_shock)
            | (depth_known & depth_shock)
        ) & ((times - times[0]).total_seconds() >= 60)
        stress.loc[at] = (
            ((spread_known & spread_shock) | (depth_known & depth_shock))
            .astype(float)
            .where(known)
            .to_numpy()
        )
        for width in (30, 60):
            history = pd.Series(stress.loc[at].to_numpy(), index=times)
            complete = history.isna().astype(int).rolling(f"{width}s").sum().eq(0) & (
                (times - times[0]).total_seconds() >= 60 + width
            )
            stress_windows[width].loc[at] = (
                history.rolling(f"{width}s").mean().where(complete).to_numpy()
            )
    cohort = work.timestamp
    peer_keys = pd.MultiIndex.from_arrays(
        [cohort, daily.industry_standard_id, daily.industry_level1_code]
    )
    peer_ids = pd.Series(pd.factorize(peer_keys)[0], index=work.index)
    valid_industry = (
        daily.industry_standard_id.notna() & daily.industry_level1_code.notna()
    )
    for width in (30, 60, 300):
        own = returns[width]
        peers = leave_one_out_mean(own.where(valid_industry), peer_ids)
        for kind, value in (
            ("return_relative", own - peers),
            ("catchup", (peers - own).clip(lower=0).where(peers > 0, 0)),
            ("catchdown", (own - peers).clip(lower=0).where(peers < 0, 0)),
        ):
            name = f"industry_{kind}_{width}s"
            if name in columns:
                result[name] = value.where(own.notna() & peers.notna() & valid_industry)
        if width in (30, 60):
            for kind, values in (
                ("sell_pressure", (-flows[width]).clip(lower=0)),
                ("liquidity_stress", stress_windows[width]),
            ):
                name = f"industry_synchronous_{kind}_{width}s"
                if name in columns:
                    result[name] = pd.Series(
                        leave_one_out_mean(values.where(valid_industry), peer_ids),
                        index=work.index,
                    ).where(valid_industry)
            name = f"lagged_industry_return_{width}s"
            if name in columns:
                series = pd.Series(peers, index=work.index)
                for _, at in work.groupby(
                    [work.asset_id, day, work.segment_id], sort=False
                ).indices.items():
                    result.loc[at, name] = _lag(
                        series.iloc[at],
                        pd.DatetimeIndex(work.timestamp.iloc[at]),
                        width,
                    )
    for name, values in (
        ("intraday_positive_return_fraction", returns[30]),
        ("intraday_positive_flow_fraction", flows[30]),
        ("intraday_liquidity_stress_fraction", stress),
    ):
        if name in columns:
            result[name] = (
                values.gt(0)
                .astype(float)
                .where(values.notna())
                .groupby(cohort)
                .transform("mean")
            )
    if linked_panel is not None:
        for width in (30, 60):
            name = f"lagged_linked_benchmark_return_{width}s"
            if name not in columns:
                continue
            frames = []
            for (asset, _), g in linked_panel.groupby(
                ["asset_id", "segment_id"], sort=False
            ):
                g = g.sort_values("timestamp")
                t = pd.DatetimeIndex(g.timestamp)
                recent = g.mid.to_numpy() / _lag(g.mid, t, width) - 1
                frames.append(
                    pd.DataFrame(
                        {
                            "asset_id": asset,
                            "timestamp": t,
                            "available_at": t,
                            "value": _lag(recent, t, width),
                        }
                    )
                )
            if frames:
                query = work.assign(asset_id=work.asset_id.map(linked_assets or {}))
                result[name] = latest_observations(
                    pd.concat(frames, ignore_index=True), query, columns=("value",)
                ).value.to_numpy()
    interactions = {
        "prior_low_position_x_depth_recovery": (
            "observed_depth_recovery_60s",
            "price_position_60d",
        ),
        "prior_drawdown_x_sell_pressure_decay": (
            "observed_sell_pressure_decay_60s",
            "peak_drawdown_20d",
        ),
        "prior_drawdown_x_activity_x_depth_recovery": (
            "observed_depth_recovery_60s",
            "peak_drawdown_20d",
        ),
        "prior_residual_drawdown_x_activity_x_depth_recovery": (
            "observed_depth_recovery_60s",
            "risk_residual_drawdown_20d",
        ),
    }
    paths = tuple(
        dict.fromkeys(
            path for name, (path, _) in interactions.items() if name in columns
        )
    )
    if paths:
        aligned = (
            path_factors(work, columns=paths)
            .set_index(["asset_id", "timestamp"])
            .reindex(pd.MultiIndex.from_frame(work[["asset_id", "timestamp"]]))
        )
    for name, (path, context) in interactions.items():
        if name in columns:
            state = pd.to_numeric(daily[context])
            state = 1 - state if "low_position" in name else state
            result[name] = aligned[path].to_numpy() * state
            if "_activity_x_depth_recovery" in name:
                result[name] *= relative_activity
    if "prior_financing_x_intraday_liquidity_stress" in columns:
        result["prior_financing_x_intraday_liquidity_stress"] = (
            pd.to_numeric(daily.financing_balance_to_float_cap) * stress
        )
    if "prior_ownership_crowding_x_intraday_liquidity_stress" in columns:
        result["prior_ownership_crowding_x_intraday_liquidity_stress"] = (
            pd.to_numeric(daily.cn_fund_holder_concentration) * stress
        )
    if "intraday_short_squeeze_pressure" in columns:
        result["intraday_short_squeeze_pressure"] = (
            pd.to_numeric(daily.lending_days_to_cover_20d)
            * pd.to_numeric(daily.lending_net_cover_fraction_5d)
            * returns[60].clip(lower=0)
            * flows[60].clip(lower=0)
        )
    if "intraday_long_deleveraging_pressure" in columns:
        result["intraday_long_deleveraging_pressure"] = (
            (-pd.to_numeric(daily.financing_balance_change_5d)).clip(lower=0)
            * (-returns[60]).clip(lower=0)
            * (-flows[60]).clip(lower=0)
        )
    profiles = tuple(name for name in columns if name.startswith("trade_profile_"))
    if profiles:
        profile = trade_profiles(
            work, trades, columns=profiles, session_days=session_days
        )
        result[list(profiles)] = profile[list(profiles)].to_numpy()
    return result


def definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="standard",
            input_columns=("mid",),
            output_column=name,
            family="intraday_context",
            required_input_data_id="hft_research_factors",
            description=(
                "Observed intraday hypothesis on canonical UTC clocks. Peer "
                "returns exclude self and use contemporaneously known industry; "
                "no future-event or receive-time claim. Historical trade profiles"
                " allocate recorded trade volume to fixed 20bp log-price bins; "
                "the value area expands from the point of control to cover 70%. "
                "Nominal trade prices are not investor cost basis. Only explicit "
                "historical data and known prior daily context are consumed."
                " Drawdown/activity/depth-recovery interactions multiply the "
                "known prior 20-session raw or risk-residual drawdown by "
                "10 * volume(t-30,t] / volume(t-330,t-30] and the observed "
                "60-second depth recovery. A missing residual background affects "
                "only the residual interaction. These are QRP hypotheses."
            ),
            timing_semantics="known_before_quote_time",
            null_policy="missing_only_invalidates_actual_dependencies",
        )
        for name in FACTORS
    )
