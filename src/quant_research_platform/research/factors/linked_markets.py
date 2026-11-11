"""Explicit economic links, foreign trading activity and index-futures states."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.asof import (
    latest_observations,
    rolling_available_at,
)
from quant_research_platform.contracts.factors import FactorDefinition

FUTURES = (
    "index_future_near_basis",
    "index_future_basis_change_1d",
    "index_future_basis_change_5d",
    "index_future_calendar_spread",
    "index_future_annualized_basis",
    "index_future_oi_change_1d",
    "index_future_oi_change_5d",
    "index_future_volume_to_oi",
    "index_future_roll_concentration",
    "basis_shock_x_hedging_exposure",
    "oi_shock_x_hedging_exposure",
)
PAIRS = tuple(
    f"linked_pair_{kind}_{w}d"
    for kind in ("relative_return", "beta_adjusted_spread", "spread_zscore")
    for w in (20, 60)
)
US = (
    "us_short_sale_activity_fraction",
    "us_short_sale_activity_change",
    "us_short_sale_activity_percentile",
    "us_sector_etf_return_shock_1d",
    "us_sector_etf_return_shock_5d",
    "us_sector_own_beta",
    "us_sector_shock_x_own_beta",
    "us_relative_sector_shock",
    "us_relative_sector_shock_x_own_beta",
)
RECIPES = {
    **dict.fromkeys(
        (
            f"index_group_{kind}_{w}d"
            for kind in ("positive_fraction", "leader_gap")
            for w in (5, 20)
        ),
        ("index_market", "trading_calendar"),
    ),
    "implied_minus_past_realized_variance_20d": (),
    "implied_minus_past_realized_variance_60d": (),
    "index_positive_participation_5d": (
        "market_panel_1d",
        "index_membership",
        "index_weights",
    ),
    "index_positive_participation_20d": (
        "market_panel_1d",
        "index_membership",
        "index_weights",
    ),
    "index_positive_return_concentration": (
        "market_panel_1d",
        "index_membership",
        "index_weights",
    ),
    **dict.fromkeys(
        FUTURES, ("public_futures_daily", "public_futures_contracts", "index_market")
    ),
    "index_future_calendar_spread": (
        "public_futures_daily",
        "public_futures_contracts",
    ),
    **dict.fromkeys((*FUTURES[5:9], FUTURES[10]), ("public_futures_daily",)),
    **dict.fromkeys(PAIRS, ("market_panel_1d",)),
    **dict.fromkeys(US[:3], ("public_finra_short_volume",)),
    **dict.fromkeys(US[3:], ("public_etf_prices",)),
}
UNITS = dict.fromkeys(RECIPES, "dimensionless")
for _width in (5, 20):
    UNITS[f"index_group_leader_gap_{_width}d"] = "decimal_return"
for _width in (20, 60):
    UNITS[f"implied_minus_past_realized_variance_{_width}d"] = "annualized_variance"
for _name in (
    "index_future_near_basis",
    "index_future_basis_change_1d",
    "index_future_basis_change_5d",
    "index_future_calendar_spread",
    *PAIRS[:2],
    *US[3:5],
    "us_sector_shock_x_own_beta",
    "us_relative_sector_shock",
    "us_relative_sector_shock_x_own_beta",
):
    UNITS[_name] = "decimal_return"


def definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="standard",
            input_columns=dependencies
            or (
                f"implied_variance_{name.split('_')[-1]}",
                f"volatility_{name.split('_')[-1]}",
            ),
            output_column=name,
            family="linked_markets",
            required_input_data_id="linked_market_factors",
            description=(
                "Explicit mapped economic link; publication-visible observations "
                "only. Futures price differences use proportional spot/near "
                "prices, OI is total across listed contracts; annualization uses "
                "exchange expiry and actual calendar days. US short-sale volume "
                "is activity, not short interest. Pair and foreign beta estimates"
                " use only preceding sessions."
                " Fixed index-group participation uses all explicitly configured "
                "members and actual market-session endpoint returns; leader gap "
                "is the maximum return minus the mean of the other returns. "
                "No member is dropped on missing data; pre-publication backcasts "
                "are excluded by the configured first-publication dates."
            ),
            timing_semantics="available_at_before_explicit_decision",
            null_policy="only_actual_dependencies; unknown_mapping_or_clock_undefined",
        )
        for name, dependencies in RECIPES.items()
    )


def _canonical(frame, fields):
    keys = ["asset_id", "timestamp"]
    grouped = frame.groupby(keys, sort=False, dropna=False)
    values = (
        grouped[list(fields)]
        .first()
        .where(grouped[list(fields)].nunique(dropna=False).le(1))
    )
    if "available_at" in frame:
        values["available_at"] = grouped.available_at.max()
    return values.reset_index().sort_values(["asset_id", "timestamp"], kind="stable")


def _align(states, decisions, fields, *, ids=None):
    left = decisions.loc[:, ["asset_id", "timestamp"]].copy()
    if ids is not None:
        left["asset_id"] = ids
    result = pd.DataFrame(index=decisions.index)
    for field in fields:
        clock = states.get(f"{field}_available_at", states.available_at)
        result[field] = latest_observations(
            states.assign(available_at=clock), left, columns=(field,)
        )[field]
    return result


def _quote_session_rows(frame, sessions):
    """Retain interior quote gaps on the supplied common market observations."""
    return (
        pd.concat(
            [
                block.set_index("timestamp")
                .reindex(
                    sessions[
                        (sessions >= block.timestamp.min())
                        & (sessions <= block.timestamp.max())
                    ]
                )
                .assign(asset_id=asset)
                .rename_axis("timestamp")
                .reset_index()
                for asset, block in frame.groupby("asset_id", sort=False)
            ],
            ignore_index=True,
        )
        if len(frame)
        else frame.copy()
    )


def futures_states(futures, contracts, index_prices, *, product="IF"):
    index_id = {"IF": "000300", "IH": "000016", "IC": "000905", "IM": "000852"}[product]
    f = _canonical(
        futures.loc[futures["product"].eq(product)],
        ("close", "volume", "open_interest"),
    )
    if f.empty:
        return pd.DataFrame(columns=["asset_id", "timestamp", "available_at", *FUTURES])
    sessions = pd.DatetimeIndex(f.timestamp.unique()).sort_values()
    if index_prices is None:
        i = pd.DataFrame(columns=["timestamp", "spot", "_spot_available"])
        i["timestamp"] = i["_spot_available"] = pd.Series(dtype="datetime64[ns]")
    else:
        i = index_prices.loc[
            index_prices.source_asset_id.eq(index_id), ["timestamp", "close"]
        ].rename(columns={"close": "spot"})
        # Native index closes use the existing declared after-close upper bound.
        i["_spot_available"] = i.timestamp + pd.Timedelta(hours=7, seconds=1)
        if "available_at" in index_prices:
            i["_spot_available"] = index_prices.loc[
                index_prices.source_asset_id.eq(index_id), "available_at"
            ].to_numpy()
        sessions = sessions.union(pd.DatetimeIndex(i.timestamp))
    f = _quote_session_rows(f, sessions)
    # Metadata observation dates select the applicable contract facts. Their
    # publication clocks gate derived results, not inclusion of daily quotes.
    c = (
        contracts.loc[contracts["product"].eq(product)]
        if contracts is not None
        else pd.DataFrame()
    )
    if c.empty:
        f["expiry_date"] = f["_contract_available"] = pd.NaT
    else:
        fields = ("expiry_date", "open_date") if "open_date" in c else ("expiry_date",)
        c = _canonical(c, fields).rename(
            columns={"available_at": "_contract_available"}
        )
        f["_in_quote_span"] = True
        observed_start = f.groupby("asset_id", sort=False).timestamp.min()
        expected = []
        for asset, block in c.groupby("asset_id", sort=False):
            start = block.open_date.min() if "open_date" in block else pd.NaT
            if pd.isna(start):
                start = observed_start.get(asset, pd.NaT)
            if pd.notna(start):
                dates = sessions[
                    (sessions >= max(start, block.timestamp.min()))
                    & (sessions <= block.expiry_date.max())
                ]
                expected.extend((asset, date) for date in dates)
        if expected:
            f = pd.concat(
                [f, pd.DataFrame(expected, columns=["asset_id", "timestamp"])],
                ignore_index=True,
            ).drop_duplicates(["asset_id", "timestamp"], keep="first")
        f = pd.merge_asof(
            f.sort_values("timestamp", kind="stable"),
            c.sort_values("timestamp", kind="stable"),
            on="timestamp",
            by="asset_id",
            direction="backward",
        )
        active = f.expiry_date.ge(f.timestamp)
        if "open_date" in f:
            active &= f.open_date.le(f.timestamp) | (
                f.open_date.isna() & f.timestamp.ge(f.asset_id.map(observed_start))
            )
        f = f.loc[f._in_quote_span.eq(True) | active].copy()
    f = f.merge(i, on="timestamp", how="left", validate="many_to_one")
    f["_basis"] = f["close"] / f.spot.where(f.spot > 0) - 1
    f["_basis_available"] = f[["available_at", "_spot_available"]].max(
        axis=1, skipna=False
    )
    f = f.sort_values(["asset_id", "timestamp"], kind="stable")
    for w in (1, 5):
        f[f"_basis_change_{w}"] = f.groupby("asset_id", sort=False)._basis.diff(w)
        f[f"_basis_change_{w}_available"] = pd.concat(
            [
                f._basis_available,
                f.groupby("asset_id", sort=False)._basis_available.shift(w),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
    output = []
    for date, group in f.groupby("timestamp", sort=True):
        oi = group.open_interest.sum(min_count=len(group))
        row = {
            "asset_id": product,
            "timestamp": date,
            "available_at": group.available_at.max(skipna=False),
            "index_future_near_basis": np.nan,
            "index_future_calendar_spread": np.nan,
            "index_future_annualized_basis": np.nan,
            "index_future_volume_to_oi": group.volume.sum(min_count=len(group)) / oi
            if oi > 0
            else np.nan,
            "index_future_roll_concentration": np.sum((group.open_interest / oi) ** 2)
            if oi > 0
            else np.nan,
            "_oi": oi,
        }
        for name in FUTURES[:5]:
            row[f"{name}_available_at"] = pd.NaT
        for width in (1, 5):
            row[f"index_future_basis_change_{width}d"] = np.nan
        eligible = group.expiry_date.ge(date)
        if "open_date" in group:
            # Existing quote history proves listing even without an open date.
            eligible &= group.open_date.le(date) | group.open_date.isna()
        front = group.loc[eligible].sort_values(
            ["expiry_date", "asset_id"], kind="stable"
        )
        metadata_known = group.expiry_date.notna().all()
        if len(front) and metadata_known:
            first = front.iloc[0]
            days = (first.expiry_date - date).days
            # All expiries determine the front contract; far-month quotes do not
            # enter its basis. Spread consumes only the first two quote prices.
            selection_clock = group._contract_available.max(skipna=False)
            basis_clock = pd.Series([selection_clock, first._basis_available]).max(
                skipna=False
            )
            row["index_future_near_basis"] = first._basis
            row["index_future_near_basis_available_at"] = basis_clock
            row["index_future_annualized_basis"] = (
                first._basis * 365 / days if days > 0 else np.nan
            )
            row["index_future_annualized_basis_available_at"] = basis_clock
            if len(front) > 1 and first["close"] > 0:
                row["index_future_calendar_spread"] = (
                    front["close"].iloc[1] / first["close"] - 1
                )
                row["index_future_calendar_spread_available_at"] = pd.Series(
                    [selection_clock, *front.available_at.iloc[:2]]
                ).max(skipna=False)
            for w in (1, 5):
                row[f"index_future_basis_change_{w}d"] = first[f"_basis_change_{w}"]
                row[f"index_future_basis_change_{w}d_available_at"] = pd.Series(
                    [selection_clock, first[f"_basis_change_{w}_available"]]
                ).max(skipna=False)
        output.append(row)
    states = pd.DataFrame(output)
    for w in (1, 5):
        states[f"index_future_oi_change_{w}d"] = (
            states._oi / states._oi.shift(w).where(states._oi.shift(w) > 0) - 1
        )
        states[f"index_future_oi_change_{w}d_available_at"] = pd.concat(
            [states.available_at, states.available_at.shift(w)], axis=1
        ).max(axis=1, skipna=False)
    return states


def calculate(decisions, sources, *, columns, parameters=None):
    parameters = {} if parameters is None else parameters
    work = decisions.sort_values(["asset_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    result = work.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    for name in columns:
        result[name] = np.nan
    selected = set(columns)
    missing = {
        name: [table for table in RECIPES[name] if table not in sources]
        for name in columns
    }
    usable = {name for name in selected if not missing[name]}
    group_names = {name for name in usable if name.startswith("index_group_")}
    if group_names:
        from quant_research_platform.markets.implementations.a_share.calendar import (
            unified_trading_sessions,
        )

        group = parameters.get("index_group", {})
        if not group:
            for name in group_names:
                missing[name] = ["parameters.index_group"]
        else:
            raw = sources["index_market"]
            axis = unified_trading_sessions(
                sources["trading_calendar"].loc[
                    sources["trading_calendar"].source_market_type.isin((1, 4))
                ]
            )
            prices = (
                raw.loc[raw.source_asset_id.isin(group)]
                .pivot(index="timestamp", columns="source_asset_id", values="close")
                .reindex(index=axis, columns=list(group))
            )
            for asset, first_published in group.items():
                prices.loc[prices.index < pd.Timestamp(first_published), asset] = np.nan
            signal = pd.DatetimeIndex(work.signal_session)
            for width in (5, 20):
                changes = prices / prices.shift(width).where(lambda x: x > 0) - 1
                complete = changes.notna().all(axis=1)
                for kind, values in (
                    ("positive_fraction", changes.gt(0).mean(axis=1)),
                    (
                        "leader_gap",
                        (len(group) * changes.max(axis=1) - changes.sum(axis=1))
                        / (len(group) - 1),
                    ),
                ):
                    name = f"index_group_{kind}_{width}d"
                    if name in group_names:
                        value = values.where(complete).reindex(signal).to_numpy()
                        # Index closes become visible after the signal day's close.
                        visible = work.timestamp.ge(
                            work.signal_session + pd.Timedelta(hours=7, seconds=1)
                        )
                        result[name] = np.where(visible, value, np.nan)
    if usable & set(US[3:]) and "adjusted_close" not in sources["public_etf_prices"]:
        for name in usable & set(US[3:]):
            missing[name] = ["public_etf_prices.adjusted_close"]
        usable.difference_update(US[3:])
    for width in (20, 60):
        name = f"implied_minus_past_realized_variance_{width}d"
        iv, rv = f"implied_variance_{width}d", f"volatility_{width}d"
        if name in usable and {iv, rv} <= set(work):
            result[name] = work[iv] - 252 * work[rv] ** 2
    if usable & set(FUTURES):
        product = parameters.get("future_product", "IF")
        states = futures_states(
            sources["public_futures_daily"],
            sources.get("public_futures_contracts"),
            sources.get("index_market"),
            product=product,
        )
        fields = [
            name
            for name in usable & set(FUTURES)
            if not name.endswith("hedging_exposure")
        ]
        if fields and not states.empty:
            aligned = _align(states, work, fields, ids=np.repeat(product, len(work)))
            result[fields] = aligned[fields].to_numpy()
        # Exposure must refer to this benchmark; an all-A beta is different.
        exposure = work.get("hedging_beta")
        if (
            exposure is None
            and "index_market" in sources
            and any(name.endswith("hedging_exposure") for name in usable)
            and {"period_return", "signal_session"} <= set(work)
        ):
            from quant_research_platform.compute.dataframe.history import liquidity_beta

            index_id = {"IF": "000300", "IH": "000016", "IC": "000905", "IM": "000852"}[
                product
            ]
            benchmark = (
                sources["index_market"]
                .loc[sources["index_market"].source_asset_id.eq(index_id)]
                .sort_values("timestamp")
            )
            returns = (
                benchmark.close
                / benchmark.close.shift().where(benchmark.close.shift() > 0)
                - 1
            ).set_axis(benchmark.timestamp)
            exposure = pd.Series(np.nan, index=work.index)
            for _, at in work.groupby("asset_id", sort=False).indices.items():
                exposure.loc[at] = liquidity_beta(
                    work.iloc[at].period_return,
                    work.iloc[at].signal_session.map(returns),
                )
        if exposure is not None:
            interactions = {
                "basis_shock_x_hedging_exposure": "index_future_basis_change_1d",
                "oi_shock_x_hedging_exposure": "index_future_oi_change_1d",
            }
            for name, source in interactions.items():
                if name in usable and not states.empty:
                    result[name] = (
                        _align(
                            states, work, (source,), ids=np.repeat(product, len(work))
                        )[source].to_numpy()
                        * exposure
                    )
    if usable & set(PAIRS):
        panel = sources["market_panel_1d"]
        prices = (
            _canonical(panel, ("adj_close",))
            .pivot(index="timestamp", columns="asset_id", values="adj_close")
            .sort_index()
        )
        pairs = parameters.get("pair_assets", {})
        for asset, positions in work.groupby("asset_id", sort=False).indices.items():
            peer = pairs.get(asset)
            if asset not in prices or peer not in prices:
                continue
            x, y = np.log(prices[peer]), np.log(prices[asset])
            xr, yr = x.diff(), y.diff()
            states = pd.DataFrame(
                {
                    "asset_id": asset,
                    "timestamp": prices.index,
                    "available_at": prices.index
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(1, "ns"),
                }
            )
            for w in (20, 60):
                beta = yr.rolling(w, min_periods=w).cov(xr) / xr.rolling(
                    w, min_periods=w
                ).var().where(lambda v: v > 0)
                spread = y - beta.shift() * x
                values = {
                    f"linked_pair_relative_return_{w}d": np.expm1(y - y.shift(w))
                    - np.expm1(x - x.shift(w)),
                    f"linked_pair_beta_adjusted_spread_{w}d": yr - beta.shift() * xr,
                    f"linked_pair_spread_zscore_{w}d": (
                        spread - spread.shift().rolling(w, min_periods=w).mean()
                    )
                    / spread.shift()
                    .rolling(w, min_periods=w)
                    .std()
                    .where(lambda v: v > 0),
                }
                for name, series in values.items():
                    states[name] = series.to_numpy()
            fields = tuple(usable & set(PAIRS))
            result.loc[positions, list(fields)] = _align(
                states, work.iloc[positions], fields
            )[list(fields)].to_numpy()
    index_names = {name for name in usable if name.startswith("index_positive_")}
    if index_names and "signal_session" in work:
        index_id = parameters.get("index_id", "000300")
        events = sources["index_membership"]
        events = events.loc[
            events.source_asset_id.eq(index_id)
            & pd.to_numeric(events.security_type, errors="coerce").eq(1)
        ].copy()
        events["_release"] = pd.to_datetime(events.announcement_date) + pd.Timedelta(
            hours=16
        )
        panel = sources["market_panel_1d"]
        prices = panel.pivot(
            index="timestamp", columns="asset_id", values="adj_close"
        ).sort_index()
        for signal, at in work.groupby("signal_session", sort=False).indices.items():
            decision = work.iloc[at].timestamp.min()
            raw = sources["index_weights"]
            weights = raw.loc[
                raw.source_asset_id.eq(index_id) & raw.timestamp.lt(signal)
            ]
            if "available_at" in weights:
                weights = weights.loc[weights.available_at.le(decision)]
            if weights.empty:
                continue
            snapshot_date = weights.timestamp.max()
            weights = (
                weights.loc[weights.timestamp.eq(snapshot_date)]
                .set_index("constituent_asset_id")
                .weight
            )
            visible = events.loc[
                events.effective_date.gt(snapshot_date)
                & events.effective_date.le(signal)
                & events._release.le(decision)
            ].sort_values(["effective_date", "_release"], kind="stable")
            current = visible.drop_duplicates("constituent_stock_code", keep="last")
            additions = current.loc[
                pd.to_numeric(current.change_type, errors="coerce").eq(1),
                "constituent_stock_code",
            ]
            removals = current.loc[
                pd.to_numeric(current.change_type, errors="coerce").eq(2),
                "constituent_stock_code",
            ]
            # Change events alone omit constituents that have never changed.
            members = weights.index.difference(removals).union(additions)
            if not len(members) or signal not in prices.index:
                continue
            position = prices.index.get_loc(signal)
            selected_prices = prices.reindex(columns=members)
            for width in (5, 20):
                name = f"index_positive_participation_{width}d"
                if name not in usable or position < width:
                    continue
                current_price = selected_prices.iloc[position]
                old = selected_prices.iloc[position - width]
                returns = current_price / old.where(old > 0) - 1
                if returns.notna().all():
                    result.loc[at, name] = returns.gt(0).mean()
            name = "index_positive_return_concentration"
            if name in usable and position:
                values = (
                    selected_prices.iloc[position]
                    / selected_prices.iloc[position - 1].where(lambda x: x > 0)
                    - 1
                )
                lagged = weights.reindex(values.index)
                if lagged.notna().all() and values.notna().all():
                    contribution = lagged * values.clip(lower=0)
                    if contribution.sum() > 0:
                        result.loc[at, name] = np.sum(
                            (contribution / contribution.sum()) ** 2
                        )
    if usable & set(US[:3]):
        raw = sources["public_finra_short_volume"].copy()
        raw["asset_id"] = raw.asset_id.str.split(":").str[0]
        grouped = raw.groupby(["asset_id", "timestamp"], sort=False)
        volumes = grouped[["short_volume", "total_volume"]]
        states = volumes.sum(min_count=1).where(
            volumes.count().eq(grouped.size(), axis=0)
        )
        states["available_at"] = grouped.available_at.max().where(
            grouped.available_at.count().eq(grouped.size())
        )
        states = states.reset_index().sort_values(
            ["asset_id", "timestamp"], kind="stable"
        )
        states[US[0]] = states.short_volume / states.total_volume.where(
            states.total_volume > 0
        )
        states[US[1]] = states.groupby("asset_id", sort=False)[US[0]].diff(5)
        states[f"{US[1]}_available_at"] = pd.concat(
            [
                states.available_at,
                states.groupby("asset_id", sort=False).available_at.shift(5),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        states[US[2]] = states.groupby("asset_id", sort=False)[US[0]].transform(
            lambda s: s.rolling(756, min_periods=756).rank(pct=True)
        )
        states[f"{US[2]}_available_at"] = states.groupby(
            "asset_id", sort=False
        ).available_at.transform(lambda s: rolling_available_at(s, 756))
        fields = tuple(usable & set(US[:3]))
        benchmark = parameters.get("us_activity_asset", "SPY")
        result[list(fields)] = _align(
            states, work, fields, ids=np.repeat(benchmark, len(work))
        )[list(fields)].to_numpy()
    if usable & set(US[3:]):
        raw = _canonical(sources["public_etf_prices"], ("adjusted_close",))
        states = _quote_session_rows(
            raw, pd.DatetimeIndex(raw.timestamp.unique()).sort_values()
        )
        prices = states.groupby("asset_id", sort=False).adjusted_close
        states[US[3]] = states.adjusted_close / prices.shift() - 1
        states[US[4]] = states.adjusted_close / prices.shift(5) - 1
        for name, width in ((US[3], 1), (US[4], 5)):
            states[f"{name}_available_at"] = pd.concat(
                [
                    states.available_at,
                    states.groupby("asset_id", sort=False).available_at.shift(width),
                ],
                axis=1,
            ).max(axis=1, skipna=False)
        spy = states.loc[
            states.asset_id.eq("SPY"), ["timestamp", US[3], f"{US[3]}_available_at"]
        ].rename(
            columns={US[3]: "_market", f"{US[3]}_available_at": "_market_available"}
        )
        states = states.merge(spy, on="timestamp", how="left", validate="many_to_one")
        states["us_relative_sector_shock"] = states[US[3]] - states._market
        states["us_relative_sector_shock_available_at"] = states[
            [f"{US[3]}_available_at", "_market_available"]
        ].max(axis=1, skipna=False)
        mapping = parameters.get("industry_etfs", {})
        ids = (
            (
                work.industry_standard_id.astype("str")
                + ":"
                + work.industry_level1_code.astype("str")
            ).map(mapping)
            if {"industry_standard_id", "industry_level1_code"} <= set(work)
            else pd.Series(np.nan, index=work.index)
        )
        ids = work.asset_id.map(parameters.get("asset_us_tickers", {})).fillna(ids)
        values = _align(
            states, work, (US[3], US[4], "us_relative_sector_shock"), ids=ids
        )
        for name in (US[3], US[4], "us_relative_sector_shock"):
            if name in usable:
                result[name] = values[name].to_numpy()
        if {"period_return", "market_return", "signal_session"} <= set(work):
            from quant_research_platform.compute.dataframe.history import liquidity_beta

            signal_open = work.assign(
                timestamp=pd.to_datetime(work.signal_session)
                + pd.Timedelta(hours=1, minutes=30)
            )
            training = _align(
                states, signal_open, (US[3], "us_relative_sector_shock"), ids=ids
            )
            for _, positions in work.groupby("asset_id", sort=False).indices.items():
                if not ids.iloc[positions].notna().any():
                    continue
                block = work.iloc[positions]
                x = values.iloc[positions][US[3]].to_numpy(float)
                relative = values.iloc[positions]["us_relative_sector_shock"].to_numpy(
                    float
                )
                beta = pd.Series(
                    liquidity_beta(
                        block.period_return,
                        training.iloc[positions][US[3]],
                        market_returns=block.market_return,
                    )
                ).to_numpy()
                relative_beta = pd.Series(
                    liquidity_beta(
                        block.period_return,
                        training.iloc[positions]["us_relative_sector_shock"],
                        market_returns=block.market_return,
                    )
                ).to_numpy()
                for name, v in (
                    ("us_sector_own_beta", beta),
                    ("us_sector_shock_x_own_beta", x * beta),
                    ("us_relative_sector_shock_x_own_beta", relative * relative_beta),
                ):
                    if name in usable:
                        result.loc[positions, name] = v
    result.attrs["missing_source_dependencies"] = {
        k: v for k, v in missing.items() if v
    }
    return result
