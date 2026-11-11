"""Explicit economic links, foreign trading activity and index-futures states."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.asof import latest_observations
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
    "implied_minus_past_realized_variance_20d": (),
    "implied_minus_past_realized_variance_60d": (),
    "index_positive_participation_5d": ("market_panel_1d", "index_membership"),
    "index_positive_participation_20d": ("market_panel_1d", "index_membership"),
    "index_positive_return_concentration": (
        "market_panel_1d",
        "index_membership",
        "index_weights",
    ),
    **dict.fromkeys(
        FUTURES, ("public_futures_daily", "public_futures_contracts", "index_market")
    ),
    **dict.fromkeys(PAIRS, ("market_panel_1d",)),
    **dict.fromkeys(US[:3], ("public_finra_short_volume",)),
    **dict.fromkeys(US[3:], ("public_etf_prices",)),
}
UNITS = dict.fromkeys(RECIPES, "dimensionless")
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
    return latest_observations(states, left, columns=tuple(fields))


def futures_states(futures, contracts, index_prices, *, product="IF"):
    index_id = {"IF": "000300", "IH": "000016", "IC": "000905", "IM": "000852"}[product]
    f = _canonical(
        futures.loc[futures["product"].eq(product)],
        ("close", "volume", "open_interest"),
    )
    c = contracts.loc[contracts["product"].eq(product)]
    if f.empty or c.empty:
        return pd.DataFrame(columns=["asset_id", "timestamp", "available_at", *FUTURES])
    coordinates = f[["asset_id", "timestamp"]].assign(
        timestamp=lambda d: d.timestamp + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    )
    expiry = _align(c, coordinates, ("expiry_date",))
    f["expiry_date"] = pd.to_datetime(expiry.expiry_date).to_numpy()
    i = index_prices.loc[
        index_prices.source_asset_id.eq(index_id), ["timestamp", "close"]
    ].rename(columns={"close": "spot"})
    f = f.merge(i, on="timestamp", how="left", validate="many_to_one")
    f = f.loc[f.expiry_date.ge(f.timestamp)].copy()
    f["_basis"] = f["close"] / f.spot.where(f.spot > 0) - 1
    f = f.sort_values(["asset_id", "timestamp"], kind="stable")
    for w in (1, 5):
        f[f"_basis_change_{w}"] = f.groupby("asset_id", sort=False)._basis.diff(w)
    output = []
    for date, group in f.groupby("timestamp", sort=True):
        front = group.sort_values(["expiry_date", "asset_id"], kind="stable")
        first = front.iloc[0]
        oi = group.open_interest.sum(min_count=len(group))
        days = (first.expiry_date - date).days
        row = {
            "asset_id": product,
            "timestamp": date,
            "available_at": group.available_at.max(),
            "index_future_near_basis": first._basis,
            "index_future_calendar_spread": front["close"].iloc[1] / first["close"] - 1
            if len(front) > 1 and first["close"] > 0
            else np.nan,
            "index_future_annualized_basis": first._basis * 365 / days
            if days > 0
            else np.nan,
            "index_future_volume_to_oi": group.volume.sum(min_count=len(group)) / oi
            if oi > 0
            else np.nan,
            "index_future_roll_concentration": np.sum((group.open_interest / oi) ** 2)
            if oi > 0
            else np.nan,
            "_oi": oi,
        }
        for w in (1, 5):
            row[f"index_future_basis_change_{w}d"] = first[f"_basis_change_{w}"]
        output.append(row)
    states = pd.DataFrame(output)
    for w in (1, 5):
        states[f"index_future_oi_change_{w}d"] = (
            states._oi / states._oi.shift(w).where(states._oi.shift(w) > 0) - 1
        )
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
            sources["public_futures_contracts"],
            sources["index_market"],
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
            visible = events.loc[
                events.effective_date.le(signal) & events._release.le(decision)
            ].sort_values(["effective_date", "_release"], kind="stable")
            current = visible.drop_duplicates("constituent_stock_code", keep="last")
            members = current.loc[
                pd.to_numeric(current.change_type, errors="coerce").eq(1),
                "constituent_stock_code",
            ]
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
                if returns.notna().any():
                    result.loc[at, name] = returns.gt(0).where(returns.notna()).mean()
            name = "index_positive_return_concentration"
            if name in usable and position:
                raw = sources["index_weights"]
                weights = raw.loc[
                    raw.source_asset_id.eq(index_id) & raw.timestamp.lt(signal)
                ]
                if weights.empty:
                    continue
                weights = (
                    weights.loc[weights.timestamp.eq(weights.timestamp.max())]
                    .set_index("constituent_asset_id")
                    .weight
                )
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
        states = grouped[["short_volume", "total_volume"]].sum(min_count=1)
        states["available_at"] = grouped.available_at.max()
        states = states.reset_index().sort_values(
            ["asset_id", "timestamp"], kind="stable"
        )
        states[US[0]] = states.short_volume / states.total_volume.where(
            states.total_volume > 0
        )
        states[US[1]] = states.groupby("asset_id", sort=False)[US[0]].diff(5)
        states[US[2]] = states.groupby("asset_id", sort=False)[US[0]].transform(
            lambda s: s.rolling(756, min_periods=756).rank(pct=True)
        )
        fields = tuple(usable & set(US[:3]))
        benchmark = parameters.get("us_activity_asset", "SPY")
        result[list(fields)] = _align(
            states, work, fields, ids=np.repeat(benchmark, len(work))
        )[list(fields)].to_numpy()
    if usable & set(US[3:]):
        raw = _canonical(sources["public_etf_prices"], ("adjusted_close",))
        states = raw.copy()
        prices = states.groupby("asset_id", sort=False).adjusted_close
        states[US[3]] = states.adjusted_close / prices.shift() - 1
        states[US[4]] = states.adjusted_close / prices.shift(5) - 1
        spy = states.loc[states.asset_id.eq("SPY"), ["timestamp", US[3]]].rename(
            columns={US[3]: "_market"}
        )
        states = states.merge(spy, on="timestamp", how="left", validate="many_to_one")
        states["us_relative_sector_shock"] = states[US[3]] - states._market
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
