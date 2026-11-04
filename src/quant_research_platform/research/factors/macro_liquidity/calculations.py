"""Compute release-frequency histories once, then align states to decision cutoffs."""

from bisect import bisect_right, insort

import numpy as np
import pandas as pd

from quant_research_platform.research.factors.macro_liquidity.definitions import (
    COMPOSITES,
    PRESSURE_COMPONENTS,
    RECIPES,
    VARIABLES,
)

DEFAULT_WINDOWS = {"D": 60, "W": 26, "M": 24, "Q": 12, "event": 20}


def parameters(values):
    allowed = {
        "windows",
        "minimum_fraction",
        "ewma_halflife",
        "pressure_components",
        "beta_window",
        "beta_min_observations",
        "beta_min_changes",
    }
    if set(values) - allowed:
        raise ValueError("macro_parameter_not_consumed")
    result = {
        "windows": DEFAULT_WINDOWS,
        "minimum_fraction": 1.0,
        "ewma_halflife": 10.0,
        "pressure_components": PRESSURE_COMPONENTS,
        "beta_window": 252,
        "beta_min_observations": 126,
        "beta_min_changes": 20,
        **values,
    }
    windows = result["windows"]
    if (
        type(windows) is not dict
        or set(windows) != set(DEFAULT_WINDOWS)
        or any(type(value) is not int or value < 2 for value in windows.values())
    ):
        raise ValueError("macro_windows_invalid")
    for field in ("beta_window", "beta_min_observations", "beta_min_changes"):
        if type(result[field]) is not int or result[field] < 2:
            raise ValueError("macro_beta_window_invalid")
    if (
        max(result["beta_min_observations"], result["beta_min_changes"])
        > result["beta_window"]
    ):
        raise ValueError("macro_beta_minimum_exceeds_window")
    if (
        type(result["minimum_fraction"]) not in (int, float)
        or not 0 < result["minimum_fraction"] <= 1
        or type(result["ewma_halflife"]) not in (int, float)
        or not 0 < result["ewma_halflife"] < float("inf")
    ):
        raise ValueError("macro_window_parameters_invalid")
    if set(result["pressure_components"]) != set(PRESSURE_COMPONENTS):
        raise ValueError("macro_pressure_groups_invalid")
    for components in result["pressure_components"].values():
        if (
            type(components) is not dict
            or not components
            or any(
                f"ml_{column}" not in RECIPES
                or RECIPES[f"ml_{column}"][0] != "series"
                or type(weight) not in (int, float)
                or not np.isfinite(weight)
                for column, weight in components.items()
            )
        ):
            raise ValueError("macro_pressure_components_invalid")
    return result


def _composite_history(observations, name):
    weights, _, _ = COMPOSITES[name]
    rows = observations.loc[observations.series_id.isin(weights)]
    state = {key: {} for key in weights}
    output = []
    for timestamp, update in rows.groupby("timestamp", sort=True):
        changed = set()
        for row in update.itertuples(index=False):
            state[row.series_id][row.observation_date] = row.value
            changed.add(row.observation_date)
        for observed in sorted(changed):
            if all(observed in state[key] for key in weights):
                output.append(
                    (
                        name,
                        observed,
                        timestamp,
                        sum(
                            weight * state[key][observed]
                            for key, weight in weights.items()
                        ),
                    )
                )
    return pd.DataFrame(
        output, columns=("series_id", "observation_date", "timestamp", "value")
    )


def _history_features(history, transforms, unit, frequency, window, config):
    """Update known observations and revisions without counting daily carries."""
    columns = tuple(transforms)
    dates, state, output = [], {}, []
    minimum = max(2, int(np.ceil(window * config["minimum_fraction"])))
    scale = 100 if unit == "percent" else 1
    for timestamp, changes in history.groupby("timestamp", sort=True):
        for row in changes.itertuples(index=False):
            observed = row.observation_date
            if observed not in state:
                insort(dates, observed)
            state[observed] = row.value
        latest = dates[-1]
        x = state[latest]
        tail = np.asarray([state[day] for day in dates[-(window + 1) :]], dtype=float)
        current = tail[-window:]
        prior = tail[:-1]
        finite = np.isfinite(current)
        previous = tail[-2] if len(tail) >= 2 else np.nan
        year_target = latest - pd.DateOffset(years=1)
        year_index = bisect_right(dates, year_target) - 1
        tolerance = {"D": 7, "W": 7, "M": 0, "Q": 0, "event": 0}[frequency]
        year_value = (
            state[dates[year_index]]
            if (year_index >= 0 and (year_target - dates[year_index]).days <= tolerance)
            else np.nan
        )
        change = (x - previous) * scale
        values = {
            "level": x,
            "change": change,
            "acceleration": (x - 2 * previous + tail[-3]) * scale
            if len(tail) >= 3
            else np.nan,
            "yoy_change": (x - year_value) * scale,
        }
        if finite.sum() >= minimum:
            sample = current[finite]
            std = sample.std(ddof=1)
            values["zscore"] = (x - sample.mean()) / std if std > 0 else np.nan
            values["percentile"] = (
                (np.count_nonzero(sample < x) + 0.5 * np.count_nonzero(sample == x))
                / len(sample)
                if np.isfinite(x)
                else np.nan
            )
            values["sum"] = sample.sum() * scale
        differences = np.diff(tail) * scale
        valid_differences = differences[np.isfinite(differences)]
        if len(valid_differences) >= minimum:
            values["volatility"] = valid_differences.std(ddof=1)
        known = np.isfinite(prior)
        if known.sum() >= minimum:
            values["innovation"] = (x - prior[known].mean()) * scale
            std = prior[known].std(ddof=1)
            values["innovation_zscore"] = (
                (x - prior[known].mean()) / std if std > 0 else np.nan
            )
            weights = np.exp2(
                -np.arange(len(prior) - 1, -1, -1) / config["ewma_halflife"]
            )
            values["ewma_deviation"] = (
                x - np.average(prior[known], weights=weights[known])
            ) * scale
        if set(transforms) & {"log_level", "growth", "yoy_growth"}:
            values["log_level"] = np.log(x) if x > 0 else np.nan
            values["growth"] = x / previous - 1 if x > 0 and previous > 0 else np.nan
            values["yoy_growth"] = (
                x / year_value - 1 if x > 0 and year_value > 0 else np.nan
            )
        output.append(
            (timestamp, latest, *(values.get(column, np.nan) for column in columns))
        )
    return pd.DataFrame(output, columns=("timestamp", "observation_date", *columns))


def _align(history, cutoffs, columns):
    if history.empty:
        return pd.DataFrame(np.nan, index=range(len(cutoffs)), columns=columns)
    index = (
        history.timestamp.to_numpy(dtype="datetime64[ns]").searchsorted(
            cutoffs.to_numpy(dtype="datetime64[ns]"), side="right"
        )
        - 1
    )
    result = history.loc[:, columns].iloc[np.maximum(index, 0)].reset_index(drop=True)
    result.loc[index < 0] = np.nan
    return result


def macro_states(observations, cutoffs, *, columns, config):
    """Return one row per distinct actual decision time in chronological order."""
    cutoffs = (
        pd.DatetimeIndex(pd.to_datetime(cutoffs, utc=True))
        .tz_localize(None)
        .unique()
        .sort_values()
    )
    selected = set(columns)
    unknown = selected - RECIPES.keys()
    if unknown:
        raise ValueError("unknown_macro_factors: " + ", ".join(sorted(unknown)))
    needed = set(selected)
    for column in tuple(needed):
        recipe = RECIPES[column]
        if recipe[0] == "interaction":
            needed.add("ml_" + recipe[1])
        elif recipe[0] == "beta":
            needed.add("ml_" + recipe[1] + "_level")
    if "ml_us_cn_pressure_difference" in needed:
        needed.update(("ml_us_funding_pressure", "ml_cn_funding_pressure"))
    for name, components in config["pressure_components"].items():
        if "ml_" + name in needed:
            needed.update("ml_" + item for item in components)
    ratio_inputs = {
        "ml_us_reserve_asset_ratio": "us_reserves_wed",
        "ml_us_tga_asset_ratio": "us_tga_wed",
        "ml_us_rrp_asset_ratio": "us_on_rrp",
    }
    if needed & {"ml_us_net_liquidity_latest", *ratio_inputs}:
        needed.update(
            "ml_" + item + "_level"
            for item in ("us_fed_assets", "us_tga_wed", "us_on_rrp", "us_reserves_wed")
        )
    series_requests = {}
    for column in sorted(needed):
        recipe = RECIPES[column]
        if recipe[0] == "series":
            series_requests.setdefault(recipe[1], []).append(recipe[2])
    data = observations.loc[
        :, ["series_id", "observation_date", "timestamp", "value"]
    ].copy()
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True).dt.tz_localize(None)
    data["observation_date"] = pd.to_datetime(data.observation_date)
    if not cutoffs.empty:
        data = data.loc[data.timestamp.le(cutoffs[-1])]
    output = {"timestamp": cutoffs}
    groups = {name: rows for name, rows in data.groupby("series_id", sort=False)}
    for variable, transforms in series_requests.items():
        history = (
            _composite_history(data, variable)
            if variable in COMPOSITES
            else groups.get(variable, data.iloc[:0])
        )
        unit, frequency = VARIABLES[variable]
        history = _history_features(
            history, transforms, unit, frequency, config["windows"][frequency], config
        )
        aligned = _align(history, cutoffs, transforms)
        for transform in transforms:
            output[f"ml_{variable}_{transform}"] = aligned[transform]
    for name, components in config["pressure_components"].items():
        if "ml_" + name in needed:
            output["ml_" + name] = sum(
                output["ml_" + item] * weight for item, weight in components.items()
            )
    if "ml_us_cn_pressure_difference" in needed:
        output["ml_us_cn_pressure_difference"] = (
            output["ml_us_funding_pressure"] - output["ml_cn_funding_pressure"]
        )
    if "ml_us_net_liquidity_latest" in needed:
        output["ml_us_net_liquidity_latest"] = (
            output["ml_us_fed_assets_level"]
            - output["ml_us_tga_wed_level"]
            - 1000 * output["ml_us_on_rrp_level"]
        )
    for name, variable in ratio_inputs.items():
        if name in needed:
            denominator = output["ml_us_fed_assets_level"].where(lambda x: x.gt(0))
            output[name] = (
                output["ml_" + variable + "_level"]
                * (1000 if variable == "us_on_rrp" else 1)
                / denominator
            )
    return pd.DataFrame(output)


def liquidity_beta(
    returns,
    shocks,
    *,
    market_returns=None,
    window=252,
    min_observations=126,
    min_changes=20,
):
    """Estimate historical sensitivity with explicit effective shock counts."""
    data = pd.DataFrame(
        {"x": np.asarray(shocks, float), "y": np.asarray(returns, float)}
    )
    if market_returns is not None:
        data["m"] = np.asarray(market_returns, float)
    data.loc[~np.isfinite(data).all(axis=1)] = np.nan
    x, y = data.x, data.y
    rolling = x.rolling(window, min_periods=min_observations)
    variance = rolling.var()
    covariance = rolling.cov(y)
    valid = data.x.notna() & data.x.ne(0)
    enough = valid.rolling(window, min_periods=1).sum().ge(min_changes)
    if market_returns is None:
        result = covariance / variance.where(variance.gt(0))
    else:
        market_variance = data.m.rolling(window, min_periods=min_observations).var()
        xm = rolling.cov(data.m)
        ym = y.rolling(window, min_periods=min_observations).cov(data.m)
        product = variance * market_variance
        determinant = product - xm * xm
        invertible = determinant.gt(16 * np.finfo(float).eps * product) & product.gt(0)
        result = (covariance * market_variance - ym * xm) / determinant.where(
            invertible
        )
    return result.where(enough).to_numpy()


def calculate_macro_factors(observations, panel, *, columns, config, states=None):
    """Broadcast macro histories; equity interactions operate cross-sectionally."""
    config = parameters(config)
    work = panel.sort_values(["timestamp", "asset_id"], kind="stable").reset_index(
        drop=True
    )
    if states is None:
        states = macro_states(
            observations, work.timestamp, columns=columns, config=config
        )
    output = work.loc[:, ["market_id", "asset_id", "timestamp"]].merge(
        states, on="timestamp", how="left", validate="many_to_one"
    )
    for column in columns:
        recipe = RECIPES[column]
        if recipe[0] == "interaction":
            _, state, exposure = recipe
            if exposure not in work:
                output[column] = np.nan
                continue
            value = pd.to_numeric(work[exposure], errors="raise")
            if "eligible" in work:
                value = value.where(work.eligible)
            group = value.groupby(work.timestamp, sort=False)
            std = group.transform("std").where(lambda x: x.gt(0))
            value = (value - group.transform("mean")) / std
            output[column] = output["ml_" + state] * value
    beta_columns = [name for name in columns if RECIPES[name][0] == "beta"]
    output[beta_columns] = np.nan
    if beta_columns and {"listing_date", "period_return"} <= set(work.columns):
        if "return_available_at" not in work:
            raise ValueError("liquidity_beta_requires_observed_return_clock")
        clocks = pd.to_datetime(work.return_available_at, utc=True).dt.tz_localize(None)
        if (clocks > work.timestamp).any() or clocks.isna().any():
            raise ValueError("liquidity_beta_return_not_available")
        variables = {RECIPES[name][1] for name in beta_columns}
        historical = macro_states(
            observations,
            clocks,
            columns=[f"ml_{name}_level" for name in variables],
            config=config,
        ).set_index("timestamp")
        listing = work.listing_date
        prior = listing.groupby(work.asset_id, sort=False).shift()
        segment = listing.ne(prior).fillna(True).groupby(work.asset_id).cumsum()
        for _, group in work.loc[listing.notna()].groupby(
            [work.asset_id, segment], sort=False
        ):
            for variable in variables:
                level = historical.loc[
                    clocks.loc[group.index], f"ml_{variable}_level"
                ].to_numpy()
                shocks = np.r_[np.nan, np.diff(level)]
                for kind in ("beta", "partial_beta"):
                    name = f"ml_{variable}_{kind}"
                    if name not in beta_columns:
                        continue
                    if kind == "partial_beta" and "market_return" not in group:
                        output.loc[group.index, name] = np.nan
                        continue
                    output.loc[group.index, name] = liquidity_beta(
                        group.period_return,
                        shocks,
                        market_returns=group.market_return
                        if kind == "partial_beta"
                        else None,
                        window=config["beta_window"],
                        min_observations=config["beta_min_observations"],
                        min_changes=config["beta_min_changes"],
                    )
    return output.loc[:, ["market_id", "asset_id", "timestamp", *columns]]
