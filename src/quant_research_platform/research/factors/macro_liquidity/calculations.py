"""Compute release-frequency histories once, then align states to decision cutoffs."""

from bisect import bisect_right, insort
from itertools import groupby

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _industry_group_codes,
)
from quant_research_platform.compute.dataframe.history import liquidity_beta
from quant_research_platform.markets.implementations.a_share.calendar import (
    market_type_calendar_state,
)
from quant_research_platform.research.factors.macro_liquidity.definitions import (
    COMPOSITES,
    PRESSURE_COMPONENTS,
    RATIOS,
    RECIPES,
    STATE_PRODUCTS,
    VARIABLES,
)

DEFAULT_WINDOWS = {"D": 60, "W": 26, "M": 24, "Q": 12, "event": 20}
_LATEST_BALANCE_RATIOS = {
    "ml_us_reserve_asset_ratio": "us_reserves_wed",
    "ml_us_tga_asset_ratio": "us_tga_wed",
    "ml_us_rrp_asset_ratio": "us_on_rrp",
}
BETA_INPUT_COLUMNS = (
    "listing_date",
    "period_return",
    "market_type",
    "signal_session",
    "return_available_at",
)


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
    ratio = name in RATIOS
    weights = dict.fromkeys(RATIOS[name], 1) if ratio else COMPOSITES[name][0]
    rows = observations.loc[observations.series_id.isin(weights)]
    state = {key: {} for key in weights}
    output = []
    for timestamp, update in groupby(
        rows.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .itertuples(index=False),
        key=lambda row: row.timestamp,
    ):
        changed = set()
        for row in update:
            state[row.series_id][row.observation_date] = row.value
            changed.add(row.observation_date)
        for observed in sorted(changed):
            if ratio or all(observed in state[key] for key in weights):
                if ratio:
                    numerator, denominator = (
                        state[key].get(observed, np.nan) for key in weights
                    )
                    value = numerator / denominator if denominator > 0 else np.nan
                else:
                    value = sum(
                        weight * state[key][observed] for key, weight in weights.items()
                    )
                output.append(
                    (
                        name,
                        observed,
                        timestamp,
                        value,
                    )
                )
    return pd.DataFrame(
        output, columns=("series_id", "observation_date", "timestamp", "value")
    )


def _history_features(history, transforms, unit, frequency, window, config):
    """Update known observations and revisions without counting daily carries."""
    columns = tuple(transforms)
    selected = set(columns)
    previous_transforms = selected & {"change", "acceleration", "growth"}
    yearly_transforms = selected & {"yoy_change", "yoy_growth"}
    current_transforms = selected & {"zscore", "percentile", "sum"}
    prior_transforms = selected & {"innovation", "innovation_zscore", "ewma_deviation"}
    prior_mean = bool(selected & {"innovation", "innovation_zscore"})
    volatility = "volatility" in selected
    dates, state, output = [], {}, []
    minimum = max(2, int(np.ceil(window * config["minimum_fraction"])))
    scale = 100 if unit == "percent" else 1
    if "ewma_deviation" in selected:
        weights = np.exp2(-np.arange(window - 1, -1, -1) / config["ewma_halflife"])
    for timestamp, changes in groupby(
        history.dropna(subset=["timestamp"])
        .sort_values("timestamp", kind="stable")
        .itertuples(index=False),
        key=lambda row: row.timestamp,
    ):
        for row in changes:
            observed = row.observation_date
            if observed not in state:
                insort(dates, observed)
            state[observed] = row.value
        latest = dates[-1]
        x = state[latest]
        values = {}
        if "level" in selected:
            values["level"] = x
        if previous_transforms:
            previous = state[dates[-2]] if len(dates) >= 2 else np.nan
            if "change" in selected:
                values["change"] = (x - previous) * scale
            if "acceleration" in selected:
                values["acceleration"] = (
                    (x - 2 * previous + state[dates[-3]]) * scale
                    if len(dates) >= 3
                    else np.nan
                )
            if "growth" in selected:
                values["growth"] = (
                    x / previous - 1 if x > 0 and previous > 0 else np.nan
                )
        if yearly_transforms:
            year_target = latest - pd.DateOffset(years=1)
            if frequency in {"M", "Q"}:
                period = year_target.to_period(frequency)
                year_index = bisect_right(dates, period.end_time) - 1
                same_period = (
                    year_index >= 0 and dates[year_index].to_period(frequency) == period
                )
            else:
                year_index = bisect_right(dates, year_target) - 1
                tolerance = {"D": 7, "W": 7, "event": 0}[frequency]
                same_period = (
                    year_index >= 0
                    and (year_target - dates[year_index]).days <= tolerance
                )
            year_value = state[dates[year_index]] if same_period else np.nan
            if "yoy_change" in selected:
                values["yoy_change"] = (x - year_value) * scale
            if "yoy_growth" in selected:
                values["yoy_growth"] = (
                    x / year_value - 1 if x > 0 and year_value > 0 else np.nan
                )
        if current_transforms or prior_transforms or volatility:
            tail = np.asarray(
                [state[day] for day in dates[-(window + 1) :]], dtype=float
            )
        if current_transforms:
            current = tail[-window:]
            sample = current[np.isfinite(current)]
            if len(sample) >= minimum:
                if "zscore" in selected:
                    std = sample.std(ddof=1)
                    values["zscore"] = (x - sample.mean()) / std if std > 0 else np.nan
                if "percentile" in selected:
                    values["percentile"] = (
                        (
                            np.count_nonzero(sample < x)
                            + 0.5 * np.count_nonzero(sample == x)
                        )
                        / len(sample)
                        if np.isfinite(x)
                        else np.nan
                    )
                if "sum" in selected:
                    values["sum"] = sample.sum() * scale
        if volatility:
            differences = np.diff(tail) * scale
            valid_differences = differences[np.isfinite(differences)]
            if len(valid_differences) >= minimum:
                values["volatility"] = valid_differences.std(ddof=1)
        if prior_transforms:
            prior = tail[:-1]
            known = np.isfinite(prior)
            if known.sum() >= minimum:
                sample = prior[known]
                if prior_mean:
                    mean = sample.mean()
                    if "innovation" in selected:
                        values["innovation"] = (x - mean) * scale
                    if "innovation_zscore" in selected:
                        std = sample.std(ddof=1)
                        values["innovation_zscore"] = (
                            (x - mean) / std if std > 0 else np.nan
                        )
                if "ewma_deviation" in selected:
                    values["ewma_deviation"] = (
                        x - np.average(sample, weights=weights[-len(prior) :][known])
                    ) * scale
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


def required_factors(columns, config):
    """Resolve the existing finite macro recipe dependencies once for all callers."""
    selected = set(columns)
    unknown = selected - RECIPES.keys()
    if unknown:
        raise ValueError("unknown_macro_factors: " + ", ".join(sorted(unknown)))
    needed = set(selected)
    for column in tuple(needed):
        recipe = RECIPES[column]
        if recipe[0] in {"interaction", "industry_interaction"}:
            needed.add("ml_" + recipe[1])
        elif recipe[0] == "beta":
            needed.add("ml_" + recipe[1] + "_level")
        elif recipe[0] == "matched_beta":
            needed.update(
                f"ml_{recipe[1]}_{suffix}" for suffix in ("beta", "level", "change")
            )
    if "ml_us_cn_pressure_difference" in needed:
        needed.update(("ml_us_funding_pressure", "ml_cn_funding_pressure"))
    for name, components in config["pressure_components"].items():
        if "ml_" + name in needed:
            needed.update("ml_" + item for item in components)
    for name, components in STATE_PRODUCTS.items():
        if "ml_" + name in needed:
            needed.update("ml_" + item for item in components)
    if "ml_us_net_liquidity_latest" in needed:
        needed.update(
            "ml_" + item + "_level"
            for item in ("us_fed_assets", "us_tga_wed", "us_on_rrp")
        )
    for name, variable in _LATEST_BALANCE_RATIOS.items():
        if name in needed:
            needed.update(("ml_us_fed_assets_level", f"ml_{variable}_level"))
    return needed


def macro_states(observations, cutoffs, *, columns, config):
    """Return one row per distinct actual decision time in chronological order."""
    cutoffs = (
        pd.DatetimeIndex(pd.to_datetime(cutoffs, utc=True))
        .tz_localize(None)
        .unique()
        .sort_values()
    )
    needed = required_factors(columns, config)
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
            if variable in COMPOSITES or variable in RATIOS
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
    for name, (left, right) in STATE_PRODUCTS.items():
        if "ml_" + name in needed:
            output["ml_" + name] = output["ml_" + left] * output["ml_" + right]
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
    for name, variable in _LATEST_BALANCE_RATIOS.items():
        if name in needed:
            denominator = output["ml_us_fed_assets_level"].where(lambda x: x.gt(0))
            output[name] = (
                output["ml_" + variable + "_level"]
                * (1000 if variable == "us_on_rrp" else 1)
                / denominator
            )
    return pd.DataFrame(output)


def calculate_macro_factors(
    observations, panel, *, columns, config, states=None, calendar=None
):
    """Align both clocks once, or use explicit precomputed decision states."""
    config = parameters(config)
    work = panel.sort_values(["timestamp", "asset_id"], kind="stable").reset_index(
        drop=True
    )
    beta_columns = [
        name
        for name in sorted(required_factors(columns, config))
        if RECIPES[name][0] == "beta"
    ]
    variables = {RECIPES[name][1] for name in beta_columns}
    clocks = None
    if (
        beta_columns
        and calendar is not None
        and set(BETA_INPUT_COLUMNS) <= set(work.columns)
    ):
        clocks = pd.to_datetime(work.return_available_at, utc=True).dt.tz_localize(None)
        # Unknown or immature returns are unusable, not a reason to invent a close.
        clocks = clocks.where(clocks.le(work.timestamp))
    if states is None:
        cutoffs = (
            work.timestamp
            if clocks is None
            else pd.concat((work.timestamp, clocks.dropna()), ignore_index=True)
        )
        states = macro_states(observations, cutoffs, columns=columns, config=config)
        historical = states
    elif clocks is not None:
        historical = macro_states(
            observations,
            clocks.dropna(),
            columns=[f"ml_{name}_level" for name in variables],
            config=config,
        )
    output = work.loc[:, ["market_id", "asset_id", "timestamp"]].merge(
        states, on="timestamp", how="left", validate="many_to_one"
    )
    interactions = {
        column: RECIPES[column]
        for column in columns
        if RECIPES[column][0] in {"interaction", "industry_interaction"}
    }
    if interactions:
        exposures = list(
            dict.fromkeys(
                recipe[2] for recipe in interactions.values() if recipe[2] in work
            )
        )
        values = work.loc[:, exposures].apply(pd.to_numeric, errors="raise")
        if "eligible" in work:
            values = values.where(work.eligible, axis=0)
        groups = values.groupby(work.timestamp, sort=False)
        std = groups.transform("std").where(lambda x: x.gt(0))
        values = (values - groups.transform("mean")) / std
        sector_exposures = list(
            dict.fromkeys(
                recipe[2]
                for recipe in interactions.values()
                if recipe[0] == "industry_interaction" and recipe[2] in values
            )
        )
        sectors = pd.DataFrame(index=work.index)
        if sector_exposures and {"industry_standard_id", "industry_level1_code"} <= set(
            work
        ):
            # Sector means use only the current, eligible cross-section.
            industries = _industry_group_codes(work)
            sectors = (
                values.loc[industries.notna(), sector_exposures]
                .groupby([work.timestamp, industries], sort=False)
                .transform("mean")
                .reindex(work.index)
            )
            if "eligible" in work:
                sectors = sectors.where(work.eligible, axis=0)
        products = {}
        for column, (kind, state, exposure) in interactions.items():
            exposure_values = sectors if kind == "industry_interaction" else values
            products[column] = (
                output["ml_" + state] * exposure_values[exposure]
                if exposure in exposure_values
                else np.nan
            )
        output = pd.concat((output, pd.DataFrame(products, index=work.index)), axis=1)
    output[beta_columns] = np.nan
    if clocks is not None:
        historical = historical.set_index("timestamp")
        listing = work.listing_date
        previous = work.groupby("asset_id", sort=False)[
            ["listing_date", "market_type"]
        ].shift()
        segment = (
            (
                listing.ne(previous.listing_date)
                | work.market_type.ne(previous.market_type)
            )
            .fillna(True)
            .groupby(work.asset_id)
            .cumsum()
        )
        calendars = {
            market: market_type_calendar_state(calendar, market).sort_index()
            for market in work.market_type.dropna().unique()
        }
        calendar_coverage = {
            market: state.index.to_series().diff().gt(pd.Timedelta(days=1)).cumsum()
            for market, state in calendars.items()
        }
        for _, group in work.loc[listing.notna() & work.market_type.notna()].groupby(
            [work.asset_id, segment], sort=False
        ):
            market = group.market_type.iloc[0]
            calendar_state = calendars[market]
            sessions = calendar_state.index[calendar_state.fillna(False)]
            signals = pd.DatetimeIndex(group.signal_session)
            if (
                calendar_state.reindex(signals).eq(False).any()
                or signals.has_duplicates
            ):
                raise ValueError("liquidity_beta_signal_session_invalid")
            axis = sessions[(sessions >= signals.min()) & (sessions <= signals.max())]
            if axis.empty:
                continue
            # Unknown calendar dates cannot prove which observations belong in W.
            boundary = calendar_coverage[market].reindex(axis).diff().gt(0)
            boundary.iloc[0] = signals.min() < axis[0]
            known_window = (
                boundary.rolling(config["beta_window"] - 1, min_periods=1).sum().eq(0)
            )
            returns = group.set_index("signal_session").period_return.reindex(axis)
            market_returns = (
                group.set_index("signal_session").market_return.reindex(axis)
                if "market_return" in group
                else None
            )
            for variable in variables:
                level = pd.Series(
                    historical[f"ml_{variable}_level"]
                    .reindex(clocks.loc[group.index])
                    .to_numpy(),
                    index=signals,
                ).reindex(axis)
                shocks = level.diff().mask(boundary)
                for kind in ("beta", "partial_beta"):
                    name = f"ml_{variable}_{kind}"
                    if name not in beta_columns:
                        continue
                    if kind == "partial_beta" and market_returns is None:
                        continue
                    result = liquidity_beta(
                        returns,
                        shocks,
                        market_returns=market_returns
                        if kind == "partial_beta"
                        else None,
                        window=config["beta_window"],
                        min_observations=config["beta_min_observations"],
                        min_changes=config["beta_min_changes"],
                    )
                    output.loc[group.index, name] = (
                        pd.Series(result, index=axis)
                        .where(known_window)
                        .reindex(signals)
                        .to_numpy()
                    )
    for name in columns:
        recipe = RECIPES[name]
        if recipe[0] == "matched_beta":
            # Rate changes are displayed in bp; beta uses the level's percent unit.
            scale = 100 if VARIABLES[recipe[1]][0] == "percent" else 1
            output[name] = (
                output[f"ml_{recipe[1]}_change"]
                / scale
                * output[f"ml_{recipe[1]}_beta"]
            )
    return output.loc[:, ["market_id", "asset_id", "timestamp", *columns]]
