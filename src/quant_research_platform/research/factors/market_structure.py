"""Causal breadth, industry participation and leadership on a declared full panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)

_INDUSTRY = ("industry_standard_id", "industry_level1_code")
INPUTS = {
    "market_up_fraction": ("period_return",),
    "market_above_ma20_fraction": ("adj_close",),
    "market_above_ma60_fraction": ("adj_close",),
    "market_high_low_balance_60d": ("adj_close",),
    "market_breadth_change_5d": ("period_return",),
    "industry_return_dispersion_20d": ("adj_close", *_INDUSTRY),
    "industry_positive_return_hhi_20d": ("adj_close", *_INDUSTRY),
    "industry_return_relative_20d": ("adj_close", *_INDUSTRY),
    "industry_turnover_relative": ("turnover", *_INDUSTRY),
    "industry_amount_share": ("amount", *_INDUSTRY),
    "industry_amount_hhi": ("amount", *_INDUSTRY),
    "industry_activity_pressure": ("amount", "float_market_cap", *_INDUSTRY),
    "industry_comovement_20d": ("period_return", *_INDUSTRY),
    "leader_breadth_fragility": ("adj_close",),
    "industry_isolation_20d": ("adj_close", *_INDUSTRY),
}


def _lookback(name):
    if name == "market_high_low_balance_60d":
        return 60
    if name.startswith("market_above_ma"):
        return int(name.removeprefix("market_above_ma").removesuffix("_fraction")) - 1
    if name == "market_breadth_change_5d":
        return 5
    if name == "industry_comovement_20d":
        return 19
    return 20 if "adj_close" in INPUTS[name] else 0


DESCRIPTIONS = {
    "market_up_fraction": (
        "Market context: positive-return fraction among fresh valid A-share "
        "returns; unchanged returns remain in the denominator."
    ),
    "market_above_ma20_fraction": (
        "Market context: fraction above own trailing 20-session adjusted-close "
        "mean among fresh stocks with complete positive prices."
    ),
    "market_above_ma60_fraction": (
        "Market context: fraction above own trailing 60-session adjusted-close "
        "mean among fresh stocks with complete positive prices."
    ),
    "market_high_low_balance_60d": (
        "Market context: (new 60-session highs minus new lows)/valid count, "
        "comparing today's close with the preceding 60 closes. Ties are neither."
    ),
    "market_breadth_change_5d": (
        "Market context: current up fraction minus its value five declared market "
        "sessions earlier; missing sessions are not compressed."
    ),
    "industry_return_dispersion_20d": (
        "Market context: population standard deviation across industries' "
        "contemporaneous-member equal-weight 20-session stock returns; at least "
        "two industries. No assumed bearish direction."
    ),
    "industry_positive_return_hhi_20d": (
        "Market context: HHI of positive industry mean 20-session returns, with "
        "equal industry weights. No positive industry is undefined; not "
        "market-cap contribution concentration."
    ),
    "industry_return_relative_20d": (
        "Stock context: own 20-session return minus leave-one-out equal-weight "
        "same-industry return, using PIT standard+level1 affiliation. "
        "Positive/negative identifies past leaders/laggards."
    ),
    "industry_turnover_relative": (
        "Stock context: own turnover divided by leave-one-out industry mean "
        "turnover. Zero peer activity or no peers is undefined."
    ),
    "industry_amount_share": (
        "Stock context: own amount / known nonnegative industry amount sum. An "
        "observed activity share, not investor identity or holdings."
    ),
    "industry_amount_hhi": (
        "Industry context broadcast to members: (sum(amount shares "
        "squared)-1/n)/(1-1/n), n valid fresh members>1. Zero total amount is "
        "undefined."
    ),
    "industry_activity_pressure": (
        "Industry context: industry share of market amount divided by industry "
        "share of valid current float cap, requiring the same jointly valid "
        "stocks. Unknown industries excluded from both totals."
    ),
    "industry_comovement_20d": (
        "Stock context: 20-complete-session correlation of own return with "
        "leave-one-out industry equal-weight return. Each date uses its "
        "contemporaneous PIT industry; constant series is undefined."
    ),
    "leader_breadth_fragility": (
        "User hypothesis: max(own 20-session return,0) * "
        "(1-market_above_ma20_fraction). Tests whether past winners are exposed "
        "when market participation is narrow; no validated return direction."
    ),
    "industry_isolation_20d": (
        "User hypothesis, industry context: max(own industry's mean 20-session "
        "stock return,0) * fraction of other industries with negative mean "
        "returns. Undefined without another valid industry; tests isolated "
        "leadership, not generic dispersion."
    ),
}

MARKET_CONTEXT_FACTORS = frozenset(
    name for name in DESCRIPTIONS if name.startswith("market_")
) | {"industry_return_dispersion_20d", "industry_positive_return_hhi_20d"}
INDUSTRY_CONTEXT_FACTORS = frozenset(
    {
        "industry_amount_hhi",
        "industry_activity_pressure",
        "industry_isolation_20d",
    }
)
# A common breadth multiplier is a state hypothesis, not a new stock ranking.
PRESERVE_AMPLITUDE_FACTORS = (
    MARKET_CONTEXT_FACTORS | INDUSTRY_CONTEXT_FACTORS | {"leader_breadth_fragility"}
)
FACTOR_UNITS = dict.fromkeys(DESCRIPTIONS, "dimensionless")
FACTOR_UNITS.update(
    dict.fromkeys(
        ("industry_return_dispersion_20d", "industry_return_relative_20d"),
        "decimal_return",
    )
)


def factor_definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set=FactorSet.STANDARD,
            family="market_structure",
            input_columns=(*INPUTS[name], "filling_flag", "is_a_share_scope"),
            output_column=name,
            required_input_data_id="market_structure_factors",
            required_windows=(_lookback(name) + 1,),
            lookback_window=_lookback(name) or None,
            description=description,
            timing_semantics="same_session_close_available_at_next_session",
            null_policy=(
                "preserve_nulls_per_dependency; full historical input universe required"
            ),
        )
        for name, description in DESCRIPTIONS.items()
    )


def calculate(panel, *, factor_columns=None, history_keys=("asset_id",)):
    """Use the supplied historical universe; never infer it from today's members.

    Market and industry contexts are broadcast for explicit stock interactions.
    They must not be interpreted as cross-sectional stock-selection scores.
    """
    requested = tuple(DESCRIPTIONS if factor_columns is None else factor_columns)
    if set(requested) - DESCRIPTIONS.keys():
        raise ValueError("unknown_market_structure_factor")
    keys = list(dict.fromkeys((*history_keys, "timestamp")))
    inputs = tuple(
        dict.fromkeys(column for name in requested for column in INPUTS[name])
    )
    frame = panel.loc[
        :,
        list(
            dict.fromkeys(
                ("asset_id", *keys, *inputs, "filling_flag", "is_a_share_scope")
            )
        ),
    ].copy()
    frame["timestamp"] = strict_timestamp_series(
        frame.timestamp, error="market_structure_timestamp_invalid"
    )
    if frame.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("market_structure_key_duplicate")
    for name in set(inputs) - set(_INDUSTRY):
        frame[name] = strict_finite_numeric_series(
            frame[name], error=f"market_structure_input_invalid:{name}", allow_null=True
        ).to_numpy(dtype="float64", na_value=np.nan)
    frame = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    fresh = frame.filling_flag.eq(0) & frame.is_a_share_scope.eq(True)
    for source, target in (
        ("period_return", "_r"),
        ("turnover", "_t"),
        ("amount", "_a"),
    ):
        if source in inputs:
            frame[target] = frame[source].where(
                fresh if source == "period_return" else fresh & frame[source].ge(0)
            )
    history = frame.groupby(list(history_keys), sort=False, dropna=False)
    if "adj_close" in inputs:
        frame["_c"] = frame.adj_close.where(frame.adj_close.gt(0))
        frame["_mom"] = (frame._c / history._c.shift(20) - 1).where(fresh)
    out = frame.loc[:, ["asset_id", "timestamp"]].copy()
    timestamps = frame.timestamp

    def market_mean(values):
        return values.groupby(timestamps, sort=False).transform("mean")

    if {"market_up_fraction", "market_breadth_change_5d"}.intersection(requested):
        out["market_up_fraction"] = market_mean(
            frame._r.gt(0).astype(float).where(frame._r.notna())
        )
    for window in (20, 60):
        if f"market_above_ma{window}_fraction" not in requested and not (
            window == 20 and "leader_breadth_fragility" in requested
        ):
            continue
        frame[f"_ma{window}"] = history._c.transform(
            lambda x, window=window: x.rolling(window, min_periods=window).mean()
        )
        valid = fresh & frame[f"_ma{window}"].notna() & frame._c.notna()
        above = frame._c.gt(frame[f"_ma{window}"]).astype(float).where(valid)
        out[f"market_above_ma{window}_fraction"] = market_mean(above)
    if "market_high_low_balance_60d" in requested:
        prior = history._c.shift()
        prior_groups = prior.groupby(
            [frame[key] for key in history_keys], sort=False, dropna=False
        )
        high = prior_groups.transform(lambda x: x.rolling(60, min_periods=60).max())
        low = prior_groups.transform(lambda x: x.rolling(60, min_periods=60).min())
        valid = fresh & frame._c.notna() & high.notna() & low.notna()
        high_low = frame._c.gt(high).astype(float) - frame._c.lt(low).astype(float)
        out["market_high_low_balance_60d"] = market_mean(high_low.where(valid))
    if "market_breadth_change_5d" in requested:
        breadth = out.groupby("timestamp", sort=True).market_up_fraction.first()
        out["market_breadth_change_5d"] = timestamps.map(breadth - breadth.shift(5))
    if "leader_breadth_fragility" in requested:
        out["leader_breadth_fragility"] = frame._mom.clip(lower=0) * (
            1 - out.market_above_ma20_fraction
        )

    if any(name.startswith("industry_") for name in requested):
        _industry_context(frame, out, requested, history_keys)
    return (
        out.loc[:, ["asset_id", "timestamp", *requested]]
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )


def _industry_context(frame, out, requested, history_keys):
    timestamps = frame.timestamp
    sector_columns = list(_INDUSTRY)
    known = frame[sector_columns].notna().all(axis=1) & frame[sector_columns].ne(
        ""
    ).all(axis=1)
    sector_keys = [frame[name].where(known) for name in sector_columns]
    groups = frame.groupby([timestamps, *sector_keys], sort=False, observed=True)
    index = pd.MultiIndex.from_arrays([timestamps, *sector_keys])
    if {
        "industry_return_dispersion_20d",
        "industry_positive_return_hhi_20d",
        "industry_isolation_20d",
    }.intersection(requested):
        means = groups._mom.mean()
        sectors_by_date = means.groupby(level=0)
        if "industry_return_dispersion_20d" in requested:
            dispersion = sectors_by_date.std(ddof=0).where(sectors_by_date.count() > 1)
            out["industry_return_dispersion_20d"] = timestamps.map(dispersion)
        if "industry_positive_return_hhi_20d" in requested:
            positive = means.clip(lower=0)
            total = positive.groupby(level=0).transform("sum")
            hhi = (
                (positive / total.where(total > 0))
                .pow(2)
                .groupby(level=0)
                .sum(min_count=1)
            )
            out["industry_positive_return_hhi_20d"] = timestamps.map(hhi)
        if "industry_isolation_20d" in requested:
            counts = sectors_by_date.transform("count")
            negatives = (
                means.lt(0)
                .astype(float)
                .where(means.notna())
                .groupby(level=0)
                .transform("sum")
            )
            peer_fraction = (negatives - means.lt(0)) / (counts - 1).where(counts > 1)
            out["industry_isolation_20d"] = (
                (means.clip(lower=0) * peer_fraction).reindex(index).to_numpy()
            )
    for source, target, ratio in (
        ("_mom", "industry_return_relative_20d", False),
        ("_t", "industry_turnover_relative", True),
        ("_r", "_loo_r", None),
    ):
        if target not in requested and not (
            target == "_loo_r" and "industry_comovement_20d" in requested
        ):
            continue
        values = frame[source]
        count = groups[source].transform("count") - values.notna().astype(int)
        peers = (groups[source].transform("sum") - values.fillna(0)) / count.where(
            count > 0
        )
        if ratio is None:
            frame[target] = peers
        else:
            out[target] = values / peers.where(peers > 0) if ratio else values - peers
    if {"industry_amount_share", "industry_amount_hhi"}.intersection(requested):
        amount_total = groups._a.transform("sum")
        shares = frame._a / amount_total.where(amount_total > 0)
        if "industry_amount_share" in requested:
            out["industry_amount_share"] = shares
        if "industry_amount_hhi" in requested:
            n = groups._a.transform("count")
            squares = (
                shares.pow(2)
                .groupby([timestamps, *sector_keys], sort=False)
                .transform("sum")
            )
            out["industry_amount_hhi"] = ((squares - 1 / n) / (1 - 1 / n)).where(
                (n > 1) & (amount_total > 0)
            )
    if "industry_activity_pressure" in requested:
        paired = known & frame._a.notna() & frame.float_market_cap.gt(0)
        paired_amount = frame._a.where(paired)
        paired_cap = frame.float_market_cap.where(paired)
        market_amount = paired_amount.groupby(timestamps).transform("sum")
        market_cap = paired_cap.groupby(timestamps).transform("sum")
        sector_amount = paired_amount.groupby(
            [timestamps, *sector_keys], sort=False
        ).transform("sum")
        sector_cap = paired_cap.groupby(
            [timestamps, *sector_keys], sort=False
        ).transform("sum")
        out["industry_activity_pressure"] = (
            sector_amount / market_amount.where(market_amount > 0)
        ) / (sector_cap / market_cap.where(market_cap > 0)).where(sector_cap > 0)
    if "industry_comovement_20d" in requested:
        correlation = pd.Series(np.nan, index=frame.index)
        for _, group in frame.groupby(list(history_keys), sort=False, dropna=False):
            correlation.loc[group.index] = (
                group._r.rolling(20, min_periods=20).corr(group._loo_r).to_numpy()
            )
        out["industry_comovement_20d"] = correlation
