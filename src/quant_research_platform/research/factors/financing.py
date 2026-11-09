"""PIT stock leverage proxies; balances, activity and liquidation are distinct."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.validation import (
    strict_real_numeric_series,
    strict_string_series,
)

RECIPES = {
    "financing_balance_to_float_cap": (
        "financing_balance",
        "float_market_cap",
        "Known same-observation financing balance / float market cap.",
    ),
    "financing_buy_intensity": (
        "financing_buy_amount",
        "amount",
        "Financing purchases / SAME observation session's total traded amount.",
    ),
    "financing_repayment_intensity": (
        "financing_repayment_amount",
        "amount",
        "Reported financing repayments / same-session total traded amount; "
        "missing repayments are not inferred from a different balance identity.",
    ),
    "lending_balance_to_float_cap": (
        "lending_balance",
        "float_market_cap",
        "Known lending balance / same-observation float cap, not short interest.",
    ),
    "financing_balance_change_1d": (
        "financing_balance",
        "",
        "Balance / exact preceding market-session balance - 1.",
    ),
    "financing_balance_change_5d": (
        "financing_balance",
        "",
        "Balance / balance five market sessions earlier - 1.",
    ),
    "leveraged_liquidity_pressure": (
        "financing_balance",
        "float_market_cap",
        "(financing balance / float cap) * peak_drawdown_20d * "
        "max(amihud_20d / amihud_60d - 1, 0); observable leverage and liquidity "
        "pressure, not proof of forced liquidation.",
    ),
}
BALANCE_CHANGE_WINDOWS = {
    "financing_balance_change_1d": 1,
    "financing_balance_change_5d": 5,
}


def decision_columns(columns):
    required = {"signal_session", "listing_date", "eligible"}
    required.update(RECIPES[name][1] for name in columns if RECIPES[name][1])
    if "leveraged_liquidity_pressure" in columns:
        required.update(("peak_drawdown_20d", "amihud_20d", "amihud_60d"))
    return tuple(sorted(required))


def factor_definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="standard",
            family="financing",
            input_columns=(recipe[0], "available_at", *decision_columns((name,))),
            output_column=name,
            required_input_data_id="financing_factors",
            required_windows=(BALANCE_CHANGE_WINDOWS[name],)
            if name in BALANCE_CHANGE_WINDOWS
            else (),
            description=recipe[2],
            timing_semantics="publication_available_before_explicit_decision",
            null_policy="per_field_and_exact_observation_session; no_stale_fill",
        )
        for name, recipe in RECIPES.items()
    )


def calculate(
    decisions,
    observations,
    *,
    factor_columns=None,
    sessions=None,
    observation_lag_sessions=0,
):
    """Match each report field/version to the known daily signal session.

    Decisions use UTC instants and retain signal_session/listing_date. A missing
    current report does not silently become yesterday's balance. Exact lagged
    observations come from the declared market calendar, never compressed rows.
    """
    columns = tuple(RECIPES if factor_columns is None else factor_columns)
    if type(observation_lag_sessions) is not int or observation_lag_sessions not in (
        0,
        1,
    ):
        raise ValueError("financing_observation_lag_invalid")
    if not set(columns) <= RECIPES.keys():
        raise ValueError("unknown_financing_factor")
    required = {"market_id", "asset_id", "timestamp", *decision_columns(columns)}
    if not required <= set(decisions):
        raise ValueError("financing_decision_inputs_missing")
    frame = decisions.loc[:, list(required)].reset_index(drop=True).copy()
    frame["asset_id"] = strict_string_series(
        frame.asset_id, error="financing_asset_id_invalid"
    )
    if frame.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("financing_decision_key_duplicate")
    counts = {}
    numeric_context = required - {
        "market_id",
        "asset_id",
        "timestamp",
        "signal_session",
        "listing_date",
        "eligible",
    }
    for field in numeric_context:
        raw = frame[field]
        numeric = strict_real_numeric_series(
            raw, error="financing_numeric_invalid", errors="coerce"
        )
        valid = np.isfinite(numeric) & numeric.ge(0)
        counts[field] = int((raw.notna() & ~valid).sum())
        frame[field] = numeric.where(valid)
    cutoffs = (
        pd.to_datetime(frame.timestamp, utc=True)
        .astype("datetime64[ns, UTC]")
        .dt.tz_localize(None)
    )
    dates = pd.to_datetime(frame.signal_session).astype("datetime64[ns]").dt.normalize()
    listed = pd.to_datetime(frame.listing_date).astype("datetime64[ns]").dt.normalize()
    axis = (
        None
        if sessions is None
        else pd.DatetimeIndex(sessions).as_unit("ns").sort_values().unique()
    )

    def observation_dates(lag):
        if not lag:
            return dates
        target = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        if axis is not None:
            positions = axis.get_indexer(dates)
            valid = positions >= lag
            target.loc[valid] = axis.take(positions[valid] - lag)
        return target

    base_dates = observation_dates(observation_lag_sessions)
    denominators = tuple(
        dict.fromkeys(RECIPES[name][1] for name in columns if RECIPES[name][1])
    )
    historical = None
    if observation_lag_sessions and denominators:
        reference = frame.assign(signal_session=dates, listing_date=listed).groupby(
            ["asset_id", "signal_session", "listing_date"],
            sort=False,
        )[list(denominators)]
        historical = (
            reference.first()
            .where(reference.nunique(dropna=False).le(1))
            .reindex(pd.MultiIndex.from_arrays([frame.asset_id, base_dates, listed]))
            .set_axis(frame.index)
        )
    fields = tuple(dict.fromkeys(RECIPES[name][0] for name in columns))
    facts = observations.loc[
        :, ["asset_id", "timestamp", "available_at", *fields]
    ].copy()
    facts["asset_id"] = strict_string_series(
        facts.asset_id, error="financing_asset_id_invalid"
    )
    facts["_obs"] = (
        pd.to_datetime(facts.pop("timestamp")).astype("datetime64[ns]").dt.normalize()
    )
    facts["_clock"] = (
        pd.to_datetime(facts.pop("available_at"), utc=True)
        .astype("datetime64[ns, UTC]")
        .dt.tz_localize(None)
    )
    facts = facts.loc[facts._clock.notna() & facts._obs.notna()]
    # Completed daily reports cannot be used before the local after-hours close.
    closing = facts._obs + pd.Timedelta(hours=7, minutes=30)
    facts["_clock"] = facts._clock.where(facts._clock.ge(closing), closing)
    for field in fields:
        raw = facts[field]
        numeric = strict_real_numeric_series(
            raw, error="financing_numeric_invalid", errors="coerce"
        )
        valid = np.isfinite(numeric) & numeric.ge(0)
        counts[field] = int((raw.notna() & ~valid).sum())
        facts[field] = numeric.where(valid)
    key = ["asset_id", "_obs", "_clock"]
    # Identical archives may overlap; conflicting cells stay unknown at that clock.
    grouped = facts.groupby(key, sort=False, dropna=False)[list(fields)]
    values = grouped.first().where(grouped.nunique(dropna=False).le(1)).reset_index()
    # pandas group keys may infer StringDtype when reset, unlike canonical PANEL keys.
    values["asset_id"] = values.asset_id.astype("object")
    values = values.sort_values("_clock", kind="mergesort")
    cache = {}

    def known(field, lag=0):
        if (field, lag) in cache:
            return cache[field, lag]
        target = observation_dates(lag + observation_lag_sessions)
        valid = target.notna() & listed.notna() & target.ge(listed)
        left = (
            pd.DataFrame(
                {
                    "asset_id": frame.asset_id,
                    "_obs": target,
                    "_cutoff": cutoffs,
                    "_row": frame.index,
                }
            )
            .loc[valid]
            .sort_values("_cutoff", kind="mergesort")
        )
        result = pd.Series(np.nan, index=frame.index, dtype="float64")
        if len(left) and len(values):
            joined = pd.merge_asof(
                left,
                values[[*key, field]],
                left_on="_cutoff",
                right_on="_clock",
                by=["asset_id", "_obs"],
                direction="backward",
            )
            result.loc[joined._row] = joined[field].to_numpy()
        cache[field, lag] = result
        return result

    output = (
        decisions.loc[:, ["market_id", "asset_id", "timestamp"]]
        .reset_index(drop=True)
        .copy()
    )
    for name in columns:
        field, denominator, _ = RECIPES[name]
        if denominator:
            divisor = (
                historical[denominator]
                if historical is not None
                else frame[denominator]
            )
            result = known(field) / divisor.where(divisor.gt(0))
        else:
            lag = BALANCE_CHANGE_WINDOWS[name]
            baseline = known(field, lag)
            result = known(field) / baseline.where(baseline.gt(0)) - 1
        if name == "leveraged_liquidity_pressure":
            descriptors = {
                field: frame[field]
                for field in ("peak_drawdown_20d", "amihud_20d", "amihud_60d")
            }
            pressure = (
                descriptors["amihud_20d"]
                / descriptors["amihud_60d"].where(descriptors["amihud_60d"].gt(0))
                - 1
            ).clip(lower=0)
            result *= (
                descriptors["peak_drawdown_20d"].where(
                    descriptors["peak_drawdown_20d"].ge(0)
                )
                * pressure
            )
        output[name] = result.where(np.isfinite(result) & frame.eligible.fillna(False))
    output.attrs["input_invalid_numeric_cells"] = counts
    return output
