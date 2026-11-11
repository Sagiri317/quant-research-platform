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
RECIPES.update(
    {
        "price_acceleration_x_leverage_expansion": (
            "financing_balance",
            "",
            (
                "Positive log-price up-acceleration over 20 sessions times "
                "positive 5-session growth in disclosed financing balance; "
                "scale-free leverage expansion, not a nominal balance comparison."
            ),
        ),
        "short_interest_to_float_shares": (
            "lending_quantity",
            "a_float_shares",
            (
                "Reported stock-lending quantity / same-observation float shares;"
                " covers disclosed margin lending, not every short position."
            ),
        ),
        "lending_days_to_cover_20d": (
            "lending_quantity",
            "avg_volume_20d",
            (
                "Reported lending quantity / same-observation trailing-20 mean "
                "daily share volume; days of observed trading capacity, not "
                "actual liquidation time."
            ),
        ),
        "lending_repayment_intensity": (
            "lending_repayment_quantity",
            "volume",
            (
                "Reported returned lending units / same-observation share volume;"
                " direct returns and market purchases are not distinguished."
            ),
        ),
        **{
            f"lending_net_cover_fraction_{w}d": (
                "lending_quantity",
                "",
                "max(prior lending quantity-current lending quantity,0) / "
                f"quantity {w} market sessions earlier; net quantity reduction, "
                "not gross buying.",
            )
            for w in (1, 5)
        },
        "short_squeeze_pressure": (
            "lending_quantity",
            "avg_volume_20d",
            (
                "Disclosed days-to-cover times positive observed 5-session return"
                " times net 5-session lending reduction; a squeeze proxy, not "
                "observed margin calls."
            ),
        ),
        "long_deleveraging_pressure": (
            "financing_balance",
            "avg_amount_20d",
            (
                "Positive 5-session reduction in disclosed financing balance / "
                "average daily amount, times negative observed 5-session return; "
                "a deleveraging proxy, not identified forced sales."
            ),
        ),
        "financing_buy_after_drawdown": (
            "financing_buy_amount",
            "amount",
            (
                "Financing buying / same-observation amount times positive "
                "20-session drawdown; no account-level reentry is inferred."
            ),
        ),
    }
)
BALANCE_CHANGE_WINDOWS.update(
    {
        "lending_net_cover_fraction_1d": 1,
        "lending_net_cover_fraction_5d": 5,
        "short_squeeze_pressure": 5,
        "long_deleveraging_pressure": 5,
        "price_acceleration_x_leverage_expansion": 5,
    }
)
FACTOR_UNITS = dict.fromkeys(RECIPES, "dimensionless")
FACTOR_UNITS["lending_days_to_cover_20d"] = "trading_days"
REPORT_IDENTITY = ("asset_id", "timestamp", "source", "available_at")
SOURCE_PRIORITY = (("sse_margin", "szse_margin"), ("eastmoney_margin_detail",))


def decision_columns(columns):
    required = {"signal_session", "listing_date", "eligible"}
    if "price_acceleration_x_leverage_expansion" in columns:
        required.add("price_up_acceleration_20d")
    required.update(RECIPES[name][1] for name in columns if RECIPES[name][1])
    if "leveraged_liquidity_pressure" in columns:
        required.update(("peak_drawdown_20d", "amihud_20d", "amihud_60d"))
    if {"short_squeeze_pressure", "long_deleveraging_pressure"}.intersection(columns):
        required.add("reversal_5d")
    if "financing_buy_after_drawdown" in columns:
        required.add("peak_drawdown_20d")
    return tuple(sorted(required))


def factor_definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="standard",
            family="financing",
            input_columns=(
                recipe[0],
                "source",
                "available_at",
                *decision_columns((name,)),
            ),
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
    session_successors=None,
    observation_lag_sessions=0,
):
    """Match each report field/version to the known daily signal session.

    Decisions use UTC instants and retain signal_session/listing_date. A missing
    current report does not silently become yesterday's balance. Exact lagged
    observations come from the declared market calendar, never compressed rows.
    Standalone sessions declare a complete axis; native callers supply proven
    session_successors so unknown calendar dates cannot become closed days.
    Resolve each source's available version before applying official priority;
    supplements fill unreported official cells, never disputed official values.
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
    mandatory = {
        "market_id",
        "asset_id",
        "timestamp",
        "signal_session",
        "listing_date",
        "eligible",
    }
    if not mandatory <= set(decisions):
        raise ValueError("financing_decision_inputs_missing")
    missing_context = sorted(required - set(decisions))
    frame = decisions.reindex(columns=sorted(required)).reset_index(drop=True).copy()
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
        valid = np.isfinite(numeric)
        if field != "reversal_5d":
            valid &= numeric.ge(0)
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
    gaps = None
    if axis is not None and session_successors is not None:
        proved = session_successors.reindex(axis[:-1]).eq(axis[1:]).to_numpy()
        gaps = np.r_[0, np.cumsum(~proved)]

    def observation_dates(lag):
        if not lag:
            return dates
        target = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        if axis is not None:
            positions = axis.get_indexer(dates)
            valid = positions >= lag
            if gaps is not None:
                current = positions[valid]
                valid[valid] = gaps[current] == gaps[current - lag]
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
    if not set(REPORT_IDENTITY) <= set(observations):
        raise ValueError("financing_report_identity_missing")
    facts = observations.reindex(columns=[*REPORT_IDENTITY, *fields]).copy()
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
    if not facts.source.isin(sum(SOURCE_PRIORITY, ())).all():
        raise ValueError("financing_source_unsupported")
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
    key = ["asset_id", "_obs", "source", "_clock"]
    # Keep conflicts distinct from unreported fields: neither stale versions nor
    # a lower-authority supplement may fill a disputed official cell.
    grouped = facts.groupby(key, sort=False, dropna=False)[list(fields)]
    conflicts = grouped.nunique(dropna=False).gt(1).add_suffix("_conflict")
    conflict_columns = list(conflicts)
    values = grouped.first().join(conflicts).reset_index()
    # pandas group keys may infer StringDtype when reset, unlike canonical PANEL keys.
    values["asset_id"] = values.asset_id.astype("object")
    values = values.sort_values("_clock", kind="mergesort")
    versions = dict(tuple(values.groupby("source", sort=False)))
    cache = {}

    def known(field, lag=0):
        if lag in cache:
            return cache[lag][field]
        selected = fields
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
        result = pd.DataFrame(np.nan, index=frame.index, columns=selected)
        if len(left) and len(values):
            chosen = np.full((len(left), len(selected)), np.nan)
            resolved = np.zeros(chosen.shape, dtype=bool)
            for sources in SOURCE_PRIORITY:
                current = np.full(chosen.shape, np.nan)
                disputed = np.zeros(chosen.shape, dtype=bool)
                for source in sources:
                    if source not in versions:
                        continue
                    joined = pd.merge_asof(
                        left,
                        versions[source],
                        left_on="_cutoff",
                        right_on="_clock",
                        by=["asset_id", "_obs"],
                        direction="backward",
                    )
                    incoming = joined[list(selected)].to_numpy()
                    disputed |= joined[conflict_columns].fillna(False).to_numpy(
                        dtype=bool
                    ) | (
                        np.isfinite(current)
                        & np.isfinite(incoming)
                        & (current != incoming)
                    )
                    current = np.where(np.isnan(current), incoming, current)
                chosen = np.where(~resolved & ~disputed, current, chosen)
                resolved |= disputed | np.isfinite(current)
            result.loc[left._row] = chosen
        cache[lag] = result
        return result[field]

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
        if name.startswith("lending_net_cover_fraction_"):
            result = (-result).clip(lower=0)
        elif name == "short_squeeze_pressure":
            baseline = known(field, 5)
            reduction = ((baseline - known(field)) / baseline.where(baseline > 0)).clip(
                lower=0
            )
            result *= (-frame.reversal_5d).clip(lower=0) * reduction
        elif name == "long_deleveraging_pressure":
            divisor = (
                historical[denominator]
                if historical is not None
                else frame[denominator]
            )
            result = (
                (known(field, 5) - known(field)).clip(lower=0)
                / divisor.where(divisor > 0)
                * frame.reversal_5d.clip(lower=0)
            )
        elif name == "financing_buy_after_drawdown":
            result *= frame.peak_drawdown_20d
        elif name == "price_acceleration_x_leverage_expansion":
            result = result.clip(lower=0) * frame.price_up_acceleration_20d
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
    output.attrs["missing_context_columns"] = missing_context
    return output
