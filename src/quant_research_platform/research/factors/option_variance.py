"""Same-underlying ATM implied variance with explicit expiry interpolation."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.asof import (
    latest_observations,
    observation_versions,
)


def implied_variances(decisions, risk, contracts):
    """Interpolate total ATM variance to fixed 20/252 and 60/252 year tenors."""
    names = [f"implied_variance_{width}d" for width in (20, 60)]
    result = pd.DataFrame(np.nan, index=decisions.index, columns=names)
    meta = contracts.copy()
    meta["asset_id"] = meta.security_id + ":" + meta.contract_id
    meta["available_at"] = pd.concat(
        [
            pd.to_datetime(meta.available_at, utc=True).dt.tz_localize(None),
            pd.to_datetime(meta.timestamp) + pd.Timedelta(hours=7),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    meta = meta.rename(columns={"expiry_date": "expiry", "exercise_price": "strike"})
    fields = ("expiry", "strike", "underlying_id")
    cutoffs = (
        decisions[["signal_session", "timestamp"]]
        .drop_duplicates()
        .set_axis(["timestamp", "_decision"], axis=1)
    )
    risk = observation_versions(
        risk,
        keys=("asset_id", "contract_id", "timestamp"),
        columns=("underlying_id", "option_type", "implied_volatility"),
    )
    visible = risk.merge(cutoffs, on="timestamp", how="inner")
    visible["available_at"] = pd.to_datetime(
        visible.available_at, utc=True
    ).dt.tz_localize(None)
    visible = visible.loc[
        visible.available_at.notna() & visible.available_at.le(visible._decision)
    ]
    visible = visible.sort_values("available_at", kind="stable").drop_duplicates(
        ["asset_id", "contract_id", "timestamp", "_decision"], keep="last"
    )
    visible = visible.loc[visible.underlying_id.isin(decisions.asset_id.unique())]
    if visible.empty:
        return result
    visible["asset_id"] = visible.asset_id + ":" + visible.contract_id
    query = visible[["asset_id"]].assign(timestamp=visible._decision)
    known = latest_observations(meta, query, columns=fields)
    visible[["expiry", "strike"]] = known[["expiry", "strike"]]
    visible = visible.loc[visible.underlying_id.eq(known.underlying_id)].copy()
    visible["expiry"] = pd.to_datetime(visible.expiry)
    visible = visible.loc[
        visible.implied_volatility.gt(0)
        & visible.strike.gt(0)
        & visible.expiry.gt(visible.timestamp)
    ]
    surface = (
        (visible.implied_volatility**2)
        .groupby(
            [
                visible.timestamp,
                visible._decision,
                visible.underlying_id,
                visible.expiry,
                visible.strike,
                visible.option_type,
            ]
        )
        .mean()
        .unstack("option_type")
        .reindex(columns=["认购", "认沽"])
        .dropna()
        .mean(axis=1)
        .rename("variance")
        .reset_index()
    )
    surfaces = surface.groupby(["timestamp", "_decision", "underlying_id"], sort=False)
    priced = decisions.close.gt(0) & decisions.asset_id.isin(surface.underlying_id)
    for row in decisions.loc[priced].itertuples():
        key = (row.signal_session, row.timestamp, row.asset_id)
        if key not in surfaces.indices:
            continue
        own = surfaces.get_group(key)
        curve = []
        for expiry, block in own.groupby("expiry", sort=False):
            strike = np.abs(np.log(block.strike.to_numpy(float) / row.close)).argmin()
            variance = block.variance.iloc[strike]
            curve.append(((expiry - row.signal_session).days / 365, variance))
        curve.sort()
        tenor, variance = np.asarray(curve).T
        for width, name in zip((20, 60), names, strict=True):
            horizon = width / 252
            if tenor[0] <= horizon <= tenor[-1]:
                result.loc[row.Index, name] = (
                    np.interp(horizon, tenor, tenor * variance) / horizon
                )
    return result
