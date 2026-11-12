"""Same-underlying ATM implied variance with explicit expiry interpolation."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.asof import latest_observations


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
    cutoffs = decisions.groupby("signal_session", sort=False).timestamp.min()
    visible = risk.loc[
        risk.timestamp.isin(cutoffs.index)
        & risk.underlying_id.isin(decisions.asset_id.unique())
    ].copy()
    visible["available_at"] = pd.to_datetime(
        visible.available_at, utc=True
    ).dt.tz_localize(None)
    visible["_decision"] = visible.timestamp.map(cutoffs)
    visible = visible.loc[
        visible.available_at.notna() & visible.available_at.le(visible._decision)
    ]
    if visible.empty:
        return result
    visible["asset_id"] = visible.asset_id + ":" + visible.contract_id
    query = visible[["asset_id"]].assign(timestamp=visible._decision)
    known = latest_observations(meta, query, columns=fields)
    for field in fields:
        visible[field] = known[field].to_numpy()
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
    observed = {date: block for date, block in surface.groupby("timestamp", sort=False)}
    for date, at in decisions.groupby("signal_session", sort=False).groups.items():
        if date not in observed:
            continue
        r = observed[date]
        for i in at:
            row = decisions.loc[i]
            own = r.loc[r.underlying_id.eq(row.asset_id)]
            curve = []
            for expiry, block in own.groupby("expiry", sort=False):
                if not row.close > 0:
                    continue
                strike = np.abs(
                    np.log(block.strike.to_numpy(float) / row.close)
                ).argmin()
                variance = block.variance.iloc[strike]
                curve.append(((expiry - date).days / 365, variance))
            if not curve:
                continue
            curve.sort()
            tenor, variance = np.asarray(curve).T
            for width, name in zip((20, 60), names, strict=True):
                horizon = width / 252
                if tenor[0] <= horizon <= tenor[-1]:
                    result.loc[i, name] = (
                        np.interp(horizon, tenor, tenor * variance) / horizon
                    )
    return result
