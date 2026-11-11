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
    observed = {date: block for date, block in risk.groupby("timestamp", sort=False)}
    for date, at in decisions.groupby("signal_session", sort=False).groups.items():
        if date not in observed:
            continue
        r = observed[date].copy()
        r = r.loc[r.underlying_id.isin(decisions.asset_id.unique())]
        r["available_at"] = pd.to_datetime(r.available_at, utc=True).dt.tz_localize(
            None
        )
        r["asset_id"] = r.asset_id + ":" + r.contract_id
        decision = decisions.loc[at, "timestamp"].min()
        r = r.loc[r.available_at.notna() & r.available_at.le(decision)]
        if r.empty:
            continue
        query = r[["asset_id"]].assign(timestamp=decision)
        known = latest_observations(meta, query, columns=fields)
        for field in fields:
            r[field] = known[field].to_numpy()
        r["expiry"] = pd.to_datetime(r.expiry)
        r = r.loc[r.implied_volatility.gt(0) & r.strike.gt(0) & r.expiry.gt(date)]
        for i in at:
            row = decisions.loc[i]
            own = r.loc[r.underlying_id.eq(row.asset_id)]
            curve = []
            for expiry, block in own.groupby("expiry", sort=False):
                surface = (
                    (block.implied_volatility**2)
                    .groupby([block.strike, block.option_type])
                    .mean()
                    .unstack()
                )
                if not {"认购", "认沽"} <= set(surface):
                    continue
                surface = surface.loc[surface[["认购", "认沽"]].notna().all(axis=1)]
                if surface.empty or not row.close > 0:
                    continue
                strike = np.abs(
                    np.log(surface.index.to_numpy(float) / row.close)
                ).argmin()
                variance = surface.iloc[strike][["认购", "认沽"]].mean()
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
