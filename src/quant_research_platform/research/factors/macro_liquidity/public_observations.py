"""Use typed public disclosures in the existing release-history calculator."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_real_numeric_series

from .definitions import PUBLIC_SERIES


def observations_for_series(frame, series_names):
    """Return only release versions actually known, without a second persisted table.

    CFTC net shares use one contract's open interest. Futures-only and combined
    books never mix. FINRA's native month-start key denotes the full reference
    month; its economic observation date here is month-end.
    """
    blocks = []
    counts = {}
    if series_names:
        identities = {PUBLIC_SERIES[name][1] for name in series_names}
        fields = tuple(
            dict.fromkeys(
                field for name in series_names for field in PUBLIC_SERIES[name][2]
            )
        )
        frame = frame.loc[
            frame.asset_id.isin(identities),
            ["asset_id", "timestamp", "available_at", *fields],
        ].copy()
        for field in fields:
            raw = frame[field]
            numeric = strict_real_numeric_series(
                raw, error="public_macro_numeric_invalid", errors="coerce"
            )
            valid = np.isfinite(numeric) & numeric.ge(0)
            counts[field] = int((raw.notna() & ~valid).sum())
            frame[field] = numeric.where(valid)
    for name in series_names:
        table, identity, fields, _unit, _frequency, _transforms = PUBLIC_SERIES[name]
        work = frame.loc[
            frame.asset_id.eq(identity), ["timestamp", "available_at", *fields]
        ].copy()
        value = work[fields[0]]
        if table == "public_cftc_positions":
            value = value - work[fields[1]]
            if len(fields) == 3:
                value /= work[fields[2]].where(work[fields[2]].gt(0))
        observed = pd.to_datetime(work.timestamp).astype("datetime64[ns]")
        if table == "public_finra_margin":
            observed += pd.offsets.MonthEnd(0)
        available = pd.to_datetime(work.available_at, utc=True).astype(
            "datetime64[ns, UTC]"
        )
        block = pd.DataFrame(
            {
                "series_id": name,
                "observation_date": observed,
                "timestamp": available,
                "value": value.where(np.isfinite(value)),
            }
        )
        blocks.append(block.loc[available.notna()])
    if not blocks:
        result = pd.DataFrame(
            {
                "series_id": pd.Series(dtype="str"),
                "observation_date": pd.Series(dtype="datetime64[ns]"),
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )
    else:
        values = pd.concat(blocks, ignore_index=True)
        keys = ["series_id", "observation_date", "timestamp"]
        grouped = values.groupby(keys, sort=False, dropna=False).value
        # Conflicting snapshots with no distinct historical clock are unresolved.
        result = (
            grouped.first()
            .where(grouped.nunique(dropna=False).le(1))
            .rename("value")
            .reset_index()
        )
    result.attrs["input_invalid_numeric_cells"] = counts
    return result
