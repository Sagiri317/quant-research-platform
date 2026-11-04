"""Future observed quote returns and their actual maturity timestamps."""

import numpy as np
import pandas as pd

from quant_research_platform.research.protocols import a_share_hft as protocol


def compute_labels(panel, definitions):
    output = []
    for _, group in panel.groupby(["asset_id", "segment_id"], sort=False):
        work = group.sort_values("timestamp").reset_index(drop=True)
        values = work.loc[:, protocol.KEY].copy()
        times = work.timestamp.to_numpy(dtype="datetime64[ns]")
        mid = work.mid.to_numpy(dtype=float)
        for definition in definitions:
            future = np.searchsorted(
                times, times + np.timedelta64(definition.horizon, "ms")
            )
            valid = (future < len(times)) & work.eligible.to_numpy()
            future = np.minimum(future, len(times) - 1)
            returns = mid[future] / mid - 1
            valid &= np.isfinite(returns)
            values[definition.output_column] = np.where(
                valid,
                np.sign(returns)
                if definition.family == "future_mid_direction"
                else returns,
                np.nan,
            )
            values[definition.available_at_column] = pd.to_datetime(
                np.where(valid, times[future], np.datetime64("NaT", "ns"))
            )
        output.append(values)
    return pd.concat(output, ignore_index=True).sort_values(
        ["timestamp", "asset_id"], kind="stable"
    )
