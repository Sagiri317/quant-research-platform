"""Evaluate a frozen Hawkes model on the later observed event history."""

import pandas as pd

from quant_research_platform.markets.implementations.a_share.level2 import (
    observed_segments,
)
from quant_research_platform.research.models.hawkes import intensities, trade_events
from quant_research_platform.research.protocols.a_share_hft import KEY


def compute_hawkes_factors(panel, trades, model):
    params, cutoff = model
    output = []
    for quotes, selected in observed_segments(panel, trades=trades):
        start = quotes.timestamp.iloc[0]
        times, counts = trade_events(selected["trades"], start)
        queries = (quotes.timestamp - start).dt.total_seconds().to_numpy()
        intensity = intensities(times, counts, queries, params)
        total = intensity.sum(axis=1)
        factors = pd.DataFrame(
            {
                "hawkes_buy_intensity": intensity[:, 0],
                "hawkes_sell_intensity": intensity[:, 1],
                "hawkes_intensity_imbalance": (intensity[:, 0] - intensity[:, 1])
                / total,
                "hawkes_excitation_fraction": 1 - params[0].sum() / total,
            }
        )
        frame = pd.concat([quotes.loc[:, KEY], factors], axis=1)
        output.append(frame.loc[frame.timestamp.gt(cutoff)])
    return pd.concat(output, ignore_index=True).sort_values(
        ["timestamp", "asset_id"], kind="stable"
    )
