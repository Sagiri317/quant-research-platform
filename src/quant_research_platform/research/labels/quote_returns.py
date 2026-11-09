"""Future observed quote returns and their actual maturity timestamps."""

import numpy as np
import pandas as pd

from quant_research_platform.research.protocols import a_share_hft as protocol


class QuoteReturnLookup:
    """Index price-only quote arrays once, including a streamed session."""

    def __init__(self, frames, definitions):
        self.definitions = tuple(definitions)
        pieces = {}
        for panel in frames:
            for key, group in panel.groupby(["asset_id", "segment_id"], sort=False):
                times, prices = pieces.setdefault(key, ([], []))
                times.append(group.timestamp.to_numpy(dtype="datetime64[ns]"))
                prices.append(group.mid.to_numpy(dtype=float))
        self.quotes, self.observed_times = {}, {}
        for key, (times, prices) in pieces.items():
            times, prices = np.concatenate(times), np.concatenate(prices)
            order = np.argsort(times, kind="stable")
            times, prices = times[order], prices[order]
            self.observed_times[key] = times
            valid = np.isfinite(prices) & (prices > 0)
            self.quotes[key] = times[valid], prices[valid]

    def labels(
        self,
        decisions,
        *,
        sampled=False,
        max_lateness_seconds=None,
        visibility_delay=None,
    ):
        """Price from each decision time, never from its older as-of quote."""
        visibility_delay = (
            pd.Timedelta(0) if visibility_delay is None else visibility_delay
        )
        decisions = decisions.reset_index(drop=True)
        values = decisions.loc[:, protocol.KEY].copy()
        for definition in self.definitions:
            values[definition.output_column] = np.nan
            values[definition.available_at_column] = pd.Series(
                pd.NaT, index=values.index, dtype="datetime64[ns]"
            )
            if sampled:
                values[definition.output_column + "_lateness_seconds"] = np.nan
        for key, group in decisions.groupby(["asset_id", "segment_id"], sort=False):
            if key not in self.quotes:
                continue
            times, prices = self.quotes[key]
            starts = group.timestamp.to_numpy(dtype="datetime64[ns]")
            mid = group.mid.to_numpy(dtype=float)
            valid_mid = np.isfinite(mid) & (mid > 0)
            for definition in self.definitions:
                target = starts + np.timedelta64(definition.horizon, "ms")
                future = np.searchsorted(times, target)
                found = (future < len(times)) & valid_mid
                rows = group.index[found]
                selected = future[found]
                lateness = (times[selected] - target[found]) / np.timedelta64(1, "s")
                returns = prices[selected] / mid[found] - 1
                valid = np.isfinite(returns)
                if max_lateness_seconds is not None:
                    valid &= lateness <= max_lateness_seconds
                values.loc[rows[valid], definition.output_column] = (
                    np.sign(returns[valid])
                    if definition.family == "future_mid_direction"
                    else returns[valid]
                )
                # Retain observed maturity even when a late quote fails tolerance.
                values.loc[rows, definition.available_at_column] = (
                    times[selected] + visibility_delay.to_timedelta64()
                )
                if sampled:
                    values.loc[rows, definition.output_column + "_lateness_seconds"] = (
                        lateness
                    )
        return values.sort_values(["timestamp", "asset_id"], kind="stable").reset_index(
            drop=True
        )


def compute_labels(panel, definitions, *, decisions=None, max_lateness_seconds=None):
    return QuoteReturnLookup((panel,), definitions).labels(
        panel if decisions is None else decisions,
        sampled=decisions is not None,
        max_lateness_seconds=max_lateness_seconds,
    )
