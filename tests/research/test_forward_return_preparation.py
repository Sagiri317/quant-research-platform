"""Shared entry preparation preserves each horizon's real session endpoints."""

import numpy as np
import pandas as pd

from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)


def test_shared_entries_preserve_sparse_quotes_and_horizon_availability(monkeypatch):
    sessions = pd.bdate_range("2024-01-02", periods=12)
    frame = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": day,
                "adj_open": 10.0 + position,
                "can_enter": position != 3,
            }
            for asset in ("a", "b")
            for position, day in enumerate(sessions)
            if not (asset == "b" and position in (2, 7))
        ]
    )
    frame.loc[(frame.asset_id == "a") & frame.timestamp.eq(sessions[5]), "adj_open"] = (
        np.nan
    )
    lookup = pd.MultiIndex.get_indexer
    calls = []

    def record(self, target, *args, **kwargs):
        calls.append(len(target))
        return lookup(self, target, *args, **kwargs)

    monkeypatch.setattr(pd.MultiIndex, "get_indexer", record)
    result = add_forward_return_labels(frame, horizons=(1, 3, 5), sessions=sessions)
    assert len(calls) == 4  # One common entry lookup and one exit lookup per horizon.
    prices = frame.set_index(["asset_id", "timestamp"]).adj_open
    for row in result.itertuples(index=False):
        position = sessions.get_loc(row.timestamp)
        for horizon in (1, 3, 5):
            expected, available = np.nan, pd.NaT
            if row.can_enter and position + 1 + horizon < len(sessions):
                entry = prices.get((row.asset_id, sessions[position + 1]), np.nan)
                exit_ = prices.get(
                    (row.asset_id, sessions[position + 1 + horizon]), np.nan
                )
                if (
                    np.isfinite(entry)
                    and entry > 0
                    and np.isfinite(exit_)
                    and exit_ > 0
                ):
                    expected = (exit_ - entry) / entry
                    available = sessions[position + 1 + horizon]
            value = getattr(row, f"forward_return_{horizon}d")
            time = getattr(row, f"forward_return_{horizon}d_available_at")
            assert np.isnan(value) if np.isnan(expected) else value == expected
            assert pd.isna(time) if pd.isna(available) else time == available
