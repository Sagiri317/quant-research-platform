"""Merge observed Level-2 streams without claiming a shared exchange sequence."""

from heapq import merge

import pandas as pd


def observed_segments(panel, **streams):
    """Slice only observed continuous quote intervals; never bridge an archive gap."""
    by_asset = {
        name: {
            asset: group.sort_values(["timestamp", "source_row"], kind="stable")
            for asset, group in frame.groupby("asset_id", sort=False)
        }
        for name, frame in streams.items()
    }
    for (asset, _, _), group in panel.groupby(
        ["asset_id", "session_date", "segment_id"], sort=False
    ):
        work = group.sort_values("timestamp").reset_index(drop=True)
        selected = {}
        for name, frame in streams.items():
            events = by_asset[name].get(asset, frame.iloc[:0])
            start = events.timestamp.searchsorted(work.timestamp.iloc[0], side="left")
            stop = events.timestamp.searchsorted(work.timestamp.iloc[-1], side="right")
            selected[name] = events.iloc[start:stop]
        yield work, selected


def order_reference(value):
    """Zero, missing, and the unsigned sentinel do not identify an order."""
    return None if pd.isna(value) or value <= 0 or value == 2**64 - 1 else int(value)


def first_unknown_event_time(*streams):
    """Locate the causal barrier for recursive histories within one segment."""
    return pd.concat(
        [frame.loc[frame.event_kind.eq("unknown"), "timestamp"] for frame in streams]
    ).min()


def iter_level2_events(orders, trades):
    """Tie convention: trade stream before order stream, then each source's row order.

    A Shanghai remainder may be recorded after its immediate fills. Same-millisecond
    cross-stream order is assumed, so lifecycle age at that timestamp is undefined.
    """

    def stream(frame, priority):
        for row in frame.sort_values(
            ["timestamp", "source_row"], kind="stable"
        ).itertuples(index=False):
            yield row.timestamp, priority, row.source_row, row

    for _, priority, _, row in merge(
        stream(trades, 0), stream(orders, 1), key=lambda item: item[:3]
    ):
        yield "trade" if priority == 0 else "order", row
