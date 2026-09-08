"""Small explicit market tapes for maker action tests."""

import numpy as np

from quant_research_platform.contracts.order_book import BookBoundary, TradeEvent
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    _market_events,
)


def calibration_from_paths(paths, errors, policy):
    return MakerCalibration(*calibration_inputs(paths, errors), resource_policy=policy)


def calibration_inputs(paths, errors):
    event_rows, offsets, samples = [], [], []
    event_id = 0
    for sample_id, ((books, trades), error) in enumerate(
        zip(paths, errors, strict=True)
    ):
        start = sample_id * 1_000_000_000
        initial = event_id
        for _key, event in _market_events(books, trades):
            if event.observed_time_ns >= 500_000_000:
                continue
            offsets.append(len(event_rows))
            event_rows.extend(maker_event_rows(event_id, event, offset=start))
            event_id += 1
        samples.append(
            {
                "sample_id": sample_id,
                "decision_at": start,
                "forecast_error": error,
                "book_segment_id": books[0].segment_id,
                "trade_connection_id": "trade-connection-1",
                "initial_book_event_id": initial,
                "event_start": initial + 1,
                "event_end": event_id,
            }
        )
    offsets.append(len(event_rows))
    return (
        {column: np.array([row[column] for row in samples]) for column in samples[0]},
        {
            column: np.array([row[column] for row in event_rows])
            for column in event_rows[0]
        },
        np.array(offsets, dtype=np.int64),
    )


def maker_event_rows(event_id, event, *, offset=0):
    if isinstance(event, TradeEvent):
        kind, context = "trade", event.connection_id
        levels = (
            (
                "buy" if event.buyer_is_maker else "sell",
                0,
                event.price_tick,
                event.visible_quantity,
            ),
        )
    elif isinstance(event, BookBoundary):
        kind, context = "boundary", event.segment_id
        levels = (("", 0, 0, 0.0),)
    else:
        kind, context = "book", event.segment_id
        levels = (
            (side, index, tick, quantity)
            for side, rows in (("buy", event.bids), ("sell", event.asks))
            for index, (tick, quantity) in enumerate(rows)
        )
    for side, level, tick, quantity in levels:
        yield {
            "event_id": event_id,
            "event_at": offset + event.observed_time_ns,
            "event_kind": kind,
            "context_id": context,
            "side": side,
            "level_index": level,
            "price_tick": tick,
            "quantity": quantity,
        }
