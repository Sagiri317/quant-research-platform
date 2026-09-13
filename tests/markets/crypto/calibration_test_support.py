"""Small explicit market tapes for maker action tests."""

from types import SimpleNamespace

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
                "trade_observation_id": "trade-connection-1",
                "initial_book_event_id": initial,
                "initial_bid_index": books[0].bids[0][0],
                "initial_ask_index": books[0].asks[0][0],
                "event_start": initial + 1,
                "event_end": event_id,
            }
        )
    offsets.append(len(event_rows))

    def sample_rows():
        yield from samples

    def events(start, stop):
        for event_id in range(start, stop):
            rows = event_rows[offsets[event_id] : offsets[event_id + 1]]
            first = rows[0]
            yield SimpleNamespace(
                kind=first["event_kind"],
                event_at=first["event_at"],
                event_id=event_id,
                segment_id=first["context_id"],
                observation_id=first["context_id"],
                price_index=first["price_index"],
                visible_quantity=first["quantity"],
                buyer_is_maker=first["side"] == "buy",
                bids=tuple(
                    (row["price_index"], row["quantity"])
                    for row in rows
                    if row["side"] == "buy"
                ),
                asks=tuple(
                    (row["price_index"], row["quantity"])
                    for row in rows
                    if row["side"] == "sell"
                ),
            )

    return sample_rows, events


def maker_event_rows(event_id, event, *, offset=0):
    if isinstance(event, TradeEvent):
        kind, context = "trade", event.observation_id
        levels = (
            (
                "buy" if event.buyer_is_maker else "sell",
                0,
                event.price_index,
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
            "price_index": tick,
            "quantity": quantity,
        }
