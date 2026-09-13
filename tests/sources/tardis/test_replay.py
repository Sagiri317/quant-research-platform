"""Late vendor snapshots cannot expose states or coverage early."""

import json

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import BookState
from quant_research_platform.profiles.research import _open_crypto_source
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import (
    ReconstructionStats,
    iter_reconstructed_depth,
)
from quant_research_platform.sources.tardis.raw import import_day
from tests.helpers.runtime_resources import capture_test_policy
from tests.sources.tardis.test_raw import _slices


def _line(second, channel, payload):
    value = {"stream": "btcusdt@" + channel, "data": payload}
    if channel == "depthSnapshot":
        value["generated"] = True
    return (
        f"2026-09-01T00:00:{second:02d}.000000Z " + json.dumps(value) + "\n"
    ).encode()


def _delta(second, first, last, previous, quantity="3"):
    return _line(
        second,
        "depth@0ms",
        {
            "e": "depthUpdate",
            "E": 1788220800000 + second * 1000,
            "T": 1788220800000 + second * 1000,
            "s": "BTCUSDT",
            "st": 1,
            "ps": "BTCUSDT",
            "U": first,
            "u": last,
            "pu": previous,
            "b": [["100.00", quantity]],
            "a": [],
        },
    )


@pytest.mark.parametrize("bridge", [True, False])
def test_late_snapshot_has_no_early_book_or_coverage(tmp_path, bridge):
    first = 99 if bridge else 101
    native = (
        _delta(1, first, 101, 98)
        + _line(
            2,
            "depthSnapshot",
            {
                "lastUpdateId": 100,
                "bids": [[str(100 - i), "2"] for i in range(12)],
                "asks": [[str(101 + i), "2"] for i in range(12)],
            },
        )
        + _delta(3, 102, 102, 101, "4")
    )
    slices = _slices(tmp_path / "native", native)
    published = tmp_path / "published"
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        import_day(
            published,
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        with _open_crypto_source(published, resource_policy=policy) as source:
            stats = ReconstructionStats()
            books = []
            for event in iter_reconstructed_depth(
                source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
            ):
                if isinstance(event, OwnedValue):
                    try:
                        book = event.value
                        assert isinstance(book, BookState)
                        books.append(
                            (
                                book.source_sequence,
                                book.observed_time_ns,
                                book.receive_monotonic_ns,
                                book.best_bid,
                                book.update_id,
                            )
                        )
                    finally:
                        book = None
                        event.release(owner="tardis replay test")
            if bridge:
                assert books == [
                    (
                        3,
                        pd.Timestamp("2026-09-01T00:00:03Z").value,
                        None,
                        (100, 4.0),
                        102,
                    )
                ]
                assert list(stats.coverage_segments.values()) == [
                    (books[0][1], books[0][1] + 1)
                ]
            else:
                assert books == []
                assert stats.coverage_segments == {}
                assert stats.gaps == 1
        assert policy.used_memory_bytes == 0


def test_vendor_clock_panel_persists_nullable_monotonic_and_native_identity(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_panel_frames,
    )
    from quant_research_platform.sources.binance.replay import iter_agg_trades

    midnight = pd.Timestamp("2026-09-01T00:00:00Z").value // 1_000_000

    def trade(second, sequence):
        return _line(
            second,
            "aggTrade",
            {
                "e": "aggTrade",
                "E": midnight + second * 1000,
                "T": midnight + second * 1000,
                "s": "BTCUSDT",
                "st": 1,
                "a": sequence,
                "f": sequence,
                "l": sequence,
                "p": "100",
                "q": "1",
                "nq": "1",
                "m": False,
            },
        )

    native = (
        _delta(1, 99, 101, 98)
        + trade(1, 1)
        + _line(
            2,
            "depthSnapshot",
            {
                "lastUpdateId": 100,
                "bids": [[str(100 - i), "2"] for i in range(12)],
                "asks": [[str(101 + i), "2"] for i in range(12)],
            },
        )
        + _delta(3, 102, 102, 101)
        + trade(3, 2)
        + _delta(4, 103, 103, 102)
        + trade(4, 3)
    )
    slices = _slices(tmp_path / "native", native)
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        import_day(
            tmp_path / "published",
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        with _open_crypto_source(tmp_path / "published", policy) as source:
            stats = ReconstructionStats()
            frames = iter_panel_frames(
                iter_reconstructed_depth(
                    source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
                ),
                iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"),
                market_id="crypto",
                asset_id="btc_usdt",
                start_ms=midnight,
                end_ms=midnight + 4000,
                book_coverage=stats.coverage_segments,
                trade_coverage=source.observation_coverage(
                    market_type=FUTURES_MARKET_TYPE,
                    data_kind="agg_trades",
                    symbol="BTCUSDT",
                ).covering,
                batch_rows=100,
            )
            panel = pd.concat(frames, ignore_index=True)
            assert not panel.empty
            assert panel.timestamp.min() >= pd.Timestamp("2026-09-01T00:00:03")
            assert panel.book_receive_monotonic_ns.isna().all()
            assert str(panel.book_receive_monotonic_ns.dtype) == "Int64"
            assert set(panel.book_source_sequence) <= {4, 6}
            table = pa.Table.from_pandas(panel, preserve_index=False)
            path = tmp_path / "panel.parquet"
            pq.write_table(table, path)
            persisted = pq.read_table(path)
            assert (
                persisted.schema.field("book_receive_monotonic_ns").type == pa.int64()
            )
            assert persisted.column("book_receive_monotonic_ns").null_count == len(
                panel
            )
            assert (
                persisted.column("book_source_sequence").to_pylist()
                == panel.book_source_sequence.tolist()
            )
        assert policy.used_memory_bytes == 0


def test_execution_end_stops_inside_invalid_book_segment(tmp_path, monkeypatch):
    import gzip
    from functools import partial

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _books_until,
    )
    from quant_research_platform.sources.tardis import replay as tardis_replay

    snapshot = {
        "lastUpdateId": 100,
        "bids": [[str(100 - i), "2"] for i in range(12)],
        "asks": [[str(101 + i), "2"] for i in range(12)],
    }
    remove = {
        "e": "depthUpdate",
        "E": 1788220804000,
        "T": 1788220804000,
        "s": "BTCUSDT",
        "st": 1,
        "ps": "BTCUSDT",
        "U": 103,
        "u": 103,
        "pu": 102,
        "b": [[str(100 - i), "0"] for i in range(12)],
        "a": [],
    }
    native = (
        _delta(1, 99, 101, 98)
        + _line(2, "depthSnapshot", snapshot)
        + _delta(3, 102, 102, 101)
        + _line(4, "depth@0ms", remove)
        + _delta(5, 104, 104, 103)
        + _delta(6, 105, 105, 104)
    )
    slices = _slices(tmp_path / "native", native)
    future = (
        b"\n"
        + _line(1, "depthSnapshot", dict(snapshot, lastUpdateId=200))
        + _delta(2, 199, 201, 198)
    ).replace(b"T00:00:", b"T00:10:")
    slices[1].write_bytes(gzip.compress(future))
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        import_day(
            tmp_path / "published",
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        seen = []
        original = tardis_replay.iter_native_slice

        def counted(*args, **kwargs):
            from quant_research_platform.policy.lifecycle import closing_iterator

            records = original(*args, **kwargs)
            with closing_iterator(records):
                for row in records:
                    seen.append((kwargs["offset_minutes"], row[0]))
                    yield row

        monkeypatch.setattr(tardis_replay, "iter_native_slice", counted)
        with _open_crypto_source(tmp_path / "published", policy) as source:
            events = _books_until(
                source,
                reconstruct_depth=partial(
                    iter_reconstructed_depth, market_type=FUTURES_MARKET_TYPE
                ),
                symbol="BTCUSDT",
                end_ns=pd.Timestamp("2026-09-01T00:00:05Z").value,
            )
            for event in events:
                if isinstance(event, OwnedValue):
                    event.release(owner="tardis execution bound test")
            assert seen == [(0, number) for number in range(1, 7)]
        assert policy.used_memory_bytes == 0
