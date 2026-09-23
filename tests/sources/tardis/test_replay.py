"""Late vendor snapshots cannot expose states or coverage early."""

import json

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import BookState
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources import open_market_source
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import (
    ReconstructionStats,
    iter_reconstructed_depth,
)
from quant_research_platform.sources.tardis.raw import import_slices
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
        import_slices(
            published,
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        with open_market_source(published, resource_policy=policy) as source:
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


@pytest.mark.parametrize("date", ["2025-01-01", "2026-09-01"])
def test_vendor_clock_panel_persists_nullable_monotonic_and_native_identity(
    tmp_path, date
):
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
    if date != "2026-09-01":
        original_midnight = midnight
        midnight = pd.Timestamp(date, tz="UTC").value // 1_000_000
        converted = []
        for line in native.splitlines():
            timestamp, body = line.decode().split(" ", 1)
            value = json.loads(body)
            for key in ("st", "ps", "nq"):
                value["data"].pop(key, None)
            for key in ("E", "T"):
                if key in value["data"]:
                    value["data"][key] += midnight - original_midnight
            converted.append(
                timestamp.replace("2026-09-01", date) + " " + json.dumps(value) + "\n"
            )
        native = "".join(converted).encode()
    slices = _slices(tmp_path / "native", native)
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        import_slices(
            tmp_path / "published",
            date=date,
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        with open_market_source(tmp_path / "published", policy) as source:
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
            assert panel.timestamp.min() >= pd.Timestamp(date + "T00:00:03")
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
        import_slices(
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
        with open_market_source(tmp_path / "published", policy) as source:
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


@pytest.mark.parametrize(
    "date,extensions,expected_visible",
    [
        ("2025-01-01", {}, None),
        ("2025-12-30", {}, None),
        ("2025-12-31", {"nq": "0.25"}, 0.25),
        ("2026-06-30", {"nq": "0.25", "st": 1}, 0.25),
        ("2026-09-01", {"nq": "0.25", "st": 1}, 0.25),
    ],
)
def test_native_schema_full_import_and_shared_trade_replay(
    tmp_path, date, extensions, expected_visible
):
    from quant_research_platform.sources.binance.replay import iter_agg_trades

    midnight = pd.Timestamp(date, tz="UTC")
    payload = {
        "e": "aggTrade",
        "E": midnight.value // 1_000_000 + 1000,
        "T": midnight.value // 1_000_000 + 1000,
        "s": "BTCUSDT",
        "a": 1,
        "f": 1,
        "l": 2,
        "p": "100",
        "q": "2",
        "m": True,
        **extensions,
    }
    snapshot = {
        "stream": "btcusdt@depthSnapshot",
        "generated": True,
        "data": {"lastUpdateId": 1, "bids": [["100", "2"]], "asks": [["101", "2"]]},
    }
    trade = {"stream": "btcusdt@aggTrade", "data": payload}
    native = (
        date
        + "T00:00:00Z "
        + json.dumps(snapshot)
        + "\n"
        + date
        + "T00:00:01Z "
        + json.dumps(trade)
        + "\n"
    ).encode()
    slices = _slices(tmp_path / "native", native)
    original = slices[0].read_bytes()
    published = tmp_path / "published"
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        result = import_slices(
            published,
            date=date,
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        assert len(result["files"]) == 144
        assert (published / result["files"][0]["path"]).read_bytes() == original
        with open_market_source(published, resource_policy=policy) as source:
            actual = list(
                iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT")
            )
            assert len(actual) == 1
            assert actual[0].quantity == 2.0
            assert actual[0].visible_quantity == expected_visible
            assert actual[0].receive_monotonic_ns is None
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "date,changes",
    [
        ("2025-12-31", {}),
        ("2026-06-30", {"nq": "1"}),
        ("2025-01-01", {"nq": None}),
        ("2025-01-01", {"nq": "3"}),
        ("2025-01-01", {"st": 2}),
        ("2025-01-01", {"st": True}),
        ("2025-01-01", {"q": "0"}),
    ],
)
def test_native_schema_invalid_fields_fail_atomic_full_import(tmp_path, date, changes):
    payload = {
        "e": "aggTrade",
        "E": pd.Timestamp(date, tz="UTC").value // 1_000_000,
        "T": pd.Timestamp(date, tz="UTC").value // 1_000_000,
        "s": "BTCUSDT",
        "a": 1,
        "f": 1,
        "l": 1,
        "p": "100",
        "q": "2",
        "m": False,
        **changes,
    }
    native = (
        date
        + "T00:00:00Z "
        + json.dumps({"stream": "btcusdt@aggTrade", "data": payload})
        + "\n"
    ).encode()
    slices = _slices(tmp_path / "native", native)
    published = tmp_path / "published"
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        with pytest.raises(ValueError):
            import_slices(
                published,
                date=date,
                slices=slices,
                byte_budget=1024**2,
                resource_policy=policy,
            )
        assert not published.exists()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("boundary", ("manifest_stat", "member_open", "coverage_open"))
def test_tardis_identity_preserves_shared_owner_io_failure(
    tmp_path, monkeypatch, boundary
):
    import errno
    from pathlib import Path

    from quant_research_platform.sources import market_source_identity
    from tests.sources.tardis.test_raw import _snapshot

    root = tmp_path / "published"
    with runtime_resource_policy_scope() as policy:
        manifest = import_slices(
            root,
            date="2026-09-01",
            slices=_slices(tmp_path / "native", _snapshot()),
            byte_budget=1024**2,
            resource_policy=policy,
        )
        assert market_source_identity(root, "panel")
        target = root / (
            "manifest.json"
            if boundary == "manifest_stat"
            else manifest["observation_intervals"]["path"]
            if boundary == "coverage_open"
            else manifest["files"][0]["path"]
        )
        operation = "lstat" if boundary == "manifest_stat" else "open"
        original = getattr(Path, operation)
        primary = OSError(errno.EIO, "controlled_io_failure")

        def fail(path, *args, **kwargs):
            if path == target:
                raise primary
            return original(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, operation, fail)
            with pytest.raises(OSError) as caught:
                market_source_identity(root, "panel")
        assert caught.value is primary
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("midnight_snapshot", [False, True])
def test_midnight_preserves_native_book_factor_and_label_continuity(
    tmp_path, midnight_snapshot
):
    import gzip

    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_factor_frames,
        iter_label_frames,
        iter_panel_frames,
    )
    from quant_research_platform.sources.binance.replay import iter_agg_trades

    base = pd.Timestamp("2026-12-31T23:59:56Z")

    def at(native, seconds):
        _, body = native.decode().split(" ", 1)
        row = json.loads(body)
        timestamp = base + pd.Timedelta(seconds=seconds)
        for key in ("E", "T"):
            if key in row["data"]:
                row["data"][key] = timestamp.value // 1_000_000
        return (
            timestamp.isoformat().replace("+00:00", "Z") + " " + json.dumps(row) + "\n"
        ).encode()

    records = [(0, at(_delta(0, 99, 101, 98), 0))]
    records.append(
        (
            1,
            at(
                _line(
                    0,
                    "depthSnapshot",
                    {
                        "lastUpdateId": 100,
                        "bids": [[str(100 - i), "2"] for i in range(12)],
                        "asks": [[str(101 + i), "2"] for i in range(12)],
                    },
                ),
                1,
            ),
        )
    )
    for second in range(2, 8):
        records.append(
            (second, at(_delta(0, 100 + second, 100 + second, 99 + second), second))
        )
        records.append(
            (
                second,
                at(
                    _line(
                        0,
                        "aggTrade",
                        {
                            "e": "aggTrade",
                            "E": 0,
                            "T": 0,
                            "s": "BTCUSDT",
                            "st": 1,
                            "a": second,
                            "f": second,
                            "l": second,
                            "p": "100",
                            "q": "1",
                            "nq": "1",
                            "m": False,
                        },
                    ),
                    second,
                ),
            )
        )
    if midnight_snapshot:
        records.append(
            (
                4,
                at(
                    _line(
                        0,
                        "depthSnapshot",
                        {
                            "lastUpdateId": 104,
                            "bids": [[str(100 - i), "2"] for i in range(12)],
                            "asks": [[str(101 + i), "2"] for i in range(12)],
                        },
                    ),
                    4,
                ),
            )
        )
        records.sort(key=lambda item: item[0])
    slices = []
    for index in range(2):
        path = tmp_path / f"native-{index}.gz"
        path.write_bytes(
            gzip.compress(
                b"".join(row for second, row in records if (second >= 4) == bool(index))
            )
        )
        slices.append(path)

    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        import_slices(
            tmp_path / "published",
            date="2026-12-31",
            offset_minutes=1430,
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        with open_market_source(tmp_path / "published", policy) as source:
            stats = ReconstructionStats()
            panel = pd.concat(
                iter_panel_frames(
                    iter_reconstructed_depth(
                        source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
                    ),
                    iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"),
                    market_id="crypto",
                    asset_id="btc_usdt",
                    start_ms=base.value // 1_000_000,
                    end_ms=base.value // 1_000_000 + 7000,
                    book_coverage=stats.coverage_segments,
                    trade_coverage=source.observation_coverage(
                        market_type=FUTURES_MARKET_TYPE,
                        data_kind="agg_trades",
                        symbol="BTCUSDT",
                    ).covering,
                    batch_rows=7,
                ),
                ignore_index=True,
            )
            assert panel.timestamp.dt.date.nunique() == 2
            assert panel.segment_id.nunique() == 1
            assert stats.gaps == 0
            assert panel.book_source_sequence.is_monotonic_increasing
            midnight = pd.Timestamp("2027-01-01")
            chunks = [
                panel.loc[panel.timestamp < midnight],
                panel.loc[panel.timestamp >= midnight],
            ]
            factors = pd.concat(iter_factor_frames(chunks), ignore_index=True)
            labels = pd.concat(
                iter_label_frames(chunks, horizons_ms=(500,)), ignore_index=True
            )
            assert (
                factors.loc[factors.timestamp == midnight, "return_100ms"].iloc[0] == 0
            )
            before = labels.loc[
                labels.timestamp == midnight - pd.Timedelta(milliseconds=100)
            ].iloc[0]
            assert before.future_mid_return_500ms == 0
            assert (
                before.future_mid_return_500ms_available_at
                == midnight + pd.Timedelta(milliseconds=400)
            )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("exchange", ["binance", "binance-futures"])
@pytest.mark.parametrize("resend", ["duplicate", "conflict", "unseen"])
def test_trade_resends_preserve_raw_bytes_and_count_each_identity_once(
    tmp_path, exchange, resend
):
    from quant_research_platform.sources.binance.ids import SPOT_MARKET_TYPE
    from quant_research_platform.sources.binance.replay import iter_agg_trades

    def trade(second, identifier, quantity, *, transaction_second=None):
        return _line(
            second,
            "aggTrade",
            {
                "e": "aggTrade",
                "E": 1788220800000 + second * 1000,
                "T": 1788220800000
                + (second if transaction_second is None else transaction_second) * 1000,
                "s": "BTCUSDT",
                "st": 1,
                "a": identifier,
                "p": "100",
                "q": quantity,
                "nq": quantity,
                "f": identifier,
                "l": identifier,
                "m": True,
            },
        )

    native = (
        _line(
            0,
            "depthSnapshot",
            {"lastUpdateId": 100, "bids": [["100", "2"]], "asks": [["101", "2"]]},
        )
        + trade(1, 10, "2")
        + trade(2, 11, "3")
        + trade(
            3,
            9 if resend == "unseen" else 10,
            "9" if resend == "conflict" else "2",
            transaction_second=1,
        )
        + trade(4, 12, "4")
    )
    slices = _slices(tmp_path / "native", native, count=1)
    published = tmp_path / "published"
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        kwargs = dict(
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
            exchange=exchange,
            symbols=("btcusdt",),
            channels=("depth", "depthSnapshot", "aggTrade"),
        )
        if resend != "duplicate":
            with pytest.raises(ValueError, match="sequence_non_monotonic"):
                import_slices(published, **kwargs)
            assert not published.exists()
            assert policy.used_memory_bytes == 0
            return
        manifest = import_slices(published, **kwargs)
        assert (published / manifest["files"][0]["path"]).read_bytes() == slices[
            0
        ].read_bytes()
        assert manifest["gaps"]["record_count"] == 0
        market = SPOT_MARKET_TYPE if exchange == "binance" else FUTURES_MARKET_TYPE
        with open_market_source(published, resource_policy=policy) as source:
            trades = list(iter_agg_trades(source, market, symbol="BTCUSDT"))
            assert [item.source_sequence for item in trades] == [2, 3, 5]
            assert sum(item.quantity for item in trades) == 9
            assert [item.event_time_ms for item in trades] == [
                1788220801000,
                1788220802000,
                1788220804000,
            ]
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("cancel", [False, True, "early"])
def test_archive_four_cursors_share_one_native_scan_and_close(
    tmp_path, monkeypatch, cancel
):
    import quant_research_platform.sources.tardis.replay as replay
    from quant_research_platform.policy.lifecycle import closing_iterator

    snapshot = _line(
        0,
        "depthSnapshot",
        {
            "lastUpdateId": 100,
            "bids": [[str(100 - i), "2"] for i in range(12)],
            "asks": [[str(101 + i), "2"] for i in range(12)],
        },
    )
    trade = _line(
        2,
        "aggTrade",
        {
            "e": "aggTrade",
            "E": 1788220802000,
            "T": 1788220802000,
            "s": "BTCUSDT",
            "st": 1,
            "a": 10,
            "p": "100",
            "q": "2",
            "nq": "1",
            "f": 10,
            "l": 10,
            "m": True,
        },
    )
    native = b"".join(
        row + row.replace(b"btcusdt", b"ethusdt").replace(b"BTCUSDT", b"ETHUSDT")
        for row in (snapshot, _delta(1, 99, 101, 98), trade, _delta(3, 102, 102, 101))
    )
    slices = _slices(tmp_path / "native", native, count=1)
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    published = tmp_path / "published"
    with runtime_resource_policy_scope(policy):
        import_slices(
            published,
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        original = replay.iter_native_slice
        calls = rows = 0

        def counted(*args, **kwargs):
            nonlocal calls, rows
            calls += 1
            with closing_iterator(original(*args, **kwargs)) as source:
                for row in source:
                    rows += 1
                    if cancel is True and rows == 4:
                        raise KeyboardInterrupt("cancel_native_scan")
                    yield row

        monkeypatch.setattr(replay, "iter_native_slice", counted)
        spools = []
        try:
            with open_market_source(published, resource_policy=policy) as source:
                assert calls == 0
                slow = source.iter_market_records(
                    symbol="BTCUSDT",
                    market_type=FUTURES_MARKET_TYPE,
                    data_kinds=("depth_snapshot", "depth"),
                )
                with closing_iterator(slow):
                    first = next(slow)
                    assert first["data_kind"] == "depth_snapshot"
                    assert rows == 1
                    if cancel != "early":
                        for symbol in ("ETHUSDT", "BTCUSDT"):
                            for kinds in (("agg_trades",), ("depth_snapshot", "depth")):
                                result = list(
                                    source.iter_market_records(
                                        symbol=symbol,
                                        market_type=FUTURES_MARKET_TYPE,
                                        data_kinds=kinds,
                                    )
                                )
                                assert len(result) == (
                                    1 if kinds == ("agg_trades",) else 3
                                )
                        assert len(list(slow)) == 2
                        assert calls == 1 and rows == 8
                        repeated = list(
                            source.iter_market_records(
                                symbol="BTCUSDT",
                                market_type=FUTURES_MARKET_TYPE,
                                data_kinds=("agg_trades",),
                            )
                        )
                        assert len(repeated) == 1 and calls == 1
                spools = [pair[0] for pair in source._spools.values()]
        except KeyboardInterrupt as error:
            assert cancel and str(error) == "cancel_native_scan"
            spools = [pair[0] for pair in source._spools.values()]
        assert all(spool.closed for spool in spools)
    assert policy.used_memory_bytes == 0


def test_tardis_spool_reservation_keeps_small_cap_usable(monkeypatch):
    import quant_research_platform.sources.tardis.replay as replay
    import quant_research_platform.storage._scratch_ownership as scratch

    monkeypatch.setattr(scratch, "_memory_filesystem", lambda _path: True)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=2 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        with scratch.reserved_temporary_file(resource_policy=policy) as (
            spool,
            reservation,
        ):
            with policy.lease("test.occupied", policy.working_set_headroom_bytes - 128):
                replay._reserve_spool_capacity(reservation, 128, 128)
                spool.write(b"x" * 128)
                assert reservation.reserved_bytes == 128
                with pytest.raises(
                    ValueError, match="storage_scratch_exceeds_memory_capacity"
                ):
                    replay._reserve_spool_capacity(reservation, 129, 1)
    assert policy.used_memory_bytes == 0


def test_tardis_spool_rechecks_disk_before_each_append(monkeypatch):
    from types import SimpleNamespace

    import quant_research_platform.sources.tardis.replay as replay
    from quant_research_platform.storage._scratch_ownership import (
        reserved_temporary_file,
    )

    def no_disk(_path, *, reserve_bytes):
        assert reserve_bytes == 1
        raise RuntimeError("insufficient_free_disk")

    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with reserved_temporary_file(resource_policy=policy) as (spool, reservation):
            replay._reserve_spool_capacity(reservation, 128, 128)
            spool.write(b"x" * 128)
            assert reservation.reserved_bytes >= 129
            if hasattr(replay.os, "statvfs"):
                monkeypatch.setattr(
                    replay.os,
                    "statvfs",
                    lambda _path: SimpleNamespace(f_bavail=0, f_frsize=4096),
                )
            else:
                monkeypatch.setattr(replay, "require_free_disk_headroom", no_disk)
            with pytest.raises(RuntimeError, match="insufficient_free_disk"):
                replay._reserve_spool_capacity(reservation, 129, 1)
            assert spool.tell() == 128
            spool.seek(0)
            assert spool.read() == b"x" * 128
    assert policy.used_memory_bytes == 0
