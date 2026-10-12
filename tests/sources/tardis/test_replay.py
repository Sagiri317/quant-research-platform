"""Late vendor snapshots cannot expose states or coverage early."""

import json

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import BookState
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import (
    ReconstructionStats,
    iter_reconstructed_depth,
)
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.sources.tardis.raw import import_slices
from tests.helpers.tardis import _line, _slices


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
    import_slices(published, date="2026-09-01", slices=slices)
    with open_market_source(published) as source:
        stats = ReconstructionStats()
        books = []
        for event in iter_reconstructed_depth(
            source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
        ):
            if isinstance(event, BookState):
                book = event
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
        if bridge:
            assert books == [
                (3, pd.Timestamp("2026-09-01T00:00:03Z").value, None, (100, 4.0), 102)
            ]
            assert list(stats.coverage_segments.values()) == [
                (books[0][1], books[0][1] + 1)
            ]
        else:
            assert books == []
            assert stats.coverage_segments == {}
            assert stats.gaps == 1


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

    midnight = pd.Timestamp("2026-09-01T00:00:00Z").value // 1000000

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
        midnight = pd.Timestamp(date, tz="UTC").value // 1000000
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
    import_slices(tmp_path / "published", date=date, slices=slices)
    with open_market_source(tmp_path / "published") as source:
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
        assert persisted.schema.field("book_receive_monotonic_ns").type == pa.int64()
        assert persisted.column("book_receive_monotonic_ns").null_count == len(panel)
        assert (
            persisted.column("book_source_sequence").to_pylist()
            == panel.book_source_sequence.tolist()
        )


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
    slices = _slices(tmp_path / "native", native, count=2)
    future = (
        b"\n"
        + _line(1, "depthSnapshot", dict(snapshot, lastUpdateId=200))
        + _delta(2, 199, 201, 198)
    ).replace(b"T00:00:", b"T00:10:")
    slices[1].write_bytes(gzip.compress(future))
    import_slices(tmp_path / "published", date="2026-09-01", slices=slices)
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
    with open_market_source(tmp_path / "published") as source:
        list(
            _books_until(
                source,
                reconstruct_depth=partial(
                    iter_reconstructed_depth, market_type=FUTURES_MARKET_TYPE
                ),
                symbol="BTCUSDT",
                end_ns=pd.Timestamp("2026-09-01T00:00:05Z").value,
            )
        )
        assert seen == [(0, number) for number in range(1, 7)]


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
        "E": midnight.value // 1000000 + 1000,
        "T": midnight.value // 1000000 + 1000,
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
    result = import_slices(published, date=date, slices=slices)
    assert len(result["files"]) == 1
    assert (published / result["files"][0]["path"]).read_bytes() == original
    with open_market_source(published) as source:
        actual = list(iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
        assert len(actual) == 1
        assert actual[0].quantity == 2.0
        assert actual[0].visible_quantity == expected_visible
        assert actual[0].receive_monotonic_ns is None


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
        "E": pd.Timestamp(date, tz="UTC").value // 1000000,
        "T": pd.Timestamp(date, tz="UTC").value // 1000000,
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
    with pytest.raises(ValueError):
        import_slices(published, date=date, slices=slices)
    assert not published.exists()


@pytest.mark.parametrize("boundary", ("manifest_stat", "member_open", "coverage_open"))
def test_tardis_identity_preserves_shared_owner_io_failure(
    tmp_path, monkeypatch, boundary
):
    import errno

    import quant_research_platform.sources.binance.replay as binance_replay
    from quant_research_platform.sources.market import market_source_identity
    from tests.helpers.tardis import _snapshot

    root = tmp_path / "published"
    manifest = import_slices(
        root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", _snapshot()),
    )
    assert market_source_identity(root, "panel")
    target = root / (
        "manifest.json"
        if boundary == "manifest_stat"
        else manifest["observation_intervals"]["path"]
        if boundary == "coverage_open"
        else manifest["files"][0]["path"]
    )
    owner = binance_replay
    operation = (
        "filesystem_stat" if boundary == "manifest_stat" else "open_filesystem_file"
    )
    original = getattr(owner, operation)
    primary = OSError(errno.EIO, "controlled_io_failure")

    def fail(path, *args, **kwargs):
        if path == target:
            raise primary
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(owner, operation, fail)
        with pytest.raises(OSError) as caught:
            market_source_identity(root, "panel")
    assert caught.value is primary


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
                row["data"][key] = timestamp.value // 1000000
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
                b"".join(
                    (row for second, row in records if (second >= 4) == bool(index))
                )
            )
        )
        slices.append(path)
    import_slices(
        tmp_path / "published",
        date="2026-12-31",
        offset_minutes=1430,
        slices=slices,
    )
    with open_market_source(tmp_path / "published") as source:
        stats = ReconstructionStats()
        panel = pd.concat(
            iter_panel_frames(
                iter_reconstructed_depth(
                    source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
                ),
                iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"),
                market_id="crypto",
                asset_id="btc_usdt",
                start_ms=base.value // 1000000,
                end_ms=base.value // 1000000 + 7000,
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
        assert factors.loc[factors.timestamp == midnight, "return_100ms"].iloc[0] == 0
        before = labels.loc[
            labels.timestamp == midnight - pd.Timedelta(milliseconds=100)
        ].iloc[0]
        assert before.future_mid_return_500ms == 0
        assert before.future_mid_return_500ms_available_at == midnight + pd.Timedelta(
            milliseconds=400
        )


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
    kwargs = dict(
        date="2026-09-01",
        slices=slices,
        exchange=exchange,
        symbols=("btcusdt",),
        channels=("depth", "depthSnapshot", "aggTrade"),
    )
    if resend != "duplicate":
        with pytest.raises(ValueError, match="sequence_non_monotonic"):
            import_slices(published, **kwargs)
        assert not published.exists()
        return
    manifest = import_slices(published, **kwargs)
    assert (published / manifest["files"][0]["path"]).read_bytes() == slices[
        0
    ].read_bytes()
    assert manifest["gaps"]["record_count"] == 0
    market = SPOT_MARKET_TYPE if exchange == "binance" else FUTURES_MARKET_TYPE
    with open_market_source(published) as source:
        trades = list(iter_agg_trades(source, market, symbol="BTCUSDT"))
        assert [item.source_sequence for item in trades] == [2, 3, 5]
        assert sum(item.quantity for item in trades) == 9
        assert [item.event_time_ms for item in trades] == [
            1788220801000,
            1788220802000,
            1788220804000,
        ]


@pytest.mark.parametrize("cancel", [False, True, "early"])
def test_archive_four_cursors_share_one_native_scan_and_close(
    tmp_path, monkeypatch, cancel
):
    import quant_research_platform.sources.tardis.raw as raw
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
        for row in (
            snapshot,
            _delta(1, 99, 101, 98),
            trade,
            _delta(3, 102, 102, 101),
        )
    )
    slices = _slices(tmp_path / "native", native, count=1)
    published = tmp_path / "published"
    import_slices(published, date="2026-09-01", slices=slices)
    original = replay.iter_native_slice
    native_open = raw.open_filesystem_file
    temporary_file = replay.TemporaryFile
    handles, buffers = [], []

    def opened(path, *args, **kwargs):
        handle = native_open(path, *args, **kwargs)
        if path.suffix == ".gz":
            handles.append(handle)
        return handle

    def buffered(*args, **kwargs):
        handle = temporary_file(*args, **kwargs)
        buffers.append(handle)
        return handle

    monkeypatch.setattr(raw, "open_filesystem_file", opened)
    monkeypatch.setattr(replay, "TemporaryFile", buffered)
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
    interrupted = False
    try:
        with open_market_source(published) as source:
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
                            assert len(result) == (1 if kinds == ("agg_trades",) else 3)
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
    except KeyboardInterrupt as error:
        interrupted = True
        assert cancel and str(error) == "cancel_native_scan"
    assert interrupted is (cancel is True)
    assert handles and all(handle.closed for handle in handles)
    assert buffers and all(handle.closed for handle in buffers)


def test_archive_tail_shares_native_scan_and_preserves_replay(tmp_path, monkeypatch):
    from collections import Counter
    from functools import partial

    import quant_research_platform.sources.tardis.raw as raw
    import quant_research_platform.sources.tardis.replay as replay
    from quant_research_platform.policy.lifecycle import closing_iterator

    rows = [
        _line(
            0,
            "depthSnapshot",
            {"lastUpdateId": 100, "bids": [["100", "2"]], "asks": [["101", "2"]]},
        )
    ]
    for identifier in range(1, 257):
        second = (identifier - 1) // 8 + 1
        rows.append(
            _line(
                second,
                "aggTrade",
                {
                    "e": "aggTrade",
                    "E": 1788220800000 + second * 1000,
                    "T": 1788220800000 + second * 1000,
                    "s": "BTCUSDT",
                    "st": 1,
                    "a": identifier,
                    "f": identifier,
                    "l": identifier,
                    "p": "100",
                    "q": "2",
                    "nq": "1",
                    "m": True,
                },
            )
        )
    native = b"".join(
        row + row.replace(b"btcusdt", b"ethusdt").replace(b"BTCUSDT", b"ETHUSDT")
        for row in rows
    )
    published = tmp_path / "published"
    import_slices(
        published,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native, count=1),
    )
    calls = Counter()
    original_records = replay.iter_native_slice
    native_open = raw.open_filesystem_file
    handles = []

    def opened(path, *args, **kwargs):
        handle = native_open(path, *args, **kwargs)
        if path.suffix == ".gz":
            handles.append(handle)
        return handle

    monkeypatch.setattr(raw, "open_filesystem_file", opened)

    def counted_records(*args, **kwargs):
        calls["native_scans"] += 1
        with closing_iterator(original_records(*args, **kwargs)) as records:
            for record in records:
                calls["native_rows"] += 1
                yield record

    monkeypatch.setattr(replay, "iter_native_slice", counted_records)
    with open_market_source(published) as source:
        read = partial(
            source.iter_market_records,
            market_type=FUTURES_MARKET_TYPE,
            data_kinds=("agg_trades",),
        )
        leading = list(read(symbol="BTCUSDT"))
        assert [row["source_sequence"] for row in leading] == list(range(3, 515, 2))
        expected = json.dumps(leading)
        leading[0]["payload"]["a"] = -1
        lagging = list(read(symbol="ETHUSDT"))
        assert [row["source_sequence"] for row in lagging] == list(range(4, 516, 2))
        assert json.dumps(list(read(symbol="BTCUSDT"))) == expected
        assert (
            len(
                list(
                    read(
                        symbol="BTCUSDT",
                        end_ns=pd.Timestamp("2026-09-01T00:00:01Z").value,
                    )
                )
            )
            == 8
        )
        assert calls["native_scans"] == 1 and calls["native_rows"] == 514
    assert handles and all(handle.closed for handle in handles)


def test_cursor_spool_batches_native_appends(tmp_path, monkeypatch):
    import io
    from collections import Counter
    from contextlib import ExitStack

    import quant_research_platform.sources.tardis.replay as replay

    calls = Counter()
    streams = []

    class CountedFile(io.FileIO):
        def write(self, value):
            calls["write"] += 1
            return super().write(value)

        def seek(self, offset, whence=0):
            calls["seek"] += 1
            return super().seek(offset, whence)

    def temporary_file():
        stream = io.BufferedRandom(CountedFile(tmp_path / "buffer", "w+"))
        streams.append(stream)
        return stream

    def records(self):
        for number in range(5000):
            yield (
                ("BTCUSDT", "agg_trades"),
                {
                    "sequence": number,
                    "payload": "x" * 256,
                },
            )

    monkeypatch.setattr(replay, "TemporaryFile", temporary_file)
    monkeypatch.setattr(replay.TardisArchive, "_iter_canonical_records", records)
    with ExitStack() as resources:
        archive = replay.TardisArchive(tmp_path, {}, {}, None, resources, lambda: None)
        for number, (_start, record) in enumerate(
            archive._stream_records("BTCUSDT", "agg_trades", None)
        ):
            assert record == {"sequence": number, "payload": "x" * 256}
        assert number == 4999
    assert streams and all(stream.closed for stream in streams)
    assert calls["write"] < 500 and calls["seek"] < 500


def test_cursor_spool_releases_payloads_with_native_decoder_replaced(
    tmp_path, monkeypatch
):
    import tracemalloc
    from contextlib import ExitStack

    import quant_research_platform.sources.tardis.replay as replay

    def records(self):
        for number in range(5000):
            yield ("BTCUSDT", "agg_trades"), {"payload": "x" * 4096 + str(number)}

    monkeypatch.setattr(replay.TardisArchive, "_iter_canonical_records", records)
    tracemalloc.start()
    try:
        with ExitStack() as resources:
            archive = replay.TardisArchive(
                tmp_path, {}, {}, None, resources, lambda: None
            )
            assert (
                sum(1 for _ in archive._stream_records("BTCUSDT", "agg_trades", None))
                == 5000
            )
            retained, _peak = tracemalloc.get_traced_memory()
            assert retained < 1024**2
            buffers = tuple(archive._records_by_stream.values())
        assert all(stream.closed for stream, _end in buffers)
    finally:
        tracemalloc.stop()


def test_archive_buffer_close_keeps_native_decode_failure(tmp_path, monkeypatch):
    import io
    from contextlib import ExitStack

    import quant_research_platform.sources.tardis.replay as replay

    primary = ValueError("controlled_native_decode_failure")
    buffers = []

    class BrokenClose(io.BytesIO):
        def close(self):
            if not self.closed:
                super().close()
                raise OSError("controlled_buffer_close_failure")

    def temporary_file():
        stream = BrokenClose()
        buffers.append(stream)
        return stream

    def records(self):
        yield ("BTCUSDT", "agg_trades"), {"payload": "synthetic"}
        raise primary

    monkeypatch.setattr(replay, "TemporaryFile", temporary_file)
    monkeypatch.setattr(replay.TardisArchive, "_iter_canonical_records", records)
    with pytest.raises(ValueError) as caught:
        with ExitStack() as resources:
            archive = replay.TardisArchive(
                tmp_path, {}, {}, None, resources, lambda: None
            )
            list(archive._stream_records("BTCUSDT", "agg_trades", None))
    assert caught.value is primary
    assert all(stream.closed for stream in buffers)
    assert any(
        "iterator close also failed: OSError" in note for note in primary.__notes__
    )


@pytest.mark.parametrize("snapshot_depths", [(12, 12), (12, 13), (13, 12), (12, 9)])
def test_disjoint_archives_preserve_reset_ids_through_processed_publication(
    tmp_path, snapshot_depths
):
    from functools import partial

    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_label_frames,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        catalog as processed_catalog,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer as processed_producer,
    )
    from quant_research_platform.sources.binance.replay import iter_agg_trades
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout

    dates = ("2026-07-01", "2026-08-01")
    window = TimeWindow("2026-07-01T00:00:00", "2026-08-01T00:00:04")
    paths = ("july", "august")
    catalog = processed_catalog.hft_processed_catalog()
    spec = catalog.get("binance_futures_l2_panel_100ms")
    layout = StorageLayout(project_root=tmp_path / "processed")
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    opener = partial(open_market_source, archive_paths=paths, time_window=window)
    for date, path, depth in zip(dates, paths, snapshot_depths, strict=True):
        midnight = pd.Timestamp(date, tz="UTC").value // 1000000
        native = _delta(1, 99, 101, 98) + _line(
            2,
            "depthSnapshot",
            {
                "lastUpdateId": 100,
                "bids": [[str(100 - i), "2"] for i in range(depth)],
                "asks": [[str(101 + i), "2"] for i in range(depth)],
            },
        )
        for second in (3, 4):
            native += _delta(second, 99 + second, 99 + second, 98 + second)
            native += _line(
                second,
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
            )
        converted = []
        for line in native.decode().splitlines():
            timestamp, body = line.split(" ", 1)
            record = json.loads(body)
            seconds = pd.Timestamp(timestamp).second
            if date == dates[1]:
                for side in ("bids", "asks", "b", "a"):
                    levels = record["data"].get(side)
                    if isinstance(levels, list):
                        for level in levels:
                            level[0] = str(float(level[0]) + 0.5)
                if "p" in record["data"]:
                    record["data"]["p"] = "100.5"
            for key in ("E", "T"):
                if key in record["data"]:
                    record["data"][key] = midnight + seconds * 1000
            converted.append(
                timestamp.replace("2026-09-01", date) + " " + json.dumps(record) + "\n"
            )
        import_slices(
            tmp_path / path,
            date=date,
            slices=_slices(
                tmp_path / ("native-" + path), "".join(converted).encode(), count=1
            ),
        )
    with opener(tmp_path) as source:
        assert [
            archive.market_metadata(FUTURES_MARKET_TYPE, "BTCUSDT")["price_unit"]
            for archive in source.archives
        ] == ["1", "0.1"]
        assert (
            source.market_metadata(FUTURES_MARKET_TYPE, "BTCUSDT")["price_unit"]
            == "0.1"
        )
        assert [
            archive.market_metadata(FUTURES_MARKET_TYPE, "BTCUSDT")["snapshot_limit"]
            for archive in source.archives
        ] == list(snapshot_depths)
        assert source.market_metadata(FUTURES_MARKET_TYPE, "BTCUSDT")[
            "snapshot_limit"
        ] == max(snapshot_depths)
        identity = canonical_json_hash(source.source_identity)
        rows = list(
            source.iter_market_records(
                symbol="BTCUSDT",
                market_type=FUTURES_MARKET_TYPE,
                data_kinds=("agg_trades",),
            )
        )
        assert [row["payload"]["a"] for row in rows] == [3, 4, 3, 4]
        assert {row["price_unit"] for row in rows} == {"0.1"}
        assert len({row["source_sequence"] for row in rows}) == 4
        observation_ids = [row["observation_id"] for row in rows]
        assert observation_ids[0] == observation_ids[1]
        assert observation_ids[2] == observation_ids[3]
        assert observation_ids[0] != observation_ids[2]
        assert (
            observation_ids[0].split(":", 1)[1] == observation_ids[2].split(":", 1)[1]
        )
        stats = ReconstructionStats()
        books, boundaries = ([], [])
        for event in iter_reconstructed_depth(
            source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
        ):
            if isinstance(event, BookState):
                book = event
                books.append(
                    (
                        pd.Timestamp(book.observed_time_ns, tz="UTC").strftime(
                            "%Y-%m-%d"
                        ),
                        len(book.bids),
                        len(book.asks),
                        book.update_id,
                    )
                )
            else:
                boundaries.append(event.reason)
        expected_books = [
            (date, depth, depth, update_id)
            for date, depth in zip(dates, snapshot_depths, strict=True)
            if depth >= 10
            for update_id in (102, 103)
        ]
        assert books == expected_books
        if snapshot_depths[1] < 10:
            assert boundaries == ["reconnect_boundary", "unsupported_coverage"]
            assert stats.invalid_segments == stats.unsupported_coverage == 1
            assert len(stats.coverage_segments) == 1
        else:
            assert boundaries == ["reconnect_boundary"]
            assert stats.invalid_segments == stats.unsupported_coverage == 0
            assert len(stats.coverage_segments) == 2
    if snapshot_depths[1] < 10:
        return
    with processed_producer.produce_hft_processed_data(
        spec,
        layout=layout,
        time_window=window,
        processed_catalog=catalog,
        source_catalog=None,
        open_run=opener,
        reconstruct_depth=iter_reconstructed_depth,
        read_trades=iter_agg_trades,
        make_stats=ReconstructionStats,
        raw_root=tmp_path,
        expected_source_identity=identity,
        symbols=("BTCUSDT",),
    ) as (data, metadata, verify):
        write_dataset(
            root, data, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
    panel = read_dataset(root)
    assert set(panel.timestamp.dt.strftime("%Y-%m-%d")) == set(dates)
    assert panel.loc[panel.timestamp.dt.month == 7, "mid"].dropna().eq(100.5).all()
    assert panel.loc[panel.timestamp.dt.month == 8, "mid"].dropna().eq(101.0).all()
    assert panel.segment_id.nunique() == 2
    assert panel.book_segment_id.nunique() == 2
    assert panel.book_observation_id.nunique() == 2
    assert panel.trade_observation_id.nunique() == 2
    assert panel.book_source_sequence.is_monotonic_increasing
    counted = panel.loc[panel.trade_count_100ms.gt(0)]
    assert counted.timestamp.tolist() == [
        pd.Timestamp(day + "T00:00:04") for day in dates
    ]
    assert counted.trade_count_100ms.tolist() == [1, 1]
    first = panel.loc[panel.timestamp.dt.month == 7]
    second = panel.loc[panel.timestamp.dt.month == 8]
    assert set(first.book_source_sequence).isdisjoint(second.book_source_sequence)
    labels = pd.concat(
        iter_label_frames((first, second), horizons_ms=(500,)), ignore_index=True
    )
    first_end = first.timestamp.max()
    last_july = labels.loc[labels.timestamp == first_end].iloc[0]
    assert pd.isna(last_july.future_mid_return_500ms)
    assert pd.isna(last_july.future_mid_return_500ms_available_at)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    manifest_path = tmp_path / paths[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    member = manifest["files"][0]
    member["line_count"] = 1
    member["record_counts"] = {"btcusdt@depthSnapshot": 1}
    member["disconnect_markers"] = 0
    manifest_path.write_text(json.dumps(manifest))
    seen = []
    with opener(tmp_path) as source:
        bad_identity = canonical_json_hash(source.source_identity)
        with pytest.raises(ValueError, match="tardis_native_line_count_mismatch"):
            for row in source.iter_market_records(
                symbol="BTCUSDT",
                market_type=FUTURES_MARKET_TYPE,
                data_kinds=("agg_trades",),
            ):
                seen.append(row["receive_time"])
    assert seen and all(value.startswith(dates[0]) for value in seen)
    candidate_root = StorageLayout(
        project_root=tmp_path / "candidate"
    ).processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with pytest.raises(ValueError, match="tardis_native_line_count_mismatch"):
        with processed_producer.produce_hft_processed_data(
            spec,
            layout=layout,
            time_window=window,
            processed_catalog=catalog,
            source_catalog=None,
            open_run=opener,
            reconstruct_depth=iter_reconstructed_depth,
            read_trades=iter_agg_trades,
            make_stats=ReconstructionStats,
            raw_root=tmp_path,
            expected_source_identity=bad_identity,
            symbols=("BTCUSDT",),
        ) as (data, metadata, verify):
            write_dataset(
                candidate_root,
                data,
                spec=spec,
                instance_metadata=metadata,
                staged_validator=verify,
            )
    assert not candidate_root.exists()
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before


def test_disconnect_snapshot_and_bridge_share_recovered_observation(tmp_path):
    snapshot = {
        "lastUpdateId": 100,
        "bids": [[str(100 - i), "2"] for i in range(12)],
        "asks": [[str(101 + i), "2"] for i in range(12)],
    }
    native = (
        _line(0, "depthSnapshot", snapshot)
        + _delta(1, 100, 101, 99)
        + _delta(2, 102, 102, 101)
        + b"\n"
        + _line(4, "depthSnapshot", {**snapshot, "lastUpdateId": 200})
        + _delta(5, 200, 201, 199)
        + _delta(6, 202, 202, 201)
    )
    published = tmp_path / "published"
    import_slices(
        published,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native, count=1),
    )
    with open_market_source(published) as source:
        stats = ReconstructionStats()
        identifiers = []
        for event in iter_reconstructed_depth(
            source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
        ):
            if isinstance(event, BookState):
                identifiers.append(event.update_id)
        assert identifiers == [101, 102, 201, 202]
        assert len(stats.coverage_segments) == 2


def test_archive_actual_spans_sort_equivalent_cross_midnight_slices(tmp_path):
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.sources.tardis.replay import _archive_span

    tape = (
        _delta(1, 99, 101, 98)
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
        + _delta(4, 103, 103, 102)
    )
    slices = {}
    anchors = {
        "before": pd.Timestamp("2026-12-31T23:59:55Z"),
        "after": pd.Timestamp("2027-01-01T00:00:00Z"),
    }
    for name, anchor in anchors.items():
        shift = anchor - pd.Timestamp("2026-09-01T00:00:00Z")
        rows = []
        for line in tape.decode().splitlines():
            timestamp, body = line.split(" ", 1)
            record = json.loads(body)
            for clock in ("E", "T"):
                if clock in record["data"]:
                    record["data"][clock] += shift.value // 1000000
            observed = (pd.Timestamp(timestamp) + shift).isoformat(
                timespec="microseconds"
            )
            rows.append(observed.replace("+00:00", "Z") + " " + json.dumps(record))
        slices[name] = _slices(
            tmp_path / ("native-" + name), ("\n".join(rows) + "\n").encode(), count=1
        )
    manifests = {}
    for name, date, offset, native in (
        ("before", "2026-12-31", 1430, "before"),
        ("offset", "2026-12-31", 1440, "after"),
        ("next_day", "2027-01-01", 0, "after"),
        ("older_anchor", "2026-12-30", 2880, "after"),
    ):
        manifests[name] = import_slices(
            tmp_path / name,
            date=date,
            offset_minutes=offset,
            slices=slices[native],
        )
    offset_file = manifests["offset"]["files"][0]
    next_day_file = manifests["next_day"]["files"][0]
    assert offset_file["path"] == next_day_file["path"]
    assert offset_file["sha256"] == next_day_file["sha256"]
    assert (tmp_path / "offset" / offset_file["path"]).read_bytes() == (
        tmp_path / "next_day" / next_day_file["path"]
    ).read_bytes()
    assert canonical_json_hash(manifests["offset"]) != canonical_json_hash(
        manifests["next_day"]
    )
    summaries = []
    for paths in (
        ("next_day", "before"),
        ("offset", "before"),
        ("before", "offset"),
        ("before", "older_anchor"),
    ):
        with open_market_source(tmp_path, archive_paths=paths) as source:
            spans = tuple(
                _archive_span(archive.manifest) for archive in source.archives
            )
            assert spans == tuple(sorted(spans))
            assert spans[0][1] == spans[1][0]
            ordered_paths = (
                "before",
                next(name for name in paths if name != "before"),
            )
            assert [
                item["manifest_hash"] for item in source.source_identity["archives"]
            ] == [canonical_json_hash(manifests[name]) for name in ordered_paths]
            stats = ReconstructionStats()
            books, boundaries, identities = ([], [], [])
            for event in iter_reconstructed_depth(
                source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
            ):
                if isinstance(event, BookState):
                    book = event
                    books.append(
                        (
                            book.observed_time_ns,
                            book.update_id,
                            book.bids,
                            book.asks,
                        )
                    )
                    identities.append(book.observation_id)
                else:
                    boundaries.append((event.observed_time_ns, event.reason))
            assert [(row[0], row[1]) for row in books] == [
                (anchor.value + second * 1000000000, 99 + second)
                for anchor in anchors.values()
                for second in (3, 4)
            ]
            assert len(set(identities)) == 2
            assert identities[0] == identities[1]
            assert identities[2] == identities[3]
            coverage = sorted(stats.coverage_segments.values())
            assert coverage == [
                (anchor.value + 3000000000, anchor.value + 4000000001)
                for anchor in anchors.values()
            ]
            assert boundaries == [
                (anchors["after"].value + 3000000000, "reconnect_boundary")
            ]
            assert stats.gaps == 0
            summaries.append((books, coverage, boundaries))

            from quant_research_platform.storage.datasets import (
                inspect_dataset,
                read_dataset,
                write_dataset,
            )
            from tests.storage.low_level_fixtures import make_processed_dataset_spec

            spec = make_processed_dataset_spec(
                f"span_identity_{len(summaries)}", market_id="crypto"
            )
            destination = (
                tmp_path / "data" / "processed" / "crypto" / "factor" / spec.data_id
            )
            frame = pd.DataFrame(
                {
                    "market_id": ["crypto"] * len(books),
                    "asset_id": ["btc_usdt"] * len(books),
                    "timestamp": pd.to_datetime([row[0] for row in books], unit="ns"),
                    "value": [float(row[2][0][0]) for row in books],
                }
            )
            write_dataset(
                destination,
                frame,
                spec=spec,
                instance_metadata={"source_artifact_identity": source.source_identity},
            )
            pd.testing.assert_frame_equal(read_dataset(destination), frame)
            assert (
                inspect_dataset(destination).manifest["source_artifact_identity"]
                == source.source_identity
            )
    assert all(summary == summaries[0] for summary in summaries)
    for paths, reason in (
        (("before", "before"), "tardis_archive_set_invalid"),
        (("offset", "next_day"), "tardis_archive_spans_overlap"),
    ):
        with pytest.raises(ValueError, match=reason):
            with open_market_source(tmp_path, archive_paths=paths):
                pass


@pytest.mark.parametrize(
    "failure", ("depth_exhausted", "first_bridge", "healthy", "disconnect", "no_bridge")
)
def test_real_midnight_snapshot_recovers_book_without_inventing_observation(
    tmp_path, failure
):
    import gzip

    from quant_research_platform.contracts.order_book import BookBoundary

    base = pd.Timestamp("2026-12-31T23:59:57Z")

    def line(second, channel, payload):
        stamp = base + pd.Timedelta(seconds=second)
        payload = dict(payload)
        for key in ("E", "T"):
            if key in payload:
                payload[key] = stamp.value // 1000000
        row = {"stream": "btcusdt@" + channel, "data": payload}
        if channel == "depthSnapshot":
            row["generated"] = True
        return (
            stamp.isoformat().replace("+00:00", "Z") + " " + json.dumps(row) + "\n"
        ).encode()

    def delta(second, first, last, previous, bids):
        return line(
            second,
            "depth@0ms",
            {
                "e": "depthUpdate",
                "E": 0,
                "T": 0,
                "s": "BTCUSDT",
                "st": 1,
                "ps": "BTCUSDT",
                "U": first,
                "u": last,
                "pu": previous,
                "b": bids,
                "a": [],
            },
        )

    bids = [[str(100 - i), "2"] for i in range(12)]
    asks = [[str(101 + i), "2"] for i in range(12)]
    removes = [[price, "0"] for price, _ in bids]
    broken = failure in {"depth_exhausted", "no_bridge"}
    first_day = (
        line(0, "depthSnapshot", dict(lastUpdateId=100, bids=bids, asks=asks))
        + delta(1, 101 if failure == "first_bridge" else 100, 101, 99, [])
        + delta(2, 102, 102, 101, removes if broken else [])
    )
    next_bids = [[str(88 - i), "2"] for i in range(12)] if broken else bids
    second_day = (
        (b"\n" if failure == "disconnect" else b"")
        + line(3, "depthSnapshot", dict(lastUpdateId=102, bids=next_bids, asks=asks))
        + delta(4, 103 if failure == "no_bridge" else 102, 103, 102, [])
        + delta(5, 104, 104, 103, [])
    )
    paths = []
    for index, content in enumerate((first_day, second_day)):
        path = tmp_path / f"native-{index}.gz"
        path.write_bytes(gzip.compress(content))
        paths.append(path)
    published = tmp_path / "published"
    import_slices(published, date="2026-12-31", offset_minutes=1430, slices=paths)
    with open_market_source(published) as source:
        stats = ReconstructionStats()
        events = list(
            iter_reconstructed_depth(
                source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
            )
        )
        states = [event for event in events if isinstance(event, BookState)]
        expected = {
            "depth_exhausted": [101, 103, 104],
            "first_bridge": [103, 104],
            "healthy": [101, 102, 103, 104],
            "disconnect": [101, 102, 103, 104],
            "no_bridge": [101],
        }
        assert [state.update_id for state in states] == expected[failure]
        assert len({state.observation_id for state in states}) == (
            2 if failure == "disconnect" else 1
        )
        assert len(stats.coverage_segments) == (
            2 if failure in {"depth_exhausted", "disconnect"} else 1
        )
        if failure == "depth_exhausted":
            assert states[0].segment_id != states[1].segment_id
            assert states[1].segment_id == states[2].segment_id
            assert states[1].best_bid[0] == 88
            boundary = next(
                event for event in events if isinstance(event, BookBoundary)
            )
            assert boundary.segment_id == states[0].segment_id
            assert (
                stats.coverage_segments[states[0].segment_id][1]
                == (base + pd.Timedelta(seconds=2)).value
            )
            assert (
                stats.coverage_segments[states[1].segment_id][0]
                == (base + pd.Timedelta(seconds=4)).value
            )
        if failure == "healthy":
            assert len({state.segment_id for state in states}) == 1


def test_late_snapshot_prefix_cursors_are_independent_and_canonical(tmp_path):
    from quant_research_platform.policy.lifecycle import closing_iterator

    count = 300
    native = b"".join(_delta(1, 100 + n, 100 + n, 99 + n) for n in range(count))
    native += _line(
        2,
        "depthSnapshot",
        {
            "lastUpdateId": 100 + count - 1,
            "bids": [[str(100 - i), "2"] for i in range(12)],
            "asks": [[str(101 + i), "2"] for i in range(12)],
        },
    )
    native += _delta(3, 100 + count, 100 + count, 99 + count)
    published = tmp_path / "published"
    import_slices(
        published, date="2026-09-01", slices=_slices(tmp_path / "native", native)
    )
    options = dict(
        symbol="BTCUSDT",
        market_type=FUTURES_MARKET_TYPE,
        data_kinds=("depth_snapshot", "depth"),
    )
    with open_market_source(published) as source:
        assert (
            list(
                source.iter_market_records(
                    **options, end_ns=pd.Timestamp("2026-09-01T00:00:01Z").value
                )
            )
            == []
        )
        with (
            closing_iterator(source.iter_market_records(**options)) as first,
            closing_iterator(source.iter_market_records(**options)) as second,
        ):
            snapshot = next(first)
            assert snapshot["data_kind"] == "depth_snapshot"
            assert next(second) == snapshot
            prefix = [next(first) for _ in range(17)]
            other = list(second)
            rest = list(first)
            assert prefix + rest == other
            assert len(other) == count + 1
            assert all(row["data_kind"] == "depth_bootstrap" for row in other[:-1])
            assert other[-1]["data_kind"] == "depth"
        # Reopening must preserve canonical bytes and cutoff behavior.
        assert list(source.iter_market_records(**options)) == [snapshot, *other]
        assert (
            list(
                source.iter_market_records(
                    **options, end_ns=pd.Timestamp("2026-09-01T00:00:01Z").value
                )
            )
            == []
        )


@pytest.mark.parametrize(
    "case", ["ordinary", "bucket_cancellation", "event_cancellation", "true_overflow"]
)
@pytest.mark.parametrize("batch_rows", [1, 10])
def test_native_panel_exact_ofi_publication(tmp_path, case, batch_rows):
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_factor_frames,
        iter_panel_frames,
    )
    from quant_research_platform.markets.implementations.crypto.processed import catalog
    from quant_research_platform.sources.binance.replay import iter_agg_trades
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout

    midnight = pd.Timestamp("2026-09-01T00:00:00Z").value
    quantity = "1" if case == "ordinary" else "1e308"

    def line(ms, channel, payload):
        return (
            pd.Timestamp(midnight + ms * 1000000, tz="UTC")
            .isoformat()
            .replace("+00:00", "Z")
            + " "
            + json.dumps({"stream": "btcusdt@" + channel, "data": payload})
            + "\n"
        ).encode()

    def trade(ms, identity):
        return line(
            ms,
            "aggTrade",
            dict(
                e="aggTrade",
                E=midnight // 1000000 + ms,
                T=midnight // 1000000 + ms,
                s="BTCUSDT",
                st=1,
                a=identity,
                f=identity,
                l=identity,
                p="100",
                q="1",
                nq="1",
                m=False,
            ),
        )

    native = line(
        0,
        "depthSnapshot",
        dict(
            lastUpdateId=100,
            bids=[["100", quantity], *[[str(100 - i), "1"] for i in range(1, 12)]],
            asks=[["200", quantity], *[[str(200 + i), "1"] for i in range(1, 12)]],
        ),
    )
    native += trade(0, 1)
    updates = [(0, [], []), (10, [["100", "0"], ["101", quantity]], [])]
    if case == "event_cancellation":
        updates[1] = (
            10,
            [["100", "0"], ["101", quantity]],
            [["200", "0"], ["201", quantity]],
        )
        updates.append((20, [["101", "0"], ["100", quantity]], []))
    else:
        updates.append((20, [["101", "0"], ["102", quantity]], []))
        if case != "true_overflow":
            updates.append((30, [["102", "0"], ["101", quantity]], []))
    updates.append((200, [], []))
    for number, (ms, bids, asks) in enumerate(updates):
        native += line(
            ms,
            "depth@0ms",
            dict(
                e="depthUpdate",
                E=midnight // 1000000 + ms,
                T=midnight // 1000000 + ms,
                s="BTCUSDT",
                st=1,
                ps="BTCUSDT",
                U=100 + number,
                u=100 + number,
                pu=99 + number,
                b=bids,
                a=asks,
            ),
        )
    native += trade(200, 2)
    published = tmp_path / "raw"
    import_slices(
        published,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native, count=1),
    )
    with open_market_source(published) as source:
        stats = ReconstructionStats()
        frames = iter_panel_frames(
            iter_reconstructed_depth(
                source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
            ),
            iter_agg_trades(source, FUTURES_MARKET_TYPE, symbol="BTCUSDT"),
            market_id="crypto",
            asset_id="btc_usdt",
            start_ms=midnight // 1000000,
            end_ms=midnight // 1000000 + 100,
            book_coverage=stats.coverage_segments,
            trade_coverage=source.observation_coverage(
                market_type=FUTURES_MARKET_TYPE,
                data_kind="agg_trades",
                symbol="BTCUSDT",
            ).covering,
            batch_rows=batch_rows,
        )
        spec = catalog.hft_processed_catalog().get("binance_futures_l2_panel_100ms")
        layout = StorageLayout(tmp_path)
        output = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        if case == "true_overflow":
            with pytest.raises(OverflowError):
                write_dataset(output, frames, spec=spec)
            assert not output.exists()
        else:
            write_dataset(output, frames, spec=spec)
            panel = read_dataset(output)
            assert panel.book_ofi_100ms.iloc[-1] == float(quantity)
            factor = pd.concat(iter_factor_frames([panel], columns=("ofi_100ms",)))
            assert factor.ofi_100ms.iloc[-1] == float(quantity)
        assert stats.invalid_segments == 0


def test_late_snapshot_prefix_excludes_previous_observation(tmp_path):
    snapshot = {
        "lastUpdateId": 200,
        "bids": [[str(100 - i), "2"] for i in range(12)],
        "asks": [[str(101 + i), "2"] for i in range(12)],
    }
    native = (
        _delta(1, 100, 100, 99)
        + b"\n"
        + _delta(2, 200, 200, 199)
        + _line(3, "depthSnapshot", snapshot)
        + _delta(4, 201, 201, 200)
    )
    published = tmp_path / "published"
    import_slices(
        published,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native, count=1),
    )
    with open_market_source(published) as source:
        rows = list(
            source.iter_market_records(
                symbol="BTCUSDT",
                market_type=FUTURES_MARKET_TYPE,
                data_kinds=("depth_snapshot", "depth"),
            )
        )
        assert [row["source_sequence"] for row in rows] == [4, 3, 5]
        assert [row["data_kind"] for row in rows] == [
            "depth_snapshot",
            "depth_bootstrap",
            "depth",
        ]
        assert len({row["observation_id"] for row in rows}) == 1
        stats = ReconstructionStats()
        books = list(
            iter_reconstructed_depth(
                source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
            )
        )
        assert len(books) == 1 and books[0].source_sequence == 5
        assert (
            list(stats.coverage_segments.values())[0][0]
            == pd.Timestamp("2026-09-01T00:00:04Z").value
        )
