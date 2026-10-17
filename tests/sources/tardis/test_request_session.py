"""Request-local Tardis buffers retain exact source and cursor lifetimes."""

import pytest

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.sources.tardis import replay
from quant_research_platform.sources.tardis.raw import import_slices
from tests.helpers.tardis import _line, _slices
from tests.sources.tardis.test_replay import _delta


@pytest.fixture
def archive(tmp_path):
    snapshot = _line(
        0,
        "depthSnapshot",
        {
            "lastUpdateId": 100,
            "bids": [[str(100 - i), "2"] for i in range(12)],
            "asks": [[str(101 + i), "2"] for i in range(12)],
        },
    )
    native = snapshot + _delta(1, 99, 101, 98) + _delta(3, 102, 102, 101)
    root = tmp_path / "published"
    import_slices(
        root, date="2026-09-01", slices=_slices(tmp_path / "native", native, count=1)
    )
    return root


def _records(source):
    return source.iter_market_records(
        symbol="BTCUSDT",
        market_type=FUTURES_MARKET_TYPE,
        data_kinds=("depth_snapshot", "depth"),
    )


@pytest.mark.parametrize("stop", ["normal", "early", "blocked", "error", "interrupt"])
def test_each_computation_closes_owned_decoders_without_replay_spools(
    archive, monkeypatch, stop
):
    native = replay.iter_native_slice
    calls, closed, buffers = [], [], []
    temporary = replay.TemporaryFile

    def counted(*args, **kwargs):
        calls.append(args[0])
        try:
            yield from native(*args, **kwargs)
        finally:
            closed.append(args[0])

    def tracked():
        handle = temporary()
        buffers.append(handle)
        return handle

    monkeypatch.setattr(replay, "iter_native_slice", counted)
    monkeypatch.setattr(replay, "TemporaryFile", tracked)
    with open_market_source(archive) as control:
        expected = list(_records(control))
    calls.clear()
    closed.clear()
    primary = {
        "blocked": PortfolioTargetBlocked,
        "error": ValueError,
        "interrupt": KeyboardInterrupt,
    }.get(stop, ValueError)("controlled_consumer_failure")
    try:
        with open_market_source(archive) as first:
            cursor = _records(first)
            assert next(cursor) == expected[0]
            if stop == "normal":
                assert list(cursor) == expected[1:]
            if stop in {"blocked", "error", "interrupt"}:
                raise primary
    except BaseException as error:
        assert error is primary
    assert cursor.gi_frame is None
    assert calls == closed
    # Snapshot-first input never needs any temporary replay buffer.
    assert not buffers
    with open_market_source(archive) as second:
        assert list(_records(second)) == expected
    assert len(calls) == 2
    assert calls == closed


def test_slow_fast_cursors_are_independent_and_close(archive):
    with open_market_source(archive) as source:
        slow = _records(source)
        head = next(slow)
        fast = _records(source)
        assert next(fast) == head
        tail = list(fast)
        assert list(slow) == tail
    assert slow.gi_frame is None and fast.gi_frame is None


@pytest.mark.parametrize(
    "failure", [ValueError, KeyboardInterrupt, GeneratorExit, SystemExit]
)
def test_failed_decoder_is_not_reused_as_truncated_eof(archive, monkeypatch, failure):
    original = replay.iter_native_slice
    primary = failure("controlled_native_decode_failure")
    calls = []

    def broken(*args, **kwargs):
        calls.append(args[0])
        with replay.closing_iterator(original(*args, **kwargs)) as rows:
            yield next(rows)
            raise primary

    monkeypatch.setattr(replay, "iter_native_slice", broken)
    with open_market_source(archive) as source:
        for _ in range(2):
            with pytest.raises(failure) as caught:
                list(_records(source))
            assert caught.value is primary
    assert len(calls) == 2


def test_many_archives_keep_only_active_bootstrap_prefixes(tmp_path, monkeypatch):
    import json

    import pandas as pd

    paths = []
    for number in range(6):
        native = (
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
        )
        shifted = []
        for line in native.decode().splitlines():
            stamp, payload = line.split(" ", 1)
            body = json.loads(payload)
            for key in ("E", "T"):
                if key in body["data"]:
                    body["data"][key] += number * 600_000
            timestamp = pd.Timestamp(stamp) + pd.Timedelta(minutes=number * 10)
            shifted.append(
                timestamp.isoformat().replace("+00:00", "Z") + " " + json.dumps(body)
            )
        name = f"archive-{number}"
        import_slices(
            tmp_path / name,
            date="2026-09-01",
            offset_minutes=number * 10,
            slices=_slices(
                tmp_path / f"native-{number}",
                ("\n".join(shifted) + "\n").encode(),
                count=1,
            ),
        )
        paths.append(name)
    buffers = []
    peak = 0
    temporary = replay.TemporaryFile

    def tracked():
        nonlocal peak
        stream = temporary()
        buffers.append(stream)
        peak = max(peak, sum(not item.closed for item in buffers))
        return stream

    monkeypatch.setattr(replay, "TemporaryFile", tracked)
    with open_market_source(tmp_path, archive_paths=tuple(reversed(paths))) as source:
        slow = _records(source)
        first = next(slow)
        assert first["data_kind"] == "depth_snapshot"
        expected = list(_records(source))
        assert len(expected) == 18
        assert sum(not item.closed for item in buffers) == 1
        assert [first, *slow] == expected
        assert all(item.closed for item in buffers)
    assert peak == 2
    assert len(buffers) == 12 and all(item.closed for item in buffers)


def test_decoder_failure_does_not_contaminate_an_independent_read(archive, monkeypatch):
    original = replay.iter_native_slice
    failed = False

    def fail_once(*args, **kwargs):
        nonlocal failed
        with replay.closing_iterator(original(*args, **kwargs)) as rows:
            if not failed:
                failed = True
                yield next(rows)
                raise ValueError("first_computation_decode_failure")
            yield from rows

    monkeypatch.setattr(replay, "iter_native_slice", fail_once)
    with open_market_source(archive) as source:
        with pytest.raises(ValueError, match="first_computation_decode_failure"):
            list(_records(source))
        assert len(list(_records(source))) == 3


@pytest.mark.parametrize("kind", ["depth", "state", "agg_trades"])
def test_cursor_tracks_only_requested_stream_and_keeps_global_disconnect(
    tmp_path, monkeypatch, kind
):
    from tests.helpers.tardis import _snapshot

    def trade(second, identifier):
        return _line(
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
                "q": "1",
                "nq": "1",
                "m": False,
            },
        )

    def other(row):
        return row.replace(b"btcusdt", b"ethusdt").replace(b"BTCUSDT", b"ETHUSDT")

    state = {"symbol": "BTCUSDT", "openInterest": "2", "time": 1788220803000}
    native = b"".join(trade(1, i) + other(trade(1, i)) for i in range(20))
    native += (
        _snapshot("2026-09-01T00:00:02.000000Z")
        + _line(3, "openInterest", state)
        + b"\n"
        + other(trade(4, 20))
        + _line(5, "openInterest", dict(state, time=1788220805000))
    )
    root = tmp_path / "published"
    import_slices(
        root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native),
        channels=("depth", "depthSnapshot", "aggTrade", "openInterest"),
    )
    factory = replay.native_trade_identities
    histories = []

    def tracked():
        observed = factory()
        history = []
        histories.append(history)

        def observe(market, symbol, payload):
            history.append((symbol, payload["a"]))
            return observed(market, symbol, payload)

        return observe

    monkeypatch.setattr(replay, "native_trade_identities", tracked)
    with open_market_source(root) as source:
        if kind == "state":
            records = list(
                source.iter_state_observations(
                    symbol="BTCUSDT", channels=("openInterest",)
                )
            )
            assert [(r[0], r[1], r[3]) for r in records] == [
                (1788220803000000000, "openInterest", 0),
                (1788220804000000000, None, 1),
                (1788220805000000000, "openInterest", 1),
            ]
        else:
            records = list(
                source.iter_market_records(
                    symbol="BTCUSDT",
                    market_type=FUTURES_MARKET_TYPE,
                    data_kinds=("depth_snapshot", "depth")
                    if kind == "depth"
                    else ("agg_trades",),
                )
            )
            assert [r["source_sequence"] for r in records] == (
                [41] if kind == "depth" else list(range(1, 41, 2))
            )
        assert histories == (
            [[("BTCUSDT", i) for i in range(20)]] if kind == "agg_trades" else []
        )


def test_cutoff_on_unrequested_stream_does_not_decode_later_error(tmp_path):
    import gzip

    from tests.helpers.tardis import _snapshot

    native = _snapshot("2026-09-01T00:00:00.000000Z") + _delta(1, 99, 101, 98).replace(
        b"btcusdt", b"ethusdt"
    ).replace(b"BTCUSDT", b"ETHUSDT")
    root = tmp_path / "published"
    manifest = import_slices(
        root, date="2026-09-01", slices=_slices(tmp_path / "native", native)
    )
    (root / manifest["files"][0]["path"]).write_bytes(
        gzip.compress(native + b"invalid later row\n")
    )
    with open_market_source(root) as source:
        result = list(
            source.iter_market_records(
                symbol="BTCUSDT",
                market_type=FUTURES_MARKET_TYPE,
                data_kinds=("depth_snapshot", "depth"),
                end_ns=1788220800000000000,
            )
        )
        assert [r["source_sequence"] for r in result] == [1]
        with pytest.raises(ValueError, match="tardis_observed_time_invalid"):
            list(_records(source))
