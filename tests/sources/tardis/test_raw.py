"""Native archive identity, date, gap and atomic failure regressions."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from quant_research_platform.sources.tardis.raw import import_slices, iter_native_slice


def _slices(root: Path, first: bytes = b"", *, count=144):
    root.mkdir()
    result = []
    for offset in range(0, count * 10, 10):
        path = root / f"native-{offset:04d}.gz"
        path.write_bytes(gzip.compress(first if offset == 0 else b"", mtime=0))
        result.append(path)
    return result


def _snapshot(timestamp="2026-09-01T00:00:01.000000Z"):
    value = {
        "stream": "btcusdt@depthSnapshot",
        "generated": True,
        "data": {"lastUpdateId": 100, "bids": [["100", "2"]], "asks": [["101", "2"]]},
    }
    return (timestamp + " " + json.dumps(value) + "\n").encode()


def test_import_preserves_native_compressed_bytes_and_disconnect_line(tmp_path):
    slices = _slices(tmp_path / "source", _snapshot() + b"\n")
    destination = tmp_path / "published"
    result = import_slices(
        destination, date="2026-09-01", slices=slices, byte_budget=1024**2
    )
    assert len(result["files"]) == 144
    for source, member in zip(slices, result["files"], strict=True):
        assert member["path"].endswith(".txt.gz")
        actual = (destination / member["path"]).read_bytes()
        assert actual == source.read_bytes()
        assert hashlib.sha256(actual).hexdigest() == member["sha256"]
    gap = result["gaps"]
    with gzip.open(destination / gap["path"], "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows == [
        {
            "reason": "vendor_disconnect",
            "offset_minutes": 0,
            "line": 2,
            "observed_utc": None,
        }
    ]
    assert result["observation_clock"] == "vendor_utc"
    assert result["ordering"] == "slice_offset_then_native_line"
    assert "capture" not in result
    assert "market_metadata" not in result


@pytest.mark.parametrize(
    "first",
    [_snapshot("2026-09-02T00:00:01Z"), b"2026-09-01T00:00:01Z {}\n", b"broken"],
)
def test_bad_native_input_is_not_published(tmp_path, first):
    slices = _slices(tmp_path / "source", first)
    destination = tmp_path / "published"
    with pytest.raises(ValueError):
        import_slices(
            destination, date="2026-09-01", slices=slices, byte_budget=1024**2
        )
    assert not destination.exists()


def test_byte_budget_failure_preserves_input_and_removes_staging(tmp_path):
    slices = _slices(tmp_path / "source", _snapshot())
    before = slices[0].read_bytes()
    with pytest.raises(ValueError, match="tardis_download_budget_exceeded"):
        import_slices(
            tmp_path / "published", date="2026-09-01", slices=slices, byte_budget=1
        )
    assert slices[0].read_bytes() == before
    assert not (tmp_path / "published").exists()


def test_native_line_order_is_preserved_without_a_fabricated_clock(tmp_path):
    path = tmp_path / "native.gz"
    path.write_bytes(gzip.compress(_snapshot() + b"\n" + _snapshot(), mtime=0))
    records = iter_native_slice(path, date="2026-09-01", offset_minutes=0)
    rows = [(number, timestamp) for number, timestamp, _ in records]
    assert rows == [
        (1, "2026-09-01T00:00:01.000000Z"),
        (2, None),
        (3, "2026-09-01T00:00:01.000000Z"),
    ]


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("termination", ["exhaust", "close", "primary"])
def test_native_consumed_digest_is_verified_on_every_exit(
    tmp_path, changed, termination
):
    path = tmp_path / "native.gz"
    original = gzip.compress(_snapshot() + b"\n", mtime=0)
    expected = hashlib.sha256(original).hexdigest()
    path.write_bytes(
        gzip.compress(_snapshot().replace(b'"2"', b'"9"') + b"\n", mtime=0)
        if changed
        else original
    )
    rows = iter_native_slice(
        path, date="2026-09-01", offset_minutes=0, expected_sha256=expected
    )
    next(rows)
    if termination == "primary":
        primary = KeyboardInterrupt("body_primary")
        with pytest.raises(KeyboardInterrupt) as caught:
            rows.throw(primary)
        assert caught.value is primary
        assert bool(getattr(primary, "__notes__", [])) == changed
    elif changed:
        with pytest.raises(ValueError, match="tardis_raw_identity_changed"):
            list(rows) if termination == "exhaust" else rows.close()
    elif termination == "exhaust":
        list(rows)
    else:
        rows.close()


@pytest.mark.parametrize("count", [1, 2, 289])
def test_archive_duration_and_source_identity_are_not_limited_to_one_day(
    tmp_path, count
):
    from quant_research_platform.sources import open_market_source

    slices = _slices(tmp_path / "native", count=count)
    manifest = import_slices(
        tmp_path / "published",
        date="2026-12-31",
        offset_minutes=1430,
        slices=slices,
        byte_budget=1024**2,
    )
    assert len(manifest["files"]) == count
    assert manifest["files"][-1]["offset_minutes"] == 1430 + (count - 1) * 10
    with open_market_source(tmp_path / "published") as source:
        assert source.hft_source_info()["source_file_count"] == count


def test_download_crosses_midnight_using_native_calendar_addresses(
    tmp_path, monkeypatch
):
    import io
    from urllib.parse import parse_qs, urlparse

    from quant_research_platform.sources.tardis.raw import download_slices

    requests = []

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Encoding": "gzip"}

    def receive(request, *, timeout):
        assert timeout == 60
        requests.append(parse_qs(urlparse(request.full_url).query))
        return Response(gzip.compress(b"", mtime=0))

    monkeypatch.setattr("urllib.request.urlopen", receive)
    download_slices(
        tmp_path / "published",
        date="2026-12-31",
        offset_minutes=1430,
        slice_count=3,
        byte_budget=1024**2,
    )
    assert [(r["from"][0], r["offset"][0]) for r in requests] == [
        ("2026-12-31", "1430"),
        ("2027-01-01", "0"),
        ("2027-01-01", "10"),
    ]


@pytest.mark.parametrize("offset,count", [(-10, 1), (1, 1), (0, 0), (True, 1)])
def test_invalid_slice_range_is_not_published(tmp_path, offset, count):
    slices = _slices(tmp_path / "native", count=count)
    with pytest.raises(ValueError, match="tardis_slice_range_invalid"):
        import_slices(
            tmp_path / "published",
            date="2026-09-01",
            offset_minutes=offset,
            slices=slices,
            byte_budget=1024**2,
        )
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize(
    "exchange,symbol,channels",
    [
        ("binance", "btcusdt", ("depth", "depthSnapshot", "aggTrade")),
        (
            "binance-futures",
            "solusdt",
            ("depth", "depthSnapshot", "aggTrade", "markPrice"),
        ),
        ("binance-futures", "ethusdt", ("openInterest",)),
    ],
)
def test_approved_native_scopes_publish_and_reopen(
    tmp_path, exchange, symbol, channels
):
    from quant_research_platform.sources.tardis.replay import open_tardis_archive

    if channels == ("openInterest",):
        native = (
            "2026-09-01T00:00:01Z "
            + json.dumps(
                {
                    "stream": symbol + "@openInterest",
                    "generated": True,
                    "data": {
                        "symbol": symbol.upper(),
                        "openInterest": "2",
                        "time": 1788220801000,
                    },
                }
            )
            + "\n"
        ).encode()
    else:
        native = _snapshot().replace(b"btcusdt", symbol.encode())
    slices = _slices(tmp_path / "input", native, count=1)
    root = tmp_path / "archive"
    result = import_slices(
        root,
        date="2026-09-01",
        slices=slices,
        exchange=exchange,
        symbols=(symbol,),
        channels=channels,
        byte_budget=1024**2,
    )
    assert result["exchange"] == exchange
    assert result["symbols"] == [symbol]
    assert result["requested_channels"] == list(channels)
    with open_tardis_archive(root) as source:
        (archive,) = source.archives
        assert archive.manifest == result
        assert archive.source_identity["manifest_hash"]
        assert source.source_identity["archives"] == [
            {"date": result["date"], **archive.source_identity}
        ]


@pytest.mark.parametrize(
    "exchange,symbols,channels",
    [
        ("bybit", ("btcusdt",), ("depth",)),
        ("binance", ("solusdt",), ("depth",)),
        ("binance", ("btcusdt",), ("markPrice",)),
        ("binance-futures", ("xrpusdt",), ("depth",)),
        ("binance-futures", ("btcusdt",), ("not_allowed",)),
    ],
)
def test_native_scope_rejects_unapproved_requests(
    tmp_path, exchange, symbols, channels
):
    with pytest.raises(ValueError, match="tardis_request_scope_invalid"):
        import_slices(
            tmp_path / "out",
            date="2026-09-01",
            slices=[],
            exchange=exchange,
            symbols=symbols,
            channels=channels,
            byte_budget=1024,
        )
    assert not (tmp_path / "out").exists()


def test_depth_repeats_preserve_high_water_and_raw_observation_boundaries():
    from quant_research_platform.sources.binance.replay import native_trade_identities
    from quant_research_platform.sources.tardis.raw import _ObservedStreams

    observe_trade = native_trade_identities()
    tracker = _ObservedStreams(exchange="binance", observe_trade=observe_trade)

    def depth(last, first=None):
        return {
            "stream": "btcusdt@depth",
            "data": {"U": last if first is None else first, "u": last},
        }

    first, _, gap = tracker.observe(0, 1, "2026-07-01T00:00:00Z", depth(10))
    assert gap is None
    for number, last in ((2, 10), (3, 9)):
        identity, closed, gap = tracker.observe(
            0, number, f"2026-07-01T00:00:0{number}Z", depth(last)
        )
        assert identity == first and closed == [] and (gap is None)
        assert tracker.previous["btcusdt", "depth"][0] == 10
    identity, _, gap = tracker.observe(0, 4, "2026-07-01T00:00:04Z", depth(11))
    assert identity == first and gap is None
    _, closed, gap = tracker.observe(0, 5, "2026-07-01T00:00:05Z", None)
    assert len(closed) == 1 and gap["reason"] == "vendor_disconnect"
    identity, _, gap = tracker.observe(0, 6, "2026-07-01T00:00:06Z", depth(11))
    assert identity != first and gap is None
    _, closed, gap = tracker.observe(0, 7, "2026-07-01T00:00:07Z", depth(14))
    assert len(closed) == 1
    assert gap["reason"] == "native_sequence_gap"
    assert gap["previous_id"] == 11
    trade = {"stream": "btcusdt@aggTrade", "data": {"a": 1}}
    tracker.observe(0, 8, "2026-07-01T00:00:08Z", trade)
    trade["data"]["q"] = "2"
    with pytest.raises(ValueError, match="sequence_non_monotonic"):
        tracker.observe(0, 9, "2026-07-01T00:00:09Z", trade)


def test_native_trade_history_preserves_exact_identities():
    from quant_research_platform.sources.binance import replay

    count = 512
    observe = replay.native_trade_identities()
    for trade_id in range(count):
        assert observe("future", "SYNTHETIC", {"a": trade_id, "q": "1"}) == (
            True,
            None if trade_id == 0 else trade_id - 1,
        )
    for trade_id in (0, count - 1):
        assert observe("future", "SYNTHETIC", {"a": trade_id, "q": "1", "E": 99}) == (
            False,
            count - 1,
        )
        with pytest.raises(ValueError, match="binance_native_sequence_non_monotonic"):
            observe("future", "SYNTHETIC", {"a": trade_id, "q": "2"})
    assert observe("future", "SYNTHETIC", {"a": count}) == (True, count - 1)


def test_trade_order_failure_retains_safe_source_coordinates(tmp_path):
    from tests.sources.tardis.test_replay import _line

    def trade(identifier):
        return _line(
            1,
            "aggTrade",
            {
                "e": "aggTrade",
                "E": 1788220801000,
                "T": 1788220801000,
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

    with pytest.raises(
        ValueError, match="binance_native_sequence_non_monotonic"
    ) as caught:
        import_slices(
            tmp_path / "published",
            date="2026-09-01",
            slices=_slices(tmp_path / "native", trade(10) + trade(9), count=1),
            byte_budget=1024**2,
        )
    note = " ".join(caught.value.__notes__)
    assert all(
        value in note
        for value in (
            "channel=aggTrade",
            "slice_offset=0",
            "line=2",
            "previous_id=10",
            "current_id=9",
        )
    )
    assert "payload" not in note and "price" not in note
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("body_fails", [False, True])
def test_import_iterator_close_preserves_first_failure(
    tmp_path, monkeypatch, body_fails
):
    import quant_research_platform.sources.tardis.raw as raw

    primary = ValueError("synthetic_observation_failure")
    secondary = OSError("synthetic_close_failure")
    original = raw.iter_native_slice

    def records(*args, **kwargs):
        with raw.closing_iterator(original(*args, **kwargs)) as source:
            try:
                yield from source
            finally:
                raise secondary

    def observe(*args):
        raise primary

    monkeypatch.setattr(raw, "iter_native_slice", records)
    if body_fails:
        monkeypatch.setattr(raw._ObservedStreams, "observe", observe)
    with pytest.raises(ValueError if body_fails else OSError) as caught:
        import_slices(
            tmp_path / "published",
            date="2026-09-01",
            slices=_slices(tmp_path / "native", _snapshot(), count=1),
            byte_budget=1024**2,
        )
    assert caught.value is (primary if body_fails else secondary)
    if body_fails:
        assert any("OSError" in note for note in primary.__notes__)
    assert not (tmp_path / "published").exists()
