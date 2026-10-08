"""Native archive identity, date, gap and atomic failure regressions."""

import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from quant_research_platform.sources.tardis.raw import import_slices, iter_native_slice
from tests.helpers.tardis import _slices, _snapshot


def test_import_preserves_native_compressed_bytes_and_disconnect_line(tmp_path):
    slices = _slices(tmp_path / "source", _snapshot() + b"\n")
    destination = tmp_path / "published"
    result = import_slices(destination, date="2026-09-01", slices=slices)
    assert len(result["files"]) == 1
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
        import_slices(destination, date="2026-09-01", slices=slices)
    assert not destination.exists()


def test_import_preserves_input_without_byte_quota(tmp_path):
    slices = _slices(tmp_path / "source", _snapshot())
    before = slices[0].read_bytes()
    destination = tmp_path / "published"
    import_slices(destination, date="2026-09-01", slices=slices)
    assert slices[0].read_bytes() == before
    assert destination.exists()


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


@pytest.mark.parametrize("outer_handled", [False, True])
@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("termination", ["exhaust", "close", "primary"])
def test_native_consumed_digest_is_verified_on_every_exit(
    tmp_path, changed, termination, outer_handled
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

    def finish():
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

    if outer_handled:
        try:
            raise RuntimeError("handled_outer")
        except RuntimeError as outer:
            finish()
            assert not getattr(outer, "__notes__", [])
    else:
        finish()


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
    from tests.helpers.tardis import _line

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
        )
    assert caught.value is (primary if body_fails else secondary)
    if body_fails:
        assert any("OSError" in note for note in primary.__notes__)
    assert not (tmp_path / "published").exists()


class _DownloadResponse(io.BytesIO):
    status = 200
    headers = {"Content-Encoding": "gzip"}

    def __init__(self, data, *, read_failure=None, close_failure=None):
        super().__init__(data)
        self.read_failure = read_failure
        self.close_failure = close_failure
        self.close_calls = 0

    def read(self, size=-1):
        if self.read_failure is not None:
            raise self.read_failure
        return super().read(size)

    def close(self):
        was_closed = self.closed
        super().close()
        if not was_closed:
            self.close_calls += 1
            if self.close_failure is not None:
                raise self.close_failure


def _download_archive(kind, root):
    from quant_research_platform.sources.tardis.normalized import download_dataset
    from quant_research_platform.sources.tardis.raw import download_slices

    if kind == "native":
        return download_slices(root, date="2026-08-01", slice_count=1)
    return download_dataset(
        root,
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
    )


@pytest.mark.parametrize("kind", ["native", "normalized"])
@pytest.mark.parametrize("status", [403, 500])
@pytest.mark.parametrize("close_fails", [False, True])
def test_http_error_response_is_closed_without_replacing_status_failure(
    tmp_path, monkeypatch, kind, status, close_fails
):
    from urllib.error import HTTPError

    body = _DownloadResponse(
        b"synthetic error response",
        close_failure=OSError("synthetic_close_failure") if close_fails else None,
    )
    response = HTTPError("https://synthetic.invalid/", status, "synthetic", {}, body)

    def receive(*args, **kwargs):
        raise response

    monkeypatch.setattr("urllib.request.urlopen", receive)
    destination = tmp_path / "published"
    try:
        with pytest.raises(PermissionError if status == 403 else ValueError) as caught:
            _download_archive(kind, destination)
        assert str(caught.value) == (
            "tardis_access_not_authorized"
            if status == 403
            else "tardis_http_status_500"
        )
        assert body.closed and body.close_calls == 1
        assert bool(getattr(caught.value, "__notes__", [])) == close_fails
        if close_fails:
            assert any("OSError" in note for note in caught.value.__notes__)
        assert not destination.exists()
        assert not list(tmp_path.glob(".a-*"))
    finally:
        body.close_failure = None
        response.close()


@pytest.mark.parametrize("kind", ["native", "normalized"])
@pytest.mark.parametrize("failure_type", [None, OSError, KeyboardInterrupt, SystemExit])
def test_download_response_close_preserves_primary_and_published_archive(
    tmp_path, monkeypatch, kind, failure_type
):
    from quant_research_platform.storage.errors import OutputAlreadyExistsError

    raw = gzip.compress(
        b""
        if kind == "native"
        else (
            b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
            b"bybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\n"
        ),
        mtime=0,
    )
    previous = tmp_path / "previous"
    previous_response = _DownloadResponse(raw)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: previous_response)
    _download_archive(kind, previous)
    assert previous_response.closed and previous_response.close_calls == 1
    original = {
        path.relative_to(previous): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in previous.iterdir()
    }
    primary = None if failure_type is None else failure_type("synthetic_read_failure")
    secondary = OSError("synthetic_close_failure")
    response = _DownloadResponse(raw, read_failure=primary, close_failure=secondary)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
    destination = tmp_path / "candidate"
    with pytest.raises(OSError if primary is None else failure_type) as caught:
        _download_archive(kind, destination)
    assert caught.value is (secondary if primary is None else primary)
    assert response.closed and response.close_calls == 1
    if primary is not None:
        assert any("OSError" in note for note in primary.__notes__)
    assert not destination.exists()
    assert not list(tmp_path.glob(".a-*"))
    with pytest.raises(OutputAlreadyExistsError):
        _download_archive(kind, previous)
    assert response.close_calls == 1
    assert original == {
        path.relative_to(previous): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in previous.iterdir()
    }


@pytest.mark.parametrize("member_kind", ["files", "gaps", "observation_intervals"])
@pytest.mark.parametrize("during_read", [False, True])
def test_declared_native_member_missing_is_invalid_not_unacquired(
    tmp_path, member_kind, during_read
):
    from quant_research_platform.sources import (
        market_source_identity,
        open_market_source,
    )
    from quant_research_platform.sources.tardis.replay import native_source_identity

    root = tmp_path / "published"
    scope = dict(
        exchange="binance-futures",
        symbols=("btcusdt", "ethusdt"),
        channels=("depth", "depthSnapshot", "aggTrade", "markPrice"),
    )
    assert market_source_identity(root, "synthetic") is None
    assert native_source_identity(root, "synthetic", **scope) is None
    manifest = import_slices(
        root,
        date="2026-08-01",
        slices=_slices(tmp_path / "native", count=1),
    )
    assert market_source_identity(root, "synthetic") is not None
    assert native_source_identity(root, "synthetic", **scope) is not None
    member = (
        manifest[member_kind][0] if member_kind == "files" else manifest[member_kind]
    )
    path = root / member["path"]
    if during_read:
        primary = KeyboardInterrupt("synthetic_consumer_failure")
        with pytest.raises(KeyboardInterrupt) as caught:
            with open_market_source(root):
                path.unlink()
                raise primary
        assert caught.value is primary
        assert any("ValueError" in note for note in primary.__notes__)
    else:
        path.unlink()
    with pytest.raises(ValueError, match="identity"):
        market_source_identity(root, "synthetic")
    with pytest.raises(ValueError, match="identity"):
        native_source_identity(root, "synthetic", **scope)
    with pytest.raises(ValueError, match="identity"):
        with open_market_source(root):
            pass


def test_native_manifest_removed_after_selection_is_identity_failure(
    tmp_path, monkeypatch
):
    import quant_research_platform.sources.tardis.replay as replay
    from quant_research_platform.sources import market_source_identity

    root = tmp_path / "published"
    import_slices(
        root,
        date="2026-08-01",
        slices=_slices(tmp_path / "native", count=1),
    )
    original = replay._capture_raw_file_identity
    calls = 0

    def capture(path, **kwargs):
        nonlocal calls
        if path == root / "manifest.json":
            calls += 1
            if calls == 2:
                path.unlink()
        return original(path, **kwargs)

    monkeypatch.setattr(replay, "_capture_raw_file_identity", capture)
    with pytest.raises(ValueError, match="identity"):
        market_source_identity(root, "synthetic")
    assert calls == 2


def test_verified_file_preserves_non_missing_contract_failure(tmp_path):
    from quant_research_platform.sources.tardis.raw import _verified_file

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$") as caught:
        with _verified_file(tmp_path):
            pass
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("kind", ("import", "download"))
@pytest.mark.parametrize("member", ("gaps.jsonl.gz", "observation_intervals.jsonl.gz"))
@pytest.mark.parametrize("failure", ("none", "close", "both"))
def test_native_sidecar_digest_preserves_failure_and_existing_archive(
    tmp_path, monkeypatch, kind, member, failure
):
    from quant_research_platform.sources.tardis import raw
    from quant_research_platform.sources.tardis.replay import native_source_identity

    slices = _slices(tmp_path / "source", _snapshot() + b"\n", count=1)
    source_before = slices[0].read_bytes()
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_k: _DownloadResponse(source_before)
    )

    def publish(destination):
        if kind == "import":
            return raw.import_slices(destination, date="2026-09-01", slices=slices)
        return raw.download_slices(destination, date="2026-09-01", slice_count=1)

    good = tmp_path / "good"
    manifest = publish(good)
    good_before = {path.name: path.read_bytes() for path in good.iterdir()}
    identity = native_source_identity(
        good,
        "fixture",
        exchange=raw.EXCHANGE,
        symbols=raw.SYMBOLS,
        channels=raw.CHANNELS,
    )
    assert identity is not None
    for key in ("gaps", "observation_intervals"):
        descriptor = manifest[key]
        assert (
            descriptor["sha256"]
            == hashlib.sha256(good_before[descriptor["path"]]).hexdigest()
        )
        assert descriptor["compressed_bytes"] == len(good_before[descriptor["path"]])
    original = raw.open_filesystem_file
    primary = OSError("synthetic_sidecar_read")
    secondary = RuntimeError("synthetic_sidecar_close")
    opened = []

    class Handle:
        def __init__(self, native):
            self.native = native
            self.close_calls = 0

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def readinto(self, buffer):
            if failure == "both":
                raise primary
            return self.native.readinto(buffer)

        def close(self):
            self.close_calls += 1
            self.native.close()
            if failure in {"close", "both"}:
                raise secondary

    def observed_open(path, mode="rb", **kwargs):
        native = original(path, mode, **kwargs)
        if mode != "rb" or Path(path).name != member:
            return native
        handle = Handle(native)
        opened.append(handle)
        return handle

    monkeypatch.setattr(raw, "open_filesystem_file", observed_open)
    destination = tmp_path / "candidate"
    if failure == "none":
        result = publish(destination)
        key = "gaps" if member == "gaps.jsonl.gz" else "observation_intervals"
        descriptor = result[key]
        assert (
            descriptor["sha256"]
            == hashlib.sha256(
                (destination / descriptor["path"]).read_bytes()
            ).hexdigest()
        )
    else:
        with pytest.raises(BaseException) as caught:
            publish(destination)
        assert caught.value is (primary if failure == "both" else secondary)
        if failure == "both":
            assert primary.__notes__ == ["iterator close also failed: RuntimeError"]
        assert not destination.exists()
        assert not list(tmp_path.glob(".a-*"))
    assert len(opened) == 1
    assert opened[0].close_calls == 1
    assert opened[0].native.closed
    assert source_before == slices[0].read_bytes()
    assert good_before == {path.name: path.read_bytes() for path in good.iterdir()}
    assert identity == native_source_identity(
        good,
        "fixture",
        exchange=raw.EXCHANGE,
        symbols=raw.SYMBOLS,
        channels=raw.CHANNELS,
    )
