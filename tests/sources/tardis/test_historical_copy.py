"""Historical member dates and real-file copy lifecycle regressions."""

import datetime as dt
import gzip
import io
import json

import pytest

from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.tardis import normalized, raw
from quant_research_platform.sources.tardis.replay import open_tardis_archive


@pytest.mark.parametrize(
    ("exchange", "date", "offset", "frequency", "error"),
    (
        ("binance-futures", "2020-01-06", 0, "100ms", None),
        ("binance-futures", "2020-01-08", 0, "0ms", None),
        ("binance-futures", "2020-01-06", 2880, "0ms", None),
        ("binance-futures", "2020-01-08", 0, "100ms", "frequency_unsupported"),
        ("binance-futures", "2020-01-06", 0, "0ms", "frequency_unsupported"),
        ("binance-futures", "2020-01-06", 1440, "100ms", "transition_unproven"),
        ("binance-futures", "2020-01-06", 1440, "0ms", "transition_unproven"),
        ("binance", "2020-01-07", 0, "100ms", None),
        ("binance", "2020-01-07", 0, "0ms", "frequency_unsupported"),
    ),
)
def test_historical_frequency_binds_actual_member_date(
    tmp_path, exchange, date, offset, frequency, error
):
    observed = dt.datetime.combine(dt.date.fromisoformat(date), dt.time(), dt.UTC)
    observed += dt.timedelta(minutes=offset, milliseconds=1)
    milliseconds = int(observed.timestamp() * 1000)
    snapshot = {
        "stream": "btcusdt@depthSnapshot",
        "generated": True,
        "data": {"lastUpdateId": 100, "bids": [["100", "2"]], "asks": [["101", "2"]]},
    }
    depth = {
        "stream": f"btcusdt@depth@{frequency}",
        "data": {
            "e": "depthUpdate",
            "E": milliseconds,
            "T": milliseconds,
            "s": "BTCUSDT",
            "U": 101,
            "u": 101,
            "pu": 100,
            "b": [["100", "2"]],
            "a": [["101", "2"]],
        },
    }
    timestamp = observed.isoformat().replace("+00:00", "Z")
    original = gzip.compress(
        b"".join(
            (timestamp + " " + json.dumps(value) + "\n").encode()
            for value in (snapshot, depth)
        ),
        mtime=0,
    )
    source = tmp_path / "native.gz"
    source.write_bytes(original)
    destination = tmp_path / "published"
    options = dict(
        date=date,
        offset_minutes=offset,
        exchange=exchange,
        symbols=("btcusdt",),
        channels=("depthSnapshot", "depth"),
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            list(raw.iter_native_slice(source, **options))
        with pytest.raises(ValueError, match=error):
            raw.import_slices(destination, slices=(source,), **options)
        assert not destination.exists()
        assert source.read_bytes() == original
        return
    manifest = raw.import_slices(destination, slices=(source,), **options)
    member = manifest["files"][0]
    assert member["record_counts"][depth["stream"]] == 1
    assert (destination / member["path"]).read_bytes() == original
    with open_tardis_archive(destination) as archive:
        records = list(
            archive.iter_market_records(
                symbol="BTCUSDT",
                market_type=FUTURES_MARKET_TYPE
                if exchange == "binance-futures"
                else SPOT_MARKET_TYPE,
                data_kinds=("depth_snapshot", "depth"),
            )
        )
        assert [record["data_kind"] for record in records] == [
            "depth_snapshot",
            "depth",
        ]
        assert records[-1]["payload"]["u"] == 101


@pytest.mark.parametrize("offset,count", ((1440, 1), (1430, 2)))
def test_transition_empty_depth_members_are_rejected_before_publication(
    tmp_path, offset, count
):
    source = tmp_path / "empty.gz"
    source.write_bytes(gzip.compress(b"", mtime=0))
    with pytest.raises(ValueError, match="transition_unproven"):
        raw.import_slices(
            tmp_path / "published",
            date="2020-01-06",
            offset_minutes=offset,
            slices=(source,) * count,
            channels=("depth",),
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["empty.gz"]


@pytest.mark.parametrize("backend", ("raw", "normalized"))
@pytest.mark.parametrize(
    "boundary,cancel,close_error",
    [
        ("none", False, "none"),
        ("read", False, "both"),
        ("write", False, "both"),
        ("read", True, "both"),
        ("flush", False, "none"),
        ("none", False, "response"),
        ("none", False, "output"),
        ("none", False, "both"),
    ],
)
def test_copy_closes_real_files_and_preserves_primary_identity(
    tmp_path, monkeypatch, backend, boundary, cancel, close_error
):
    module = raw if backend == "raw" else normalized
    primary = (
        KeyboardInterrupt("synthetic_copy_cancel")
        if cancel
        else OSError("synthetic_copy_primary")
    )
    response_error = OSError("synthetic_response_close")
    output_error = OSError("synthetic_output_close")
    content = (
        b""
        if backend == "raw"
        else (
            b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n"
            b"bybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\n"
        )
    )
    original = gzip.compress(content, mtime=0)
    outputs = []

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Encoding": "gzip"}
        close_count = 0

        def read(self, size=-1):
            if boundary == "read" and self.tell():
                raise primary
            return super().read(size)

        def close(self):
            self.close_count += 1
            super().close()
            if close_error in {"response", "both"}:
                raise response_error

    class Output:
        def __init__(self, handle):
            self.handle = handle
            self.close_count = 0

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, data):
            if boundary == "write":
                self.handle.write(data[:1])
                raise primary
            return self.handle.write(data)

        def flush(self):
            self.handle.flush()
            if boundary == "flush":
                raise primary

        def close(self):
            self.close_count += 1
            self.handle.close()
            if close_error in {"output", "both"}:
                raise output_error

    response = Response(original)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: response)
    open_file = module.open_filesystem_file

    def open_output(path, mode, **kwargs):
        handle = open_file(path, mode, **kwargs)
        if mode != "xb":
            return handle
        output = Output(handle)
        outputs.append(output)
        return output

    monkeypatch.setattr(module, "open_filesystem_file", open_output)
    destination = tmp_path / "published"

    def publish():
        if backend == "raw":
            return raw.download_slices(destination, date="2026-08-01", slice_count=1)
        return normalized.download_dataset(
            destination,
            exchange="bybit",
            dataset_type="trades",
            symbol="BTCUSDT",
            date="2026-08-01",
        )

    expected = (
        primary
        if boundary != "none"
        else output_error
        if close_error in {"output", "both"}
        else response_error
        if close_error == "response"
        else None
    )
    if expected is None:
        manifest = publish()
        path = manifest["files"][0]["path"] if backend == "raw" else manifest["path"]
        assert (destination / path).read_bytes() == original
    else:
        with pytest.raises(type(expected)) as caught:
            publish()
        assert caught.value is expected
        assert not destination.exists()
        if boundary != "none" and close_error != "none":
            assert any("OSError" in note for note in primary.__notes__)
    assert response.closed and response.close_count == 1
    assert len(outputs) == 1
    assert outputs[0].handle.closed and outputs[0].close_count == 1
    assert all(not path.name.startswith(".a-") for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    "termination", ("exhaust", "early", "consumer", "read", "cancel")
)
@pytest.mark.parametrize("close_fails", (False, True))
def test_verified_input_closes_on_every_exit_without_masking_primary(
    tmp_path, monkeypatch, termination, close_fails
):
    source = tmp_path / "source.gz"
    original = gzip.compress(b"", mtime=0)
    source.write_bytes(original)
    primary = (
        KeyboardInterrupt("synthetic_input_cancel")
        if termination == "cancel"
        else OSError("synthetic_input_primary")
    )
    secondary = OSError("synthetic_input_close")
    handles = []
    open_file = raw.open_filesystem_file

    class Input:
        def __init__(self, handle):
            self.handle = handle
            self.close_count = 0

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            if termination == "read":
                raise primary
            return self.handle.read(size)

        def close(self):
            self.close_count += 1
            self.handle.close()
            if close_fails:
                raise secondary

    def open_input(path, mode):
        handle = Input(open_file(path, mode))
        handles.append(handle)
        return handle

    monkeypatch.setattr(raw, "open_filesystem_file", open_input)

    def consume():
        with raw._verified_file(source) as handle:
            handle.read(1 if termination == "early" else -1)
            if termination in {"consumer", "cancel"}:
                raise primary

    expected = (
        primary
        if termination in {"consumer", "read", "cancel"}
        else secondary
        if close_fails
        else None
    )
    if expected is None:
        consume()
    else:
        with pytest.raises(type(expected)) as caught:
            consume()
        assert caught.value is expected
        if expected is primary and close_fails:
            assert any("OSError" in note for note in primary.__notes__)
    assert len(handles) == 1
    assert handles[0].handle.closed and handles[0].close_count == 1
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    ("date", "offset", "suffix", "error"),
    (
        ("2020-01-01", 0, "markPrice", None),
        ("2020-02-12", 0, "markPrice", None),
        ("2020-02-14", 0, "markPrice@1s", None),
        ("2020-02-12", 2880, "markPrice@1s", None),
        ("2020-02-12", 0, "markPrice@1s", "frequency_unsupported"),
        ("2020-02-14", 0, "markPrice", "frequency_unsupported"),
        ("2020-02-12", 1440, "markPrice", "transition_unproven"),
        ("2020-02-12", 1440, "markPrice@1s", "transition_unproven"),
    ),
)
def test_historical_markprice_defaults_and_funding_binds_member_date(
    tmp_path, date, offset, suffix, error
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.sources.binance.funding import (
        read_funding_observations,
    )
    from quant_research_platform.sources.tardis.funding import import_funding_history

    start = dt.datetime.combine(dt.date.fromisoformat(date), dt.time(), dt.UTC)
    start += dt.timedelta(minutes=offset)
    start_ms = int(start.timestamp()) * 1000
    observed = start + dt.timedelta(milliseconds=1)
    stream = "btcusdt@" + suffix
    payload = {
        "stream": stream,
        "data": {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "E": start_ms + 1,
            "p": "100",
            "r": "0.05",
            "T": start_ms + 10000,
        },
    }
    original = gzip.compress(
        (
            observed.isoformat().replace("+00:00", "Z")
            + " "
            + json.dumps(payload)
            + "\n"
        ).encode(),
        mtime=0,
    )
    source = tmp_path / "native.gz"
    source.write_bytes(original)
    destination = tmp_path / "published"
    options = dict(date=date, offset_minutes=offset)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            list(raw.iter_native_slice(source, **options))
        with pytest.raises(ValueError, match=error):
            raw.import_slices(destination, slices=(source,), **options)
        assert not destination.exists()
        return
    manifest = raw.import_slices(destination, slices=(source,), **options)
    assert manifest["requested_channels"] == list(raw.CHANNELS)
    member = manifest["files"][0]
    assert member["record_counts"] == {stream: 1}
    assert (destination / member["path"]).read_bytes() == original
    assert (
        list(raw.iter_native_slice(destination / member["path"], **options))[0][2]
        == payload
    )
    response = tmp_path / "settlements.json"
    response.write_text(
        json.dumps(
            [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": start_ms + 10000,
                    "fundingRate": "0.1",
                    "markPrice": "200",
                }
            ]
        )
    )
    with open_tardis_archive(destination) as archive:
        import_funding_history(
            archive,
            symbol="BTCUSDT",
            source_id="binance",
            request_limit=1000,
            response_path=response,
            http_status=200,
            start_ms=start_ms,
            end_ms=start_ms + 600000 - 1,
        )
        roots = archive.funding_roots(
            "BTCUSDT",
            time_window=TimeWindow(start.date().isoformat(), start.date().isoformat()),
        )
        with read_funding_observations(roots, symbol="BTCUSDT") as funding:
            forecasts = tuple(funding.forecasts)
            assert len(forecasts) == 1
            assert forecasts[0].available_at_ns == (start_ms + 1) * 1000000
            assert forecasts[0].cashflows == (((start_ms + 10000) * 1000000, -5.0),)
            assert funding.events[0].rate == 0.1


@pytest.mark.parametrize("offset,count", ((1440, 1), (1430, 2)))
def test_markprice_transition_empty_default_members_are_not_published(
    tmp_path, offset, count
):
    source = tmp_path / "empty.gz"
    source.write_bytes(gzip.compress(b"", mtime=0))
    with pytest.raises(ValueError, match="mark_price_frequency_transition_unproven"):
        raw.import_slices(
            tmp_path / "published",
            date="2020-02-12",
            offset_minutes=offset,
            slices=(source,) * count,
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["empty.gz"]
