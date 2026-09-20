"""Native archive identity, date, gap and atomic failure regressions."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.tardis.raw import import_slices, iter_native_slice
from tests.helpers.runtime_resources import capture_test_policy


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
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        result = import_slices(
            destination,
            date="2026-09-01",
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
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
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "first",
    [_snapshot("2026-09-02T00:00:01Z"), b"2026-09-01T00:00:01Z {}\n", b"broken"],
)
def test_bad_native_input_is_not_published(tmp_path, first):
    slices = _slices(tmp_path / "source", first)
    destination = tmp_path / "published"
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        with pytest.raises(ValueError):
            import_slices(
                destination,
                date="2026-09-01",
                slices=slices,
                byte_budget=1024**2,
                resource_policy=policy,
            )
        assert not destination.exists()
        assert policy.used_memory_bytes == 0


def test_byte_budget_failure_preserves_input_and_removes_staging(tmp_path):
    slices = _slices(tmp_path / "source", _snapshot())
    before = slices[0].read_bytes()
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        with pytest.raises(ValueError, match="tardis_download_budget_exceeded"):
            import_slices(
                tmp_path / "published",
                date="2026-09-01",
                slices=slices,
                byte_budget=1,
                resource_policy=policy,
            )
        assert slices[0].read_bytes() == before
        assert not (tmp_path / "published").exists()
        assert policy.used_memory_bytes == 0


def test_native_line_order_is_preserved_without_a_fabricated_clock(tmp_path):
    path = tmp_path / "native.gz"
    path.write_bytes(gzip.compress(_snapshot() + b"\n" + _snapshot(), mtime=0))
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        records = iter_native_slice(
            path, date="2026-09-01", offset_minutes=0, resource_policy=policy
        )
        rows = [(number, timestamp) for number, timestamp, _ in records]
        assert rows == [
            (1, "2026-09-01T00:00:01.000000Z"),
            (2, None),
            (3, "2026-09-01T00:00:01.000000Z"),
        ]
        assert policy.used_memory_bytes == 0


def test_bad_native_decode_releases_payload_inside_record_lease(tmp_path, monkeypatch):
    import weakref

    from quant_research_platform.sources.tardis import raw

    path = tmp_path / "invalid.gz"
    path.write_bytes(gzip.compress(_snapshot().replace(b"depthSnapshot", b"unknown")))
    decoder = raw.parse_qrp_json_object
    released = []

    class Tracked(dict):
        pass

    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:

        def parse(value):
            result = Tracked(decoder(value))
            weakref.finalize(result, lambda: released.append(policy.used_memory_bytes))
            return result

        monkeypatch.setattr(raw, "parse_qrp_json_object", parse)
        records = raw.iter_native_slice(
            path, date="2026-09-01", offset_minutes=0, resource_policy=policy
        )
        with pytest.raises(ValueError, match="tardis_native_channel_invalid"):
            next(records)
        assert released and released[0] > 0
        assert policy.used_memory_bytes == 0


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
    with runtime_resource_policy_scope() as policy:
        rows = iter_native_slice(
            path,
            date="2026-09-01",
            offset_minutes=0,
            resource_policy=policy,
            expected_sha256=expected,
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
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("count", [1, 2, 289])
def test_archive_duration_and_source_identity_are_not_limited_to_one_day(
    tmp_path, count
):
    from quant_research_platform.sources.binance.replay import open_market_source

    slices = _slices(tmp_path / "native", count=count)
    with runtime_resource_policy_scope() as policy:
        manifest = import_slices(
            tmp_path / "published",
            date="2026-12-31",
            offset_minutes=1430,
            slices=slices,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        assert len(manifest["files"]) == count
        assert manifest["files"][-1]["offset_minutes"] == 1430 + (count - 1) * 10
        with open_market_source(tmp_path / "published", policy) as source:
            assert source.hft_source_info()["source_file_count"] == count
        assert policy.used_memory_bytes == 0


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
    with runtime_resource_policy_scope() as policy:
        download_slices(
            tmp_path / "published",
            date="2026-12-31",
            offset_minutes=1430,
            slice_count=3,
            byte_budget=1024**2,
            resource_policy=policy,
        )
        assert policy.used_memory_bytes == 0
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
