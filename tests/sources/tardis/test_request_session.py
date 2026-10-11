"""Request-local Tardis buffers retain exact source and cursor lifetimes."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from threading import Barrier

import pytest

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.time import TimeWindow
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


@contextmanager
def _request():
    with ExitStack() as resources:
        resources.enter_context(replay._tardis_request_session(resources))
        yield


@pytest.mark.parametrize("stop", ["normal", "early", "blocked", "error"])
def test_request_reuses_scan_and_closes_each_lease_cursor(archive, monkeypatch, stop):
    native = replay.iter_native_slice
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        yield from native(*args, **kwargs)

    monkeypatch.setattr(replay, "iter_native_slice", counted)
    with open_market_source(archive) as control:
        expected = list(_records(control))
    calls.clear()
    primary = (
        PortfolioTargetBlocked("controlled_coverage_gap")
        if stop == "blocked"
        else ValueError("controlled_consumer_failure")
    )
    with _request():
        try:
            with open_market_source(archive) as first:
                slow = _records(first)
                assert next(slow) == expected[0]
                owner = first.archives[0]
                if stop == "normal":
                    assert list(slow) == expected[1:]
                if stop in {"blocked", "error"}:
                    raise primary
        except (PortfolioTargetBlocked, ValueError) as error:
            assert error is primary
        assert slow.gi_frame is None
        assert owner._records_by_stream
        assert all(not f.closed for f in owner._records_by_stream.values())
        with open_market_source(archive) as second:
            assert second is not first
            assert second.archives[0] is owner
            assert list(_records(second)) == expected
        assert len(calls) == 1
    assert all(f.closed for f in owner._records_by_stream.values())
    assert replay._REQUEST_ARCHIVES.get() is None
    with _request():
        with open_market_source(archive) as third:
            assert third.archives[0] is not owner
            assert list(_records(third)) == expected
    assert len(calls) == 2


def test_nested_leases_have_independent_positions_and_close(archive):
    with _request():
        with open_market_source(archive) as first:
            one = _records(first)
            head = next(one)
            with open_market_source(archive) as second:
                two = _records(second)
                assert next(two) == head
                tail = list(two)
            assert two.gi_frame is None and one.gi_frame is not None
            assert list(one) == tail


def test_request_keys_separate_roots_paths_windows_and_nested_scopes(archive, tmp_path):
    import shutil

    other = tmp_path / "other"
    shutil.copytree(archive, other)
    with _request():
        with open_market_source(archive) as source:
            original = source.archives[0]
        for root, kwargs in (
            (other, {}),
            (archive, {"time_window": TimeWindow("2026-09-01", "2026-09-01")}),
            (archive.parent, {"archive_paths": (archive.name,)}),
        ):
            with open_market_source(root, **kwargs) as separate:
                assert separate.archives[0] is not original
                assert separate.source_identity == source.source_identity
        with _request():
            with open_market_source(archive) as nested:
                assert nested.archives[0] is not original
        with open_market_source(archive) as restored:
            assert restored.archives[0] is original
    assert replay._REQUEST_ARCHIVES.get() is None


def test_simultaneous_requests_have_separate_buffers(archive):
    barrier = Barrier(2)

    def consume(_):
        with _request():
            with open_market_source(archive) as source:
                records = list(_records(source))
                owner = source.archives[0]
                barrier.wait(timeout=20)
            return owner, records

    with ThreadPoolExecutor(max_workers=2) as threads:
        left, right = list(threads.map(consume, range(2)))
    assert left[0] is not right[0] and left[1] == right[1]
    assert all(
        f.closed for item in (left, right) for f in item[0]._records_by_stream.values()
    )
    assert replay._REQUEST_ARCHIVES.get() is None


@pytest.mark.parametrize("phase", ["entry", "exit"])
@pytest.mark.parametrize("member", ["manifest", "raw"])
def test_each_lease_rechecks_source_before_return(archive, phase, member):
    import json

    manifest = archive / "manifest.json"
    metadata = json.loads(manifest.read_text())
    target = (
        manifest if member == "manifest" else archive / metadata["files"][0]["path"]
    )
    original = target.read_bytes()
    with pytest.raises(ValueError):
        with _request():
            with open_market_source(archive) as first:
                list(_records(first))
            if phase == "entry":
                target.write_bytes(original + b" ")
            with pytest.raises(ValueError):
                with open_market_source(archive):
                    assert phase == "exit"
                    target.write_bytes(original + b" ")
    assert replay._REQUEST_ARCHIVES.get() is None


def test_failed_lazy_decoder_cannot_be_reused_as_truncated_eof(archive, monkeypatch):
    original = replay.iter_native_slice
    primary = ValueError("controlled_native_decode_failure")
    calls = 0

    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        with replay.closing_iterator(original(*args, **kwargs)) as rows:
            yield next(rows)
            raise primary

    monkeypatch.setattr(replay, "iter_native_slice", broken)
    with _request():
        for _ in range(2):
            with pytest.raises(ValueError) as caught:
                with open_market_source(archive) as source:
                    list(_records(source))
            assert caught.value is primary
        assert calls == 1
    assert replay._REQUEST_ARCHIVES.get() is None


@pytest.mark.parametrize("phase", ["entry", "exit"])
@pytest.mark.parametrize("change", ["invalid", "newly_selected"])
def test_each_lease_checks_declared_but_excluded_manifests(tmp_path, phase, change):
    from tests.workflows.test_tardis_frozen_calibration import _publish

    root = tmp_path / "raw"
    root.mkdir()
    paths = tuple(
        _publish(root, date, fine=False) for date in ("2026-08-01", "2026-09-01")
    )
    window = TimeWindow("2026-08-01", "2026-08-01")
    excluded = root / paths[1] / "manifest.json"
    replacement = (
        b"{}"
        if change == "invalid"
        else (root / paths[0] / "manifest.json").read_bytes()
    )
    with _request():
        with open_market_source(
            root, archive_paths=paths, time_window=window
        ) as source:
            assert len(source.archives) == 1
        if phase == "entry":
            excluded.write_bytes(replacement)
        with pytest.raises(ValueError):
            with open_market_source(root, archive_paths=paths, time_window=window):
                assert phase == "exit"
                excluded.write_bytes(replacement)
    assert replay._REQUEST_ARCHIVES.get() is None
