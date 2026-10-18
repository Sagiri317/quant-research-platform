"""Native settlement identity and vendor forecast availability are distinct."""

import json
import math

import pandas as pd
import pytest

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.sources.binance.funding import read_funding_observations
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.sources.tardis.funding import import_funding_history
from quant_research_platform.sources.tardis.raw import import_slices
from quant_research_platform.sources.tardis.replay import open_tardis_archive
from tests.helpers.tardis import _line, _slices


@pytest.mark.parametrize("source_id", ["binance", "xoomar"])
@pytest.mark.parametrize("slice_count", [1, 144, 289])
def test_forecast_does_not_use_future_settlement_and_projection_is_verified(
    tmp_path, slice_count, source_id
):
    start_ms = pd.Timestamp("2026-09-01T00:00:00Z").value // 1000000
    native = _line(
        1,
        "markPrice@1s",
        {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "E": start_ms + 900,
            "p": "100",
            "r": "0.05",
            "T": start_ms + 10000,
        },
    )
    slices = _slices(tmp_path / "native", native, count=slice_count)
    response = tmp_path / "official.json"
    original = json.dumps(
        [
            {
                "symbol": "BTCUSDT",
                "fundingTime": start_ms + 10000,
                "fundingRate": "0.1",
                "markPrice": "200",
            }
        ]
    ).encode()
    if source_id == "xoomar":
        rows = [
            {
                "exchange": "binance",
                "baseAsset": "BTC",
                "slug": "btc",
                "fundingTime": pd.Timestamp(stamp, unit="ms", tz="UTC").isoformat(),
                "fundingRate": "0.1",
                "markPrice": "200",
            }
            for stamp in (
                start_ms + slice_count * 600000,
                start_ms + 10000,
                start_ms - 1,
            )
        ]
        original = json.dumps({"data": {"current": [], "history": rows}}).encode()
    response.write_bytes(original)
    import_slices(tmp_path / "published", date="2026-09-01", slices=slices)
    with open_tardis_archive(tmp_path / "published") as day:
        import_funding_history(
            day,
            symbol="BTCUSDT",
            source_id=source_id,
            request_limit=1000 if source_id == "binance" else 5000,
            response_path=response,
            http_status=200,
            start_ms=start_ms,
            end_ms=start_ms + slice_count * 600000 - 1,
        )
        full_window = TimeWindow(
            pd.Timestamp(start_ms, unit="ms"),
            pd.Timestamp(start_ms + slice_count * 600000 - 1, unit="ms"),
        )
        roots = day.funding_roots("BTCUSDT", time_window=full_window)
        assert len(roots) == 1
        funding_root = roots[0]
        with read_funding_observations(roots, symbol="BTCUSDT") as source:
            assert source.events[0].rate == 0.1
            assert source.events[0].mark_price == 200.0
            forecasts = tuple(source.forecasts)
            assert len(forecasts) == 1
            assert forecasts[0].available_at_ns == (start_ms + 1000) * 1000000
            assert forecasts[0].cashflows == (((start_ms + 10000) * 1000000, -5.0),)
        outside = pd.Timestamp(start_ms + slice_count * 600000, unit="ms")
        outside_window = TimeWindow(outside, outside)
        assert day.funding_roots("BTCUSDT", time_window=outside_window) == ()
        if slice_count > 144:
            later_window = TimeWindow("2026-09-02T00:00:00", "2026-09-02T00:01:00")
            assert day.funding_roots("BTCUSDT", time_window=later_window) == roots
            with open_tardis_archive(
                tmp_path / "published", time_window=later_window
            ) as later:
                assert later.funding_roots("BTCUSDT", time_window=later_window) == roots
                with read_funding_observations(roots, symbol="BTCUSDT") as source:
                    assert source.coverage == (
                        (
                            start_ms * 1000000,
                            (start_ms + slice_count * 600000) * 1000000 - 1,
                        ),
                    )
                    assert source.events[0].rate == 0.1
        with pytest.raises(PortfolioTargetBlocked, match="tardis_archive_window_empty"):
            with open_tardis_archive(
                tmp_path / "published", time_window=outside_window
            ):
                pytest.fail("half-open archive end cannot cover the next instant")
        metadata = json.loads(
            (funding_root / "response.metadata.json").read_text(encoding="utf-8")
        )["metadata"]["settlement_source"]
        assert metadata["source_id"] == source_id
        assert ("xoomar.com" in metadata["endpoint"]) == (source_id == "xoomar")
        preserved = funding_root.parent / "settlements.json"
        assert preserved.read_bytes() == original == response.read_bytes()
        # Byte count and hash describe the imported bytes, not read admission.
        preserved.write_text(json.dumps(json.loads(original), indent=2) + "\n")
        with read_funding_observations(roots, symbol="BTCUSDT") as source:
            assert source.events[0].rate == 0.1
            assert source.events[0].mark_price == 200.0
            assert len(tuple(source.forecasts)) == 1
        preserved.write_bytes(original.replace(b"200", b"201"))
        with pytest.raises(
            ValueError, match="binance_funding_settlement_projection_invalid"
        ):
            with read_funding_observations(roots, symbol="BTCUSDT"):
                pass


def _publish_funding_archive(root, date, *, include_funding=True):
    start_ms = pd.Timestamp(date).value // 1000000
    native = _line(
        1,
        "markPrice@1s",
        {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "E": start_ms + 900,
            "p": "100",
            "r": "0.05",
            "T": start_ms + 10000,
        },
    ).replace(b"2026-09-01T", (date + "T").encode())
    slices = _slices(root / ("native-" + date), native, count=1)
    destination = root / ("archive-" + date)
    import_slices(destination, date=date, slices=slices)
    if include_funding:
        response = root / (date + "-settlements.json")
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
            ),
            encoding="utf-8",
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
    return destination


@pytest.mark.parametrize("changed_field", ("time", "rate", "mark"))
def test_native_settlements_are_authoritative_after_coordinated_csv_rewrite(
    tmp_path, changed_field
):
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.raw_snapshots import (
        iter_csv_snapshot,
        write_csv_snapshot,
    )

    destination = _publish_funding_archive(tmp_path, "2026-09-01")
    window = TimeWindow("2026-09-01T00:00:00", "2026-09-01T00:09:59")
    with open_tardis_archive(destination) as archive:
        (original_root,) = archive.funding_roots("BTCUSDT", time_window=window)
        original_manifest = read_manifest(original_root / "response.metadata.json")
        original_bytes = (original_root.parent / "settlements.json").read_bytes()
        rows = list(
            iter_csv_snapshot(original_root, expected_manifest=original_manifest)
        )
        changed = []
        for row in rows:
            if row[0] == "settlement":
                row = list(row)
                index = {"time": 1, "rate": 3, "mark": 4}[changed_field]
                row[index] = (
                    row[index] + 1
                    if changed_field == "time"
                    else 0.9
                    if changed_field == "rate"
                    else 201.0
                )
                row = tuple(row)
            changed.append(row)
        rewritten = tmp_path / "coordinated" / "observations"
        rewritten.parent.mkdir()
        (rewritten.parent / "settlements.json").write_bytes(original_bytes)

        def validate_csv(root):
            manifest = read_manifest(root / "response.metadata.json")
            assert tuple(iter_csv_snapshot(root, expected_manifest=manifest)) == tuple(
                changed
            )

        write_csv_snapshot(
            rewritten,
            columns=original_manifest["columns"],
            rows=changed,
            metadata=original_manifest["metadata"],
            validate=validate_csv,
        )
        rewritten_manifest = read_manifest(rewritten / "response.metadata.json")
        assert canonical_json_hash(rewritten_manifest) != canonical_json_hash(
            original_manifest
        )
        with pytest.raises(
            ValueError, match="^binance_funding_settlement_projection_invalid$"
        ):
            with read_funding_observations((rewritten,), symbol="BTCUSDT"):
                pytest.fail(
                    "a new manifest cannot change the original settlement facts"
                )
        with read_funding_observations((original_root,), symbol="BTCUSDT") as source:
            assert [
                (event.timestamp_ns, event.rate, event.mark_price)
                for event in source.events
            ] == [(pd.Timestamp("2026-09-01T00:00:10Z").value, 0.1, 200.0)]
            assert len(tuple(source.forecasts)) == 1
        assert (
            original_root.parent / "settlements.json"
        ).read_bytes() == original_bytes


def test_selected_validation_funding_never_reads_other_months(tmp_path, monkeypatch):
    import quant_research_platform.sources.binance.funding as funding

    dates = ("2026-07-01", "2026-08-01", "2026-09-01")
    validation = TimeWindow("2026-08-01T00:00:00", "2026-08-01T00:09:59")
    archives = tuple(_publish_funding_archive(tmp_path, date) for date in dates)
    archive_paths = tuple(path.name for path in archives)
    inspected = []
    original_manifest = funding._manifest

    def observed_manifest(root, symbol):
        inspected.append(root)
        return original_manifest(root, symbol)

    monkeypatch.setattr(funding, "_manifest", observed_manifest)
    with open_market_source(
        tmp_path, archive_paths=archive_paths, time_window=validation
    ) as source:
        assert [item["date"] for item in source.source_identity["archives"]] == [
            "2026-08-01"
        ]
        roots = source.funding_roots("BTCUSDT", time_window=validation)
        assert len(roots) == 1 and roots[0].is_relative_to(archives[1])
        manifest = json.loads((roots[0] / "response.metadata.json").read_text())
        with read_funding_observations(roots, symbol="BTCUSDT") as observations:
            assert observations.identity == canonical_json_hash(
                {"sources": [canonical_json_hash(manifest)]}
            )
            forecasts = tuple(observations.forecasts)
            assert [item.available_at_ns for item in forecasts] == [
                pd.Timestamp("2026-08-01T00:00:01").value
            ]
        no_roots = source.funding_roots(
            "BTCUSDT",
            time_window=TimeWindow("2026-09-01T00:00:00", "2026-09-01T00:01:00"),
        )
        assert no_roots == ()
        with pytest.raises(
            PortfolioTargetBlocked, match="binance_funding_input_missing"
        ):
            with read_funding_observations(no_roots, symbol="BTCUSDT"):
                pytest.fail("selected source cannot discover another month")
    assert inspected and set(inspected) == set(roots)


@pytest.mark.parametrize("mode", ("complete", "early", "primary"))
def test_native_funding_scope_decodes_once_and_preserves_primary(
    tmp_path, monkeypatch, mode
):
    import quant_research_platform.sources.binance.funding as funding

    destination = _publish_funding_archive(tmp_path, "2026-09-01")
    root = destination / "funding/BTCUSDT/observations"
    original = funding._settlement_response
    decoded = []
    opened = []
    reads = []
    original_open = funding.open_filesystem_file
    import quant_research_platform.storage.raw_snapshots as snapshots

    csv_open = snapshots.open_filesystem_file
    csv_paths = []
    csv_handles = []

    def measured_csv_open(path, *args, **kwargs):
        csv_paths.append(path)
        handle = csv_open(path, *args, **kwargs)
        csv_handles.append(handle)
        return handle

    def forbidden_byte_encoding(*args, **kwargs):
        pytest.fail("ordinary funding reads must not re-encode canonical bytes")

    monkeypatch.setattr(snapshots, "open_filesystem_file", measured_csv_open)
    monkeypatch.setattr(snapshots, "_record", forbidden_byte_encoding)
    monkeypatch.setattr(snapshots, "encode_raw_cell", forbidden_byte_encoding)
    primary = MemoryError("consumer primary")

    class MeasuredHandle:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            reads.append(size)
            return self.handle.read(size)

        def close(self):
            self.handle.close()

    def measured_open(path, *args, **kwargs):
        opened.append(path)
        return MeasuredHandle(original_open(path, *args, **kwargs))

    monkeypatch.setattr(funding, "open_filesystem_file", measured_open)

    def decode(*args, **kwargs):
        result = original(*args, **kwargs)
        decoded.append(result)
        return result

    monkeypatch.setattr(funding, "_settlement_response", decode)

    def consume():
        with read_funding_observations((root,), symbol="BTCUSDT") as source:
            assert len(source.events) == 1
            assert (source.events[0].rate, source.events[0].mark_price) == (0.1, 200.0)
            if mode == "complete":
                assert len(tuple(source.forecasts)) == 1
            else:
                assert next(source.forecasts).cashflows == (
                    (pd.Timestamp("2026-09-01T00:00:10Z").value, -5.0),
                )
            if mode == "primary":
                path = root.parent / "settlements.json"
                path.write_bytes(path.read_bytes() + b" ")
            if mode == "primary":
                raise primary

    if mode == "primary":
        with pytest.raises(MemoryError) as raised:
            consume()
        assert raised.value is primary
        assert not getattr(primary, "__notes__", [])
    else:
        consume()
    assert len(decoded) == 1
    assert opened == [root.parent / "settlements.json"]
    assert reads == [-1]
    assert csv_paths == [root / "response.csv"]
    assert all(handle.closed for handle in csv_handles)


def test_separate_native_funding_archives_cannot_supply_continuous_months(tmp_path):
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _qualified_episodes,
    )

    dates = ("2026-07-01", "2026-08-01", "2026-09-01")
    starts = tuple(pd.Timestamp(date).value for date in dates)
    intervals = tuple((start, start + 600000000000 - 1) for start in starts)
    window = TimeWindow(pd.Timestamp(intervals[0][0]), pd.Timestamp(intervals[-1][1]))
    bounds = ((intervals[0][0], intervals[-1][1]),)
    archives = tuple(_publish_funding_archive(tmp_path, date) for date in dates)
    with open_market_source(
        tmp_path,
        archive_paths=tuple(path.name for path in archives),
        time_window=window,
    ) as source:
        roots = source.funding_roots("BTCUSDT", time_window=window)
        assert len(roots) == 3
        with read_funding_observations(roots, symbol="BTCUSDT") as observations:
            assert observations.coverage == intervals
            market_span = (bounds[0][0], bounds[0][1] + 1)
            episodes = list(
                _qualified_episodes(
                    {"book": market_span},
                    SimpleNamespace(intervals=lambda: iter((market_span,))),
                    observations,
                    bounds,
                )
            )
            assert [(item[0], item[1]) for item in episodes] == [
                (intervals[0][0], intervals[0][1] + 1),
                (intervals[0][1] + 1, intervals[1][0]),
                (intervals[1][0], intervals[1][1] + 1),
                (intervals[1][1] + 1, intervals[2][0]),
                (intervals[2][0], intervals[2][1] + 1),
            ]
            assert [item[2] for item in episodes] == ["TRUNCATED"] * 4 + ["PLANNED_END"]


def test_selected_archive_or_funding_missing_does_not_fall_back(tmp_path):
    window = TimeWindow("2026-08-01T00:00:00", "2026-08-01T00:09:59")
    july = _publish_funding_archive(tmp_path, "2026-07-01")
    august = _publish_funding_archive(tmp_path, "2026-08-01", include_funding=False)
    with open_market_source(
        tmp_path, archive_paths=(july.name, august.name), time_window=window
    ) as source:
        roots = source.funding_roots("BTCUSDT", time_window=window)
        assert len(roots) == 1 and roots[0].is_relative_to(august)
        with pytest.raises(
            PortfolioTargetBlocked, match="binance_funding_input_missing"
        ):
            with read_funding_observations(roots, symbol="BTCUSDT"):
                pytest.fail("July funding cannot substitute for missing August funding")
    with pytest.raises(FileNotFoundError):
        with open_market_source(
            tmp_path,
            archive_paths=(july.name, "missing-august-archive"),
            time_window=window,
        ):
            pytest.fail("missing declared archive cannot fall back to another source")


@pytest.mark.parametrize(
    "rate,expected",
    [("+0.0001", -0.01), ("-0.0001", 0.01), ("0", 0.0), (True, None), (False, None)],
)
def test_native_forecast_type_is_checked_before_numeric_conversion(
    tmp_path, rate, expected
):
    start = pd.Timestamp("2026-09-01").value // 1000000
    native = _line(
        1,
        "markPrice@1s",
        {
            "e": "markPriceUpdate",
            "s": "BTCUSDT",
            "E": start + 900,
            "p": "100",
            "r": rate,
            "T": start + 10000,
        },
    )
    response = tmp_path / "settlement.json"
    response.write_bytes(b"[]")
    destination = tmp_path / "published"
    import_slices(
        destination,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native, count=1),
    )
    original = {p.name: p.read_bytes() for p in destination.glob("*.gz")}
    with open_tardis_archive(destination) as archive:
        kwargs = dict(
            symbol="BTCUSDT",
            source_id="binance",
            response_path=response,
            http_status=200,
            start_ms=start,
            end_ms=start + 600000 - 1,
            request_limit=1000,
        )
        import_funding_history(archive, **kwargs)
        with read_funding_observations(
            (destination / "funding/BTCUSDT/observations",), symbol="BTCUSDT"
        ) as observed:
            forecasts = tuple(observed.forecasts)
            if expected is None:
                assert forecasts[0].cashflows[0][1] is None
            else:
                assert forecasts[0].cashflows[0][1] == pytest.approx(expected)
                if rate == "0":
                    assert math.copysign(1.0, forecasts[0].cashflows[0][1]) == -1.0
    assert {p.name: p.read_bytes() for p in destination.glob("*.gz")} == original


def test_effective_limit_cannot_certify_a_truncated_settlement_window(tmp_path):
    import hashlib

    start = pd.Timestamp("2026-09-01").value // 1000000
    response = tmp_path / "settlement.json"
    rows = [
        {
            "symbol": "BTCUSDT",
            "fundingTime": start + offset,
            "fundingRate": "0.01",
            "markPrice": "100",
        }
        for offset in (10000, 20000)
    ]
    response.write_text(json.dumps(rows))
    destination = _publish_funding_archive(
        tmp_path, "2026-09-01", include_funding=False
    )
    with open_tardis_archive(destination) as archive:
        kwargs = dict(
            symbol="BTCUSDT",
            source_id="binance",
            response_path=response,
            http_status=200,
            start_ms=start,
            end_ms=start + 600000 - 1,
        )
        import_funding_history(archive, **kwargs, request_limit=1000)
        root = destination / "funding/BTCUSDT/observations"
        with read_funding_observations((root,), symbol="BTCUSDT") as source:
            assert len(source.events) == 2
            assert sum(-event.rate * event.mark_price for event in source.events) == -2
        old_good = {
            p.relative_to(root.parent): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.parent.rglob("*")
            if p.is_file()
        }
        response.write_text(json.dumps(rows[:1]))
        with pytest.raises(
            PortfolioTargetBlocked, match="binance_funding_coverage_unproven"
        ):
            import_funding_history(archive, **kwargs, request_limit=1)
        assert old_good == {
            p.relative_to(root.parent): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.parent.rglob("*")
            if p.is_file()
        }
        metadata_path = root / "response.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["metadata"]["settlement_source"]["request_limit"] = 1
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(
            PortfolioTargetBlocked, match="binance_funding_coverage_unproven"
        ):
            with read_funding_observations((root,), symbol="BTCUSDT"):
                pass


@pytest.mark.parametrize("body_fails", [False, True])
def test_settlement_read_handle_close_preserves_primary(
    tmp_path, monkeypatch, body_fails
):
    import quant_research_platform.sources.binance.funding as funding

    primary = MemoryError("synthetic_hash_read")
    secondary = OSError("synthetic_hash_close")
    destination = _publish_funding_archive(tmp_path, "2026-09-01")
    original = funding.open_filesystem_file
    handles = []

    class Handle:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size=-1):
            if body_fails:
                raise primary
            return self.handle.read(size)

        def close(self):
            self.handle.close()
            raise secondary

    def open_handle(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if path.name == "settlements.json":
            handles.append(handle)
            return Handle(handle)
        return handle

    monkeypatch.setattr(funding, "open_filesystem_file", open_handle)
    with pytest.raises(MemoryError if body_fails else OSError) as caught:
        with read_funding_observations(
            (destination / "funding/BTCUSDT/observations",), symbol="BTCUSDT"
        ):
            pass
    assert caught.value is (primary if body_fails else secondary)
    assert handles and all(handle.closed for handle in handles)


@pytest.mark.parametrize("body_fails", (False, True))
def test_import_settlement_handle_close_preserves_primary(
    tmp_path, monkeypatch, body_fails
):
    import quant_research_platform.sources.tardis.funding as funding

    destination = _publish_funding_archive(
        tmp_path, "2026-09-01", include_funding=False
    )
    response = tmp_path / "settlements.json"
    response.write_bytes(b"[]")
    primary = MemoryError("synthetic_import_read")
    secondary = OSError("synthetic_import_close")
    original_open = funding.open_filesystem_file
    handles = []

    class Handle:
        def __init__(self, handle):
            self.handle = handle

        def read(self, count=-1):
            if body_fails:
                raise primary
            return self.handle.read(count)

        def close(self):
            self.handle.close()
            raise secondary

    def open_handle(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        handles.append(handle)
        return Handle(handle)

    monkeypatch.setattr(funding, "open_filesystem_file", open_handle)
    start = pd.Timestamp("2026-09-01").value // 1000000
    with open_tardis_archive(destination) as archive:
        with pytest.raises(MemoryError if body_fails else OSError) as caught:
            funding.import_funding_history(
                archive,
                symbol="BTCUSDT",
                source_id="binance",
                request_limit=1000,
                response_path=response,
                http_status=200,
                start_ms=start,
                end_ms=start + 600000 - 1,
            )
    assert caught.value is (primary if body_fails else secondary)
    if body_fails:
        assert primary.__notes__ == ["iterator close also failed: OSError"]
    assert handles and all(handle.closed for handle in handles)
    assert not (destination / "funding/BTCUSDT").exists()


@pytest.mark.parametrize(
    "expired,conflict", ((False, False), (True, False), (False, True), (True, True))
)
def test_equivalent_utc_funding_retransmission_is_idempotent(
    tmp_path, expired, conflict
):
    start = pd.Timestamp("2026-09-01T00:00:00Z").value // 1_000_000
    data = {
        "e": "markPriceUpdate",
        "s": "BTCUSDT",
        "E": start + 900,
        "p": "100",
        "r": "0.05",
        "T": start + (1000 if expired else 10000),
    }
    native = _line(1, "markPrice@1s", data) + _line(
        1, "markPrice@1s", {**data, "r": "0.06" if conflict else "0.05"}
    )
    slices = _slices(tmp_path / "native", native, count=1)
    response = tmp_path / "official.json"
    response.write_text(
        json.dumps(
            [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": start + 10000,
                    "fundingRate": "0.1",
                    "markPrice": "200",
                }
            ]
        )
    )
    published = tmp_path / "archive"
    import_slices(published, date="2026-09-01", slices=slices)
    with open_tardis_archive(published) as day:

        def publish():
            import_funding_history(
                day,
                symbol="BTCUSDT",
                source_id="binance",
                request_limit=1000,
                response_path=response,
                http_status=200,
                start_ms=start,
                end_ms=start + 599999,
            )

        publish()
        roots = day.funding_roots(
            "BTCUSDT",
            time_window=TimeWindow(
                pd.Timestamp(start, unit="ms"), pd.Timestamp(start + 599999, unit="ms")
            ),
        )
        with read_funding_observations(roots, symbol="BTCUSDT") as source:
            forecasts = tuple(source.forecasts)
            assert len(forecasts) == 1
            if expired:
                assert forecasts[0].known_until_ns == forecasts[0].available_at_ns
                assert forecasts[0].cashflows == ()
            else:
                assert forecasts[0].cashflows == (
                    ((start + 10000) * 1_000_000, None if conflict else -5.0),
                )
            assert source.events[0].rate == 0.1
            assert source.events[0].mark_price == 200.0
            assert (
                source.settlement_qualification[0]["verification"]
                == "official_response_validated"
            )


@pytest.mark.parametrize("representation", ["crlf", "json_spacing", "equal_number"])
def test_funding_csv_read_accepts_equivalent_bytes_but_audit_remains_strict(
    tmp_path, representation
):
    import csv
    import io

    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.raw_snapshots import iter_csv_snapshot

    destination = _publish_funding_archive(tmp_path, "2026-09-01")
    root = destination / "funding/BTCUSDT/observations"
    path = root / "response.csv"
    original = path.read_bytes()
    with read_funding_observations((root,), symbol="BTCUSDT") as source:
        expected = (source.events, tuple(source.forecasts))
    if representation == "crlf":
        replacement = original.replace(b"\n", b"\r\n")
    else:
        rows = list(csv.reader(io.StringIO(original.decode())))
        if representation == "json_spacing":
            rows[-1][-1] = json.dumps(json.loads(rows[-1][-1]), separators=(", ", ": "))
        else:
            assert rows[1][4] == "200.0"
            rows[1][4] = "200"
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        replacement = buffer.getvalue().encode()
    assert replacement != original
    path.write_bytes(replacement)
    with read_funding_observations((root,), symbol="BTCUSDT") as source:
        assert (source.events, tuple(source.forecasts)) == expected
    with pytest.raises(
        ValueError, match="raw_csv_(record|cell)_not_canonical|raw_csv_content_invalid"
    ):
        list(
            iter_csv_snapshot(
                root, expected_manifest=read_manifest(root / "response.metadata.json")
            )
        )


@pytest.mark.parametrize("change", ["rate", "omit_forecast"])
def test_funding_csv_read_keeps_projection_and_row_count_checks(tmp_path, change):
    destination = _publish_funding_archive(tmp_path, "2026-09-01")
    root = destination / "funding/BTCUSDT/observations"
    path = root / "response.csv"
    original = path.read_bytes()
    if change == "rate":
        replacement = original.replace(b",0.1,200.0,", b",0.2,200.0,")
        reason = "binance_funding_settlement_projection_invalid"
    else:
        replacement = b"".join(original.splitlines(keepends=True)[:-1])
        reason = "raw_csv_content_invalid"
    assert replacement != original
    path.write_bytes(replacement)
    with pytest.raises(ValueError, match=reason):
        with read_funding_observations((root,), symbol="BTCUSDT"):
            pytest.fail("changed settlement or omitted forecast cannot be accepted")


def test_funding_import_still_checks_projected_csv_bytes(tmp_path, monkeypatch):
    import quant_research_platform.sources.tardis.funding as importer

    original_write = importer.write_csv_snapshot

    def write_with_reformatted_projection(root, *, validate, **kwargs):
        def changed_validate(staging):
            path = staging / "response.csv"
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            validate(staging)

        return original_write(root, validate=changed_validate, **kwargs)

    monkeypatch.setattr(
        importer, "write_csv_snapshot", write_with_reformatted_projection
    )
    with pytest.raises(ValueError, match="raw_csv_record_not_canonical"):
        _publish_funding_archive(tmp_path, "2026-09-01")
    assert not (tmp_path / "archive-2026-09-01/funding/BTCUSDT").exists()


def test_import_touch_uses_consumed_settlement_bytes(tmp_path, monkeypatch):
    import hashlib
    import os

    import quant_research_platform.sources.tardis.funding as funding

    original = funding._settlement_response
    response = tmp_path / "2026-09-01-settlements.json"

    def decode_then_touch(raw, **kwargs):
        result = original(raw, **kwargs)
        before = response.stat()
        os.utime(response, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(funding, "_settlement_response", decode_then_touch)
    destination = _publish_funding_archive(tmp_path, "2026-09-01")
    root = destination / "funding/BTCUSDT"
    consumed = (root / "settlements.json").read_bytes()
    evidence = json.loads((root / "observations/response.metadata.json").read_text())[
        "metadata"
    ]["settlement_source"]
    assert consumed == response.read_bytes()
    assert evidence["sha256"] == hashlib.sha256(consumed).hexdigest()
    assert evidence["byte_count"] == len(consumed)
