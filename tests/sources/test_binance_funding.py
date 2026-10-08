from copy import deepcopy

import pytest

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.sources.binance.funding import (
    FUNDING_COLUMNS,
    FUNDING_SCHEMA_ID,
    funding_source_identity,
    read_funding_observations,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.raw_snapshots import (
    iter_csv_snapshot,
    write_csv_snapshot,
)


def _write(raw, *, rows=(), metadata_changes=None):
    metadata = {
        "schema_id": FUNDING_SCHEMA_ID,
        "symbol": "BTCUSDT",
        "market_type": FUTURES_MARKET_TYPE,
        "valuation_currency": "USDT",
        "coverage": {"start_ns": 0, "end_ns": 100, "status": "COMPLETE"},
        "provenance": "explicit_synthetic_funding_calendar",
        "settlement_source": None,
    }
    metadata.update(metadata_changes or {})

    def validate(root):
        manifest = read_manifest(root / "response.metadata.json")
        assert manifest["metadata"] == metadata
        for _row in iter_csv_snapshot(
            root, expected_manifest=manifest, max_bytes=1024 * 1024
        ):
            pass

    write_csv_snapshot(
        raw.parent / "funding" / "BTCUSDT",
        columns=FUNDING_COLUMNS,
        rows=rows,
        metadata=metadata,
        max_bytes=1024 * 1024,
        validate=validate,
    )


def test_funding_missing_and_explicit_zero_have_distinct_evidence(tmp_path):
    raw = tmp_path / "raw"
    with pytest.raises(PortfolioTargetBlocked, match="binance_funding_input_missing"):
        funding_source_identity(
            (raw.parent / "funding" / "BTCUSDT",), "ledger", symbol="BTCUSDT"
        )
    _write(raw, rows=(("forecast", 0, 100, None, None, []),))
    with read_funding_observations(
        (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
    ) as source:
        assert source.events == () and source.coverage == ((0, 100),)
        forecasts = tuple(source.forecasts)
        assert len(forecasts) == 1 and forecasts[0].cashflows == ()
        original = source.identity
    assert (
        funding_source_identity(
            (raw.parent / "funding" / "BTCUSDT",), "ledger", symbol="BTCUSDT"
        )
        == original
    )
    with pytest.raises(PortfolioTargetBlocked, match="binance_funding_input_missing"):
        funding_source_identity(
            (raw.parent / "funding" / "ETHUSDT",), "ledger", symbol="ETHUSDT"
        )
    changed = tmp_path / "changed" / "raw"
    _write(changed, rows=(("forecast", 0, 100, None, None, [[50, -1.0]]),))
    assert (
        funding_source_identity(
            (changed.parent / "funding" / "BTCUSDT",), "ledger", symbol="BTCUSDT"
        )
        != original
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"coverage": {"start_ns": 0, "end_ns": 100, "status": "UNPROVEN"}},
        {"provenance": ""},
    ),
)
def test_funding_unproven_coverage_never_becomes_zero(tmp_path, changes):
    raw = tmp_path / "raw"
    _write(raw, metadata_changes=deepcopy(changes))
    with pytest.raises(
        PortfolioTargetBlocked, match="binance_funding_coverage_unproven"
    ):
        funding_source_identity(
            (raw.parent / "funding" / "BTCUSDT",), "ledger", symbol="BTCUSDT"
        )


@pytest.mark.parametrize(
    "rows",
    (
        (("settlement", 101, None, 0.001, 100.0, None),),
        (
            ("settlement", 50, None, 0.001, 100.0, None),
            ("settlement", 50, None, 0.001, 100.0, None),
        ),
        (("forecast", 0, 100, None, None, [[101, 1.0]]),),
    ),
)
def test_funding_malformed_calendar_rejected_with_resources_released(tmp_path, rows):
    raw = tmp_path / "raw"
    _write(raw, rows=rows)
    with pytest.raises(ValueError, match="binance_funding_input_invalid"):
        funding_source_identity(
            (raw.parent / "funding" / "BTCUSDT",), "ledger", symbol="BTCUSDT"
        )


def test_funding_changed_during_consumption_is_rejected_and_primary_preserved(tmp_path):
    raw = tmp_path / "raw"
    row = ("forecast", 0, 100, None, None, [[50, 0.0]])
    _write(raw, rows=(row,))
    changed = tmp_path / "changed" / "raw"
    _write(changed, rows=(("forecast", 0, 100, None, None, [[50, 1.0]]),))
    with pytest.raises(ValueError, match="binance_funding_identity_changed"):
        with read_funding_observations(
            (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
        ) as source:
            assert tuple(source.forecasts)
            for name in ("response.csv", "response.metadata.json"):
                (raw.parent / "funding" / "BTCUSDT" / name).write_bytes(
                    (changed.parent / "funding" / "BTCUSDT" / name).read_bytes()
                )
    primary = KeyboardInterrupt("primary")
    with pytest.raises(KeyboardInterrupt) as raised:
        with read_funding_observations(
            (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
        ):
            raise primary
    assert raised.value is primary


@pytest.mark.parametrize("failure", ("io", "permission", "memory", "interrupt"))
def test_funding_identity_preserves_fatal_file_failure(tmp_path, monkeypatch, failure):
    import errno

    import quant_research_platform.storage.identity as identity

    raw = tmp_path / "raw"
    _write(raw)
    root = raw.parent / "funding" / "BTCUSDT"
    primary = {
        "io": OSError(errno.EIO, "controlled_io_failure"),
        "permission": PermissionError(errno.EACCES, "controlled_access_failure"),
        "memory": MemoryError("controlled_memory_failure"),
        "interrupt": KeyboardInterrupt("controlled_interrupt"),
    }[failure]
    original = identity.filesystem_stat

    def fail(path, *args, **kwargs):
        if path == root / "response.metadata.json":
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(identity, "filesystem_stat", fail)
    with pytest.raises(type(primary)) as caught:
        funding_source_identity((root,), "ledger", symbol="BTCUSDT")
    assert caught.value is primary


def test_funding_sources_preserve_gaps_and_ordered_pit_forecasts(tmp_path):
    roots = []
    for number, start in enumerate((0, 200, 400)):
        raw = tmp_path / str(number) / "raw"
        _write(
            raw,
            rows=(
                ("settlement", start + 50, None, 0.1, 200.0, None),
                ("forecast", start, start + 100, None, None, [[start + 50, -5.0]]),
            ),
            metadata_changes={
                "coverage": {
                    "start_ns": start,
                    "end_ns": start + 100,
                    "status": "COMPLETE",
                }
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    roots = tuple(roots)
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        assert source.coverage == ((0, 100), (200, 300), (400, 500))
        assert [event.timestamp_ns for event in source.events] == [50, 250, 450]
        assert all(event.rate * event.mark_price == 20.0 for event in source.events)
        forecasts = tuple(source.forecasts)
        assert [item.available_at_ns for item in forecasts] == [0, 200, 400]
        assert [item.cashflows for item in forecasts] == [
            ((50, -5.0),),
            ((250, -5.0),),
            ((450, -5.0),),
        ]
        identity = source.identity
    assert funding_source_identity(roots, "ledger", symbol="BTCUSDT") == identity
    assert funding_source_identity((roots[1],), "ledger", symbol="BTCUSDT") != identity


@pytest.mark.parametrize("second_start", (0, 50, 100))
def test_funding_overlapping_closed_intervals_are_rejected(tmp_path, second_start):
    roots = []
    for number, start in enumerate((0, second_start)):
        raw = tmp_path / str(number) / "raw"
        _write(
            raw,
            metadata_changes={
                "coverage": {
                    "start_ns": start,
                    "end_ns": start + 100,
                    "status": "COMPLETE",
                }
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    with pytest.raises(ValueError, match="binance_funding_source_intervals_invalid"):
        with read_funding_observations(tuple(roots), symbol="BTCUSDT"):
            pytest.fail("overlap cannot be silently merged")


def test_funding_adjacent_sources_require_declared_order_and_all_roots(tmp_path):
    roots = []
    for number, start in enumerate((0, 101)):
        raw = tmp_path / str(number) / "raw"
        _write(
            raw,
            metadata_changes={
                "coverage": {
                    "start_ns": start,
                    "end_ns": start + 100,
                    "status": "COMPLETE",
                }
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    with read_funding_observations(tuple(roots), symbol="BTCUSDT") as source:
        assert source.coverage == ((0, 100), (101, 201))
    for selected in (tuple(reversed(roots)), (roots[0], roots[0])):
        with pytest.raises(
            ValueError, match="binance_funding_source_intervals_invalid"
        ):
            with read_funding_observations(selected, symbol="BTCUSDT"):
                pytest.fail("unordered or repeated sources")
    for selected in ((), (roots[0], tmp_path / "missing")):
        with pytest.raises(
            PortfolioTargetBlocked, match="binance_funding_input_missing"
        ):
            with read_funding_observations(selected, symbol="BTCUSDT"):
                pytest.fail("missing funding is not zero")


@pytest.mark.parametrize("interrupt", (False, True))
def test_funding_unconsumed_later_source_is_verified_at_exit(
    tmp_path, monkeypatch, interrupt
):
    import quant_research_platform.storage.raw_snapshots as snapshots

    handles = []
    original_open = snapshots.open_filesystem_file

    def tracked_open(*args, **kwargs):
        handle = original_open(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(snapshots, "open_filesystem_file", tracked_open)
    roots = []
    for number, start in enumerate((0, 200)):
        raw = tmp_path / str(number) / "raw"
        _write(
            raw,
            rows=(("forecast", start, start + 100, None, None, []),),
            metadata_changes={
                "coverage": {
                    "start_ns": start,
                    "end_ns": start + 100,
                    "status": "COMPLETE",
                }
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    primary = KeyboardInterrupt("controlled_primary")
    with pytest.raises(KeyboardInterrupt if interrupt else ValueError) as caught:
        with read_funding_observations(tuple(roots), symbol="BTCUSDT") as source:
            assert next(source.forecasts).available_at_ns == 0
            path = roots[1] / "response.csv"
            path.write_bytes(path.read_bytes().replace(b"200", b"201"))
            if interrupt:
                raise primary
    if interrupt:
        assert caught.value is primary
        assert primary.__notes__ == [
            "Funding source verification also failed: ValueError"
        ]
    else:
        assert str(caught.value) == "raw_csv_content_invalid"
    assert handles and all(handle.closed for handle in handles)


def test_funding_forecast_order_is_checked_across_sources(tmp_path):
    roots = []
    for number, start in enumerate((0, 200)):
        raw = tmp_path / str(number) / "raw"
        _write(
            raw,
            rows=(("forecast", 0, 500, None, None, []),),
            metadata_changes={
                "coverage": {
                    "start_ns": start,
                    "end_ns": start + 100,
                    "status": "COMPLETE",
                }
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    with pytest.raises(ValueError, match="binance_funding_record_order_invalid"):
        with read_funding_observations(tuple(roots), symbol="BTCUSDT"):
            pytest.fail("cross-source forecasts must remain strictly ordered")


def test_funding_tail_io_error_retains_original_exception(tmp_path, monkeypatch):
    import errno

    import quant_research_platform.storage.identity as identity

    raw = tmp_path / "raw"
    _write(raw, rows=(("forecast", 0, 100, None, None, []),))
    root = raw.parent / "funding" / "BTCUSDT"
    failure = OSError(errno.EIO, "controlled_tail_io_failure")
    original_stat = identity.filesystem_stat

    def failed_stat(path, *args, **kwargs):
        if path == root / "response.metadata.json":
            raise failure
        return original_stat(path, *args, **kwargs)

    with pytest.raises(OSError) as caught:
        with read_funding_observations((root,), symbol="BTCUSDT"):
            monkeypatch.setattr(identity, "filesystem_stat", failed_stat)
    assert caught.value is failure


def test_funding_numeric_strings_are_normalized_through_persisted_reader(tmp_path):
    _write(
        tmp_path / "raw",
        rows=(
            ("settlement", 50, None, "0.01", "100", None),
            ("forecast", 0, 100, None, None, [[50, "-1"]]),
        ),
    )
    with read_funding_observations(
        (tmp_path / "funding" / "BTCUSDT",), symbol="BTCUSDT"
    ) as source:
        event = source.events[0]
        forecast = next(source.forecasts)
        assert type(event.rate) is float and type(event.mark_price) is float
        assert -event.rate * event.mark_price == -1.0
        assert forecast.cashflows == ((50, -1.0),)
        assert type(forecast.cashflows[0][1]) is float


@pytest.mark.parametrize("bad", ([], {}), ids=("array", "object"))
def test_funding_coverage_status_types_reject_in_reader_and_identity(tmp_path, bad):
    good_raw = tmp_path / "good" / "raw"
    row = ("forecast", 0, 100, None, None, [])
    _write(good_raw, rows=(row,))
    good = good_raw.parent / "funding" / "BTCUSDT"
    before = {path.name: path.read_bytes() for path in good.iterdir()}
    with read_funding_observations((good,), symbol="BTCUSDT") as source:
        assert source.coverage == ((0, 100),)
        assert source.events == ()
        assert len(tuple(source.forecasts)) == 1
        identity = source.identity
    assert funding_source_identity((good,), "ledger", symbol="BTCUSDT") == identity

    changed_raw = tmp_path / "changed" / "raw"
    _write(
        changed_raw,
        rows=(row,),
        metadata_changes={"coverage": {"start_ns": 0, "end_ns": 100, "status": bad}},
    )
    changed = changed_raw.parent / "funding" / "BTCUSDT"
    changed_before = {path.name: path.read_bytes() for path in changed.iterdir()}
    with pytest.raises(ValueError, match="^binance_funding_coverage_invalid$"):
        with read_funding_observations((changed,), symbol="BTCUSDT"):
            pytest.fail("Malformed coverage must fail before observations are yielded")
    with pytest.raises(ValueError, match="^binance_funding_input_invalid$") as raised:
        funding_source_identity((changed,), "ledger", symbol="BTCUSDT")
    assert type(raised.value.__cause__) is ValueError
    assert str(raised.value.__cause__) == "binance_funding_coverage_invalid"
    assert {
        path.name: path.read_bytes() for path in changed.iterdir()
    } == changed_before
    assert {path.name: path.read_bytes() for path in good.iterdir()} == before
    assert funding_source_identity((good,), "ledger", symbol="BTCUSDT") == identity
