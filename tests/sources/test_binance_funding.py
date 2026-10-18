from copy import deepcopy

import pytest

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.sources.binance.funding import (
    FUNDING_COLUMNS,
    FUNDING_SCHEMA_ID,
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
        for _row in iter_csv_snapshot(root, expected_manifest=manifest):
            pass

    write_csv_snapshot(
        raw.parent / "funding" / "BTCUSDT",
        columns=FUNDING_COLUMNS,
        rows=rows,
        metadata=metadata,
        validate=validate,
    )


def test_funding_missing_and_explicit_zero_have_distinct_evidence(tmp_path):
    raw = tmp_path / "raw"
    with pytest.raises(PortfolioTargetBlocked, match="binance_funding_input_missing"):
        with read_funding_observations(
            (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
        ):
            pytest.fail("Invalid funding must fail before observations are yielded")
    _write(raw, rows=(("forecast", 0, 100, None, None, []),))
    with read_funding_observations(
        (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
    ) as source:
        assert source.events == () and source.coverage == ((0, 100),)
        forecasts = tuple(source.forecasts)
        assert len(forecasts) == 1 and forecasts[0].cashflows == ()
    with pytest.raises(PortfolioTargetBlocked, match="binance_funding_input_missing"):
        with read_funding_observations(
            (raw.parent / "funding" / "ETHUSDT",), symbol="ETHUSDT"
        ):
            pytest.fail("Invalid funding must fail before observations are yielded")
    changed = tmp_path / "changed" / "raw"
    _write(changed, rows=(("forecast", 0, 100, None, None, [[50, -1.0]]),))
    with read_funding_observations(
        (changed.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
    ) as source:
        assert tuple(source.forecasts)[0].cashflows == ((50, -1.0),)


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
        with read_funding_observations(
            (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
        ):
            pytest.fail("Invalid funding must fail before observations are yielded")


@pytest.mark.parametrize(
    ("rows", "error"),
    (
        (
            (("settlement", 101, None, 0.001, 100.0, None),),
            "binance_funding_settlement_invalid",
        ),
        (
            (
                ("settlement", 50, None, 0.001, 100.0, None),
                ("settlement", 50, None, 0.001, 100.0, None),
            ),
            "binance_funding_record_order_invalid",
        ),
        (
            (("forecast", 0, 100, None, None, [[101, 1.0]]),),
            "portfolio_funding_forecast_events_invalid",
        ),
    ),
)
def test_funding_malformed_calendar_rejected_with_resources_released(
    tmp_path, rows, error
):
    raw = tmp_path / "raw"
    _write(raw, rows=rows)
    with pytest.raises(ValueError, match=f"^{error}$"):
        with read_funding_observations(
            (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
        ):
            pytest.fail("Invalid funding must fail before observations are yielded")


def test_funding_consumer_primary_is_preserved(tmp_path):
    raw = tmp_path / "raw"
    row = ("forecast", 0, 100, None, None, [[50, 0.0]])
    _write(raw, rows=(row,))
    primary = KeyboardInterrupt("primary")
    with pytest.raises(KeyboardInterrupt) as raised:
        with read_funding_observations(
            (raw.parent / "funding" / "BTCUSDT",), symbol="BTCUSDT"
        ):
            raise primary
    assert raised.value is primary


@pytest.mark.parametrize("failure", ("io", "permission", "memory", "interrupt"))
def test_funding_reader_preserves_fatal_file_failure(tmp_path, monkeypatch, failure):
    import errno

    import quant_research_platform.sources.binance.funding as funding

    raw = tmp_path / "raw"
    _write(raw)
    root = raw.parent / "funding" / "BTCUSDT"
    primary = {
        "io": OSError(errno.EIO, "controlled_io_failure"),
        "permission": PermissionError(errno.EACCES, "controlled_access_failure"),
        "memory": MemoryError("controlled_memory_failure"),
        "interrupt": KeyboardInterrupt("controlled_interrupt"),
    }[failure]
    original = funding.read_manifest

    def fail(path, *args, **kwargs):
        if path == root / "response.metadata.json":
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(funding, "read_manifest", fail)
    with pytest.raises(type(primary)) as caught:
        with read_funding_observations((root,), symbol="BTCUSDT"):
            pytest.fail("Invalid funding must fail before observations are yielded")
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


def test_funding_identical_prior_forecasts_are_deduplicated_across_sources(tmp_path):
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
    with read_funding_observations(tuple(roots), symbol="BTCUSDT") as source:
        forecasts = tuple(source.forecasts)
        assert len(forecasts) == 1
        assert forecasts[0].available_at_ns == 0
        assert forecasts[0].known_until_ns == 500


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
def test_funding_coverage_status_types_reject_in_reader(tmp_path, bad):
    good_raw = tmp_path / "good" / "raw"
    row = ("forecast", 0, 100, None, None, [])
    _write(good_raw, rows=(row,))
    good = good_raw.parent / "funding" / "BTCUSDT"
    before = {path.name: path.read_bytes() for path in good.iterdir()}
    with read_funding_observations((good,), symbol="BTCUSDT") as source:
        assert source.coverage == ((0, 100),)
        assert source.events == ()
        assert len(tuple(source.forecasts)) == 1

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
    assert {
        path.name: path.read_bytes() for path in changed.iterdir()
    } == changed_before
    assert {path.name: path.read_bytes() for path in good.iterdir()} == before


def test_funding_forecasts_merge_prior_observations_without_reordering_sources(
    tmp_path,
):
    roots = []
    for index, stamps in enumerate(((0, 100), (50, 100, 250), (25, 300))):
        raw = tmp_path / str(index) / "raw"
        start = index * 200
        _write(
            raw,
            rows=(
                ("settlement", start + 50, None, 0.1, 100.0, None),
                *(
                    ("forecast", stamp, 600, None, None, [[550, -float(stamp)]])
                    for stamp in stamps
                ),
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
    with read_funding_observations(tuple(roots), symbol="BTCUSDT") as source:
        assert source.coverage == ((0, 100), (200, 300), (400, 500))
        assert [event.timestamp_ns for event in source.events] == [50, 250, 450]
        forecasts = tuple(source.forecasts)
        assert [item.available_at_ns for item in forecasts] == [
            0,
            25,
            50,
            100,
            250,
            300,
        ]
        assert forecasts[1].cashflows == ((550, -25.0),)


@pytest.mark.parametrize("conflict", ["horizon", "cashflow", "local_order"])
def test_funding_merged_forecasts_reject_conflicting_facts_and_local_order(
    tmp_path, conflict
):
    roots = []
    for index in range(2):
        raw = tmp_path / str(index) / "raw"
        rows = [
            (
                "forecast",
                100,
                600 if conflict != "horizon" or index == 0 else 601,
                None,
                None,
                [[550, float(index) if conflict == "cashflow" else 0.0]],
            )
        ]
        if conflict == "local_order" and index == 1:
            rows.append(("forecast", 50, 600, None, None, []))
        _write(
            raw,
            rows=rows,
            metadata_changes={
                "coverage": {
                    "start_ns": index * 200,
                    "end_ns": index * 200 + 100,
                    "status": "COMPLETE",
                }
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    with pytest.raises(
        ValueError,
        match="binance_funding_record_"
        + ("order_invalid" if conflict == "local_order" else "conflict"),
    ):
        with read_funding_observations(tuple(roots), symbol="BTCUSDT"):
            pytest.fail("conflicting or out-of-order observations cannot be accepted")


@pytest.mark.parametrize("mode", ("complete", "early", "untouched", "primary"))
def test_interleaved_declared_archive_decodes_once_and_closes(
    tmp_path, monkeypatch, mode
):
    import quant_research_platform.storage.raw_snapshots as snapshots
    from quant_research_platform.contracts.portfolios import (
        FundingEvent,
        FundingForecast,
    )

    _write(
        tmp_path / "raw",
        rows=(
            ("forecast", 0, 100, None, None, [[80, -1.0]]),
            ("settlement", 20, None, None, 100.0, None),
            ("forecast", 30, 100, None, None, [[80, None]]),
            ("settlement", 80, None, 0.01, 200.0, None),
            ("forecast", 60, 100, None, None, [[80, -2.0]]),
        ),
    )
    opened = []
    original_open = snapshots.open_filesystem_file

    def measured_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(snapshots, "open_filesystem_file", measured_open)
    primary = RuntimeError("consumer primary")
    streams = []

    def consume():
        with read_funding_observations(
            (tmp_path / "funding/BTCUSDT",), symbol="BTCUSDT"
        ) as source:
            streams.append(source.forecasts)
            assert source.events == (
                FundingEvent(20, None, 100.0),
                FundingEvent(80, 0.01, 200.0),
            )
            assert len(opened) == 1 and opened[0].closed
            if mode == "complete":
                assert tuple(source.forecasts) == (
                    FundingForecast(0, 100, ((80, -1.0),)),
                    FundingForecast(30, 100, ((80, None),)),
                    FundingForecast(60, 100, ((80, -2.0),)),
                )
            elif mode != "untouched":
                assert next(source.forecasts) == FundingForecast(0, 100, ((80, -1.0),))
            if mode == "primary":
                raise primary

    if mode == "primary":
        with pytest.raises(RuntimeError) as raised:
            consume()
        assert raised.value is primary
    else:
        consume()
    assert len(opened) == 1 and opened[0].closed
    assert list(streams[0]) == []
