from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
)
from quant_research_platform.markets.implementations.crypto.hft import (
    HFT_GRID_MS,
    iter_factor_frames,
    iter_label_frames,
    iter_panel_frames,
    l1_ofi,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from tests.helpers.binance_hft import (
    _coverage,
    _panel_rows,
    _row_timestamp_ms,
    coverage_lookup,
)


def test_panel_uses_exact_validated_receipt_nanoseconds() -> None:
    observed = pd.Timestamp("2023-01-01T00:00:00.000000123").value
    origin_ms = observed // 1000000
    state = replace(
        _state(origin_ms),
        observed_time_ns=observed,
        receive_time=pd.Timestamp(observed, unit="ns").isoformat() + "Z",
    )
    frame = pd.concat(
        tuple(
            iter_panel_frames(
                [state, _state(origin_ms + 300)],
                [],
                start_ms=origin_ms + 100,
                end_ms=origin_ms + 300,
                book_coverage={"s1": _coverage(origin_ms, origin_ms + 400)},
                trade_coverage=coverage_lookup(
                    {"trade-connection-1": _coverage(origin_ms, origin_ms + 400)}
                ),
                market_id="crypto",
                asset_id="btc_usdt",
                batch_rows=4,
            )
        ),
        ignore_index=True,
    )
    assert pd.Timestamp(frame.observed_utc.iloc[0]).value == observed


def _state(
    timestamp_ms: int,
    *,
    bid_index: int = 995,
    ask_index: int = 1005,
    bid_qty: float = 2.0,
    ask_qty: float = 1.0,
    segment_id: str = "s1",
) -> BookState:
    return BookState(
        source_sequence=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time=pd.Timestamp(timestamp_ms, unit="ms").isoformat() + "Z",
        receive_monotonic_ns=timestamp_ms + 1,
        update_id=timestamp_ms + 1,
        segment_id=segment_id,
        observation_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        bids=((bid_index, bid_qty), (bid_index - 10, 3.0)),
        asks=((ask_index, ask_qty), (ask_index + 10, 2.0)),
    )


def _boundary(timestamp_ms: int, *, segment_id: str = "s1") -> BookBoundary:
    return BookBoundary(
        source_sequence=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time=pd.Timestamp(timestamp_ms, unit="ms").isoformat() + "Z",
        receive_monotonic_ns=timestamp_ms + 1,
        segment_id=segment_id,
        observation_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        reason="sequence_gap",
    )


def _trade(timestamp_ms: int, quantity: float, *, maker_buyer: bool) -> TradeEvent:
    return TradeEvent(
        source_sequence=timestamp_ms + 10,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time=pd.Timestamp(timestamp_ms, unit="ms").isoformat() + "Z",
        receive_monotonic_ns=timestamp_ms + 10,
        observation_id="trade-connection-1",
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        price_index=1005,
        quantity=quantity,
        visible_quantity=quantity,
        buyer_is_maker=maker_buyer,
    )


def test_l1_ofi_uses_exact_integer_tick_identity() -> None:
    assert l1_ofi((1000, 2.0), (1010, 3.0), (1020, 1.0), (1020, 2.0)) == 2.0


def test_panel_frames_build_from_column_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pd.DataFrame,
        "from_records",
        lambda *_args, **_kwargs: pytest.fail("hft_panel_row_graph_used"),
    )
    frames = list(
        iter_panel_frames(
            [_state(0), _state(200)],
            [],
            market_id="crypto",
            asset_id="btc_usdt",
            start_ms=0,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 300)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 300)}),
            batch_rows=2,
        )
    )
    assert [len(frame) for frame in frames] == [2, 1]
    assert frames[0]["market_id"].tolist() == ["crypto", "crypto"]
    assert frames[0]["asset_id"].tolist() == ["btc_usdt", "btc_usdt"]


@pytest.mark.parametrize("market_type", (SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE))
def test_panel_transaction_clock_is_nullable_integer(market_type) -> None:
    import pyarrow as pa

    from quant_research_platform.storage.datasets.schema import (
        physical_schema_from_schema,
    )

    states = [
        replace(
            _state(stamp),
            market_type=market_type,
            transaction_time_ms=None if market_type == SPOT_MARKET_TYPE else stamp,
        )
        for stamp in (0, 200)
    ]
    frames = list(
        iter_panel_frames(
            states,
            [],
            market_id="crypto",
            asset_id="btc_usdt",
            start_ms=0,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 300)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 300)}),
            batch_rows=2,
        )
    )
    assert [len(frame) for frame in frames] == [2, 1]
    for frame in frames:
        assert str(frame["transaction_time_ms"].dtype) == "Int64"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        assert table.schema.field("transaction_time_ms").type == pa.int64()
        physical_schema_from_schema(table.schema)
    transaction = pd.concat(frames, ignore_index=True)["transaction_time_ms"]
    if market_type == SPOT_MARKET_TYPE:
        assert transaction.isna().all()
    else:
        assert transaction.tolist() == [0, 0, 200]


@pytest.mark.parametrize("close_early", (False, True))
def test_panel_releases_consumed_buffers_before_yield(close_early) -> None:
    frames = iter_panel_frames(
        [_state(0), _state(200)],
        [],
        market_id="crypto",
        asset_id="btc_usdt",
        start_ms=0,
        end_ms=200,
        book_coverage={"s1": _coverage(0, 300)},
        trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 300)}),
        batch_rows=2,
    )
    first = next(frames)
    expected = first.copy(deep=True)
    assert not frames.gi_frame.f_locals["buffers"]
    if close_early:
        frames.close()
    else:
        last = next(frames)
        assert not frames.gi_frame.f_locals["buffers"]
        assert last["timestamp"].tolist() == [pd.Timestamp(200, unit="ms")]
        with pytest.raises(StopIteration):
            next(frames)
    pd.testing.assert_frame_equal(first, expected)


def test_quiet_valid_book_persists_without_fabricated_gap() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _state(500)],
            [],
            start_ms=0,
            end_ms=500,
            book_coverage={"s1": _coverage(0, 501)},
            trade_coverage={"trade-connection-1": _coverage(0, 501)},
        )
    )
    assert [_row_timestamp_ms(row) for row in rows] == [0, 100, 200, 300, 400, 500]
    assert {row["segment_id"] for row in rows} == {"s1:trade-connection-1"}


def test_panel_excludes_pre_interval_trades_and_keeps_stream_clocks_separate() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _state(600)],
            [_trade(100, 9.0, maker_buyer=False), _trade(500, 2.0, maker_buyer=True)],
            start_ms=500,
            end_ms=500,
            book_coverage={"s1": _coverage(0, 601)},
            trade_coverage={"trade-connection-1": _coverage(0, 601)},
        )
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["trade_volume_100ms"] == 2.0
    assert row["observed_signed_trade_volume_100ms"] == -2.0
    assert row["book_observed_time_ms"] == 0
    assert row["exchange_event_time_ms"] == 1
    assert row["transaction_time_ms"] == 0
    assert row["book_source_sequence"] == 1


def test_panel_ofi_excludes_book_events_before_interval_lower_bound() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _state(50, bid_qty=3.0)],
            [],
            start_ms=200,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 301)},
            trade_coverage={"trade-connection-1": _coverage(0, 301)},
        )
    )
    assert len(rows) == 1
    assert rows[0]["book_ofi_100ms"] == 0.0


def test_explicit_book_boundary_stops_panel_until_new_segment() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _boundary(250), _state(400, segment_id="s2")],
            [],
            start_ms=0,
            end_ms=400,
            book_coverage={"s1": _coverage(0, 250), "s2": _coverage(400, 500)},
            trade_coverage={"trade-connection-1": _coverage(0, 500)},
        )
    )
    assert [(_row_timestamp_ms(row), row["segment_id"]) for row in rows] == [
        (0, "s1:trade-connection-1"),
        (100, "s1:trade-connection-1"),
        (200, "s1:trade-connection-1"),
    ]


@pytest.mark.parametrize("phase", (0, 37))
@pytest.mark.parametrize("gap_ms", (1000, 31 * 86400 * 1000))
@pytest.mark.parametrize("boundary", (False, True))
def test_panel_uncovered_gap_keeps_events_ofi_labels_and_grid_phase(
    phase, gap_ms, boundary
):
    resume = phase + gap_ms
    books = [_state(phase), _state(phase + 50, bid_qty=3.0)]
    if boundary:
        books.append(_boundary(phase + 250))
    books.extend(
        (
            _state(resume - 50, segment_id="s2", bid_qty=4.0),
            _state(resume + 25, segment_id="s2", bid_qty=6.0),
            _state(resume + 150, segment_id="s2", bid_qty=7.0),
        )
    )
    trades = [
        _trade(phase + 275, 11.0, maker_buyer=False),
        _trade(resume - 75, 9.0, maker_buyer=False),
        _trade(resume, 3.0, maker_buyer=False),
        _trade(resume + 50, 2.0, maker_buyer=True),
        _trade(resume + 200, 4.0, maker_buyer=False),
        _trade(resume + 250, -1.0, maker_buyer=False),
    ]
    lookup = coverage_lookup({"trade-connection-1": _coverage(0, resume + 1000)})
    queries = []

    def covering(start_ns, end_ns):
        queries.append((start_ns, end_ns))
        assert len(queries) <= 32, "uncovered wall-clock gap was scanned grid by grid"
        return lookup(start_ns, end_ns)

    panel = pd.concat(
        iter_panel_frames(
            books,
            trades,
            start_ms=phase,
            end_ms=resume + 250,
            book_coverage={
                "s1": _coverage(0, phase + 250),
                "s2": _coverage(resume - 50, resume + 201),
            },
            trade_coverage=covering,
            market_id="crypto",
            asset_id="btc_usdt",
            batch_rows=2,
        ),
        ignore_index=True,
    )
    assert panel.timestamp.tolist() == [
        pd.Timestamp(stamp, unit="ms")
        for stamp in (phase, phase + 100, phase + 200, resume + 100, resume + 200)
    ]
    assert panel.book_ofi_100ms.tolist() == [0.0, 1.0, 0.0, 2.0, 1.0]
    assert panel.trade_volume_100ms.tolist() == [0.0, 0.0, 0.0, 2.0, 4.0]
    assert panel.observed_signed_trade_volume_100ms.tolist() == [
        0.0,
        0.0,
        0.0,
        -2.0,
        4.0,
    ]
    assert panel.book_segment_id.tolist() == ["s1"] * 3 + ["s2"] * 2
    assert str(panel.transaction_time_ms.dtype) == "Int64"
    assert queries == sorted(queries)
    assert queries[-1][1] == (resume + 200) * 1000000
    labels = pd.concat(
        iter_label_frames((panel,), horizons_ms=(100, 500, 1000, 5000)),
        ignore_index=True,
    )
    assert labels.future_mid_return_100ms.notna().tolist() == [
        True,
        True,
        False,
        True,
        False,
    ]
    assert labels.future_mid_return_100ms.dropna().eq(0.0).all()
    for horizon in (500, 1000, 5000):
        assert labels[f"future_mid_return_{horizon}ms"].isna().all()
    assert labels.maker_bid_sell_pressure_proxy_500ms.isna().all()


@pytest.mark.parametrize("coverage_end", (400, 401))
def test_panel_gap_wakes_when_first_book_interval_becomes_fully_covered(coverage_end):
    phase = 37
    resume = phase + 31 * 86400 * 1000
    queries = []

    def covering(start_ns, end_ns):
        queries.append((start_ns, end_ns))
        assert len(queries) <= 10, "uncovered leading gap was scanned grid by grid"
        return ("trade-connection-1",)

    frames = list(
        iter_panel_frames(
            [_state(resume + 50)],
            [],
            start_ms=phase,
            end_ms=resume + 450,
            book_coverage={"s1": _coverage(resume + 50, resume + coverage_end)},
            trade_coverage=covering,
            market_id="crypto",
            asset_id="btc_usdt",
            batch_rows=2,
        )
    )
    panel = pd.concat(frames, ignore_index=True)
    assert panel.timestamp.tolist() == [
        pd.Timestamp(resume + offset, unit="ms")
        for offset in range(200, min(coverage_end, 450), 100)
    ]
    assert panel.book_ofi_100ms.eq(0.0).all()
    assert panel.trade_count_100ms.eq(0).all()


@pytest.mark.parametrize(
    "failure,expected",
    (
        ("book", "hft_book_time_non_monotonic"),
        ("trade", "hft_trade_time_non_monotonic"),
        ("earlier_trade", "hft_trade_values_invalid"),
        ("same_grid", "hft_book_time_non_monotonic"),
        ("last_grid", "hft_trade_values_invalid"),
    ),
)
def test_panel_uncovered_gap_preserves_event_failures_and_their_order(
    failure, expected
):
    gap = 31 * 86400 * 1000
    books = [_state(0), _boundary(150), _state(gap), _state(gap - 1)]
    trades = []
    if failure == "trade":
        trades = [
            _trade(250, 1.0, maker_buyer=False),
            _trade(249, 1.0, maker_buyer=False),
        ]
    elif failure in {"earlier_trade", "same_grid"}:
        trades = [
            _trade(250 if failure == "earlier_trade" else gap, -1.0, maker_buyer=False)
        ]
    elif failure == "last_grid":
        books = []
        trades = [_trade(gap, -1.0, maker_buyer=False)]
    calls = 0

    def covering(_start_ns, _end_ns):
        nonlocal calls
        calls += 1
        assert calls <= 16, "uncovered gap delayed validation of a stream event"
        return ("trade-connection-1",)

    with pytest.raises(ValueError, match=expected):
        list(
            iter_panel_frames(
                books,
                trades,
                start_ms=0,
                end_ms=gap + 99,
                book_coverage={"s1": _coverage(0, 150)},
                trade_coverage=covering,
                market_id="crypto",
                asset_id="btc_usdt",
                batch_rows=2,
            )
        )


def test_factor_rows_are_hand_calculated_from_panel_sufficient_statistics() -> None:
    panel = list(
        _panel_rows(
            [
                _state(0),
                _state(100, bid_index=1000, ask_index=1010, bid_qty=3.0, ask_qty=1.0),
                _state(200, bid_index=1005, ask_index=1015, bid_qty=4.0, ask_qty=2.0),
            ],
            [_trade(100, 2.0, maker_buyer=False)],
            start_ms=0,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
        )
    )
    factors = pd.concat(
        iter_factor_frames((pd.DataFrame(panel),)), ignore_index=True
    ).to_dict("records")
    assert factors[1]["spread"] == 1.0
    assert math.isclose(factors[1]["imbalance_1"], 0.5)
    assert factors[1]["ofi_100ms"] == 4.0
    assert factors[1]["signed_trade_flow_100ms"] == 2.0
    assert math.isclose(factors[1]["return_100ms"], 0.5 / 100.0)
    assert math.isnan(factors[0]["return_100ms"])
    assert not {"best_bid_price", "trade_count_100ms"}.intersection(factors[0])


def test_labels_are_exact_horizon_same_segment_and_keep_zero_distinct_from_nan() -> (
    None
):
    rows = [
        {
            "timestamp_ms": index * HFT_GRID_MS,
            "timestamp": f"1970-01-01T00:00:00.{index}00",
            "segment_id": "s1" if index < 3 else "s2",
            "mid": 100.0 if index != 1 else 101.0,
            "market_id": "crypto",
            "asset_id": "btc_usdt",
        }
        for index in range(5)
    ]
    labels = pd.concat(
        iter_label_frames((pd.DataFrame(rows),), horizons_ms=(HFT_GRID_MS, 200)),
        ignore_index=True,
    ).to_dict("records")
    assert math.isclose(labels[0]["future_mid_return_100ms"], 0.01)
    assert labels[0]["future_mid_direction_100ms"] == 1.0
    assert labels[3]["future_mid_direction_100ms"] == 0.0
    assert math.isnan(labels[1]["future_mid_return_200ms"])


def test_maker_bid_sell_pressure_proxy_uses_future_observed_sell_flow() -> None:
    sell_volume = [0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "1970-01-01", periods=len(sell_volume), freq="100ms"
            ),
            "segment_id": ["s1"] * len(sell_volume),
            "mid": [100.0] * len(sell_volume),
            "best_bid_qty": [2.0] * len(sell_volume),
            "observed_taker_sell_volume_100ms": sell_volume,
            "market_id": ["crypto"] * len(sell_volume),
            "asset_id": ["btc_usdt"] * len(sell_volume),
        }
    )
    labels = pd.concat(
        iter_label_frames((frame,), horizons_ms=(500,)), ignore_index=True
    )
    assert "maker_bid_sell_pressure_proxy_500ms" in labels
    assert set(column for column in labels if column.startswith("maker_bid_")) == {
        "maker_bid_sell_pressure_proxy_500ms",
        "maker_bid_sell_pressure_proxy_500ms_available_at",
    }
    assert labels.loc[0, "maker_bid_sell_pressure_proxy_500ms"] == 1.0
    assert labels.loc[1, "maker_bid_sell_pressure_proxy_500ms"] == 0.75
    assert labels["maker_bid_sell_pressure_proxy_500ms"].iloc[-1:].isna().all()
    available = labels["maker_bid_sell_pressure_proxy_500ms_available_at"]
    assert available.iloc[:2].tolist() == [
        pd.Timestamp("1970-01-01T00:00:00.500000"),
        pd.Timestamp("1970-01-01T00:00:00.600000"),
    ]
    assert available.iloc[2:].isna().all()


def test_factor_and_label_rows_reject_non_grid_gap_in_same_segment() -> None:
    complete = list(
        _panel_rows(
            [_state(0), _state(200)],
            [],
            start_ms=0,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
        )
    )
    rows = [complete[0], complete[2]]
    with pytest.raises(ValueError, match="hft_grid_non_contiguous"):
        list(iter_factor_frames((pd.DataFrame(rows),)))
    with pytest.raises(ValueError, match="hft_grid_non_contiguous"):
        list(iter_label_frames((pd.DataFrame(rows),), horizons_ms=(HFT_GRID_MS,)))


def test_factor_and_label_frames_match_across_chunk_boundaries() -> None:
    panel = pd.DataFrame(
        _panel_rows(
            [_state(index * HFT_GRID_MS) for index in range(15)],
            [],
            start_ms=0,
            end_ms=14 * HFT_GRID_MS,
            book_coverage={"s1": _coverage(0, 14 * HFT_GRID_MS + 1)},
            trade_coverage={"trade-connection-1": _coverage(0, 14 * HFT_GRID_MS + 1)},
        )
    )
    chunks = (panel.iloc[:7].copy(), panel.iloc[7:].copy())
    factor_full = pd.concat(iter_factor_frames((panel,)), ignore_index=True)
    factor_chunked = pd.concat(iter_factor_frames(chunks), ignore_index=True)
    label_full = pd.concat(
        iter_label_frames((panel,), horizons_ms=(HFT_GRID_MS, 500)), ignore_index=True
    )
    label_chunked = pd.concat(
        iter_label_frames(chunks, horizons_ms=(HFT_GRID_MS, 500)), ignore_index=True
    )
    pd.testing.assert_frame_equal(factor_chunked, factor_full, check_exact=True)
    pd.testing.assert_frame_equal(label_chunked, label_full, check_exact=True)


@pytest.mark.parametrize("price_unit,ticks", [("0.01", 247900), ("0.1", 600000)])
def test_persisted_tick_spread_preserves_constant_and_real_tick_changes(
    tmp_path, price_unit, ticks
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout

    states = [
        replace(
            _state(
                i * 100,
                bid_index=ticks + i,
                ask_index=ticks + i + (2 if i == 9 else 1),
                bid_qty=0.1,
                ask_qty=0.1,
            ),
            price_unit=price_unit,
        )
        for i in range(10)
    ]
    panel = pd.concat(
        iter_panel_frames(
            states,
            [_trade(i * 100, 1.0, maker_buyer=False) for i in range(10)],
            market_id="crypto",
            asset_id="btc_usdt",
            batch_rows=2,
            start_ms=0,
            end_ms=900,
            book_coverage={"s1": _coverage(0, 1001)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 1001)}),
        ),
        ignore_index=True,
    )
    spec = hft_processed_catalog().get("binance_futures_l2_panel_100ms")
    layout = StorageLayout(project_root=tmp_path)
    root = layout.target_root(
        spec.data_id,
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
    )
    write_dataset(root, panel, spec=spec)
    restored = read_dataset(root)
    factors = pd.concat(iter_factor_frames((restored,)), ignore_index=True)
    assert factors["spread"].iloc[:9].tolist() == [float(price_unit)] * 9
    assert factors["spread"].iloc[9] == 2 * float(price_unit)
    assert factors["microprice_deviation"].tolist() == [0.0] * 10


@pytest.mark.parametrize("frame_rows", [2, 11])
@pytest.mark.parametrize("numerator", [11, 13, 19])
@pytest.mark.parametrize("perturb", [False, True])
def test_persisted_return_volatility_preserves_constant_windows_and_tick_changes(
    tmp_path, numerator, perturb, frame_rows
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout

    value = 10**10
    states = []
    for index in range(11):
        mid = value + int(perturb and index == 10)
        states.append(
            replace(
                _state(index * 100, bid_index=2 * mid - 1, ask_index=2 * mid + 1),
                price_unit="0.5",
            )
        )
        value = value * numerator // 10
    panel = pd.concat(
        iter_panel_frames(
            states,
            [_trade(i * 100, 1.0, maker_buyer=False) for i in range(11)],
            market_id="crypto",
            asset_id="btc_usdt",
            batch_rows=2,
            start_ms=0,
            end_ms=1000,
            book_coverage={"s1": _coverage(0, 1101)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 1101)}),
        ),
        ignore_index=True,
    )
    layout = StorageLayout(project_root=tmp_path)
    catalog = hft_processed_catalog()
    restored = panel
    for data_id in (
        "binance_futures_l2_panel_100ms",
        "binance_futures_microstructure_factors_100ms",
    ):
        spec = catalog.get(data_id)
        root = layout.target_root(
            spec.data_id,
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
        )
        frame = (
            panel
            if data_id == "binance_futures_l2_panel_100ms"
            else pd.concat(
                iter_factor_frames(
                    restored.iloc[start : start + frame_rows]
                    for start in range(0, len(restored), frame_rows)
                ),
                ignore_index=True,
            )
        )
        frame = frame.loc[:, list(spec.column_roles)]
        write_dataset(root, frame, spec=spec)
        restored = read_dataset(root)
    returns = restored["return_100ms"].dropna()
    volatility = restored["realized_volatility_1s"]
    assert len(returns) == 10
    assert returns.iloc[:9].nunique() == 1
    assert volatility.iloc[:10].isna().all()
    if perturb:
        assert returns.iloc[-1] != returns.iloc[0]
        expected = abs(returns.iloc[-1] - returns.iloc[0]) / math.sqrt(10)
        assert volatility.iloc[-1] == pytest.approx(expected, rel=1e-14, abs=0)
        assert volatility.iloc[-1] > 0
    else:
        assert returns.nunique() == 1
        assert volatility.iloc[-1] == 0.0
    reset_panel = panel.copy()
    reset_panel.loc[8:, "segment_id"] = "s2:trade-connection-1"
    reset_factors = pd.concat(
        iter_factor_frames((reset_panel.iloc[:9], reset_panel.iloc[9:])),
        ignore_index=True,
    )
    assert reset_factors["realized_volatility_1s"].iloc[8:].isna().all()
    assert reset_factors["return_100ms"].iloc[8:9].isna().all()


def test_price_unit_is_only_a_coordinate_for_panel_factors_and_all_labels():
    from dataclasses import replace

    from tests.helpers.binance_hft import coverage_lookup

    baseline = None
    for multiplier, unit in ((1, "0.1"), (10, "0.01"), (100, "0.001")):
        states = []
        for stamp in range(0, 6200, 100):
            state = _state(
                stamp,
                bid_index=995 + stamp // 100 % 4,
                ask_index=1005 + stamp // 100 % 3,
                bid_qty=2 + stamp // 100 % 3,
                ask_qty=1 + stamp // 100 % 2,
            )
            states.append(
                replace(
                    state,
                    price_unit=unit,
                    bids=tuple(
                        ((index * multiplier, qty) for index, qty in state.bids)
                    ),
                    asks=tuple(
                        ((index * multiplier, qty) for index, qty in state.asks)
                    ),
                )
            )
        trades = []
        for stamp in range(40, 6050, 80):
            trade = _trade(
                stamp, 0.2 + stamp // 80 % 3, maker_buyer=stamp // 80 % 2 == 0
            )
            trades.append(
                replace(
                    trade, price_unit=unit, price_index=trade.price_index * multiplier
                )
            )
        panel = pd.concat(
            list(
                iter_panel_frames(
                    states,
                    trades,
                    market_id="crypto",
                    asset_id="btc_usdt",
                    start_ms=0,
                    end_ms=6000,
                    batch_rows=17,
                    book_coverage={"s1": (0, 7000000000)},
                    trade_coverage=coverage_lookup(
                        {"trade-connection-1": (0, 7000000000)}
                    ),
                )
            ),
            ignore_index=True,
        )
        factor = pd.concat(list(iter_factor_frames([panel])), ignore_index=True)
        label = pd.concat(
            list(iter_label_frames([panel], horizons_ms=(100, 500, 1000, 5000))),
            ignore_index=True,
        )
        actual = (panel, factor, label)
        if baseline is None:
            baseline = actual
        else:
            for expected, frame in zip(baseline, actual, strict=True):
                pd.testing.assert_frame_equal(expected, frame, check_exact=True)


@pytest.mark.parametrize("frame_rows", [2, 10])
def test_persisted_depth_ratios_keep_scale_invariance_and_one_ulp_change(
    tmp_path, frame_rows
):
    from fractions import Fraction

    from quant_research_platform.research.models.scaler import ModelMatrixScaler
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout

    quantities = [0.01, 0.015, 0.03, 0.035, 0.07, 0.1, 0.15, 0.3, 0.5, 0.15]
    states = []
    for index, ask in enumerate(quantities):
        bid = (
            math.nextafter(2 * ask, math.inf)
            if index == len(quantities) - 1
            else 2 * ask
        )
        states.append(
            replace(
                _state(index * 100),
                bids=tuple((995 - level * 10, bid) for level in range(10)),
                asks=tuple((1005 + level * 10, ask) for level in range(10)),
            )
        )
    panel = pd.concat(
        iter_panel_frames(
            states,
            [_trade(i * 100, 1.0, maker_buyer=False) for i in range(10)],
            market_id="crypto",
            asset_id="btc_usdt",
            batch_rows=2,
            start_ms=0,
            end_ms=900,
            book_coverage={"s1": _coverage(0, 1001)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 1001)}),
        ),
        ignore_index=True,
    )
    catalog = hft_processed_catalog()
    layout = StorageLayout(project_root=tmp_path)
    restored = panel
    for data_id in (
        "binance_futures_l2_panel_100ms",
        "binance_futures_microstructure_factors_100ms",
    ):
        spec = catalog.get(data_id)
        root = layout.target_root(
            spec.data_id,
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
        )
        frame = (
            panel
            if data_id == "binance_futures_l2_panel_100ms"
            else pd.concat(
                iter_factor_frames(
                    restored.iloc[start : start + frame_rows]
                    for start in range(0, len(restored), frame_rows)
                ),
                ignore_index=True,
            )
        )
        frame = frame.loc[:, list(spec.column_roles)]
        write_dataset(root, frame, spec=spec)
        restored = read_dataset(root)
        if data_id == "binance_futures_l2_panel_100ms":
            for levels in (1, 5, 10):
                for bid, ask in zip(
                    restored[f"bid_depth_{levels}"].iloc[:-1],
                    restored[f"ask_depth_{levels}"].iloc[:-1],
                    strict=True,
                ):
                    assert Fraction(bid) / Fraction(ask) == 2
    columns = [f"imbalance_{levels}" for levels in (1, 5, 10)] + [
        "microprice_deviation"
    ]
    for column in columns[:3]:
        assert restored[column].iloc[:-1].tolist() == [1 / 3] * 9
    assert restored.microprice_deviation.iloc[:-1].tolist() == [1 / 6] * 9
    assert restored.imbalance_1.iloc[-1] > 1 / 3
    assert restored.microprice_deviation.iloc[-1] > 1 / 6
    scaler = ModelMatrixScaler.fit(restored.loc[:8, columns])
    assert scaler.scale_ == (1.0,) * 4
    assert not scaler.transform(restored.loc[:8, columns]).to_numpy().any()
    assert (
        scaler.transform(restored.loc[9:, columns]).iloc[0, 0]
        == restored.imbalance_1.iloc[-1] - 1 / 3
    )


@pytest.mark.parametrize(
    "unit,bid,ask,expected",
    [
        ("1e300", 100000000, 150000000, 1.25e308),
        ("0.0000000000000001", 10000000000000000, 10000000000000001, 1.0),
    ],
)
@pytest.mark.parametrize("owner", ["execution", "panel"])
def test_panel_and_execution_midpoint_round_only_final_exact_price(
    unit, bid, ask, expected, tmp_path, owner
):
    from quant_research_platform.markets.implementations.crypto.execution import _mid
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    state = replace(_state(0, bid_index=bid, ask_index=ask), price_unit=unit)
    if owner == "execution":
        assert _mid(state) == expected
        return
    panel = pd.concat(
        iter_panel_frames(
            [state],
            [],
            market_id="crypto",
            asset_id="btc_usdt",
            batch_rows=2,
            start_ms=0,
            end_ms=0,
            book_coverage={"s1": _coverage(0, 1001)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 1001)}),
        ),
        ignore_index=True,
    )
    assert panel.mid.tolist() == [expected]
    spec = hft_processed_catalog().get("binance_futures_l2_panel_100ms")
    from quant_research_platform.storage.layout import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, panel, spec=spec)
    assert read_dataset(root).mid.tolist() == [expected]


def test_l0_full_windows_chunk_segments_assets_causality_and_persistence(tmp_path):
    import numpy as np

    size = 110
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp_ms": np.arange(size) * 100,
            "segment_id": ["a"] * 55 + ["b"] * 55,
            "mid": 100.0,
            "book_spread": 2.0,
            "bid_depth_1": 3.0,
            "ask_depth_1": 1.0,
            "bid_depth_5": 8.0,
            "ask_depth_5": 2.0,
            "bid_depth_10": 12.0,
            "ask_depth_10": 8.0,
            "book_ofi_100ms": 2.0,
            "observed_signed_trade_volume_100ms": 2.0,
            "observed_taker_buy_volume_100ms": 3.0,
            "observed_taker_sell_volume_100ms": 1.0,
        }
    )
    full = pd.concat(iter_factor_frames([panel]), ignore_index=True)
    for names in (
        (
            "return_100ms",
            "ofi_500ms",
            "depth_normalized_ofi_500ms",
            "trade_flow_imbalance_500ms",
            "return_500ms",
        ),
        ("microprice",),
        ("imbalance_5_minus_10",),
        ("realized_volatility_1s",),
    ):
        projected = pd.concat(
            iter_factor_frames([panel.iloc[:7], panel.iloc[7:]], columns=names),
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(
            projected.loc[:, list(names)], full.loc[:, list(names)]
        )
    extreme = panel.iloc[:25].copy()
    extreme.loc[:, "mid"] = 100.0
    extreme.loc[12, "mid"] = np.finfo(float).tiny
    extreme.loc[13:, "mid"] = 1e308
    with np.errstate(over="ignore", invalid="ignore"):
        extreme_vol = pd.concat(
            iter_factor_frames([extreme], columns=("realized_volatility_1s",)),
            ignore_index=True,
        )["realized_volatility_1s"]
    assert extreme_vol.iloc[13:23].isna().all()
    assert extreme_vol.iloc[23] == 0.0
    chunked = pd.concat(
        iter_factor_frames([panel.iloc[i : i + 7] for i in range(0, size, 7)]),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(full, chunked, check_exact=True)
    oracle = {
        "trade_flow_imbalance_100ms": 0.5,
        "trade_flow_imbalance_500ms": 0.5,
        "trade_flow_imbalance_1s": 0.5,
        "trade_flow_imbalance_5s": 0.5,
        "ofi_500ms": 10.0,
        "ofi_1s": 20.0,
        "ofi_5s": 100.0,
        "depth_normalized_ofi_500ms": 2.5,
        "depth_normalized_ofi_1s": 5.0,
        "depth_normalized_ofi_5s": 25.0,
        "imbalance_1_minus_10": 0.3,
        "imbalance_5_minus_10": 0.4,
        "near_depth_share_1_10": 0.2,
        "taker_sell_pressure_500ms": 5 / 3,
        "taker_buy_pressure_500ms": 15.0,
    }
    assert len(oracle) == 15
    assert set(oracle) <= set(full.columns)
    for name, value in oracle.items():
        assert full.loc[54, name] == pytest.approx(value), name
        assert full.loc[109, name] == pytest.approx(value), name
    assert full.ofi_5s.iloc[:49].isna().all()
    assert full.ofi_5s.iloc[55:104].isna().all()
    changed = panel.copy()
    changed.loc[54:, "observed_signed_trade_volume_100ms"] = 0
    future = pd.concat(iter_factor_frames([changed]), ignore_index=True)
    pd.testing.assert_frame_equal(future.iloc[:54], full.iloc[:54], check_exact=True)
    zero = panel.assign(
        observed_taker_buy_volume_100ms=0.0,
        observed_taker_sell_volume_100ms=0.0,
        observed_signed_trade_volume_100ms=0.0,
    )
    zeros = pd.concat(iter_factor_frames([zero]), ignore_index=True)
    assert zeros.trade_flow_imbalance_5s.isna().all()
    assert zeros.taker_sell_pressure_500ms.iloc[54] == 0
    other = panel.iloc[:10].assign(asset_id="eth_usdt")
    separate = list(iter_factor_frames([panel.iloc[:55], other]))[-1]
    assert separate.ofi_1s.iloc[:9].isna().all()
    path = tmp_path / "l0.parquet"
    full.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), full, check_exact=True)
