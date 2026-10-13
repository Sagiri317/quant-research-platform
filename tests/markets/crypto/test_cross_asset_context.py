"""Information assets use comparable matured features without selecting execution."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_cross_asset_factor_frames,
    iter_factor_frames,
)
from tests.markets.crypto.test_spot_context import panel


def test_cross_asset_units_lag_segment_and_chunk_causality():
    btc = panel(spot=True)
    sol = btc.copy()
    sol["asset_id"] = "sol_usdt"
    sol[["mid", "book_spread"]] *= 0.01
    quantity = [
        key
        for key in sol
        if key.startswith(("bid_depth", "ask_depth", "observed_"))
        and key != "observed_utc"
    ]
    sol[quantity] *= 100
    sol["book_ofi_100ms"] *= 100
    btc_state = next(iter_factor_frames([btc]))
    sol_state = next(iter_factor_frames([sol]))
    targets = btc[["timestamp"]]

    def compute(target_frames, btc_frames, sol_frames):
        return pd.concat(
            iter_cross_asset_factor_frames(
                target_frames,
                information_frames={"btc_usdt": btc_frames, "sol_usdt": sol_frames},
                max_age_ms=200,
                batch_rows=7,
            ),
            ignore_index=True,
        )

    full = compute([targets], [btc_state], [sol_state])
    split = compute(
        [targets.iloc[i : i + 3] for i in range(0, 20, 3)],
        [btc_state.iloc[i : i + 4] for i in range(0, 20, 4)],
        [sol_state.iloc[i : i + 6] for i in range(0, 20, 6)],
    )
    pd.testing.assert_frame_equal(full, split, check_exact=True)
    for metric in ("return_100ms", "return_500ms", "normalized_ofi_500ms", "tfi_500ms"):
        np.testing.assert_allclose(
            full["btc_" + metric],
            full["sol_" + metric],
            rtol=1e-11,
            atol=1e-14,
            equal_nan=True,
        )
    assert full.sol_return_500ms.iloc[5] == pytest.approx(0.025)
    assert full.sol_lagged_return_100ms.iloc[5] == sol_state.return_100ms.iloc[4]
    assert full.sol_lagged_tfi_500ms.iloc[5] == 0.25
    changed = sol_state.copy()
    changed.loc[9:, "return_100ms"] = 99
    pd.testing.assert_frame_equal(
        full.iloc[:9],
        compute([targets], [btc_state], [changed]).iloc[:9],
        check_exact=True,
    )
    gap = sol_state.drop(index=range(8, 14)).copy()
    gap.loc[14, "return_100ms"] = np.nan
    limited = compute([targets], [btc_state], [gap])
    assert limited.sol_grid_age_ms.iloc[10] == 300
    assert pd.isna(limited.sol_return_100ms.iloc[10])
    assert pd.isna(limited.sol_lagged_return_100ms.iloc[14])
    assert limited.btc_return_100ms.equals(full.btc_return_100ms)
    missing = compute([targets], [btc_state], [])
    assert missing.filter(regex="^sol_").isna().all().all()
    assert missing.btc_return_100ms.equals(full.btc_return_100ms)


@pytest.mark.parametrize("chunk_rows", ((1, 3), (4, 1), (20, 20)))
@pytest.mark.parametrize("batch_rows", (1, 4, 20))
def test_information_union_preserves_complete_keys_and_exact_nullable_values(
    chunk_rows, batch_rows
):
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer,
    )

    def source(asset, positions):
        return pd.DataFrame(
            {
                "market_id": pd.Series(["crypto"] * len(positions), dtype="string"),
                "asset_id": pd.Series([asset] * len(positions), dtype="string"),
                "timestamp": pd.to_datetime(positions, unit="ms").as_unit("ns"),
                "segment_id": pd.Series(
                    [f"{asset}-native-{i // 2}" for i in positions], dtype="string"
                ),
                "transaction_time_ms": pd.array(
                    [None if i % 2 == 0 else 2**63 - 1 for i in positions],
                    dtype="Int64",
                ),
                "receive_monotonic_ns": pd.array(
                    [None if i % 3 == 0 else 2**53 + 1 for i in positions],
                    dtype="Int64",
                ),
                "trade_source_sequence_min": pd.array(
                    [None if i % 2 == 0 else 2**53 - 1 for i in positions],
                    dtype="Int64",
                ),
                "value": pd.array(
                    [None if i % 2 else float(i) for i in positions], dtype="Float64"
                ),
            }
        )

    base = pd.concat(
        [source("btc_usdt", (0, 1, 3, 4)), source("eth_usdt", (0, 1, 3, 4))],
        ignore_index=True,
    ).sort_values(["timestamp", "asset_id"], ignore_index=True)
    sol = source("sol_usdt", (-1, 0, 2, 4, 5))
    snapshots = [frame.copy(deep=True) for frame in (base, sol)]
    chunks = [
        [frame.iloc[i : i + width] for i in range(0, len(frame), width)]
        for frame, width in zip((base, sol), chunk_rows, strict=True)
    ]
    output = list(
        producer._union_panel_frames(
            chunks, columns=tuple(base.columns), batch_rows=batch_rows
        )
    )
    assert all(0 < len(frame) <= batch_rows for frame in output)
    assert len(output) == (len(base) + len(sol) + batch_rows - 1) // batch_rows
    combined = pd.concat(output, ignore_index=True)
    expected = pd.concat((base, sol), ignore_index=True).sort_values(
        ["timestamp", "asset_id"], ignore_index=True
    )
    pd.testing.assert_frame_equal(combined, expected, check_exact=True)
    assert not combined.duplicated(["market_id", "asset_id", "timestamp"]).any()
    for original, saved in zip((base, sol), snapshots, strict=True):
        pd.testing.assert_frame_equal(original, saved, check_exact=True)


@pytest.mark.parametrize("empty_kind", ("one", "both", "no_schema", "duplicate"))
def test_information_union_preserves_empty_schema_and_never_deduplicates(empty_kind):
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([0, 1], unit="ms").as_unit("ns"),
            "asset_id": pd.Series(["btc_usdt", "btc_usdt"], dtype="string"),
            "transaction_time_ms": pd.array([None, 2**63 - 1], dtype="Int64"),
        }
    )
    streams = {
        "one": ([frame.iloc[:0]], [frame]),
        "both": ([frame.iloc[:0]], [frame.iloc[:0]]),
        "no_schema": ([], []),
        "duplicate": ([frame], [frame]),
    }[empty_kind]
    output = list(
        producer._union_panel_frames(
            streams, columns=tuple(frame.columns), batch_rows=1
        )
    )
    if empty_kind == "no_schema":
        assert output == []
    else:
        expected = {
            "one": frame,
            "both": frame.iloc[:0],
            "duplicate": pd.concat((frame, frame), ignore_index=True).sort_values(
                ["timestamp", "asset_id"], ignore_index=True
            ),
        }[empty_kind]
        pd.testing.assert_frame_equal(
            pd.concat(output, ignore_index=True),
            expected.reset_index(drop=True),
            check_exact=True,
        )


@pytest.mark.parametrize("exit_mode", ("early_close", "source_error", "source_cancel"))
def test_information_union_closes_all_inputs_and_preserves_primary(exit_mode):
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([0, 1, 2], unit="ms").as_unit("ns"),
            "asset_id": pd.Series(["btc_usdt"] * 3, dtype="string"),
            "transaction_time_ms": pd.array(
                [None, 2**53 + 1, 2**63 - 1], dtype="Int64"
            ),
        }
    )
    closed = []
    primary = (
        KeyboardInterrupt("union_primary")
        if exit_mode == "source_cancel"
        else RuntimeError("union_primary")
    )

    def source(name):
        try:
            yield frame
            if name == "second" and exit_mode != "early_close":
                raise primary
        finally:
            closed.append(name)

    output = producer._union_panel_frames(
        (source("first"), source("second")), columns=tuple(frame.columns), batch_rows=2
    )
    first = next(output)
    saved = first.copy(deep=True)
    if exit_mode == "early_close":
        output.close()
    else:
        with pytest.raises(type(primary)) as raised:
            list(output)
        assert raised.value is primary
    assert sorted(closed) == ["first", "second"]
    pd.testing.assert_frame_equal(first, saved, check_exact=True)


def test_information_union_persists_complete_ordered_panel_for_research_windows(
    tmp_path,
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_panel_frames,
    )
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_INFORMATION_PANEL_DATA_ID,
        HFT_PANEL_DATA_ID,
        HFT_SOL_PANEL_DATA_ID,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        catalog,
        producer,
    )
    from quant_research_platform.research.datasets.assembler import (
        _iter_research_chunk_windows_from_base_input,
    )
    from quant_research_platform.storage.datasets.reading import (
        iter_dataset_chunks,
        open_dataset_session,
        read_dataset,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.binance_hft import _coverage, coverage_lookup
    from tests.markets.crypto.test_hft import _state

    start = pd.Timestamp("2026-08-01T00:00:01")
    start_ms = start.value // 1000000
    window = TimeWindow(
        start - pd.Timedelta(milliseconds=100), start + pd.Timedelta(milliseconds=400)
    )
    processed = catalog.hft_processed_catalog()
    layout = StorageLayout(tmp_path / "project")

    def source(asset, first, last):
        frame = pd.concat(
            list(
                iter_panel_frames(
                    [_state(first), _state(last)],
                    [],
                    market_id="crypto",
                    asset_id=asset,
                    start_ms=first,
                    end_ms=last,
                    book_coverage={"s1": _coverage(first, last + 100)},
                    trade_coverage=coverage_lookup(
                        {"trade": _coverage(first, last + 100)}
                    ),
                    batch_rows=2,
                )
            ),
            ignore_index=True,
        )
        frame["transaction_time_ms"] = pd.array(
            [None if index % 2 == 0 else 2**63 - 1 for index in range(len(frame))],
            dtype="Int64",
        )
        frame["book_receive_monotonic_ns"] = pd.array(
            [None if index % 2 == 0 else 2**53 + 1 for index in range(len(frame))],
            dtype="Int64",
        )
        return frame

    base = pd.concat(
        [source(asset, start_ms, start_ms + 200) for asset in ("btc_usdt", "eth_usdt")],
        ignore_index=True,
    ).sort_values(["timestamp", "asset_id"], ignore_index=True)
    sol = source("sol_usdt", start_ms - 100, start_ms + 400)
    inputs = {HFT_PANEL_DATA_ID: base, HFT_SOL_PANEL_DATA_ID: sol}
    expected_parts = []
    expected_identities = {}
    for data_id, frame in inputs.items():
        source_spec = processed.get(data_id)
        root = layout.processed_data_root(
            source_spec.market_id, source_spec.kind, data_id
        )
        write_dataset(
            root, (frame.iloc[:1].copy(), frame.iloc[1:].copy()), spec=source_spec
        )
        expected_parts.append(read_dataset(root))
        session = open_dataset_session(root, columns=None, time_window=None)
        expected_identities[data_id] = str(root)
    expected = pd.concat(expected_parts, ignore_index=True).sort_values(
        ["timestamp", "asset_id"], ignore_index=True
    )
    expected_parts.clear()
    spec = processed.get(HFT_INFORMATION_PANEL_DATA_ID)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_information_panel(
        spec,
        inputs=expected_identities,
        time_window=window,
        processed_catalog=processed,
        source_catalog=None,
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    frames = None
    restored = read_dataset(root)
    pd.testing.assert_frame_equal(restored, expected, check_exact=True)
    assert len(restored) == len(base) + len(sol)
    assert restored["timestamp"].is_monotonic_increasing
    assert not restored.duplicated(list(spec.primary_key)).any()
    session = open_dataset_session(root, columns=None, time_window=None)
    assert session.manifest["inputs"] == expected_identities
    previous = None
    rows = 0
    for frame in iter_dataset_chunks(session):
        if len(frame):
            if previous is not None:
                assert frame["timestamp"].iloc[0] >= previous
            previous = frame["timestamp"].iloc[-1]
        rows += len(frame)
        frame = None
    assert rows == len(expected)
    windows = list(
        _iter_research_chunk_windows_from_base_input(session, time_window=window)
    )
    assert windows[0].start == window.start
    assert windows[-1].end == window.end
    restored = expected = None


def test_input_delay_postpones_visibility_without_shifting_target_or_source_clock():
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_derivative_factor_frames,
        iter_external_context_frames,
        iter_spot_perp_factor_frames,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_DERIVATIVE_CHANNELS,
    )

    target, information = panel(), panel(spot=True)
    state = next(iter_factor_frames([information]))
    cross = pd.concat(
        iter_cross_asset_factor_frames(
            [target],
            information_frames={"btc_usdt": [state]},
            max_age_ms=150,
            batch_rows=7,
            input_delay_ms=100,
        ),
        ignore_index=True,
    )
    assert cross.timestamp.equals(target.timestamp.dt.as_unit("ns"))
    assert pd.isna(cross.btc_grid_at.iloc[0])
    assert cross.btc_grid_at.iloc[1] == target.timestamp.iloc[0]
    assert cross.btc_grid_age_ms.iloc[1] == 100
    assert cross.btc_return_500ms.iloc[6] == state.return_500ms.iloc[5]
    spot = pd.concat(
        iter_spot_perp_factor_frames(
            [target],
            spot_frames=[information],
            max_age_ms=150,
            batch_rows=7,
            input_delay_ms=100,
        ),
        ignore_index=True,
    )
    assert pd.isna(spot.spot_perp_basis.iloc[0])
    assert spot.spot_perp_basis.iloc[1] == 1.0
    assert spot.spot_book_age_ms.iloc[1] == 101
    origin = target.timestamp.iloc[0].value
    derivative = pd.concat(
        iter_derivative_factor_frames(
            [target],
            observations=[
                (
                    origin,
                    "openInterest",
                    {"openInterest": "10", "time": origin // 1_000_000},
                    0,
                )
            ],
            max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 150),
            input_delay_ms=100,
        ),
        ignore_index=True,
    )
    assert pd.isna(derivative.log_open_interest.iloc[0])
    assert derivative.log_open_interest.iloc[1] == pytest.approx(np.log(10))
    assert derivative.openInterest_observed_at.iloc[1] == target.timestamp.iloc[0]
    assert pd.isna(derivative.log_open_interest.iloc[2])
    quotes = [
        {
            "local_timestamp": origin // 1000,
            "timestamp": origin // 1000 - 1000,
            "bid_price": 199.0,
            "ask_price": 201.0,
        }
    ]
    trades = [
        {
            "local_timestamp": origin // 1000,
            "timestamp": origin // 1000,
            "side": "buy",
            "price": 200.0,
            "amount": 3.0,
        }
    ]
    external = pd.concat(
        iter_external_context_frames(
            [target],
            quotes=quotes,
            trades=trades,
            venue="bybit",
            base_units_per_amount=1.0,
            max_age_ms=1000,
            input_delay_ms=100,
        ),
        ignore_index=True,
    )
    assert pd.isna(external.bybit_mid_usdt.iloc[0])
    assert external.bybit_mid_usdt.iloc[1] == 200.0
    assert external.bybit_quote_age_ms.iloc[1] == 100
    assert external.bybit_quote_observed_at.iloc[1] == target.timestamp.iloc[0]
    assert pd.isna(external.bybit_observed_signed_base_flow_500ms.iloc[5])
    assert pd.isna(external.bybit_observed_signed_base_flow_500ms.iloc[6])


@pytest.mark.parametrize("delay", (0, 50, 100, 150, 199, 200, 201, 250))
def test_delayed_flow_windows_wait_for_complete_history_after_each_reset(delay):
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_derivative_factor_frames,
        iter_external_context_frames,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_DERIVATIVE_CHANNELS,
    )

    frame = pd.concat([panel()] * 6, ignore_index=True)
    frame["timestamp"] = pd.date_range(
        "2026-08-01T00:00:00.100", periods=120, freq="100ms"
    )
    frame.loc[60:, "segment_id"] = "b"
    stamps = frame.timestamp.dt.as_unit("ns").astype("int64").tolist()
    trades = [
        {
            "local_timestamp": stamp // 1000,
            "timestamp": stamp // 1000,
            "side": "buy",
            "price": 100.0,
            "amount": 1.0,
        }
        for stamp in stamps
    ]
    external = pd.concat(
        iter_external_context_frames(
            [frame.iloc[:13], frame.iloc[13:74], frame.iloc[74:]],
            quotes=[],
            trades=trades,
            venue="bybit",
            base_units_per_amount=1.0,
            max_age_ms=1000,
            input_delay_ms=delay,
        ),
        ignore_index=True,
    )
    events = [
        (stamp, "forceOrder", {"E": stamp // 1_000_000, "o": {"z": "1", "S": "BUY"}}, 0)
        for stamp in stamps
    ]
    derivative = pd.concat(
        iter_derivative_factor_frames(
            [frame.iloc[:13], frame.iloc[13:74], frame.iloc[74:]],
            observations=events,
            max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 1000),
            input_delay_ms=delay,
        ),
        ignore_index=True,
    )
    lag = (delay + 99) // 100
    for offset in (0, 60):
        flow = external.bybit_observed_signed_base_flow_500ms.iloc[offset : offset + 60]
        force = derivative.observed_force_order_imbalance_5s.iloc[offset : offset + 60]
        assert flow.iloc[: 5 + lag].isna().all()
        assert flow.iloc[5 + lag] == 5.0
        assert force.iloc[: 49 + lag].isna().all()
        assert force.iloc[49 + lag] == 1.0


@pytest.mark.parametrize("prior_epoch", (False, True))
def test_delayed_force_order_epoch_reset_does_not_restart_delay_twice(prior_epoch):
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_derivative_factor_frames,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_DERIVATIVE_CHANNELS,
    )

    frame = pd.concat([panel()] * 4, ignore_index=True)
    frame["timestamp"] = pd.date_range(
        "2026-08-01T00:00:00.100", periods=80, freq="100ms"
    )
    origin = pd.Timestamp("2026-08-01").value
    events = [
        (origin, None, None, 0),
        (origin + 1_000_000_000, None, None, 1),
        (
            origin + 1_100_000_000,
            "forceOrder",
            {"E": (origin + 1_100_000_000) // 1_000_000, "o": {"z": "1", "S": "BUY"}},
            1,
        ),
    ]

    if not prior_epoch:
        events = events[1:]

    def compute(delay):
        return pd.concat(
            iter_derivative_factor_frames(
                [frame],
                observations=events,
                max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 10000),
                input_delay_ms=delay,
            ),
            ignore_index=True,
        )

    baseline, delayed = compute(0), compute(200)
    assert baseline.observed_force_order_imbalance_5s.iloc[:59].isna().all()
    assert baseline.observed_force_order_imbalance_5s.iloc[59] == 1.0
    assert delayed.observed_force_order_imbalance_5s.iloc[61] == 1.0
    assert delayed.observed_force_order_imbalance_5s.iloc[:61].isna().all()


def test_delayed_source_epoch_cannot_move_panel_flow_boundary_backwards():
    from quant_research_platform.markets.implementations.crypto.hft import (
        iter_derivative_factor_frames,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_DERIVATIVE_CHANNELS,
    )

    frame = pd.concat([panel()] * 5, ignore_index=True)
    frame["timestamp"] = pd.date_range(
        "2026-08-01T00:00:00.100", periods=100, freq="100ms"
    )
    frame.loc[40:, "segment_id"] = "b"
    origin = pd.Timestamp("2026-08-01").value
    events = [
        (origin, None, None, 0),
        (origin + 3_900_000_000, None, None, 1),
        (
            origin + 4_000_000_000,
            "forceOrder",
            {"E": (origin + 4_000_000_000) // 1_000_000, "o": {"z": "1", "S": "BUY"}},
            1,
        ),
    ]
    result = pd.concat(
        iter_derivative_factor_frames(
            [frame],
            observations=events,
            max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 10000),
            input_delay_ms=200,
        ),
        ignore_index=True,
    )
    assert result.observed_force_order_imbalance_5s.isna().all()
