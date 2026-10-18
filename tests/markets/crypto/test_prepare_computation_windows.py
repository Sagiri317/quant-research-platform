"""Output projection must not truncate causal warmup or label maturity."""

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto import ids
from quant_research_platform.markets.implementations.crypto.processed import (
    catalog,
    producer,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_DERIVATIVE_CHANNELS,
)
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.helpers.binance_hft import write_binance_hft_run


@pytest.fixture(scope="module")
def native_panel(tmp_path_factory):
    root = tmp_path_factory.mktemp("computation-windows")
    write_binance_hft_run(root / "raw")
    path = prepare(
        {
            "market_id": "crypto",
            "data_id": ids.HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(root / "raw")},
            "time_window": {
                "start": "1970-01-01T00:00:00",
                "end": "1970-01-01T00:00:06",
            },
            "output_root": str(root / "panel"),
        }
    )
    return read_dataset(path)


def persist_panel(path, frame, data_id=ids.HFT_PANEL_DATA_ID):
    write_dataset(path, frame, spec=catalog.hft_processed_catalog().get(data_id))
    return path


def run_prepare(path, inputs, data_id, start, end):
    return read_dataset(
        prepare(
            {
                "market_id": "crypto",
                "data_id": data_id,
                "inputs": {key: str(value) for key, value in inputs.items()},
                "time_window": {"start": start.isoformat(), "end": end.isoformat()},
                "output_root": str(path),
                **(
                    {"parameters": {"max_age_ms": 200}}
                    if data_id == ids.HFT_SPOT_CONTEXT_DATA_ID
                    else {
                        "parameters": {
                            "max_age_ms": 200,
                            "information_assets": ["btc_usdt", "eth_usdt"],
                            "target_assets": ["btc_usdt", "eth_usdt"],
                        }
                    }
                    if data_id == ids.HFT_CROSS_ASSET_DATA_ID
                    else {}
                ),
            }
        )
    )


def assert_projection(broad, narrow, start, end):
    expected = broad.loc[broad.timestamp.between(start, end)].reset_index(drop=True)
    actual = narrow.reset_index(drop=True)
    continuous = [
        name
        for name in expected.select_dtypes(include="floating")
        if "direction" not in name
    ]
    exact = expected.columns.difference(continuous)
    pd.testing.assert_frame_equal(expected[exact], actual[exact], check_exact=True)
    pd.testing.assert_frame_equal(expected.isna(), actual.isna(), check_exact=True)
    pd.testing.assert_frame_equal(
        expected[continuous],
        actual[continuous],
        check_exact=False,
        rtol=2e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("data_id", [ids.HFT_FACTOR_DATA_ID, ids.HFT_LABEL_DATA_ID])
@pytest.mark.parametrize("gap", [False, True])
def test_native_prepare_window_partition_chunk_equivalence(
    tmp_path, monkeypatch, native_panel, data_id, gap
):
    frame = native_panel.copy()
    offset = pd.Timestamp("2025-12-31T23:59:57") - pd.Timestamp("1970-01-01")
    frame["timestamp"] += offset
    frame["observed_utc"] += offset
    if gap:
        boundary = offset + pd.Timestamp("1970-01-01T00:00:02.5")
        frame = frame.loc[frame.timestamp != boundary].copy()
        frame.loc[frame.timestamp > boundary, "segment_id"] += ":new"
    panel = persist_panel(tmp_path / "panel", frame)
    start, end = frame.timestamp.min(), frame.timestamp.max()
    narrow_start, narrow_end = (
        start + pd.Timedelta(seconds=1.9),
        end - pd.Timedelta(seconds=1),
    )
    inputs = {ids.HFT_PANEL_DATA_ID: panel}
    broad = run_prepare(tmp_path / "broad", inputs, data_id, start, end)
    original = producer.iter_dataset_chunks
    monkeypatch.setattr(
        producer,
        "iter_dataset_chunks",
        lambda session, **kwargs: original(session, batch_rows=3),
    )
    narrow = run_prepare(tmp_path / "narrow", inputs, data_id, narrow_start, narrow_end)
    assert_projection(broad, narrow, narrow_start, narrow_end)
    assert frame.timestamp.dt.year.nunique() == 2
    if data_id == ids.HFT_FACTOR_DATA_ID:
        first = narrow.loc[narrow.asset_id == "btc_usdt"].iloc[0]
        assert pd.notna(first.return_500ms)
        if gap:
            assert (
                narrow.loc[
                    narrow.timestamp == boundary + pd.Timedelta(milliseconds=100),
                    "return_500ms",
                ]
                .isna()
                .all()
            )
    else:
        assert broad.loc[broad.timestamp == end, "future_mid_return_100ms"].isna().all()
        assert (
            narrow.loc[
                narrow.timestamp == narrow.timestamp.max(), "future_mid_return_100ms"
            ]
            .notna()
            .all()
        )


def test_spot_and_cross_keep_warmup_and_stale_clocks(tmp_path, native_panel):
    frame = native_panel
    start, end = frame.timestamp.min(), frame.timestamp.max()
    cut = start + pd.Timedelta(seconds=2)
    panel = persist_panel(tmp_path / "panel", frame)
    spot = persist_panel(
        tmp_path / "spot", frame.loc[frame.timestamp <= cut], ids.HFT_SPOT_PANEL_DATA_ID
    )
    inputs = {ids.HFT_PANEL_DATA_ID: panel, ids.HFT_SPOT_PANEL_DATA_ID: spot}
    narrow_start = cut + pd.Timedelta(milliseconds=100)
    broad = run_prepare(
        tmp_path / "spot-broad", inputs, ids.HFT_SPOT_CONTEXT_DATA_ID, start, end
    )
    narrow = run_prepare(
        tmp_path / "spot-narrow",
        inputs,
        ids.HFT_SPOT_CONTEXT_DATA_ID,
        narrow_start,
        end,
    )
    assert_projection(broad, narrow, narrow_start, end)
    assert narrow.spot_grid_at.notna().all()
    assert narrow.spot_grid_age_ms.max() > 1000
    info_panel = persist_panel(
        tmp_path / "info-panel",
        frame.loc[frame.timestamp <= cut],
        ids.HFT_INFORMATION_PANEL_DATA_ID,
    )
    info = prepare(
        {
            "market_id": "crypto",
            "data_id": ids.HFT_INFORMATION_STATE_DATA_ID,
            "base_panel": ids.HFT_INFORMATION_PANEL_DATA_ID,
            "inputs": {ids.HFT_INFORMATION_PANEL_DATA_ID: str(info_panel)},
            "time_window": {"start": start.isoformat(), "end": cut.isoformat()},
            "output_root": str(tmp_path / "info"),
        }
    )
    spec = catalog.hft_processed_catalog().get(ids.HFT_CROSS_ASSET_DATA_ID)
    inputs = {
        spec.processed_input_data_ids[0]: panel,
        spec.processed_input_data_ids[1]: info,
    }
    broad = run_prepare(
        tmp_path / "cross-broad", inputs, ids.HFT_CROSS_ASSET_DATA_ID, start, end
    )
    narrow = run_prepare(
        tmp_path / "cross-narrow",
        inputs,
        ids.HFT_CROSS_ASSET_DATA_ID,
        narrow_start,
        end,
    )
    assert_projection(broad, narrow, narrow_start, end)
    assert narrow.btc_grid_at.notna().all()


def test_derivative_five_minute_warmup_is_bounded(tmp_path, monkeypatch, native_panel):
    template = native_panel.loc[native_panel.asset_id == "btc_usdt"].iloc[[0]]
    frame = pd.concat([template] * 3102, ignore_index=True)
    frame["timestamp"] = pd.date_range("2026-01-01", periods=len(frame), freq="100ms")
    panel = persist_panel(tmp_path / "panel", frame)
    origin = frame.timestamp.iloc[0]
    events = [
        (
            origin.value,
            "openInterest",
            {"openInterest": "10", "time": origin.value // 1_000_000},
            0,
        ),
        (
            (origin + pd.Timedelta(seconds=300)).value,
            "openInterest",
            {"openInterest": "12", "time": origin.value // 1_000_000 + 300000},
            0,
        ),
    ]

    @contextmanager
    def archive(_root, *, time_window):
        assert time_window.start == pd.Timestamp("1970-01-01")
        yield SimpleNamespace(
            source_identity={}, iter_state_observations=lambda **kwargs: iter(events)
        )

    spec = catalog.hft_processed_catalog().get(ids.HFT_DERIVATIVE_DATA_ID)
    params = json.loads(spec.parameters_json)
    params["max_age_ms"] = dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000)
    spec = replace(spec, parameters_json=json.dumps(params))
    windows = []
    original = producer.iter_dataset_chunks

    def chunks(session, **kwargs):
        windows.append(session.time_window)
        with closing_iterator(original(session, **kwargs)) as frames:
            for part in frames:
                assert part.timestamp.between(
                    session.time_window.start, session.time_window.end
                ).all()
                yield part

    monkeypatch.setattr(producer, "iter_dataset_chunks", chunks)

    def compute(start):
        with producer.produce_derivative_processed_data(
            spec,
            inputs={ids.HFT_PANEL_DATA_ID: panel},
            time_window=TimeWindow(start, frame.timestamp.iloc[-1]),
            processed_catalog=None,
            open_archive=archive,
            source_catalog=None,
        ) as (frames, _metadata):
            return pd.concat(frames, ignore_index=True)

    broad = compute(origin)
    start = origin + pd.Timedelta(seconds=305)
    narrow = compute(start)
    assert_projection(broad, narrow, start, frame.timestamp.iloc[-1])
    assert narrow.open_interest_change_5m.iloc[0] == pytest.approx(0.2)
    assert windows[-1].start == start - pd.Timedelta(seconds=300)


@pytest.mark.parametrize("missing", [False, True])
def test_external_prior_day_prefix_and_delayed_warmup(tmp_path, native_panel, missing):
    frame = native_panel.copy()
    offset = pd.Timestamp("2025-12-31T23:59:58") - pd.Timestamp("1970-01-01")
    frame["timestamp"] += offset
    frame["observed_utc"] += offset
    panel = persist_panel(tmp_path / "panel", frame)
    start, end = frame.timestamp.min(), frame.timestamp.max()
    quote_time = start - pd.Timedelta(milliseconds=250)
    quote = {
        "local_timestamp": quote_time.value // 1000,
        "timestamp": quote_time.value // 1000,
        "bid_price": 99.0,
        "ask_price": 101.0,
    }
    trades = [
        {
            "local_timestamp": stamp.value // 1000,
            "timestamp": stamp.value // 1000,
            "amount": 1.0,
            "side": "buy",
            "price": 100.0,
        }
        for stamp in frame.timestamp.drop_duplicates()
    ]
    reads = []

    def records(_root, *, archive_paths, time_window, expected_scope, on_source):
        reads.append(time_window)
        yield from (
            [] if missing else [quote] if archive_paths == ["quotes"] else trades
        )
        on_source("a" * 64)

    spec = catalog.hft_processed_catalog().get(ids.HFT_EXTERNAL_DATA_IDS["bybit"])
    params = json.loads(spec.parameters_json)
    params.update(
        max_age_ms=10000,
        input_delay_ms=150,
        venue="bybit",
        symbols={"btc_usdt": "BTCUSDT", "eth_usdt": "ETHUSDT"},
        base_units_per_amount={"btc_usdt": 1.0, "eth_usdt": 1.0},
    )
    spec = replace(spec, parameters_json=json.dumps(params))

    def compute(begin):
        with producer.produce_normalized_context(
            spec,
            inputs={ids.HFT_PANEL_DATA_ID: panel},
            time_window=TimeWindow(begin, end),
            processed_catalog=None,
            source_catalog=None,
            raw_root=tmp_path,
            read_records=records,
            source_schema_id="test",
            archives={
                symbol: {"quotes": ["quotes"], "trades": ["trades"]}
                for symbol in ("BTCUSDT", "ETHUSDT")
            },
        ) as (frames, _metadata):
            return pd.concat(frames, ignore_index=True)

    broad = compute(start)
    narrow_start = pd.Timestamp("2026-01-01T00:00:00")
    narrow = compute(narrow_start)
    assert_projection(broad, narrow, narrow_start, end)
    assert all(window.start < quote_time and window.end == end for window in reads)
    if missing:
        assert narrow.bybit_mid_usdt.isna().all()
        assert narrow.bybit_quote_observed_at.isna().all()
    else:
        assert narrow.bybit_quote_observed_at.eq(quote_time).all()
        assert narrow.bybit_observed_signed_base_flow_500ms.eq(5).all()


def test_output_projection_closes_upstream_on_early_exit():
    closed = []

    def frames():
        try:
            while True:
                yield pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]})
        finally:
            closed.append(True)

    result = producer._output_frames(
        frames(), TimeWindow("2026-01-01T00:00:00", "2026-01-01T00:00:01")
    )
    next(result)
    result.close()
    assert closed == [True]


@pytest.mark.parametrize("data_id", [ids.HFT_FACTOR_DATA_ID, ids.HFT_LABEL_DATA_ID])
def test_core_reads_only_finite_dependency_interval(
    tmp_path, monkeypatch, native_panel, data_id
):
    panel = persist_panel(tmp_path / "panel", native_panel)
    inputs = {ids.HFT_PANEL_DATA_ID: panel}
    first, last = native_panel.timestamp.min(), native_panel.timestamp.max()
    broad = run_prepare(tmp_path / "broad", inputs, data_id, first, last)
    windows = []
    original = producer.iter_dataset_chunks

    def chunks(session, **kwargs):
        windows.append(session.time_window)
        with closing_iterator(original(session, **kwargs)) as frames:
            for part in frames:
                assert part.timestamp.between(
                    session.time_window.start, session.time_window.end
                ).all()
                yield part

    monkeypatch.setattr(producer, "iter_dataset_chunks", chunks)
    start, end = first + pd.Timedelta(seconds=5.2), first + pd.Timedelta(seconds=5.3)
    narrow = run_prepare(tmp_path / "narrow", inputs, data_id, start, end)
    assert_projection(broad, narrow, start, end)
    if data_id == ids.HFT_FACTOR_DATA_ID:
        assert windows[0] == TimeWindow(start - pd.Timedelta(milliseconds=4900), end)
        assert narrow.ofi_5s.notna().all()
    else:
        params = json.loads(
            catalog.hft_processed_catalog().get(data_id).parameters_json
        )
        horizon = max(item["horizon"] for item in params["labels"])
        assert windows[0] == TimeWindow(start, end + pd.Timedelta(milliseconds=horizon))
        assert narrow.future_mid_return_100ms.notna().all()
        assert narrow.future_mid_return_5000ms.isna().all()
