from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    catalog,
    producer,
)
from quant_research_platform.storage.datasets.reading import (
    read_dataset,
)
from tests.helpers.binance_hft import write_binance_hft_run
from tests.helpers.paths import StorageLayout


def test_panel_batch_cap_preserves_persisted_rows(tmp_path, monkeypatch):
    from unittest.mock import patch

    from quant_research_platform.markets.prepare import prepare

    raw = tmp_path / "raw"
    write_binance_hft_run(raw)
    panels = []
    for limit in (producer.DEFAULT_BATCH_ROWS, 3):
        monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", limit)
        with patch.object(
            producer, "iter_panel_frames", wraps=producer.iter_panel_frames
        ) as frames:
            root = prepare(
                {
                    "market_id": "crypto",
                    "data_id": HFT_PANEL_DATA_ID,
                    "inputs": {},
                    "source": {"root": str(raw)},
                    "time_window": {
                        "start": "1970-01-01T00:00:00",
                        "end": "1970-01-01T00:00:06",
                    },
                    "output_root": str(tmp_path / str(limit)),
                }
            )
            assert frames.call_count == 2
            assert {call.kwargs["batch_rows"] for call in frames.call_args_list} == {
                limit // 2
            }
            panel = read_dataset(root).sort_values(
                ["timestamp", "asset_id"], ignore_index=True
            )
            assert len(panel) == 122
            assert not panel.duplicated(["timestamp", "asset_id"]).any()
            assert panel.groupby("timestamp").asset_id.nunique().eq(2).all()
            panels.append(panel)
    pd.testing.assert_frame_equal(panels[0], panels[1], check_exact=True)


def test_hft_direct_preparation_assembly_and_fractional_window(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.sources.binance.replay import ReconstructionStats
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

    raw = tmp_path / "raw"
    write_binance_hft_run(raw)
    window = {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:06"}
    roots = {}
    for data_id in (HFT_PANEL_DATA_ID, HFT_FACTOR_DATA_ID, HFT_LABEL_DATA_ID):
        request = {
            "market_id": "crypto",
            "data_id": data_id,
            "inputs": {}
            if data_id == HFT_PANEL_DATA_ID
            else {HFT_PANEL_DATA_ID: str(roots[HFT_PANEL_DATA_ID])},
            "time_window": window,
            "output_root": str(tmp_path / data_id),
        }
        if data_id == HFT_PANEL_DATA_ID:
            request["source"] = {"root": str(raw)}
        roots[data_id] = prepare(request)
    manifest = read_dataset_manifest_evidence(roots[HFT_PANEL_DATA_ID])
    assert manifest["row_count"] == 122
    assert manifest["source_evidence"]["artifact_id"] == "workflow-run"
    assert set(manifest["source_evidence"]["market_metadata"]) == {"BTCUSDT", "ETHUSDT"}
    assert (
        manifest["producer_diagnostics"]["observation_grid"]
        == "independent_asset_observed_timestamps"
    )
    for payload in manifest["producer_diagnostics"]["reconstruction"].values():
        stats = ReconstructionStats.from_payload(payload)
        assert stats.emitted_states == 61
        assert any(
            start <= 0 and stop > 6_000_000_000
            for start, stop in stats.coverage_segments.values()
        )
    output = assemble(
        {
            "market_id": "crypto",
            "dataset_id": HFT_RESEARCH_DATASET_ID,
            "inputs": {key: str(value) for key, value in roots.items()},
            "time_window": window,
            "output_root": str(tmp_path / "assembled"),
        }
    )
    assert len(read_dataset(output)) == 122
    fractional = {
        "start": "1970-01-01T00:00:00.100001",
        "end": "1970-01-01T00:00:05.999999",
    }
    root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(raw)},
            "time_window": fractional,
            "output_root": str(tmp_path / "fractional"),
        }
    )
    frame = read_dataset(root, columns=("timestamp", "asset_id"))
    assert frame.timestamp.min() >= pd.Timestamp(fractional["start"])
    assert frame.timestamp.max() <= pd.Timestamp(fractional["end"])
    assert not frame.timestamp.eq(pd.Timestamp("1970-01-01T00:00:00.100000")).any()
    assert frame.groupby("timestamp").asset_id.nunique().eq(2).all()
    assert read_dataset_manifest_evidence(root)["row_count"] == len(frame)


@pytest.mark.parametrize("batch_rows", [2, 3, 4, 9])
@pytest.mark.parametrize("monotonic", [None, 2**53 + 1])
def test_independent_panel_persistence_preserves_source_nullable_dtypes(
    tmp_path, batch_rows, monotonic
):
    from dataclasses import replace

    from quant_research_platform.markets.implementations.crypto import hft
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.binance_hft import _coverage, coverage_lookup
    from tests.markets.crypto.test_hft import _state, _trade

    spec = catalog.hft_processed_catalog().get(HFT_PANEL_DATA_ID)
    inputs = {}
    for symbol, asset, frame_rows in (
        ("BTCUSDT", "btc_usdt", 2),
        ("ETHUSDT", "eth_usdt", 3),
    ):
        states = [
            replace(_state(stamp), receive_monotonic_ns=monotonic)
            for stamp in range(0, 500, 100)
        ]
        trades = [
            replace(
                _trade(200, 1.0, maker_buyer=False),
                source_sequence=2**53 + 1,
                receive_monotonic_ns=monotonic,
            )
        ]
        inputs[symbol] = tuple(
            frame.loc[:, list(spec.required_columns)]
            for frame in hft.iter_panel_frames(
                states,
                trades,
                market_id="crypto",
                asset_id=asset,
                start_ms=0,
                end_ms=400,
                batch_rows=frame_rows,
                book_coverage={"s1": _coverage(0, 501)},
                trade_coverage=coverage_lookup(
                    {"trade-connection-1": _coverage(0, 501)}
                ),
            )
        )
    frames = tuple(
        frame for symbol_frames in inputs.values() for frame in symbol_frames
    )
    expected_dtypes = inputs["BTCUSDT"][0].dtypes
    for frame in frames:
        pd.testing.assert_series_equal(frame.dtypes, expected_dtypes)
    merged = pd.concat(frames, ignore_index=True)
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    write_dataset(root, merged, spec=spec)
    restored = read_dataset(root)
    pd.testing.assert_frame_equal(
        restored.sort_values(["asset_id", "timestamp"]).reset_index(drop=True),
        merged.assign(timestamp=merged.timestamp.astype("datetime64[ns]"))
        .sort_values(["asset_id", "timestamp"])
        .reset_index(drop=True),
        check_exact=True,
    )
    for symbol, asset in (("BTCUSDT", "btc_usdt"), ("ETHUSDT", "eth_usdt")):
        expected = pd.concat(inputs[symbol], ignore_index=True)
        expected["timestamp"] = expected.timestamp.astype("datetime64[ns]")
        actual = restored.loc[restored.asset_id.eq(asset)].reset_index(drop=True)
        columns = [name for name in spec.required_columns if name != "segment_id"]
        pd.testing.assert_frame_equal(
            actual[columns], expected[columns], check_exact=True
        )
    assert restored.trade_source_sequence_min.dropna().tolist() == [2**53 + 1] * 2
    assert restored.book_receive_monotonic_ns.isna().all() == (monotonic is None)
