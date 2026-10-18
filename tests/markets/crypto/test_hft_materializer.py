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
    continuous = panels[0].select_dtypes(include="floating").columns
    exact = panels[0].columns.difference(continuous)
    pd.testing.assert_frame_equal(panels[0][exact], panels[1][exact], check_exact=True)
    pd.testing.assert_frame_equal(panels[0].isna(), panels[1].isna(), check_exact=True)
    pd.testing.assert_frame_equal(
        panels[0][continuous],
        panels[1][continuous],
        check_exact=False,
        rtol=2e-12,
        atol=1e-14,
    )


def test_hft_direct_preparation_assembly_and_fractional_window(tmp_path):
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.predictions.runner import predict
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
        else:
            request["parameters"] = {"asset_ids": ["eth_usdt", "btc_usdt"]}
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
    factor_base = assemble(
        {
            "market_id": "crypto",
            "dataset_id": "direct_crypto_factors",
            "inputs": {"factors": str(roots[HFT_FACTOR_DATA_ID])},
            "factor_columns": ["relative_spread"],
            "feature_recipe": None,
            "label_columns": [],
            "time_window": window,
            "output_root": str(tmp_path / "factor-base"),
        }
    )
    for path in (roots[HFT_FACTOR_DATA_ID], roots[HFT_LABEL_DATA_ID], factor_base):
        frame = read_dataset(path)
        pd.testing.assert_frame_equal(
            frame,
            frame.sort_values(["timestamp", "asset_id"], ignore_index=True),
            check_exact=True,
        )
    model = fit(
        {
            "market_id": "crypto",
            "inputs": str(output),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "direct-factor-model",
                "model_algorithm": "linear_regression",
                "input_factor_columns": ["relative_spread"],
                "target_column": "future_mid_return_100ms",
                "sample_filter_mode": "unfiltered",
                "random_seed": 7,
            },
            "fit_windows": [{"start": window["start"], "end": "1970-01-01T00:00:02"}],
            "fit_cutoff": "1970-01-01T00:00:03",
            "label_unit": "decimal_return",
        }
    )
    predictions, _ = read_predictions(
        predict(
            {
                "model": str(model),
                "inputs": str(roots[HFT_FACTOR_DATA_ID]),
                "prediction_window": {
                    "start": "1970-01-01T00:00:03.100",
                    "end": window["end"],
                },
                "output": str(tmp_path / "direct-factor-predictions"),
            }
        )
    )
    assert len(predictions) == 60
    assert predictions.groupby("timestamp").asset_id.nunique().eq(2).all()
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


@pytest.mark.parametrize("monotonic", [None, 2**53 + 1])
def test_independent_panel_persistence_preserves_source_nullable_dtypes(
    tmp_path, monotonic
):
    from dataclasses import replace

    from quant_research_platform.markets.implementations.crypto import hft
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.binance_hft import _coverage, coverage_lookup
    from tests.helpers.binance_hft import panel_state as _state
    from tests.helpers.binance_hft import panel_trade as _trade

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


@pytest.mark.parametrize("bid_levels,ask_levels", [(1, 1), (3, 7), (10, 10)])
def test_partial_depth_preserves_bbo_grid_and_local_persisted_dependencies(
    tmp_path, bid_levels, ask_levels
):
    import json

    from quant_research_platform.markets.prepare import prepare
    from tests.markets.crypto.test_independent_execution_source import _mutate

    raw = tmp_path / "raw"
    write_binance_hft_run(raw, count=12)

    def change(row, member):
        if member["symbol"] != "BTCUSDT" or member["data_kind"] != "depth":
            return
        payload = row["payload"]
        for side, best, count, sign in (
            ("b", 99.5, bid_levels, -1),
            ("a", 100.5, ask_levels, 1),
        ):
            if payload["T"] in (100, 600):
                quantity = "0" if payload["T"] == 100 else "2"
                payload[side].extend(
                    [f"{best + sign * index:.1f}", quantity]
                    for index in range(count, 10)
                )

    manifest = _mutate(raw, change)
    (raw / "run.json").write_text(json.dumps(manifest))
    # The output begins after the actual snapshot and removal event.
    window = {"start": "1970-01-01T00:00:00.200", "end": "1970-01-01T00:00:01.200"}
    panel_root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(raw)},
            "time_window": window,
            "output_root": str(tmp_path / "panel"),
        }
    )
    panel = read_dataset(panel_root)
    btc = panel.loc[panel.asset_id.eq("btc_usdt")].reset_index(drop=True)
    assert (
        btc.timestamp.tolist()
        == pd.date_range(window["start"], periods=11, freq="100ms").tolist()
    )
    assert btc.segment_id.nunique() == btc.book_observation_id.nunique() == 1
    assert btc.mid.notna().all()
    assert btc.trade_volume_100ms.eq(0.1).all()
    partial = btc.timestamp.lt(pd.Timestamp("1970-01-01T00:00:00.600"))
    for side, count in (("bid", bid_levels), ("ask", ask_levels)):
        for level in (1, 5, 10):
            column = f"{side}_depth_{level}"
            if count < level:
                assert btc.loc[partial, column].isna().all()
            else:
                assert btc.loc[partial, column].eq(2.0 * level).all()
            assert btc.loc[~partial, column].eq(2.0 * level).all()
    factors_root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_FACTOR_DATA_ID,
            "factor_columns": [
                "imbalance_1",
                "imbalance_5",
                "imbalance_10",
                "return_100ms",
                "signed_trade_flow_100ms",
            ],
            "inputs": {HFT_PANEL_DATA_ID: str(panel_root)},
            "time_window": window,
            "output_root": str(tmp_path / "factors"),
        }
    )
    factors = read_dataset(factors_root)
    factors = factors.loc[factors.asset_id.eq("btc_usdt")].reset_index(drop=True)
    assert factors.imbalance_1.eq(0.0).all()
    assert factors.return_100ms.iloc[1:].notna().all()
    assert factors.signed_trade_flow_100ms.eq(0.1).all()
    for level in (5, 10):
        if min(bid_levels, ask_levels) < level:
            assert factors.loc[partial, f"imbalance_{level}"].isna().all()
        assert factors.loc[~partial, f"imbalance_{level}"].eq(0.0).all()
    labels_root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_LABEL_DATA_ID,
            "label_columns": ["future_mid_return_100ms"],
            "inputs": {HFT_PANEL_DATA_ID: str(panel_root)},
            "time_window": window,
            "output_root": str(tmp_path / "labels"),
        }
    )
    labels = read_dataset(labels_root)
    labels = labels.loc[labels.asset_id.eq("btc_usdt")].reset_index(drop=True)
    expected = (btc.mid.shift(-1) - btc.mid) / btc.mid
    pd.testing.assert_series_equal(
        labels.future_mid_return_100ms,
        expected.rename("future_mid_return_100ms"),
        check_exact=False,
        rtol=2e-12,
        atol=1e-14,
    )
    assert (
        labels.future_mid_return_100ms_available_at.iloc[:-1].tolist()
        == btc.timestamp.iloc[1:].tolist()
    )
    assert pd.isna(labels.future_mid_return_100ms_available_at.iloc[-1])
