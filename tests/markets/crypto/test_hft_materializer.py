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
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.binance_hft import write_binance_hft_run


def test_common_panel_alignment_ignores_asymmetric_frame_boundaries() -> None:
    columns = ("timestamp", "asset_id", "segment_id", "value")
    timestamps = pd.date_range("2024-01-01", periods=4, freq="100ms")
    eth_tail_consumed = []

    def frame(asset_id: str, positions: tuple[int, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": timestamps[position],
                    "asset_id": asset_id,
                    "segment_id": "source_segment",
                    "value": position,
                }
                for position in positions
            ],
            columns=columns,
        )

    def eth_frames():
        yield frame("eth_usdt", (0, 1, 2))
        yield frame("eth_usdt", (3,))
        eth_tail_consumed.append(True)

    output = list(
        producer._common_symbol_panel_frames(
            {
                "BTCUSDT": iter((frame("btc_usdt", (0,)), frame("btc_usdt", (1, 2)))),
                "ETHUSDT": eth_frames(),
            },
            columns=columns,
            batch_rows=4,
        )
    )
    combined = pd.concat(output, ignore_index=True)
    assert [len(frame) for frame in output] == [4, 2]
    assert combined.groupby("timestamp")["asset_id"].nunique().tolist() == [2, 2, 2]
    assert len(combined) == 6
    assert eth_tail_consumed == [True]


def test_common_panel_initial_empty_stream_still_drains_required_symbols() -> None:
    columns = ("timestamp", "asset_id", "segment_id", "value")
    exhausted = []

    def btc_frames():
        yield pd.DataFrame(
            [(pd.Timestamp("2024-01-01"), "btc_usdt", "source_segment", 1)],
            columns=columns,
        )
        exhausted.append("BTCUSDT")

    def eth_frames():
        exhausted.append("ETHUSDT")
        return
        yield

    output = list(
        producer._common_symbol_panel_frames(
            {"BTCUSDT": btc_frames(), "ETHUSDT": eth_frames()},
            columns=columns,
            batch_rows=2,
        )
    )
    assert output == []
    assert exhausted == ["ETHUSDT", "BTCUSDT"]


@pytest.mark.parametrize("batch_rows", (2, 4, 8))
def test_common_panel_gaps_start_new_segments_without_inventing_rows(batch_rows):
    from quant_research_platform.markets.implementations.crypto import hft
    from tests.helpers.binance_hft import _coverage, coverage_lookup
    from tests.markets.crypto.test_hft import _boundary, _state

    def panels(events, asset, coverage):
        return tuple(
            hft.iter_panel_frames(
                events,
                [],
                market_id="crypto",
                asset_id=asset,
                start_ms=0,
                end_ms=300,
                book_coverage=coverage,
                trade_coverage=coverage_lookup(
                    {"trade-connection-1": _coverage(0, 400)}
                ),
                batch_rows=4,
            )
        )

    btc = panels([_state(0), _state(300)], "btc_usdt", {"s1": _coverage(0, 400)})
    eth = panels(
        [
            _state(0),
            _boundary(100),
            _state(200, segment_id="s2"),
            _state(300, segment_id="s2"),
        ],
        "eth_usdt",
        {"s1": _coverage(0, 100), "s2": _coverage(200, 400)},
    )
    tuple(hft.iter_factor_frames(btc))
    tuple(hft.iter_factor_frames(eth))
    parts = tuple(
        producer._common_symbol_panel_frames(
            {"BTCUSDT": btc, "ETHUSDT": eth},
            columns=tuple(btc[0].columns),
            batch_rows=batch_rows,
        )
    )
    joined = pd.concat(parts, ignore_index=True)
    assert len(joined) == 4
    assert (
        joined.timestamp.tolist()
        == [pd.Timestamp(0, unit="ms")] * 2 + [pd.Timestamp(300, unit="ms")] * 2
    )
    for asset in ("btc_usdt", "eth_usdt"):
        frames = tuple(part.loc[part.asset_id.eq(asset)] for part in parts)
        factors = pd.concat(tuple(hft.iter_factor_frames(frames)), ignore_index=True)
        assert factors.return_100ms.isna().all()
        labels = pd.concat(
            tuple(hft.iter_label_frames(frames, horizons_ms=(100,))), ignore_index=True
        )
        assert labels.future_mid_return_100ms.isna().all()
        assert joined.loc[joined.asset_id.eq(asset), "segment_id"].nunique() == 2


def test_panel_batch_cap_preserves_persisted_rows(tmp_path, monkeypatch):
    from unittest.mock import patch

    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        research_profile,
    )
    from quant_research_platform.workflows import research

    raw_root = tmp_path / "raw"
    write_binance_hft_run(raw_root)
    profile = research_profile(CRYPTO_PROFILE_ID)
    panels = []
    for limit in (producer.DEFAULT_BATCH_ROWS, 3):
        monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", limit)
        layout = StorageLayout(tmp_path / str(limit))
        with patch.object(
            producer, "iter_panel_frames", wraps=producer.iter_panel_frames
        ) as symbol_frames:
            result = research.data_materialize(
                profile=profile,
                layout=layout,
                raw_root=raw_root,
                start_date="1970-01-01T00:00:00",
                end_date="1970-01-01T00:00:06",
                data_ids=(HFT_PANEL_DATA_ID,),
            )
            assert result["status"] == "COMPLETE", result["target_results"]
            assert symbol_frames.call_count == 2
            assert {
                call.kwargs["batch_rows"] for call in symbol_frames.call_args_list
            } == {limit // 2}
            target = result["target_results"][0]
            assert target["status"] == "BUILT"
            root = research._target_instance_root(
                profile, layout, HFT_PANEL_DATA_ID, target["instance_identity"]
            )
            panel = read_dataset(root).sort_values(
                ["timestamp", "asset_id"], ignore_index=True
            )
            assert len(panel) == 122
            assert not panel.duplicated(["timestamp", "asset_id"]).any()
            assert panel.groupby("timestamp").asset_id.nunique().eq(2).all()
            panels.append(panel)
    pd.testing.assert_frame_equal(panels[0], panels[1], check_exact=True)


def test_generic_hft_data_nodes_materialize_reuse_and_build_dataset(tmp_path):
    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        research_profile,
    )
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from quant_research_platform.workflows import research

    raw_root = tmp_path / "raw"
    write_binance_hft_run(raw_root)
    layout = StorageLayout(tmp_path / "project")
    profile = research_profile(CRYPTO_PROFILE_ID)
    window = dict(start_date="1970-01-01T00:00:00", end_date="1970-01-01T00:00:06")
    result = research.data_materialize(
        profile=profile, layout=layout, raw_root=raw_root, **window
    )
    assert result["status"] == "COMPLETE", result["target_results"]
    assert {row["target_id"] for row in result["target_results"]} == {
        HFT_PANEL_DATA_ID,
        HFT_FACTOR_DATA_ID,
        HFT_LABEL_DATA_ID,
    }
    targets = {row["target_id"]: row for row in result["target_results"]}
    panel_spec = profile.processed.get(HFT_PANEL_DATA_ID)
    root = layout.instance_root(
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
        targets[HFT_PANEL_DATA_ID]["instance_identity"],
    )
    manifest = read_dataset_manifest_evidence(root)
    assert manifest["row_count"] == 122
    assert manifest["source_artifact_identity"]["artifact_id"] == "workflow-run"
    assert set(manifest["producer_diagnostics"]["reconstruction"]) == {
        "BTCUSDT",
        "ETHUSDT",
    }
    assert all(
        value["emitted_states"] == 61
        for value in manifest["producer_diagnostics"]["reconstruction"].values()
    )
    again = research.data_materialize(
        profile=profile, layout=layout, raw_root=raw_root, **window
    )
    assert all(row["status"] == "REUSED" for row in again["target_results"])
    fractional = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        data_ids=(HFT_PANEL_DATA_ID,),
        start_date="1970-01-01T00:00:00.100001",
        end_date="1970-01-01T00:00:05.999999",
    )
    assert fractional["status"] == "COMPLETE"
    fractional_root = layout.instance_root(
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
        fractional["target_results"][0]["instance_identity"],
    )
    sampled = read_dataset(
        fractional_root,
        columns=("timestamp", "asset_id"),
        mode=DatasetReadMode.MATERIALIZED,
    )
    assert sampled.timestamp.min() >= pd.Timestamp("1970-01-01T00:00:00.100001")
    assert sampled.timestamp.max() <= pd.Timestamp("1970-01-01T00:00:05.999999")
    assert not sampled.timestamp.eq(pd.Timestamp("1970-01-01T00:00:00.100000")).any()
    assert sampled.groupby("timestamp").asset_id.nunique().eq(2).all()
    fractional_manifest = read_dataset_manifest_evidence(fractional_root)
    assert fractional_manifest["row_count"] == len(sampled)
    assert (
        fractional_manifest["producer_diagnostics"]["observation_grid"]
        == "common_btc_eth_observed_timestamps"
    )
    sampled = None
    built = research.dataset_build(
        profile=profile,
        layout=layout,
        dataset_id=HFT_RESEARCH_DATASET_ID,
        raw_root=raw_root,
        **window,
    )
    assert built["status"] == "COMPLETE", built["target_results"]
    assert [row["status"] for row in built["target_results"]].count("BUILT") == 1
    from contextlib import contextmanager

    from quant_research_platform.sources.binance.replay import ReconstructionStats

    assert (
        manifest["producer_diagnostics"]["observation_grid"]
        == "common_btc_eth_observed_timestamps"
    )
    stats = {
        symbol: ReconstructionStats.from_payload(value)
        for symbol, value in manifest["producer_diagnostics"]["reconstruction"].items()
    }
    panel = read_dataset(
        root, columns=("timestamp", "asset_id"), mode=DatasetReadMode.MATERIALIZED
    )
    assert len(panel) == manifest["row_count"] == 122
    assert panel.groupby("timestamp").asset_id.nunique().eq(2).all()
    assert set(panel.timestamp) == set(
        pd.date_range("1970-01-01", periods=61, freq="100ms")
    )
    for values in stats.values():
        assert any(
            (
                start <= 0 and stop > 6000000000
                for start, stop in values.coverage_segments.values()
            )
        )
    panel = None
    from copy import deepcopy

    from quant_research_platform.artifacts.research import read_research_run_record
    from tests.helpers.research_configuration import research_configuration
    from tests.workflows.test_research_dag import _crypto_plan_payload

    configuration = research_configuration(
        "crypto",
        "baseline",
        development=("1970-01-01T00:00:04", "1970-01-01T00:00:06"),
        plan=_crypto_plan_payload(),
    )
    configuration.update(
        stage="evaluation", models=[], portfolios=[], comparison_model_id=None
    )
    request = dict(profile=profile, layout=layout, raw_root=raw_root)
    offset_configuration = deepcopy(configuration)
    offset_configuration["windows"]["development"] = {
        "start": "1970-01-01T08:00:04+08:00",
        "end": "1970-01-01T08:00:06+08:00",
    }
    first = research.research_run(**request, research_plan=offset_configuration)
    assert first["status"] == "COMPLETE"
    first_record = read_research_run_record(layout.research_run_path(first["run_id"]))
    changed = deepcopy(configuration)
    changed["research_plan"]["initial_capital"] = 2000.0
    changed["research_plan"]["test"] = {
        "start": "1970-01-01T00:00:08",
        "end": "1970-01-01T00:00:09",
    }
    changed["windows"]["holdout"] = changed["research_plan"]["test"].copy()
    second = research.research_run(**request, research_plan=changed)
    second_record = read_research_run_record(layout.research_run_path(second["run_id"]))
    assert first_record.protocol_identity != second_record.protocol_identity
    for run in (first, second):
        data_results = {
            item["target_id"]: item
            for item in run["target_results"]
            if item["target_id"] in targets
        }
        assert all(item["status"] == "REUSED" for item in data_results.values())
        assert {
            key: item["instance_identity"] for key, item in data_results.items()
        } == {key: item["instance_identity"] for key, item in targets.items()}
    assert first_record.protocol["research_plan"]["initial_capital"] == 1000.0
    assert second_record.protocol["research_plan"]["initial_capital"] == 2000.0
    assert "research_plan" not in profile.protocol().manifest()
    primary = KeyboardInterrupt("raw_exit_verification")
    real_open = profile.producer_bindings[HFT_PANEL_DATA_ID].producer.keywords[
        "open_run"
    ]

    @contextmanager
    def failing_exit(*args, **kwargs):
        with real_open(*args, **kwargs) as opened:
            yield opened
            raise primary

    from dataclasses import replace
    from functools import partial

    binding = profile.producer_bindings[HFT_PANEL_DATA_ID]
    failure_profile = replace(
        profile,
        producer_bindings={
            **profile.producer_bindings,
            HFT_PANEL_DATA_ID: replace(
                binding, producer=partial(binding.producer, open_run=failing_exit)
            ),
        },
    )
    with pytest.raises(KeyboardInterrupt):
        research.data_materialize(
            profile=failure_profile,
            layout=layout,
            raw_root=raw_root,
            rebuild=True,
            **window,
        )
    assert read_dataset_manifest_evidence(root) == manifest


@pytest.mark.parametrize("batch_rows", [2, 3, 4, 9])
@pytest.mark.parametrize("monotonic", [None, 2**53 + 1])
def test_common_panel_persistence_preserves_source_nullable_dtypes(
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
        producer._common_symbol_panel_frames(
            inputs, columns=spec.required_columns, batch_rows=batch_rows
        )
    )
    expected_dtypes = inputs["BTCUSDT"][0].dtypes
    for frame in frames:
        pd.testing.assert_series_equal(frame.dtypes, expected_dtypes)
    merged = pd.concat(frames, ignore_index=True)
    layout = StorageLayout(tmp_path)
    root = layout.target_root(
        spec.data_id,
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
    )
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
