from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import producer
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.binance_hft import write_binance_hft_run
from tests.helpers.runtime_resources import capture_test_policy


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

    def eth_frames():  # noqa: ANN202
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

    def btc_frames():  # noqa: ANN202
        yield pd.DataFrame(
            [(pd.Timestamp("2024-01-01"), "btc_usdt", "source_segment", 1)],
            columns=columns,
        )
        exhausted.append("BTCUSDT")

    def eth_frames():  # noqa: ANN202
        exhausted.append("ETHUSDT")
        return
        yield  # pragma: no cover

    output = list(
        producer._common_symbol_panel_frames(
            {"BTCUSDT": btc_frames(), "ETHUSDT": eth_frames()},
            columns=columns,
            batch_rows=2,
        )
    )

    assert output == []
    assert exhausted == ["ETHUSDT", "BTCUSDT"]


@pytest.mark.parametrize(
    ("fail_transform", "empty_symbols"),
    [
        (False, ()),
        (False, ("BTCUSDT",)),
        (False, ("ETHUSDT",)),
        (False, ("BTCUSDT", "ETHUSDT")),
        (True, ()),
    ],
)
def test_panel_transform_waits_for_reservation_and_closes_source(
    tmp_path, monkeypatch, fail_transform, empty_symbols
):
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    frame = make_default_dataset_frame()
    root = make_dataset_artifact_root(tmp_path, "handshake")
    pulls = []
    closed = []
    primary = KeyboardInterrupt("transform_primary")

    def chunks(asset_session, **_kwargs):
        try:
            pulls.append("source")
            asset = asset_session.parquet_filters[-1][2]
            if asset not in {
                producer.canonical_crypto_asset_id(symbol) for symbol in empty_symbols
            }:
                yield frame
        finally:
            closed.append("source")

    def transform(source):
        try:
            for chunk in source:
                pulls.append("transform")
                if fail_transform:
                    raise primary
                yield chunk
        finally:
            closed.append("transform")

    policy = capture_test_policy(available_memory_bytes=128 * 1024**2)
    with runtime_resource_policy_scope(policy):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("handshake"),
            resource_policy=policy,
            borrowed_input=True,
        )
        with open_dataset_session(
            root, columns=None, time_window=None, resource_policy=policy
        ) as session:
            baseline = policy.used_memory_bytes
            monkeypatch.setattr(producer, "iter_dataset_chunks", chunks)
            output = producer._panel_frame_producer(
                session,
                transform=transform,
                columns=tuple(frame.columns),
                resource_policy=policy,
            )
            try:
                bound = next(output)
                assert pulls == []
                if fail_transform:
                    with policy.lease("test.panel.output", bound) as lease:
                        with pytest.raises(KeyboardInterrupt) as raised:
                            output.send(lease)
                        assert raised.value is primary
                    assert closed == ["transform", "source"]
                else:
                    emitted = 0
                    while True:
                        with policy.lease("test.panel.output", bound) as lease:
                            try:
                                owned = output.send(lease)
                            except StopIteration:
                                break
                            assert owned.lease is lease
                            pd.testing.assert_frame_equal(owned.value, frame)
                            owned = None
                            emitted += 1
                        try:
                            bound = output.send(None)
                        except StopIteration:
                            break
                    assert emitted == 2 - len(empty_symbols)
                    assert closed.count("transform") == closed.count("source") == 2
                assert policy.used_memory_bytes == baseline
            finally:
                output.close()
    assert policy.used_memory_bytes == 0


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
    with runtime_resource_policy_scope() as policy:
        sampled = read_dataset(
            fractional_root,
            columns=("timestamp", "asset_id"),
            mode=DatasetReadMode.MATERIALIZED,
            resource_policy=policy,
        )
        assert sampled.timestamp.min() >= pd.Timestamp("1970-01-01T00:00:00.100001")
        assert sampled.timestamp.max() <= pd.Timestamp("1970-01-01T00:00:05.999999")
        assert not sampled.timestamp.eq(
            pd.Timestamp("1970-01-01T00:00:00.100000")
        ).any()
        assert sampled.groupby("timestamp").asset_id.nunique().eq(2).all()
        fractional_manifest = read_dataset_manifest_evidence(fractional_root)
        assert fractional_manifest["row_count"] == len(sampled)
        assert (
            fractional_manifest["producer_diagnostics"]["observation_grid"]
            == "common_btc_eth_observed_timestamps"
        )
        sampled = None
        assert policy.used_memory_bytes == 0
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
    with runtime_resource_policy_scope() as policy:
        panel = read_dataset(
            root,
            columns=("timestamp", "asset_id"),
            mode=DatasetReadMode.MATERIALIZED,
            resource_policy=policy,
        )
        assert len(panel) == manifest["row_count"] == 122
        assert panel.groupby("timestamp").asset_id.nunique().eq(2).all()
        assert set(panel.timestamp) == set(
            pd.date_range("1970-01-01", periods=61, freq="100ms")
        )
        for values in stats.values():
            assert any(
                start <= 0 and stop > 6_000_000_000
                for start, stop in values.coverage_segments.values()
            )
        panel = None
        assert policy.used_memory_bytes == 0
    from copy import deepcopy

    from quant_research_platform.artifacts.research import read_research_run_record
    from tests.workflows.test_research_dag import _crypto_plan_payload

    plan = _crypto_plan_payload()
    request = dict(
        profile=profile,
        layout=layout,
        scenario_id=HFT_RESEARCH_DATASET_ID,
        raw_root=raw_root,
        start_date="1970-01-01T00:00:04",
        end_date="1970-01-01T00:00:06",
    )
    first = research.research_run(
        **{
            **request,
            "start_date": "1970-01-01T08:00:04+08:00",
            "end_date": "1970-01-01T08:00:06+08:00",
        },
        research_plan=plan,
    )
    assert first["status"] == "BLOCKED"
    first_record = read_research_run_record(layout.research_run_path(first["run_id"]))
    changed = deepcopy(plan)
    changed["initial_capital"] = 2000.0
    changed["final_holdout"] = {
        "start": "1970-01-01T00:00:08",
        "end": "1970-01-01T00:00:09",
    }
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

    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    with runtime_resource_policy_scope(policy):
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
        with pytest.raises(KeyboardInterrupt) as raised:
            research.data_materialize(
                profile=failure_profile,
                layout=layout,
                raw_root=raw_root,
                rebuild=True,
                **window,
            )
        assert raised.value is primary and policy.used_memory_bytes == 0
    assert read_dataset_manifest_evidence(root) == manifest
