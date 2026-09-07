from __future__ import annotations

import weakref
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.execution import (
    market_name_for_type,
)
from quant_research_platform.markets.implementations.a_share.processed import producer
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.portfolios import (
    production as portfolio_production,
)
from quant_research_platform.runtime.dataframes import iter_complete_session_frames
from quant_research_platform.runtime.resources import (
    MemoryLease,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.runtime_resources import capture_test_policy
from tests.markets.a_share.test_target_dag_chain import (
    _synthetic_financial_frames,
    _synthetic_interim_frames,
)


@pytest.mark.parametrize("boundary", ["soft", "captured", "explicit"])
def test_financial_event_builder_is_admitted_before_expansion(monkeypatch, boundary):
    source = _synthetic_financial_frames()
    dates = pd.bdate_range("2019-01-01", "2022-01-01")
    source["trading_calendar"] = pd.DataFrame(
        {"calendar_date": dates, "is_trading_day": True}
    )
    event_rows = sum(
        len(source[name]) * len(columns)
        for name, columns in producer.FINANCIAL_VALUE_COLUMNS.items()
    )
    bound = 1024**2 + event_rows * 1024
    policy = capture_test_policy(
        available_memory_bytes=bound - 1 if boundary == "captured" else bound * 5 // 4,
        explicit_cap_bytes=bound - 1 if boundary == "explicit" else None,
    )
    calls = []
    builder = producer._financial_pit_events

    def observed_builder(source):
        calls.append(True)
        events = builder(source)
        assert not events.empty
        return events

    monkeypatch.setattr(producer, "_processed_axis", lambda *args, **kwargs: ((), {}))
    monkeypatch.setattr(producer, "_financial_pit_events", observed_builder)
    with runtime_resource_policy_scope(policy):
        with (
            ExitStack() as retained,
            closing_iterator(
                producer._processed_frames(
                    research_profile().processed.get("fundamental_pit"),
                    source=source,
                    source_sessions={},
                    dependencies={"panel": object()},
                    policy=policy,
                    retained=retained,
                )
            ) as frames,
        ):
            if boundary == "soft":
                assert bound > policy.memory_budget_bytes
                assert list(frames) == []
                assert calls == [True]
            else:
                with pytest.raises(
                    ValueError,
                    match="a_share_financial_events_exceeds_task_memory_budget",
                ):
                    next(frames)
                assert calls == []
        assert policy.used_memory_bytes == 0


def test_persisted_panel_market_codes_reach_real_fee_mapping(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=66)
    source = _synthetic_interim_frames(dates)
    source["market_bars_1d"]["market_type"] = source["market_bars_1d"][
        "source_asset_id"
    ].map({"000001": 1.0, "000002": 4.0, "000003": 16.0})
    source["trading_calendar"] = pd.concat(
        [
            source["trading_calendar"].assign(source_market_type=code)
            for code in (1, 4, 16)
        ],
        ignore_index=True,
    )
    assert "market_type" not in source["stock_master"]
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    window = TimeWindow(dates[0], dates[-1])
    columns = ("market_id", "asset_id", "timestamp", "market_type")
    with runtime_resource_policy_scope(explicit_cap_bytes=128 * 1024**2) as policy:
        for data_id in spec.source_input_data_ids:
            source_spec = profile.interim.get(data_id)
            write_dataset(
                layout.interim_data_root(source_spec.source_id, data_id),
                source[data_id],
                spec=source_spec,
                resource_policy=policy,
                borrowed_input=True,
            )
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        with producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=window,
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
            resource_policy=policy,
        ) as (frames, metadata, verify):
            write_dataset(
                root,
                frames,
                spec=spec,
                resource_policy=policy,
                instance_metadata=metadata,
                staged_validator=verify,
            )
        with closing_iterator(
            portfolio_production._iter_input_sessions(root, columns, window, policy)
        ) as sessions:
            for timestamp in (dates[0], dates[64], dates[65]):
                with ExitStack() as retained:
                    panel = portfolio_production._read_next_session(
                        sessions, columns, timestamp, retained
                    ).set_index("asset_id")
                    assert {
                        asset: market_name_for_type(panel.loc[asset, "market_type"])
                        for asset in panel.index
                    } == {
                        "000001": "shanghai",
                        "000002": "shenzhen",
                        "000003": "shenzhen",
                    }
        assert policy.used_memory_bytes == 0


def test_processed_blocks_preserve_full_history_math_and_release_failed_inputs(
    tmp_path,
    monkeypatch,
):
    dates = pd.bdate_range("2020-01-02", periods=400)
    source = _synthetic_interim_frames(dates)
    source.update(_synthetic_financial_frames())
    disclosures = source["financial_first_disclosure"]
    disclosures["first_disclosure_date"] = disclosures["first_disclosure_date"].clip(
        lower=dates[80]
    )
    bars = source["market_bars_1d"]
    keep = ~(
        (
            bars["source_asset_id"].eq("000001")
            & bars["timestamp"].between(dates[66], dates[250])
        )
        | (
            bars["source_asset_id"].eq("000002")
            & bars["timestamp"].between(dates[1], dates[190])
        )
    )
    retained_rows = bars.loc[
        (bars["source_asset_id"].eq("000001") & bars["timestamp"].eq(dates[100]))
        | (bars["source_asset_id"].eq("000002") & bars["timestamp"].eq(dates[5]))
    ].copy()
    retained_rows["filling_flag"] = 1
    retained_rows["adj_close"] = 777.0
    bars = (
        pd.concat([bars.loc[keep], retained_rows], ignore_index=True)
        .sort_values(
            ["timestamp", "source_asset_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    first = bars["source_asset_id"].eq("000002") & bars["timestamp"].eq(dates[0])
    bars.loc[first, ["open", "high", "low", "close", "adj_open", "adj_close"]] = np.nan
    source["market_bars_1d"] = bars
    stock = source["stock_master"]
    stock.loc[stock["source_asset_id"].eq("000003"), "listing_date"] = dates[70]
    stock.loc[stock["source_asset_id"].eq("000003"), "delisting_date"] = dates[330]
    profile = research_profile()
    layout = StorageLayout(tmp_path)
    window = TimeWindow(pd.Timestamp("2019-01-01"), dates[-1])
    target_ids = (
        "market_panel_1d",
        "execution_mask",
        "forward_returns",
        "price_volume_facts",
        "fundamental_pit",
        "fundamental_factors",
    )

    compute = producer._compute_processed_frame
    materialize = producer.materialize_dataset_session_owned
    issued_inputs = []
    pit_rows = []

    def materialize_owned(*args, **kwargs):
        owned = materialize(*args, **kwargs)
        issued_inputs.append((weakref.ref(owned.value), owned.lease))
        return owned

    def owned_compute(spec, **kwargs):
        payloads = {
            id(frame): frame
            for frame in (*kwargs["source"].values(), *kwargs["dependencies"].values())
        }
        assert policy.used_memory_bytes >= sum(
            pandas_frame_payload_bytes(frame) for frame in payloads.values()
        )
        for frame in payloads.values():
            if not frame.empty:
                assert any(
                    reference() is frame and lease.active
                    for reference, lease in issued_inputs
                )
        if spec.data_id == "fundamental_pit":
            pit_rows.append(len(kwargs["dependencies"]["market_panel_1d"]))
        return compute(spec, **kwargs)

    with runtime_resource_policy_scope(explicit_cap_bytes=128 * 1024**2) as policy:
        for data_id, frame in source.items():
            spec = profile.interim.get(data_id)
            write_dataset(
                layout.interim_data_root(spec.source_id, data_id),
                frame,
                spec=spec,
                resource_policy=policy,
                borrowed_input=True,
            )
        source = {
            data_id: read_dataset(
                layout.interim_data_root(
                    profile.interim.get(data_id).source_id, data_id
                ),
                time_window=None
                if data_id in producer._FULL_HISTORY_SOURCE_IDS
                else window,
                resource_policy=policy,
            )
            for data_id in source
        }
        eager = {}
        for data_id in target_ids:
            spec = profile.processed.get(data_id)
            eager[data_id] = (
                producer._compute_processed_frame(
                    spec,
                    source=source,
                    dependencies={
                        key: eager[key] for key in spec.processed_input_data_ids
                    },
                    trading_sessions=tuple(dates),
                )
                .sort_values(["timestamp", "market_id", "asset_id"], kind="mergesort")
                .reset_index(drop=True)
            )

        monkeypatch.setattr(producer, "_compute_processed_frame", owned_compute)
        monkeypatch.setattr(
            producer, "materialize_dataset_session_owned", materialize_owned
        )
        for data_id in target_ids:
            spec = profile.processed.get(data_id)
            root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
            with producer.produce_a_share_processed_data(
                spec,
                layout=layout,
                time_window=window,
                processed_catalog=profile.processed,
                source_catalog=profile.interim,
                resource_policy=policy,
            ) as (frames, metadata, verify):
                write_dataset(
                    root,
                    frames,
                    spec=spec,
                    resource_policy=policy,
                    instance_metadata=metadata,
                    staged_validator=verify,
                )
            actual = read_dataset(root, resource_policy=policy)
            pd.testing.assert_frame_equal(actual, eager[data_id], check_exact=True)
            assert len(tuple((root / "data").rglob("*.parquet"))) > 1
            chunks = list(
                iter_complete_session_frames(
                    read_dataset(
                        root, mode=DatasetReadMode.CHUNKED, resource_policy=policy
                    ),
                    timestamp_column="timestamp",
                    resource_policy=policy,
                )
            )
            pd.testing.assert_frame_equal(
                pd.concat(chunks, ignore_index=True), eager[data_id], check_exact=True
            )
            assert policy.used_memory_bytes == 0
            assert all(not lease.active for _reference, lease in issued_inputs)
        assert max(pit_rows) <= 64 * 3
        mask = eager["execution_mask"]
        assert mask.loc[
            mask["asset_id"].eq("000001") & mask["timestamp"].eq(dates[63]), "can_enter"
        ].item()
        panel = eager["market_panel_1d"]
        reopened = panel.loc[
            panel["asset_id"].eq("000002") & panel["timestamp"].eq(dates[191])
        ]
        assert reopened["period_return"].item() == pytest.approx(
            reopened["adj_close"].item() / 777.0 - 1.0
        )

        references = []
        read = producer._owned_frame
        close = MemoryLease.close
        injected_cleanup = False
        primary = KeyboardInterrupt("synthetic_processed_primary")

        def tracked_read(*args):
            frame = read(*args)
            references.append(weakref.ref(frame))
            return frame

        def fail_compute(*args, **kwargs):
            raise primary

        def fail_cleanup(lease):
            nonlocal injected_cleanup
            close(lease)
            if lease.owner == "storage.materialized.output" and not injected_cleanup:
                injected_cleanup = True
                raise RuntimeError("synthetic_processed_cleanup")

        monkeypatch.setattr(producer, "_owned_frame", tracked_read)
        monkeypatch.setattr(producer, "_compute_processed_frame", fail_compute)
        monkeypatch.setattr(MemoryLease, "close", fail_cleanup)
        with pytest.raises(KeyboardInterrupt) as caught:
            with producer.produce_a_share_processed_data(
                profile.processed.get("fundamental_pit"),
                layout=layout,
                time_window=window,
                processed_catalog=profile.processed,
                source_catalog=profile.interim,
                resource_policy=policy,
            ):
                pytest.fail("failed computation must not reach publication")
        assert caught.value is primary
        assert injected_cleanup
        assert any("RuntimeError" in note for note in primary.__notes__)
        assert references and all(reference() is None for reference in references)
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "case",
    [
        "empty_window",
        "no_lifecycle",
        "holiday_fill",
        "closed_calendar",
        "late_lifecycle",
        "invalid_market_type",
    ],
)
def test_processed_window_boundaries_preserve_publish_state(tmp_path, case):
    dates = pd.bdate_range("2020-01-02", periods=130 if case == "late_lifecycle" else 4)
    frames = _synthetic_interim_frames(dates)
    if case == "no_lifecycle":
        frames["stock_master"]["listing_date"] = pd.Timestamp("2030-01-01")
    elif case == "late_lifecycle":
        frames["stock_master"]["listing_date"] = dates[70]
    elif case == "closed_calendar":
        frames["trading_calendar"]["is_trading_day"] = False
    elif case == "invalid_market_type":
        frames["market_bars_1d"]["market_type"] = 4.5
    elif case == "holiday_fill":
        bars = frames["market_bars_1d"]
        bars.loc[bars["timestamp"].eq(dates[-1]), "filling_flag"] = 2
    base = research_profile()

    def source_identity(_root, data_id):
        return (data_id.encode().hex() + "0" * 64)[:64]

    def interim_producer(**kwargs):
        (data_id,) = kwargs["data_ids"]
        spec = base.interim.get(data_id)
        root = kwargs["layout"].target_root(
            data_id,
            kwargs["layout"].interim_data_root(spec.source_id, data_id),
        )
        write_dataset(
            root,
            frames[data_id],
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={data_id: root},
            skipped=(),
            pruning_report_text="synthetic boundary",
        )

    profile = replace(
        base,
        producer_bindings={
            **base.producer_bindings,
            **{
                data_id: ProducerBinding(
                    ProducerKind.INTERIM,
                    interim_producer,
                    source_identity=source_identity,
                )
                for data_id in frames
            },
        },
    )
    layout = StorageLayout(tmp_path)

    def build():
        return research.dataset_build(
            profile=profile,
            layout=layout,
            dataset_id="pv_all",
            start_date="2030-01-01" if case == "empty_window" else str(dates[0].date()),
            end_date="2030-01-04" if case == "empty_window" else str(dates[-1].date()),
            raw_root=tmp_path / "synthetic-raw",
        )

    errors = {
        "holiday_fill": "holiday_fill_conflicts_with_open_session",
        "closed_calendar": "a_share_trading_calendar_empty",
        "invalid_market_type": "invalid_market_type",
    }
    if case in errors:
        with pytest.raises(ValueError, match=errors[case]):
            build()
    else:
        result = build()
        market = next(
            row
            for row in result["target_results"]
            if row["target_id"] == "market_panel_1d"
        )
        if case == "late_lifecycle":
            assert market["status"] == "BUILT"
        else:
            assert market["status"] == "BLOCKED"
            assert market["reason"] == "processed_target_empty"
    spec = profile.processed.get("market_panel_1d")
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    if case == "late_lifecycle":
        with runtime_resource_policy_scope() as policy:
            saved = read_dataset(
                layout.instance_root(root, market["instance_identity"]),
                resource_policy=policy,
            )
        assert len(saved) == 3 * (130 - 70)
        assert saved["timestamp"].min() == dates[70]
    else:
        assert not tuple(root.rglob("manifest.json"))
