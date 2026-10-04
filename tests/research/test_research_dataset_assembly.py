from __future__ import annotations

import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.research.datasets.assembler as assembler_module
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    validate_evaluation_input,
    validate_research_input,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_processed_inputs import (
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import (
    make_research_dataset_catalog,
)
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
    capture_test_policy,
)
from tests.research.workflow_test_support import (
    _passthrough_session,
    _research_rows,
    _research_rows_for_dates,
)


def test_research_dataset_assembles_from_processed_inputs(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    root = assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        ),
        ignore_index=True,
    )
    inspection = inspect_dataset(root)

    assert root == tmp_path / "data" / "research" / "a_share" / "demo"
    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "momentum_20d",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert len(result) == 4
    assert set(inspection.manifest["input_dataset_identities"]) == {
        "demo_factors",
        "demo_labels",
        "demo_annotations",
    }
    validate_evaluation_input(
        result,
        dataset_catalog.get("demo"),
        signal_columns=("momentum_20d",),
        label_columns=("forward_return_20d",),
    )
    validate_research_input(
        result,
        dataset_catalog.get("demo"),
        input_columns=("momentum_20d",),
    )


def test_research_join_bound_admission_precedes_join_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024,
        explicit_cap_bytes=32 * 1024,
    )
    time_key_pulls: list[bool] = []
    pulled: list[bool] = []

    def fake_iter_dataset_chunks(*_args, **_kwargs):  # noqa: ANN002, ANN003
        time_key_pulls.append(True)
        yield pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]})

    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        fake_iter_dataset_chunks,
    )
    monkeypatch.setattr(
        assembler_module,
        "_iter_research_chunk_windows_from_base_input",
        lambda *_args, **_kwargs: iter((object(),)),
    )
    monkeypatch.setattr(
        assembler_module,
        "_join_processed_inputs",
        lambda **_kwargs: (pulled.append(True), pd.DataFrame({"id": [1]}))[1],
    )
    dataset_spec = SimpleNamespace(
        primary_key=("id",),
        feature_transforms=(),
        input_data_ids=("base",),
        base_input_data_id="base",
    )
    sessions = {
        "base": _passthrough_session(
            retained_memory_bytes=0,
            path_selection=None,
            close=lambda: None,
        )
    }
    chunks = assembler_module._iter_joined_processed_chunks(
        dataset_spec=dataset_spec,
        processed_catalog=SimpleNamespace(get=lambda _key: object()),
        sessions=sessions,
        time_window=object(),
    )

    with runtime_resource_policy_scope(policy):
        blocker = policy.lease("test.research.blocker", policy.memory_budget_bytes)
        try:
            bound = next(chunks)
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                policy.lease("test.research.consumer", bound)
            assert time_key_pulls == []
            assert pulled == []
        finally:
            blocker.close()


def test_research_time_window_driver_uses_one_window_per_physical_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_targets: list[int] = []
    timestamps = pd.date_range("2026-01-01", periods=20, freq="min")
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )
    session = _passthrough_session(path_selection=None, close=lambda: None)

    def fake_iter_dataset_chunks(_context, *, target_chunk_bytes):  # noqa: ANN001
        observed_targets.append(target_chunk_bytes)
        yield pd.DataFrame({"timestamp": timestamps})

    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        fake_iter_dataset_chunks,
        raising=False,
    )

    with runtime_resource_policy_scope(policy):
        result = list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                session,
                time_window=TimeWindow(
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:19:00",
                ),
            )
        )

    assert result == [
        TimeWindow(
            "2026-01-01T00:00:00",
            "2026-01-01T00:19:00",
        )
    ]
    assert observed_targets == [policy.planning_headroom_bytes]


def test_research_time_window_driver_preserves_single_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 240,
    )
    session = _passthrough_session(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: (
            frame
            for frame in (pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]}),)
        ),
    )

    with runtime_resource_policy_scope(policy):
        assert list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                session,
                time_window=TimeWindow(
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:00",
                ),
            )
        ) == [TimeWindow("2026-01-01T00:00:00", "2026-01-01T00:00:00")]


def test_research_time_window_driver_normalizes_and_deduplicates_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=128 * 1024)
    session = _passthrough_session(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (
                pd.DataFrame(
                    {
                        "timestamp": [
                            pd.Timestamp("2026-01-02 15:00"),
                            pd.Timestamp("2026-01-01 09:00"),
                        ]
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": [
                            pd.Timestamp("2026-01-02 09:00"),
                            pd.Timestamp("2026-01-03 09:00"),
                        ]
                    }
                ),
            )
        ),
    )

    with runtime_resource_policy_scope(policy):
        result = list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                session,
                time_window=TimeWindow("2026-01-01", "2026-01-03"),
            )
        )

    assert result == [
        TimeWindow("2026-01-01", "2026-01-02"),
        TimeWindow("2026-01-03", "2026-01-03"),
    ]


def test_research_time_window_driver_rejects_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=128 * 1024)
    session = _passthrough_session(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]")}),)
        ),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="no_research_dataset_sessions"):
            list(
                assembler_module._iter_research_chunk_windows_from_base_input(
                    session,
                    time_window=TimeWindow("2026-01-01", "2026-01-01"),
                )
            )


def test_research_time_window_driver_rejects_non_monotonic_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=128 * 1024)
    session = _passthrough_session(path_selection=None, close=lambda: None)
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (
                pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-02")]}),
                pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]}),
            )
        ),
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="non_monotonic_research_time_keys"):
            list(
                assembler_module._iter_research_chunk_windows_from_base_input(
                    session,
                    time_window=TimeWindow("2026-01-01", "2026-01-02"),
                )
            )


def test_research_time_window_driver_closes_iterator_before_view_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )

    view = SimpleNamespace(path_selection=1)

    @contextmanager
    def filtered_view(_time_window):
        try:
            yield view
        finally:
            events.extend(("selection_release", "view_close"))

    base_session = SimpleNamespace(filtered_view=filtered_view)

    def chunks():
        try:
            yield pd.DataFrame({"not_timestamp": [1]})
        finally:
            events.append("iterator_close")

    iterator = chunks()
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iterator,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyError, match="timestamp"):
            list(
                assembler_module._iter_research_chunk_windows_from_base_input(
                    base_session,
                    time_window=TimeWindow("2026-01-01", "2026-01-01"),
                )
            )
    assert events == ["iterator_close", "selection_release", "view_close"]


def test_research_processed_input_uses_live_budget_for_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_catalog = make_demo_processed_catalog()
    dataset_spec = make_research_dataset_catalog().get("demo")
    processed_spec = processed_catalog.get("demo_factors")
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 9216,
    )
    observed: dict[str, object] = {}
    columns = assembler_module._processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=(),
    )
    context = _passthrough_session(
        logical_columns=columns,
        path_selection=None,
        close=lambda: None,
    )

    def materialize(context, **kwargs):  # noqa: ANN001
        observed["context"] = context
        observed.update(kwargs)
        return OwnedValue(
            pd.DataFrame(),
            policy.lease("test.research.input", 0),
        )

    monkeypatch.setattr(
        assembler_module,
        "materialize_dataset_session_owned",
        materialize,
    )

    with runtime_resource_policy_scope(policy):
        owned = assembler_module._read_processed_input(
            dataset_spec=dataset_spec,
            processed_spec=processed_spec,
            session=context,
            existing_columns=(),
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
        )
        owned.lease.close()

    assert observed == {
        "context": context,
        "target_chunk_bytes": policy.planning_headroom_bytes,
    }


def test_research_join_preflights_merged_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 750,
    )
    base = pd.DataFrame({"id": [1], "left": [1]})
    right = pd.DataFrame({"id": [1], "right": [1]})
    frames = iter((base, right))
    monkeypatch.setattr(
        assembler_module,
        "_ordered_input_data_ids",
        lambda *_args, **_kwargs: ("left", "right"),
    )
    monkeypatch.setattr(
        assembler_module,
        "_read_processed_input",
        lambda **_kwargs: OwnedValue(
            next(frames),
            policy.lease("test.research.input", 0),
        ),
    )
    monkeypatch.setattr(
        assembler_module,
        "pandas_frame_payload_bytes",
        lambda frame: {
            ("id",): 100,
            ("id", "left"): 200,
            ("id", "right"): 151,
        }[tuple(frame.columns)],
    )

    with runtime_resource_policy_scope(policy):
        blocking_lease = policy.lease(
            "test.research.merge.capacity",
            policy.memory_budget_bytes - 700,
        )
        try:
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                assembler_module._join_processed_inputs(
                    dataset_spec=SimpleNamespace(primary_key=("id",)),
                    processed_catalog=SimpleNamespace(get=lambda _key: object()),
                    sessions={
                        "left": SimpleNamespace(),
                        "right": SimpleNamespace(),
                    },
                    time_window=object(),
                )
        finally:
            blocking_lease.close()


def test_research_join_clears_transient_chunk_before_next_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    holder = {
        "frame": pd.DataFrame(
            {"timestamp": [pd.Timestamp("2024-01-02")], "id": [1], "value": [1.0]}
        ),
    }
    frame_ref = weakref.ref(holder["frame"])
    monkeypatch.setattr(
        assembler_module,
        "_base_input_data_id",
        lambda *_args, **_kwargs: "one",
    )
    monkeypatch.setattr(
        assembler_module,
        "_iter_research_chunk_windows_from_base_input",
        lambda *_args, **_kwargs: iter((object(), object())),
    )
    monkeypatch.setattr(
        assembler_module,
        "_join_processed_inputs",
        lambda **_kwargs: holder["frame"],
    )
    monkeypatch.setattr(
        assembler_module,
        "validate_research_dataset",
        lambda *_args, **_kwargs: None,
    )

    chunks = assembler_module._iter_joined_processed_chunks(
        dataset_spec=SimpleNamespace(primary_key=("timestamp", "id")),
        processed_catalog=SimpleNamespace(get=lambda _key: object()),
        sessions={"one": SimpleNamespace(retained_memory_bytes=0)},
        time_window=object(),
    )
    with runtime_resource_policy_scope(policy):
        bound = next(chunks)
        lease = policy.lease("test.research.consumer", bound)
        owned = chunks.send(lease)
        assert policy.used_memory_bytes == lease.reserved_bytes
        holder["frame"] = None
        del owned
        next_bound = chunks.send(None)
        assert type(next_bound) is int and next_bound > 0
        assert frame_ref() is None
        lease.close()
        chunks.close()


def test_research_dataset_assembles_intraday_timestamp_window(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    primary_key = PLATFORM_ASSET_TIME_KEY
    factor_spec = ProcessedDataSpec(
        data_id="minute_factors",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "minute_factor": ColumnRole.FACTOR,
        },
        primary_key=primary_key,
        partition_columns=("year",),
    )
    label_spec = ProcessedDataSpec(
        data_id="minute_labels",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.LABEL,
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "minute_label": ColumnRole.LABEL,
        },
        primary_key=primary_key,
        partition_columns=("year",),
    )
    processed_catalog = ProcessedDataCatalog((factor_spec, label_spec))
    dataset_catalog = make_research_dataset_catalog(
        data_id="minute_research",
        market_id="crypto",
        input_data_ids=("minute_factors", "minute_labels"),
        base_input_data_id="minute_factors",
        factor_columns=("minute_factor",),
        label_columns=("minute_label",),
        filter_columns=(),
        context_columns=(),
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
    )
    timestamps = pd.to_datetime(
        [
            "2026-07-06T00:00:00",
            "2026-07-06T00:01:00",
            "2026-07-06T00:02:00",
        ]
    )
    keys = pd.DataFrame(
        [
            {
                "market_id": "crypto",
                "asset_id": asset_id,
                "timestamp": timestamp,
            }
            for timestamp in timestamps
            for asset_id in ("asset_a", "asset_b")
        ]
    )
    for spec, value_column in (
        (factor_spec, "minute_factor"),
        (label_spec, "minute_label"),
    ):
        frame = keys.assign(**{value_column: range(len(keys))})
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame,
            spec=spec,
        )

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="minute_research",
        time_window=TimeWindow(
            "2026-07-06T00:01:00",
            "2026-07-06T00:02:00",
        ),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("crypto"),
            dataset_id="minute_research",
            time_window=TimeWindow(
                "2026-07-06T00:01:00",
                "2026-07-06T00:02:00",
            ),
        ),
        ignore_index=True,
    )

    assert len(result) == 4
    assert sorted(result["timestamp"].unique()) == list(timestamps[1:])


def test_research_dataset_assembler_generates_feature_transforms_from_factors(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id="demo",
                market_id=MarketId("a_share"),
                input_data_ids=("demo_factors", "demo_labels", "demo_annotations"),
                base_input_data_id="demo_annotations",
                factor_columns=("momentum_20d",),
                feature_columns=("momentum_20d_zscore",),
                feature_transforms=(
                    FeatureTransformSpec(
                        transform_id="momentum_20d_zscore",
                        source_columns=("momentum_20d",),
                        output_columns=("momentum_20d_zscore",),
                        transform="cross_sectional_zscore",
                    ),
                ),
                label_columns=("forward_return_20d",),
                filter_columns=("eligible", "can_enter", "can_exit"),
                context_columns=("market_type",),
                frequency=DataFrequency.DAILY,
                grain=DataGrain.RESEARCH_ROW,
                column_roles={
                    "market_id": ColumnRole.KEY,
                    "asset_id": ColumnRole.KEY,
                    "timestamp": ColumnRole.TIMESTAMP,
                    "momentum_20d": ColumnRole.FACTOR,
                    "momentum_20d_zscore": ColumnRole.FEATURE,
                    "forward_return_20d": ColumnRole.LABEL,
                    "eligible": ColumnRole.UNIVERSE,
                    "can_enter": ColumnRole.MASK,
                    "can_exit": ColumnRole.MASK,
                    "market_type": ColumnRole.CONTEXT,
                },
                primary_key=PLATFORM_ASSET_TIME_KEY,
                partition_columns=("year",),
            ),
        )
    )
    write_demo_processed_inputs(layout, processed_catalog)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        ),
        ignore_index=True,
    ).sort_values(["timestamp", "asset_id"])

    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "momentum_20d",
        "momentum_20d_zscore",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert result["momentum_20d_zscore"].tolist() == pytest.approx(
        [-1.0, 1.0, 1.0, -1.0]
    )
    validate_research_input(
        result,
        dataset_catalog.get("demo"),
        input_columns=("momentum_20d", "momentum_20d_zscore"),
    )


def test_assembly_keeps_nanosecond_windows_through_persisted_inputs(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(tmp_path)
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
    }
    source = ProcessedDataSpec(
        data_id="nanosecond_input",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.TICK,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
    dataset = make_research_dataset_catalog(
        data_id="nanosecond_research",
        market_id="crypto",
        input_data_ids=(source.data_id,),
        base_input_data_id=source.data_id,
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=(),
        context_columns=(),
        frequency=DataFrequency.TICK,
    )
    times = pd.Timestamp("2020-01-01") + pd.to_timedelta([1, 2], unit="ns")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "asset",
            "timestamp": times,
            "factor": [1.0, 2.0],
            "label": [3.0, 4.0],
        }
    )
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        frame,
        spec=source,
    )
    window = TimeWindow(times[0], times[-1])
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset,
        processed_catalog=ProcessedDataCatalog((source,)),
        dataset_id="nanosecond_research",
        time_window=window,
    )
    result = pd.concat(
        tuple(
            iter_research_dataset_chunks(
                layout,
                market_id=MarketId("crypto"),
                dataset_id="nanosecond_research",
                time_window=window,
            )
        ),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(result, frame, check_exact=True)


def test_custom_feature_spec_survives_context_and_persisted_assembly(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
        "eligible": ColumnRole.UNIVERSE,
        "industry_level1_code": ColumnRole.CONTEXT,
        "log_float_market_cap": ColumnRole.CONTEXT,
    }
    source = ProcessedDataSpec(
        data_id="custom_inputs",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
    datasets = make_research_dataset_catalog(
        data_id="custom_rank",
        input_data_ids=(source.data_id,),
        base_input_data_id=source.data_id,
        factor_columns=("factor",),
        feature_columns=("rank_1",),
        feature_transforms=(
            FeatureTransformSpec(
                transform_id="rank_1",
                source_columns=("factor",),
                output_columns=("rank_1",),
                transform="cross_sectional_rank",
            ),
        ),
        label_columns=("label",),
        filter_columns=("eligible",),
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        context_columns=("industry_level1_code", "log_float_market_cap"),
    )
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2020-01-03"] * 3),
            "factor": [1.0, 2.0, 3.0],
            "label": 0.0,
            "eligible": True,
            "industry_level1_code": "A",
            "log_float_market_cap": [1.0, 2.0, 3.0],
        }
    )
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        frame,
        spec=source,
    )
    policy = capture_storage_test_policy(working_headroom_bytes=16 * 1024**2)
    with runtime_resource_policy_scope(policy):
        window = TimeWindow("2020-01-03", "2020-01-03")
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=datasets,
            processed_catalog=ProcessedDataCatalog((source,)),
            dataset_id="custom_rank",
            time_window=window,
            resource_policy=policy,
        )
        result = pd.concat(
            iter_research_dataset_chunks(
                layout,
                market_id=MarketId("a_share"),
                dataset_id="custom_rank",
                time_window=window,
            ),
            ignore_index=True,
        )
        assert result["rank_1"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
        assert not any(column.startswith("feature_") for column in result)
        assert policy.used_memory_bytes == 0


def test_research_dataset_session_complete_chunks_preserve_output(
    tmp_path: Path,
) -> None:
    frame = _research_rows_for_dates(
        [
            "2014-01-02",
            "2014-01-03",
            "2014-01-06",
            "2014-01-07",
            "2014-01-08",
        ]
    )
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(default_layout, processed_catalog, frame=frame)
    write_demo_processed_inputs(tiny_layout, processed_catalog, frame=frame)

    default_root = assemble_research_dataset(
        layout=default_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    tiny_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    tiny_root = assemble_research_dataset(
        layout=tiny_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        resource_policy=tiny_policy,
    )

    default = pd.concat(
        iter_research_dataset_chunks(
            default_layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])
    tiny = pd.concat(
        iter_research_dataset_chunks(
            tiny_layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-08"),
        ),
        ignore_index=True,
    ).sort_values(["market_id", "asset_id", "timestamp"])

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
    assert (
        inspect_dataset(tiny_root).file_count
        == inspect_dataset(default_root).file_count
    )


def test_research_chunk_view_prunes_top_level_multi_year_physical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    frame = _research_rows_for_dates(["2014-12-31", "2015-01-02"])
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    processed_spec = processed_catalog.get("demo_factors")
    dataset_spec = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    ).get("demo")
    columns = assembler_module._processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=(),
    )
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    with assembler_module.open_dataset_session(
        layout.processed_data_root(
            processed_spec.market_id,
            processed_spec.kind,
            processed_spec.data_id,
        ),
        columns=columns,
        time_window=TimeWindow("2014-12-31", "2015-01-02"),
        resource_policy=policy,
    ) as session:
        with runtime_resource_policy_scope(policy):
            with session.filtered_view(TimeWindow("2015-01-02", "2015-01-02")) as view:
                assert len(view.read_paths) == 1
                assert view.read_paths[0].parent.name == "year=2015"
                assert view.path_selection is not None
                view_selection = view.path_selection
        assert len(session.read_paths) == 2
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=view_selection,
            )
            == 0
        )

        observed_selections: list[int] = []

        def materialize(current, **_kwargs):  # noqa: ANN001
            assert len(current.read_paths) == 1
            assert current.path_selection is not None
            observed_selections.append(current.path_selection)
            return OwnedValue(
                pd.DataFrame(),
                policy.lease("test.research.input", 0),
            )

        monkeypatch.setattr(
            assembler_module,
            "materialize_dataset_session_owned",
            materialize,
        )
        with runtime_resource_policy_scope(policy):
            owned = assembler_module._read_processed_input(
                dataset_spec=dataset_spec,
                processed_spec=processed_spec,
                session=session,
                existing_columns=(),
                time_window=TimeWindow("2014-12-31", "2014-12-31"),
            )
            owned.lease.close()
        normal_selection = observed_selections.pop()
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=normal_selection,
            )
            == 0
        )

        def fail_materialization(current, **_kwargs):  # noqa: ANN001
            assert current.path_selection is not None
            observed_selections.append(current.path_selection)
            raise RuntimeError("research_test_materialization_failure")

        monkeypatch.setattr(
            assembler_module,
            "materialize_dataset_session_owned",
            fail_materialization,
        )
        with runtime_resource_policy_scope(policy):
            with pytest.raises(
                RuntimeError,
                match="research_test_materialization_failure",
            ):
                assembler_module._read_processed_input(
                    dataset_spec=dataset_spec,
                    processed_spec=processed_spec,
                    session=session,
                    existing_columns=(),
                    time_window=TimeWindow("2015-01-02", "2015-01-02"),
                )
        error_selection = observed_selections.pop()
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=error_selection,
            )
            == 0
        )


def test_research_assembly_rejects_dependency_replacement_after_identity_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    write_demo_processed_inputs(
        replacement_layout,
        processed_catalog,
        frame=_research_rows_for_dates(["2014-01-02", "2014-01-03", "2014-01-04"]),
    )
    factor_spec = processed_catalog.get("demo_factors")
    current_root = layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    replacement_root = replacement_layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    real_identity = assembler_module.dataset_identity_payload
    real_open = assembler_module.open_dataset_session
    opened_roots: list[Path] = []
    replaced = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def replace_after_identity(spec, manifest):  # noqa: ANN001
        nonlocal replaced
        identity = real_identity(spec, manifest)
        if spec["data_id"] == factor_spec.data_id and not replaced:
            current_root.rename(backup_root)
            replacement_root.rename(current_root)
            replaced = True
        return identity

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(
        assembler_module,
        "dataset_identity_payload",
        replace_after_identity,
    )

    with pytest.raises(
        ValueError,
        match="dataset_(spec|manifest)_identity_changed",
    ):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )

    assert replaced
    assert len(opened_roots) == len(processed_catalog.list())
    assert len(opened_roots) == len(set(opened_roots))

    reopened_policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    with runtime_resource_policy_scope(reopened_policy):
        assert current_runtime_resource_policy() is reopened_policy


def test_research_assembly_verifies_every_input_inside_staged_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    real_open = assembler_module.open_dataset_session
    real_write = assembler_module.write_dataset
    real_verify = assembler_module.DatasetSession.verify
    opened_roots: list[Path] = []
    opened_sessions: list[assembler_module.DatasetSession] = []
    staged_verified_roots: list[Path] = []
    post_publish_verified_roots: list[Path] = []
    inside_staged_validator = False
    writer_returned = False

    @contextmanager
    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        with real_open(root, **kwargs) as session:
            opened_sessions.append(session)
            yield session

    def record_verify(session) -> None:  # noqa: ANN001
        if inside_staged_validator:
            staged_verified_roots.append(session.root)
        elif writer_returned and session.root in opened_roots:
            post_publish_verified_roots.append(session.root)
        real_verify(session)

    def record_write(root, data, *, staged_validator, **kwargs):  # noqa: ANN001
        nonlocal inside_staged_validator, writer_returned
        assert staged_validator is not None
        assert "retained_payload_bytes" not in kwargs

        def record_staged(receipt) -> None:  # noqa: ANN001
            nonlocal inside_staged_validator
            inside_staged_validator = True
            try:
                staged_validator(receipt)
            finally:
                inside_staged_validator = False

        result = real_write(
            root,
            data,
            staged_validator=record_staged,
            **kwargs,
        )
        writer_returned = True
        return result

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(assembler_module.DatasetSession, "verify", record_verify)
    monkeypatch.setattr(assembler_module, "write_dataset", record_write)

    artifact_root = assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )

    assert artifact_root.is_dir()
    assert len(opened_roots) == len(processed_catalog.list())
    assert len(opened_roots) == len(set(opened_roots))
    assert staged_verified_roots == opened_roots
    assert post_publish_verified_roots == []


def test_research_assembly_rejects_replacement_before_publish_and_cleans_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    write_demo_processed_inputs(layout, processed_catalog)
    write_demo_processed_inputs(
        replacement_layout,
        processed_catalog,
        frame=_research_rows_for_dates(["2014-01-02", "2014-01-03", "2014-01-04"]),
    )
    factor_spec = processed_catalog.get("demo_factors")
    current_root = layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    replacement_root = replacement_layout.processed_data_root(
        factor_spec.market_id,
        factor_spec.kind,
        factor_spec.data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    artifact_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    real_open = assembler_module.open_dataset_session
    real_joined_chunks = assembler_module._iter_joined_processed_chunks
    real_write = assembler_module.write_dataset
    opened_roots: list[Path] = []
    replaced = False
    staged_validator_supplied = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def replace_after_last_input(**kwargs):  # noqa: ANN003, ANN202
        nonlocal replaced
        yield from real_joined_chunks(**kwargs)
        current_root.rename(backup_root)
        replacement_root.rename(current_root)
        replaced = True

    def record_write(root, data, *, staged_validator, **kwargs):  # noqa: ANN001
        nonlocal staged_validator_supplied
        staged_validator_supplied = staged_validator is not None
        return real_write(
            root,
            data,
            staged_validator=staged_validator,
            **kwargs,
        )

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(
        assembler_module,
        "_iter_joined_processed_chunks",
        replace_after_last_input,
    )
    monkeypatch.setattr(assembler_module, "write_dataset", record_write)

    with pytest.raises(
        ValueError,
        match="dataset_(spec|manifest)_identity_changed",
    ):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )

    assert replaced
    assert staged_validator_supplied
    assert len(opened_roots) == len(processed_catalog.list())
    assert len(opened_roots) == len(set(opened_roots))
    assert not artifact_root.exists()
    assert list(artifact_root.parent.glob(".a-*")) == []


def test_sparse_research_dataset_left_joins_missing_non_base_input_key(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        key_coverage={"demo_labels": "sparse"},
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    frame = _research_rows()
    labels = frame.drop(index=frame.index[-1]).reset_index(drop=True)
    write_demo_processed_inputs(layout, processed_catalog, label_frame=labels)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id="demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        ),
        ignore_index=True,
    )
    missing_row = result.loc[
        result["asset_id"].eq("000002")
        & result["timestamp"].eq(pd.Timestamp("2014-01-03"))
    ].iloc[0]

    assert len(result) == 4
    assert pd.isna(missing_row["forward_return_20d"])


def test_research_dataset_role_validation_rejects_wrong_factor_source(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id="bad_demo",
                market_id=MarketId("a_share"),
                input_data_ids=("demo_labels", "demo_annotations"),
                base_input_data_id="demo_annotations",
                factor_columns=("market_type",),
                label_columns=("forward_return_20d",),
                frequency=DataFrequency.DAILY,
                grain=DataGrain.RESEARCH_ROW,
                column_roles={
                    "market_id": ColumnRole.KEY,
                    "asset_id": ColumnRole.KEY,
                    "timestamp": ColumnRole.TIMESTAMP,
                    "market_type": ColumnRole.FACTOR,
                    "forward_return_20d": ColumnRole.LABEL,
                },
                primary_key=PLATFORM_ASSET_TIME_KEY,
                partition_columns=("year",),
            ),
        )
    )
    write_demo_processed_inputs(layout, processed_catalog)

    with pytest.raises(ValueError, match="invalid_research_factor_columns"):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="bad_demo",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_column", "expected_error"),
    [
        ("label_columns", "momentum_20d", "research_input_column_overlap"),
        ("filter_columns", "market_type", "research_semantic_column_role_invalid"),
        ("context_columns", "eligible", "research_semantic_column_role_invalid"),
    ],
)
def test_research_dataset_role_validation_rejects_wrong_label_filter_and_context(
    field_name: str,
    bad_column: str,
    expected_error: str,
) -> None:
    spec_kwargs = {
        "factor_columns": ("momentum_20d",),
        "label_columns": ("forward_return_20d",),
        "filter_columns": ("eligible", "can_enter", "can_exit"),
        "context_columns": ("market_type",),
    }
    spec_kwargs[field_name] = (bad_column,)
    with pytest.raises(ValueError, match=expected_error):
        ResearchDatasetCatalog(
            (
                ResearchDatasetSpec(
                    data_id=f"bad_{field_name}",
                    market_id=MarketId("a_share"),
                    input_data_ids=("demo_factors", "demo_labels", "demo_annotations"),
                    base_input_data_id="demo_annotations",
                    **spec_kwargs,
                    frequency=DataFrequency.DAILY,
                    grain=DataGrain.RESEARCH_ROW,
                    column_roles={
                        "market_id": ColumnRole.KEY,
                        "asset_id": ColumnRole.KEY,
                        "timestamp": ColumnRole.TIMESTAMP,
                        "momentum_20d": ColumnRole.FACTOR,
                        "forward_return_20d": ColumnRole.LABEL,
                        "eligible": ColumnRole.UNIVERSE,
                        "can_enter": ColumnRole.MASK,
                        "can_exit": ColumnRole.MASK,
                        "market_type": ColumnRole.CONTEXT,
                    },
                    primary_key=PLATFORM_ASSET_TIME_KEY,
                    partition_columns=("year",),
                ),
            )
        )


@pytest.mark.parametrize("coverage", ("strict", "sparse"))
@pytest.mark.parametrize("physical_rows", (1, 2))
@pytest.mark.parametrize(
    "extra_date", (None, "2023-12-31", "2024-01-02", "2024-01-04", "2024-01-06")
)
def test_full_request_key_domain_is_independent_of_base_file_boundaries(
    tmp_path, coverage, physical_rows, extra_date
):
    import hashlib

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_specs import (
        DEMO_FACTOR_COLUMN,
        DEMO_LABEL_COLUMN,
        make_demo_research_dataset_spec,
    )

    layout = StorageLayout(tmp_path)
    processed = make_demo_processed_catalog()
    spec = make_demo_research_dataset_spec(key_coverage={"demo_factors": coverage})
    rows = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset_a",
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            DEMO_FACTOR_COLUMN: 1.0,
            DEMO_LABEL_COLUMN: 0.01,
            "eligible": True,
            "can_enter": True,
            "can_exit": True,
            "market_type": 1,
        }
    )
    factors = (
        rows
        if extra_date is None
        else pd.concat(
            (rows, rows.iloc[:1].assign(timestamp=pd.Timestamp(extra_date))),
            ignore_index=True,
        ).sort_values("timestamp")
    )
    roots = []
    for data_id, frame in (
        ("demo_annotations", rows),
        ("demo_labels", rows),
        ("demo_factors", factors),
    ):
        definition = processed.get(data_id)
        root = layout.processed_data_root(
            definition.market_id, definition.kind, data_id
        )
        selected = frame.loc[:, list(definition.required_columns)]
        chunks = tuple(
            selected.iloc[start : start + physical_rows]
            for start in range(0, len(selected), physical_rows)
        )
        write_dataset(root, chunks, spec=definition)
        roots.append(root)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }

    def assemble():
        return assemble_research_dataset(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            processed_catalog=processed,
            dataset_id=spec.data_id,
            time_window=TimeWindow("2023-12-31", "2024-01-04"),
        )

    if extra_date in {"2023-12-31", "2024-01-02", "2024-01-04"}:
        with pytest.raises(
            ValueError, match="extra_research_input_keys: demo_factors: 1"
        ):
            assemble()
        assert not layout.research_dataset_root(spec.market_id, spec.data_id).exists()
    else:
        frame = read_dataset(assemble())
        assert frame.timestamp.tolist() == rows.timestamp.tolist()
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )


def test_research_assembly_can_finish_after_soft_planning_target_is_used(
    tmp_path, monkeypatch
):
    from quant_research_platform.runtime import resources

    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(layout, processed_catalog)
    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 256 * 1024**2)
    with runtime_resource_policy_scope() as policy:
        with policy.lease("tests.existing", policy.memory_budget_bytes):
            assert policy.planning_headroom_bytes == 0
            root = assemble_research_dataset(
                layout=layout,
                dataset_catalog=dataset_catalog,
                processed_catalog=processed_catalog,
                dataset_id="demo",
                time_window=TimeWindow("2014-01-02", "2014-01-03"),
                resource_policy=policy,
            )
            assert root.is_dir()
        assert policy.used_memory_bytes == 0
    assert inspect_dataset(root).row_count > 0
