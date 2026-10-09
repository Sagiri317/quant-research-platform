from __future__ import annotations

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
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_processed_inputs import (
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import make_research_dataset_catalog
from tests.research.workflow_test_support import (
    _passthrough_session,
    _research_rows,
    _research_rows_for_dates,
)


def test_research_dataset_assembles_from_processed_inputs(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
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
        result, dataset_catalog.get("demo"), input_columns=("momentum_20d",)
    )


@pytest.mark.parametrize(
    "scenario", ("pv_all", "pv_expanded", "fusion_nf_core", "fusion_nf_expectations")
)
def test_non_a_share_named_scenario_keeps_input_semantics_and_requires_real_columns(
    tmp_path, scenario
):
    from dataclasses import replace

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.research_processed_inputs import (
        make_dummy_processed_catalog,
        write_dummy_processed_inputs,
    )
    from tests.helpers.research_specs import make_dummy_research_dataset_spec

    layout = StorageLayout(tmp_path)
    processed = make_dummy_processed_catalog()
    spec = make_dummy_research_dataset_spec(data_id=scenario)
    write_dummy_processed_inputs(layout, processed)
    root = assemble_research_dataset(
        layout=layout,
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        processed_catalog=processed,
        dataset_id=scenario,
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    frame = read_dataset(root)
    assert frame.eligible.tolist() == [True, False]
    assert assembler_module._generated_semantic_columns(spec) == ()
    assert assembler_module._add_a_share_semantic_columns(frame, spec) is frame
    a_share = replace(spec, market_id=MarketId("a_share"))
    assert assembler_module._generated_semantic_columns(a_share) == ("eligible",)
    with pytest.raises(ValueError, match="^a_share_eligibility_facts_missing:"):
        assembler_module._add_a_share_semantic_columns(
            frame.assign(market_id="a_share"), a_share
        )
    missing = ProcessedDataCatalog(
        tuple(
            replace(
                item,
                column_roles={
                    key: role
                    for key, role in item.column_roles.items()
                    if key != "eligible"
                },
            )
            if str(item.data_id) == "filters"
            else item
            for item in processed.list()
        )
    )
    with pytest.raises(
        ValueError, match="^missing_research_required_columns: eligible$"
    ):
        assembler_module._validate_research_input_specs(spec, missing)


def test_research_time_window_driver_uses_one_window_per_physical_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = pd.date_range("2026-01-01", periods=20, freq="min")
    session = _passthrough_session()

    def fake_iter_dataset_chunks(_context):
        yield pd.DataFrame({"timestamp": timestamps})

    monkeypatch.setattr(
        assembler_module, "iter_dataset_chunks", fake_iter_dataset_chunks, raising=False
    )
    result = list(
        assembler_module._iter_research_chunk_windows_from_base_input(
            session,
            time_window=TimeWindow("2026-01-01T00:00:00", "2026-01-01T00:19:00"),
        )
    )
    assert result == [TimeWindow("2026-01-01T00:00:00", "2026-01-01T00:19:00")]


def test_research_time_window_driver_preserves_single_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _passthrough_session()
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: (
            frame
            for frame in (pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]}),)
        ),
    )
    assert list(
        assembler_module._iter_research_chunk_windows_from_base_input(
            session,
            time_window=TimeWindow("2026-01-01T00:00:00", "2026-01-01T00:00:00"),
        )
    ) == [TimeWindow("2026-01-01T00:00:00", "2026-01-01T00:00:00")]


def test_research_time_window_driver_normalizes_and_deduplicates_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _passthrough_session()
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
    result = list(
        assembler_module._iter_research_chunk_windows_from_base_input(
            session, time_window=TimeWindow("2026-01-01", "2026-01-03")
        )
    )
    assert result == [
        TimeWindow("2026-01-01", "2026-01-02"),
        TimeWindow("2026-01-03", "2026-01-03"),
    ]


def test_research_time_window_driver_rejects_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _passthrough_session()
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]")}),)
        ),
    )
    with pytest.raises(ValueError, match="no_research_dataset_sessions"):
        list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                session, time_window=TimeWindow("2026-01-01", "2026-01-01")
            )
        )


def test_research_time_window_driver_rejects_non_monotonic_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _passthrough_session()
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
    with pytest.raises(ValueError, match="non_monotonic_research_time_keys"):
        list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                session, time_window=TimeWindow("2026-01-01", "2026-01-02")
            )
        )


def test_research_time_window_driver_closes_iterator_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    view = SimpleNamespace()

    def filtered_view(_time_window):
        return view

    base_session = SimpleNamespace(filtered_view=filtered_view)

    def chunks():
        try:
            yield pd.DataFrame({"not_timestamp": [1]})
        finally:
            events.append("iterator_close")

    iterator = chunks()
    monkeypatch.setattr(
        assembler_module, "iter_dataset_chunks", lambda *_args, **_kwargs: iterator
    )
    with pytest.raises(KeyError, match="timestamp"):
        list(
            assembler_module._iter_research_chunk_windows_from_base_input(
                base_session, time_window=TimeWindow("2026-01-01", "2026-01-01")
            )
        )
    assert events == ["iterator_close"]


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
        ["2026-07-06T00:00:00", "2026-07-06T00:01:00", "2026-07-06T00:02:00"]
    )
    keys = pd.DataFrame(
        [
            {"market_id": "crypto", "asset_id": asset_id, "timestamp": timestamp}
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
        time_window=TimeWindow("2026-07-06T00:01:00", "2026-07-06T00:02:00"),
    )
    result = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("crypto"),
            dataset_id="minute_research",
            time_window=TimeWindow("2026-07-06T00:01:00", "2026-07-06T00:02:00"),
        ),
        ignore_index=True,
    )
    assert len(result) == 4
    assert sorted(result["timestamp"].unique()) == list(timestamps[1:])


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


def test_research_dataset_session_complete_chunks_preserve_output(
    tmp_path: Path,
) -> None:
    frame = _research_rows_for_dates(
        ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
    )
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(default_layout, processed_catalog, frame=frame)
    for spec in processed_catalog.list():
        source = frame.loc[:, list(spec.column_roles)]
        root = tiny_layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset(
            root,
            (source.iloc[position : position + 1] for position in range(len(source))),
            spec=spec,
        )
        assert inspect_dataset(root).file_count == len(source)
    base_spec = processed_catalog.get(dataset_catalog.get("demo").base_input_data_id)
    base_root = tiny_layout.processed_data_root(
        base_spec.market_id, base_spec.kind, base_spec.data_id
    )
    from quant_research_platform.storage.datasets.reading import (
        iter_dataset_chunks,
        open_dataset_session,
    )

    batches = list(
        iter_dataset_chunks(
            open_dataset_session(base_root, columns=None, time_window=None)
        )
    )
    assert len(batches) == len(frame)
    assert all(len(batch) == 1 for batch in batches)
    assert batches[0].timestamp.iloc[0] == batches[1].timestamp.iloc[0]
    default_root = assemble_research_dataset(
        layout=default_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    tiny_root = assemble_research_dataset(
        layout=tiny_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
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
        default.reset_index(drop=True), tiny.reset_index(drop=True)
    )
    assert (
        inspect_dataset(tiny_root).file_count > inspect_dataset(default_root).file_count
    )


def test_research_chunk_view_prunes_top_level_multi_year_physical_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    frame = _research_rows_for_dates(["2014-12-31", "2015-01-02"])
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    processed_spec = processed_catalog.get("demo_factors")
    dataset_spec = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    ).get("demo")
    columns = assembler_module._processed_input_columns(
        dataset_spec, processed_spec, existing_columns=()
    )
    session = assembler_module.open_dataset_session(
        layout.processed_data_root(
            processed_spec.market_id, processed_spec.kind, processed_spec.data_id
        ),
        columns=columns,
        time_window=TimeWindow("2014-12-31", "2015-01-02"),
    )
    view = session.filtered_view(TimeWindow("2015-01-02", "2015-01-02"))
    assert len(view.read_paths) == 1
    assert view.read_paths[0].parent.name == "year=2015"
    assert len(session.read_paths) == 2


def test_research_assembly_rejects_dependency_replacement_after_identity_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(layout, processed_catalog)
    write_demo_processed_inputs(
        replacement_layout,
        processed_catalog,
        frame=_research_rows_for_dates(["2014-01-02", "2014-01-03", "2014-01-04"]),
    )
    factor_spec = processed_catalog.get("demo_factors")
    current_root = layout.processed_data_root(
        factor_spec.market_id, factor_spec.kind, factor_spec.data_id
    )
    replacement_root = replacement_layout.processed_data_root(
        factor_spec.market_id, factor_spec.kind, factor_spec.data_id
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    real_identity = assembler_module.dataset_identity_payload
    real_open = assembler_module.open_dataset_session
    opened_roots: list[Path] = []
    replaced = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def replace_after_identity(spec, manifest):
        nonlocal replaced
        identity = real_identity(spec, manifest)
        if spec["data_id"] == factor_spec.data_id and (not replaced):
            current_root.rename(backup_root)
            replacement_root.rename(current_root)
            replaced = True
        return identity

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(
        assembler_module, "dataset_identity_payload", replace_after_identity
    )
    with pytest.raises(ValueError, match="dataset_(spec|manifest)_identity_changed"):
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


def test_research_assembly_verifies_every_input_inside_staged_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
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

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        session = real_open(root, **kwargs)
        opened_sessions.append(session)
        return session

    def record_verify(session) -> None:
        if inside_staged_validator:
            staged_verified_roots.append(session.root)
        elif writer_returned and session.root in opened_roots:
            post_publish_verified_roots.append(session.root)
        real_verify(session)

    def record_write(root, data, *, staged_validator, **kwargs):
        nonlocal inside_staged_validator, writer_returned
        assert staged_validator is not None

        def record_staged(receipt) -> None:
            nonlocal inside_staged_validator
            inside_staged_validator = True
            try:
                staged_validator(receipt)
            finally:
                inside_staged_validator = False

        result = real_write(root, data, staged_validator=record_staged, **kwargs)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE}
    )
    write_demo_processed_inputs(layout, processed_catalog)
    write_demo_processed_inputs(
        replacement_layout,
        processed_catalog,
        frame=_research_rows_for_dates(["2014-01-02", "2014-01-03", "2014-01-04"]),
    )
    factor_spec = processed_catalog.get("demo_factors")
    current_root = layout.processed_data_root(
        factor_spec.market_id, factor_spec.kind, factor_spec.data_id
    )
    replacement_root = replacement_layout.processed_data_root(
        factor_spec.market_id, factor_spec.kind, factor_spec.data_id
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

    def replace_after_last_input(**kwargs):
        nonlocal replaced
        yield from real_joined_chunks(**kwargs)
        current_root.rename(backup_root)
        replacement_root.rename(current_root)
        replaced = True

    def record_write(root, data, *, staged_validator, **kwargs):
        nonlocal staged_validator_supplied
        staged_validator_supplied = staged_validator is not None
        return real_write(root, data, staged_validator=staged_validator, **kwargs)

    monkeypatch.setattr(assembler_module, "open_dataset_session", record_open)
    monkeypatch.setattr(
        assembler_module, "_iter_joined_processed_chunks", replace_after_last_input
    )
    monkeypatch.setattr(assembler_module, "write_dataset", record_write)
    with pytest.raises(ValueError, match="dataset_(spec|manifest)_identity_changed"):
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
        (
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for path, digest in before.items()
        )
    )


def test_model_target_uses_last_participating_label_maturity_and_keeps_raw_clock(
    tmp_path,
):
    from dataclasses import replace

    from quant_research_platform.contracts.experiments import (
        legal_training_rows,
        resolve_training_availability_column,
    )
    from quant_research_platform.research.protocols import a_share
    from quant_research_platform.storage.datasets import read_dataset

    base = a_share.research_dataset_catalog().get("pv_all")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "ineligible"],
            "timestamp": pd.Timestamp("2020-01-02"),
            "forward_return_20d": [0.1, 0.3, 0.5],
            "forward_return_20d_available_at": pd.to_datetime(
                ["2020-02-01", "2020-02-04", "2020-02-10"]
            ),
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": [False, False, True],
            "is_pt": False,
            "listing_age_trading_days": 400,
            "has_252_trading_sessions": True,
            "close": 10.0,
            "adj_close": 10.0,
            "volume": 10.0,
            "amount": 100.0,
            "turnover": 0.1,
        }
    )
    output = assembler_module._add_a_share_semantic_columns(frame, base)
    output["momentum_20d"] = [1.0, 2.0, 3.0]
    pd.testing.assert_series_equal(
        output.forward_return_20d_available_at, frame.forward_return_20d_available_at
    )
    assert (
        output.model_target_available_at.iloc[:2].eq(pd.Timestamp("2020-02-04")).all()
    )
    assert pd.isna(output.model_target_available_at.iloc[2])
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        "forward_return_20d",
        "forward_return_20d_available_at",
        "model_target",
        "model_target_available_at",
        "eligible",
        "momentum_20d",
    )
    spec = replace(
        base,
        factor_columns=("momentum_20d",),
        feature_columns=(),
        feature_transforms=(),
        context_columns=("model_target_available_at",),
        label_columns=(
            "forward_return_20d",
            "forward_return_20d_available_at",
            "model_target",
        ),
        column_roles={key: base.column_roles[key] for key in columns},
    )
    root = StorageLayout(tmp_path).research_dataset_root(spec.market_id, spec.data_id)
    write_dataset(root, output.loc[:, columns], spec=spec)
    saved = read_dataset(root)
    maturity = resolve_training_availability_column(saved, label_column="model_target")
    assert maturity == "model_target_available_at"
    assert legal_training_rows(
        saved,
        fitting_cutoff=pd.Timestamp("2020-02-02"),
        availability_column=maturity,
        target_column="model_target",
        feature_columns=(),
    ).empty
    assert (
        len(
            legal_training_rows(
                saved,
                fitting_cutoff=pd.Timestamp("2020-02-04"),
                availability_column=maturity,
                target_column="model_target",
                feature_columns=(),
            )
        )
        == 2
    )


@pytest.mark.parametrize("recipe", ("a_share_canonical", "a_share_quantile_1_99"))
def test_named_feature_recipe_assembles_complete_sessions_and_persists_identity(
    tmp_path, recipe
):
    import numpy as np

    from quant_research_platform.research.features.transforms import (
        a_share_feature_pipeline,
    )
    from quant_research_platform.storage.datasets.schema import data_spec_payload

    layout = StorageLayout(project_root=tmp_path)
    transform = FeatureTransformSpec(recipe, ("factor",))
    context = ("industry_standard_id", "industry_level1_code", "log_float_market_cap")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{i:03d}" for i in range(120)],
            "timestamp": pd.Timestamp("2020-01-03"),
            "factor": np.sin(np.arange(120)),
            "label": 0.0,
            "eligible": True,
            "industry_standard_id": "fixture",
            "industry_level1_code": "A",
            "log_float_market_cap": np.arange(120) * 0.01,
        }
    )
    frame.loc[119, "factor"] = 1000.0
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "factor": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
        "eligible": ColumnRole.UNIVERSE,
        **dict.fromkeys(context, ColumnRole.CONTEXT),
    }
    source = ProcessedDataSpec(
        data_id="recipe_inputs",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
    datasets = make_research_dataset_catalog(
        data_id="recipe_case",
        input_data_ids=(source.data_id,),
        base_input_data_id=source.data_id,
        factor_columns=("factor",),
        feature_columns=transform.output_columns,
        feature_transforms=(transform,),
        label_columns=("label",),
        filter_columns=("eligible",),
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        context_columns=context,
    )
    write_dataset(
        layout.processed_data_root(source.market_id, source.kind, source.data_id),
        (frame.iloc[i : i + 25] for i in range(0, len(frame), 25)),
        spec=source,
    )
    window = TimeWindow("2020-01-03", "2020-01-03")
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=datasets,
        processed_catalog=ProcessedDataCatalog((source,)),
        dataset_id="recipe_case",
        time_window=window,
    )
    saved = pd.concat(
        iter_research_dataset_chunks(
            layout,
            market_id=MarketId("a_share"),
            dataset_id="recipe_case",
            time_window=window,
        ),
        ignore_index=True,
    )
    expected = a_share_feature_pipeline(
        frame, factor_columns=("factor",), recipe_id=recipe
    )
    for column in ("factor", *transform.output_columns):
        pd.testing.assert_series_equal(
            saved[column], expected[column], check_exact=True
        )
    assert saved.feature_factor.notna().all()
    assert data_spec_payload(datasets.get("recipe_case"))["feature_transforms"] == [
        {"recipe_id": recipe, "source_columns": ["factor"]},
    ]
