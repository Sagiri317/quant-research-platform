from __future__ import annotations

import ast
import inspect
import json
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.research.datasets.assembler as assembler_module
import quant_research_platform.research.evaluations.runner as evaluation_runner_module
import quant_research_platform.research.predictions.runner as prediction_runner_module
from quant_research_platform.artifacts.outputs import (
    write_evaluation_outputs,
    write_prediction_outputs,
)
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_PRIMARY_KEY,
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import (
    assemble_research_dataset,
    read_research_dataset,
    validate_evaluation_input,
    validate_prediction_input,
)
from quant_research_platform.research.evaluations.engine import evaluate_signal_frame
from quant_research_platform.research.evaluations.runner import (
    build_evaluation_result_from_research_dataset,
)
from quant_research_platform.research.models import fit_model_result
from quant_research_platform.research.predictions import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage import atomic as storage_atomic
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_processed_inputs import (
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import (
    make_research_dataset_catalog,
    make_signal_spec,
)
from tests.research.prediction_output_helpers import (
    write_prediction_output_and_read,
)


def _output_spec(
    *,
    output_definition_id: str,
    run_id: str,
    run_type: str,
    market_id: MarketId | None = None,
) -> OutputSpec:
    return OutputSpec(
        output_id=run_id,
        output_definition_id=output_definition_id,
        run_id="test_run",
        market_id=market_id or MarketId("a_share"),
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def _write_prediction_result(layout: StorageLayout, result) -> Path:
    return write_prediction_outputs(
        spec=result.spec,
        output_spec=_output_spec(
            output_definition_id="prediction_outputs",
            run_id=result.spec.prediction_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
        output_chunks=result.output_chunks,
        model_metadata=result.model_metadata,
        prediction_metadata_factory=result.prediction_metadata_factory,
    )


def _write_evaluation_result(layout: StorageLayout, result) -> Path:
    return write_evaluation_outputs(
        result.metrics,
        result.grouped_label_values,
        ic_summary=result.ic_summary,
        group_spread_summary=result.group_spread_summary,
        evaluation_diagnostics=result.diagnostics,
        spec=result.spec,
        output_spec=_output_spec(
            output_definition_id="evaluation_outputs",
            run_id=result.spec.run_id,
            market_id=result.spec.market_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
    )


def _research_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share", "a_share"],
            "asset_id": ["000001", "000002", "000001", "000002"],
            "timestamp": pd.to_datetime(
                ["2014-01-02", "2014-01-02", "2014-01-03", "2014-01-03"]
            ),
            "mom_20d": [1.0, 2.0, 2.0, 1.0],
            "forward_return_20d": [0.01, 0.03, 0.02, 0.00],
            "eligible": [True, True, True, True],
            "can_enter": [True, True, True, True],
            "can_exit": [True, True, True, True],
            "market_type": [4, 4, 4, 4],
        }
    )


def _research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "mom_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)


def test_research_dataset_assembles_from_processed_inputs(tmp_path: Path) -> None:
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
    result = read_research_dataset(
        layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    inspection = inspect_dataset(root)

    assert root == tmp_path / "data" / "research" / "a_share" / "demo"
    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "mom_20d",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert len(result) == 4
    assert "input_fingerprints" not in inspection.manifest
    assert set(inspection.manifest["input_dataset_identities"]) == {
        "demo_factors",
        "demo_labels",
        "demo_annotations",
    }
    validate_evaluation_input(
        result,
        dataset_catalog.get("demo"),
        signal_columns=("mom_20d",),
        label_columns=("forward_return_20d",),
    )
    validate_prediction_input(
        result,
        dataset_catalog.get("demo"),
        input_columns=("mom_20d",),
    )


def test_research_output_disk_guard_checks_each_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        assembler_module,
        "require_free_disk_headroom",
        lambda path, **_kwargs: observed.append(Path(path)),
    )
    frames = (pd.DataFrame({"value": [1]}), pd.DataFrame({"value": [2]}))

    result = list(
        assembler_module._iter_disk_guarded_output_chunks(
            frames,
            scratch_parent=tmp_path,
        )
    )

    assert [len(frame) for frame in result] == [1, 1]
    assert observed == [tmp_path, tmp_path]
    assert "_iter_disk_guarded_output_chunks(" in inspect.getsource(
        assembler_module._assemble_research_dataset_with_policy
    )


def test_research_time_key_scan_uses_payload_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_targets: list[int] = []
    timestamps = pd.date_range("2026-01-01", periods=20, freq="min")
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024,
        environ={},
    )
    session = SimpleNamespace()
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )

    def fake_iter_dataset_chunks(context, *, target_chunk_bytes):  # noqa: ANN001
        observed_targets.append(target_chunk_bytes)
        yield pd.DataFrame({"timestamp": timestamps})

    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        fake_iter_dataset_chunks,
        raising=False,
    )

    result = assembler_module._time_keys_from_base_input(
        session,
        primary_key=platform_entity_time_key(),
        time_window=TimeWindow(
            "2026-01-01T00:00:00",
            "2026-01-01T00:19:00",
        ),
        resource_policy=policy,
    )

    assert len(result) == 20
    assert observed_targets == [policy.payload_target_bytes]


def test_research_time_key_scan_preserves_single_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=320,
        environ={},
    )
    session = SimpleNamespace()
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )
    monkeypatch.setattr(
        assembler_module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(
            (pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")]}),)
        ),
    )

    assert assembler_module._time_keys_from_base_input(
        session,
        primary_key=platform_entity_time_key(),
        time_window=TimeWindow(
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
        resource_policy=policy,
    ) == (pd.Timestamp("2026-01-01"),)


def test_research_processed_input_uses_task_budget_for_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_catalog = make_demo_processed_catalog()
    dataset_spec = make_research_dataset_catalog().get("demo")
    processed_spec = processed_catalog.get("demo_factors")
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=12 * 1024,
        environ={},
    )
    observed: dict[str, object] = {}
    columns = assembler_module._processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=(),
    )
    context = SimpleNamespace(logical_columns=columns)
    monkeypatch.setattr(
        assembler_module,
        "_research_session_view",
        lambda current, **_kwargs: current,
    )

    def materialize(context, **kwargs):  # noqa: ANN001
        observed["context"] = context
        observed.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(assembler_module, "materialize_dataset_session", materialize)

    assembler_module._read_processed_input(
        dataset_spec=dataset_spec,
        processed_spec=processed_spec,
        session=context,
        existing_columns=(),
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        resource_policy=policy,
    )

    assert observed == {
        "context": context,
        "target_chunk_bytes": policy.payload_target_bytes,
        "materialized_budget_bytes": policy.task_memory_budget_bytes,
    }


def test_research_join_rejects_known_payloads_over_task_budget() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1_024,
        environ={},
    )

    with pytest.raises(
        ValueError,
        match="research_known_join_payloads_exceed_task_memory_budget",
    ):
        assembler_module._require_research_known_join_payload_bound(
            retained_payload_bytes=500,
            input_bytes=500,
            base_key_bytes=0,
            resource_policy=policy,
        )


def test_research_join_bound_includes_retained_base_keys() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1_000,
        environ={},
    )

    with pytest.raises(
        ValueError,
        match="research_known_join_payloads_exceed_task_memory_budget",
    ):
        assembler_module._require_research_known_join_payload_bound(
            retained_payload_bytes=300,
            input_bytes=400,
            base_key_bytes=300,
            resource_policy=policy,
        )


def test_research_final_write_rejects_output_over_payload_target() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1_000,
        environ={},
    )
    output = pd.DataFrame({"value": ["x" * 2_000]})

    with pytest.raises(
        ValueError,
        match="research_indivisible_join_unit_exceeds_payload_target",
    ):
        assembler_module._require_research_final_write_payload(
            output,
            resource_policy=policy,
        )


def test_research_dataset_assembles_intraday_timestamp_window(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    primary_key = platform_entity_time_key()
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
    processed_catalog = ProcessedDataCatalog()
    processed_catalog.register(factor_spec)
    processed_catalog.register(label_spec)
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
    result = read_research_dataset(
        layout,
        market_id=MarketId("crypto"),
        dataset_id="minute_research",
        time_window=TimeWindow(
            "2026-07-06T00:01:00",
            "2026-07-06T00:02:00",
        ),
    )

    assert len(result) == 4
    assert sorted(result["timestamp"].unique()) == list(timestamps[1:])


def test_research_dataset_assembler_generates_feature_transforms_from_factors(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        ResearchDatasetSpec(
            data_id="demo",
            market_id=MarketId("a_share"),
            input_data_ids=("demo_factors", "demo_labels", "demo_annotations"),
            base_input_data_id="demo_annotations",
            factor_columns=("mom_20d",),
            feature_columns=("mom_20d_zscore",),
            feature_transforms=(
                FeatureTransformSpec(
                    transform_id="mom_20d_zscore",
                    source_columns=("mom_20d",),
                    output_columns=("mom_20d_zscore",),
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
                "mom_20d": ColumnRole.FACTOR,
                "mom_20d_zscore": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
                "market_type": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
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
    result = read_research_dataset(
        layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    ).sort_values(["timestamp", "asset_id"])

    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "mom_20d",
        "mom_20d_zscore",
        "forward_return_20d",
        "eligible",
        "can_enter",
        "can_exit",
        "market_type",
    ]
    assert result["mom_20d_zscore"].tolist() == pytest.approx([-1.0, 1.0, 1.0, -1.0])
    validate_prediction_input(
        result,
        dataset_catalog.get("demo"),
        input_columns=("mom_20d", "mom_20d_zscore"),
    )


def test_research_dataset_assembler_does_not_mutate_storage_manifest() -> None:
    source = inspect.getsource(assembler_module)
    tree = ast.parse(source)
    string_constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "replace_internal_artifact_manifest" not in source
    assert "_write_input_fingerprints" not in source
    assert "fingerprint_dataset" not in source
    assert "_manifest" not in string_constants
    assert "manifest.json" not in string_constants


def test_research_dataset_forced_tiny_session_chunks_equal_default_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setenv("QRP_TASK_MEMORY_BUDGET_MB", "1")
    monkeypatch.setenv("QRP_PAYLOAD_TARGET_MB", "1")
    monkeypatch.setattr(
        assembler_module,
        "_estimate_research_join_bytes_per_session",
        lambda **_kwargs: 600_000,
    )
    tiny_root = assemble_research_dataset(
        layout=tiny_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )

    default = read_research_dataset(
        default_layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    ).sort_values(["market_id", "asset_id", "timestamp"])
    tiny = read_research_dataset(
        tiny_layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    ).sort_values(["market_id", "asset_id", "timestamp"])

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
    assert (
        inspect_dataset(tiny_root).file_count > inspect_dataset(default_root).file_count
    )


def test_research_dataset_sizing_accounts_for_assembly_peak_memory() -> None:
    frame = _research_rows_for_dates(["2014-01-02", "2014-01-03"])

    measured = assembler_module._measured_bytes_per_session(
        frame,
        sample_sessions=tuple(pd.to_datetime(["2014-01-02", "2014-01-03"])),
        target_session_count=2,
    )

    raw_per_session = max(1, ceil(frame.memory_usage(index=True, deep=True).sum() / 2))
    assert (
        measured
        == raw_per_session
        * assembler_module.RESEARCH_ASSEMBLY_PEAK_MEMORY_MULTIPLIER
    )


def test_research_sizing_probe_materializes_one_session_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = tuple(pd.date_range("2026-01-01", periods=20, freq="D"))
    observed_windows: list[TimeWindow] = []

    def join(**kwargs):
        observed_windows.append(kwargs["time_window"])
        return pd.DataFrame({"timestamp": [kwargs["time_window"].start]})

    monkeypatch.setattr(assembler_module, "_join_processed_inputs", join)

    assembler_module._estimate_research_join_bytes_per_session(
        dataset_spec=SimpleNamespace(),
        processed_catalog=SimpleNamespace(),
        sessions={},
        time_keys=sessions,
        time_window=TimeWindow("2026-01-01", "2026-01-20"),
        base_data_id="base",
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=1024**3,
            environ={},
        ),
    )

    assert len(observed_windows) == 20
    assert all(window.start == window.end for window in observed_windows)


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
        frame=_research_rows_for_dates(
            ["2014-01-02", "2014-01-03", "2014-01-04"]
        ),
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
        match="dataset_manifest_identity_changed",
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
    staged_verified_roots: list[Path] = []
    post_publish_verified_roots: list[Path] = []
    inside_staged_validator = False
    writer_returned = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def record_verify(session) -> None:  # noqa: ANN001
        if inside_staged_validator:
            staged_verified_roots.append(session.root)
        elif writer_returned and session.root in opened_roots:
            post_publish_verified_roots.append(session.root)
        real_verify(session)

    def record_write(root, data, *, staged_validator, **kwargs):  # noqa: ANN001
        nonlocal inside_staged_validator, writer_returned
        assert staged_validator is not None

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
        frame=_research_rows_for_dates(
            ["2014-01-02", "2014-01-03", "2014-01-04"]
        ),
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
    real_guarded_chunks = assembler_module._iter_disk_guarded_output_chunks
    real_write = assembler_module.write_dataset
    opened_roots: list[Path] = []
    replaced = False
    staged_validator_supplied = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def replace_after_last_input(chunks, *, scratch_parent):  # noqa: ANN001
        nonlocal replaced
        yield from real_guarded_chunks(chunks, scratch_parent=scratch_parent)
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
        "_iter_disk_guarded_output_chunks",
        replace_after_last_input,
    )
    monkeypatch.setattr(assembler_module, "write_dataset", record_write)

    with pytest.raises(ValueError, match="dataset_manifest_identity_changed"):
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
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(artifact_root)
    assert list(artifact_root.parent.glob(f"{scratch_prefix}*")) == []


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
    result = read_research_dataset(
        layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
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
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
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
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
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
        ("label_columns", "mom_20d", "research_input_column_overlap"),
        ("filter_columns", "market_type", "research_semantic_column_role_invalid"),
        ("context_columns", "eligible", "research_semantic_column_role_invalid"),
    ],
)
def test_research_dataset_role_validation_rejects_wrong_label_filter_and_context(
    tmp_path: Path,
    field_name: str,
    bad_column: str,
    expected_error: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_demo_processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    spec_kwargs = {
        "factor_columns": ("mom_20d",),
        "label_columns": ("forward_return_20d",),
        "filter_columns": ("eligible", "can_enter", "can_exit"),
        "context_columns": ("market_type",),
    }
    spec_kwargs[field_name] = (bad_column,)
    with pytest.raises(ValueError, match=expected_error):
        dataset_catalog.register(
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
                    "mom_20d": ColumnRole.FACTOR,
                    "forward_return_20d": ColumnRole.LABEL,
                    "eligible": ColumnRole.UNIVERSE,
                    "can_enter": ColumnRole.MASK,
                    "can_exit": ColumnRole.MASK,
                    "market_type": ColumnRole.CONTEXT,
                },
                primary_key=platform_entity_time_key(),
                partition_columns=("year",),
            )
        )
        write_demo_processed_inputs(layout, processed_catalog)
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id=f"bad_{field_name}",
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
        )


def test_evaluation_prediction_and_report_boundary(tmp_path: Path) -> None:
    frame = _research_rows()
    eval_spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("mom_20d"),),
        label_columns=("forward_return_20d",),
    )
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("mom_20d",),
        label_column="forward_return_20d",
        max_train_rows=len(frame),
    )
    prediction_spec = PredictionSpec(
        prediction_id="pred_demo",
        model_id=model_spec.model_id,
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )

    evaluation = evaluate_signal_frame(frame, eval_spec)
    model_result = fit_model_result(
        frame.loc[frame["timestamp"] <= pd.Timestamp(prediction_spec.train_end)],
        model_spec,
    )
    prediction, output_bundle, _ = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        prediction_spec,
        model_result.coefficients,
        model_spec.input_columns,
        model_metadata=model_result.metadata,
    )

    assert len(evaluation.metrics) == 2
    assert len(evaluation.ic_summary) == 1
    assert len(evaluation.group_spread_summary) == 1
    assert evaluation.diagnostics.rows_input == 4
    assert prediction_spec.prediction_column in prediction.columns
    assert prediction["timestamp"].min() >= pd.Timestamp("2014-01-03")
    assert prediction["timestamp"].max() <= pd.Timestamp("2014-01-03")
    assert output_bundle.as_posix().endswith("outputs/pred_demo")


def test_prediction_handoff_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_algorithm="linear_regression",
            input_factor_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            train_end="2014-01-02",
            test_start="2014-01-03",
            test_end="2014-01-03",
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    )
    output_bundle = _write_prediction_result(layout, result)

    assert output_bundle.as_posix().endswith("outputs/pred_demo")
    assert (output_bundle / "prediction.parquet").is_file()
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["is_oos"] is True
    assert manifest["metadata"]["prediction_mode"] == "minimal_oos"
    assert manifest["metadata"]["prediction_column"] == PREDICTION_SCORE_COLUMN
    assert manifest["metadata"]["primary_key"] == list(PREDICTION_PRIMARY_KEY)
    assert manifest["metadata"]["model_metadata"]["input_count"] == 1
    assert manifest["metadata"]["model_metadata"]["coefficient_count"] == 2
    assert manifest["metadata"]["model_metadata"]["train_start"] == "2014-01-02"
    assert manifest["metadata"]["model_metadata"]["train_end"] == "2014-01-02"
    assert manifest["metadata"]["prediction_metadata"]["prediction_rows"] == 2
    assert (
        manifest["metadata"]["prediction_metadata"]["requested_test_end"]
        == "2014-01-03"
    )
    assert (
        manifest["metadata"]["prediction_metadata"]["effective_test_end"]
        == "2014-01-03"
    )


def test_prediction_handoff_writes_prediction_report_without_materialized_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_algorithm="linear_regression",
            input_factor_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            train_end="2014-01-02",
            test_start="2014-01-03",
            test_end="2014-01-03",
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    )
    output_bundle = _write_prediction_result(layout, result)

    assert (output_bundle / "prediction.parquet").is_file()


def test_prediction_handoff_streams_research_chunks_for_model_and_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    monkeypatch.setenv("QRP_TASK_MEMORY_BUDGET_MB", "1")
    monkeypatch.setenv("QRP_PAYLOAD_TARGET_MB", "1")
    monkeypatch.setattr(
        assembler_module,
        "_estimate_research_join_bytes_per_session",
        lambda **_kwargs: 600_000,
    )
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    monkeypatch.setenv("QRP_TASK_MEMORY_BUDGET_MB", "1")
    monkeypatch.setenv("QRP_PAYLOAD_TARGET_MB", "1")
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_algorithm="linear_regression",
            input_factor_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            train_end="2014-01-03",
            test_start="2014-01-06",
            test_end="2014-01-08",
        )
    )

    original_iter = prediction_runner_module.iter_research_dataset_chunks
    original_fit = prediction_runner_module.fitting.fit_model_result_from_frames
    original_write = prediction_runner_module.build_prediction_result_from_frames
    read_chunk_rows: list[int] = []
    fit_chunk_rows: list[int] = []
    prediction_chunk_rows: list[int] = []
    read_columns_seen: list[tuple[str, ...]] = []

    def wrapped_iter(*args, **kwargs):
        read_columns_seen.append(tuple(kwargs["columns"]))
        for chunk in original_iter(*args, **kwargs):
            read_chunk_rows.append(len(chunk))
            yield chunk

    def wrapped_fit(frames, spec, *, estimator):
        chunks = list(frames)
        fit_chunk_rows.extend(len(chunk) for chunk in chunks)
        return original_fit(iter(chunks), spec, estimator=estimator)

    def wrapped_write(frames, spec, coefficients, input_columns, **kwargs):
        def counted_frames():
            for chunk in frames:
                prediction_chunk_rows.append(len(chunk))
                yield chunk

        return original_write(
            counted_frames(),
            spec,
            coefficients,
            input_columns,
            **kwargs,
        )

    monkeypatch.setattr(
        prediction_runner_module,
        "iter_research_dataset_chunks",
        wrapped_iter,
    )
    monkeypatch.setattr(
        prediction_runner_module.fitting,
        "fit_model_result_from_frames",
        wrapped_fit,
    )
    monkeypatch.setattr(
        prediction_runner_module,
        "build_prediction_result_from_frames",
        wrapped_write,
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    )
    output_bundle = _write_prediction_result(layout, result)

    assert (output_bundle / "prediction.parquet").is_file()
    assert len([rows for rows in read_chunk_rows if rows]) > 1
    assert max(read_chunk_rows) < len(frame)
    assert fit_chunk_rows
    assert max(fit_chunk_rows) < len(frame)
    assert len(prediction_chunk_rows) > 1
    assert max(prediction_chunk_rows) < len(frame)
    assert any("forward_return_20d" in columns for columns in read_columns_seen)
    assert any("forward_return_20d" not in columns for columns in read_columns_seen)


def test_evaluation_runner_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    evaluation_catalog = EvaluationCatalog()
    evaluation_catalog.register(
        EvaluationSpec(
            run_id="eval_demo",
            market_id=MarketId("a_share"),
            research_dataset_id="demo",
            signals=(make_signal_spec("mom_20d"),),
            label_columns=("forward_return_20d",),
        )
    )

    result = build_evaluation_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        evaluation_catalog=evaluation_catalog,
        run_id="eval_demo",
    )
    output_bundle = _write_evaluation_result(layout, result)

    assert output_bundle.as_posix().endswith("outputs/eval_demo")
    assert (output_bundle / "metrics.parquet").is_file()
    assert (output_bundle / "grouped_label_values.parquet").is_file()
    assert (output_bundle / "ic_summary.parquet").is_file()
    assert (output_bundle / "group_spread_summary.parquet").is_file()
    assert (output_bundle / "summary.md").is_file()
    assert (output_bundle / "manifest.json").is_file()
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
        "summary.md",
    ]


def test_evaluation_runner_streams_research_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    monkeypatch.setenv("QRP_TASK_MEMORY_BUDGET_MB", "1")
    monkeypatch.setenv("QRP_PAYLOAD_TARGET_MB", "1")
    monkeypatch.setattr(
        assembler_module,
        "_estimate_research_join_bytes_per_session",
        lambda **_kwargs: 600_000,
    )
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    evaluation_catalog = EvaluationCatalog()
    evaluation_catalog.register(
        EvaluationSpec(
            run_id="eval_demo",
            market_id=MarketId("a_share"),
            research_dataset_id="demo",
            signals=(make_signal_spec("mom_20d"),),
            label_columns=("forward_return_20d",),
        )
    )
    original_iter = evaluation_runner_module.iter_research_dataset_chunks
    read_chunk_rows: list[int] = []

    def wrapped_iter(*args, **kwargs):
        for chunk in original_iter(*args, **kwargs):
            read_chunk_rows.append(len(chunk))
            yield chunk

    monkeypatch.setattr(
        evaluation_runner_module,
        "iter_research_dataset_chunks",
        wrapped_iter,
    )

    result = build_evaluation_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        evaluation_catalog=evaluation_catalog,
        run_id="eval_demo",
    )
    output_bundle = _write_evaluation_result(layout, result)

    assert (output_bundle / "metrics.parquet").is_file()
    assert len([rows for rows in read_chunk_rows if rows]) > 1
    assert max(read_chunk_rows) < len(frame)
