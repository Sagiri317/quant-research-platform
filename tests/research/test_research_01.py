from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.research.datasets.assembler as assembler_module
import quant_research_platform.research.evaluations.runner as evaluation_runner_module
import quant_research_platform.research.predictions.runner as prediction_runner_module
from quant_research_platform.artifacts.run_outputs import (
    write_evaluation_run_outputs,
    write_prediction_run_outputs,
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
from quant_research_platform.contracts.predictions import (
    PREDICTION_PRIMARY_KEY,
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.run_outputs import OutputScope, RunOutputSpec
from quant_research_platform.contracts.signals import SignalSpec
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
from quant_research_platform.research.predictions.outputs import (
    build_prediction_result_from_frames,
)
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.research.prediction_artifact_helpers import (
    write_prediction_artifact_and_read,
)


def _run_output_spec(
    *,
    output_definition_id: str,
    run_id: str,
    run_type: str,
    market_id: MarketId | None = None,
) -> RunOutputSpec:
    return RunOutputSpec(
        output_id=output_definition_id,
        output_definition_id=output_definition_id,
        run_id=run_id,
        market_id=market_id or MarketId("a_share"),
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def _write_prediction_result(layout: StorageLayout, result) -> Path:
    return write_prediction_run_outputs(
        spec=result.spec,
        output_spec=_run_output_spec(
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
    return write_evaluation_run_outputs(
        result.metrics,
        result.grouped_label_values,
        ic_summary=result.ic_summary,
        group_spread_summary=result.group_spread_summary,
        evaluation_diagnostics=result.diagnostics,
        spec=result.spec,
        output_spec=_run_output_spec(
            output_definition_id="evaluation_outputs",
            run_id=result.spec.run_id,
            market_id=result.spec.market_id,
            run_type=result.spec.run_type,
        ),
        layout=layout,
    )


def _signal(column: str, *, source_kind: str = "factor") -> SignalSpec:
    return SignalSpec(
        signal_id=column,
        source_column=column,
        source_kind=source_kind,
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


def _processed_catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_factors",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_labels",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.LABEL,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_annotations",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.ANNOTATION,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
                "market_type": ColumnRole.CONTEXT,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def _research_dataset_catalog(
    *,
    key_coverage: dict[str, str] | None = None,
) -> ResearchDatasetCatalog:
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            data_id="demo",
            market_id=MarketId("a_share"),
            input_data_ids=("demo_factors", "demo_labels", "demo_annotations"),
            base_input_data_id="demo_annotations",
            factor_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
            filter_columns=("eligible", "can_enter", "can_exit"),
            context_columns=("market_type",),
            key_coverage=key_coverage or {},
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
    return catalog


def test_research_dataset_assembles_from_processed_inputs(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = _research_dataset_catalog()
    _write_processed_inputs(layout, processed_catalog)

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
    assert set(inspection.manifest["input_fingerprints"]) == {
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


def test_research_dataset_assembler_generates_feature_transforms_from_factors(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
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
    _write_processed_inputs(layout, processed_catalog)

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
    processed_catalog = _processed_catalog()
    dataset_catalog = _research_dataset_catalog()
    _write_processed_inputs(default_layout, processed_catalog, frame=frame)
    _write_processed_inputs(tiny_layout, processed_catalog, frame=frame)

    default_root = assemble_research_dataset(
        layout=default_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")
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


def test_sparse_research_dataset_left_joins_missing_non_base_input_key(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = _research_dataset_catalog(
        key_coverage={"demo_labels": "sparse"},
    )
    frame = _research_rows()
    labels = frame.drop(index=frame.index[-1]).reset_index(drop=True)
    _write_processed_inputs(layout, processed_catalog, label_frame=labels)

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
    processed_catalog = _processed_catalog()
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
    _write_processed_inputs(layout, processed_catalog)

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
        ("filter_columns", "market_type", "invalid_research_filter_columns"),
        ("context_columns", "eligible", "invalid_research_context_columns"),
    ],
)
def test_research_dataset_role_validation_rejects_wrong_label_filter_and_context(
    tmp_path: Path,
    field_name: str,
    bad_column: str,
    expected_error: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
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
        _write_processed_inputs(layout, processed_catalog)
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
        signals=(_signal("mom_20d"),),
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
    prediction, run_root, _ = write_prediction_artifact_and_read(
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
    assert run_root.as_posix().endswith(
        "run_outputs/local/a_share/prediction/pred_demo"
    )
    assert (
        "output_root"
        not in inspect.signature(build_prediction_result_from_frames).parameters
    )


def test_prediction_handoff_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _research_dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog)
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
    run_root = _write_prediction_result(layout, result)

    assert run_root.as_posix().endswith(
        "run_outputs/local/a_share/prediction/pred_demo"
    )
    assert (run_root / "prediction.parquet").is_file()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_oos"] is True
    assert manifest["prediction_mode"] == "minimal_oos"
    assert manifest["prediction_column"] == PREDICTION_SCORE_COLUMN
    assert manifest["primary_key"] == list(PREDICTION_PRIMARY_KEY)
    assert manifest["model_metadata"]["contract_version"] == "0.1"
    assert manifest["model_metadata"]["input_count"] == 1
    assert manifest["model_metadata"]["coefficient_count"] == 2
    assert manifest["model_metadata"]["train_start"] == "2014-01-02"
    assert manifest["model_metadata"]["train_end"] == "2014-01-02"
    assert manifest["prediction_metadata"]["contract_version"] == "0.1"
    assert manifest["prediction_metadata"]["prediction_rows"] == 2
    assert manifest["prediction_metadata"]["requested_test_end"] == "2014-01-03"
    assert manifest["prediction_metadata"]["effective_test_end"] == "2014-01-03"
    assert "metadata" not in manifest


def test_prediction_handoff_writes_prediction_report_without_materialized_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _research_dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog)
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
    run_root = _write_prediction_result(layout, result)

    assert (run_root / "prediction.parquet").is_file()


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
    dataset_catalog = _research_dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame=frame)
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
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
            train_end="2014-01-03",
            test_start="2014-01-06",
            test_end="2014-01-08",
        )
    )

    original_iter = prediction_runner_module._iter_research_dataset_chunks
    original_fit = prediction_runner_module._fit_model_result_from_frames
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

    def wrapped_fit(frames, spec):
        chunks = list(frames)
        fit_chunk_rows.extend(len(chunk) for chunk in chunks)
        return original_fit(iter(chunks), spec)

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
        "_iter_research_dataset_chunks",
        wrapped_iter,
    )
    monkeypatch.setattr(
        prediction_runner_module,
        "_fit_model_result_from_frames",
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
    run_root = _write_prediction_result(layout, result)

    assert (run_root / "prediction.parquet").is_file()
    assert len([rows for rows in read_chunk_rows if rows]) > 1
    assert max(read_chunk_rows) < len(frame)
    assert len(fit_chunk_rows) > 1
    assert max(fit_chunk_rows) < len(frame)
    assert len(prediction_chunk_rows) > 1
    assert max(prediction_chunk_rows) < len(frame)
    assert any("forward_return_20d" in columns for columns in read_columns_seen)
    assert any("forward_return_20d" not in columns for columns in read_columns_seen)


def test_evaluation_runner_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _research_dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog)
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
            signals=(_signal("mom_20d"),),
            label_columns=("forward_return_20d",),
        )
    )

    result = build_evaluation_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        evaluation_catalog=evaluation_catalog,
        run_id="eval_demo",
    )
    run_root = _write_evaluation_result(layout, result)

    assert run_root.as_posix().endswith(
        "run_outputs/local/a_share/evaluation/eval_demo"
    )
    assert (run_root / "metrics.parquet").is_file()
    assert (run_root / "grouped_label_values.parquet").is_file()
    assert (run_root / "ic_summary.parquet").is_file()
    assert (run_root / "group_spread_summary.parquet").is_file()
    assert (run_root / "manifest.json").is_file()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
    ]
    assert manifest["evaluation_diagnostics"]["contract_version"] == "0.1"
    assert "metadata" not in manifest


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
    dataset_catalog = _research_dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame=frame)
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")
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
            signals=(_signal("mom_20d"),),
            label_columns=("forward_return_20d",),
        )
    )
    original_iter = evaluation_runner_module._iter_research_dataset_chunks
    read_chunk_rows: list[int] = []

    def wrapped_iter(*args, **kwargs):
        for chunk in original_iter(*args, **kwargs):
            read_chunk_rows.append(len(chunk))
            yield chunk

    monkeypatch.setattr(
        evaluation_runner_module,
        "_iter_research_dataset_chunks",
        wrapped_iter,
    )

    result = build_evaluation_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        evaluation_catalog=evaluation_catalog,
        run_id="eval_demo",
    )
    run_root = _write_evaluation_result(layout, result)

    assert (run_root / "metrics.parquet").is_file()
    assert len([rows for rows in read_chunk_rows if rows]) > 1
    assert max(read_chunk_rows) < len(frame)


def _write_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    frame: pd.DataFrame | None = None,
    label_frame: pd.DataFrame | None = None,
) -> None:
    frame = _research_rows() if frame is None else frame
    labels = frame if label_frame is None else label_frame
    columns_by_id = {
        "demo_factors": ["market_id", "asset_id", "timestamp", "mom_20d"],
        "demo_annotations": [
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            "can_enter",
            "can_exit",
            "market_type",
        ],
    }
    for data_id, columns in columns_by_id.items():
        spec = processed_catalog.get(data_id)
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame.loc[:, columns],
            spec=spec,
        )
    spec = processed_catalog.get("demo_labels")
    write_dataset(
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
        labels.loc[
            :,
            ["market_id", "asset_id", "timestamp", "forward_return_20d"],
        ],
        spec=spec,
    )
