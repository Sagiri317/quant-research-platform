from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import (
    PredictionCatalog,
    default_prediction_column,
)
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.markets.implementations.a_share.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.research.datasets import (
    assemble_research_dataset,
    assemble_research_dataset_from_processed,
    read_research_dataset,
    validate_evaluation_input,
    validate_prediction_input,
)
from quant_research_platform.research.evaluations.engine import evaluate_factor_frame
from quant_research_platform.research.evaluations.runner import (
    run_evaluation_from_research_dataset,
)
from quant_research_platform.research.models import fit_linear_baseline
from quant_research_platform.research.predictions import (
    build_prediction_bundle,
    run_prediction_from_research_dataset,
    write_prediction_bundle,
)
from quant_research_platform.storage.layout import (
    StorageLayout,
    layout_from_processed_market_root,
)
from quant_research_platform.storage.parquet import write_processed_data


def _research_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share", "a_share"],
            "asset_id": ["000001", "000002", "000001", "000002"],
            "timestamp": pd.to_datetime(
                ["2014-01-02", "2014-01-02", "2014-01-03", "2014-01-03"]
            ),
            "mom_20d": [1.0, 2.0, 2.0, 1.0],
            "forward_return_20d": [0.01, 0.03, 0.02, 0.00],
        }
    )


def _research_dataset_catalog() -> ResearchDatasetCatalog:
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            dataset_id="demo",
            market_id=MarketId("a_share"),
            processed_data_ids=(
                PRICE_VOLUME_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
            ),
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def test_research_dataset_assemble_read_validate(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"

    assemble_research_dataset(
        _research_frame(),
        _research_dataset_catalog(),
        "demo",
        processed_root,
    )
    result = read_research_dataset(processed_root, "demo")

    assert len(result) == 4
    assert set(result.columns) == {
        "market_id",
        "asset_id",
        "timestamp",
        "mom_20d",
        "forward_return_20d",
    }


def test_research_dataset_assembles_from_processed_specs(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    processed_catalog = ProcessedDataCatalog()
    processed_catalog.register(
        ProcessedDataSpec(
            market_id=MarketId("a_share"),
            category="factors",
            processed_data_id="demo_factors",
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
    processed_catalog.register(
        ProcessedDataSpec(
            market_id=MarketId("a_share"),
            category="labels",
            processed_data_id="demo_labels",
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
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        ResearchDatasetSpec(
            dataset_id="demo_from_processed",
            market_id=MarketId("a_share"),
            processed_data_ids=("demo_factors", "demo_labels"),
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )

    layout = layout_from_processed_market_root(processed_root)
    write_processed_data(
        _research_frame().loc[:, ["market_id", "asset_id", "timestamp", "mom_20d"]],
        layout=layout,
        catalog=processed_catalog,
        processed_data_id="demo_factors",
    )
    write_processed_data(
        _research_frame().loc[
            :, ["market_id", "asset_id", "timestamp", "forward_return_20d"]
        ],
        layout=layout,
        catalog=processed_catalog,
        processed_data_id="demo_labels",
    )

    assemble_research_dataset_from_processed(
        processed_root=processed_root,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_from_processed",
        years=(2014,),
    )
    result = read_research_dataset(processed_root, "demo_from_processed")

    assert list(result.columns) == [
        "market_id",
        "asset_id",
        "timestamp",
        "mom_20d",
        "forward_return_20d",
    ]
    assert len(result) == 4
    validate_evaluation_input(
        result,
        dataset_catalog.get("demo_from_processed"),
        feature_columns=("mom_20d",),
        label_columns=("forward_return_20d",),
    )
    validate_prediction_input(
        result,
        dataset_catalog.get("demo_from_processed"),
        feature_columns=("mom_20d",),
    )


def test_evaluation_prediction_and_report_boundary(tmp_path: Path) -> None:
    frame = _research_frame()
    eval_spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        feature_columns=("mom_20d",),
        label_columns=("forward_return_20d",),
    )
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_family="linear",
        feature_columns=("mom_20d",),
        label_column="forward_return_20d",
    )
    prediction_spec = PredictionSpec(
        prediction_id="pred_demo",
        model_id=model_spec.model_id,
        research_dataset_id="demo",
        prediction_column=default_prediction_column(),
        primary_key=platform_entity_time_key(),
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )

    evaluation = evaluate_factor_frame(frame, eval_spec)
    coefficients = fit_linear_baseline(frame, model_spec)
    prediction = build_prediction_bundle(
        frame,
        prediction_spec,
        coefficients,
        model_spec.feature_columns,
    )
    layout = StorageLayout(project_root=tmp_path)
    report_spec = ReportSpec(
        report_id="prediction_bundle",
        report_definition_id="prediction_bundle_private",
        run_id=prediction_spec.prediction_id,
        market_id=MarketId("a_share"),
        run_type=prediction_spec.run_type,
    )
    run_root = write_prediction_bundle(
        prediction,
        prediction_spec,
        report_spec=report_spec,
        layout=layout,
    )

    assert len(evaluation.metrics) == 2
    assert prediction_spec.prediction_column in prediction.columns
    assert prediction["timestamp"].min() >= pd.Timestamp("2014-01-03")
    assert prediction["timestamp"].max() <= pd.Timestamp("2014-01-03")
    assert run_root.as_posix().endswith(
        "reports/private_real/a_share/prediction/pred_demo"
    )
    assert "output_root" not in inspect.signature(write_prediction_bundle).parameters


def test_prediction_handoff_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    dataset_catalog = _research_dataset_catalog()
    assemble_research_dataset(
        _research_frame(),
        dataset_catalog,
        "demo",
        processed_root,
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_family="linear",
            feature_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            prediction_column=default_prediction_column(),
            primary_key=platform_entity_time_key(),
            train_end="2014-01-02",
            test_start="2014-01-03",
            test_end="2014-01-03",
        )
    )

    run_root = run_prediction_from_research_dataset(
        processed_root=processed_root,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    )

    assert run_root.as_posix().endswith(
        "reports/private_real/a_share/prediction/pred_demo"
    )
    assert (run_root / "prediction.parquet").is_file()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_oos"] is True
    assert manifest["prediction_mode"] == "minimal_oos"
    assert manifest["train_end"] == "2014-01-02"
    assert manifest["test_start"] == "2014-01-03"
    assert manifest["test_end"] == "2014-01-03"
    assert manifest["is_final_oos_window"] is False
    written = pd.read_parquet(run_root / "prediction.parquet")
    assert written["timestamp"].min() >= pd.Timestamp("2014-01-03")
    assert written["timestamp"].max() <= pd.Timestamp("2014-01-03")


def test_prediction_handoff_rejects_overlapping_oos_split(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    dataset_catalog = _research_dataset_catalog()
    assemble_research_dataset(
        _research_frame(),
        dataset_catalog,
        "demo",
        processed_root,
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_demo",
            model_family="linear",
            feature_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            prediction_column=default_prediction_column(),
            primary_key=platform_entity_time_key(),
            train_end="2014-01-03",
            test_start="2014-01-03",
        )
    )

    with pytest.raises(
        ValueError,
        match="prediction_train_end_must_precede_test_start",
    ):
        run_prediction_from_research_dataset(
            processed_root=processed_root,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="pred_demo",
        )


def test_prediction_spec_rejects_open_ended_non_final_window() -> None:
    with pytest.raises(ValueError, match="prediction_open_ended_window_must_be_final"):
        PredictionSpec(
            prediction_id="pred_demo",
            model_id="linear_demo",
            research_dataset_id="demo",
            prediction_column=default_prediction_column(),
            primary_key=platform_entity_time_key(),
            train_end="2014-01-02",
            test_start="2014-01-03",
            is_final_oos_window=False,
        )


def test_prediction_bundle_rejects_empty_oos_test_window() -> None:
    frame = _research_frame()
    spec = PredictionSpec(
        prediction_id="pred_demo",
        model_id="linear_demo",
        research_dataset_id="demo",
        prediction_column=default_prediction_column(),
        primary_key=platform_entity_time_key(),
        train_end="2014-01-02",
        test_start="2099-01-01",
    )

    with pytest.raises(ValueError, match="prediction_oos_test_window_empty"):
        build_prediction_bundle(
            frame,
            spec,
            coefficients=pd.Series([0.0, 1.0]).to_numpy(),
            feature_columns=("mom_20d",),
        )


def test_evaluation_runner_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    processed_root = tmp_path / "data" / "processed" / "a_share"
    dataset_catalog = _research_dataset_catalog()
    assemble_research_dataset(
        _research_frame(),
        dataset_catalog,
        "demo",
        processed_root,
    )
    evaluation_catalog = EvaluationCatalog()
    evaluation_catalog.register(
        EvaluationSpec(
            run_id="eval_demo",
            market_id=MarketId("a_share"),
            research_dataset_id="demo",
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
        )
    )

    run_root = run_evaluation_from_research_dataset(
        processed_root=processed_root,
        dataset_catalog=dataset_catalog,
        evaluation_catalog=evaluation_catalog,
        run_id="eval_demo",
    )

    assert run_root.as_posix().endswith(
        "reports/private_real/a_share/evaluation/eval_demo"
    )
    assert (run_root / "metrics.parquet").is_file()
    assert (run_root / "grouped_returns.parquet").is_file()
    assert (run_root / "manifest.json").is_file()
