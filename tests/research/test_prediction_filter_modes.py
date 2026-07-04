from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import assemble_research_dataset
from quant_research_platform.research.predictions import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_processed_inputs import (
    make_filter_processed_catalog,
    write_filter_processed_inputs,
)
from tests.helpers.research_specs import (
    make_research_dataset_catalog,
)
from tests.research.filter_mode_fixtures import (
    make_filter_mode_frame,
)
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
    write_prediction_result,
)


def test_prediction_sample_filter_modes_metadata_and_label_free_generation(
    tmp_path: Path,
) -> None:
    frame = pd.concat(
        [
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        ],
        ignore_index=True,
    )
    spec = PredictionSpec(
        prediction_id="pred_eligible",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )
    prediction = prediction_frame_from_artifact(
        frame.drop(columns=["label"]),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
    )

    assert "metadata" not in prediction.attrs
    assert prediction.attrs["prediction_metadata"] == {
        "prediction_mode": "minimal_oos",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 12,
        "rows_after_filter_mode": 10,
        "rows_after_input_dropna": 10,
        "prediction_rows": 5,
        "train_end": "2024-01-02",
        "test_start": "2024-01-03",
        "requested_test_end": "2024-01-03",
        "effective_test_end": "2024-01-03",
        "is_oos": True,
        "is_final_oos_window": False,
        "input_columns": ["factor"],
        "input_count": 1,
        "prediction_column": PREDICTION_SCORE_COLUMN,
    }
    assert len(prediction) == 5
    assert set(prediction.columns) == {
        "asset_id",
        "timestamp",
        PREDICTION_SCORE_COLUMN,
    }

    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        data_id="demo_filter",
        input_data_ids=("demo_factor", "demo_label", "demo_annotation"),
        base_input_data_id="demo_annotation",
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=("eligible", "can_enter", "can_exit"),
        context_columns=(),
    )
    processed_catalog = make_filter_processed_catalog()
    write_filter_processed_inputs(layout, processed_catalog, frame=frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(spec)

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_eligible",
    )
    output_bundle = write_prediction_result(layout, result)
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["model_metadata"]["input_count"] == 1
    assert manifest["metadata"]["model_metadata"]["coefficient_count"] == 2
    assert manifest["metadata"]["model_metadata"]["sample_filter_mode"] == "eligible"
    assert manifest["metadata"]["model_metadata"]["rows_after_input_label_dropna"] == 4
    assert manifest["metadata"]["model_metadata"]["train_start"] == "2024-01-02"
    assert manifest["metadata"]["model_metadata"]["train_end"] == "2024-01-02"
    assert manifest["metadata"]["prediction_metadata"]["prediction_rows"] == 5
    assert (
        manifest["metadata"]["prediction_metadata"]["sample_filter_mode"] == "eligible"
    )
    assert manifest["metadata"]["prediction_metadata"]["rows_after_input_dropna"] == 10
    assert manifest["metadata"]["prediction_metadata"]["train_end"] == "2024-01-02"
    assert (
        manifest["metadata"]["prediction_metadata"]["requested_test_end"]
        == "2024-01-03"
    )
    assert (
        manifest["metadata"]["prediction_metadata"]["effective_test_end"]
        == "2024-01-03"
    )


def test_prediction_open_ended_oos_model_metadata_uses_effective_test_end(
    tmp_path: Path,
) -> None:
    frame = pd.concat(
        [
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-04")),
        ],
        ignore_index=True,
    )
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        data_id="demo_filter",
        input_data_ids=("demo_factor", "demo_label", "demo_annotation"),
        base_input_data_id="demo_annotation",
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=("eligible", "can_enter", "can_exit"),
        context_columns=(),
    )
    processed_catalog = make_filter_processed_catalog()
    write_filter_processed_inputs(layout, processed_catalog, frame=frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-04"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_open_ended",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_open_ended",
            model_id="linear_open_ended",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            test_end=None,
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_open_ended",
    )
    output_bundle = write_prediction_result(layout, result)
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["prediction_metadata"]["requested_test_end"] is None
    assert (
        manifest["metadata"]["prediction_metadata"]["effective_test_end"]
        == "2024-01-04"
    )
    assert manifest["metadata"]["model_metadata"]["oos_end"] == "2024-01-04"


def test_prediction_handoff_records_window_override_metadata(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        ],
        ignore_index=True,
    )
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        data_id="demo_filter",
        input_data_ids=("demo_factor", "demo_label", "demo_annotation"),
        base_input_data_id="demo_annotation",
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=("eligible", "can_enter", "can_exit"),
        context_columns=(),
    )
    processed_catalog = make_filter_processed_catalog()
    write_filter_processed_inputs(layout, processed_catalog, frame=frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_override",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-01",
            test_start="2024-01-04",
            test_end="2024-01-04",
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_override",
        prediction_train_end="2024-01-02",
        prediction_test_start="2024-01-03",
        prediction_test_end="2024-01-03",
    )
    output_bundle = write_prediction_result(layout, result)
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["train_end"] == "2024-01-02"
    assert manifest["metadata"]["test_start"] == "2024-01-03"
    assert manifest["metadata"]["test_end"] == "2024-01-03"
    assert manifest["metadata"]["model_metadata"]["train_start"] == "2024-01-02"
    assert manifest["metadata"]["model_metadata"]["train_end"] == "2024-01-02"
    assert manifest["metadata"]["prediction_metadata"]["train_end"] == "2024-01-02"
    assert manifest["metadata"]["prediction_metadata"]["test_start"] == "2024-01-03"
    assert (
        manifest["metadata"]["prediction_metadata"]["requested_test_end"]
        == "2024-01-03"
    )
    assert (
        manifest["metadata"]["prediction_metadata"]["effective_test_end"]
        == "2024-01-03"
    )


def test_prediction_handoff_model_fit_does_not_read_test_window_labels(
    tmp_path: Path,
) -> None:
    train_rows = make_filter_mode_frame().assign(
        timestamp=pd.Timestamp("2024-01-02"),
        factor=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        label=[5.0, 8.0, 11.0, 14.0, 17.0, 20.0],
        eligible=True,
    )
    test_rows = make_filter_mode_frame().assign(
        timestamp=pd.Timestamp("2024-01-03"),
        factor=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        label=[-10.0, -20.0, -30.0, -40.0, -50.0, -60.0],
        eligible=True,
    )
    frame = pd.concat([train_rows, test_rows], ignore_index=True)
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        data_id="demo_filter",
        input_data_ids=("demo_factor", "demo_label", "demo_annotation"),
        base_input_data_id="demo_annotation",
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=("eligible", "can_enter", "can_exit"),
        context_columns=(),
    )
    processed_catalog = make_filter_processed_catalog()
    write_filter_processed_inputs(layout, processed_catalog, frame=frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_no_leakage",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            test_end="2024-01-03",
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_no_leakage",
    )
    output_bundle = write_prediction_result(layout, result)
    prediction = pd.read_parquet(output_bundle / "prediction.parquet").sort_values(
        "asset_id",
        kind="mergesort",
    )

    assert prediction[PREDICTION_SCORE_COLUMN].tolist() == pytest.approx(
        [5.0, 8.0, 11.0, 14.0, 17.0, 20.0]
    )


def test_prediction_unfiltered_mode_and_tradable_rejection() -> None:
    spec = PredictionSpec(
        prediction_id="pred_unfiltered",
        model_id="linear_unfiltered",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        sample_filter_mode="unfiltered",
    )
    prediction = prediction_frame_from_artifact(
        make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
    )

    assert "metadata" not in prediction.attrs
    assert prediction.attrs["prediction_metadata"]["rows_after_filter_mode"] == 6
    assert prediction.attrs["prediction_metadata"]["rows_after_input_dropna"] == 6

    with pytest.raises(ValueError, match="unsupported_prediction_sample_filter_mode"):
        PredictionSpec(
            prediction_id="pred_tradable",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            sample_filter_mode="tradable",
        )


def test_prediction_outputs_fail_closed_errors_are_specific() -> None:
    with pytest.raises(
        ValueError,
        match="prediction_train_end_must_precede_test_start",
    ):
        prediction_frame_from_artifact(
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            PredictionSpec(
                prediction_id="pred_bad_order",
                model_id="linear_eligible",
                research_dataset_id="demo_filter",
                train_end="2024-01-03",
                test_start="2024-01-03",
            ),
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("factor",),
        )

    with pytest.raises(
        ValueError,
        match="prediction_test_end_must_not_precede_test_start",
    ):
        prediction_frame_from_artifact(
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            PredictionSpec(
                prediction_id="pred_bad_test_end",
                model_id="linear_eligible",
                research_dataset_id="demo_filter",
                train_end="2024-01-02",
                test_start="2024-01-04",
                test_end="2024-01-03",
            ),
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("factor",),
        )

    with pytest.raises(ValueError, match="empty_prediction_window"):
        prediction_frame_from_artifact(
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            PredictionSpec(
                prediction_id="pred_empty_test",
                model_id="linear_eligible",
                research_dataset_id="demo_filter",
                train_end="2024-01-02",
                test_start="2024-01-03",
                test_end="2024-01-03",
            ),
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("factor",),
        )


def test_prediction_handoff_empty_training_window_fails_closed(
    tmp_path: Path,
) -> None:
    frame = make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03"))
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        data_id="demo_filter",
        input_data_ids=("demo_factor", "demo_label", "demo_annotation"),
        base_input_data_id="demo_annotation",
        factor_columns=("factor",),
        label_columns=("label",),
        filter_columns=("eligible", "can_enter", "can_exit"),
        context_columns=(),
    )
    processed_catalog = make_filter_processed_catalog()
    write_filter_processed_inputs(layout, processed_catalog, frame=frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-03", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_empty_train",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            test_end="2024-01-03",
        )
    )

    with pytest.raises(ValueError, match="empty_training_window"):
        build_prediction_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="pred_empty_train",
        )


def test_prediction_outputs_rejects_filter_columns_as_direct_features() -> None:
    spec = PredictionSpec(
        prediction_id="pred_bad_feature",
        model_id="linear_unfiltered",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        sample_filter_mode="unfiltered",
    )

    with pytest.raises(ValueError, match="filter_column_in_input_columns: eligible"):
        prediction_frame_from_artifact(
            make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            spec,
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("eligible",),
        )
