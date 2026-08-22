from __future__ import annotations

import json
import warnings
from contextlib import nullcontext
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts._output_common_validation import (
    validate_output_frame,
)
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
    _validate_prediction_output_values,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.predictions.outputs import (
    build_prediction_result_from_frames,
)
from quant_research_platform.research.predictions.runner import (
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
from tests.helpers.runtime_resources import capture_test_policy
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_input", True),
        ("rows_input", 1.5),
        ("rows_input", -1),
        ("rows_after_filter_mode", True),
        ("rows_after_filter_mode", 1.5),
        ("rows_after_filter_mode", -1),
    ],
)
def test_prediction_build_rejects_invalid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    metadata = {
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
    }
    metadata[field] = value

    def fake_filter_mode_scope(frame: pd.DataFrame, **_: object):
        return nullcontext((frame, metadata))

    monkeypatch.setattr(
        "quant_research_platform.research.predictions.outputs._filter_mode_scope",
        fake_filter_mode_scope,
    )
    spec = PredictionSpec(
        prediction_id="pred_bad_metadata_count",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )
    frame = make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03"))
    result = build_prediction_result_from_frames(
        (frame.drop(columns=["label"]),),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
        model_metadata=_prediction_model_metadata(),
    )

    with pytest.raises(ValueError, match="invalid_prediction_metadata_count"):
        tuple(result.output_chunks)


def _prediction_model_metadata() -> dict[str, object]:
    return {
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
        "test_start": None,
        "test_end": None,
        "oos_start": None,
        "oos_end": None,
        "input_factor_columns": ["factor"],
        "input_feature_columns": [],
        "input_columns": ["factor"],
        "input_count": 1,
        "label_column": "label",
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": None,
        "max_train_rows": None,
        "purge_gap_steps": 0,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
        "coefficient_count": 2,
        "feature_importance": {"factor": 0.1},
    }


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


def test_prediction_outputs_accept_equivalent_timezone_representations() -> None:
    spec = PredictionSpec(
        prediction_id="pred_equivalent_timezones",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-02T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                pd.Timestamp(
                    "2024-01-02T12:00:00",
                    tz=ZoneInfo("Europe/Paris"),
                ),
            ],
            "asset_id": ["asset_a", "asset_b"],
            PREDICTION_SCORE_COLUMN: [0.1, 0.2],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_prediction_output_values(
            frame,
            primary_key=spec.primary_key,
            prediction_column=spec.prediction_column,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("nan", "invalid_prediction_values"),
        ("numeric_text", "invalid_prediction_values"),
        ("positive_inf", "invalid_prediction_values"),
        ("negative_inf", "invalid_prediction_values"),
        ("null_timestamp", "invalid_prediction_keys"),
        ("null_asset", "invalid_prediction_keys"),
        ("duplicate_key", "duplicate_prediction_keys"),
    ),
)
def test_prediction_outputs_require_finite_values_and_complete_unique_keys(
    mutation: str,
    error: str,
) -> None:
    spec = PredictionSpec(
        prediction_id="pred_strict_values",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "asset_id": ["asset_a", "asset_b"],
            PREDICTION_SCORE_COLUMN: [0.1, 0.2],
        }
    )
    if mutation == "nan":
        frame.loc[0, PREDICTION_SCORE_COLUMN] = np.nan
    elif mutation == "numeric_text":
        frame[PREDICTION_SCORE_COLUMN] = ["0.1", "0.2"]
    elif mutation == "positive_inf":
        frame.loc[0, PREDICTION_SCORE_COLUMN] = np.inf
    elif mutation == "negative_inf":
        frame.loc[0, PREDICTION_SCORE_COLUMN] = -np.inf
    elif mutation == "null_timestamp":
        frame.loc[0, "timestamp"] = pd.NaT
    elif mutation == "null_asset":
        frame.loc[0, "asset_id"] = None
    else:
        frame.loc[1, ["timestamp", "asset_id"]] = frame.loc[
            0, ["timestamp", "asset_id"]
        ]

    with pytest.raises(ValueError, match=f"^{error}$"):
        _validate_prediction_output_values(
            frame,
            primary_key=spec.primary_key,
            prediction_column=spec.prediction_column,
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
        policy = capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024**2,
        )
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
            resource_policy=policy,
        )
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("non_string", "invalid_prediction_output_columns"),
        ("duplicate_key", "duplicate_prediction_output_columns"),
        ("duplicate_payload", "duplicate_prediction_output_columns"),
        ("missing", "missing_prediction_output_columns: prediction_score"),
        ("extra", "unregistered_prediction_output_columns: extra"),
        ("misordered", "misordered_prediction_output_columns"),
    ),
)
def test_prediction_outputs_require_exact_ordered_unique_columns(
    mutation: str,
    error: str,
) -> None:
    spec = PredictionSpec(
        prediction_id="pred_exact_output_columns",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "asset_id": ["asset_a"],
            PREDICTION_SCORE_COLUMN: [0.1],
        }
    )
    if mutation == "non_string":
        frame = frame.rename(columns={"timestamp": 1})
    elif mutation == "duplicate_key":
        frame = pd.concat((frame, frame.loc[:, ["timestamp"]]), axis=1)
    elif mutation == "duplicate_payload":
        frame = pd.concat(
            (frame, frame.loc[:, [PREDICTION_SCORE_COLUMN]]),
            axis=1,
        )
    elif mutation == "missing":
        frame = frame.drop(columns=[PREDICTION_SCORE_COLUMN])
    elif mutation == "extra":
        frame = frame.assign(extra=1)
    elif mutation == "misordered":
        frame = frame.loc[:, ["asset_id", "timestamp", PREDICTION_SCORE_COLUMN]]

    with pytest.raises(ValueError, match=f"^{error}$"):
        validate_output_frame(
            frame,
            required=(*spec.primary_key, spec.prediction_column),
            error_name="prediction_output",
        )


def test_prediction_builder_outputs_canonical_columns_for_validation() -> None:
    spec = PredictionSpec(
        prediction_id="pred_canonical_builder_output",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )
    frame = make_filter_mode_frame().assign(timestamp=pd.Timestamp("2024-01-03"))
    result = build_prediction_result_from_frames(
        (frame.drop(columns=["label"]),),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
        model_metadata=_prediction_model_metadata(),
    )
    output = pd.concat(tuple(result.output_chunks), ignore_index=True)

    assert tuple(output.columns) == (*spec.primary_key, spec.prediction_column)
    validate_output_frame(
        output,
        required=(*spec.primary_key, spec.prediction_column),
        error_name="prediction_output",
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

    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024**2,
    )
    with pytest.raises(ValueError, match="empty_training_window"):
        build_prediction_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="pred_empty_train",
            resource_policy=policy,
        )
    assert policy.used_memory_bytes == 0


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
