from __future__ import annotations

import warnings
from copy import deepcopy
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts import (
    _parquet_writers as parquet_writers,
)
from quant_research_platform.artifacts import (
    _prediction_output_validation as validation,
)
from quant_research_platform.artifacts import outputs
from quant_research_platform.artifacts._output_manifest import (
    prediction_output_metadata,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_manifest
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_prediction_spec,
    make_output_edge_report_spec,
)
from tests.helpers.report_output_frames import (
    make_artifact_prediction_output_frame,
)
from tests.helpers.runtime_resources import capture_test_policy


def _valid_prediction_output_metadata_with_selection() -> dict[str, object]:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-03",
        candidate_model_ids=("candidate_model",),
        model_selection_validation_start="2024-01-02",
        model_selection_validation_end="2024-01-02",
    )
    frame = make_artifact_prediction_output_frame()
    model_metadata = deepcopy(frame.attrs["model_metadata"])
    model_metadata.update(
        {
            "test_start": spec.test_start,
            "oos_start": spec.test_start,
            "oos_end": spec.test_start,
        }
    )
    model_selection = _valid_model_selection_metadata()
    model_selection["selected_model_input_columns"] = model_metadata["input_columns"]
    model_metadata["model_selection"] = model_selection
    prediction_metadata = deepcopy(frame.attrs["prediction_metadata"])
    prediction_metadata.update(
        {
            "test_start": spec.test_start,
            "effective_test_end": spec.test_start,
        }
    )
    return prediction_output_metadata(
        spec=spec,
        model_metadata=model_metadata,
        prediction_metadata=prediction_metadata,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "prediction_mode",
        "is_oos",
        "primary_key",
        "sample_filter_mode",
        "model_filter_columns",
        "model_filter_definition",
        "prediction_filter_columns",
        "prediction_filter_definition",
        "noncanonical_final_window",
        "selection_configured_model",
        "selection_candidate_identity",
        "model_algorithm",
        "model_label_input_overlap",
        "model_factor_filter_column",
        "model_feature_filter_column",
        "model_factor_duplicate",
        "model_feature_duplicate",
        "model_factor_feature_overlap",
        "model_inputs_empty",
        "model_random_seed_bool",
        "model_random_seed_float",
        "model_purge_gap_negative",
        "model_purge_gap_bool",
        "model_max_train_rows_zero",
        "model_max_train_rows_bool",
    ),
)
def test_prediction_metadata_reuses_complete_business_owner_graph(
    mutation: str,
) -> None:
    metadata = _valid_prediction_output_metadata_with_selection()
    assert validation.validate_prediction_output_metadata_shape(metadata) is metadata
    metadata = deepcopy(metadata)
    model_metadata = metadata["model_metadata"]
    prediction_metadata = metadata["prediction_metadata"]
    selection = model_metadata["model_selection"]
    if mutation == "prediction_mode":
        metadata["prediction_mode"] = "unknown"
        prediction_metadata["prediction_mode"] = "unknown"
    elif mutation == "is_oos":
        metadata["is_oos"] = False
        prediction_metadata["is_oos"] = False
    elif mutation == "primary_key":
        metadata["primary_key"] = ["asset_id", "timestamp"]
    elif mutation == "sample_filter_mode":
        metadata["sample_filter_mode"] = "tradable"
        for item in (model_metadata, prediction_metadata):
            item["sample_filter_mode"] = "tradable"
            item["filter_columns_used"] = ["eligible", "can_enter", "can_exit"]
            item["filter_definition"] = (
                "eligible == True and can_enter == True and can_exit == True"
            )
    elif mutation == "model_filter_columns":
        model_metadata["filter_columns_used"] = []
    elif mutation == "model_filter_definition":
        model_metadata["filter_definition"] = "wrong"
    elif mutation == "prediction_filter_columns":
        prediction_metadata["filter_columns_used"] = []
    elif mutation == "prediction_filter_definition":
        prediction_metadata["filter_definition"] = "wrong"
    elif mutation == "noncanonical_final_window":
        metadata["test_end"] = metadata["test_start"]
        prediction_metadata["requested_test_end"] = metadata["test_start"]
        model_metadata["test_end"] = metadata["test_start"]
    elif mutation == "selection_configured_model":
        selection["configured_model_id"] = "other_model"
    elif mutation == "selection_candidate_identity":
        selection["candidate_model_ids"][0] = "other_model"
        selection["trial_metrics"][0]["model_id"] = "other_model"
    elif mutation == "model_algorithm":
        model_metadata["model_algorithm"] = ""
    elif mutation == "model_label_input_overlap":
        model_metadata["label_column"] = model_metadata["input_columns"][0]
    elif mutation == "model_factor_filter_column":
        model_metadata["input_factor_columns"] = ["eligible"]
        model_metadata["input_columns"] = [
            "eligible",
            *model_metadata["input_feature_columns"],
        ]
    elif mutation == "model_feature_filter_column":
        model_metadata["input_feature_columns"] = ["can_enter"]
        model_metadata["input_columns"] = [
            *model_metadata["input_factor_columns"],
            "can_enter",
        ]
    elif mutation == "model_factor_duplicate":
        model_metadata["input_factor_columns"] = ["factor", "factor"]
        model_metadata["input_columns"] = [
            "factor",
            "factor",
            *model_metadata["input_feature_columns"],
        ]
    elif mutation == "model_feature_duplicate":
        model_metadata["input_feature_columns"] = ["feature", "feature"]
        model_metadata["input_columns"] = [
            *model_metadata["input_factor_columns"],
            "feature",
            "feature",
        ]
    elif mutation == "model_factor_feature_overlap":
        model_metadata["input_feature_columns"] = ["factor"]
        model_metadata["input_columns"] = [
            *model_metadata["input_factor_columns"],
            "factor",
        ]
    elif mutation == "model_inputs_empty":
        model_metadata["input_factor_columns"] = []
        model_metadata["input_feature_columns"] = []
        model_metadata["input_columns"] = []
    elif mutation == "model_random_seed_bool":
        model_metadata["random_seed"] = True
    elif mutation == "model_random_seed_float":
        model_metadata["random_seed"] = 1.0
    elif mutation == "model_purge_gap_negative":
        model_metadata["purge_gap_steps"] = -1
    elif mutation == "model_purge_gap_bool":
        model_metadata["purge_gap_steps"] = True
    elif mutation == "model_max_train_rows_zero":
        model_metadata["max_train_rows"] = 0
    else:
        model_metadata["max_train_rows"] = True

    if mutation.startswith("model_") and mutation in {
        "model_factor_filter_column",
        "model_feature_filter_column",
        "model_factor_duplicate",
        "model_feature_duplicate",
        "model_factor_feature_overlap",
        "model_inputs_empty",
    }:
        model_metadata["input_count"] = len(model_metadata["input_columns"])
        model_metadata["feature_importance"] = {
            column: 1.0 for column in model_metadata["input_columns"]
        }
        selection["selected_model_input_columns"] = model_metadata["input_columns"]

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        validation.validate_prediction_output_metadata_shape(metadata)


def test_prediction_output_validation_rejects_bad_frames(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)
    valid = make_artifact_prediction_output_frame()

    with pytest.raises(ValueError, match="missing_prediction_output_columns"):
        outputs.write_prediction_outputs(
            valid.drop(columns=["prediction_score"]),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="unregistered_prediction_output_columns"):
        outputs.write_prediction_outputs(
            valid.assign(extra=1),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="duplicate_prediction_keys"):
        outputs.write_prediction_outputs(
            pd.concat([valid, valid], ignore_index=True),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


@pytest.mark.parametrize(
    ("model_algorithm", "dependency_package"),
    (
        ("linear_regression", "sklearn"),
        ("ridge_regression", "internal"),
        ("single_factor_baseline", "sklearn"),
    ),
)
def test_prediction_outputs_bind_dependency_package_to_model_algorithm(
    tmp_path: Path,
    model_algorithm: str,
    dependency_package: str,
) -> None:
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    frame = make_artifact_prediction_output_frame()
    frame.attrs["model_metadata"] = {
        **frame.attrs["model_metadata"],
        "model_algorithm": model_algorithm,
        "dependency_status": {
            "status": "available",
            "package": dependency_package,
            "version": "test",
        },
    }
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            frame,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
        )
    assert not (tmp_path / "outputs" / output_spec.output_id).exists()


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("equal_ids", "output_manifest_output_id_invalid"),
        ("unknown_definition", "output_spec_mismatch"),
    ),
)
def test_prediction_output_identity_failure_never_publishes_bundle(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    if mutation == "equal_ids":
        object.__setattr__(
            output_spec,
            "output_id",
            output_spec.output_definition_id,
        )
    else:
        object.__setattr__(
            output_spec,
            "output_definition_id",
            "unknown_outputs",
        )
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match=error):
        outputs.write_prediction_outputs(
            make_artifact_prediction_output_frame(),
            spec=spec,
            output_spec=output_spec,
            layout=layout,
        )

    assert not (tmp_path / "outputs" / output_spec.output_id).exists()


def test_prediction_output_nested_schema_identity_never_publishes_bundle(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    frame = make_artifact_prediction_output_frame()
    frame.attrs["model_metadata"]["hyperparameters"]["schema_id"] = "forbidden"
    frame.attrs["model_metadata"]["resolved_hyperparameters"]["schema_id"] = "forbidden"
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        outputs.write_prediction_outputs(
            frame,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
        )

    assert not (tmp_path / "outputs" / output_spec.output_id).exists()


def test_prediction_chunk_writer_rejects_cross_chunk_duplicate_and_cleans_scratch(
    tmp_path: Path,
) -> None:
    frame = make_artifact_prediction_output_frame()
    final_path = tmp_path / "prediction.parquet"

    with pytest.raises(ValueError, match="duplicate_prediction_keys"):
        parquet_writers.write_parquet_chunks(
            (frame, frame.copy()),
            final_path,
            required=tuple(frame.columns),
            key_columns=("asset_id", "timestamp"),
            compression="zstd",
        )

    assert not final_path.exists()
    assert not list(tmp_path.glob(".k-*"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_artifact_parquet_writers_admit_before_native_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_artifact_prediction_output_frame()
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1024 * 1024,
    )

    def reject_index(_self):
        raise AssertionError("SQLite state allocated before artifact admission")

    def reject_parquet(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("native parquet state allocated before artifact admission")

    monkeypatch.setattr(ExactPrimaryKeyIndex, "__enter__", reject_index)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", reject_parquet)

    with runtime_resource_policy_scope(policy):
        with policy.reserve("outer", policy.task_memory_budget_bytes):
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                parquet_writers.write_parquet_chunks(
                    (frame,),
                    tmp_path / "chunked.parquet",
                    required=tuple(frame.columns),
                    key_columns=("asset_id", "timestamp"),
                    compression="zstd",
                )
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                parquet_writers.write_parquet(
                    frame,
                    tmp_path / "frame.parquet",
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )


def test_prediction_model_reservation_precedes_writer_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_artifact_prediction_output_frame()
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1024 * 1024,
    )
    retained_model_bytes = 128 * 1024
    writer_bytes = max(64 * 1024, policy.payload_target_bytes // 8)
    original_enter = ExactPrimaryKeyIndex.__enter__

    def assert_model_reserved(index):  # noqa: ANN001, ANN202
        assert policy.reserved_memory_bytes >= retained_model_bytes + writer_bytes
        return original_enter(index)

    def frames():
        with policy.reserve("research.model.fitted_state", retained_model_bytes):
            yield frame

    monkeypatch.setattr(ExactPrimaryKeyIndex, "__enter__", assert_model_reserved)

    with runtime_resource_policy_scope(policy):
        parquet_writers.write_parquet_chunks(
            frames(),
            tmp_path / "prediction.parquet",
            required=tuple(frame.columns),
            key_columns=("asset_id", "timestamp"),
            compression="zstd",
        )


def test_prediction_outputs_require_standard_metadata_attrs(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="missing_model_metadata"):
        outputs.write_prediction_outputs(
            make_artifact_prediction_output_frame(with_metadata=False),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    incomplete_model = make_artifact_prediction_output_frame()
    incomplete_model.attrs["model_metadata"] = {"unexpected": True}
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            incomplete_model,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    incomplete_prediction = make_artifact_prediction_output_frame()
    incomplete_prediction.attrs["prediction_metadata"] = {"unexpected": True}
    with pytest.raises(ValueError, match="invalid_prediction_metadata"):
        outputs.write_prediction_outputs(
            incomplete_prediction,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_require_prediction_metadata_attrs(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)
    frame = make_artifact_prediction_output_frame(with_metadata=False)
    frame.attrs["model_metadata"] = make_artifact_prediction_output_frame().attrs[
        "model_metadata"
    ]

    with pytest.raises(ValueError, match="missing_prediction_metadata"):
        outputs.write_prediction_outputs(
            frame,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_inconsistent_metadata(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_rows = make_artifact_prediction_output_frame()
    bad_rows.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "prediction_rows": 2,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_rows,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_window = make_artifact_prediction_output_frame()
    bad_window.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "test_start": "2024-01-03",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_window,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_train_end = make_artifact_prediction_output_frame()
    bad_train_end.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "train_end": "2024-01-02",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_train_end,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model = make_artifact_prediction_output_frame()
    bad_model.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "coefficient_count": 0,
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            bad_model,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_feature_columns = make_artifact_prediction_output_frame()
    bad_feature_columns.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "input_columns": ["factor", "other_input"],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_feature_columns,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_tampered_selected_model_input_columns(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-03",
        candidate_model_ids=("candidate_model",),
        model_selection_validation_start="2024-01-02",
        model_selection_validation_end="2024-01-02",
    )
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    tampered = make_artifact_prediction_output_frame().assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    tampered.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "test_start": "2024-01-03",
        "oos_start": "2024-01-03",
        "oos_end": "2024-01-03",
        "model_selection": {
            "status": "selected",
            "configured_model_id": "model_demo",
            "selected_model_id": "candidate_model",
            "selected_model_input_columns": ["other_input"],
            "candidate_model_ids": ["model_demo", "candidate_model"],
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "validation_start": "2024-01-02",
            "validation_end": "2024-01-02",
            "candidate_count": 2,
            "trial_metrics": [
                {
                    "model_id": "model_demo",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 1.0,
                },
                {
                    "model_id": "candidate_model",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 0.0,
                },
            ],
        },
    }
    tampered.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "test_start": "2024-01-03",
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            tampered,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


@pytest.mark.parametrize(
    "feature_importance",
    (
        {"factor": 0.1},
        {"factor": 0.1, "feature": 0.2, "extra": 0.3},
        {"unrelated": 1.0},
    ),
    ids=("missing", "extra", "unrelated"),
)
@pytest.mark.parametrize("chunked", (False, True), ids=("direct", "chunked"))
def test_prediction_outputs_bind_feature_importance_to_exact_model_inputs(
    tmp_path: Path,
    feature_importance: dict[str, float],
    chunked: bool,
) -> None:
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    frame = make_artifact_prediction_output_frame()
    model_metadata = {
        **frame.attrs["model_metadata"],
        "feature_importance": feature_importance,
    }
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="invalid_model_metadata"):
        if chunked:
            outputs.write_prediction_outputs(
                spec=spec,
                output_spec=output_spec,
                layout=layout,
                output_chunks=(frame,),
                model_metadata=model_metadata,
                prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
            )
        else:
            frame.attrs["model_metadata"] = model_metadata
            outputs.write_prediction_outputs(
                frame,
                spec=spec,
                output_spec=output_spec,
                layout=layout,
            )

    assert not (tmp_path / "outputs" / output_spec.output_id).exists()


def test_prediction_outputs_preserve_six_digit_aggregate_counts(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    frame = make_artifact_prediction_output_frame()
    model_metadata = {
        **frame.attrs["model_metadata"],
        "rows_input": 687_688,
        "rows_after_filter_mode": 600_000,
        "rows_after_input_label_dropna": 5_000,
    }

    output_root = outputs.write_prediction_outputs(
        spec=spec,
        output_spec=output_spec,
        layout=StorageLayout(project_root=tmp_path),
        output_chunks=(frame,),
        model_metadata=model_metadata,
        prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
    )

    persisted = read_manifest(output_root / "manifest.json")
    persisted_model = persisted["metadata"]["model_metadata"]
    assert persisted_model["rows_input"] == 687_688
    assert persisted_model["rows_after_filter_mode"] == 600_000
    assert persisted_model["rows_after_input_label_dropna"] == 5_000


def test_prediction_outputs_reject_tampered_model_selection_direction(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-03",
        candidate_model_ids=("candidate_model",),
        model_selection_validation_start="2024-01-02",
        model_selection_validation_end="2024-01-02",
    )
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    tampered = make_artifact_prediction_output_frame().assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    tampered.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "test_start": "2024-01-03",
        "oos_start": "2024-01-03",
        "oos_end": "2024-01-03",
        "model_selection": {
            "status": "selected",
            "configured_model_id": "model_demo",
            "selected_model_id": "candidate_model",
            "selected_model_input_columns": ["factor"],
            "candidate_model_ids": ["model_demo", "candidate_model"],
            "selection_metric": "validation_rankic",
            "selection_direction": "minimize",
            "validation_start": "2024-01-02",
            "validation_end": "2024-01-02",
            "candidate_count": 2,
            "trial_metrics": [
                {
                    "model_id": "model_demo",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 1.0,
                },
                {
                    "model_id": "candidate_model",
                    "validation_rows": 1,
                    "validation_ic": 0.0,
                    "validation_rankic": 0.0,
                    "validation_mse": 0.0,
                },
            ],
        },
    }
    tampered.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "test_start": "2024-01-03",
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            tampered,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )
    assert not (tmp_path / "outputs" / report_spec.output_id).exists()


def _valid_model_selection_metadata() -> dict[str, object]:
    return {
        "status": "selected",
        "configured_model_id": "model_demo",
        "selected_model_id": "candidate_model",
        "selected_model_input_columns": ["factor"],
        "candidate_model_ids": ["model_demo", "candidate_model"],
        "selection_metric": "validation_rankic",
        "selection_direction": "maximize",
        "validation_start": "2024-01-02",
        "validation_end": "2024-01-02",
        "candidate_count": 2,
        "trial_metrics": [
            {
                "model_id": "model_demo",
                "validation_rows": 1,
                "validation_ic": 0.0,
                "validation_rankic": 0.0,
                "validation_mse": 1.0,
            },
            {
                "model_id": "candidate_model",
                "validation_rows": 1,
                "validation_ic": 0.0,
                "validation_rankic": 0.0,
                "validation_mse": 0.0,
            },
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "candidate_missing",
        "candidate_extra",
        "candidate_duplicate",
        "trial_missing",
        "trial_extra",
        "trial_duplicate",
        "trial_out_of_order",
        "trial_field_missing",
        "trial_field_extra",
        "trial_validation_rows_bool",
        "trial_validation_rows_string",
        "trial_validation_rows_numpy",
        "trial_validation_rows_negative",
        "candidate_count_bool",
        "candidate_count_string",
        "candidate_count_numpy",
        "candidate_count_mismatch",
        "trial_metric_bool",
        "trial_metric_nan",
        "trial_metric_infinity",
    ),
)
def test_model_selection_candidate_trial_exact_mutation_matrix(
    mutation: str,
) -> None:
    metadata = deepcopy(_valid_model_selection_metadata())
    candidates = metadata["candidate_model_ids"]
    trials = metadata["trial_metrics"]
    if mutation == "candidate_missing":
        candidates.pop()
    elif mutation == "candidate_extra":
        candidates.append("extra_model")
    elif mutation == "candidate_duplicate":
        candidates[1] = candidates[0]
    elif mutation == "trial_missing":
        trials.pop()
    elif mutation == "trial_extra":
        trials.append(deepcopy(trials[0]))
    elif mutation == "trial_duplicate":
        trials[1]["model_id"] = trials[0]["model_id"]
    elif mutation == "trial_out_of_order":
        trials.reverse()
    elif mutation == "trial_field_missing":
        trials[0].pop("validation_mse")
    elif mutation == "trial_field_extra":
        trials[0]["unexpected"] = 0
    elif mutation == "trial_validation_rows_bool":
        trials[0]["validation_rows"] = True
    elif mutation == "trial_validation_rows_string":
        trials[0]["validation_rows"] = "1"
    elif mutation == "trial_validation_rows_numpy":
        trials[0]["validation_rows"] = np.int64(1)
    elif mutation == "trial_validation_rows_negative":
        trials[0]["validation_rows"] = -1
    elif mutation == "candidate_count_bool":
        metadata["candidate_count"] = True
    elif mutation == "candidate_count_string":
        metadata["candidate_count"] = "2"
    elif mutation == "candidate_count_numpy":
        metadata["candidate_count"] = np.int64(2)
    elif mutation == "candidate_count_mismatch":
        metadata["candidate_count"] = 1
    elif mutation == "trial_metric_bool":
        trials[0]["validation_mse"] = True
    elif mutation == "trial_metric_nan":
        trials[0]["validation_mse"] = float("nan")
    else:
        trials[0]["validation_mse"] = float("inf")

    with pytest.raises(ValueError, match="invalid_model_metadata"):
        validation.validate_model_selection_metadata_shape(metadata)


def test_prediction_outputs_reject_bad_row_count_chain_and_filter_contract(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_prediction_counts = make_artifact_prediction_output_frame()
    bad_prediction_counts.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "rows_after_input_dropna": 0,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_prediction_counts,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model_counts = make_artifact_prediction_output_frame()
    bad_model_counts.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "rows_after_filter_mode": 1,
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_counts,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_prediction_filter = make_artifact_prediction_output_frame()
    bad_prediction_filter.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "filter_columns_used": [],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_prediction_filter,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model_filter = make_artifact_prediction_output_frame()
    bad_model_filter.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "filter_definition": "no eligible/can_enter/can_exit filter",
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_filter,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_bad_effective_test_end(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_effective_end = make_artifact_prediction_output_frame()
    bad_effective_end.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_effective_end,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_unparseable_prediction_timestamp(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_timestamp = make_artifact_prediction_output_frame()
    bad_timestamp["timestamp"] = ["bad-date"]

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_timestamp,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_accept_equivalent_timezone_representations(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    original = make_artifact_prediction_output_frame()
    frame = pd.concat(
        [original, original.assign(asset_id="asset_b")],
        ignore_index=True,
    )
    frame["timestamp"] = [
        pd.Timestamp(
            "2024-01-02T12:00:00",
            tz=timezone(timedelta(hours=1)),
        ),
        pd.Timestamp(
            "2024-01-02T12:00:00",
            tz=ZoneInfo("Europe/Paris"),
        ),
    ]
    frame.attrs = original.attrs
    frame.attrs["prediction_metadata"] = {
        **original.attrs["prediction_metadata"],
        "prediction_rows": 2,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outputs.write_prediction_outputs(
            frame,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_chunked_prediction_outputs_accept_equivalent_timezone_representations(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    original = make_artifact_prediction_output_frame()
    chunk = pd.concat(
        [original, original.assign(asset_id="asset_b")],
        ignore_index=True,
    )
    chunk["timestamp"] = [
        pd.Timestamp(
            "2024-01-02T12:00:00",
            tz=timezone(timedelta(hours=1)),
        ),
        pd.Timestamp(
            "2024-01-02T12:01:00",
            tz=ZoneInfo("Europe/Paris"),
        ),
    ]
    prediction_metadata = {
        **original.attrs["prediction_metadata"],
        "prediction_rows": 2,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outputs.write_prediction_outputs(
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
            output_chunks=(chunk,),
            model_metadata=original.attrs["model_metadata"],
            prediction_metadata_factory=lambda: prediction_metadata,
        )


def test_prediction_outputs_reject_effective_test_end_after_requested_end(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    late_prediction = make_artifact_prediction_output_frame().assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    late_prediction.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "test_end": "2024-01-02",
        "oos_end": "2024-01-02",
    }
    late_prediction.attrs["prediction_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["prediction_metadata"],
        "requested_test_end": "2024-01-02",
        "effective_test_end": "2024-01-03",
        "is_final_oos_window": False,
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            late_prediction,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_model_train_window_after_spec_train_end(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_prediction_spec()
    report_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_model_window = make_artifact_prediction_output_frame()
    bad_model_window.attrs["model_metadata"] = {
        **make_artifact_prediction_output_frame().attrs["model_metadata"],
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
    }

    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_window,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )
