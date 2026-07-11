from __future__ import annotations

import inspect
import warnings
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_research_platform.artifacts import _parquet_writers as parquet_writers
from quant_research_platform.artifacts import outputs
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_prediction_spec,
    make_output_edge_report_spec,
)
from tests.helpers.report_output_frames import (
    make_artifact_prediction_output_frame,
)


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


def test_prediction_chunk_writer_has_no_row_proportional_python_key_set() -> None:
    source = inspect.getsource(parquet_writers.write_parquet_chunks)

    assert "seen_keys" not in source


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
    incomplete_model.attrs["model_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            incomplete_model,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    incomplete_prediction = make_artifact_prediction_output_frame()
    incomplete_prediction.attrs["prediction_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_prediction_metadata"):
        outputs.write_prediction_outputs(
            incomplete_prediction,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


@pytest.mark.parametrize("include_generic_metadata", [False, True])
def test_prediction_outputs_require_prediction_metadata_attrs(
    tmp_path: Path,
    include_generic_metadata: bool,
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
    if include_generic_metadata:
        frame.attrs["metadata"] = {"contract_version": "0.1"}

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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
        **make_artifact_prediction_output_frame().attrs[
            "prediction_metadata"
        ],
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
