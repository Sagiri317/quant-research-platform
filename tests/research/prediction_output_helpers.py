from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quant_research_platform.artifacts._prediction_outputs import (
    write_prediction_outputs as write_prediction_outputs_in_workflow,
)
from quant_research_platform.artifacts.outputs import write_prediction_outputs
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.research.predictions.outputs import (
    build_prediction_result_from_frames,
)
from quant_research_platform.storage.layout import StorageLayout


def prediction_frame_from_artifact(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    *,
    prediction_model: object | None = None,
    model_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    with TemporaryDirectory() as temp_dir:
        prediction, _, _ = write_prediction_output_and_read(
            Path(temp_dir),
            (frame,),
            spec,
            coefficients,
            input_columns,
            prediction_model=prediction_model,
            model_metadata=model_metadata,
        )
        return prediction


def prediction_frame_from_artifact_frames(
    frames: tuple[pd.DataFrame, ...],
    spec: PredictionSpec,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    *,
    prediction_model: object | None = None,
    model_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    with TemporaryDirectory() as temp_dir:
        prediction, _, _ = write_prediction_output_and_read(
            Path(temp_dir),
            frames,
            spec,
            coefficients,
            input_columns,
            prediction_model=prediction_model,
            model_metadata=model_metadata,
        )
        return prediction


def write_prediction_output_and_read(
    project_root: Path,
    frames: tuple[pd.DataFrame, ...],
    spec: PredictionSpec,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    *,
    prediction_model: object | None = None,
    model_metadata: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    layout = StorageLayout(project_root=project_root)
    output_spec = OutputSpec(
        output_id=spec.prediction_id,
        output_definition_id="prediction_outputs",
        run_id="test_run",
        market_id=MarketId("a_share"),
        run_type=spec.run_type,
        scope=OutputScope.LOCAL_ONLY,
    )
    result = build_prediction_result_from_frames(
        frames,
        spec,
        coefficients,
        input_columns,
        prediction_model=prediction_model,
        model_metadata=model_metadata
        or synthetic_prediction_model_metadata(spec, input_columns),
    )
    output_bundle = write_prediction_outputs(
        spec=result.spec,
        output_spec=output_spec,
        layout=layout,
        output_chunks=result.output_chunks,
        model_metadata=result.model_metadata,
        prediction_metadata_factory=result.prediction_metadata_factory,
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    prediction = pd.read_parquet(output_bundle / "prediction.parquet")
    prediction.attrs["model_metadata"] = manifest["metadata"]["model_metadata"]
    prediction.attrs["prediction_metadata"] = manifest["metadata"][
        "prediction_metadata"
    ]
    return prediction, output_bundle, manifest


def write_prediction_result(layout: StorageLayout, result) -> Path:
    output_spec = OutputSpec(
        output_id=result.spec.prediction_id,
        output_definition_id="prediction_outputs",
        run_id="test_run",
        market_id=MarketId("a_share"),
        run_type=result.spec.run_type,
        scope=OutputScope.LOCAL_ONLY,
    )
    return write_prediction_outputs_in_workflow(
        spec=result.spec,
        output_spec=output_spec,
        layout=layout,
        output_chunks=result.output_chunks,
        model_metadata=result.model_metadata,
        prediction_metadata_factory=result.prediction_metadata_factory,
    )


def synthetic_prediction_model_metadata(
    spec: PredictionSpec,
    input_columns: tuple[str, ...],
) -> dict[str, object]:
    return {
        "model_algorithm": "linear_regression",
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": list(required_filter_columns(spec.sample_filter_mode)),
        "filter_definition": filter_definition(spec.sample_filter_mode),
        "rows_input": 1,
        "rows_after_filter_mode": 1,
        "rows_after_input_label_dropna": 1,
        "train_start": spec.train_end,
        "train_end": spec.train_end,
        "test_start": None,
        "test_end": None,
        "oos_start": None,
        "oos_end": None,
        "input_factor_columns": list(input_columns),
        "input_feature_columns": [],
        "input_columns": list(input_columns),
        "input_count": len(input_columns),
        "label_column": "label",
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": None,
        "max_train_rows": 1,
        "purge_gap_steps": 0,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "1",
        },
        "coefficient_count": len(input_columns) + 1,
        "feature_importance": {column: 1.0 for column in input_columns},
    }
