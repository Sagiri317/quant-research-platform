from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.research.predictions.artifacts import (
    _write_prediction_artifacts_from_frames,
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
        prediction, _, _ = write_prediction_artifact_and_read(
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
        prediction, _, _ = write_prediction_artifact_and_read(
            Path(temp_dir),
            frames,
            spec,
            coefficients,
            input_columns,
            prediction_model=prediction_model,
            model_metadata=model_metadata,
        )
        return prediction


def write_prediction_artifact_and_read(
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
    report_spec = ReportSpec(
        report_id="prediction_artifacts",
        report_definition_id="prediction_artifacts_private",
        run_id=spec.prediction_id,
        market_id=MarketId("a_share"),
        run_type=spec.run_type,
    )
    run_root = _write_prediction_artifacts_from_frames(
        frames,
        spec,
        coefficients,
        input_columns,
        prediction_model=prediction_model,
        model_metadata=model_metadata
        or synthetic_prediction_model_metadata(spec, input_columns),
        report_spec=report_spec,
        layout=layout,
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    prediction = pd.read_parquet(run_root / "prediction.parquet")
    prediction.attrs["model_metadata"] = manifest["model_metadata"]
    prediction.attrs["prediction_metadata"] = manifest["prediction_metadata"]
    return prediction, run_root, manifest


def synthetic_prediction_model_metadata(
    spec: PredictionSpec,
    input_columns: tuple[str, ...],
) -> dict[str, object]:
    return {
        "contract_version": "0.1",
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
        "random_seed": None,
        "max_train_rows": 1,
        "purge_gap_steps": 0,
        "dependency_status": {"status": "available", "package": "synthetic_test"},
        "coefficient_count": len(input_columns) + 1,
        "feature_importance": {column: 1.0 for column in input_columns},
    }
