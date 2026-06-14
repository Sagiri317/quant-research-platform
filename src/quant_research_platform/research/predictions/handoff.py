"""Spec-driven prediction handoff from ResearchDataset inputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import (
    read_research_dataset,
    validate_prediction_input,
)
from quant_research_platform.research.models import fit_linear_baseline_result
from quant_research_platform.research.predictions.bundle import (
    build_prediction_bundle,
    write_prediction_bundle,
)
from quant_research_platform.storage.layout import StorageLayout


def run_prediction_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    model_catalog: ModelCatalog,
    prediction_catalog: PredictionCatalog,
    prediction_id: str,
    time_window: TimeWindow | None = None,
    prediction_train_end: str | None = None,
    prediction_test_start: str | None = None,
    prediction_test_end: str | None = None,
) -> Path:
    prediction_spec = prediction_catalog.get(prediction_id)
    if (
        prediction_train_end is not None
        or prediction_test_start is not None
        or prediction_test_end is not None
    ):
        prediction_spec = replace(
            prediction_spec,
            train_end=prediction_train_end or prediction_spec.train_end,
            test_start=prediction_test_start or prediction_spec.test_start,
            test_end=(
                prediction_test_end
                if prediction_test_end is not None
                else prediction_spec.test_end
            ),
        )
    model_spec = model_catalog.get(prediction_spec.model_id)
    dataset_spec = dataset_catalog.get(prediction_spec.research_dataset_id)
    read_columns = tuple(
        dict.fromkeys(
            (
                *prediction_spec.primary_key,
                *model_spec.feature_columns,
                model_spec.label_column,
                *required_filter_columns(model_spec.sample_filter_mode),
                *required_filter_columns(prediction_spec.sample_filter_mode),
            )
        )
    )
    frame = read_research_dataset(
        layout,
        market_id=dataset_spec.market_id,
        dataset_id=dataset_spec.data_id,
        time_window=time_window,
        columns=read_columns,
    )
    validate_prediction_input(
        frame,
        dataset_spec,
        feature_columns=model_spec.feature_columns,
    )
    train_end = pd.Timestamp(prediction_spec.train_end)
    test_start = pd.Timestamp(prediction_spec.test_start)
    if train_end >= test_start:
        raise ValueError("prediction_train_end_must_precede_test_start")
    train_frame = frame.loc[frame["timestamp"] <= train_end].copy()
    test_mask = frame["timestamp"] >= test_start
    if prediction_spec.test_end is not None:
        test_mask &= frame["timestamp"] <= pd.Timestamp(prediction_spec.test_end)
    test_frame = frame.loc[test_mask]
    if train_frame.empty or test_frame.empty:
        raise ValueError("prediction_oos_split_has_empty_side")
    model_result = fit_linear_baseline_result(train_frame, model_spec)
    prediction = build_prediction_bundle(
        frame,
        prediction_spec,
        model_result.coefficients,
        model_spec.feature_columns,
    )
    prediction.attrs["model_metadata"] = model_result.metadata
    report_spec = ReportSpec(
        report_id="prediction_bundle",
        report_definition_id="prediction_bundle_private",
        run_id=prediction_spec.prediction_id,
        market_id=dataset_spec.market_id,
        run_type=prediction_spec.run_type,
    )
    return write_prediction_bundle(
        prediction,
        prediction_spec,
        report_spec=report_spec,
        layout=layout,
    )


__all__ = ["run_prediction_from_research_dataset"]
