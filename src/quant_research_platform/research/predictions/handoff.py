"""Spec-driven prediction handoff from ResearchDataset inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.research.datasets import (
    read_research_dataset,
    validate_prediction_input,
)
from quant_research_platform.research.models import fit_linear_baseline
from quant_research_platform.research.predictions.bundle import (
    build_prediction_bundle,
    write_prediction_bundle,
)
from quant_research_platform.storage.layout import layout_from_processed_market_root


def run_prediction_from_research_dataset(
    *,
    processed_root: Path,
    dataset_catalog: ResearchDatasetCatalog,
    model_catalog: ModelCatalog,
    prediction_catalog: PredictionCatalog,
    prediction_id: str,
    years: tuple[int, ...] | None = None,
) -> Path:
    prediction_spec = prediction_catalog.get(prediction_id)
    model_spec = model_catalog.get(prediction_spec.model_id)
    dataset_spec = dataset_catalog.get(prediction_spec.research_dataset_id)
    read_columns = (
        *prediction_spec.primary_key,
        *model_spec.feature_columns,
        model_spec.label_column,
    )
    frame = read_research_dataset(
        processed_root,
        prediction_spec.research_dataset_id,
        years=years,
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
    train_frame = frame.loc[frame["timestamp"] <= train_end].dropna()
    test_frame = frame.loc[frame["timestamp"] >= test_start]
    if train_frame.empty or test_frame.empty:
        raise ValueError("prediction_oos_split_has_empty_side")
    coefficients = fit_linear_baseline(train_frame, model_spec)
    prediction = build_prediction_bundle(
        frame,
        prediction_spec,
        coefficients,
        model_spec.feature_columns,
    )
    layout = layout_from_processed_market_root(processed_root)
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
