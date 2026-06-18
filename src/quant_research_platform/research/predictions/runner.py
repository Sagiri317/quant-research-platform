"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import validate_prediction_input
from quant_research_platform.research.datasets.io import _iter_research_dataset_chunks
from quant_research_platform.research.models.supervised import (
    _fit_model_result_from_frames,
)
from quant_research_platform.research.predictions.artifacts import (
    _write_prediction_artifacts_from_frames,
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
    train_read_columns = tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                *model_spec.input_columns,
                model_spec.label_column,
                *required_filter_columns(model_spec.sample_filter_mode),
            )
        )
    )
    prediction_read_columns = tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                *model_spec.input_columns,
                *required_filter_columns(prediction_spec.sample_filter_mode),
            )
        )
    )

    def _chunks(read_columns: tuple[str, ...]) -> Iterable[pd.DataFrame]:
        return _validated_prediction_chunks(
            _iter_research_dataset_chunks(
                layout,
                market_id=dataset_spec.market_id,
                dataset_id=dataset_spec.data_id,
                time_window=time_window,
                columns=read_columns,
            ),
            dataset_spec=dataset_spec,
            input_columns=model_spec.input_columns,
        )

    _validate_prediction_window_order(prediction_spec)
    train_end = pd.Timestamp(prediction_spec.train_end)
    try:
        model_result = _fit_model_result_from_frames(
            _prediction_train_chunks(
                _chunks(train_read_columns),
                train_end=train_end,
            ),
            model_spec,
        )
    except ValueError as exc:
        if str(exc) == "model_training_sample_empty":
            raise ValueError("empty_training_window") from exc
        raise
    report_spec = ReportSpec(
        report_id="prediction_artifacts",
        report_definition_id="prediction_artifacts_private",
        run_id=prediction_spec.prediction_id,
        market_id=dataset_spec.market_id,
        run_type=prediction_spec.run_type,
    )
    return _write_prediction_artifacts_from_frames(
        _chunks(prediction_read_columns),
        prediction_spec,
        model_result.coefficients,
        model_spec.input_columns,
        prediction_model=model_result.prediction_model,
        model_metadata=model_result.metadata,
        report_spec=report_spec,
        layout=layout,
    )


def run_predictions_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    model_catalog: ModelCatalog,
    prediction_catalog: PredictionCatalog,
    prediction_ids: Iterable[str],
    time_window: TimeWindow | None = None,
    prediction_train_end: str | None = None,
    prediction_test_start: str | None = None,
    prediction_test_end: str | None = None,
) -> tuple[Path, ...]:
    return tuple(
        run_prediction_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id=prediction_id,
            time_window=time_window,
            prediction_train_end=prediction_train_end,
            prediction_test_start=prediction_test_start,
            prediction_test_end=prediction_test_end,
        )
        for prediction_id in prediction_ids
    )


def _validated_prediction_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    dataset_spec,
    input_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    for chunk in chunks:
        validate_prediction_input(
            chunk,
            dataset_spec,
            input_columns=input_columns,
        )
        yield chunk


def _validate_prediction_window_order(spec) -> None:
    train_end = pd.Timestamp(spec.train_end)
    test_start = pd.Timestamp(spec.test_start)
    if train_end >= test_start:
        raise ValueError("prediction_train_end_must_precede_test_start")
    test_end = pd.Timestamp(spec.test_end) if spec.test_end is not None else None
    if test_end is not None and test_end < test_start:
        raise ValueError("prediction_test_end_must_not_precede_test_start")


def _prediction_train_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    train_end: pd.Timestamp,
) -> Iterable[pd.DataFrame]:
    for chunk in chunks:
        timestamps = _prediction_timestamps(chunk)
        train = chunk.loc[timestamps <= train_end]
        if not train.empty:
            yield train


def _prediction_timestamps(chunk: pd.DataFrame) -> pd.Series:
    if "timestamp" not in chunk.columns:
        raise ValueError("missing_prediction_columns: timestamp")
    timestamps = pd.to_datetime(chunk["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_prediction_timestamp_values")
    return timestamps


__all__ = [
    "run_prediction_from_research_dataset",
    "run_predictions_from_research_dataset",
]
