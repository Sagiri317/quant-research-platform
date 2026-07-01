"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import validate_prediction_input
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    TEST_METRIC_COLUMNS,
    VALIDATION_METRIC_COLUMNS,
    build_model_selection_metadata,
    evaluate_model_trial_metrics,
    select_best_validation_trial,
)
from quant_research_platform.research.predictions.outputs import (
    PredictionResult,
    build_prediction_result_from_frames,
    prediction_callable,
)
from quant_research_platform.storage.layout import StorageLayout


def build_prediction_result_from_research_dataset(
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
) -> PredictionResult:
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

    def _chunks(
        read_columns: tuple[str, ...],
        *,
        input_columns: tuple[str, ...],
    ) -> Iterable[pd.DataFrame]:
        return _validated_prediction_chunks(
            iter_research_dataset_chunks(
                layout,
                market_id=dataset_spec.market_id,
                dataset_id=dataset_spec.data_id,
                time_window=time_window,
                columns=read_columns,
            ),
            dataset_spec=dataset_spec,
            input_columns=input_columns,
        )

    _validate_prediction_window_order(prediction_spec)
    model_selection_metadata: dict[str, object] | None = None
    if prediction_spec.model_selection_enabled:
        model_spec, model_selection_metadata = _select_prediction_model_spec(
            prediction_spec=prediction_spec,
            dataset_spec=dataset_spec,
            model_catalog=model_catalog,
            chunks=_chunks,
        )
    train_read_columns = _prediction_train_read_columns(dataset_spec, model_spec)
    prediction_read_columns = _prediction_read_columns(
        dataset_spec,
        prediction_spec,
        model_spec,
    )
    train_end = pd.Timestamp(prediction_spec.train_end)
    try:
        model_result = fitting.fit_model_result_from_frames(
            _prediction_train_chunks(
                _chunks(train_read_columns, input_columns=model_spec.input_columns),
                train_end=train_end,
            ),
            model_spec,
            estimator=build_estimator_from_model_spec(model_spec),
        )
    except ValueError as exc:
        if str(exc) == "model_training_sample_empty":
            raise ValueError("empty_training_window") from exc
        raise
    model_metadata = model_result.metadata
    if model_selection_metadata is not None:
        model_metadata = {**model_metadata, "model_selection": model_selection_metadata}
    return build_prediction_result_from_frames(
        _chunks(prediction_read_columns, input_columns=model_spec.input_columns),
        prediction_spec,
        model_result.coefficients,
        model_spec.input_columns,
        prediction_model=model_result.prediction_model,
        model_metadata=model_metadata,
    )


def build_prediction_results_from_research_dataset(
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
) -> tuple[PredictionResult, ...]:
    return tuple(
        build_prediction_result_from_research_dataset(
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


def _prediction_train_read_columns(dataset_spec, model_spec) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                *model_spec.input_columns,
                model_spec.label_column,
                *required_filter_columns(model_spec.sample_filter_mode),
            )
        )
    )


def _prediction_read_columns(
    dataset_spec,
    prediction_spec,
    model_spec,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                *model_spec.input_columns,
                *required_filter_columns(prediction_spec.sample_filter_mode),
            )
        )
    )


def _select_prediction_model_spec(
    *,
    prediction_spec,
    dataset_spec,
    model_catalog: ModelCatalog,
    chunks,
):
    candidate_specs = tuple(
        model_catalog.get(model_id)
        for model_id in (prediction_spec.model_id, *prediction_spec.candidate_model_ids)
    )
    label_columns = {spec.label_column for spec in candidate_specs}
    if len(label_columns) != 1:
        raise ValueError("model_selection_label_column_mismatch")
    label_column = candidate_specs[0].label_column
    validation_start = pd.Timestamp(prediction_spec.model_selection_validation_start)
    validation_end = pd.Timestamp(prediction_spec.model_selection_validation_end)
    trial_metric_records: list[dict[str, object]] = []
    for candidate_spec in candidate_specs:
        train_read_columns = _prediction_train_read_columns_from_inputs(
            dataset_spec.primary_key,
            candidate_spec.input_columns,
            label_column=label_column,
            sample_filter_mode=candidate_spec.sample_filter_mode,
        )
        model_result = fitting.fit_model_result_from_frames(
            _prediction_train_chunks(
                chunks(train_read_columns, input_columns=candidate_spec.input_columns),
                train_end=pd.Timestamp(prediction_spec.train_end),
            ),
            candidate_spec,
            estimator=build_estimator_from_model_spec(candidate_spec),
        )
        validation_read_columns = tuple(
            dict.fromkeys(
                (
                    *dataset_spec.primary_key,
                    *candidate_spec.input_columns,
                    label_column,
                    *required_filter_columns(prediction_spec.sample_filter_mode),
                )
            )
        )
        validation_frame = _validation_prediction_frame(
            chunks(validation_read_columns, input_columns=candidate_spec.input_columns),
            prediction_spec=prediction_spec,
            candidate_spec=candidate_spec,
            model_result=model_result,
            validation_start=validation_start,
            validation_end=validation_end,
        )
        trial_metric_records.append(
            evaluate_model_trial_metrics(
                validation_frame,
                model_id=candidate_spec.model_id,
                label_column=label_column,
                prediction_column=prediction_spec.prediction_column,
            )
        )
    trial_metrics = pd.DataFrame.from_records(trial_metric_records)
    direction = (
        MODEL_SELECTION_METRIC_DIRECTIONS[prediction_spec.model_selection_metric]
        == "maximize"
    )
    selected_model_id = select_best_validation_trial(
        trial_metrics,
        trial_id_column="model_id",
        selection_metric=prediction_spec.model_selection_metric,
        maximize=direction,
        validation_metric_columns=VALIDATION_METRIC_COLUMNS,
        test_metric_columns=TEST_METRIC_COLUMNS,
    )
    metadata = build_model_selection_metadata(
        configured_model_id=prediction_spec.model_id,
        selected_model_id=selected_model_id,
        candidate_model_ids=tuple(spec.model_id for spec in candidate_specs),
        selection_metric=prediction_spec.model_selection_metric,
        validation_start=prediction_spec.model_selection_validation_start,
        validation_end=prediction_spec.model_selection_validation_end,
        selected_model_input_columns=model_catalog.get(selected_model_id).input_columns,
        trial_metrics=trial_metrics,
    )
    return model_catalog.get(selected_model_id), metadata


def _prediction_train_read_columns_from_inputs(
    primary_key: tuple[str, ...],
    input_columns: tuple[str, ...],
    *,
    label_column: str,
    sample_filter_mode,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *primary_key,
                *input_columns,
                label_column,
                *required_filter_columns(sample_filter_mode),
            )
        )
    )


def _validation_prediction_frame(
    chunks: Iterable[pd.DataFrame],
    *,
    prediction_spec,
    candidate_spec,
    model_result: fitting.ModelFitResult,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    predictor = prediction_callable(
        model_result.coefficients,
        candidate_spec.input_columns,
        model_result.prediction_model,
    )
    records: list[dict[str, object]] = []
    for chunk in chunks:
        timestamps = _prediction_timestamps(chunk)
        validation = chunk.loc[
            (timestamps >= validation_start) & (timestamps <= validation_end)
        ]
        validation, _filter_metadata = apply_filter_mode(
            validation,
            mode=prediction_spec.sample_filter_mode,
            metadata_key="sample_filter_mode",
        )
        validation = validation.dropna(
            subset=[*candidate_spec.input_columns, candidate_spec.label_column]
        )
        if validation.empty:
            continue
        output = validation.loc[
            :,
            ["timestamp", "asset_id", candidate_spec.label_column],
        ].copy()
        output[prediction_spec.prediction_column] = predictor(validation)
        records.extend(output.to_dict("records"))
    if not records:
        raise ValueError("empty_model_selection_validation_window")
    return pd.DataFrame.from_records(records)


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
    "build_prediction_result_from_research_dataset",
    "build_prediction_results_from_research_dataset",
]
