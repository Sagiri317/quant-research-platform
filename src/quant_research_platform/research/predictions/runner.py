"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import replace

import pandas as pd

import quant_research_platform.research.models.fitting as fitting
import quant_research_platform.research.models.validation_split as validation_split
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.research._memory import (
    _ResearchFrameAccumulator,
    _selected_frame_payload_bytes,
)
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    validate_prediction_input,
)
from quant_research_platform.research.filtering import apply_filter_mode
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
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
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
        yield from _validated_prediction_chunks(
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
    train_read_columns = _prediction_train_read_columns(
        dataset_spec.primary_key,
        model_spec.input_columns,
        label_column=model_spec.label_column,
        sample_filter_mode=model_spec.sample_filter_mode,
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
        retained_model_bytes=model_result.retained_memory_bytes,
        resource_policy=current_runtime_resource_policy(),
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
    common_validation_input_columns = tuple(
        dict.fromkeys(
            column for spec in candidate_specs for column in spec.input_columns
        )
    )
    common_purge_gap_steps = max(spec.purge_gap_steps for spec in candidate_specs)
    validation_start = pd.Timestamp(prediction_spec.model_selection_validation_start)
    validation_end = pd.Timestamp(prediction_spec.model_selection_validation_end)
    trial_metric_records: list[dict[str, object]] = []
    for candidate_spec in candidate_specs:
        train_read_columns = _prediction_train_read_columns(
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
                    *common_validation_input_columns,
                    label_column,
                    *required_filter_columns(prediction_spec.sample_filter_mode),
                )
            )
        )
        policy = current_runtime_resource_policy()
        with ExitStack() as model_state:
            if policy is not None:
                model_state.enter_context(
                    policy.reserve(
                        "research.model.selection_state",
                        model_result.retained_memory_bytes,
                    )
                )
            validation_frame = _validation_prediction_frame(
                chunks(
                    validation_read_columns,
                    input_columns=common_validation_input_columns,
                ),
                prediction_spec=prediction_spec,
                candidate_spec=candidate_spec,
                model_result=model_result,
                validation_start=validation_start,
                validation_end=validation_end,
                usable_input_columns=common_validation_input_columns,
                purge_gap_steps=common_purge_gap_steps,
            )
            trial_metric_records.append(
                evaluate_model_trial_metrics(
                    validation_frame,
                    model_id=candidate_spec.model_id,
                    label_column=label_column,
                    prediction_column=prediction_spec.prediction_column,
                )
            )
        del model_result, validation_frame
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


def _prediction_train_read_columns(
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
    usable_input_columns: tuple[str, ...] | None = None,
    purge_gap_steps: int | None = None,
) -> pd.DataFrame:
    predictor = prediction_callable(
        model_result.coefficients,
        candidate_spec.input_columns,
        model_result.prediction_model,
    )
    retained_policy = current_runtime_resource_policy()
    with _ResearchFrameAccumulator(
        "research.prediction.validation",
        resource_policy=retained_policy,
    ) as accumulator:
        for chunk in chunks:
            resource_policy = current_runtime_resource_policy()
            row_count = len(chunk)
            index_bytes = int(chunk.index.memory_usage(deep=True))
            timestamp_bytes = index_bytes + 8 * row_count
            window_mask_bytes = index_bytes + row_count
            window_output_bytes = pandas_frame_payload_bytes(chunk)
            with ExitStack() as window_reservation:
                if resource_policy is not None:
                    window_reservation.enter_context(
                        resource_policy.reserve(
                            "research.prediction.validation.window",
                            timestamp_bytes + window_mask_bytes + window_output_bytes,
                        )
                    )
                timestamps = _prediction_timestamps(chunk)
                validation = chunk.loc[
                    (timestamps >= validation_start) & (timestamps <= validation_end)
                ]
                del timestamps
            if validation.empty:
                del validation
                continue
            validation_bytes = pandas_frame_payload_bytes(validation)
            with ExitStack() as retained:
                if resource_policy is not None:
                    retained.enter_context(
                        resource_policy.reserve(
                            "research.prediction.validation.input",
                            validation_bytes,
                        )
                    )
                filtered, _filter_metadata = apply_filter_mode(
                    validation,
                    mode=prediction_spec.sample_filter_mode,
                    metadata_key="sample_filter_mode",
                )
                if filtered is not validation and resource_policy is not None:
                    retained.enter_context(
                        resource_policy.reserve(
                            "research.prediction.validation.filtered",
                            pandas_frame_payload_bytes(filtered),
                        )
                    )
                filtered_bytes = pandas_frame_payload_bytes(filtered)
                usable_mask_bytes = int(filtered.index.memory_usage(deep=True)) + len(
                    filtered
                )
                output_columns = (
                    "timestamp",
                    "asset_id",
                    candidate_spec.label_column,
                )
                output_frame_bytes = _selected_frame_payload_bytes(
                    filtered,
                    output_columns,
                ) + 8 * len(filtered)
                model_matrix_bytes = (
                    8 * len(filtered) * (len(candidate_spec.input_columns) + 1)
                )
                with ExitStack() as compute_reservation:
                    if resource_policy is not None:
                        compute_reservation.enter_context(
                            resource_policy.reserve(
                                "research.prediction.validation.compute",
                                usable_mask_bytes
                                + filtered_bytes
                                + output_frame_bytes
                                + model_matrix_bytes,
                            )
                        )
                    usable = filtered.dropna(
                        subset=[
                            *(
                                usable_input_columns
                                if usable_input_columns is not None
                                else candidate_spec.input_columns
                            ),
                            candidate_spec.label_column,
                        ]
                    )
                    if not usable.empty:
                        output = usable.loc[:, list(output_columns)].copy()
                        output[prediction_spec.prediction_column] = predictor(usable)
                        accumulator.append(output)
                        del output
                    del usable
            del filtered, validation
        if accumulator.row_count == 0:
            raise ValueError("empty_model_selection_validation_window")
        result = accumulator.materialize()
    input_bytes = pandas_frame_payload_bytes(result)
    purged_output_bound = input_bytes
    with ExitStack() as purge_reservation:
        if retained_policy is not None:
            purge_reservation.enter_context(
                retained_policy.reserve(
                    "research.prediction.validation.purge",
                    input_bytes + purged_output_bound,
                )
            )
        result = validation_split._purge_trailing_timestamps(
            result,
            (
                candidate_spec.purge_gap_steps
                if purge_gap_steps is None
                else purge_gap_steps
            ),
            group_column="asset_id",
        )
        if pandas_frame_payload_bytes(result) > purged_output_bound:
            raise ValueError(
                "research_prediction_working_set_exceeds_task_memory_budget"
            )
    if result.empty:
        raise ValueError("empty_model_selection_validation_window")
    return result


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
        policy = current_runtime_resource_policy()
        index_bytes = int(chunk.index.memory_usage(deep=True))
        timestamp_bytes = index_bytes + 8 * len(chunk)
        mask_bytes = index_bytes + len(chunk)
        working_bytes = timestamp_bytes + mask_bytes + pandas_frame_payload_bytes(chunk)
        if policy is not None and (
            policy.reserved_memory_bytes + working_bytes
            > policy.task_memory_budget_bytes
        ):
            raise ValueError(
                "prediction_training_working_set_exceeds_task_memory_budget"
            )
        with ExitStack() as working:
            if policy is not None:
                working.enter_context(
                    policy.reserve("research.prediction.training", working_bytes)
                )
            timestamps = _prediction_timestamps(chunk)
            train = chunk.loc[timestamps <= train_end]
        if not train.empty:
            retained_bytes = pandas_frame_payload_bytes(train)
            if policy is not None and (
                policy.reserved_memory_bytes + retained_bytes
                > policy.task_memory_budget_bytes
            ):
                raise ValueError(
                    "prediction_training_working_set_exceeds_task_memory_budget"
                )
            with ExitStack() as retained:
                if policy is not None:
                    retained.enter_context(
                        policy.reserve("research.prediction.train", retained_bytes)
                    )
                yield train


def _prediction_timestamps(chunk: pd.DataFrame) -> pd.Series:
    if "timestamp" not in chunk.columns:
        raise ValueError("missing_prediction_columns: timestamp")
    return strict_timestamp_series(
        chunk["timestamp"], error="invalid_prediction_timestamp_values"
    )


__all__ = [
    "build_prediction_result_from_research_dataset",
]
