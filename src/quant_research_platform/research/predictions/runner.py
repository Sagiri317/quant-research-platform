"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
from quant_research_platform.research._memory import _selected_frame_payload_bytes
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    validate_prediction_input,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
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
    RuntimeResourcePolicy,
    _OwnedValue,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator
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
    resource_policy: RuntimeResourcePolicy | None = None,
) -> PredictionResult:
    configured_prediction = prediction_catalog.get(prediction_id)
    preflight_model_specs(
        model_catalog.get(model_id)
        for model_id in dict.fromkeys(
            (
                configured_prediction.model_id,
                *configured_prediction.candidate_model_ids,
            )
        )
    )
    operation = ExitStack()
    operation.enter_context(runtime_resource_policy_scope(resource_policy))
    try:
        result = _build_prediction_result_from_research_dataset(
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
    except BaseException as primary:
        try:
            operation.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "prediction operation cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    source_chunks = operation.enter_context(
        closing_dataset_iterator(iter(result.output_chunks))
    )
    operation.enter_context(result)
    chunks_complete = False
    metadata_value: dict[str, object] | None = None
    terminal_error: BaseException | None = None
    operation_open = True

    def close_operation(*, suppress_errors: bool = False) -> None:
        nonlocal operation_open
        if operation_open:
            operation_open = False
            try:
                operation.close()
            except BaseException:
                if not suppress_errors:
                    raise

    def close_result() -> None:
        nonlocal terminal_error
        if terminal_error is not None or not operation_open:
            return
        closed = RuntimeError("prediction_result_closed")
        try:
            close_operation()
        except BaseException as primary:
            terminal_error = primary
            raise
        terminal_error = closed

    def output_chunks() -> Iterable[pd.DataFrame]:
        nonlocal chunks_complete, terminal_error
        if terminal_error is not None:
            raise terminal_error
        try:
            yield from source_chunks
        except BaseException as primary:
            terminal_error = primary
            try:
                close_operation()
            except BaseException as cleanup_error:
                primary.add_note(
                    "prediction operation cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        chunks_complete = True

    def prediction_metadata() -> dict[str, object]:
        nonlocal metadata_value, terminal_error
        if terminal_error is not None:
            raise terminal_error
        if not chunks_complete:
            raise RuntimeError("prediction_metadata_before_output_complete")
        if metadata_value is not None:
            return metadata_value
        try:
            candidate = result.prediction_metadata_factory()
        except BaseException as primary:
            terminal_error = primary
            try:
                close_operation()
            except BaseException as cleanup_error:
                primary.add_note(
                    "prediction operation cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        metadata_value = candidate
        return metadata_value

    owned_result = PredictionResult(
        spec=result.spec,
        output_chunks=output_chunks(),
        model_metadata=result.model_metadata,
        prediction_metadata_factory=prediction_metadata,
        _close=close_result,
    )
    return owned_result


def _build_prediction_result_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    model_catalog: ModelCatalog,
    prediction_catalog: PredictionCatalog,
    prediction_id: str,
    time_window: TimeWindow | None,
    prediction_train_end: str | None,
    prediction_test_start: str | None,
    prediction_test_end: str | None,
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
        owned_model = fitting._fit_model_result_owned_from_frames(  # noqa: SLF001
            _prediction_train_chunks(
                _chunks(train_read_columns, input_columns=model_spec.input_columns),
                train_end=train_end,
            ),
            model_spec,
            estimator=build_estimator_from_model_spec(model_spec),
        )
        model_result = (
            owned_model.value
            if isinstance(owned_model, _OwnedValue)
            else owned_model
        )
        model_lease = (
            owned_model.lease if isinstance(owned_model, _OwnedValue) else None
        )
    except ValueError as exc:
        if str(exc) == "model_training_sample_empty":
            raise ValueError("empty_training_window") from exc
        raise
    model_metadata = model_result.metadata
    if model_selection_metadata is not None:
        model_metadata = {**model_metadata, "model_selection": model_selection_metadata}
    try:
        return build_prediction_result_from_frames(
            _chunks(prediction_read_columns, input_columns=model_spec.input_columns),
            prediction_spec,
            model_result.coefficients,
            model_spec.input_columns,
            prediction_model=model_result.prediction_model,
            model_metadata=model_metadata,
            model_lease=model_lease,
            resource_policy=current_runtime_resource_policy(),
        )
    except BaseException as primary:
        model_result = None
        owned_model = None
        model_metadata = None
        try:
            if model_lease is not None:
                model_lease.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "fitted model lease release also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise


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
        owned_model = fitting._fit_model_result_owned_from_frames(  # noqa: SLF001
            _prediction_train_chunks(
                chunks(train_read_columns, input_columns=candidate_spec.input_columns),
                train_end=pd.Timestamp(prediction_spec.train_end),
            ),
            candidate_spec,
            estimator=build_estimator_from_model_spec(candidate_spec),
        )
        model_result = (
            owned_model.value
            if isinstance(owned_model, _OwnedValue)
            else owned_model
        )
        model_lease = (
            owned_model.lease if isinstance(owned_model, _OwnedValue) else None
        )
        validation_frame = None
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
        try:
            validation_frame = _validation_prediction_frame(
                lambda validation_columns=validation_read_columns: chunks(
                    validation_columns,
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
            try:
                trial_metric_records.append(
                    evaluate_model_trial_metrics(
                        validation_frame.value
                        if isinstance(validation_frame, _OwnedValue)
                        else validation_frame,
                        model_id=candidate_spec.model_id,
                        label_column=label_column,
                        prediction_column=prediction_spec.prediction_column,
                    )
                )
            except BaseException as primary:
                if isinstance(validation_frame, _OwnedValue):
                    validation_lease = validation_frame.lease
                    validation_frame = None
                    try:
                        validation_lease.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            "validation lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                raise
            else:
                if isinstance(validation_frame, _OwnedValue):
                    validation_lease = validation_frame.lease
                    validation_frame = None
                    validation_lease.close()
        except BaseException as primary:
            model_result = None
            owned_model = None
            if model_lease is not None:
                try:
                    model_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "candidate model lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise
        else:
            model_result = None
            owned_model = None
            if model_lease is not None:
                model_lease.close()
            model_lease = None
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
    chunks: Callable[[], Iterable[pd.DataFrame]],
    *,
    prediction_spec,
    candidate_spec,
    model_result: fitting.ModelFitResult,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    usable_input_columns: tuple[str, ...] | None = None,
    purge_gap_steps: int | None = None,
) -> pd.DataFrame | _OwnedValue[pd.DataFrame]:
    predictor = prediction_callable(
        model_result.coefficients,
        candidate_spec.input_columns,
        model_result.prediction_model,
    )
    retained_policy = current_runtime_resource_policy()
    output_columns = (
        "timestamp",
        "asset_id",
        candidate_spec.label_column,
    )
    selected_output_bytes = 0
    max_selected_output_bytes = 0
    max_full_chunk_payload = 0
    max_chunk_rows = 0
    eligible_rows = 0
    for chunk in chunks():
        full_chunk_payload = pandas_frame_payload_bytes(chunk)
        max_full_chunk_payload = max(max_full_chunk_payload, full_chunk_payload)
        max_chunk_rows = max(max_chunk_rows, len(chunk))
        index_bytes = int(chunk.index.memory_usage(deep=True))
        chunk_bound = (
            3 * full_chunk_payload
            + index_bytes
            + len(chunk)
            * 8
            * (len(usable_input_columns or candidate_spec.input_columns) + 6)
        )
        chunk_lease = None
        if retained_policy is not None:
            chunk_lease = retained_policy.lease(
                "research.prediction.validation_preflight",
                max(1, chunk_bound),
            )
        timestamps = None
        validation = None
        filtered = None
        usable = None
        try:
            timestamps = _prediction_timestamps(chunk)
            validation = chunk.loc[
                (timestamps >= validation_start) & (timestamps <= validation_end)
            ]
            filtered, _filter_metadata = apply_filter_mode(
                validation,
                mode=prediction_spec.sample_filter_mode,
                metadata_key="sample_filter_mode",
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
                selected_columns = [
                    column for column in output_columns if column in usable.columns
                ]
                selected_bytes = _selected_frame_payload_bytes(
                    usable,
                    selected_columns,
                ) + 8 * len(usable)
                selected_output_bytes += selected_bytes
                max_selected_output_bytes = max(
                    max_selected_output_bytes,
                    selected_bytes,
                )
                eligible_rows += int(len(usable))
        except BaseException as primary:
            timestamps = None
            validation = None
            filtered = None
            usable = None
            if chunk_lease is not None:
                try:
                    chunk_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "validation preflight lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise
        else:
            timestamps = None
            validation = None
            filtered = None
            usable = None
            if chunk_lease is not None:
                chunk_lease.close()
    if eligible_rows == 0:
        raise ValueError("empty_model_selection_validation_window")
    retained_bound = _validation_result_bound(
        selected_output_bytes=selected_output_bytes,
        max_selected_output_bytes=max_selected_output_bytes,
        max_full_chunk_payload=max_full_chunk_payload,
        max_chunk_rows=max_chunk_rows,
        eligible_rows=eligible_rows,
        input_count=len(candidate_spec.input_columns),
    )
    validation_lease = None
    if retained_policy is not None:
        validation_lease = retained_policy.lease(
            "research.prediction.validation",
            retained_bound,
        )
    try:
        output_frames: list[pd.DataFrame] = []
        for chunk in chunks():
            timestamps = _prediction_timestamps(chunk)
            validation = chunk.loc[
                (timestamps >= validation_start) & (timestamps <= validation_end)
            ]
            del timestamps
            if validation.empty:
                continue
            filtered, _filter_metadata = apply_filter_mode(
                validation,
                mode=prediction_spec.sample_filter_mode,
                metadata_key="sample_filter_mode",
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
                output_frames.append(output)
        if not output_frames:
            raise ValueError("empty_model_selection_validation_window")
        result = pd.concat(output_frames, ignore_index=True)
        output_frames = []
        result = validation_split._purge_trailing_timestamps(
            result,
            (
                candidate_spec.purge_gap_steps
                if purge_gap_steps is None
                else purge_gap_steps
            ),
            group_column="asset_id",
        )
        if result.empty:
            raise ValueError("empty_model_selection_validation_window")
        timestamps = None
        validation = None
        filtered = None
        usable = None
        output = None
        if validation_lease is not None:
            validation_lease.shrink_to(pandas_frame_payload_bytes(result))
            validation_lease.transfer("research.model.selection.validation")
            return _OwnedValue(result, validation_lease)
        return result
    except BaseException as primary:
        if "timestamps" in locals():
            del timestamps
        validation = None
        filtered = None
        usable = None
        output = None
        output_frames = []
        result = None
        if validation_lease is not None:
            try:
                validation_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "validation lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise


def _validation_result_bound(
    *,
    selected_output_bytes: int,
    max_selected_output_bytes: int,
    max_full_chunk_payload: int,
    max_chunk_rows: int,
    eligible_rows: int,
    input_count: int,
) -> int:
    working_columns = input_count + 6
    window_filter_usable = 3 * max_full_chunk_payload
    mask_and_matrix = (
        max_chunk_rows * 8 * (working_columns + 3)
        + eligible_rows * 8 * 4
    )
    output_frames = selected_output_bytes
    concat_copy = output_frames
    purge_copy = output_frames
    peak_output = max_selected_output_bytes
    return max(
        1,
        window_filter_usable
        + mask_and_matrix
        + output_frames
        + concat_copy
        + purge_copy
        + peak_output,
    )


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
            policy.used_memory_bytes + working_bytes
            > policy.memory_budget_bytes
        ):
            raise ValueError(
                "prediction_training_working_set_exceeds_task_memory_budget"
            )
        working_lease = None
        if policy is not None:
            working_lease = policy.lease(
                "research.prediction.training",
                working_bytes,
            )
        timestamps = None
        train = None
        try:
            timestamps = _prediction_timestamps(chunk)
            train = chunk.loc[timestamps <= train_end]
            if train.empty:
                train = None
                timestamps = None
                if working_lease is not None:
                    working_lease.close()
                continue
            retained_bytes = pandas_frame_payload_bytes(train)
            if working_lease is not None:
                working_lease.shrink_to(retained_bytes)
                working_lease.transfer("research.prediction.train")
            timestamps = None
            try:
                yield train
            except BaseException as primary:
                train = None
                if working_lease is not None:
                    try:
                        working_lease.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            "prediction training lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                raise
            else:
                train = None
                if working_lease is not None:
                    working_lease.close()
        except BaseException as primary:
            train = None
            timestamps = None
            if working_lease is not None and working_lease.active:
                try:
                    working_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "prediction training lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise


def _prediction_timestamps(chunk: pd.DataFrame) -> pd.Series:
    if "timestamp" not in chunk.columns:
        raise ValueError("missing_prediction_columns: timestamp")
    return strict_timestamp_series(
        chunk["timestamp"], error="invalid_prediction_timestamp_values"
    )


__all__ = [
    "build_prediction_result_from_research_dataset",
]
