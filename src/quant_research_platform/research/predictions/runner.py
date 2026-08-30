"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import replace

import pandas as pd

import quant_research_platform.research.models.fitting as fitting
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    validate_prediction_input,
)
from quant_research_platform.research.filtering import (
    _filter_working_set_bound,
    apply_filter_mode,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.models.trial_selection import (
    TEST_METRIC_COLUMNS,
    VALIDATION_METRIC_COLUMNS,
    build_model_selection_metadata,
    evaluate_model_trial_metrics_from_chunks,
    select_best_validation_trial,
)
from quant_research_platform.research.predictions.outputs import (
    PredictionResult,
    build_prediction_result_from_frames,
    prediction_callable,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
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
        closing_iterator(iter(result.output_chunks))
    )
    operation.enter_context(result)
    chunks_complete = False
    metadata_value: dict[str, object] | None = None
    terminal_error: BaseException | None = None
    operation_open = True

    def close_operation() -> None:
        nonlocal operation_open
        if operation_open:
            operation_open = False
            operation.close()

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
                asset_ids=prediction_spec.training_asset_ids,
            ),
            model_spec,
            estimator=build_estimator_from_model_spec(model_spec),
        )
        model_result = (
            owned_model.value if isinstance(owned_model, OwnedValue) else owned_model
        )
        model_lease = owned_model.lease if isinstance(owned_model, OwnedValue) else None
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
    with closing_iterator(iter(chunks)) as active_chunks:
        for chunk in active_chunks:
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
                asset_ids=prediction_spec.training_asset_ids,
            ),
            candidate_spec,
            estimator=build_estimator_from_model_spec(candidate_spec),
        )
        model_result = (
            owned_model.value if isinstance(owned_model, OwnedValue) else owned_model
        )
        model_lease = owned_model.lease if isinstance(owned_model, OwnedValue) else None
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
            validation_chunks = _iter_validation_prediction_chunks(
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
            )
            trial_metric_records.append(
                evaluate_model_trial_metrics_from_chunks(
                    validation_chunks,
                    model_id=candidate_spec.model_id,
                    label_column=label_column,
                    prediction_column=prediction_spec.prediction_column,
                    purge_gap_steps=common_purge_gap_steps,
                )
            )
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


def _iter_validation_prediction_chunks(
    chunks: Callable[[], Iterable[pd.DataFrame]],
    *,
    prediction_spec,
    candidate_spec,
    model_result: fitting.ModelFitResult,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    usable_input_columns: tuple[str, ...] | None = None,
) -> Iterable[object]:
    predictor = prediction_callable(
        model_result.coefficients,
        candidate_spec.input_columns,
        model_result.prediction_model,
    )
    output_columns = (
        "timestamp",
        "asset_id",
        candidate_spec.label_column,
    )
    policy = current_runtime_resource_policy()
    with closing_iterator(iter(chunks())) as source:
        for chunk in source:
            full_chunk_payload = pandas_frame_payload_bytes(chunk)
            index_bytes = int(chunk.index.memory_usage(deep=True))
            chunk_bound = (
                3 * full_chunk_payload
                + index_bytes
                + len(chunk)
                * 8
                * (len(usable_input_columns or candidate_spec.input_columns) + 6)
            )
            chunk_bound = max(
                chunk_bound,
                _filter_working_set_bound(
                    chunk,
                    mode=prediction_spec.sample_filter_mode,
                ),
            )
            chunk_lease = (
                policy.lease(
                    "research.prediction.validation.chunk",
                    max(1, chunk_bound),
                )
                if policy is not None
                else None
            )
            try:
                timestamps = _prediction_timestamps(chunk)
                validation = chunk.loc[
                    (timestamps >= validation_start) & (timestamps <= validation_end)
                ]
                if not validation.empty:
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
                        if chunk_lease is not None:
                            output_bytes = pandas_frame_payload_bytes(output)
                            if output_bytes > chunk_lease.reserved_bytes:
                                raise ValueError(
                                    "prediction_validation_working_set_exceeds_task_memory_budget"
                                )
                            chunk_lease.shrink_to(output_bytes)
                            chunk_lease.transfer(
                                "research.model.selection.metric.chunk"
                            )
                            item = OwnedValue(output, chunk_lease)
                        else:
                            item = output
                        del chunk, timestamps, validation, filtered, usable, output
                        yield item
            except BaseException as primary:
                if chunk_lease is not None and chunk_lease.active:
                    try:
                        chunk_lease.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            "validation chunk lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                raise
            else:
                if chunk_lease is not None and chunk_lease.active:
                    chunk_lease.close()


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
    asset_ids: tuple[str, ...] = (),
) -> Iterable[pd.DataFrame]:
    for chunk in chunks:
        if asset_ids:
            if "asset_id" not in chunk.columns:
                raise ValueError("prediction_training_asset_id_missing")
            chunk = chunk.loc[chunk["asset_id"].isin(asset_ids)]
            if chunk.empty:
                continue
        policy = current_runtime_resource_policy()
        index_bytes = int(chunk.index.memory_usage(deep=True))
        timestamp_bytes = index_bytes + 8 * len(chunk)
        mask_bytes = index_bytes + len(chunk)
        working_bytes = timestamp_bytes + mask_bytes + pandas_frame_payload_bytes(chunk)
        if (
            policy is not None
            and policy.explicit_memory_cap_bytes is not None
            and policy.used_memory_bytes + working_bytes
            > policy.explicit_memory_cap_bytes
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
