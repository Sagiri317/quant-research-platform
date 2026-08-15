"""Supervised model fitting owners."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import quant_research_platform.research.models.dependencies as dependencies
import quant_research_platform.research.models.feature_importance as feature_importance
import quant_research_platform.research.models.supervised as supervised
import quant_research_platform.research.models.validation_split as validation_split
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research._memory import (
    _selected_frame_payload_bytes,
    iter_complete_time_sized_frames,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator


@dataclass(frozen=True)
class ModelFitResult:
    coefficients: np.ndarray
    metadata: dict[str, object]
    prediction_model: object | None = field(default=None, compare=False, repr=False)
    retained_memory_bytes: int = field(default=0, compare=False, repr=False)


def fit_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    estimator: object,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> ModelFitResult:
    policy = (
        bind_runtime_resource_policy(resource_policy) or RuntimeResourcePolicy.capture()
    )
    with runtime_resource_policy_scope(policy):
        fit_model_result = getattr(estimator, "fit_model_result_from_frames", None)
        if callable(fit_model_result):
            return fit_model_result(frames, spec)
        return fit_estimator_model_result_from_frames(
            frames,
            spec,
            estimator=estimator,
        )


def fit_linear_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> ModelFitResult:
    if spec.max_train_rows is not None or spec.purge_gap_steps > 0:
        return fit_controlled_linear_model_result_from_frames(frames, spec)
    input_columns = spec.input_columns
    required = set(input_columns)
    required.add(spec.label_column)
    required.add("timestamp")
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                *input_columns,
                spec.label_column,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    fit_columns = tuple(
        column
        for column in ("timestamp", *input_columns, spec.label_column)
        if column in work_columns
    )
    work_frames = iter_model_work_frames(
        frames,
        required=required,
        work_columns=work_columns,
    )
    with closing_dataset_iterator(iter(work_frames)) as active_frames:
        beta, row_counts, filter_metadata = fit_linear_coefficients(
            active_frames,
            spec=spec,
            fit_columns=fit_columns,
        )
    if row_counts["rows_after_input_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    metadata = model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        coefficient_count=int(len(beta)),
        feature_importance=feature_importance.linear_feature_importance(spec, beta),
    )
    return ModelFitResult(
        coefficients=beta,
        metadata=metadata,
        retained_memory_bytes=int(beta.nbytes),
    )


def model_metadata(
    spec: ModelSpec,
    *,
    filter_metadata: dict[str, object],
    row_counts: dict[str, object],
    coefficient_count: int,
    feature_importance: dict[str, float] | None = None,
    feature_importance_status: str | None = None,
    resolved_hyperparameters: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "model_algorithm": spec.model_algorithm,
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": list(filter_metadata["filter_columns_used"]),
        "filter_definition": str(filter_metadata["filter_definition"]),
        "rows_input": _strict_non_negative_count_value(
            filter_metadata["rows_input"],
        ),
        "rows_after_filter_mode": _strict_non_negative_count_value(
            filter_metadata["rows_after_filter_mode"],
        ),
        "rows_after_input_label_dropna": _strict_non_negative_count_value(
            row_counts["rows_after_input_label_dropna"],
        ),
        "train_start": row_counts["train_start"],
        "train_end": row_counts["train_end"],
        "test_start": None,
        "test_end": None,
        "oos_start": None,
        "oos_end": None,
        "input_factor_columns": list(spec.input_factor_columns),
        "input_feature_columns": list(spec.input_feature_columns),
        "input_columns": list(spec.input_columns),
        "input_count": len(spec.input_columns),
        "label_column": spec.label_column,
        "hyperparameters": dict(spec.hyperparameters),
        "resolved_hyperparameters": (
            supervised.resolved_model_hyperparameters(spec)
            if resolved_hyperparameters is None
            else resolved_hyperparameters
        ),
        "random_seed": spec.random_seed,
        "max_train_rows": spec.max_train_rows,
        "purge_gap_steps": spec.purge_gap_steps,
        "dependency_status": dependencies.require_model_dependency(
            spec.model_algorithm,
        ),
        "coefficient_count": coefficient_count,
    }
    if feature_importance is not None:
        metadata["feature_importance"] = feature_importance
    else:
        metadata["feature_importance_status"] = (
            feature_importance_status or "not_available_for_model_algorithm"
        )
    return metadata


def fit_controlled_linear_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> ModelFitResult:
    training, row_counts, filter_metadata = collect_training_frame(frames, spec)
    if row_counts["rows_after_input_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    beta = fit_linear_coefficients_exact(training, spec)
    metadata = model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        coefficient_count=int(len(beta)),
        feature_importance=feature_importance.linear_feature_importance(spec, beta),
    )
    return ModelFitResult(
        coefficients=beta,
        metadata=metadata,
        retained_memory_bytes=int(beta.nbytes),
    )


def fit_linear_coefficients(
    frames: Iterable[pd.DataFrame],
    *,
    spec: ModelSpec,
    fit_columns: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    input_count = len(spec.input_columns)
    chunk_count = 0
    row_count = 0
    rows_input = 0
    rows_after_filter = 0
    first_usable: pd.DataFrame | None = None
    filter_columns_used: list[str] = []
    filter_definition = ""
    r_factor: np.ndarray | None = None
    qt_y: np.ndarray | None = None
    train_start: str | None = None
    train_end: str | None = None
    factor_reservation = ExitStack()
    first_usable_reservation = ExitStack()
    try:
        for chunk in frames:
            chunk_count += 1
            policy = current_runtime_resource_policy()
            if chunk_count == 2:
                first_usable = None
                first_usable_reservation.close()
                first_usable_reservation = ExitStack()
            with ExitStack() as chunk_work:
                filtered, metadata = apply_filter_mode(
                    chunk,
                    mode=spec.sample_filter_mode,
                    metadata_key="sample_filter_mode",
                )
                if policy is not None and filtered is not chunk:
                    chunk_work.enter_context(
                        policy.reserve(
                            "research.model.linear_filtered",
                            pandas_frame_payload_bytes(filtered),
                        )
                    )
                rows_input += _strict_non_negative_count_value(metadata["rows_input"])
                rows_after_filter += _strict_non_negative_count_value(
                    metadata["rows_after_filter_mode"]
                )
                filter_columns_used = list(metadata["filter_columns_used"])
                filter_definition = str(metadata["filter_definition"])
                usable_bound = _selected_frame_payload_bytes(
                    filtered,
                    list(fit_columns),
                )
                with ExitStack() as usable_reservation:
                    if policy is not None:
                        usable_reservation.enter_context(
                            policy.reserve(
                                "research.model.linear_projection",
                                usable_bound,
                            )
                        )
                    usable = filtered.dropna(
                        subset=[*spec.input_columns, spec.label_column]
                    ).loc[:, list(fit_columns)]
                    if pandas_frame_payload_bytes(usable) > usable_bound:
                        raise ValueError(
                            "model_training_working_set_exceeds_task_memory_budget"
                        )
                    if chunk_count == 1:
                        first_usable = usable
                        first_usable_reservation = usable_reservation.pop_all()
                    row_count += int(len(usable))
                    if len(usable) == 0:
                        del usable, filtered, chunk
                        continue
                    with ExitStack() as qr_work:
                        if policy is not None:
                            qr_work.enter_context(
                                policy.reserve(
                                    "research.model.qr_work",
                                    _linear_qr_working_set_bytes(
                                        row_count=len(usable),
                                        input_count=input_count,
                                        has_retained_factor=r_factor is not None,
                                    ),
                                )
                            )
                        y = usable.loc[:, spec.label_column].to_numpy(
                            dtype="float64", copy=False
                        )
                        usable_timestamps = pd.to_datetime(
                            usable["timestamp"], errors="coerce"
                        )
                        chunk_start = date_string(usable_timestamps.min())
                        chunk_end = date_string(usable_timestamps.max())
                        train_start = (
                            chunk_start
                            if train_start is None
                            else min(train_start, chunk_start)
                        )
                        train_end = (
                            chunk_end
                            if train_end is None
                            else max(train_end, chunk_end)
                        )
                        chunk_design = linear_design_matrix(usable, spec)
                        if r_factor is None or qt_y is None:
                            stacked_design = chunk_design
                            stacked_y = y
                        else:
                            stacked_design = np.vstack([r_factor, chunk_design])
                            stacked_y = np.concatenate([qt_y, y])
                        with threadpool_limits(limits=1, user_api="blas"):
                            q_factor, next_r_factor = np.linalg.qr(
                                stacked_design, mode="reduced"
                            )
                            next_qt_y = q_factor.T @ stacked_y
                        del (
                            chunk_design,
                            q_factor,
                            stacked_design,
                            stacked_y,
                            usable_timestamps,
                            y,
                        )
                del usable, filtered, chunk
            factor_reservation.close()
            factor_reservation = ExitStack()
            if policy is not None:
                factor_reservation.enter_context(
                    policy.reserve(
                        "research.model.qr_state",
                        int(next_r_factor.nbytes + next_qt_y.nbytes),
                    )
                )
            r_factor = next_r_factor
            qt_y = next_qt_y

        row_counts = {
            "rows_input": rows_input,
            "rows_after_filter_mode": rows_after_filter,
            "rows_after_input_label_dropna": row_count,
            "train_start": train_start,
            "train_end": train_end,
        }
        filter_metadata = {
            "sample_filter_mode": spec.sample_filter_mode,
            "filter_columns_used": filter_columns_used,
            "filter_definition": filter_definition,
            "rows_input": rows_input,
            "rows_after_filter_mode": rows_after_filter,
        }
        if row_count == 0:
            return (
                np.empty(input_count + 1, dtype="float64"),
                row_counts,
                filter_metadata,
            )
        if chunk_count == 1 and first_usable is not None:
            factor_reservation.close()
            first_usable_reservation.close()
            r_factor = None
            qt_y = None
            return (
                fit_linear_coefficients_exact(first_usable, spec),
                row_counts,
                filter_metadata,
            )
        if r_factor is None or qt_y is None:
            return (
                np.empty(input_count + 1, dtype="float64"),
                row_counts,
                filter_metadata,
            )
        return (
            fit_linear_coefficients_from_tsqr(r_factor, qt_y),
            row_counts,
            filter_metadata,
        )
    finally:
        factor_reservation.close()
        first_usable_reservation.close()


def fit_estimator_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    estimator: object,
) -> ModelFitResult:
    training, row_counts, filter_metadata = collect_training_frame(frames, spec)
    if row_counts["rows_after_input_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    policy = current_runtime_resource_policy()
    training_bytes = pandas_frame_payload_bytes(training)
    target_bytes = len(training) * np.dtype(np.float64).itemsize
    model_state_bytes = _model_state_upper_bound_bytes(
        training,
        spec,
        estimator=estimator,
    )
    with ExitStack() as retained_state:
        if policy is not None:
            retained_state.enter_context(
                policy.reserve("research.model.training_result", training_bytes)
            )
            retained_state.enter_context(
                policy.reserve("research.model.fit_state", model_state_bytes)
            )
        with ExitStack() as target_state:
            if policy is not None:
                target_state.enter_context(
                    policy.reserve("research.model.target", target_bytes)
                )
            y = training.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
            with ExitStack() as fit_work:
                threads = _effective_model_thread_count(
                    training,
                    spec,
                    available_bytes=(
                        None if policy is None else policy.remaining_memory_bytes
                    ),
                )
                if policy is not None:
                    fit_work.enter_context(
                        policy.reserve(
                            "research.model.fit_work",
                            _estimator_fit_working_set_bytes(
                                training,
                                spec,
                                threads=threads,
                            ),
                        )
                    )
                predictor = _fit_predictor(
                    spec,
                    training=training,
                    y=y,
                    estimator=estimator,
                    threads=threads,
                )
            del y
        coefficient_count = int(
            getattr(predictor, "coefficient_count", len(spec.input_columns))
        )
        resolved_hyperparameters = supervised.resolved_model_hyperparameters(spec)
        if spec.model_algorithm in supervised._PARALLEL_MODEL_ALGORITHMS:  # noqa: SLF001
            fitted_threads = getattr(predictor, "n_jobs", None)
            if type(fitted_threads) is not int or fitted_threads <= 0:
                raise RuntimeError("invalid_fitted_model_thread_count")
            resolved_hyperparameters["n_jobs"] = fitted_threads
        metadata = model_metadata(
            spec,
            filter_metadata=filter_metadata,
            row_counts=row_counts,
            coefficient_count=coefficient_count,
            feature_importance=feature_importance.estimator_feature_importance(
                spec,
                predictor,
            ),
            resolved_hyperparameters=resolved_hyperparameters,
        )
        coefficients = np.asarray(
            getattr(predictor, "coefficients", np.full(coefficient_count, np.nan)),
            dtype="float64",
        )
        result = ModelFitResult(
            coefficients=coefficients,
            metadata=metadata,
            prediction_model=predictor,
            retained_memory_bytes=model_state_bytes,
        )
    return result


def collect_training_frame(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    if spec.max_train_rows is None:
        raise ValueError("unbounded_model_training_collection")
    required = set(spec.input_columns)
    required.add(spec.label_column)
    required.add("timestamp")
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *spec.input_columns,
                spec.label_column,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    rows_input = 0
    rows_after_filter = 0
    filter_columns_used: list[str] = []
    filter_definition = ""
    capped_training: pd.DataFrame | None = None
    resource_policy = None
    retained_policy = current_runtime_resource_policy()
    retained_training = ExitStack()
    try:
        for chunk in iter_model_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ):
            resource_policy = current_runtime_resource_policy()
            filtered, metadata = apply_filter_mode(
                chunk,
                mode=spec.sample_filter_mode,
                metadata_key="sample_filter_mode",
            )
            rows_input += _strict_non_negative_count_value(metadata["rows_input"])
            rows_after_filter += _strict_non_negative_count_value(
                metadata["rows_after_filter_mode"]
            )
            filter_columns_used = list(metadata["filter_columns_used"])
            filter_definition = str(metadata["filter_definition"])
            identity_columns = ("asset_id",) if "asset_id" in filtered.columns else ()
            selected_columns = [
                "timestamp",
                *identity_columns,
                *spec.input_columns,
                spec.label_column,
            ]
            with ExitStack() as chunk_work:
                if resource_policy is not None and filtered is not chunk:
                    chunk_work.enter_context(
                        resource_policy.reserve(
                            "research.model.filtered",
                            pandas_frame_payload_bytes(filtered),
                        )
                    )
                usable_bound = _selected_frame_payload_bytes(filtered, selected_columns)
                if resource_policy is not None:
                    chunk_work.enter_context(
                        resource_policy.reserve(
                            "research.model.training_projection",
                            usable_bound,
                        )
                    )
                usable = filtered.dropna(
                    subset=[*spec.input_columns, spec.label_column]
                ).loc[:, selected_columns]
                usable_payload_bytes = pandas_frame_payload_bytes(usable)
                if usable_payload_bytes > usable_bound:
                    raise ValueError(
                        "model_training_working_set_exceeds_task_memory_budget"
                    )
                if usable.empty:
                    continue
                with ExitStack() as trim_work:
                    if resource_policy is not None:
                        trim_work.enter_context(
                            resource_policy.reserve(
                                "research.model.training_trim",
                                _training_control_working_set_bytes(
                                    usable_payload_bytes,
                                    len(usable),
                                ),
                            )
                        )
                    usable = validation_split.trim_training_buffer(usable, spec)
                usable_payload_bytes = pandas_frame_payload_bytes(usable)
                if capped_training is None:
                    capped_training = usable
                    if (
                        resource_policy is not None
                        and retained_policy is resource_policy
                    ):
                        retained_training.enter_context(
                            resource_policy.reserve(
                                "research.model.training",
                                usable_payload_bytes,
                            )
                        )
                    continue
                combined_payload_bytes = (
                    pandas_frame_payload_bytes(capped_training) + usable_payload_bytes
                )
                with ExitStack() as concat_work:
                    if resource_policy is not None:
                        concat_work.enter_context(
                            resource_policy.reserve(
                                "research.model.training_concat",
                                combined_payload_bytes
                                + _training_control_working_set_bytes(
                                    combined_payload_bytes,
                                    len(capped_training) + len(usable),
                                ),
                            )
                        )
                    combined = validation_split.trim_training_buffer(
                        pd.concat([capped_training, usable], ignore_index=True),
                        spec,
                    )
                retained_training.close()
                retained_training = ExitStack()
                capped_training = combined
                if resource_policy is not None and retained_policy is resource_policy:
                    retained_training.enter_context(
                        resource_policy.reserve(
                            "research.model.training",
                            pandas_frame_payload_bytes(capped_training),
                        )
                    )
        resource_policy = current_runtime_resource_policy()
        training = capped_training if capped_training is not None else pd.DataFrame()
        with ExitStack() as final_work:
            if resource_policy is not None:
                final_work.enter_context(
                    resource_policy.reserve(
                        "research.model.training_controls",
                        _training_control_working_set_bytes(
                            pandas_frame_payload_bytes(training),
                            len(training),
                        ),
                    )
                )
            training = validation_split.apply_training_controls(training, spec)
    finally:
        retained_training.close()
    resource_policy = current_runtime_resource_policy()
    with ExitStack() as result_work:
        training_bytes = pandas_frame_payload_bytes(training)
        if resource_policy is not None:
            result_work.enter_context(
                resource_policy.reserve(
                    "research.model.training_result",
                    training_bytes,
                )
            )
        if "asset_id" in training.columns:
            output_bound = _selected_frame_payload_bytes(
                training,
                [column for column in training.columns if column != "asset_id"],
            )
            if resource_policy is not None:
                result_work.enter_context(
                    resource_policy.reserve(
                        "research.model.training_identity_drop",
                        output_bound,
                    )
                )
            training = training.drop(columns="asset_id")
        train_start = None
        train_end = None
        if not training.empty:
            if resource_policy is not None:
                result_work.enter_context(
                    resource_policy.reserve(
                        "research.model.training_timestamp",
                        len(training) * np.dtype("datetime64[ns]").itemsize,
                    )
                )
            timestamps = pd.to_datetime(training["timestamp"], errors="coerce")
            train_start = date_string(timestamps.min())
            train_end = date_string(timestamps.max())
    row_counts = {
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
        "rows_after_input_label_dropna": int(len(training)),
        "train_start": train_start,
        "train_end": train_end,
    }
    filter_metadata = {
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    return training, row_counts, filter_metadata


def _training_control_working_set_bytes(frame_bytes: int, row_count: int) -> int:
    copied_frame = frame_bytes
    sorted_frame = frame_bytes
    timestamp_values = row_count * np.dtype("datetime64[ns]").itemsize
    sort_index = row_count * np.dtype(np.intp).itemsize
    return copied_frame + sorted_frame + timestamp_values + sort_index


def fit_predictor(
    spec: ModelSpec,
    *,
    training: pd.DataFrame,
    y: np.ndarray,
    estimator: object,
) -> object:
    policy = current_runtime_resource_policy()
    with ExitStack() as fit_reservation:
        threads = supervised.model_thread_count(spec)
        if policy is not None:
            input_bytes = pandas_frame_payload_bytes(training) + int(y.nbytes)
            state_bytes = _model_state_upper_bound_bytes(
                training,
                spec,
                estimator=estimator,
            )
            if input_bytes + state_bytes > policy.remaining_memory_bytes:
                raise MemoryError("runtime_memory_budget_exceeded")
            fit_reservation.enter_context(
                policy.reserve(
                    "research.model.fit_input",
                    input_bytes,
                )
            )
            fit_reservation.enter_context(
                policy.reserve(
                    "research.model.fit_state",
                    state_bytes,
                )
            )
            threads = _effective_model_thread_count(
                training,
                spec,
                available_bytes=policy.remaining_memory_bytes,
            )
            fit_reservation.enter_context(
                policy.reserve(
                    "research.model.fit_work",
                    _estimator_fit_working_set_bytes(
                        training,
                        spec,
                        threads=threads,
                    ),
                )
            )
        return _fit_predictor(
            spec,
            training=training,
            y=y,
            estimator=estimator,
            threads=threads,
        )


def _fit_predictor(
    spec: ModelSpec,
    *,
    training: pd.DataFrame,
    y: np.ndarray,
    estimator: object,
    threads: int,
) -> object:
    fit_training_frame = getattr(estimator, "fit_training_frame", None)
    if callable(fit_training_frame):
        return fit_training_frame(training)
    if not hasattr(estimator, "fit"):
        return estimator
    if spec.model_algorithm in supervised._PARALLEL_MODEL_ALGORITHMS:  # noqa: SLF001
        set_params = getattr(estimator, "set_params", None)
        if not callable(set_params):
            raise RuntimeError("model_thread_configuration_unavailable")
        set_params(n_jobs=threads)
    openmp_threads = (
        threads
        if spec.model_algorithm in {"xgboost_regressor", "lightgbm_regressor"}
        else 1
    )
    with (
        threadpool_limits(limits=1, user_api="blas"),
        threadpool_limits(
            limits=openmp_threads,
            user_api="openmp",
        ),
    ):
        estimator.fit(training.loc[:, spec.input_columns], y)
    return estimator


def fit_linear_coefficients_exact(
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    policy = current_runtime_resource_policy()
    with ExitStack() as solve_reservation:
        if policy is not None:
            solve_reservation.enter_context(
                policy.reserve(
                    "research.model.linear_input",
                    pandas_frame_payload_bytes(frame),
                )
            )
            solve_reservation.enter_context(
                policy.reserve(
                    "research.model.linear_solve",
                    _linear_exact_working_set_bytes(frame, spec),
                )
            )
        design = linear_design_matrix(frame, spec)
        y = frame.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
        with threadpool_limits(limits=1, user_api="blas"):
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return beta


def iter_model_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    required: set[str],
    work_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    for frame in iter_complete_time_sized_frames(
        iter(frames),
        timestamp_column="timestamp",
        resource_policy=current_runtime_resource_policy(),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing_model_columns: {', '.join(missing)}")
        selected = [column for column in work_columns if column in frame.columns]
        selected_bytes = _selected_frame_payload_bytes(frame, selected)
        policy = current_runtime_resource_policy()
        if policy is not None and (
            policy.reserved_memory_bytes + selected_bytes
            > policy.task_memory_budget_bytes
        ):
            raise ValueError("model_training_working_set_exceeds_task_memory_budget")
        with ExitStack() as reservation:
            if policy is not None:
                reservation.enter_context(
                    policy.reserve("research.model.projection", selected_bytes)
                )
            yield frame.loc[:, selected]


def linear_design_matrix(
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    x = frame.loc[:, spec.input_columns].to_numpy(dtype="float64", copy=False)
    design = np.empty((len(x), len(spec.input_columns) + 1), dtype="float64")
    design[:, 0] = 1.0
    design[:, 1:] = x
    return design


def _estimator_fit_working_set_bytes(
    frame: pd.DataFrame,
    spec: ModelSpec,
    *,
    threads: int | None = None,
) -> int:
    rows = len(frame)
    inputs = len(spec.input_columns)
    feature_frame = _selected_frame_payload_bytes(frame, list(spec.input_columns))
    dense_features = rows * inputs * np.dtype(np.float64).itemsize
    backend_feature_copy = dense_features
    label_copy = rows * np.dtype(np.float64).itemsize
    row_index = rows * np.dtype(np.intp).itemsize
    coefficient_workspace = (inputs + 1) ** 2 * np.dtype(np.float64).itemsize
    parallel_workspace = _parallel_estimator_workspace_bytes(
        rows=rows,
        inputs=inputs,
        spec=spec,
        threads=supervised.model_thread_count(spec) if threads is None else threads,
    )
    return (
        feature_frame
        + dense_features
        + backend_feature_copy
        + label_copy
        + row_index
        + coefficient_workspace
        + parallel_workspace
    )


def _parallel_estimator_workspace_bytes(
    *,
    rows: int,
    inputs: int,
    spec: ModelSpec,
    threads: int | None = None,
) -> int:
    algorithm = spec.model_algorithm
    thread_count = supervised.model_thread_count(spec) if threads is None else threads
    integer = np.dtype(np.intp).itemsize
    floating = np.dtype(np.float64).itemsize
    if algorithm in {"random_forest_regressor", "extra_trees_regressor"}:
        return thread_count * (rows * (integer + floating) + inputs * floating)
    if algorithm == "xgboost_regressor":
        params = supervised.resolved_model_hyperparameters(spec)
        bins = _positive_model_int(params.get("max_bin"), default=256)
        quantized = rows * inputs * np.dtype(np.uint32).itemsize
        gradients = 2 * rows * floating
        histograms = thread_count * inputs * bins * 2 * floating
        row_positions = thread_count * rows * integer
        return quantized + gradients + histograms + row_positions
    if algorithm == "lightgbm_regressor":
        params = supervised.resolved_model_hyperparameters(spec)
        bins = _positive_model_int(params.get("max_bin"), default=255)
        bin_itemsize = 1 if bins <= 255 else 2 if bins <= 65_535 else 4
        quantized = rows * inputs * bin_itemsize
        gradients = 2 * rows * floating
        histograms = thread_count * inputs * bins * 2 * floating
        row_positions = thread_count * rows * integer
        return quantized + gradients + histograms + row_positions
    return 0


def _effective_model_thread_count(
    frame: pd.DataFrame,
    spec: ModelSpec,
    *,
    available_bytes: int | None,
) -> int:
    requested = supervised.model_thread_count(spec)
    if available_bytes is None:
        return requested
    if requested == 1:
        if _estimator_fit_working_set_bytes(frame, spec, threads=1) > available_bytes:
            raise MemoryError("runtime_memory_budget_exceeded")
        return 1
    base = _estimator_fit_working_set_bytes(frame, spec, threads=0)
    per_thread = _estimator_fit_working_set_bytes(frame, spec, threads=1) - base
    if per_thread <= 0:
        return requested
    effective = min(requested, max(1, (available_bytes - base) // per_thread))
    if base + effective * per_thread > available_bytes:
        raise MemoryError("runtime_memory_budget_exceeded")
    return effective


def _linear_exact_working_set_bytes(frame: pd.DataFrame, spec: ModelSpec) -> int:
    rows = len(frame)
    columns = len(spec.input_columns) + 1
    design = rows * columns * np.dtype(np.float64).itemsize
    solver_input_copy = design
    left_singular_vectors = design
    right_singular_vectors = columns * columns * np.dtype(np.float64).itemsize
    label_copy = rows * np.dtype(np.float64).itemsize
    solver_vectors = 2 * columns * np.dtype(np.float64).itemsize
    native_workspace = design + right_singular_vectors + solver_vectors
    return (
        design
        + solver_input_copy
        + left_singular_vectors
        + right_singular_vectors
        + label_copy
        + solver_vectors
        + native_workspace
    )


def _linear_qr_working_set_bytes(
    *,
    row_count: int,
    input_count: int,
    has_retained_factor: bool,
) -> int:
    columns = input_count + 1
    stacked_rows = row_count + (columns if has_retained_factor else 0)
    chunk_design = row_count * columns * np.dtype(np.float64).itemsize
    timestamp_values = row_count * np.dtype("datetime64[ns]").itemsize
    stacked_design = (
        stacked_rows * columns * np.dtype(np.float64).itemsize
        if has_retained_factor
        else 0
    )
    stacked_label = (
        stacked_rows * np.dtype(np.float64).itemsize if has_retained_factor else 0
    )
    qr_input_copy = stacked_rows * columns * np.dtype(np.float64).itemsize
    q_factor = qr_input_copy
    r_factor = columns * columns * np.dtype(np.float64).itemsize
    projected_label = columns * np.dtype(np.float64).itemsize
    native_workspace = qr_input_copy + r_factor + projected_label
    return (
        chunk_design
        + timestamp_values
        + stacked_design
        + stacked_label
        + qr_input_copy
        + q_factor
        + r_factor
        + projected_label
        + native_workspace
    )


def _model_state_upper_bound_bytes(
    frame: pd.DataFrame,
    spec: ModelSpec,
    *,
    estimator: object | None = None,
) -> int:
    rows = max(1, len(frame))
    inputs = max(1, len(spec.input_columns))
    params = supervised.resolved_model_hyperparameters(spec)
    get_params = getattr(estimator, "get_params", None)
    if callable(get_params):
        params = dict(get_params(deep=False))
    algorithm = spec.model_algorithm
    if algorithm in {
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elastic_net",
        "huber_regression",
        "single_factor_baseline",
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        return 4096 + 64 * inputs

    tree_count = 1
    leaves = 2 * rows - 1
    if algorithm in {"random_forest_regressor", "extra_trees_regressor"}:
        tree_count = _positive_model_int(params.get("n_estimators"), default=100)
        depth = _positive_model_int(params.get("max_depth"), default=0)
        leaves = _bounded_tree_nodes(rows, depth=depth)
    elif algorithm == "decision_tree_regressor":
        depth = _positive_model_int(params.get("max_depth"), default=0)
        leaves = _bounded_tree_nodes(rows, depth=depth)
    elif algorithm == "gradient_boosting_regressor":
        tree_count = _positive_model_int(params.get("n_estimators"), default=100)
        depth_value = params.get("max_depth")
        leaves = _bounded_tree_nodes(
            rows,
            depth=(
                0
                if depth_value is None
                else _positive_model_int(depth_value, default=3)
            ),
        )
    elif algorithm == "hist_gradient_boosting_regressor":
        tree_count = _positive_model_int(params.get("max_iter"), default=100)
        max_leaf_nodes = params.get("max_leaf_nodes")
        leaves = (
            2 * rows - 1
            if max_leaf_nodes is None
            else min(
                2 * rows - 1,
                2 * _positive_model_int(max_leaf_nodes, default=31),
            )
        )
    elif algorithm == "xgboost_regressor":
        tree_count = _positive_model_int(params.get("n_estimators"), default=100)
        depth_value = params.get("max_depth")
        leaves = _bounded_tree_nodes(
            rows,
            depth=(
                0 if depth_value == 0 else _positive_model_int(depth_value, default=6)
            ),
        )
    elif algorithm == "lightgbm_regressor":
        tree_count = _positive_model_int(params.get("n_estimators"), default=100)
        leaves = min(
            2 * rows - 1,
            2 * _positive_model_int(params.get("num_leaves"), default=31),
        )
    return tree_count * (4096 + max(1, leaves) * 256)


def _bounded_tree_nodes(rows: int, *, depth: int) -> int:
    row_bound = max(1, 2 * rows - 1)
    if depth <= 0 or depth >= row_bound.bit_length():
        return row_bound
    return min(row_bound, (1 << (depth + 1)) - 1)


def _positive_model_int(value: object, *, default: int) -> int:
    return value if type(value) is int and value > 0 else default


def fit_linear_coefficients_from_tsqr(
    r_factor: np.ndarray,
    qt_y: np.ndarray,
) -> np.ndarray:
    policy = current_runtime_resource_policy()
    with ExitStack() as solve_reservation:
        if policy is not None:
            columns = len(qt_y)
            solver_input_copy = int(r_factor.nbytes + qt_y.nbytes)
            right_singular_vectors = columns * columns * np.dtype(np.float64).itemsize
            solver_vectors = 2 * columns * np.dtype(np.float64).itemsize
            native_workspace = (
                solver_input_copy + right_singular_vectors + solver_vectors
            )
            solve_reservation.enter_context(
                policy.reserve(
                    "research.model.qr_solve",
                    solver_input_copy
                    + right_singular_vectors
                    + solver_vectors
                    + native_workspace,
                )
            )
        with threadpool_limits(limits=1, user_api="blas"):
            beta, *_ = np.linalg.lstsq(r_factor, qt_y, rcond=None)
        return beta


def date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _strict_non_negative_count_value(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid_model_metadata_count")
    return value


__all__ = [
    "ModelFitResult",
    "collect_training_frame",
    "fit_linear_model_result_from_frames",
    "fit_model_result_from_frames",
    "fit_predictor",
]
