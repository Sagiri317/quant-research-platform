"""Supervised model fitting owners."""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import quant_research_platform.research.models.dependencies as dependencies
import quant_research_platform.research.models.estimators as estimators
import quant_research_platform.research.models.feature_importance as feature_importance
import quant_research_platform.research.models.supervised as supervised
import quant_research_platform.research.models.validation_split as validation_split
from quant_research_platform.contracts.experiments import (
    legal_training_rows,
    resolve_training_availability_column,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.models import (
    ModelSpec,
    _model_algorithm_facts,
    _thaw_model_hyperparameters,
)
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.filtering import (
    _filter_mode_scope,
)
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
)
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
    ScaledPredictor,
    equal_date_sample_weights,
)
from quant_research_platform.runtime.dataframes import (
    _selected_frame_payload_bytes,
    iter_complete_time_sized_frames,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)


@dataclass
class ModelFitResult:
    metadata: dict[str, object]
    prediction_model: object = field(
        repr=False,
        compare=False,
    )


type _OwnedModelFitResult = OwnedValue[ModelFitResult]


def prediction_callable(
    input_columns: tuple[str, ...],
    prediction_model: object,
):
    """Build the model-owned prediction callable used by selection and output."""

    def predict_from_model(frame: pd.DataFrame) -> np.ndarray:
        columns = input_columns
        if isinstance(
            prediction_model, (RankEnsemblePredictor, ICWeightedRankEnsemble)
        ):
            columns = tuple(dict.fromkeys(("timestamp", *input_columns)))
        predicted = prediction_model.predict(frame.loc[:, list(columns)])
        return np.asarray(predicted, dtype="float64")

    return predict_from_model


def validate_model_state(predictor: object, metadata: dict[str, object]) -> None:
    """Bind persisted model metadata to the actual trained predictor state."""

    spec = ModelSpec(
        model_id="persisted_model",
        model_algorithm=metadata["model_algorithm"],
        input_factor_columns=tuple(metadata["input_factor_columns"]),
        input_feature_columns=tuple(metadata["input_feature_columns"]),
        label_column=metadata["label_column"],
        sample_filter_mode=metadata["sample_filter_mode"],
        hyperparameters=metadata["hyperparameters"],
        random_seed=metadata["random_seed"],
        max_train_rows=metadata["max_train_rows"],
    )
    if spec.model_algorithm == "linear_regression":
        if type(predictor) is not LinearScaledPredictor:
            raise ValueError("model_state_metadata_mismatch")
        try:
            coefficients = np.asarray(predictor.coefficients, dtype="float64")
        except (TypeError, ValueError):
            raise ValueError("model_state_metadata_mismatch") from None
        if (
            coefficients.shape != (len(spec.input_columns) + 1,)
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("model_state_metadata_mismatch")
        importance = feature_importance.linear_feature_importance(spec, coefficients)
    elif spec.model_algorithm in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        expected_type = (
            RankEnsemblePredictor
            if spec.model_algorithm == "equal_weight_rank_ensemble"
            else ICWeightedRankEnsemble
        )
        if type(predictor) is not expected_type:
            raise ValueError("model_state_metadata_mismatch")
        parameters = {"columns": list(predictor.columns)}
        if isinstance(predictor, RankEnsemblePredictor):
            parameters["directions"] = list(predictor.directions)
            if any(direction not in (-1.0, 1.0) for direction in predictor.directions):
                raise ValueError("model_state_metadata_mismatch")
        else:
            parameters["weights"] = list(predictor.weights)
            if not np.isclose(
                sum(abs(weight) for weight in predictor.weights),
                1.0,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("model_state_metadata_mismatch")
        if (
            metadata.get("rank_parameters") != parameters
            or metadata["scaler"] is not None
        ):
            raise ValueError("model_state_metadata_mismatch")
        importance = feature_importance.estimator_feature_importance(spec, predictor)
    else:
        if type(predictor) is not ScaledPredictor:
            raise ValueError("model_state_metadata_mismatch")
        expected = estimators._MODEL_ESTIMATOR_FACTORIES[spec.model_algorithm](
            spec, dict(metadata["resolved_hyperparameters"])
        )
        actual = predictor.estimator
        if type(actual) is not type(expected):
            raise ValueError("model_state_metadata_mismatch")
        actual_parameters, expected_parameters = (
            actual.get_params(deep=False),
            expected.get_params(deep=False),
        )
        if set(actual_parameters) != set(expected_parameters):
            raise ValueError("model_state_metadata_mismatch")
        for name, expected_value in expected_parameters.items():
            actual_value = actual_parameters[name]
            if isinstance(expected_value, float) and np.isnan(expected_value):
                matches = isinstance(actual_value, float) and np.isnan(actual_value)
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError("model_state_metadata_mismatch")
        try:
            inputs_match = (
                actual.n_features_in_ == len(spec.input_columns)
                and tuple(actual.feature_names_in_) == spec.input_columns
            )
        except AttributeError:
            raise ValueError("model_state_metadata_mismatch") from None
        if not inputs_match:
            raise ValueError("model_state_metadata_mismatch")
        importance = feature_importance.estimator_feature_importance(spec, predictor)
    if isinstance(predictor, (LinearScaledPredictor, ScaledPredictor)):
        if (
            type(predictor.scaler) is not ModelMatrixScaler
            or predictor.scaler.to_payload() != metadata["scaler"]
        ):
            raise ValueError("model_state_metadata_mismatch")
    if importance != metadata.get("feature_importance"):
        raise ValueError("model_state_metadata_mismatch")


def _fit_model_result_owned_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    estimator: object,
    fitting_cutoff: object,
) -> ModelFitResult | _OwnedModelFitResult:
    if spec.model_algorithm == "linear_regression":
        return _fit_linear_model_result_owned_from_frames(
            frames, spec, fitting_cutoff=fitting_cutoff
        )
    return _fit_estimator_model_result_owned_from_frames(
        frames,
        spec,
        estimator=estimator,
        fitting_cutoff=fitting_cutoff,
    )


def _fit_linear_model_result_owned_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    fitting_cutoff: object,
) -> ModelFitResult | _OwnedModelFitResult:
    policy = current_runtime_resource_policy()
    model_lease = (
        None
        if policy is None
        else policy.lease(
            "research.model.fitted",
            _linear_model_state_upper_bound_bytes(spec),
        )
    )
    try:
        return _fit_linear_model_result_with_lease(
            frames,
            spec,
            model_lease=model_lease,
            fitting_cutoff=fitting_cutoff,
        )
    except BaseException as primary:
        if model_lease is not None:
            try:
                model_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    f"model lease release also failed: {type(cleanup_error).__name__}"
                )
        raise


def _fit_linear_model_result_with_lease(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    model_lease: MemoryLease | None,
    fitting_cutoff: object,
) -> ModelFitResult | _OwnedModelFitResult:
    owned_training = _collect_training_frame_owned(
        frames, spec, fitting_cutoff=fitting_cutoff
    )
    training, row_counts, filter_metadata = (
        owned_training.value
        if isinstance(owned_training, OwnedValue)
        else owned_training
    )
    training_lease = (
        owned_training.lease if isinstance(owned_training, OwnedValue) else None
    )
    weights = matrix = transformed = design = weighted = None
    beta = scaler = predictor = metadata = result = None
    try:
        if training.empty:
            raise ValueError("model_training_sample_empty")
        if model_lease is not None:
            policy = current_runtime_resource_policy()
            fit_work_bytes = _estimator_fit_working_set_bytes(training, spec, threads=1)
            policy.require_working_set(
                fit_work_bytes,
                error="model_training_working_set_exceeds_task_memory_budget",
            )
            model_lease.grow(fit_work_bytes)
        weights = equal_date_sample_weights(training["timestamp"])
        matrix = training.loc[:, spec.input_columns]
        scaler = ModelMatrixScaler.fit(matrix, sample_weight=weights)
        transformed = scaler.transform(matrix).to_numpy(dtype="float64")
        design = np.column_stack((np.ones(len(transformed)), transformed))
        weighted = np.sqrt(weights)
        with threadpool_limits(limits=1, user_api="blas"):
            beta = np.linalg.lstsq(
                design * weighted[:, None],
                training[spec.label_column].to_numpy(dtype="float64") * weighted,
                rcond=None,
            )[0]
        predictor = LinearScaledPredictor(
            coefficients=tuple(float(value) for value in beta),
            scaler=scaler,
        )
        metadata = model_metadata(
            spec,
            filter_metadata=filter_metadata,
            row_counts=row_counts,
            feature_importance=feature_importance.linear_feature_importance(spec, beta),
            scaler=scaler,
        )
        result = ModelFitResult(
            metadata=metadata,
            prediction_model=predictor,
        )
        weights = matrix = transformed = design = weighted = None
        training = owned_training = None
        if training_lease is not None:
            training_lease.close()
            training_lease = None
        if model_lease is None:
            return result
        model_lease.shrink_to(_linear_model_state_upper_bound_bytes(spec))
        return OwnedValue(result, model_lease)
    except BaseException as primary:
        weights = matrix = transformed = design = weighted = None
        beta = scaler = predictor = metadata = result = None
        training = owned_training = None
        if training_lease is not None:
            try:
                training_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "linear training lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise


def model_metadata(
    spec: ModelSpec,
    *,
    filter_metadata: dict[str, object],
    row_counts: dict[str, object],
    feature_importance: dict[str, float] | None = None,
    feature_importance_status: str | None = None,
    resolved_hyperparameters: dict[str, object] | None = None,
    scaler: ModelMatrixScaler | None = None,
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
        "input_factor_columns": list(spec.input_factor_columns),
        "input_feature_columns": list(spec.input_feature_columns),
        "input_columns": list(spec.input_columns),
        "input_count": len(spec.input_columns),
        "label_column": spec.label_column,
        "hyperparameters": _thaw_model_hyperparameters(spec.hyperparameters),
        "resolved_hyperparameters": (
            supervised.resolved_model_hyperparameters(spec)
            if resolved_hyperparameters is None
            else _thaw_model_hyperparameters(resolved_hyperparameters)
        ),
        "random_seed": spec.random_seed,
        "max_train_rows": spec.max_train_rows,
        "dependency_status": dependencies.require_model_dependency(
            spec.model_algorithm,
        ),
        "sample_weight_protocol": "equal_date_total_equal_sum_n_train",
        "scaler": None if scaler is None else scaler.to_payload(),
    }
    if feature_importance is not None:
        metadata["feature_importance"] = feature_importance
    else:
        metadata["feature_importance_status"] = (
            feature_importance_status or "not_available_for_model_algorithm"
        )
    return metadata


def _fit_estimator_model_result_owned_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    estimator: object,
    fitting_cutoff: object,
) -> ModelFitResult | _OwnedModelFitResult:
    owned_training = _collect_training_frame_owned(
        frames, spec, fitting_cutoff=fitting_cutoff
    )
    training, row_counts, filter_metadata = (
        owned_training.value
        if isinstance(owned_training, OwnedValue)
        else owned_training
    )
    training_lease = (
        owned_training.lease if isinstance(owned_training, OwnedValue) else None
    )
    if row_counts["rows_after_input_label_dropna"] == 0:
        primary = ValueError("model_training_sample_empty")
        training = None
        owned_training = None
        if training_lease is not None:
            try:
                training_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "training lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise primary
    policy = current_runtime_resource_policy()
    model_lease = None
    y: np.ndarray | None = None
    predictor: object | None = None
    metadata: dict[str, object] | None = None
    resolved_hyperparameters: dict[str, object] | None = None
    result: ModelFitResult | None = None
    try:
        target_bytes = len(training) * np.dtype(np.float64).itemsize
        model_state_bytes = _model_state_upper_bound_bytes(
            training,
            spec,
            estimator=estimator,
        )
        thread_available_bytes = None
        if policy is not None:
            thread_available_bytes = max(
                0,
                policy.operation_headroom_bytes - target_bytes - model_state_bytes,
            )
        threads = _effective_model_thread_count(
            training,
            spec,
            available_bytes=thread_available_bytes,
        )
        if policy is not None:
            fit_work_bytes = _estimator_fit_working_set_bytes(
                training,
                spec,
                threads=threads,
            )
            policy.require_working_set(
                target_bytes + fit_work_bytes + model_state_bytes,
                error="model_training_working_set_exceeds_task_memory_budget",
            )
            model_lease = policy.lease(
                "research.model.fitted",
                target_bytes + fit_work_bytes + model_state_bytes,
            )
    except BaseException as primary:
        training = None
        owned_training = None
        if training_lease is not None:
            try:
                training_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "training lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    try:
        y = training.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
        predictor = _fit_predictor(
            spec,
            training=training,
            y=y,
            estimator=estimator,
            threads=threads,
        )
        y = None
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
            feature_importance=feature_importance.estimator_feature_importance(
                spec,
                predictor,
            ),
            resolved_hyperparameters=resolved_hyperparameters,
            scaler=(
                predictor.scaler if isinstance(predictor, ScaledPredictor) else None
            ),
        )
        result = ModelFitResult(
            metadata=metadata,
            prediction_model=predictor,
        )
        if model_lease is None:
            training = None
            owned_training = None
            if training_lease is not None:
                training_lease.close()
            return result
        training = None
        owned_training = None
        if training_lease is not None:
            training_lease.close()
            training_lease = None
        model_lease.shrink_to(model_state_bytes)
        return OwnedValue(result, model_lease)
    except BaseException as primary:
        if y is not None:
            y = None
        predictor = None
        metadata = None
        resolved_hyperparameters = None
        result = None
        training = None
        owned_training = None
        if training_lease is not None:
            try:
                training_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "training lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if model_lease is not None:
            try:
                model_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    f"model lease release also failed: {type(cleanup_error).__name__}"
                )
        raise


def _collect_training_frame_owned(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    fitting_cutoff: object,
) -> (
    tuple[pd.DataFrame, dict[str, object], dict[str, object]]
    | OwnedValue[tuple[pd.DataFrame, dict[str, object], dict[str, object]]]
):
    required = set(spec.input_columns)
    required.add(spec.label_column)
    required.add("timestamp")
    required.add("eligible")
    availability_columns = training_availability_columns(spec.label_column)
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *spec.input_columns,
                spec.label_column,
                "eligible",
                *availability_columns,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    rows_input = 0
    rows_after_filter = 0
    filter_columns_used: list[str] = []
    filter_definition = ""
    retained: deque[tuple[pd.DataFrame, MemoryLease | None, int]] = deque()
    retained_rows = 0
    retained_bytes = 0
    training_lease: MemoryLease | None = None
    projection_lease: MemoryLease | None = None
    try:
        for chunk in iter_model_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ):
            resource_policy = current_runtime_resource_policy()
            with ExitStack() as filter_work:
                filtered, metadata = filter_work.enter_context(
                    _filter_mode_scope(
                        chunk,
                        mode=spec.sample_filter_mode,
                        metadata_key="sample_filter_mode",
                    )
                )
                rows_input += _strict_non_negative_count_value(metadata["rows_input"])
                rows_after_filter += _strict_non_negative_count_value(
                    metadata["rows_after_filter_mode"]
                )
                filter_columns_used = list(metadata["filter_columns_used"])
                filter_definition = str(metadata["filter_definition"])
                identity_columns = (
                    ("asset_id",) if "asset_id" in filtered.columns else ()
                )
                selected_columns = [
                    "timestamp",
                    *identity_columns,
                    *spec.input_columns,
                    spec.label_column,
                ]
                usable_bound = _selected_frame_payload_bytes(
                    filtered,
                    selected_columns,
                )
                projection_lease = None
                if resource_policy is not None:
                    projection_bound = (
                        usable_bound
                        + _training_control_working_set_bytes(
                            usable_bound, len(filtered)
                        )
                    )
                    resource_policy.require_working_set(
                        projection_bound,
                        error="model_training_working_set_exceeds_task_memory_budget",
                    )
                    projection_lease = resource_policy.lease(
                        "research.model.training_projection", projection_bound
                    )
                availability_column = resolve_training_availability_column(
                    filtered,
                    label_column=spec.label_column,
                )
                usable = legal_training_rows(
                    filtered,
                    fitting_cutoff=fitting_cutoff,
                    availability_column=availability_column,
                    feature_columns=(
                        ()
                        if _model_algorithm_facts(spec.model_algorithm).input_kind
                        == "rank_ensemble_inputs"
                        else spec.input_columns
                    ),
                    target_column=spec.label_column,
                ).loc[:, selected_columns]
                usable_payload_bytes = pandas_frame_payload_bytes(usable)
                if usable_payload_bytes > usable_bound:
                    raise ValueError(
                        "model_training_working_set_exceeds_task_memory_budget"
                    )
                if usable.empty:
                    usable = None
                    filtered = None
                    if projection_lease is not None:
                        projection_lease.close()
                        projection_lease = None
                    continue
                if spec.max_train_rows is not None:
                    usable = validation_split.trim_training_buffer(usable, spec)
                    usable_payload_bytes = pandas_frame_payload_bytes(usable)
                if projection_lease is not None:
                    projection_lease.shrink_to(usable_payload_bytes)
                    projection_lease.transfer("research.model.training")
                retained.append((usable, projection_lease, usable_payload_bytes))
                projection_lease = None
                retained_rows += len(usable)
                retained_bytes += usable_payload_bytes
                usable = None
                filtered = None
                if spec.max_train_rows is not None:
                    while retained_rows - len(retained[0][0]) >= spec.max_train_rows:
                        removed, removed_lease, removed_bytes = retained.popleft()
                        retained_rows -= len(removed)
                        retained_bytes -= removed_bytes
                        removed = None
                        if removed_lease is not None:
                            removed_lease.close()
                    excess = retained_rows - spec.max_train_rows
                    if excess > 0:
                        oldest, oldest_lease, oldest_bytes = retained[0]
                        if resource_policy is not None:
                            resource_policy.require_working_set(
                                oldest_bytes,
                                error="model_training_working_set_exceeds_task_memory_budget",
                            )
                            projection_lease = resource_policy.lease(
                                "research.model.training_trim", oldest_bytes
                            )
                        trimmed = oldest.iloc[excess:].copy()
                        trimmed_bytes = pandas_frame_payload_bytes(trimmed)
                        if projection_lease is not None:
                            projection_lease.shrink_to(trimmed_bytes)
                            projection_lease.transfer("research.model.training")
                        retained[0] = (trimmed, projection_lease, trimmed_bytes)
                        projection_lease = None
                        retained_rows -= excess
                        retained_bytes += trimmed_bytes - oldest_bytes
                        oldest = trimmed = None
                        if oldest_lease is not None:
                            oldest_lease.close()
        resource_policy = current_runtime_resource_policy()
        if len(retained) == 1:
            training, training_lease, _ = retained.popleft()
        elif retained:
            if resource_policy is not None:
                resource_policy.require_working_set(
                    retained_bytes,
                    error="model_training_working_set_exceeds_task_memory_budget",
                )
                training_lease = resource_policy.lease(
                    "research.model.training_concat", retained_bytes
                )
            training = pd.concat([part for part, _, _ in retained], ignore_index=True)
            while retained:
                removed, removed_lease, _ = retained.popleft()
                removed = None
                if removed_lease is not None:
                    removed_lease.close()
            if training_lease is not None:
                training_lease.shrink_to(pandas_frame_payload_bytes(training))
                training_lease.transfer("research.model.training")
        else:
            training = pd.DataFrame()
        final_lease = None
        if resource_policy is not None:
            control_bound = _training_control_working_set_bytes(
                pandas_frame_payload_bytes(training), len(training)
            )
            resource_policy.require_working_set(
                control_bound,
                error="model_training_working_set_exceeds_task_memory_budget",
            )
            final_lease = resource_policy.lease(
                "research.model.training_controls", control_bound
            )
        previous_training = training
        try:
            controlled_training = validation_split.apply_training_controls(
                training,
                spec,
            )
            training = controlled_training
            if final_lease is not None:
                if controlled_training is previous_training:
                    final_lease.close()
                    final_lease = None
                else:
                    previous_training = None
                    final_lease.shrink_to(pandas_frame_payload_bytes(training))
                    final_lease.transfer("research.model.training")
                    previous_training_lease = training_lease
                    training_lease = final_lease
                    final_lease = None
                    if previous_training_lease is not None:
                        previous_training_lease.close()
            controlled_training = None
            previous_training = None
        except BaseException as primary:
            previous_training = None
            controlled_training = None
            training = None
            if final_lease is not None:
                try:
                    final_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "training controls lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise
    except BaseException as primary:
        filtered = None
        usable = None
        training = None
        while retained:
            removed, removed_lease, _ = retained.popleft()
            removed = None
            if removed_lease is not None:
                try:
                    removed_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "retained training lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
        if projection_lease is not None:
            try:
                projection_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "training projection lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if training_lease is not None:
            try:
                training_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "training lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    resource_policy = current_runtime_resource_policy()
    result_lease = None
    previous_training_lease = None
    timestamps = None
    previous_training = None
    try:
        training_bytes = pandas_frame_payload_bytes(training)
        output_bound = (
            _selected_frame_payload_bytes(
                training,
                [column for column in training.columns if column != "asset_id"],
            )
            if "asset_id" in training.columns
            else training_bytes
        )
        has_identity = "asset_id" in training.columns
        result_bound = (output_bound if has_identity else 0) + len(training) * 16
        if resource_policy is not None:
            result_lease = resource_policy.lease(
                "research.model.training_result",
                result_bound,
            )
        if has_identity:
            previous_training = training
            training = previous_training.drop(columns="asset_id")
            previous_training = None
        train_start = None
        train_end = None
        if not training.empty:
            timestamps = strict_timestamp_series(
                training["timestamp"],
                error="invalid_model_training_timestamp",
            )
            train_start = time_value_isoformat(pd.Timestamp(timestamps.min()))
            train_end = time_value_isoformat(pd.Timestamp(timestamps.max()))
        timestamps = None
        if result_lease is not None and has_identity:
            result_lease.shrink_to(pandas_frame_payload_bytes(training))
            result_lease.transfer("research.model.training")
            previous_training_lease = training_lease
            training_lease = result_lease
            result_lease = None
            if previous_training_lease is not None:
                previous_training_lease.close()
                previous_training_lease = None
        elif result_lease is not None:
            result_lease.close()
            result_lease = None
    except BaseException as primary:
        previous_training = None
        timestamps = None
        training = None
        if result_lease is not None:
            try:
                result_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "training result lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if (
            previous_training_lease is not None
            and previous_training_lease is not training_lease
        ):
            try:
                previous_training_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "previous training lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if training_lease is not None:
            try:
                training_lease.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "training lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
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
    result = training, row_counts, filter_metadata
    if training_lease is None:
        return result
    return OwnedValue(result, training_lease)


def _training_control_working_set_bytes(frame_bytes: int, row_count: int) -> int:
    copied_frame = frame_bytes
    sorted_frame = frame_bytes
    timestamp_values = row_count * np.dtype("datetime64[ns]").itemsize
    sort_index = row_count * np.dtype(np.intp).itemsize
    return copied_frame + sorted_frame + timestamp_values + sort_index


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
    from sklearn.exceptions import ConvergenceWarning

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
    if spec.model_algorithm not in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        matrix = training.loc[:, spec.input_columns]
        weights = equal_date_sample_weights(training["timestamp"])
        scaler = ModelMatrixScaler.fit(matrix, sample_weight=weights)
        scaled_matrix = scaler.transform(matrix)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                with (
                    threadpool_limits(limits=1, user_api="blas"),
                    threadpool_limits(
                        limits=openmp_threads,
                        user_api="openmp",
                    ),
                ):
                    estimator.fit(scaled_matrix, y, sample_weight=weights)
        except ConvergenceWarning:
            raise ValueError("model_fit_non_converged") from None
        return ScaledPredictor(estimator, scaler)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with (
                threadpool_limits(limits=1, user_api="blas"),
                threadpool_limits(
                    limits=openmp_threads,
                    user_api="openmp",
                ),
            ):
                estimator.fit(training.loc[:, spec.input_columns], y)
    except ConvergenceWarning:
        raise ValueError("model_fit_non_converged") from None
    return estimator


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
        if policy is not None:
            policy.require_working_set(
                selected_bytes,
                error="model_training_working_set_exceeds_task_memory_budget",
            )
        projection_lease = None
        if policy is not None:
            projection_lease = policy.lease(
                "research.model.projection",
                selected_bytes,
            )
        try:
            yield frame.loc[:, selected]
        except BaseException as primary:
            if projection_lease is not None:
                try:
                    projection_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "model projection lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise
        else:
            if projection_lease is not None:
                projection_lease.close()


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
    if spec.model_algorithm == "linear_regression":
        # Cover scaling temporaries, intercept design, weighted LAPACK inputs/copies,
        # equal-date grouping, and the dense least-squares workspace together.
        return (
            feature_frame
            + 8 * dense_features
            + 12 * label_copy
            + 4 * row_index
            + 4 * coefficient_workspace
        )
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
    if algorithm == "random_forest_regressor":
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
    if available_bytes is None or requested == 1:
        return requested
    base = _estimator_fit_working_set_bytes(frame, spec, threads=0)
    per_thread = _estimator_fit_working_set_bytes(frame, spec, threads=1) - base
    if per_thread <= 0:
        return requested
    return min(requested, max(1, (available_bytes - base) // per_thread))


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
        "huber_regression",
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        return 4096 + 64 * inputs

    tree_count = 1
    leaves = 2 * rows - 1
    if algorithm == "random_forest_regressor":
        tree_count = _positive_model_int(params.get("n_estimators"), default=100)
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
    elif algorithm == "xgboost_regressor":
        tree_count = _positive_model_int(params.get("n_estimators"), default=100)
        tree_count *= _positive_model_int(
            params.get("num_parallel_tree"),
            default=1,
        )
        depth_value = params.get("max_depth")
        leaves = _bounded_tree_nodes(
            rows,
            depth=(
                0
                if (
                    isinstance(depth_value, Integral)
                    and not isinstance(depth_value, bool)
                    and depth_value == 0
                )
                else _positive_model_int(depth_value, default=6)
            ),
        )
    elif algorithm == "lightgbm_regressor":
        tree_count = _positive_model_int(params.get("n_estimators"), default=100)
        leaves = min(
            2 * rows - 1,
            2 * _positive_model_int(params.get("num_leaves"), default=31),
        )
    return tree_count * (4096 + max(1, leaves) * 256)


def _linear_model_state_upper_bound_bytes(spec: ModelSpec) -> int:
    return 4096 + 64 * max(1, len(spec.input_columns))


def _bounded_tree_nodes(rows: int, *, depth: int) -> int:
    row_bound = max(1, 2 * rows - 1)
    if depth <= 0 or depth >= row_bound.bit_length():
        return row_bound
    return min(row_bound, (1 << (depth + 1)) - 1)


def model_training_read_columns(
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
                "eligible",
                *training_availability_columns(label_column),
                *required_filter_columns(sample_filter_mode),
            )
        )
    )


def iter_legal_training_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    train_end: pd.Timestamp,
    label_column: str,
    input_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    for chunk in chunks:
        policy = current_runtime_resource_policy()
        index_bytes = int(chunk.index.memory_usage(deep=True))
        timestamp_bytes = index_bytes + 8 * len(chunk)
        mask_bytes = index_bytes + len(chunk)
        payload_bytes = pandas_frame_payload_bytes(chunk)
        working_bytes = timestamp_bytes + mask_bytes + payload_bytes
        if (
            policy is not None
            and policy.explicit_memory_cap_bytes is not None
            and policy.used_memory_bytes + working_bytes
            > policy.explicit_memory_cap_bytes
        ):
            raise ValueError(
                "prediction_training_working_set_exceeds_task_memory_budget"
            )
        working_lease = (
            policy.lease("research.model.training.chunk", working_bytes)
            if policy is not None
            else None
        )
        train = None
        try:
            availability_column = resolve_training_availability_column(
                chunk,
                label_column=label_column,
            )
            train = legal_training_rows(
                chunk,
                fitting_cutoff=train_end,
                availability_column=availability_column,
                feature_columns=input_columns,
                target_column=label_column,
            )
            if train.empty:
                train = None
                if working_lease is not None:
                    working_lease.close()
                continue
            retained_bytes = pandas_frame_payload_bytes(train)
            if working_lease is not None:
                working_lease.shrink_to(retained_bytes)
                working_lease.transfer("research.model.training")
            try:
                yield train
            except BaseException as primary:
                train = None
                if working_lease is not None:
                    try:
                        working_lease.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            "model training lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                raise
            else:
                train = None
                if working_lease is not None:
                    working_lease.close()
        except BaseException as primary:
            train = None
            if working_lease is not None and working_lease.active:
                try:
                    working_lease.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "model training lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise


def training_scope_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    asset_ids: tuple[str, ...] = (),
) -> Iterable[pd.DataFrame]:
    """Apply only the caller's asset scope; fitting owns semantic filtering."""

    with closing_iterator(iter(chunks)) as active_chunks:
        for chunk in active_chunks:
            if not asset_ids:
                yield chunk
                continue
            if "asset_id" not in chunk.columns:
                raise ValueError("model_training_asset_id_missing")
            policy = current_runtime_resource_policy()
            scope_lease = None
            if policy is not None:
                source_bytes = pandas_frame_payload_bytes(chunk)
                mask_bytes = int(chunk.index.memory_usage(deep=True)) + len(chunk)
                scope_lease = policy.lease(
                    "research.model.training.asset_scope",
                    max(1, source_bytes + mask_bytes),
                )
            scoped = None
            primary_error: BaseException | None = None
            try:
                scoped = chunk.loc[chunk["asset_id"].isin(asset_ids)]
                if scoped.empty:
                    scoped = None
                    continue
                if scope_lease is not None:
                    scope_lease.shrink_to(max(1, pandas_frame_payload_bytes(scoped)))
                    scope_lease.transfer("research.model.training.asset_scope")
                yield scoped
            except BaseException as exc:
                primary_error = exc
                raise
            finally:
                scoped = None
                if scope_lease is not None and scope_lease.active:
                    try:
                        scope_lease.close()
                    except BaseException as cleanup_error:
                        if primary_error is not None:
                            primary_error.add_note(
                                "asset scope lease release also failed: "
                                f"{type(cleanup_error).__name__}"
                            )
                        else:
                            raise


def _positive_model_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
        return int(value)
    raise ValueError("invalid_model_state_sizing_parameter")


def _strict_non_negative_count_value(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid_model_metadata_count")
    return value
