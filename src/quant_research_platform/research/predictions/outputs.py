"""Prediction output builders."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import (
    reject_filter_input_columns,
    required_filter_columns,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.research._memory import (
    _selected_frame_payload_bytes,
    iter_complete_time_sized_frames,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.algorithm_facts import algorithm_facts
from quant_research_platform.research.models.prediction import (
    prediction_model_input_frame,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)


@dataclass(frozen=True)
class PredictionResult:
    spec: PredictionSpec
    output_chunks: Iterable[pd.DataFrame]
    model_metadata: dict[str, object]
    prediction_metadata_factory: Callable[[], dict[str, object]]
    _close: Callable[[], None] = field(
        default=lambda: None,
        repr=False,
        compare=False,
    )

    def close(self) -> None:
        self._close()

    def __enter__(self) -> PredictionResult:
        return self

    def __exit__(
        self,
        _exception_type: object,
        primary: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                "prediction result cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )


def build_prediction_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: PredictionSpec,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    *,
    prediction_model: object | None = None,
    model_metadata: dict[str, object],
    retained_model_bytes: int = 0,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> PredictionResult:
    if type(model_metadata) is not dict:
        raise ValueError("invalid_model_metadata")
    model_algorithm = model_metadata.get("model_algorithm")
    hyperparameters = model_metadata.get("hyperparameters")
    resolved_hyperparameters = model_metadata.get("resolved_hyperparameters")
    dependency_status = model_metadata.get("dependency_status")
    if (
        type(model_algorithm) is not str
        or not model_algorithm
        or type(hyperparameters) is not dict
        or type(resolved_hyperparameters) is not dict
        or type(dependency_status) is not dict
    ):
        raise ValueError("invalid_model_metadata")
    try:
        facts = algorithm_facts(model_algorithm)
    except ValueError as exc:
        raise ValueError("invalid_model_metadata") from exc
    expected_hyperparameters = dict(facts.default_hyperparameters)
    expected_package = facts.dependency_package or "internal"
    expected_hyperparameters.update(hyperparameters)
    if (
        resolved_hyperparameters != expected_hyperparameters
        or dependency_status.get("package") != expected_package
    ):
        raise ValueError("invalid_model_metadata")
    reject_filter_input_columns(input_columns)
    required = set((*spec.primary_key, *input_columns))
    _validate_oos_split(spec)
    work_columns = tuple(
        dict.fromkeys(
            (
                *spec.primary_key,
                *input_columns,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    build_state = _PredictionBuildState()
    output_frames = _iter_prediction_output_frames(
        _iter_prediction_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ),
        spec=spec,
        coefficients=coefficients,
        input_columns=input_columns,
        predictor=prediction_callable(coefficients, input_columns, prediction_model),
        build_state=build_state,
    )
    bound_policy = bind_runtime_resource_policy(resource_policy)
    if retained_model_bytes and bound_policy is not None:
        output_frames = _reserved_prediction_output_frames(
            output_frames,
            resource_policy=bound_policy,
            retained_model_bytes=retained_model_bytes,
        )
    return PredictionResult(
        spec=spec,
        output_chunks=output_frames,
        model_metadata={
            **model_metadata,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
            "oos_start": spec.test_start,
        },
        prediction_metadata_factory=lambda: build_state.prediction_metadata(
            spec=spec,
            input_columns=input_columns,
        ),
    )


def _reserved_prediction_output_frames(
    frames: Iterable[pd.DataFrame],
    *,
    resource_policy: RuntimeResourcePolicy,
    retained_model_bytes: int,
) -> Iterable[pd.DataFrame]:
    with runtime_resource_policy_scope(resource_policy), ExitStack() as retained:
        retained.enter_context(
            resource_policy.reserve(
                "research.model.fitted_state",
                retained_model_bytes,
            )
        )
        yield from frames


@dataclass
class _PredictionBuildState:
    rows_input: int = 0
    rows_after_filter: int = 0
    rows_after_dropna: int = 0
    prediction_rows: int = 0
    filter_columns_used: tuple[str, ...] = ()
    filter_definition: str = ""
    effective_test_end: str | None = None

    def record_filter_metadata(self, metadata: dict[str, object]) -> None:
        self.rows_input += _strict_non_negative_prediction_count(metadata["rows_input"])
        self.rows_after_filter += _strict_non_negative_prediction_count(
            metadata["rows_after_filter_mode"]
        )
        self.filter_columns_used = tuple(metadata["filter_columns_used"])
        self.filter_definition = str(metadata["filter_definition"])

    def record_input_dropna_rows(self, row_count: int) -> None:
        self.rows_after_dropna += int(row_count)

    def record_prediction_output(self, output: pd.DataFrame) -> None:
        if output.empty:
            return
        timestamps = strict_timestamp_series(
            output["timestamp"], error="invalid_prediction_timestamp_values"
        )
        current_end = date_string(timestamps.max())
        if self.effective_test_end is None or pd.Timestamp(current_end) > pd.Timestamp(
            self.effective_test_end
        ):
            self.effective_test_end = current_end
        self.prediction_rows += int(len(output))

    def prediction_metadata(
        self,
        *,
        spec: PredictionSpec,
        input_columns: tuple[str, ...],
    ) -> dict[str, object]:
        if self.prediction_rows == 0 or self.effective_test_end is None:
            raise ValueError("empty_prediction_window")
        return {
            "prediction_mode": spec.prediction_mode,
            "sample_filter_mode": spec.sample_filter_mode,
            "filter_columns_used": list(self.filter_columns_used),
            "filter_definition": self.filter_definition,
            "rows_input": int(self.rows_input),
            "rows_after_filter_mode": int(self.rows_after_filter),
            "rows_after_input_dropna": int(self.rows_after_dropna),
            "prediction_rows": int(self.prediction_rows),
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "requested_test_end": spec.test_end,
            "effective_test_end": self.effective_test_end,
            "is_oos": spec.is_oos,
            "is_final_oos_window": spec.is_final_oos_window,
            "input_columns": list(input_columns),
            "input_count": len(input_columns),
            "prediction_column": spec.prediction_column,
        }


def _iter_prediction_output_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    predictor: Callable[[pd.DataFrame], np.ndarray],
    build_state: _PredictionBuildState,
) -> Iterable[pd.DataFrame]:
    usable_columns = tuple(dict.fromkeys((*spec.primary_key, *input_columns)))
    for chunk in frames:
        filtered, metadata = apply_filter_mode(
            chunk,
            mode=spec.sample_filter_mode,
            metadata_key="sample_filter_mode",
        )
        build_state.record_filter_metadata(metadata)
        policy = current_runtime_resource_policy()
        row_count = len(filtered)
        index_bytes = int(filtered.index.memory_usage(deep=True))
        mask_bytes = index_bytes + row_count * (len(input_columns) + 2)
        test_frame_bytes = _selected_frame_payload_bytes(filtered, usable_columns)
        output_bytes = _selected_frame_payload_bytes(filtered, spec.primary_key) + (
            8 * row_count
        )
        model_matrix_bytes = 8 * row_count * (len(input_columns) + 1)
        working_bytes = (
            pandas_frame_payload_bytes(filtered)
            + mask_bytes
            + test_frame_bytes
            + output_bytes
            + model_matrix_bytes
        )
        if policy is not None and (
            policy.reserved_memory_bytes + working_bytes
            > policy.task_memory_budget_bytes
        ):
            raise ValueError("prediction_working_set_exceeds_task_memory_budget")
        with ExitStack() as working:
            if policy is not None:
                working.enter_context(
                    policy.reserve("research.prediction.output", working_bytes)
                )
            usable_mask = filtered.loc[:, list(input_columns)].notna().all(axis=1)
            build_state.record_input_dropna_rows(int(usable_mask.sum()))
            usable_mask &= filtered["timestamp"] >= pd.Timestamp(spec.test_start)
            if spec.test_end is not None:
                usable_mask &= filtered["timestamp"] <= pd.Timestamp(spec.test_end)
            test_frame = filtered.loc[usable_mask, list(usable_columns)]
            if test_frame.empty:
                continue
            output = test_frame.loc[:, list(spec.primary_key)].copy()
            output[spec.prediction_column] = predictor(test_frame)
            _validate_prediction_output_values(
                output,
                primary_key=spec.primary_key,
                prediction_column=spec.prediction_column,
            )
        retained_bytes = pandas_frame_payload_bytes(output)
        if policy is not None and (
            policy.reserved_memory_bytes + retained_bytes
            > policy.task_memory_budget_bytes
        ):
            raise ValueError("prediction_working_set_exceeds_task_memory_budget")
        build_state.record_prediction_output(output)
        with ExitStack() as retained:
            if policy is not None:
                retained.enter_context(
                    policy.reserve("research.prediction.result", retained_bytes)
                )
            yield output


def prediction_callable(
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    prediction_model: object | None,
) -> Callable[[pd.DataFrame], np.ndarray]:
    if prediction_model is not None:

        def predict_from_model(frame: pd.DataFrame) -> np.ndarray:
            predicted = prediction_model.predict(
                prediction_model_input_frame(
                    frame,
                    input_columns=input_columns,
                    prediction_model=prediction_model,
                )
            )
            return np.asarray(predicted, dtype="float64")

        return predict_from_model

    def predict_from_coefficients(frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, input_columns].to_numpy(dtype=float, copy=False)
        return np.asarray(float(coefficients[0]) + x @ coefficients[1:])

    return predict_from_coefficients


def _validate_oos_split(spec: PredictionSpec) -> None:
    if pd.Timestamp(spec.train_end) >= pd.Timestamp(spec.test_start):
        raise ValueError("prediction_train_end_must_precede_test_start")
    if spec.test_end is not None and pd.Timestamp(spec.test_end) < pd.Timestamp(
        spec.test_start
    ):
        raise ValueError("prediction_test_end_must_not_precede_test_start")


def date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _iter_prediction_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    required: set[str],
    work_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    for frame in iter_complete_time_sized_frames(
        iter(frames),
        timestamp_column="timestamp",
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing_prediction_columns: {', '.join(missing)}")
        selected = [column for column in work_columns if column in frame.columns]
        selected_bytes = _selected_frame_payload_bytes(frame, selected)
        policy = current_runtime_resource_policy()
        if policy is not None and (
            policy.reserved_memory_bytes + selected_bytes
            > policy.task_memory_budget_bytes
        ):
            raise ValueError("prediction_working_set_exceeds_task_memory_budget")
        with ExitStack() as reservation:
            if policy is not None:
                reservation.enter_context(
                    policy.reserve("research.prediction.projection", selected_bytes)
                )
            yield frame.loc[:, selected]


def validate_prediction_outputs(frame: pd.DataFrame, spec: PredictionSpec) -> None:
    expected = (*spec.primary_key, spec.prediction_column)
    observed = tuple(frame.columns)
    if any(type(column) is not str or not column for column in observed):
        raise ValueError("invalid_prediction_output_columns")
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate_prediction_output_columns")
    missing = [column for column in expected if column not in observed]
    if missing:
        raise ValueError("missing_prediction_output_columns: " + ", ".join(missing))
    extra = sorted(set(observed).difference(expected))
    if extra:
        raise ValueError("unregistered_prediction_output_columns: " + ", ".join(extra))
    if observed != expected:
        raise ValueError("misordered_prediction_output_columns")
    if frame.empty:
        raise ValueError("empty_prediction_window")
    _validate_prediction_output_values(
        frame,
        primary_key=spec.primary_key,
        prediction_column=spec.prediction_column,
    )
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_prediction_timestamp_values"
    )
    if bool((timestamps < pd.Timestamp(spec.test_start)).any()):
        raise ValueError("prediction_output_before_test_start")
    if spec.test_end is not None and bool(
        (timestamps > pd.Timestamp(spec.test_end)).any()
    ):
        raise ValueError("prediction_output_after_test_end")


def _validate_prediction_output_values(
    frame: pd.DataFrame,
    *,
    primary_key: tuple[str, ...],
    prediction_column: str,
) -> None:
    if bool(frame.loc[:, list(primary_key)].isna().any(axis=None)):
        raise ValueError("invalid_prediction_keys")
    if bool(frame.duplicated(list(primary_key)).any()):
        raise ValueError("duplicate_prediction_keys")
    strict_finite_numeric_series(
        frame[prediction_column],
        error="invalid_prediction_values",
    )


def _strict_non_negative_prediction_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid_prediction_metadata_count")
    return value


__all__ = [
    "PredictionResult",
    "build_prediction_result_from_frames",
    "validate_prediction_outputs",
]
