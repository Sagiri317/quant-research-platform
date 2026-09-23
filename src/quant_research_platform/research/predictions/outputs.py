"""Prediction output builders."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import (
    reject_filter_input_columns,
    required_filter_columns,
)
from quant_research_platform.contracts.models import (
    _model_algorithm_facts,
    _resolved_model_hyperparameters_match,
)
from quant_research_platform.contracts.predictions import (
    ALPHA_SCORE_COLUMN,
    RAW_PREDICTION_COLUMN,
    PredictionSpec,
    _validate_prediction_output_values,
    prediction_output_columns,
)
from quant_research_platform.contracts.time import (
    is_datetime_value,
    time_value_isoformat,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    canonical_feature_validity_columns,
)
from quant_research_platform.research.filtering import (
    _filter_mode_scope,
)
from quant_research_platform.research.models.fitting import prediction_callable
from quant_research_platform.runtime.dataframes import (
    _selected_frame_payload_bytes,
    iter_complete_time_sized_frames,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)


@dataclass(frozen=True)
class PredictionResult:
    spec: PredictionSpec
    output_chunks: Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]]
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
                f"prediction result cleanup also failed: {type(cleanup_error).__name__}"
            )


def build_prediction_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: PredictionSpec,
    input_columns: tuple[str, ...],
    *,
    prediction_model: object,
    model_metadata: dict[str, object],
    model_lease: MemoryLease | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> PredictionResult:
    bound_policy = resource_policy or current_runtime_resource_policy()
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
        facts = _model_algorithm_facts(model_algorithm)
    except ValueError as exc:
        raise ValueError("invalid_model_metadata") from exc
    expected_package = facts.dependency_package or "internal"
    if (
        not _resolved_model_hyperparameters_match(
            model_algorithm,
            hyperparameters,
            resolved_hyperparameters,
        )
        or dependency_status.get("package") != expected_package
    ):
        raise ValueError("invalid_model_metadata")
    rank_inputs_allow_missing = facts.input_kind == "rank_ensemble_inputs"
    reject_filter_input_columns(input_columns)
    coordinates = ("quarter_id",) if spec.prediction_mode == "a_share_oos" else ()
    validity_columns = canonical_feature_validity_columns(input_columns)
    required = set((*spec.primary_key, *input_columns, *validity_columns, *coordinates))
    work_columns = tuple(
        dict.fromkeys(
            (
                *spec.primary_key,
                *coordinates,
                *input_columns,
                *validity_columns,
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
        input_columns=input_columns,
        predictor=prediction_callable(
            input_columns,
            prediction_model,
        ),
        build_state=build_state,
        rank_inputs_allow_missing=rank_inputs_allow_missing,
    )
    scoped_output_frames = _scoped_prediction_output_frames(
        output_frames,
        bound_policy=bound_policy,
    )

    def close_result() -> None:
        primary: BaseException | None = None
        try:
            scoped_output_frames.close()
        except BaseException as exc:
            primary = exc
        if model_lease is not None:
            try:
                model_lease.close()
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "fitted model lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if primary is not None:
            raise primary

    return PredictionResult(
        spec=spec,
        output_chunks=scoped_output_frames,
        model_metadata={
            **model_metadata,
            "test_start": spec.test_start,
            "test_end": spec.test_end,
            "oos_start": spec.test_start,
            "oos_end": None,
        },
        prediction_metadata_factory=lambda: build_state.prediction_metadata(
            spec=spec,
            input_columns=input_columns,
        ),
        _close=close_result,
    )


def _scoped_prediction_output_frames(
    frames: Iterable[pd.DataFrame],
    *,
    bound_policy: RuntimeResourcePolicy | None,
) -> Iterator[pd.DataFrame]:
    with runtime_resource_policy_scope(bound_policy):
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

    def record_prediction_output(
        self,
        output: pd.DataFrame,
        *,
        spec: PredictionSpec,
    ) -> None:
        if output.empty:
            return
        timestamps = strict_timestamp_series(
            output["timestamp"], error="invalid_prediction_timestamp_values"
        )
        current_end = time_value_isoformat(
            pd.Timestamp(timestamps.max()),
            date_only=not is_datetime_value(spec.test_start),
        )
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
    input_columns: tuple[str, ...],
    predictor: Callable[[pd.DataFrame], np.ndarray],
    build_state: _PredictionBuildState,
    rank_inputs_allow_missing: bool = False,
) -> Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]]:
    usable_columns = tuple(
        dict.fromkeys(
            (
                *spec.primary_key,
                *input_columns,
                *(("quarter_id",) if spec.prediction_mode == "a_share_oos" else ()),
            )
        )
    )
    with closing_iterator(iter(frames)) as active_frames:
        for chunk in active_frames:
            with _filter_mode_scope(
                chunk,
                mode=spec.sample_filter_mode,
                metadata_key="sample_filter_mode",
            ) as (filtered, metadata):
                build_state.record_filter_metadata(metadata)
                policy = current_runtime_resource_policy()
                row_count = len(filtered)
                index_bytes = int(filtered.index.memory_usage(deep=True))
                mask_bytes = index_bytes + row_count * (len(input_columns) + 2)
                test_frame_bytes = _selected_frame_payload_bytes(
                    filtered,
                    usable_columns,
                )
                output_bytes = _selected_frame_payload_bytes(
                    filtered,
                    spec.primary_key,
                ) + (8 * row_count)
                if spec.prediction_mode == "a_share_oos":
                    output_bytes += row_count * (64 + 16)
                model_matrix_bytes = 8 * row_count * (len(input_columns) + 1)
                working_bytes = (
                    mask_bytes + test_frame_bytes + output_bytes + model_matrix_bytes
                )
                if (
                    policy is not None
                    and policy.explicit_memory_cap_bytes is not None
                    and policy.used_memory_bytes + working_bytes
                    > policy.explicit_memory_cap_bytes
                ):
                    raise ValueError(
                        "prediction_working_set_exceeds_task_memory_budget"
                    )
                working_lease = None
                if policy is not None:
                    working_lease = policy.lease(
                        "research.prediction.output",
                        working_bytes,
                    )
                usable_mask = None
                test_frame = None
                output = None
                owned_output: OwnedValue[pd.DataFrame] | None = None
                try:
                    validity_inputs = tuple(
                        dict.fromkeys(
                            (
                                *canonical_feature_validity_columns(input_columns),
                                *(
                                    input_columns
                                    if not rank_inputs_allow_missing
                                    else ()
                                ),
                            )
                        )
                    )
                    usable_mask = (
                        filtered.loc[:, list(validity_inputs)].notna().all(axis=1)
                    )
                    build_state.record_input_dropna_rows(int(usable_mask.sum()))
                    usable_mask &= filtered["timestamp"] >= pd.Timestamp(
                        spec.test_start
                    )
                    if spec.evaluation_asset_ids:
                        usable_mask &= filtered["asset_id"].isin(
                            spec.evaluation_asset_ids
                        )
                    if spec.test_end is not None:
                        usable_mask &= filtered["timestamp"] <= pd.Timestamp(
                            spec.test_end
                        )
                    test_frame = filtered.loc[usable_mask, list(usable_columns)]
                    if test_frame.empty:
                        test_frame = None
                        usable_mask = None
                        filtered = chunk = None
                        if working_lease is not None:
                            working_lease.close()
                        continue
                    output = test_frame.loc[:, list(spec.primary_key)].copy()
                    raw_prediction = predictor(test_frame)
                    if spec.prediction_mode == "a_share_oos":
                        output["quarter_id"] = test_frame["quarter_id"]
                        output["model_id"] = spec.model_id
                        output[RAW_PREDICTION_COLUMN] = raw_prediction
                        timestamps = strict_timestamp_series(
                            output["timestamp"],
                            error="invalid_prediction_timestamp_values",
                        )
                        ranks = (
                            pd.Series(
                                raw_prediction,
                                index=output.index,
                                dtype="float64",
                            )
                            .groupby([timestamps, output["family_id"]], sort=False)
                            .rank(method="average")
                        )
                        counts = timestamps.groupby(
                            [timestamps, output["family_id"]], sort=False
                        ).transform("size")
                        output[ALPHA_SCORE_COLUMN] = (
                            ranks - (counts + 1.0) / 2.0
                        ) / counts
                    else:
                        output[spec.prediction_column] = raw_prediction
                    if tuple(output.columns) != prediction_output_columns(
                        spec.prediction_mode,
                        spec.prediction_column,
                        spec.primary_key,
                    ):
                        raise ValueError("prediction_output_columns_invalid")
                    _validate_prediction_output_values(
                        output,
                        primary_key=spec.primary_key,
                        prediction_column=(
                            RAW_PREDICTION_COLUMN
                            if spec.prediction_mode == "a_share_oos"
                            else spec.prediction_column
                        ),
                    )
                    if spec.prediction_mode == "a_share_oos":
                        if any(
                            type(value) is not str or not value
                            for value in output["model_id"]
                        ):
                            raise ValueError("prediction_model_id_invalid")
                        if not np.isfinite(
                            output[ALPHA_SCORE_COLUMN].to_numpy(dtype="float64")
                        ).all():
                            raise ValueError("invalid_prediction_alpha_scores")
                    build_state.record_prediction_output(output, spec=spec)
                    raw_prediction = timestamps = ranks = counts = None
                    test_frame = usable_mask = None
                    retained_bytes = pandas_frame_payload_bytes(output)
                    if working_lease is not None:
                        working_lease.shrink_to(retained_bytes)
                        working_lease.transfer("research.prediction.result")
                        owned_output = OwnedValue(output, working_lease)
                        working_lease = None
                    test_frame = None
                    usable_mask = None
                    try:
                        yield owned_output if owned_output is not None else output
                    finally:
                        output = None
                        filtered = chunk = None
                        if owned_output is not None and owned_output.lease.active:
                            owned_output.release(owner="research.prediction.result")
                except BaseException as primary:
                    output = None
                    test_frame = None
                    usable_mask = None
                    filtered = chunk = None
                    if owned_output is not None and owned_output.lease.active:
                        try:
                            owned_output.release(owner="research.prediction.result")
                        except BaseException as cleanup_error:
                            primary.add_note(
                                "prediction output lease release also failed: "
                                f"{type(cleanup_error).__name__}"
                            )
                    if working_lease is not None and working_lease.active:
                        try:
                            working_lease.close()
                        except BaseException as cleanup_error:
                            primary.add_note(
                                "prediction output lease release also failed: "
                                f"{type(cleanup_error).__name__}"
                            )
                    raise
                finally:
                    raw_prediction = timestamps = ranks = counts = None
                    chunk = None


def _iter_prediction_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    required: set[str],
    work_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    complete_frames = iter_complete_time_sized_frames(
        iter(frames),
        timestamp_column="timestamp",
        resource_policy=current_runtime_resource_policy(),
    )
    with closing_iterator(complete_frames) as active_frames:
        for frame in active_frames:
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"missing_prediction_columns: {', '.join(missing)}")
            selected = [column for column in work_columns if column in frame.columns]
            selected_bytes = _selected_frame_payload_bytes(frame, selected)
            policy = current_runtime_resource_policy()
            if (
                policy is not None
                and policy.explicit_memory_cap_bytes is not None
                and policy.used_memory_bytes + selected_bytes
                > policy.explicit_memory_cap_bytes
            ):
                raise ValueError("prediction_working_set_exceeds_task_memory_budget")
            projection_lease = None
            if policy is not None:
                projection_lease = policy.lease(
                    "research.prediction.projection",
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
                            "prediction projection lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                raise
            else:
                if projection_lease is not None:
                    projection_lease.close()
            finally:
                frame = None


def _strict_non_negative_prediction_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid_prediction_metadata_count")
    return value
