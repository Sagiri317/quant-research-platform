"""Spec-driven prediction runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pandas as pd

import quant_research_platform.research.models.fitting as fitting
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    iter_validated_research_input_chunks,
)
from quant_research_platform.research.models.fitting import (
    model_training_read_columns,
    training_scope_chunks,
)
from quant_research_platform.research.models.persisted import (
    PersistedQuarterlyModelArtifact,
    read_quarterly_predictor,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.models.trial_selection import (
    select_prediction_model_spec,
)
from quant_research_platform.research.predictions.outputs import (
    PredictionResult,
    build_prediction_result_from_frames,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
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


def build_prediction_result_from_persisted_model(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    prediction_spec,
    persisted_model: PersistedQuarterlyModelArtifact,
    resource_policy: RuntimeResourcePolicy,
) -> PredictionResult:
    """Build predictions from the persisted quarterly model members."""

    dataset_spec = dataset_catalog.get(prediction_spec.research_dataset_id)
    requested_start = pd.Timestamp(prediction_spec.test_start)
    requested_end = pd.Timestamp(prediction_spec.test_end or prediction_spec.test_start)
    selected = tuple(
        item
        for item in persisted_model.quarterly_models
        if pd.Timestamp(item["prediction_window"]["end"]) >= requested_start
        and pd.Timestamp(item["prediction_window"]["start"]) <= requested_end
    )
    if not selected:
        raise ValueError("persisted_model_prediction_window_empty")
    if (
        str(pd.Period(requested_start, freq="Q")) != selected[0]["quarter_id"]
        or str(pd.Period(requested_end, freq="Q")) != selected[-1]["quarter_id"]
    ):
        raise ValueError("persisted_model_prediction_window_uncovered")
    selected_periods = tuple(
        dict.fromkeys(
            pd.Period(item["prediction_window"]["start"], freq="Q") for item in selected
        )
    )
    if any(
        current != previous + 1
        for previous, current in zip(
            selected_periods,
            selected_periods[1:],
            strict=False,
        )
    ):
        raise ValueError("persisted_model_prediction_window_uncovered")
    selected_input_columns = tuple(
        dict.fromkeys(
            column
            for item in selected
            for column in item["model_metadata"]["input_columns"]
        )
    )
    state = _PredictionMetadataState(selected_input_columns)

    def output_chunks() -> Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]]:
        with runtime_resource_policy_scope(resource_policy) as policy:
            for item in selected:
                metadata = dict(item["model_metadata"])
                q_start_value = max(
                    pd.Timestamp(item["prediction_window"]["start"]),
                    requested_start,
                )
                q_end_value = min(
                    pd.Timestamp(item["prediction_window"]["end"]),
                    requested_end,
                )
                if q_start_value > q_end_value:
                    continue
                q_start = q_start_value.date().isoformat()
                q_end = q_end_value.date().isoformat()
                q_model_id = item["model_id"]
                q_input_columns = tuple(metadata["input_columns"])
                if not q_input_columns:
                    raise ValueError("persisted_model_input_columns_missing")
                read_columns = tuple(
                    dict.fromkeys(
                        (
                            *dataset_spec.primary_key,
                            *q_input_columns,
                            *required_filter_columns(
                                prediction_spec.sample_filter_mode
                            ),
                        )
                    )
                )

                def chunks(
                    *,
                    _q_start=q_start,
                    _q_end=q_end,
                    _read_columns=read_columns,
                    _q_input_columns=q_input_columns,
                    _family=item["family_id"],
                    _quarter=item["quarter_id"],
                ) -> Iterable[pd.DataFrame]:
                    if {"family_id", "quarter_id"}.intersection(_read_columns):
                        raise ValueError("prediction_input_coordinate_collision")
                    for frame in iter_validated_research_input_chunks(
                        iter_research_dataset_chunks(
                            layout,
                            market_id=dataset_spec.market_id,
                            dataset_id=dataset_spec.data_id,
                            time_window=TimeWindow(_q_start, _q_end),
                            columns=_read_columns,
                        ),
                        dataset_spec=dataset_spec,
                        input_columns=_q_input_columns,
                    ):
                        with policy.lease(
                            "research.prediction.family_coordinates",
                            max(1, len(frame) * 512 + 65536),
                        ):
                            frame["family_id"] = _family
                            frame["quarter_id"] = _quarter
                            try:
                                yield frame
                            finally:
                                del frame["family_id"]
                                del frame["quarter_id"]

                q_spec = replace(
                    prediction_spec,
                    prediction_id=f"{prediction_spec.prediction_id}_{item['quarter_id'].lower()}",
                    model_id=q_model_id,
                    train_end=None,
                    test_start=q_start,
                    test_end=q_end,
                    candidate_model_ids=(),
                    model_selection_validation_start=None,
                    model_selection_validation_end=None,
                )
                q_metadata = dict(metadata)
                q_metadata.pop("fit_cutoff", None)
                q_metadata.pop("rank_parameters", None)
                with read_quarterly_predictor(
                    persisted_model,
                    item["quarter_id"],
                    item["family_id"],
                    resource_policy=policy,
                ) as predictor:
                    q_result = build_prediction_result_from_frames(
                        chunks(),
                        q_spec,
                        np.empty(0, dtype="float64"),
                        q_input_columns,
                        prediction_model=predictor,
                        model_metadata=q_metadata,
                        resource_policy=policy,
                    )
                    try:
                        yield from q_result.output_chunks
                        state.add(q_result.prediction_metadata_factory())
                    finally:
                        q_result.close()

    source = output_chunks()
    return PredictionResult(
        spec=prediction_spec,
        output_chunks=source,
        model_metadata=persisted_model.prediction_lineage(
            TimeWindow(requested_start, requested_end)
        ),
        prediction_metadata_factory=lambda: state.metadata(prediction_spec),
        _close=source.close,
    )


class _PredictionMetadataState:
    def __init__(self, input_columns: tuple[str, ...]) -> None:
        self.input_columns = input_columns
        self._metadata: list[dict[str, object]] = []

    def add(self, metadata: dict[str, object]) -> None:
        self._metadata.append(metadata)

    def metadata(self, spec) -> dict[str, object]:
        if len(self._metadata) == 0:
            raise ValueError("empty_prediction_window")
        first = self._metadata[0]
        effective_end = max(
            (pd.Timestamp(item["effective_test_end"]) for item in self._metadata),
            default=None,
        )
        if effective_end is None:
            raise ValueError("empty_prediction_window")
        return {
            "prediction_mode": spec.prediction_mode,
            "sample_filter_mode": spec.sample_filter_mode,
            "filter_columns_used": first["filter_columns_used"],
            "filter_definition": first["filter_definition"],
            "rows_input": sum(int(item["rows_input"]) for item in self._metadata),
            "rows_after_filter_mode": sum(
                int(item["rows_after_filter_mode"]) for item in self._metadata
            ),
            "rows_after_input_dropna": sum(
                int(item["rows_after_input_dropna"]) for item in self._metadata
            ),
            "prediction_rows": sum(
                int(item["prediction_rows"]) for item in self._metadata
            ),
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "requested_test_end": spec.test_end,
            "effective_test_end": effective_end.date().isoformat()
            if spec.prediction_mode == "a_share_oos"
            else effective_end.isoformat(),
            "is_oos": spec.is_oos,
            "is_final_oos_window": spec.is_final_oos_window,
            "input_columns": list(self.input_columns),
            "input_count": len(self.input_columns),
            "prediction_column": spec.prediction_column,
        }


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
        yield from iter_validated_research_input_chunks(
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
        model_spec, model_selection_metadata = select_prediction_model_spec(
            prediction_spec=prediction_spec,
            dataset_spec=dataset_spec,
            model_catalog=model_catalog,
            chunks=_chunks,
        )
    train_read_columns = model_training_read_columns(
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
            training_scope_chunks(
                _chunks(train_read_columns, input_columns=model_spec.input_columns),
                asset_ids=prediction_spec.training_asset_ids,
            ),
            model_spec,
            estimator=build_estimator_from_model_spec(model_spec),
            fitting_cutoff=train_end,
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


def _validate_prediction_window_order(spec) -> None:
    train_end = pd.Timestamp(spec.train_end)
    test_start = pd.Timestamp(spec.test_start)
    if train_end >= test_start:
        raise ValueError("prediction_train_end_must_precede_test_start")
    test_end = pd.Timestamp(spec.test_end) if spec.test_end is not None else None
    if test_end is not None and test_end < test_start:
        raise ValueError("prediction_test_end_must_not_precede_test_start")


def build_crypto_predictions(
    *, layout, dataset_catalog, persisted_model, time_window, resource_policy
):
    from quant_research_platform.contracts.predictions import PredictionSpec
    from quant_research_platform.research.models.persisted import (
        read_crypto_model_member,
    )

    dataset_id = persisted_model.envelope["semantic_metadata"]["target_contract"][
        "protocol"
    ]["scenario_id"]
    dataset = dataset_catalog.get(dataset_id)
    model_ids = tuple(
        dict.fromkeys(entry["model_id"] for entry in persisted_model.models)
    )
    for model_id in model_ids:
        entries = tuple(
            entry for entry in persisted_model.models if entry["model_id"] == model_id
        )
        inputs = tuple(entries[0]["model_metadata"]["input_columns"])
        spec = PredictionSpec(
            prediction_id=f"{model_id}_prediction",
            model_id=model_id,
            research_dataset_id=dataset_id,
            prediction_mode="crypto_oos",
            train_end=None,
            test_start=pd.Timestamp(time_window.start).isoformat(),
            test_end=pd.Timestamp(time_window.end).isoformat(),
            training_asset_ids=tuple(entries[0]["training_asset_ids"]),
            evaluation_asset_ids=tuple(entries[0]["evaluation_asset_ids"]),
        )
        state = _PredictionMetadataState(inputs)

        def output_chunks(entries=entries, spec=spec, state=state, inputs=inputs):
            for entry in entries:
                window = TimeWindow(
                    max(
                        pd.Timestamp(entry["prediction_window"]["start"]),
                        pd.Timestamp(time_window.start),
                    ),
                    min(
                        pd.Timestamp(entry["prediction_window"]["end"]),
                        pd.Timestamp(time_window.end),
                    ),
                )
                columns = tuple(
                    dict.fromkeys(
                        (
                            *dataset.primary_key,
                            *inputs,
                            *required_filter_columns(spec.sample_filter_mode),
                        )
                    )
                )
                source = iter_validated_research_input_chunks(
                    iter_research_dataset_chunks(
                        layout,
                        market_id=dataset.market_id,
                        dataset_id=dataset_id,
                        time_window=window,
                        columns=columns,
                    ),
                    dataset_spec=dataset,
                    input_columns=inputs,
                )
                folded = replace(
                    spec,
                    prediction_mode="minimal_oos",
                    train_end=entry["validation_window"]["end"],
                    test_start=pd.Timestamp(window.start).isoformat(),
                    test_end=pd.Timestamp(window.end).isoformat(),
                )
                with read_crypto_model_member(
                    persisted_model, entry, resource_policy=resource_policy
                ) as payload:
                    result = None
                    try:
                        result = build_prediction_result_from_frames(
                            source,
                            folded,
                            payload[0],
                            inputs,
                            prediction_model=payload[1],
                            model_metadata=dict(entry["model_metadata"]),
                            resource_policy=resource_policy,
                        )
                        with result:
                            yield from result.output_chunks
                            state.add(result.prediction_metadata_factory())
                    finally:
                        result = payload = None
            if len(state._metadata) != len(entries):
                raise ValueError("crypto_prediction_fold_coverage_invalid")

        source = output_chunks()
        yield PredictionResult(
            spec,
            source,
            persisted_model.prediction_lineage(model_id),
            lambda state=state, spec=spec: state.metadata(spec),
            source.close,
        )
