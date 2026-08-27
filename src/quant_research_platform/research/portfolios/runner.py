"""Spec-driven portfolio runner from ResearchDataset inputs."""

from __future__ import annotations

import math
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.compute.dataframe.performance import (
    ADDITIVE_CUMULATIVE_RETURN_COLUMNS,
    _new_performance_stats,
    _ordered_cumulative,
    _performance_summary_frame,
    _performance_summary_record,
    _update_performance_stats_batch,
    _validate_returns_frame,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    EXPOSURE_SUMMARY_COLUMNS,
    RETURNS_COLUMNS,
)
from quant_research_platform.compute.dataframe.position_math import validate_positions
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research._memory import (
    _selected_frame_payload_bytes,
    iter_complete_session_frames,
)
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.portfolios.engine import (
    _construct_portfolio_positions,
    _PortfolioTemporalState,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout

_PORTFOLIO_ROW_WORKING_BYTES = 2048
_PORTFOLIO_PERIOD_WORKING_BYTES = 4096
_PORTFOLIO_STATS_BATCH_ROWS = 4096
_PORTFOLIO_STATE_FIXED_WORKING_BYTES = 64 * 1024
_PORTFOLIO_STATE_PER_SIGNAL_BYTES = 16 * 1024
_PORTFOLIO_SQLITE_PAGE_BYTES = 4096
_PORTFOLIO_SQLITE_ROW_DISK_BYTES = 1024
_PORTFOLIO_SQLITE_STARTUP_BYTES = 64 * _PORTFOLIO_SQLITE_PAGE_BYTES


@dataclass(frozen=True)
class PortfolioResult:
    spec: PortfolioSpec
    output_chunks: Iterable[Mapping[str, pd.DataFrame]]
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]]
    additive_cumulative_returns_factory: Callable[
        [], Iterable[pd.DataFrame]
    ] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _close: Callable[[], None] = field(
        default=lambda: None,
        repr=False,
        compare=False,
    )

    def close(self) -> None:
        self._close()

    def __enter__(self) -> PortfolioResult:
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
                "portfolio result cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )


@dataclass
class _PortfolioSummaryState:
    position_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    resource_policy: RuntimeResourcePolicy | None = None
    returns_complete: bool = False
    state_lease: MemoryLease | None = field(default=None, repr=False)
    summary_lease: MemoryLease | None = field(default=None, repr=False)
    return_database: sqlite3.Connection | None = field(
        default=None,
        repr=False,
    )
    return_scratch: tempfile.TemporaryDirectory[str] | None = field(
        default=None,
        repr=False,
    )


def build_portfolio_result_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    portfolio_catalog: PortfolioCatalog,
    portfolio_id: str,
    time_window: TimeWindow | None = None,
    prediction_frame: pd.DataFrame | None = None,
    prediction_metadata: Mapping[str, object] | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> PortfolioResult:
    operation = ExitStack()
    policy = operation.enter_context(runtime_resource_policy_scope(resource_policy))
    try:
        result = _build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id=portfolio_id,
            time_window=time_window,
            prediction_frame=prediction_frame,
            prediction_metadata=prediction_metadata,
            resource_policy=policy,
        )
    except BaseException as primary:
        try:
            operation.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "portfolio operation cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    operation.callback(result.close)
    source_chunks = operation.enter_context(
        closing_iterator(iter(result.output_chunks))
    )
    chunks_complete = False
    summary_value: Mapping[str, pd.DataFrame] | None = None
    terminal_error: BaseException | None = None
    operation_open = True

    def close_operation() -> None:
        nonlocal operation_open
        if operation_open:
            operation_open = False
            operation.close()

    def close_result() -> None:
        nonlocal terminal_error
        if not operation_open:
            return
        closed = RuntimeError("portfolio_result_closed")
        try:
            close_operation()
        except BaseException as primary:
            terminal_error = primary
            raise
        if terminal_error is None:
            terminal_error = closed

    def output_chunks() -> Iterable[Mapping[str, pd.DataFrame]]:
        nonlocal chunks_complete, terminal_error
        if terminal_error is not None:
            raise terminal_error
        try:
            yield from source_chunks
        except BaseException as primary:
            terminal_error = primary
            try:
                source_chunks.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "portfolio output cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            if operation_open and policy.used_memory_bytes == 0:
                try:
                    close_operation()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "portfolio operation cleanup also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise
        chunks_complete = True

    def summary() -> Mapping[str, pd.DataFrame]:
        nonlocal summary_value, terminal_error
        if terminal_error is not None:
            raise terminal_error
        if not chunks_complete:
            raise RuntimeError("portfolio_summary_before_output_complete")
        if summary_value is not None:
            return summary_value
        try:
            candidate = result.summary_factory()
        except BaseException as primary:
            terminal_error = primary
            try:
                close_operation()
            except BaseException as cleanup_error:
                primary.add_note(
                    "portfolio operation cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        summary_value = candidate
        return summary_value

    owned_result = PortfolioResult(
        spec=result.spec,
        output_chunks=output_chunks(),
        summary_factory=summary,
        additive_cumulative_returns_factory=result.additive_cumulative_returns_factory,
        _close=close_result,
    )
    return owned_result


def _build_portfolio_result_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    portfolio_catalog: PortfolioCatalog,
    portfolio_id: str,
    time_window: TimeWindow | None,
    prediction_frame: pd.DataFrame | None,
    prediction_metadata: Mapping[str, object] | None,
    resource_policy: RuntimeResourcePolicy,
) -> PortfolioResult:
    spec = portfolio_catalog.get(portfolio_id)
    dataset_spec = dataset_catalog.get(spec.research_dataset_id)
    prediction_signals = tuple(
        signal for signal in spec.signals if signal.source_kind == "prediction"
    )
    prediction_input_columns: tuple[str, ...] = ()
    prediction_filter_columns: tuple[str, ...] = ()
    prediction_input = None
    if prediction_signals:
        if len(prediction_signals) != 1:
            raise ValueError("multiple_prediction_signals_unsupported")
        if prediction_frame is None or prediction_metadata is None:
            raise ValueError("persisted_prediction_input_required")
        prediction_input = _validated_prediction_input(
            prediction_frame,
            prediction_metadata,
            signal=prediction_signals[0],
            research_dataset_id=dataset_spec.data_id,
            time_window=time_window,
        )
        prediction_input_columns = prediction_input[2]
        prediction_filter_columns = required_filter_columns(prediction_input[3])
    elif prediction_frame is not None or prediction_metadata is not None:
        raise ValueError("unexpected_persisted_prediction_input")
    read_columns = _portfolio_read_columns(
        spec,
        prediction_input_columns=prediction_input_columns,
        prediction_filter_columns=prediction_filter_columns,
    )
    source_columns_by_kind = {
        "factor": tuple(dataset_spec.factor_columns),
        "feature": tuple(dataset_spec.feature_columns),
        "prediction": tuple(signal.source_column for signal in prediction_signals),
    }
    summary_state = _PortfolioSummaryState(
        resource_policy=resource_policy
    )
    try:
        summary_state.state_lease = resource_policy.lease(
            "research.portfolio.state",
            _PORTFOLIO_STATE_FIXED_WORKING_BYTES
            + max(1, len(spec.signals)) * _PORTFOLIO_STATE_PER_SIGNAL_BYTES,
        )
        input_chunks = _portfolio_input_chunks(
            layout=layout,
            market_id=dataset_spec.market_id,
            dataset_id=dataset_spec.data_id,
            time_window=time_window,
            columns=read_columns,
        )
        if prediction_input is not None:
            input_chunks = _iter_prediction_joined_chunks(
                input_chunks,
                prediction_frame=prediction_input[0],
                primary_key=prediction_input[1],
                input_columns=prediction_input[2],
                sample_filter_mode=prediction_input[3],
                test_start=prediction_input[4],
                test_end=prediction_input[5],
                resource_policy=summary_state.resource_policy,
            )
        output_chunks = _observe_portfolio_summary_chunks(
            _iter_portfolio_output_chunks(
                input_chunks,
                spec,
                source_columns_by_kind=source_columns_by_kind,
                resource_policy=summary_state.resource_policy,
            ),
            summary_state,
        )

        def additive_cumulative_returns_factory() -> Iterable[pd.DataFrame]:
            if not summary_state.returns_complete:
                raise RuntimeError(
                    "portfolio_additive_cumulative_returns_before_output_complete"
                )
            return _iter_portfolio_additive_cumulative_chunks(summary_state)

        return PortfolioResult(
            spec=spec,
            output_chunks=output_chunks,
            summary_factory=lambda: _portfolio_summary_outputs(summary_state),
            additive_cumulative_returns_factory=additive_cumulative_returns_factory,
            _close=lambda: _close_portfolio_summary_state(summary_state),
        )
    except BaseException:
        _close_portfolio_summary_state(summary_state)
        raise


def _portfolio_read_columns(
    spec: PortfolioSpec,
    *,
    prediction_input_columns: tuple[str, ...] = (),
    prediction_filter_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    optional_columns = (
        spec.weight_config.risk_column,
        spec.tradability_config.eligibility_column,
        spec.tradability_config.can_enter_column,
        spec.tradability_config.can_exit_column,
        spec.diagnostics_config.capacity_amount_column,
        spec.diagnostics_config.capacity_float_market_cap_column,
        *spec.diagnostics_config.numeric_exposure_columns,
        *spec.diagnostics_config.categorical_exposure_columns,
    )
    return tuple(
        dict.fromkeys(
            (
                "market_id",
                "asset_id",
                "timestamp",
                *(
                    signal.source_column
                    for signal in spec.signals
                    if signal.source_kind != "prediction"
                ),
                spec.return_source_column,
                *prediction_input_columns,
                *prediction_filter_columns,
                *(column for column in optional_columns if column is not None),
            )
        )
    )


def _validated_prediction_input(
    frame: object,
    metadata: Mapping[str, object],
    *,
    signal,
    research_dataset_id: str,
    time_window: TimeWindow | None,
) -> tuple[
    object,
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    str | None,
]:
    model_metadata = metadata.get("model_metadata")
    primary_key_value = metadata.get("primary_key")
    if (
        metadata.get("prediction_id") != signal.signal_id
        or metadata.get("research_dataset_id") != research_dataset_id
        or metadata.get("prediction_column") != signal.source_column
        or type(model_metadata) is not dict
        or type(primary_key_value) is not list
        or not primary_key_value
        or any(type(column) is not str or not column for column in primary_key_value)
    ):
        raise ValueError("persisted_prediction_metadata_mismatch")
    input_columns_value = model_metadata.get("input_columns")
    sample_filter_mode = metadata.get("sample_filter_mode")
    test_start = metadata.get("test_start")
    test_end = metadata.get("test_end")
    if (
        type(input_columns_value) is not list
        or not input_columns_value
        or any(type(column) is not str or not column for column in input_columns_value)
        or type(sample_filter_mode) is not str
        or not sample_filter_mode
        or type(test_start) is not str
        or not test_start
        or (test_end is not None and (type(test_end) is not str or not test_end))
    ):
        raise ValueError("persisted_prediction_metadata_mismatch")
    primary_key = tuple(primary_key_value)
    input_columns = tuple(input_columns_value)
    expected_columns = (*primary_key, signal.source_column)
    if not isinstance(frame, pd.DataFrame):
        if (
            getattr(frame, "primary_key", None) != primary_key
            or getattr(frame, "prediction_column", None) != signal.source_column
        ):
            raise ValueError("persisted_prediction_columns_mismatch")
        return (
            frame,
            primary_key,
            input_columns,
            sample_filter_mode,
            test_start,
            test_end,
        )
    if tuple(frame.columns) != expected_columns:
        raise ValueError("persisted_prediction_columns_mismatch")
    policy = current_runtime_resource_policy()
    row_count = len(frame)
    index_bytes = int(frame.index.memory_usage(deep=True))
    frame_bytes = pandas_frame_payload_bytes(frame)
    key_projection_bytes = _selected_frame_payload_bytes(frame, list(primary_key))
    key_validation_bytes = index_bytes + row_count * (32 + 16 * len(primary_key))
    timestamp_bytes = index_bytes + 8 * row_count
    value_bytes = index_bytes + 16 * row_count
    normalized_output_bytes = frame_bytes
    window_bytes = 0 if time_window is None else index_bytes + row_count + frame_bytes
    validation_bytes = (
        key_projection_bytes
        + key_validation_bytes
        + timestamp_bytes
        + value_bytes
        + normalized_output_bytes
        + window_bytes
    )
    with ExitStack() as validation_reservation:
        if policy is not None:
            validation_reservation.enter_context(
                policy.lease(
                    "research.portfolio.prediction.validate",
                    validation_bytes,
                )
            )
        if frame.loc[:, list(primary_key)].isna().any(axis=None):
            raise ValueError("persisted_prediction_key_null")
        timestamps = strict_timestamp_series(
            frame["timestamp"], error="invalid_persisted_prediction_timestamp"
        )
        if frame.duplicated(list(primary_key)).any():
            raise ValueError("duplicate_persisted_prediction_keys")
        if bool((timestamps < pd.Timestamp(test_start)).any()) or (
            test_end is not None and bool((timestamps > pd.Timestamp(test_end)).any())
        ):
            raise ValueError("persisted_prediction_window_mismatch")
        if not pd.api.types.is_numeric_dtype(frame[signal.source_column]):
            raise ValueError("persisted_prediction_values_invalid")
        values = strict_finite_numeric_series(
            frame[signal.source_column],
            error="persisted_prediction_values_invalid",
        )
        result = frame
        if not timestamps.equals(frame["timestamp"]) or not values.equals(
            frame[signal.source_column]
        ):
            result = frame.copy()
            result["timestamp"] = timestamps
            result[signal.source_column] = values
        if time_window is not None:
            in_window = timestamps.between(
                pd.Timestamp(time_window.start),
                pd.Timestamp(time_window.end),
            )
            if not bool(in_window.all()):
                result = result.loc[in_window].reset_index(drop=True)
    return (
        result,
        primary_key,
        input_columns,
        sample_filter_mode,
        test_start,
        test_end,
    )


def _iter_prediction_joined_chunks(
    chunks,
    *,
    prediction_frame: pd.DataFrame,
    primary_key: tuple[str, ...],
    input_columns: tuple[str, ...],
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
    resource_policy: RuntimeResourcePolicy,
):
    if not isinstance(prediction_frame, pd.DataFrame):
        yield from _iter_prediction_lookup_joined_chunks(
            chunks,
            prediction_lookup=prediction_frame,
            primary_key=primary_key,
            input_columns=input_columns,
            sample_filter_mode=sample_filter_mode,
            test_start=test_start,
            test_end=test_end,
        )
        return
    prediction_bytes = pandas_frame_payload_bytes(prediction_frame)
    prediction_rows = len(prediction_frame)
    prediction_key_bytes = _selected_frame_payload_bytes(
        prediction_frame,
        list(primary_key),
    )
    prediction_index_bytes = prediction_key_bytes + prediction_rows * (
        8 * len(primary_key)
    )
    matched_bytes = prediction_rows
    with ExitStack() as prediction_state:
        if resource_policy is not None:
            try:
                prediction_state.enter_context(
                    resource_policy.lease(
                        "research.portfolio.prediction.input",
                        prediction_bytes + prediction_index_bytes + matched_bytes,
                    )
                )
            except MemoryError:
                raise ValueError(
                    "portfolio_prediction_join_working_set_exceeds_task_memory_budget"
                ) from None
        prediction_index = pd.MultiIndex.from_frame(
            prediction_frame.loc[:, list(primary_key)]
        )
        matched = np.zeros(prediction_rows, dtype=bool)
        missing_required = False
        with closing_iterator(iter(chunks)) as active_chunks:
            for chunk in active_chunks:
                chunk_rows = len(chunk)
                chunk_bytes = pandas_frame_payload_bytes(chunk)
                chunk_index_bytes = int(chunk.index.memory_usage(deep=True))
                timestamp_bytes = chunk_index_bytes + 8 * chunk_rows
                required_mask_bytes = chunk_index_bytes + chunk_rows
                required_key_bytes = _selected_frame_payload_bytes(
                    chunk,
                    list(primary_key),
                )
                required_index_bytes = required_key_bytes + chunk_rows * (
                    8 * len(primary_key)
                )
                positions_bytes = chunk_rows * np.dtype(np.intp).itemsize
                join_hash_bytes = chunk_rows * (32 + 16 * len(primary_key))
                joined_output_bytes = chunk_bytes + chunk_index_bytes + 8 * chunk_rows
                chunk_working_bytes = (
                    timestamp_bytes
                    + required_mask_bytes
                    + required_index_bytes
                    + positions_bytes
                    + join_hash_bytes
                    + joined_output_bytes
                )
                with ExitStack() as chunk_state:
                    if resource_policy is not None:
                        chunk_state.enter_context(
                            resource_policy.lease(
                                "research.portfolio.prediction.join",
                                chunk_working_bytes,
                            )
                        )
                    eligible, _metadata = apply_filter_mode(
                        chunk,
                        mode=sample_filter_mode,
                        metadata_key="sample_filter_mode",
                    )
                    eligible = eligible.dropna(subset=list(input_columns))
                    timestamps = strict_timestamp_series(
                        eligible["timestamp"],
                        error="invalid_portfolio_timestamp_values",
                    )
                    required_mask = timestamps >= pd.Timestamp(test_start)
                    if test_end is not None:
                        required_mask &= timestamps <= pd.Timestamp(test_end)
                    required_index = pd.MultiIndex.from_frame(
                        eligible.loc[required_mask, list(primary_key)]
                    )
                    positions = prediction_index.get_indexer(required_index)
                    missing_required |= bool((positions < 0).any())
                    matched[positions[positions >= 0]] = True
                    try:
                        yield chunk.merge(
                            prediction_frame,
                            on=list(primary_key),
                            how="left",
                            validate="one_to_one",
                        )
                    except pd.errors.MergeError as exc:
                        raise ValueError(
                            "persisted_prediction_join_key_invalid"
                        ) from exc
        unexpected = bool((~matched).any())
        if missing_required and unexpected:
            raise ValueError("misaligned_persisted_prediction_keys")
        if missing_required:
            raise ValueError("missing_required_prediction_keys")
        if unexpected:
            raise ValueError("misaligned_persisted_prediction_keys")


def _iter_prediction_lookup_joined_chunks(
    chunks,
    *,
    prediction_lookup,
    primary_key: tuple[str, ...],
    input_columns: tuple[str, ...],
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
):
    missing_required = False
    with closing_iterator(iter(chunks)) as active_chunks:
        for chunk in active_chunks:
            row_count = len(chunk)
            frame_bytes = pandas_frame_payload_bytes(chunk)
            index_bytes = int(chunk.index.memory_usage(deep=True))
            timestamp_bytes = index_bytes + 8 * row_count
            required_mask_bytes = index_bytes + row_count
            required_output_bytes = frame_bytes
            output_frame_bytes = frame_bytes + index_bytes + 8 * row_count
            policy = current_runtime_resource_policy()
            with ExitStack() as chunk_state:
                if policy is not None:
                    chunk_state.enter_context(
                        policy.lease(
                            "research.portfolio.prediction.output",
                            timestamp_bytes
                            + required_mask_bytes
                            + required_output_bytes
                            + output_frame_bytes,
                        )
                    )
                values, _found = chunk_state.enter_context(
                    prediction_lookup.values_for_frame(
                        chunk,
                        mark_matched=False,
                    )
                )
                eligible, _metadata = apply_filter_mode(
                    chunk,
                    mode=sample_filter_mode,
                    metadata_key="sample_filter_mode",
                )
                eligible = eligible.dropna(subset=list(input_columns))
                timestamps = strict_timestamp_series(
                    eligible["timestamp"], error="invalid_portfolio_timestamp_values"
                )
                required_mask = timestamps >= pd.Timestamp(test_start)
                if test_end is not None:
                    required_mask &= timestamps <= pd.Timestamp(test_end)
                required = eligible.loc[required_mask]
                if not required.empty:
                    _required_values, required_found = chunk_state.enter_context(
                        prediction_lookup.values_for_frame(
                            required,
                            mark_matched=True,
                        )
                    )
                    missing_required |= 0 in required_found
                output = chunk.copy()
                output[prediction_lookup.prediction_column] = values
                yield output
    unexpected, extra = prediction_lookup.coverage_counts()
    if extra:
        raise RuntimeError("persisted_prediction_match_state_invalid")
    if missing_required and unexpected:
        raise ValueError("misaligned_persisted_prediction_keys")
    if missing_required:
        raise ValueError("missing_required_prediction_keys")
    if unexpected:
        raise ValueError("misaligned_persisted_prediction_keys")


def _portfolio_input_chunks(
    *,
    layout: StorageLayout,
    market_id: str,
    dataset_id: str,
    time_window: TimeWindow | None,
    columns: tuple[str, ...],
):
    return iter_complete_session_frames(
        iter_research_dataset_chunks(
            layout,
            market_id=market_id,
            dataset_id=dataset_id,
            time_window=time_window,
            columns=columns,
        ),
        timestamp_column="timestamp",
        resource_policy=current_runtime_resource_policy(),
    )


def _iter_portfolio_output_chunks(
    chunks,
    spec: PortfolioSpec,
    *,
    source_columns_by_kind,
    resource_policy: RuntimeResourcePolicy | None,
) -> Iterable[dict[str, pd.DataFrame]]:
    temporal_state = _PortfolioTemporalState()
    saw_rows = False
    rolling_lease: MemoryLease | None = None
    input_cleanup = ExitStack()
    active_chunks = input_cleanup.enter_context(
        closing_iterator(iter(chunks))
    )

    try:
        for chunk in active_chunks:
            if chunk.empty:
                continue
            saw_rows = True
            policy = current_runtime_resource_policy()
            if resource_policy is not None and policy is not resource_policy:
                raise RuntimeError("runtime_resource_policy_conflict")
            input_bytes = pandas_frame_payload_bytes(chunk)
            row_count = len(chunk)
            signal_count = max(1, len(spec.signals))
            validated_panel_bytes = input_bytes
            sorted_panel_bytes = input_bytes
            row_output_bytes = row_count * signal_count * _PORTFOLIO_ROW_WORKING_BYTES
            period_output_bytes = (
                (int(spec.execution_lag_steps) + 2)
                * signal_count
                * _PORTFOLIO_PERIOD_WORKING_BYTES
            )
            state_bytes = _portfolio_temporal_state_bytes(temporal_state)
            state_growth_bytes = (
                row_count * (signal_count + 1) * _PORTFOLIO_ROW_WORKING_BYTES
            )
            operation_bytes = (
                state_bytes
                + state_growth_bytes
                + validated_panel_bytes
                + sorted_panel_bytes
                + row_output_bytes
                + period_output_bytes
            )
            if resource_policy is not None and policy is not None:
                if rolling_lease is None:
                    rolling_lease = policy.lease(
                        "research.portfolio.operation",
                        operation_bytes,
                    )
                elif operation_bytes > rolling_lease.reserved_bytes:
                    rolling_lease.grow(operation_bytes - rolling_lease.reserved_bytes)
                else:
                    rolling_lease.shrink_to(operation_bytes)
                rolling_lease.transfer("research.portfolio.operation")
            result = _construct_portfolio_positions(
                chunk,
                spec,
                source_columns_by_kind=source_columns_by_kind,
                temporal_state=temporal_state,
                resource_lease=rolling_lease,
            )
            output = {
                "positions": result.positions,
                "turnover": result.turnover,
                "cost_breakdown": result.cost_breakdown,
                "portfolio_returns": result.portfolio_returns,
                "diagnostics": result.diagnostics,
                "capacity_diagnostics": result.capacity_diagnostics,
                "numeric_exposure_diagnostics": result.numeric_exposure_diagnostics,
                "categorical_exposure_diagnostics": (
                    result.categorical_exposure_diagnostics
                ),
            }
            output_bytes = sum(
                pandas_frame_payload_bytes(frame) for frame in output.values()
            )
            retained_state_bytes = _portfolio_temporal_state_bytes(temporal_state)
            del result, chunk
            if rolling_lease is not None:
                retained_bytes = output_bytes + retained_state_bytes
                if retained_bytes > rolling_lease.reserved_bytes:
                    rolling_lease.grow(retained_bytes - rolling_lease.reserved_bytes)
                else:
                    rolling_lease.shrink_to(retained_bytes)
                rolling_lease.transfer("research.portfolio.output_state")
            yield output
            del output
            if rolling_lease is not None:
                rolling_lease.shrink_to(retained_state_bytes)
                rolling_lease.transfer("research.portfolio.temporal_state")
    finally:
        primary = sys.exception()
        first_failure: BaseException | None = None
        try:
            if rolling_lease is not None:
                rolling_lease.close()
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    "portfolio rolling lease release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            else:
                first_failure = cleanup_error
        try:
            input_cleanup.close()
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    "portfolio input cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            elif first_failure is None:
                first_failure = cleanup_error
            else:
                first_failure.add_note(
                    "portfolio input cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if first_failure is not None:
            raise first_failure

    if not saw_rows:
        raise ValueError("empty_portfolio_input")


def _observe_portfolio_summary_chunks(
    output_chunks: Iterable[dict[str, pd.DataFrame]],
    summary_state: _PortfolioSummaryState,
) -> Iterable[dict[str, pd.DataFrame]]:
    completed = False
    try:
        with closing_iterator(iter(output_chunks)) as active_chunks:
            for output_chunk in active_chunks:
                _update_portfolio_summary_state(
                    summary_state,
                    positions=output_chunk["positions"],
                    portfolio_returns=output_chunk["portfolio_returns"],
                )
                yield output_chunk
        completed = True
        summary_state.returns_complete = True
    finally:
        if not completed:
            _close_portfolio_summary_state(summary_state)


def _update_portfolio_summary_state(
    summary_state: _PortfolioSummaryState,
    *,
    positions: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
) -> None:
    active_policy = current_runtime_resource_policy()
    if summary_state.resource_policy is not None:
        if active_policy is not summary_state.resource_policy:
            raise RuntimeError("runtime_resource_policy_conflict")
    policy = summary_state.resource_policy
    return_rows = len(portfolio_returns)
    position_rows = len(positions)
    return_index_bytes = int(portfolio_returns.index.memory_usage(deep=True))
    position_index_bytes = int(positions.index.memory_usage(deep=True))
    return_projection_bytes = pandas_frame_payload_bytes(portfolio_returns)
    return_timestamp_bytes = return_index_bytes + 8 * return_rows
    return_sort_bytes = return_projection_bytes
    position_projection_bytes = pandas_frame_payload_bytes(positions)
    position_timestamp_bytes = position_index_bytes + 8 * position_rows
    position_numeric_bytes = position_index_bytes + 3 * 8 * position_rows
    position_group_bytes = position_index_bytes + 3 * 8 * position_rows
    operation_bytes = (
        return_projection_bytes
        + return_timestamp_bytes
        + return_sort_bytes
        + position_projection_bytes
        + position_timestamp_bytes
        + position_numeric_bytes
        + position_group_bytes
    )
    with ExitStack() as operation:
        if policy is not None:
            operation.enter_context(
                policy.lease(
                    "research.portfolio.summary.update",
                    operation_bytes,
                )
            )
        return_frame = _portfolio_return_summary_frame(portfolio_returns)
        if not return_frame.empty:
            _append_portfolio_return_summary(summary_state, return_frame)
        _update_position_period_summary(summary_state.position_stats, positions)
        del return_frame


def _append_portfolio_return_summary(
    summary_state: _PortfolioSummaryState,
    frame: pd.DataFrame,
) -> None:
    if summary_state.return_database is None:
        scratch = tempfile.TemporaryDirectory(prefix="qrp-portfolio-summary-")
        database: sqlite3.Connection | None = None
        try:
            database = sqlite3.connect(
                str(Path(scratch.name) / "returns.sqlite"),
            )
            database.execute(f"PRAGMA page_size={_PORTFOLIO_SQLITE_PAGE_BYTES}")
            database.execute("PRAGMA temp_store=FILE")
            _admit_portfolio_disk_growth(
                database,
                Path(scratch.name),
                row_count=0,
            )
            database.execute(
                """
                CREATE TABLE returns (
                    timestamp TEXT NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    signal TEXT NOT NULL,
                    gross_return REAL NOT NULL,
                    net_return REAL NOT NULL,
                    cost REAL NOT NULL,
                    turnover REAL NOT NULL,
                    long_count REAL NOT NULL,
                    short_count REAL NOT NULL
                )
                """
            )
            database.execute(
                "CREATE INDEX returns_signal_time ON returns(signal, timestamp_ns)"
            )
            database.commit()
        except BaseException as primary:
            if database is not None:
                try:
                    database.close()
                except BaseException as close_error:
                    primary.add_note(
                        "portfolio return database close also failed: "
                        f"{type(close_error).__name__}"
                    )
            try:
                scratch.cleanup()
            except BaseException as cleanup_error:
                primary.add_note(
                    "portfolio return scratch cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise
        summary_state.return_scratch = scratch
        summary_state.return_database = database

    database = summary_state.return_database
    assert database is not None
    scratch = summary_state.return_scratch
    assert scratch is not None

    _admit_portfolio_disk_growth(
        database,
        Path(scratch.name),
        row_count=len(frame),
    )

    def database_rows() -> Iterable[tuple[object, ...]]:
        for row in frame.itertuples(index=False, name=None):
            (
                timestamp,
                signal,
                _source,
                _mode,
                _long,
                _short,
                gross,
                turnover,
                cost,
                net,
                long_count,
                short_count,
            ) = row
            timestamp_value = pd.Timestamp(timestamp)
            yield (
                timestamp_value.isoformat(),
                timestamp_value.value,
                str(signal),
                float(gross),
                float(net),
                float(cost),
                float(turnover),
                float(long_count),
                float(short_count),
            )
    database.executemany(
        """
        INSERT INTO returns(
            timestamp, timestamp_ns, signal, gross_return, net_return,
            cost, turnover, long_count, short_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        database_rows(),
    )
    database.commit()


def _admit_portfolio_disk_growth(
    database: sqlite3.Connection,
    scratch_path: Path,
    *,
    row_count: int,
) -> None:
    growth_bytes = max(
        _PORTFOLIO_SQLITE_STARTUP_BYTES,
        max(1, row_count) * _PORTFOLIO_SQLITE_ROW_DISK_BYTES,
    )
    require_free_disk_headroom(
        scratch_path,
        reserve_bytes=2 * growth_bytes,
    )
    growth_pages = (
        growth_bytes + _PORTFOLIO_SQLITE_PAGE_BYTES - 1
    ) // _PORTFOLIO_SQLITE_PAGE_BYTES
    for schema in ("main", "temp"):
        current_pages = int(
            database.execute(f"PRAGMA {schema}.page_count").fetchone()[0]
        )
        if row_count > 0 and schema == "temp":
            current_pages = int(
                database.execute("PRAGMA temp.max_page_count").fetchone()[0]
            )
        database.execute(
            f"PRAGMA {schema}.max_page_count={current_pages + growth_pages}"
        )


def _ordered_performance_stats(
    summary_state: _PortfolioSummaryState,
) -> dict[str, dict[str, object]]:
    database = summary_state.return_database
    if database is None:
        return {}
    stats_by_signal: dict[str, dict[str, object]] = {}
    cursor = database.execute(
        """
        SELECT signal, timestamp, gross_return, net_return, cost, turnover
        FROM returns
        ORDER BY signal, timestamp_ns, rowid
        """
    )
    try:
        columns = (
            "signal",
            "timestamp",
            "gross_return",
            "net_return",
            "cost",
            "turnover",
        )
        while rows := cursor.fetchmany(_PORTFOLIO_STATS_BATCH_ROWS):
            frame = pd.DataFrame.from_records(rows, columns=columns)
            for signal, group in frame.groupby("signal", sort=False):
                signal_id = str(signal)
                stats = stats_by_signal.setdefault(
                    signal_id,
                    _new_performance_stats(),
                )
                _update_performance_stats_batch(
                    stats,
                    timestamps=group["timestamp"].to_numpy(copy=False),
                    gross_returns=group["gross_return"].to_numpy(copy=False),
                    net_returns=group["net_return"].to_numpy(copy=False),
                    costs=group["cost"].to_numpy(copy=False),
                    turnovers=group["turnover"].to_numpy(copy=False),
                )
    finally:
        primary = sys.exception()
        try:
            cursor.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                "portfolio performance cursor close also failed: "
                f"{type(cleanup_error).__name__}"
            )
    return stats_by_signal


def _portfolio_return_summary_frame(
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    if portfolio_returns.empty:
        return _empty_summary_frame(RETURNS_COLUMNS)
    selected = _validate_returns_frame(
        portfolio_returns,
        segment_columns=(),
        include_performance_columns=True,
    )
    if tuple(selected.columns) != tuple(RETURNS_COLUMNS):
        selected = selected.loc[:, RETURNS_COLUMNS]
    return selected.sort_values(
        ["signal", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def _iter_portfolio_additive_cumulative_chunks(
    summary_state: _PortfolioSummaryState,
) -> Iterable[pd.DataFrame]:
    database = summary_state.return_database
    if database is None:
        raise RuntimeError("portfolio_additive_cumulative_returns_unavailable")
    policy = summary_state.resource_policy
    if policy is None:
        batch_rows = 1024
        batch_bound = 0
    else:
        fixed_bound = _PORTFOLIO_PERIOD_WORKING_BYTES
        per_row_bound = _PORTFOLIO_ROW_WORKING_BYTES
        available = policy.available_memory_bytes
        batch_rows = max(
            1,
            min(1024, max(1, (available - fixed_bound) // per_row_bound)),
        )
        batch_bound = fixed_bound + batch_rows * per_row_bound
    cursor = database.execute(
        """
        SELECT timestamp, signal, gross_return, net_return, cost
        FROM returns
        ORDER BY signal, timestamp_ns, rowid
        """
    )
    running: dict[str, dict[str, float]] = {}
    try:
        while True:
            lease = None
            try:
                if policy is not None:
                    lease = policy.lease(
                        "research.portfolio.additive_cumulative",
                        batch_bound,
                    )
                rows = cursor.fetchmany(batch_rows)
                if not rows:
                    return
                rows_by_signal: dict[
                    str, list[tuple[object, object, object, object]]
                ] = {}
                for timestamp, signal, gross, net, cost in rows:
                    rows_by_signal.setdefault(str(signal), []).append(
                        (timestamp, gross, net, cost)
                    )
                records: list[dict[str, object]] = []
                for signal_id, signal_rows in rows_by_signal.items():
                    state = running.setdefault(
                        signal_id,
                        {
                            "gross": 0.0,
                            "net": 0.0,
                            "cost": 0.0,
                            "high_water": 0.0,
                        },
                    )
                    gross_values = np.asarray(
                        [row[1] for row in signal_rows], dtype="float64"
                    )
                    net_values = np.asarray(
                        [row[2] for row in signal_rows], dtype="float64"
                    )
                    cost_values = np.asarray(
                        [row[3] for row in signal_rows], dtype="float64"
                    )
                    gross_cumulative = _ordered_cumulative(
                        gross_values,
                        initial=state["gross"],
                    )
                    net_cumulative = _ordered_cumulative(
                        net_values,
                        initial=state["net"],
                    )
                    cost_cumulative = _ordered_cumulative(
                        cost_values,
                        initial=state["cost"],
                    )
                    high_water = np.maximum.accumulate(
                        np.concatenate(([state["high_water"]], net_cumulative))
                    )[1:]
                    state["gross"] = float(gross_cumulative[-1])
                    state["net"] = float(net_cumulative[-1])
                    state["cost"] = float(cost_cumulative[-1])
                    state["high_water"] = float(high_water[-1])
                    for position, (timestamp, _gross, _net, _cost) in enumerate(
                        signal_rows
                    ):
                        records.append(
                            {
                                "timestamp": pd.Timestamp(timestamp),
                                "signal": signal_id,
                                "gross_portfolio_return": float(
                                    gross_values[position]
                                ),
                                "net_portfolio_return": float(
                                    net_values[position]
                                ),
                                "cost": float(cost_values[position]),
                                "additive_cumulative_gross_return": float(
                                    gross_cumulative[position]
                                ),
                                "additive_cumulative_net_return": float(
                                    net_cumulative[position]
                                ),
                                "additive_cumulative_cost": float(
                                    cost_cumulative[position]
                                ),
                                "high_water_mark": float(high_water[position]),
                                "drawdown": float(
                                    net_cumulative[position] - high_water[position]
                                ),
                            }
                        )
                frame = pd.DataFrame.from_records(
                    records,
                    columns=ADDITIVE_CUMULATIVE_RETURN_COLUMNS,
                )
                if lease is not None:
                    lease.shrink_to(max(1, pandas_frame_payload_bytes(frame)))
                yield frame
            finally:
                if lease is not None:
                    primary = sys.exception()
                    try:
                        lease.close()
                    except BaseException as cleanup_error:
                        if primary is None:
                            raise
                        primary.add_note(
                            "portfolio additive cumulative lease close also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
    finally:
        primary = sys.exception()
        try:
            cursor.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(
                "portfolio additive cumulative cursor close also failed: "
                f"{type(cleanup_error).__name__}"
            )


def _update_position_period_summary(
    position_stats: dict[str, dict[str, float | int]],
    positions: pd.DataFrame,
) -> None:
    if positions.empty:
        return
    frame = validate_positions(positions)
    frame["abs_position"] = frame["position"].abs()
    for (signal, _timestamp), group in frame.groupby(
        ["signal", "timestamp"],
        sort=True,
        dropna=False,
    ):
        signal_id = str(signal)
        gross_exposure = float(group["abs_position"].sum())
        net_exposure = float(group["position"].sum())
        position_count = int((group["position"] != 0).sum())
        if gross_exposure:
            shares = group["abs_position"] / gross_exposure
            hhi = float((shares.pow(2)).sum())
            top_weight = float(shares.max())
        else:
            hhi = 0.0
            top_weight = 0.0
        stats = position_stats.setdefault(
            signal_id,
            {
                "periods": 0,
                "gross_sum": 0.0,
                "net_sum": 0.0,
                "max_gross": 0.0,
                "max_abs_net": 0.0,
                "position_count_sum": 0,
                "hhi_sum": 0.0,
                "max_hhi": 0.0,
                "top_weight_sum": 0.0,
                "max_top_weight": 0.0,
            },
        )
        stats["periods"] += 1
        stats["gross_sum"] += gross_exposure
        stats["net_sum"] += net_exposure
        stats["max_gross"] = max(float(stats["max_gross"]), gross_exposure)
        stats["max_abs_net"] = max(float(stats["max_abs_net"]), abs(net_exposure))
        stats["position_count_sum"] += position_count
        stats["hhi_sum"] += hhi
        stats["max_hhi"] = max(float(stats["max_hhi"]), hhi)
        stats["top_weight_sum"] += top_weight
        stats["max_top_weight"] = max(float(stats["max_top_weight"]), top_weight)


def _portfolio_summary_memory_bound(
    summary_state: _PortfolioSummaryState,
) -> int:
    database = summary_state.return_database
    signal_rows = 0
    period_rows = 0
    return_rows = 0
    if database is not None:
        signal_rows = int(
            database.execute(
                "SELECT COUNT(DISTINCT signal) FROM returns"
            ).fetchone()[0]
        )
        return_rows = int(
            database.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
        )
        period_rows = int(
            database.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT signal, substr(timestamp, 1, 7) AS period
                    FROM returns
                    GROUP BY signal, period
                )
                """
            ).fetchone()[0]
        )
    summary_rows = (
        max(1, signal_rows)
        + 2 * max(1, len(summary_state.position_stats))
        + period_rows
    )
    batch_rows = min(_PORTFOLIO_STATS_BATCH_ROWS, return_rows)
    batch_bound = (
        _PORTFOLIO_PERIOD_WORKING_BYTES
        + batch_rows * _PORTFOLIO_ROW_WORKING_BYTES
        if batch_rows
        else 0
    )
    return 64 * 1024 + summary_rows * 16 * 1024 + batch_bound


def _portfolio_summary_outputs(
    summary_state: _PortfolioSummaryState,
) -> dict[str, pd.DataFrame]:
    try:
        database = summary_state.return_database
        scratch = summary_state.return_scratch
        if database is not None and scratch is not None:
            aggregate_row_count = int(
                database.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
            )
            if aggregate_row_count:
                _admit_portfolio_disk_growth(
                    database,
                    Path(scratch.name),
                    row_count=aggregate_row_count,
                )
        policy = summary_state.resource_policy
        if policy is not None:
            summary_state.summary_lease = policy.lease(
                "research.portfolio.summary.outputs",
                _portfolio_summary_memory_bound(summary_state),
            )
        performance_summary = _portfolio_performance_summary(summary_state)
        exposure_summary = _portfolio_exposure_summary(summary_state)
        concentration = _portfolio_concentration_summary(summary_state)
        period_breakdown = _portfolio_period_breakdown(summary_state)
        result = {
            "performance_summary": performance_summary,
            "exposure_summary": exposure_summary,
            "concentration": concentration,
            "period_breakdown": period_breakdown,
        }
    except BaseException:
        _close_portfolio_summary_state(summary_state)
        raise
    return result


def _portfolio_performance_summary(
    summary_state: _PortfolioSummaryState,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    database = summary_state.return_database
    ordered_stats = _ordered_performance_stats(summary_state)
    for signal in sorted(ordered_stats):
        stats = ordered_stats[signal]
        periods = int(stats["periods"])
        if periods == 0:
            continue
        var_95 = _spilled_quantile(database, signal, periods, 0.05)
        q95 = _spilled_quantile(database, signal, periods, 0.95)
        cvar_values = _spilled_cvar(database, signal, var_95)
        records.append(
            _performance_summary_record(
                signal,
                stats,
                var_95=var_95,
                q95=q95,
                cvar_95=cvar_values,
            )
        )
    return _performance_summary_frame(records)


def _spilled_quantile(
    database: sqlite3.Connection | None,
    signal: str,
    count: int,
    quantile: float,
) -> float:
    if database is None or count == 0:
        raise RuntimeError("portfolio_performance_spill_unavailable")
    position = (count - 1) * quantile
    lower_index = int(math.floor(position))
    values = database.execute(
        """
        SELECT net_return FROM returns
        WHERE signal = ? AND net_return IS NOT NULL
        ORDER BY net_return
        LIMIT 2 OFFSET ?
        """,
        (signal, lower_index),
    ).fetchall()
    if not values:
        raise RuntimeError("portfolio_performance_spill_missing")
    lower = float(values[0][0])
    upper = float(values[-1][0])
    return lower + (upper - lower) * (position - lower_index)


def _spilled_cvar(
    database: sqlite3.Connection | None,
    signal: str,
    threshold: float,
) -> float:
    if database is None:
        raise RuntimeError("portfolio_performance_spill_unavailable")
    value = database.execute(
        """
        SELECT AVG(net_return) FROM returns
        WHERE signal = ? AND net_return IS NOT NULL AND net_return <= ?
        """,
        (signal, threshold),
    ).fetchone()[0]
    if value is None:
        raise RuntimeError("portfolio_performance_spill_missing")
    return float(value)


def _portfolio_period_breakdown(
    summary_state: _PortfolioSummaryState,
) -> pd.DataFrame:
    columns = [
        "signal",
        "year",
        "month",
        "periods",
        "gross_portfolio_return",
        "net_portfolio_return",
        "cost",
        "turnover",
        "hit_rate",
    ]
    database = summary_state.return_database
    if database is None:
        return _empty_summary_frame(columns)
    records = []
    period_rows = database.execute(
        """
        SELECT
            signal,
            CAST(substr(timestamp, 1, 4) AS INTEGER) AS year,
            CAST(substr(timestamp, 6, 2) AS INTEGER) AS month,
            COUNT(*) AS periods,
            SUM(gross_return),
            SUM(net_return),
            SUM(cost),
            SUM(turnover),
            SUM(CASE WHEN net_return > 0 THEN 1 ELSE 0 END)
        FROM returns
        GROUP BY signal, year, month
        ORDER BY signal, year, month
        """
    )
    for (
        signal,
        year,
        month,
        periods,
        gross,
        net,
        cost,
        turnover,
        positive,
    ) in period_rows:
        records.append(
            {
                "signal": str(signal),
                "year": int(year),
                "month": int(month),
                "periods": int(periods),
                "gross_portfolio_return": float(gross),
                "net_portfolio_return": float(net),
                "cost": float(cost),
                "turnover": float(turnover) / int(periods),
                "hit_rate": int(positive) / int(periods),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _close_portfolio_summary_state(summary_state: _PortfolioSummaryState) -> None:
    primary = sys.exception()
    cleanup_error: BaseException | None = None
    summary_lease = summary_state.summary_lease
    summary_state.summary_lease = None
    if summary_lease is not None:
        try:
            summary_lease.close()
        except BaseException as exc:
            if primary is None:
                if cleanup_error is None:
                    cleanup_error = exc
            else:
                primary.add_note(
                    "portfolio summary lease close also failed: "
                    f"{type(exc).__name__}"
                )
    database = summary_state.return_database
    summary_state.return_database = None
    if database is not None:
        try:
            database.close()
        except BaseException as exc:
            if primary is None:
                if cleanup_error is None:
                    cleanup_error = exc
            else:
                primary.add_note(
                    "portfolio return database close also failed: "
                    f"{type(exc).__name__}"
                )
    scratch = summary_state.return_scratch
    summary_state.return_scratch = None
    if scratch is not None:
        try:
            scratch.cleanup()
        except BaseException as exc:
            if primary is None:
                if cleanup_error is None:
                    cleanup_error = exc
            else:
                primary.add_note(
                    "portfolio return scratch cleanup also failed: "
                    f"{type(exc).__name__}"
                )
    state_lease = summary_state.state_lease
    summary_state.state_lease = None
    if state_lease is not None:
        try:
            state_lease.close()
        except BaseException as exc:
            if primary is None:
                if cleanup_error is None:
                    cleanup_error = exc
            else:
                primary.add_note(
                    "portfolio state lease close also failed: "
                    f"{type(exc).__name__}"
                )
    if cleanup_error is not None:
        raise cleanup_error


def _portfolio_exposure_summary(
    summary_state: _PortfolioSummaryState,
) -> pd.DataFrame:
    if not summary_state.position_stats:
        return _empty_summary_frame(EXPOSURE_SUMMARY_COLUMNS)
    count_stats: dict[str, tuple[int, float, float]] = {}
    database = summary_state.return_database
    if database is not None:
        count_stats = {
            str(signal): (
                int(rows),
                float(long_sum),
                float(short_sum),
            )
            for signal, rows, long_sum, short_sum in database.execute(
                """
                SELECT
                    signal,
                    COUNT(*),
                    SUM(long_count),
                    SUM(short_count)
                FROM returns
                GROUP BY signal
                """
            )
        }
    records: list[dict[str, object]] = []
    for signal_id in sorted(summary_state.position_stats):
        stats = summary_state.position_stats[signal_id]
        periods = int(stats["periods"])
        count_rows, long_sum, short_sum = count_stats.get(signal_id, (0, 0.0, 0.0))
        records.append(
            {
                "signal": signal_id,
                "mean_gross_exposure": float(stats["gross_sum"]) / periods,
                "mean_net_exposure": float(stats["net_sum"]) / periods,
                "max_gross_exposure": float(stats["max_gross"]),
                "max_abs_net_exposure": float(stats["max_abs_net"]),
                "mean_long_count": (
                    long_sum / count_rows
                    if count_rows
                    else 0.0
                ),
                "mean_short_count": (
                    short_sum / count_rows
                    if count_rows
                    else 0.0
                ),
                "mean_position_count": (float(stats["position_count_sum"]) / periods),
            }
        )
    return pd.DataFrame.from_records(records, columns=EXPOSURE_SUMMARY_COLUMNS)


def _portfolio_concentration_summary(
    summary_state: _PortfolioSummaryState,
) -> pd.DataFrame:
    columns = [
        "signal",
        "periods",
        "mean_hhi",
        "max_hhi",
        "mean_top_weight",
        "max_top_weight",
        "mean_position_count",
    ]
    if not summary_state.position_stats:
        return _empty_summary_frame(columns)
    records: list[dict[str, object]] = []
    for signal_id in sorted(summary_state.position_stats):
        stats = summary_state.position_stats[signal_id]
        periods = int(stats["periods"])
        records.append(
            {
                "signal": signal_id,
                "periods": periods,
                "mean_hhi": float(stats["hhi_sum"]) / periods,
                "max_hhi": float(stats["max_hhi"]),
                "mean_top_weight": float(stats["top_weight_sum"]) / periods,
                "max_top_weight": float(stats["max_top_weight"]),
                "mean_position_count": (float(stats["position_count_sum"]) / periods),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _portfolio_temporal_state_bytes(state: _PortfolioTemporalState) -> int:
    position_count = sum(
        len(positions)
        for signals in state.previous_positions.values()
        for positions in signals.values()
    )
    return (
        pandas_frame_payload_bytes(state.lag_history)
        + position_count * _PORTFOLIO_ROW_WORKING_BYTES
    )


def _empty_summary_frame(columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(columns))
    if "position" in frame.columns:
        frame["position"] = frame["position"].astype("float64")
    return frame
