"""Spec-driven portfolio runner from ResearchDataset inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.compute.dataframe.portfolio_constants import (
    EXPOSURE_SUMMARY_COLUMNS,
    POSITION_COLUMNS,
    RETURNS_COLUMNS,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.research._memory import (
    _ResearchFrameSpill,
    _selected_frame_payload_bytes,
    iter_complete_session_frames,
)
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.portfolios.engine import (
    compute_cumulative_returns,
    compute_performance_summary,
    construct_portfolio_positions,
    summarize_period_breakdown,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage.layout import StorageLayout

_PORTFOLIO_ROW_WORKING_BYTES = 2048
_PORTFOLIO_PERIOD_WORKING_BYTES = 4096


@dataclass(frozen=True)
class PortfolioResult:
    spec: PortfolioSpec
    output_chunks: Iterable[Mapping[str, pd.DataFrame]]
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]]


@dataclass
class _PortfolioSummaryState:
    position_stats: dict[str, dict[str, float | int]] = field(default_factory=dict)
    resource_policy: RuntimeResourcePolicy | None = None
    return_spill: _ResearchFrameSpill | None = field(default=None, repr=False)


def build_portfolio_result_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    portfolio_catalog: PortfolioCatalog,
    portfolio_id: str,
    time_window: TimeWindow | None = None,
    prediction_frame: pd.DataFrame | None = None,
    prediction_metadata: Mapping[str, object] | None = None,
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
        resource_policy=current_runtime_resource_policy()
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
    return PortfolioResult(
        spec=spec,
        output_chunks=output_chunks,
        summary_factory=lambda: _portfolio_summary_outputs(summary_state),
    )


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
                policy.reserve(
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
        values = pd.to_numeric(frame[signal.source_column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64", copy=False)).all():
            raise ValueError("persisted_prediction_values_invalid")
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
    resource_policy: RuntimeResourcePolicy | None,
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
    policy = bind_runtime_resource_policy(resource_policy)
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
        if policy is not None:
            try:
                prediction_state.enter_context(
                    policy.reserve(
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
        for chunk in chunks:
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
                if policy is not None:
                    chunk_state.enter_context(
                        policy.reserve(
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
                    eligible["timestamp"], error="invalid_portfolio_timestamp_values"
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
                    raise ValueError("persisted_prediction_join_key_invalid") from exc
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
    for chunk in chunks:
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
                    policy.reserve(
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
    )


def _iter_portfolio_output_chunks(
    chunks,
    spec: PortfolioSpec,
    *,
    source_columns_by_kind,
    resource_policy: RuntimeResourcePolicy | None,
) -> Iterable[dict[str, pd.DataFrame]]:
    context = pd.DataFrame()
    context_reservation = ExitStack()
    context_periods = max(1, int(spec.execution_lag_steps) + 1)
    saw_rows = False

    try:
        for chunk in chunks:
            if chunk.empty:
                continue
            context_reservation.close()
            context_reservation = ExitStack()
            saw_rows = True
            policy = current_runtime_resource_policy()
            if resource_policy is not None and policy is not resource_policy:
                raise RuntimeError("runtime_resource_policy_conflict")
            input_bytes = pandas_frame_payload_bytes(
                context
            ) + pandas_frame_payload_bytes(chunk)
            row_count = len(context) + len(chunk)
            signal_count = max(1, len(spec.signals))
            combined_frame_bytes = 0 if context.empty else input_bytes
            validated_panel_bytes = input_bytes
            sorted_panel_bytes = input_bytes
            row_output_bytes = row_count * signal_count * _PORTFOLIO_ROW_WORKING_BYTES
            period_output_bytes = (
                (context_periods + 1) * signal_count * _PORTFOLIO_PERIOD_WORKING_BYTES
            )
            timestamp_state_bytes = len(chunk) * (
                np.dtype("datetime64[ns]").itemsize
                + np.dtype(np.intp).itemsize
                + np.dtype(np.bool_).itemsize
            )
            selection_bytes = row_output_bytes + period_output_bytes
            next_context_bytes = input_bytes
            operation_bytes = (
                combined_frame_bytes
                + validated_panel_bytes
                + sorted_panel_bytes
                + row_output_bytes
                + period_output_bytes
                + timestamp_state_bytes
                + selection_bytes
                + next_context_bytes
            )
            with ExitStack() as operation_reservation:
                if policy is not None:
                    operation_reservation.enter_context(
                        policy.reserve(
                            "research.portfolio.operation",
                            operation_bytes,
                        )
                    )
                actual_timestamps = _timestamp_set(chunk)
                working = _combine_context_and_chunk(context, chunk)
                result = construct_portfolio_positions(
                    working,
                    spec,
                    source_columns_by_kind=source_columns_by_kind,
                )
                output = {
                    "positions": _select_actual_timestamps(
                        result.positions,
                        actual_timestamps,
                    ),
                    "turnover": _select_actual_timestamps(
                        result.turnover,
                        actual_timestamps,
                    ),
                    "cost_breakdown": _select_actual_timestamps(
                        result.cost_breakdown,
                        actual_timestamps,
                    ),
                    "portfolio_returns": _select_actual_timestamps(
                        result.portfolio_returns,
                        actual_timestamps,
                    ),
                    "diagnostics": _select_actual_timestamps(
                        result.diagnostics,
                        actual_timestamps,
                    ),
                    "capacity_diagnostics": _select_actual_timestamps(
                        result.capacity_diagnostics,
                        actual_timestamps,
                    ),
                    "numeric_exposure_diagnostics": _select_actual_timestamps(
                        result.numeric_exposure_diagnostics,
                        actual_timestamps,
                    ),
                    "categorical_exposure_diagnostics": _select_actual_timestamps(
                        result.categorical_exposure_diagnostics,
                        actual_timestamps,
                    ),
                }
                next_context = _trailing_timestamp_context(
                    working,
                    periods=context_periods,
                )
                output_bytes = sum(
                    pandas_frame_payload_bytes(frame) for frame in output.values()
                )
                retained_context_bytes = pandas_frame_payload_bytes(next_context)
                del actual_timestamps, result, working, chunk
                operation_reservation.close()
                context = next_context
                if resource_policy is not None:
                    context_reservation.enter_context(
                        resource_policy.reserve(
                            "research.portfolio.context",
                            retained_context_bytes,
                        )
                    )
                with ExitStack() as output_reservation:
                    if policy is not None:
                        output_reservation.enter_context(
                            policy.reserve("research.portfolio.output", output_bytes)
                        )
                    yield output
                del output, next_context
    finally:
        context_reservation.close()

    if not saw_rows:
        raise ValueError("empty_portfolio_input")


def _observe_portfolio_summary_chunks(
    output_chunks: Iterable[dict[str, pd.DataFrame]],
    summary_state: _PortfolioSummaryState,
) -> Iterable[dict[str, pd.DataFrame]]:
    completed = False
    try:
        for output_chunk in output_chunks:
            _update_portfolio_summary_state(
                summary_state,
                positions=output_chunk["positions"],
                portfolio_returns=output_chunk["portfolio_returns"],
            )
            yield output_chunk
        completed = True
    finally:
        if not completed:
            _close_portfolio_summary_spill(summary_state)


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
                policy.reserve(
                    "research.portfolio.summary.update",
                    operation_bytes,
                )
            )
        return_frame = _portfolio_return_summary_frame(portfolio_returns)
        if not return_frame.empty:
            if summary_state.return_spill is None:
                summary_state.return_spill = _ResearchFrameSpill(
                    "research.portfolio.summary",
                    resource_policy=policy,
                )
            summary_state.return_spill.append(return_frame)
        _update_position_period_summary(summary_state.position_stats, positions)
        del return_frame


def _portfolio_return_summary_frame(
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    if portfolio_returns.empty:
        return _empty_summary_frame(RETURNS_COLUMNS)
    selected = portfolio_returns.loc[:, RETURNS_COLUMNS].copy()
    selected["timestamp"] = strict_timestamp_series(
        selected["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    return selected.sort_values(["signal", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )


def _update_position_period_summary(
    position_stats: dict[str, dict[str, float | int]],
    positions: pd.DataFrame,
) -> None:
    if positions.empty:
        return
    frame = positions.loc[:, POSITION_COLUMNS].copy()
    frame["timestamp"] = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)
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


def _portfolio_summary_outputs(
    summary_state: _PortfolioSummaryState,
) -> dict[str, pd.DataFrame]:
    try:
        portfolio_returns = _portfolio_returns_summary_frame(summary_state)
        policy = summary_state.resource_policy
        returns_bytes = pandas_frame_payload_bytes(portfolio_returns)
        aggregation_state_bytes = len(portfolio_returns) * 8 * 8
        summary_output_bound = returns_bytes + 64 * 1024
        with ExitStack() as summary_reservation:
            if policy is not None:
                summary_reservation.enter_context(
                    policy.reserve(
                        "research.portfolio.summary.outputs",
                        returns_bytes
                        + aggregation_state_bytes
                        + summary_output_bound,
                    )
                )
            cumulative_returns = compute_cumulative_returns(portfolio_returns)
            performance_summary = compute_performance_summary(
                portfolio_returns,
                cumulative_returns,
            )
            exposure_summary = _portfolio_exposure_summary(
                summary_state,
                portfolio_returns,
            )
            concentration = _portfolio_concentration_summary(summary_state)
            period_breakdown = summarize_period_breakdown(portfolio_returns)
            if (
                sum(
                    pandas_frame_payload_bytes(frame)
                    for frame in (
                        cumulative_returns,
                        performance_summary,
                        exposure_summary,
                        concentration,
                        period_breakdown,
                    )
                )
                > summary_output_bound
            ):
                raise ValueError(
                    "portfolio_summary_working_set_exceeds_task_memory_budget"
                )
        return {
            "cumulative_returns": cumulative_returns,
            "performance_summary": performance_summary,
            "exposure_summary": exposure_summary,
            "concentration": concentration,
            "period_breakdown": period_breakdown,
        }
    finally:
        _close_portfolio_summary_spill(summary_state)


def _portfolio_returns_summary_frame(
    summary_state: _PortfolioSummaryState,
) -> pd.DataFrame:
    spill = summary_state.return_spill
    if spill is None or spill.row_count == 0:
        return _empty_summary_frame(RETURNS_COLUMNS)
    frame = spill.materialize()
    policy = summary_state.resource_policy
    input_bytes = pandas_frame_payload_bytes(frame)
    sorted_output_bound = input_bytes
    with ExitStack() as sort_reservation:
        if policy is not None:
            sort_reservation.enter_context(
                policy.reserve(
                    "research.portfolio.summary.sort",
                    input_bytes + sorted_output_bound,
                )
            )
        output = frame.sort_values(
            ["signal", "timestamp"],
            kind="mergesort",
        ).reset_index(drop=True)
        if pandas_frame_payload_bytes(output) > sorted_output_bound:
            raise ValueError("portfolio_summary_working_set_exceeds_task_memory_budget")
        return output


def _close_portfolio_summary_spill(summary_state: _PortfolioSummaryState) -> None:
    if summary_state.return_spill is not None:
        summary_state.return_spill.close()
        summary_state.return_spill = None


def _portfolio_exposure_summary(
    summary_state: _PortfolioSummaryState,
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    if not summary_state.position_stats:
        return _empty_summary_frame(EXPOSURE_SUMMARY_COLUMNS)
    count_summary = _return_count_summary(portfolio_returns)
    records: list[dict[str, object]] = []
    for signal_id in sorted(summary_state.position_stats):
        stats = summary_state.position_stats[signal_id]
        periods = int(stats["periods"])
        count_values = count_summary.get(signal_id, {})
        records.append(
            {
                "signal": signal_id,
                "mean_gross_exposure": float(stats["gross_sum"]) / periods,
                "mean_net_exposure": float(stats["net_sum"]) / periods,
                "max_gross_exposure": float(stats["max_gross"]),
                "max_abs_net_exposure": float(stats["max_abs_net"]),
                "mean_long_count": float(count_values.get("mean_long_count", 0.0)),
                "mean_short_count": float(count_values.get("mean_short_count", 0.0)),
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


def _return_count_summary(
    portfolio_returns: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if portfolio_returns.empty:
        return {}
    frame = portfolio_returns.loc[:, ["signal", "long_count", "short_count"]].copy()
    frame["long_count"] = pd.to_numeric(frame["long_count"], errors="coerce")
    frame["short_count"] = pd.to_numeric(frame["short_count"], errors="coerce")
    result: dict[str, dict[str, float]] = {}
    for signal_id, group in frame.groupby("signal", sort=True, dropna=False):
        result[str(signal_id)] = {
            "mean_long_count": float(group["long_count"].mean()),
            "mean_short_count": float(group["short_count"].mean()),
        }
    return result


def _combine_context_and_chunk(
    context: pd.DataFrame,
    chunk: pd.DataFrame,
) -> pd.DataFrame:
    if context.empty:
        return chunk
    # Bounded context carries only recent timestamps for lag/turnover continuity.
    return pd.concat([context, chunk], ignore_index=True)


def _timestamp_set(frame: pd.DataFrame) -> pd.DatetimeIndex:
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    return pd.DatetimeIndex(timestamps.unique()).sort_values()


def _select_actual_timestamps(
    frame: pd.DataFrame,
    actual_timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    return frame.loc[timestamps.isin(actual_timestamps)].copy()


def _trailing_timestamp_context(frame: pd.DataFrame, *, periods: int) -> pd.DataFrame:
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    trailing = pd.DatetimeIndex(timestamps.unique()).sort_values()[-periods:]
    return frame.loc[timestamps.isin(trailing)].copy()


def _empty_summary_frame(columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(columns))
    if "position" in frame.columns:
        frame["position"] = frame["position"].astype("float64")
    return frame


__all__ = ["PortfolioResult", "build_portfolio_result_from_research_dataset"]
