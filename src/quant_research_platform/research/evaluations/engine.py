"""Spec-driven evaluation orchestration for compute kernels."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass

import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_non_negative_count_series,
    strict_non_negative_count_value,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.research._memory import (
    _ResearchFrameAccumulator,
    _selected_frame_payload_bytes,
    iter_complete_session_frames,
)
from quant_research_platform.research.evaluations.diagnostics import (
    build_evaluation_diagnostics,
    summarize_group_spread,
    summarize_period_ic,
)
from quant_research_platform.research.signals.materialization import (
    normalize_signal_frame,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)

_EVALUATION_RECORD_WORKING_BYTES = 512


@dataclass(frozen=True)
class EvaluationResult:
    spec: EvaluationSpec
    metrics: pd.DataFrame
    grouped_label_values: pd.DataFrame
    ic_summary: pd.DataFrame
    group_spread_summary: pd.DataFrame
    diagnostics: EvaluationDiagnostics


def evaluate_signal_frame_chunks(
    frames: Iterable[pd.DataFrame],
    spec: EvaluationSpec,
    *,
    n_groups: int = 5,
    asset_column: str = "asset_id",
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> EvaluationResult:
    policy = bind_runtime_resource_policy(
        resource_policy,
        allow_implicit_active=False,
    ) or RuntimeResourcePolicy.capture()
    with runtime_resource_policy_scope(policy):
        return _evaluate_signal_frame_chunks(
            frames,
            spec,
            n_groups=n_groups,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
        )


def _evaluate_signal_frame_chunks(
    frames: Iterable[pd.DataFrame],
    spec: EvaluationSpec,
    *,
    n_groups: int,
    asset_column: str,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None,
) -> EvaluationResult:
    owner_policy = current_runtime_resource_policy()
    metrics, grouped_label_values, filter_metadata, rows_after_dropna = (
        _compute_evaluation_frames(
            frames,
            spec=spec,
            signal_columns=tuple(signal.signal_id for signal in spec.signals),
            label_columns=spec.label_columns,
            n_groups=n_groups,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
        )
    )
    retained_bytes = pandas_frame_payload_bytes(metrics) + pandas_frame_payload_bytes(
        grouped_label_values
    )
    aggregation_state_bytes = (len(metrics) + len(grouped_label_values)) * 8 * 8
    summary_output_bound = retained_bytes + 64 * 1024
    with ExitStack() as retained_outputs:
        if owner_policy is not None:
            retained_outputs.enter_context(
                owner_policy.reserve("research.evaluation.result", retained_bytes)
            )
            retained_outputs.enter_context(
                owner_policy.reserve(
                    "research.evaluation.summary",
                    aggregation_state_bytes + summary_output_bound,
                )
            )
        ic_summary = summarize_period_ic(metrics)
        group_spread_summary = summarize_group_spread(grouped_label_values)
        diagnostics = build_evaluation_diagnostics(
            spec=spec,
            filter_metadata=filter_metadata,
            rows_after_signal_label_dropna=rows_after_dropna,
            metrics=metrics,
            grouped_label_values=grouped_label_values,
            ic_summary=ic_summary,
            group_spread_summary=group_spread_summary,
            n_groups=n_groups,
            timestamp_column="timestamp",
        )
        if (
            pandas_frame_payload_bytes(ic_summary)
            + pandas_frame_payload_bytes(group_spread_summary)
            > summary_output_bound
        ):
            raise ValueError(
                "research_evaluation_working_set_exceeds_task_memory_budget"
            )
    return EvaluationResult(
        spec=spec,
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        diagnostics=diagnostics,
    )


def _compute_evaluation_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: EvaluationSpec,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    n_groups: int,
    asset_column: str,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], int]:
    from quant_research_platform.research.filtering import apply_filter_mode

    rows_input = 0
    rows_after_filter = 0
    rows_after_dropna = 0
    filter_columns_used: list[str] = []
    filter_definition = ""
    retained_policy = current_runtime_resource_policy()
    with (
        _ResearchFrameAccumulator(
            "research.evaluation.metrics",
            resource_policy=retained_policy,
        ) as metric_accumulator,
        _ResearchFrameAccumulator(
            "research.evaluation.grouped",
            resource_policy=retained_policy,
        ) as grouped_accumulator,
    ):
        for chunk in _iter_evaluation_work_frames(
            frames,
            spec=spec,
            signal_columns=signal_columns,
            label_columns=label_columns,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
        ):
            filtered, metadata = apply_filter_mode(
                chunk,
                mode=spec.filter_mode,
                metadata_key="filter_mode",
            )
            rows_input += strict_non_negative_count_value(metadata["rows_input"])
            rows_after_filter += strict_non_negative_count_value(
                metadata["rows_after_filter_mode"]
            )
            filter_columns_used = list(metadata["filter_columns_used"])
            filter_definition = str(metadata["filter_definition"])
            for label_column in label_columns:
                policy = active_runtime_resource_policy()
                compute_spec = ComputePanelSpec(
                    signal_columns=signal_columns,
                    label_column=label_column,
                    timestamp_column="timestamp",
                    asset_column=asset_column,
                )
                compute_columns = (
                    compute_spec.timestamp_column,
                    compute_spec.asset_column,
                    *compute_spec.signal_columns,
                    compute_spec.label_column,
                )
                row_count = len(filtered)
                index_bytes = int(filtered.index.memory_usage(deep=True))
                usable_bytes = _selected_frame_payload_bytes(
                    filtered,
                    compute_columns,
                )
                retained_filter_bytes = (
                    0 if filtered is chunk else pandas_frame_payload_bytes(filtered)
                )
                panel_copy_bytes = usable_bytes
                sorted_panel_bytes = usable_bytes
                null_key_mask_bytes = index_bytes + row_count
                duplicate_key_mask_bytes = index_bytes + row_count
                numeric_kernel_bytes = row_count * (
                    3 * 1  # validity/group masks
                    + 10 * 8  # selected, centered, ranked, and grouped arrays
                )
                output_record_count = len(signal_columns) * (1 + n_groups)
                output_record_bytes = (
                    output_record_count * _EVALUATION_RECORD_WORKING_BYTES
                )
                kernel_bytes = (
                    retained_filter_bytes
                    + usable_bytes
                    + panel_copy_bytes
                    + sorted_panel_bytes
                    + null_key_mask_bytes
                    + duplicate_key_mask_bytes
                    + numeric_kernel_bytes
                    + output_record_bytes
                )
                with policy.reserve("research.evaluation.kernel", kernel_bytes):
                    usable = filtered.loc[:, list(compute_columns)].copy()
                    metrics, grouped_label_values = compute_ic_and_grouped_label_values(
                        usable,
                        compute_spec,
                        n_groups=n_groups,
                    )
                    del usable
                output_bytes = pandas_frame_payload_bytes(
                    metrics
                ) + pandas_frame_payload_bytes(grouped_label_values)
                with policy.reserve("research.evaluation.output", output_bytes):
                    rows_after_dropna += int(
                        strict_non_negative_count_series(metrics["n_obs"]).sum()
                    )
                    metric_accumulator.append(metrics)
                    grouped_accumulator.append(grouped_label_values)
                    del metrics, grouped_label_values

        metrics_result = metric_accumulator.materialize()
        result_reservation = ExitStack()
        try:
            if retained_policy is not None:
                result_reservation.enter_context(
                    retained_policy.reserve(
                        "research.evaluation.concat",
                        pandas_frame_payload_bytes(metrics_result),
                    )
                )
            grouped_result = grouped_accumulator.materialize()
        finally:
            result_reservation.close()

    filter_metadata = {
        "filter_mode": spec.filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    return (
        metrics_result,
        grouped_result,
        filter_metadata,
        rows_after_dropna,
    )


def _iter_evaluation_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: EvaluationSpec,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    asset_column: str,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> Iterator[pd.DataFrame]:
    required = {
        "timestamp",
        asset_column,
        *(signal.source_column for signal in spec.signals),
        *label_columns,
    }
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                asset_column,
                *(signal.source_column for signal in spec.signals),
                *label_columns,
                *required_filter_columns(spec.filter_mode),
            )
        )
    )
    for frame in iter_complete_session_frames(
        iter(frames),
        timestamp_column="timestamp",
        resource_policy=current_runtime_resource_policy(),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing_evaluation_columns: {', '.join(missing)}")
        selected = [column for column in work_columns if column in frame.columns]
        policy = current_runtime_resource_policy()
        projection_bytes = _selected_frame_payload_bytes(frame, selected)
        with ExitStack() as working:
            if policy is not None:
                working.enter_context(
                    policy.reserve("research.evaluation.projection", projection_bytes)
                )
            selected_frame = frame.loc[:, selected]
            normalized = selected_frame.pipe(
                normalize_signal_frame,
                signals=spec.signals,
                timestamp_column="timestamp",
                source_columns_by_kind=source_columns_by_kind,
            )
            del selected_frame
        normalized_bytes = pandas_frame_payload_bytes(normalized)
        if policy is not None and (
            policy.reserved_memory_bytes + normalized_bytes
            > policy.task_memory_budget_bytes
        ):
            raise ValueError(
                "research_evaluation_working_set_exceeds_task_memory_budget"
            )
        with ExitStack() as retained:
            if policy is not None:
                retained.enter_context(
                    policy.reserve("research.evaluation.input", normalized_bytes)
                )
            yield normalized


__all__ = ["EvaluationResult", "evaluate_signal_frame_chunks"]
