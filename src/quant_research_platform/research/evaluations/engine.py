"""Spec-driven evaluation orchestration for compute kernels."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
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
from quant_research_platform.research._memory import iter_complete_time_sized_frames
from quant_research_platform.research.evaluations.diagnostics import (
    build_evaluation_diagnostics,
    summarize_group_spread,
    summarize_period_ic,
)
from quant_research_platform.research.signals.materialization import (
    normalize_signal_frame,
)


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
) -> EvaluationResult:
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

    metric_parts: list[pd.DataFrame] = []
    grouped_parts: list[pd.DataFrame] = []
    rows_input = 0
    rows_after_filter = 0
    rows_after_dropna = 0
    filter_columns_used: list[str] = []
    filter_definition = ""

    for chunk in iter_complete_time_sized_frames(
        _iter_evaluation_work_frames(
            frames,
            spec=spec,
            signal_columns=signal_columns,
            label_columns=label_columns,
            asset_column=asset_column,
            source_columns_by_kind=source_columns_by_kind,
        ),
        timestamp_column="timestamp",
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
            usable = filtered.loc[:, list(compute_columns)].copy()
            metrics, grouped_label_values = compute_ic_and_grouped_label_values(
                usable,
                compute_spec,
                n_groups=n_groups,
            )
            rows_after_dropna += int(
                strict_non_negative_count_series(metrics["n_obs"]).sum()
            )
            metric_parts.append(metrics)
            grouped_parts.append(grouped_label_values)

    filter_metadata = {
        "filter_mode": spec.filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    return (
        _concat_parts(metric_parts),
        _concat_parts(grouped_parts),
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
    for frame in frames:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing_evaluation_columns: {', '.join(missing)}")
        yield frame.loc[
            :,
            [column for column in work_columns if column in frame.columns],
        ].pipe(
            normalize_signal_frame,
            signals=spec.signals,
            timestamp_column="timestamp",
            source_columns_by_kind=source_columns_by_kind,
        )


def _concat_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [part for part in parts if not part.empty]
    if not non_empty:
        return parts[0].copy() if parts else pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


__all__ = ["EvaluationResult", "evaluate_signal_frame_chunks"]
