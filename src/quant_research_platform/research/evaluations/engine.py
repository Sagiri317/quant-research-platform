"""Spec-driven evaluation orchestration for compute kernels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    compute_ic_and_grouped_label_values,
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)


@dataclass(frozen=True)
class EvaluationResult:
    spec: EvaluationSpec
    metrics: pd.DataFrame
    grouped_label_values: pd.DataFrame
    ic_summary: pd.DataFrame
    group_spread_summary: pd.DataFrame
    diagnostics: EvaluationDiagnostics


def evaluate_factor_frame(
    frame: pd.DataFrame,
    spec: EvaluationSpec,
    *,
    n_groups: int = 5,
) -> EvaluationResult:
    if len(spec.feature_columns) != 1 or len(spec.label_columns) != 1:
        raise ValueError("minimal_evaluation_requires_one_feature_and_one_label")
    feature = spec.feature_columns[0]
    label = spec.label_columns[0]
    required = {"timestamp", feature, label}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_evaluation_columns: {', '.join(missing)}")

    from quant_research_platform.research.filtering import apply_filter_mode

    filtered, filter_metadata = apply_filter_mode(
        frame,
        mode=spec.filter_mode,
        metadata_key="filter_mode",
    )
    usable_frame = filtered.dropna(subset=[feature, label]).copy()
    rows_after_feature_label_dropna = int(len(usable_frame))

    compute_spec = ComputePanelSpec(
        feature_columns=(feature,),
        label_column=label,
        timestamp_column="timestamp",
        asset_column="asset_id" if "asset_id" in usable_frame else "asset",
    )
    metrics, grouped_label_values = compute_ic_and_grouped_label_values(
        usable_frame,
        compute_spec,
        n_groups=n_groups,
    )
    ic_summary = summarize_ic(metrics)
    group_spread_summary = summarize_group_spread(grouped_label_values)
    diagnostics = _evaluation_diagnostics(
        spec=spec,
        filter_metadata=filter_metadata,
        rows_after_feature_label_dropna=rows_after_feature_label_dropna,
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        n_groups=n_groups,
        timestamp_column=compute_spec.timestamp_column,
    )
    return EvaluationResult(
        spec=spec,
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        diagnostics=diagnostics,
    )


def _evaluation_diagnostics(
    *,
    spec: EvaluationSpec,
    filter_metadata: dict[str, object],
    rows_after_feature_label_dropna: int,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    n_groups: int,
    timestamp_column: str,
) -> EvaluationDiagnostics:
    timestamps = pd.to_datetime(metrics.get(timestamp_column), errors="coerce").dropna()
    n_obs = pd.to_numeric(metrics.get("n_obs"), errors="coerce")
    skipped_timestamp_count = int(n_obs.lt(2).sum()) if n_obs is not None else 0
    return EvaluationDiagnostics(
        filter_mode=spec.filter_mode,
        filter_columns_used=tuple(
            str(column) for column in filter_metadata["filter_columns_used"]
        ),
        filter_definition=str(filter_metadata["filter_definition"]),
        rows_input=int(filter_metadata["rows_input"]),
        rows_after_filter_mode=int(filter_metadata["rows_after_filter_mode"]),
        rows_after_feature_label_dropna=rows_after_feature_label_dropna,
        metric_rows=int(len(metrics)),
        grouped_label_value_rows=int(len(grouped_label_values)),
        ic_summary_rows=int(len(ic_summary)),
        group_spread_summary_rows=int(len(group_spread_summary)),
        timestamp_count=int(timestamps.nunique()) if not timestamps.empty else 0,
        min_timestamp=_date_string(timestamps.min()) if not timestamps.empty else None,
        max_timestamp=_date_string(timestamps.max()) if not timestamps.empty else None,
        n_groups=n_groups,
        skipped_timestamp_count=skipped_timestamp_count,
    )


def _date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


__all__ = ["EvaluationResult", "evaluate_factor_frame"]
