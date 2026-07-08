"""Evaluation output validation."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from quant_research_platform.artifacts._output_common_validation import (
    validate_filter_metadata,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_finite_or_null_numeric_series,
    strict_non_negative_count_series,
    strict_positive_group_series,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)

EVALUATION_METRICS_COLUMNS = ("timestamp", "signal", "label", "ic", "rankic", "n_obs")
EVALUATION_GROUPED_LABEL_VALUE_COLUMNS = (
    "timestamp",
    "signal",
    "label",
    "group",
    "mean_label_value",
    "n_obs",
)
EVALUATION_IC_SUMMARY_COLUMNS = (
    "signal",
    "label",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_rate",
    "valid_ic_count",
    "mean_rankic",
    "std_rankic",
    "rankicir",
    "positive_rankic_rate",
    "valid_rankic_count",
)
EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS = (
    "signal",
    "label",
    "top_group_mean_label_value",
    "bottom_group_mean_label_value",
    "top_bottom_label_value_spread_mean",
    "top_bottom_label_value_spread_hit_rate",
    "group_monotonicity_score",
)
EVALUATION_ARTIFACTS = [
    "metrics.parquet",
    "grouped_label_values.parquet",
    "ic_summary.parquet",
    "group_spread_summary.parquet",
    "summary.md",
]


def validate_evaluation_diagnostics_consistency(
    *,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
) -> None:
    diagnostics_payload = asdict(evaluation_diagnostics)
    rows_input = int(diagnostics_payload["rows_input"])
    rows_after_filter_mode = int(diagnostics_payload["rows_after_filter_mode"])
    rows_after_signal_label_dropna = int(
        diagnostics_payload["rows_after_signal_label_dropna"]
    )
    max_signal_label_rows = (
        rows_after_filter_mode * len(spec.signals) * len(spec.label_columns)
    )
    if not (
        0 <= rows_after_filter_mode <= rows_input
        and 0 <= rows_after_signal_label_dropna <= max_signal_label_rows
    ):
        raise ValueError("evaluation_diagnostics_inconsistent")
    validate_filter_metadata(
        mode=evaluation_diagnostics.filter_mode,
        columns=list(evaluation_diagnostics.filter_columns_used),
        definition=evaluation_diagnostics.filter_definition,
        error_name="evaluation_diagnostics_inconsistent",
    )
    timestamps = pd.to_datetime(metrics["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("evaluation_diagnostics_inconsistent")
    expected_timestamp_count = int(timestamps.nunique()) if not timestamps.empty else 0
    expected_min_timestamp = (
        pd.Timestamp(timestamps.min()).date().isoformat()
        if not timestamps.empty
        else None
    )
    expected_max_timestamp = (
        pd.Timestamp(timestamps.max()).date().isoformat()
        if not timestamps.empty
        else None
    )
    n_obs = strict_non_negative_count_series(metrics["n_obs"])
    expected_skipped_timestamp_count = int(
        metrics.loc[n_obs.lt(2), "timestamp"].nunique()
    )
    expected_rows_after_signal_label_dropna = int(n_obs.sum())
    expected = {
        "filter_mode": spec.filter_mode,
        "filter_columns_used": tuple(required_filter_columns(spec.filter_mode)),
        "filter_definition": filter_definition(spec.filter_mode),
        "rows_after_signal_label_dropna": expected_rows_after_signal_label_dropna,
        "metric_rows": len(metrics),
        "grouped_label_value_rows": len(grouped_label_values),
        "ic_summary_rows": len(ic_summary),
        "group_spread_summary_rows": len(group_spread_summary),
        "timestamp_count": expected_timestamp_count,
        "min_timestamp": expected_min_timestamp,
        "max_timestamp": expected_max_timestamp,
        "skipped_timestamp_count": expected_skipped_timestamp_count,
    }
    for key, value in expected.items():
        if getattr(evaluation_diagnostics, key) != value:
            raise ValueError("evaluation_diagnostics_inconsistent")


def validate_evaluation_artifact_structure(
    *,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    spec: EvaluationSpec,
    n_groups: int,
) -> None:
    _reject_duplicate_keys(metrics, ("timestamp", "signal", "label"), "metrics")
    _reject_duplicate_keys(
        grouped_label_values,
        ("timestamp", "signal", "label", "group"),
        "grouped_label_values",
    )
    _reject_duplicate_keys(ic_summary, ("signal", "label"), "ic_summary")
    _reject_duplicate_keys(
        group_spread_summary,
        ("signal", "label"),
        "group_spread_summary",
    )
    for frame in (metrics, grouped_label_values, ic_summary, group_spread_summary):
        _validate_signal_label_membership(frame, spec=spec)
    strict_non_negative_count_series(metrics["n_obs"])
    strict_non_negative_count_series(grouped_label_values["n_obs"])
    groups = strict_positive_group_series(grouped_label_values["group"])
    if bool(groups.gt(n_groups).any()):
        raise ValueError("evaluation_group_out_of_range")
    for frame, columns in (
        (metrics, ("ic", "rankic")),
        (grouped_label_values, ("mean_label_value",)),
        (
            ic_summary,
            tuple(
                column
                for column in ic_summary.columns
                if column not in {"signal", "label"}
            ),
        ),
        (
            group_spread_summary,
            tuple(
                column
                for column in group_spread_summary.columns
                if column not in {"signal", "label"}
            ),
        ),
    ):
        for column in columns:
            strict_finite_or_null_numeric_series(frame[column])


def _reject_duplicate_keys(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    artifact_name: str,
) -> None:
    if bool(frame.duplicated(list(keys)).any()):
        raise ValueError(f"duplicate_evaluation_{artifact_name}_keys")


def _validate_signal_label_membership(
    frame: pd.DataFrame,
    *,
    spec: EvaluationSpec,
) -> None:
    signal_ids = {signal.signal_id for signal in spec.signals}
    if not set(frame["signal"]).issubset(signal_ids) or not set(
        frame["label"]
    ).issubset(set(spec.label_columns)):
        raise ValueError("evaluation_artifact_signal_label_mismatch")
