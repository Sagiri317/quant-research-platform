"""Evaluation output validation."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts.outputs.common_validation import (
    require_metadata_string,
    require_metadata_string_list,
    require_non_negative_metadata_int,
    require_optional_metadata_string,
    require_signal_metadata_list,
    validate_filter_metadata,
)
from quant_research_platform.contracts.evaluations import (
    CORRELATION_RAW_REASON_CODES,
    CORRELATION_SUMMARY_REASON_CODES,
    GROUP_SPREAD_REASON_CODES,
    EvaluationDiagnostics,
    EvaluationSpec,
    strict_non_negative_count_series,
    strict_positive_group_series,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_membership_series,
    strict_string_series,
    strict_timestamp_series,
)

EVALUATION_ARTIFACTS = [
    "metrics.parquet",
    "grouped_label_values.parquet",
    "ic_summary.parquet",
    "group_spread_summary.parquet",
]
_EVALUATION_OUTPUT_METADATA_KEYS = frozenset(
    {
        "evaluation_id",
        "run_type",
        "market_id",
        "research_dataset_id",
        "evaluation_axis",
        "signals",
        "label_columns",
        "label_availability_columns",
        "filter_mode",
        "evaluation_diagnostics",
    }
)
_EVALUATION_DIAGNOSTIC_KEYS = frozenset(
    {
        "filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_signal_label_dropna",
        "metric_rows",
        "grouped_label_value_rows",
        "ic_summary_rows",
        "group_spread_summary_rows",
        "timestamp_count",
        "min_timestamp",
        "max_timestamp",
        "n_groups",
        "skipped_timestamp_count",
    }
)


def validate_evaluation_science(metadata):
    """Validate the optional actual evaluation window, knowledge cutoff and role."""
    from quant_research_platform.contracts.time import TimeWindow

    if type(metadata) is not dict or set(metadata) != {
        "window",
        "cutoff",
        "sample_role",
    }:
        raise ValueError("evaluation_science_metadata_invalid")
    if metadata["sample_role"] not in {"train", "validation", "test"}:
        raise ValueError("evaluation_sample_role_invalid")
    window = TimeWindow(**metadata["window"])
    if window.end is None or any(
        pd.isna(pd.Timestamp(value))
        for value in (window.start, window.end, metadata["cutoff"])
    ):
        raise ValueError("evaluation_science_time_invalid")


def validate_evaluation_output_metadata_shape(
    metadata: object,
) -> dict[str, object]:
    science = {"window", "cutoff", "sample_role"}
    if type(metadata) is not dict or set(metadata) not in (
        _EVALUATION_OUTPUT_METADATA_KEYS,
        _EVALUATION_OUTPUT_METADATA_KEYS | science,
    ):
        raise ValueError("output_manifest_metadata_invalid")
    if science <= set(metadata):
        validate_evaluation_science({key: metadata[key] for key in science})
    for key in (
        "evaluation_id",
        "run_type",
        "market_id",
        "research_dataset_id",
        "evaluation_axis",
        "filter_mode",
    ):
        require_metadata_string(metadata, key, "output_manifest_metadata_invalid")
    label_columns = require_metadata_string_list(
        metadata, "label_columns", "output_manifest_metadata_invalid"
    )
    label_availability_columns = metadata["label_availability_columns"]
    if type(label_availability_columns) is not dict or any(
        type(label) is not str or type(column) is not str
        for label, column in label_availability_columns.items()
    ):
        raise ValueError("output_manifest_metadata_invalid")
    signals = require_signal_metadata_list(
        metadata, "signals", "output_manifest_metadata_invalid"
    )
    if not label_columns:
        raise ValueError("output_manifest_metadata_invalid")

    diagnostics = metadata["evaluation_diagnostics"]
    if type(diagnostics) is not dict or set(diagnostics) != _EVALUATION_DIAGNOSTIC_KEYS:
        raise ValueError("output_manifest_metadata_invalid")
    require_metadata_string(
        diagnostics, "filter_mode", "output_manifest_metadata_invalid"
    )
    require_metadata_string(
        diagnostics, "filter_definition", "output_manifest_metadata_invalid"
    )
    require_metadata_string_list(
        diagnostics,
        "filter_columns_used",
        "output_manifest_metadata_invalid",
        allow_empty=True,
    )
    for key in (
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_signal_label_dropna",
        "metric_rows",
        "grouped_label_value_rows",
        "ic_summary_rows",
        "group_spread_summary_rows",
        "timestamp_count",
        "n_groups",
        "skipped_timestamp_count",
    ):
        require_non_negative_metadata_int(
            diagnostics, key, "output_manifest_metadata_invalid"
        )
    min_timestamp = require_optional_metadata_string(
        diagnostics, "min_timestamp", "output_manifest_metadata_invalid"
    )
    max_timestamp = require_optional_metadata_string(
        diagnostics, "max_timestamp", "output_manifest_metadata_invalid"
    )
    if (diagnostics["min_timestamp"] is None) != (diagnostics["max_timestamp"] is None):
        raise ValueError("output_manifest_metadata_invalid")
    if science <= set(metadata) and min_timestamp is not None:
        from quant_research_platform.contracts.time import (
            TimeWindow,
            time_window_timestamp_interval,
        )

        start, stop, closed = time_window_timestamp_interval(
            TimeWindow(**metadata["window"])
        )
        maximum = pd.Timestamp(max_timestamp)
        if pd.Timestamp(min_timestamp) < pd.Timestamp(start) or (
            maximum > pd.Timestamp(stop) if closed else maximum >= pd.Timestamp(stop)
        ):
            raise ValueError("evaluation_diagnostics_outside_window")
    if not (
        diagnostics["rows_after_filter_mode"] <= diagnostics["rows_input"]
        and diagnostics["rows_after_signal_label_dropna"]
        <= diagnostics["rows_after_filter_mode"]
        * len(metadata["signals"])
        * len(label_columns)
    ):
        raise ValueError("output_manifest_metadata_invalid")
    if metadata["filter_mode"] != diagnostics["filter_mode"]:
        raise ValueError("output_manifest_metadata_invalid")
    if metadata["evaluation_axis"] == "cross_sectional":
        if diagnostics["metric_rows"] < diagnostics["timestamp_count"]:
            raise ValueError("output_manifest_metadata_invalid")
    elif metadata["evaluation_axis"] == "single_asset_time_series":
        expected_metric_rows = (
            len(signals) * len(label_columns) if diagnostics["timestamp_count"] else 0
        )
        if (
            diagnostics["metric_rows"] != expected_metric_rows
            or diagnostics["grouped_label_value_rows"] != 0
            or diagnostics["group_spread_summary_rows"] != 0
        ):
            raise ValueError("output_manifest_metadata_invalid")
    else:
        raise ValueError("output_manifest_metadata_invalid")
    try:
        evaluation_diagnostics = EvaluationDiagnostics(
            filter_mode=diagnostics["filter_mode"],
            filter_columns_used=tuple(diagnostics["filter_columns_used"]),
            filter_definition=diagnostics["filter_definition"],
            rows_input=diagnostics["rows_input"],
            rows_after_filter_mode=diagnostics["rows_after_filter_mode"],
            rows_after_signal_label_dropna=diagnostics[
                "rows_after_signal_label_dropna"
            ],
            metric_rows=diagnostics["metric_rows"],
            grouped_label_value_rows=diagnostics["grouped_label_value_rows"],
            ic_summary_rows=diagnostics["ic_summary_rows"],
            group_spread_summary_rows=diagnostics["group_spread_summary_rows"],
            timestamp_count=diagnostics["timestamp_count"],
            min_timestamp=min_timestamp,
            max_timestamp=max_timestamp,
            n_groups=diagnostics["n_groups"],
            skipped_timestamp_count=diagnostics["skipped_timestamp_count"],
        )
        validate_filter_metadata(
            mode=evaluation_diagnostics.filter_mode,
            columns=list(evaluation_diagnostics.filter_columns_used),
            definition=evaluation_diagnostics.filter_definition,
            error_name="output_manifest_metadata_invalid",
        )
        EvaluationSpec(
            evaluation_id=metadata["evaluation_id"],
            market_id=MarketId(metadata["market_id"]),
            research_dataset_id=metadata["research_dataset_id"],
            signals=signals,
            label_columns=tuple(label_columns),
            label_availability_columns=label_availability_columns,
            evaluation_axis=metadata["evaluation_axis"],
            filter_mode=metadata["filter_mode"],
            n_groups=evaluation_diagnostics.n_groups,
            run_type=metadata["run_type"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("output_manifest_metadata_invalid") from exc
    return metadata


def validate_evaluation_diagnostics_consistency(
    *,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
    rows_input: int,
    rows_after_filter_mode: int,
    rows_after_signal_label_dropna: int,
    metric_rows: int,
    grouped_label_value_rows: int,
    ic_summary_rows: int,
    group_spread_summary_rows: int,
    timestamp_count: int,
    min_timestamp: str | None,
    max_timestamp: str | None,
    skipped_timestamp_count: int,
    filter_columns_used: tuple[str, ...],
    filter_definition_value: str,
) -> None:
    if not (
        0 <= rows_after_filter_mode <= rows_input
        and 0
        <= rows_after_signal_label_dropna
        <= rows_after_filter_mode * len(spec.signals) * len(spec.label_columns)
    ):
        raise ValueError("evaluation_diagnostics_inconsistent")
    validate_filter_metadata(
        mode=spec.filter_mode,
        columns=list(filter_columns_used),
        definition=filter_definition_value,
        error_name="evaluation_diagnostics_inconsistent",
    )
    expected = {
        "filter_mode": spec.filter_mode,
        "filter_columns_used": tuple(required_filter_columns(spec.filter_mode)),
        "filter_definition": filter_definition(spec.filter_mode),
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter_mode,
        "rows_after_signal_label_dropna": rows_after_signal_label_dropna,
        "metric_rows": metric_rows,
        "grouped_label_value_rows": grouped_label_value_rows,
        "ic_summary_rows": ic_summary_rows,
        "group_spread_summary_rows": group_spread_summary_rows,
        "timestamp_count": timestamp_count,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "skipped_timestamp_count": skipped_timestamp_count,
        "n_groups": spec.n_groups,
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
    science: dict | None = None,
    validate_pairs: bool = False,
) -> None:
    for frame in (metrics, grouped_label_values, ic_summary, group_spread_summary):
        _validate_signal_label_membership(frame, spec=spec)
    for frame in (metrics, grouped_label_values):
        if frame.empty:
            continue
        if not pd.api.types.is_datetime64_any_dtype(frame.timestamp.dtype):
            raise ValueError("invalid_evaluation_timestamp_values")
        timestamps = strict_timestamp_series(
            frame.timestamp, error="invalid_evaluation_timestamp_values"
        )
        if science is not None and "window" in science:
            from quant_research_platform.contracts.time import (
                TimeWindow,
                time_window_timestamp_interval,
            )

            start, stop, closed = time_window_timestamp_interval(
                TimeWindow(**science["window"])
            )
            if timestamps.lt(pd.Timestamp(start)).any() or (
                timestamps.gt(pd.Timestamp(stop)).any()
                if closed
                else timestamps.ge(pd.Timestamp(stop)).any()
            ):
                raise ValueError("evaluation_member_outside_window")
    if spec.evaluation_axis == "single_asset_time_series":
        if (
            not grouped_label_values.empty
            or not group_spread_summary.empty
            or not metrics["rankic"].isna().all()
            or not metrics["rankic_reason"].eq("not_applicable").all()
        ):
            raise ValueError("single_asset_time_series_evaluation_artifact_invalid")
    elif metrics["rankic_reason"].eq("not_applicable").any():
        raise ValueError("cross_sectional_evaluation_artifact_invalid")
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
    strict_non_negative_count_series(metrics["n_obs"])
    strict_non_negative_count_series(grouped_label_values["n_obs"])
    groups = strict_positive_group_series(grouped_label_values["group"])
    if bool(groups.gt(spec.n_groups).any()):
        raise ValueError("evaluation_group_out_of_range")
    for frame, columns in (
        (metrics, ("ic", "rankic", "n_obs")),
        (grouped_label_values, ("mean_label_value", "group", "n_obs")),
        (
            ic_summary,
            tuple(
                column
                for column in ic_summary.columns
                if column not in {"signal", "label"} and not column.endswith("_reason")
            ),
        ),
        (
            group_spread_summary,
            tuple(
                column
                for column in group_spread_summary.columns
                if column not in {"signal", "label"} and not column.endswith("_reason")
            ),
        ),
    ):
        for column in columns:
            values = frame[column]
            if pd.api.types.is_bool_dtype(values) or (
                not pd.api.types.is_numeric_dtype(values) and not values.isna().all()
            ):
                raise ValueError("invalid_evaluation_numeric_values")
            strict_finite_numeric_series(
                frame[column],
                error="invalid_evaluation_numeric_values",
                allow_null=True,
            )
    for column in ("ic", "rankic"):
        if bool(metrics[column].dropna().abs().gt(1.0).any()):
            raise ValueError("invalid_evaluation_correlation_values")
    for column in ("mean_ic", "mean_rankic"):
        if bool(ic_summary[column].dropna().abs().gt(1.0).any()):
            raise ValueError("invalid_evaluation_correlation_values")
    for column in ("valid_ic_count", "valid_rankic_count"):
        strict_non_negative_count_series(
            ic_summary[column],
            error="invalid_evaluation_summary_values",
        )
    for column in ("std_ic", "std_rankic"):
        _validate_reason_pair(
            ic_summary[column],
            ic_summary[f"{column}_reason"],
            allowed=frozenset(
                {"no_valid_observations", "nonfinite_result", "not_applicable"}
            ),
        )
        prefix = column.removeprefix("std_")
        for count, reason in zip(
            ic_summary[f"valid_{prefix}_count"],
            ic_summary[f"{column}_reason"],
            strict=True,
        ):
            expected_empty = (
                "not_applicable"
                if prefix == "rankic"
                and spec.evaluation_axis == "single_asset_time_series"
                else "no_valid_observations"
            )
            if (count == 0 and reason != expected_empty) or (
                count > 0 and not pd.isna(reason) and reason != "nonfinite_result"
            ):
                raise ValueError("invalid_evaluation_statistic_reason")
        if bool(ic_summary[column].dropna().lt(0).any()):
            raise ValueError("invalid_evaluation_summary_values")
    for column in ("positive_ic_rate", "positive_rankic_rate"):
        values = ic_summary[column].dropna()
        if bool(values.lt(0).any() or values.gt(1).any()):
            raise ValueError("invalid_evaluation_summary_values")
    spread_hit_rate = group_spread_summary[
        "top_bottom_label_value_spread_hit_rate"
    ].dropna()
    if bool(spread_hit_rate.lt(0).any() or spread_hit_rate.gt(1).any()):
        raise ValueError("invalid_evaluation_summary_values")
    monotonicity = group_spread_summary["group_monotonicity_score"].dropna()
    if bool(monotonicity.lt(-1).any() or monotonicity.gt(1).any()):
        raise ValueError("invalid_evaluation_summary_values")
    _validate_reason_pair(
        metrics["ic"],
        metrics["ic_reason"],
        allowed=CORRELATION_RAW_REASON_CODES,
    )
    _validate_reason_pair(
        metrics["rankic"],
        metrics["rankic_reason"],
        allowed=CORRELATION_RAW_REASON_CODES,
    )
    _validate_reason_pair(
        ic_summary["mean_ic"],
        ic_summary["ic_reason"],
        allowed=CORRELATION_SUMMARY_REASON_CODES,
    )
    _validate_reason_pair(
        ic_summary["icir"],
        ic_summary["icir_reason"],
        allowed=CORRELATION_SUMMARY_REASON_CODES,
    )
    _validate_reason_pair(
        ic_summary["mean_rankic"],
        ic_summary["rankic_reason"],
        allowed=CORRELATION_SUMMARY_REASON_CODES,
    )
    _validate_reason_pair(
        ic_summary["rankicir"],
        ic_summary["rankicir_reason"],
        allowed=CORRELATION_SUMMARY_REASON_CODES,
    )
    _validate_reason_pair(
        group_spread_summary["group_monotonicity_score"],
        group_spread_summary["group_monotonicity_score_reason"],
        allowed=GROUP_SPREAD_REASON_CODES,
    )
    for value_column, reason_column in (
        ("ic", "ic_reason"),
        ("rankic", "rankic_reason"),
    ):
        if (
            value_column == "rankic"
            and spec.evaluation_axis == "single_asset_time_series"
        ):
            continue
        for n_obs, value, reason in zip(
            metrics["n_obs"],
            metrics[value_column],
            metrics[reason_column],
            strict=True,
        ):
            insufficient = type(reason) is str and reason == "insufficient_observations"
            if int(n_obs) < 2:
                if not (pd.isna(value) and insufficient):
                    raise ValueError("invalid_evaluation_statistic_reason")
            elif insufficient:
                raise ValueError("invalid_evaluation_statistic_reason")

    if validate_pairs and spec.evaluation_axis == "cross_sectional":
        expected = {
            (row.timestamp, row.signal, row.label): int(row.n_obs)
            for row in metrics.itertuples(index=False)
            if row.n_obs >= 2
        }
        observed = {}
        for row in grouped_label_values.itertuples(index=False):
            if row.n_obs <= 0 or pd.isna(row.mean_label_value):
                raise ValueError("evaluation_group_observations_invalid")
            key = (row.timestamp, row.signal, row.label)
            observed[key] = observed.get(key, 0) + int(row.n_obs)
        if observed != expected:
            raise ValueError("evaluation_metric_group_observations_mismatch")


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
    error = "evaluation_artifact_signal_label_mismatch"
    for column, allowed in (
        ("signal", {signal.signal_id for signal in spec.signals}),
        ("label", set(spec.label_columns)),
    ):
        if column not in frame.columns:
            raise ValueError(error)
        values = strict_string_series(frame[column], error=error, strip=False)
        strict_membership_series(values, allowed=allowed, error=error)


def _validate_reason_pair(
    values: pd.Series,
    reasons: pd.Series,
    *,
    allowed: frozenset[str],
) -> None:
    for value, reason in zip(values, reasons, strict=True):
        value_is_null = bool(pd.isna(value))
        reason_is_null = bool(pd.isna(reason))
        if value_is_null:
            if reason_is_null or type(reason) is not str or reason not in allowed:
                raise ValueError("invalid_evaluation_statistic_reason")
        elif not reason_is_null:
            raise ValueError("invalid_evaluation_statistic_reason")
