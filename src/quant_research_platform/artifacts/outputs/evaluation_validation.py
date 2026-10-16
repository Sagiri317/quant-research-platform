"""Evaluation output validation."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.evaluations import (
    CORRELATION_RAW_REASON_CODES,
    CORRELATION_SUMMARY_REASON_CODES,
    EVALUATION_FORECAST_AGE_BOUNDS,
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
from quant_research_platform.contracts.signals import SignalSpec
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


EVALUATION_SCIENCE_KEYS = frozenset(
    {
        "sample_scope",
        "input_facts",
        "comparison_inputs",
        "strata",
        "stratum_domains",
        "optional_member_rows",
        "common_rule",
    }
)


def validate_evaluation_science(metadata):
    """Validate actual input, window, units and optional daily-member facts."""
    from quant_research_platform.contracts.time import TimeWindow

    base = {"window", "cutoff", "sample_role"}
    if type(metadata) is not dict or set(metadata) not in (
        base,
        base | EVALUATION_SCIENCE_KEYS,
    ):
        raise ValueError("evaluation_science_metadata_invalid")
    if type(metadata["sample_role"]) is not str or not metadata["sample_role"].strip():
        raise ValueError("evaluation_sample_role_invalid")
    window = TimeWindow(**metadata["window"])
    if window.end is None or any(
        pd.isna(pd.Timestamp(value))
        for value in (window.start, window.end, metadata["cutoff"])
    ):
        raise ValueError("evaluation_science_time_invalid")
    if "sample_scope" not in metadata:
        return
    if (
        metadata["sample_scope"] not in {"own", "common"}
        or metadata["common_rule"] != "mask_only_no_retransform"
    ):
        raise ValueError("evaluation_sample_scope_invalid")
    peers = metadata["comparison_inputs"]
    if type(peers) is not list or (metadata["sample_scope"] == "own" and peers):
        raise ValueError("evaluation_comparison_inputs_invalid")
    for facts in [metadata["input_facts"], *peers]:
        if type(facts) is not dict or set(facts) != {
            "selector",
            "columns",
            "column_units",
            "observed_columns",
            "feature_transforms",
            "signal_units",
            "label_units",
            "label_availability_columns",
            "label_semantics",
            "market_id",
        }:
            raise ValueError("evaluation_input_facts_invalid")
        if (
            type(facts["selector"]) is not dict
            or type(facts["selector"].get("path")) is not str
            or not facts["selector"]["path"]
        ):
            raise ValueError("evaluation_input_facts_invalid")
        for key in (
            "columns",
            "column_units",
            "observed_columns",
            "signal_units",
            "label_units",
            "label_availability_columns",
            "label_semantics",
        ):
            if type(facts[key]) is not dict:
                raise ValueError("evaluation_input_facts_invalid")
        if type(facts["feature_transforms"]) is not list or any(
            unit is not None and (type(unit) is not str or not unit)
            for key in ("column_units", "signal_units", "label_units")
            for unit in facts[key].values()
        ):
            raise ValueError("evaluation_input_units_invalid")
        if (
            set(facts["column_units"]) != set(facts["columns"])
            or set(facts["signal_units"]) != set(facts["observed_columns"])
            or set(facts["label_units"]) != set(facts["label_availability_columns"])
        ):
            raise ValueError("evaluation_input_facts_invalid")
        for label, availability in facts["label_availability_columns"].items():
            if (
                facts["columns"].get(label) != "label"
                or availability not in facts["columns"]
                or facts["label_units"][label] != facts["column_units"].get(label)
            ):
                raise ValueError("evaluation_input_facts_invalid")
        for signal, companions in facts["observed_columns"].items():
            if (
                facts["columns"].get(signal) not in {"factor", "feature"}
                or type(companions) is not list
                or any(column not in facts["columns"] for column in companions)
                or facts["signal_units"][signal] != facts["column_units"].get(signal)
            ):
                raise ValueError("evaluation_input_facts_invalid")
        if facts["label_units"] != metadata["input_facts"]["label_units"]:
            raise ValueError("evaluation_label_units_mismatch")
    strata = metadata["strata"]
    if type(strata) is not dict or type(metadata["stratum_domains"]) is not dict:
        raise ValueError("evaluation_strata_invalid")
    if strata:
        allowed = {
            "primary_label",
            "kinds",
            "forecast_freshness_days",
            "population",
            "age_column",
            "age_bounds",
        }
        if (
            set(strata) - allowed
            or not {"primary_label", "kinds", "population", "age_bounds"} <= set(strata)
            or strata["population"]
            != "focal_eligible_primary_label_mature_before_common"
        ):
            raise ValueError("evaluation_strata_invalid")
        kinds = strata["kinds"]
        if (
            type(kinds) is not list
            or not kinds
            or len(set(kinds)) != len(kinds)
            or set(kinds) - {"industry", "size", "liquidity", "forecast_age"}
        ):
            raise ValueError("evaluation_strata_invalid")
        if "forecast_age" in kinds:
            if (
                strata.get("forecast_freshness_days") not in {90, 180}
                or strata.get("age_column") != "median_forecast_age_days_fy1"
                or strata["age_bounds"]
                != [
                    list(item)
                    for item in EVALUATION_FORECAST_AGE_BOUNDS
                    if item[2] <= strata["forecast_freshness_days"]
                ]
            ):
                raise ValueError("evaluation_age_bounds_invalid")
    if set(metadata["stratum_domains"]) != set(strata.get("kinds", [])) or any(
        type(values) is not list
        or any(value is not None and type(value) is not str for value in values)
        or len(values) != len(set(values))
        for values in metadata["stratum_domains"].values()
    ):
        raise ValueError("evaluation_stratum_domains_invalid")
    counts = metadata["optional_member_rows"]
    if (
        type(counts) is not dict
        or set(counts) not in (set(), {"daily_coverage", "daily_strata"})
        or any(type(value) is not int or value < 0 for value in counts.values())
        or ((strata or metadata["sample_scope"] == "common") and not counts)
    ):
        raise ValueError("evaluation_optional_members_invalid")


def validate_evaluation_output_metadata_shape(
    metadata: object,
) -> dict[str, object]:
    science = {"window", "cutoff", "sample_role"}
    if type(metadata) is not dict or set(metadata) not in (
        _EVALUATION_OUTPUT_METADATA_KEYS,
        _EVALUATION_OUTPUT_METADATA_KEYS | science,
        _EVALUATION_OUTPUT_METADATA_KEYS | science | EVALUATION_SCIENCE_KEYS,
    ):
        raise ValueError("output_manifest_metadata_invalid")
    if science <= set(metadata):
        validate_evaluation_science(
            {
                key: metadata[key]
                for key in science
                | (EVALUATION_SCIENCE_KEYS if "sample_scope" in metadata else set())
            }
        )
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
    if "input_facts" in metadata:
        facts = metadata["input_facts"]
        if (
            facts["label_availability_columns"] != label_availability_columns
            or set(facts["signal_units"])
            != {
                signal.source_column
                for signal in signals
                if signal.source_kind != "prediction"
            }
            or set(facts["label_units"]) != set(label_columns)
        ):
            raise ValueError("evaluation_input_facts_invalid")
        if (
            metadata["strata"]
            and metadata["strata"]["primary_label"] not in label_columns
        ):
            raise ValueError("evaluation_strata_invalid")
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
    timestamp_count: int | None = None,
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
    _validate_correlation_metrics(metrics, spec=spec)
    strict_non_negative_count_series(grouped_label_values["n_obs"])
    groups = strict_positive_group_series(grouped_label_values["group"])
    if bool(groups.gt(spec.n_groups).any()):
        raise ValueError("evaluation_group_out_of_range")
    for frame, columns in (
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
    for column in ("mean_ic", "mean_rankic"):
        if bool(ic_summary[column].dropna().abs().gt(1.0).any()):
            raise ValueError("invalid_evaluation_correlation_values")
    for column in ("valid_ic_count", "valid_rankic_count"):
        strict_non_negative_count_series(
            ic_summary[column],
            error="invalid_evaluation_summary_values",
        )
    for statistic in ("ic", "rankic"):
        counts = ic_summary[f"valid_{statistic}_count"]
        maximum = (
            min(1, timestamp_count)
            if timestamp_count is not None
            and spec.evaluation_axis == "single_asset_time_series"
            else timestamp_count
        )
        if maximum is not None and counts.gt(maximum).any():
            raise ValueError("invalid_evaluation_summary_count_values")
        empty = counts.eq(0)
        expected_empty = (
            "not_applicable"
            if statistic == "rankic"
            and spec.evaluation_axis == "single_asset_time_series"
            else "no_valid_observations"
        )
        if (
            ic_summary[f"mean_{statistic}"].notna().ne(~empty).any()
            or ic_summary[f"positive_{statistic}_rate"].notna().ne(~empty).any()
            or ic_summary.loc[empty, f"{statistic}_reason"].ne(expected_empty).any()
            or ic_summary.loc[empty, f"{statistic}ir_reason"].ne(expected_empty).any()
            or ic_summary.loc[empty, f"{statistic}ir"].notna().any()
        ):
            raise ValueError("invalid_evaluation_summary_count_values")
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


def _validate_correlation_metrics(metrics, *, spec):
    """Own correlation values, counts and undefined reasons for all daily tables."""
    _validate_signal_label_membership(metrics, spec=spec)
    strict_non_negative_count_series(metrics["n_obs"])
    for column in ("ic", "rankic"):
        values = metrics[column]
        if pd.api.types.is_bool_dtype(values) or (
            not pd.api.types.is_numeric_dtype(values) and not values.isna().all()
        ):
            raise ValueError("invalid_evaluation_numeric_values")
        strict_finite_numeric_series(
            values, error="invalid_evaluation_numeric_values", allow_null=True
        )
        if values.dropna().abs().gt(1).any():
            raise ValueError("invalid_evaluation_correlation_values")
        _validate_reason_pair(
            values, metrics[column + "_reason"], allowed=CORRELATION_RAW_REASON_CODES
        )
    if (
        spec.evaluation_axis == "cross_sectional"
        and metrics.rankic_reason.eq("not_applicable").any()
    ):
        raise ValueError("cross_sectional_evaluation_artifact_invalid")
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


_SIGNAL_METADATA_KEYS = frozenset(
    {"signal_id", "source_kind", "source_column", "direction", "transform"}
)


def validate_filter_metadata(
    *,
    mode: str,
    columns: object,
    definition: str,
    error_name: str,
) -> None:
    if type(columns) is not list or columns != list(required_filter_columns(mode)):
        raise ValueError(error_name)
    if definition != filter_definition(mode):
        raise ValueError(error_name)


def require_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str:
    value = metadata[key]
    if type(value) is not str or not value:
        raise ValueError(error_name)
    return value


def require_optional_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str | None:
    value = metadata[key]
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(error_name)
    return value


def require_metadata_string_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = metadata[key]
    if type(value) is not list:
        raise ValueError(error_name)
    if (not allow_empty and not value) or any(
        type(item) is not str or not item for item in value
    ):
        raise ValueError(error_name)
    if len(value) != len(set(value)):
        raise ValueError(error_name)
    return value


def require_non_negative_metadata_int(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> int:
    value = metadata[key]
    if type(value) is not int or value < 0:
        raise ValueError(error_name)
    return value


def require_signal_metadata_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> tuple[SignalSpec, ...]:
    value = metadata[key]
    if type(value) is not list or not value:
        raise ValueError(error_name)
    signals: list[SignalSpec] = []
    for signal in value:
        if type(signal) is not dict or set(signal) != _SIGNAL_METADATA_KEYS:
            raise ValueError(error_name)
        for field in _SIGNAL_METADATA_KEYS:
            require_metadata_string(signal, field, error_name)
        try:
            signals.append(
                SignalSpec(
                    signal_id=signal["signal_id"],
                    source_kind=signal["source_kind"],
                    source_column=signal["source_column"],
                    direction=signal["direction"],
                    transform=signal["transform"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(error_name) from exc
    return tuple(signals)


def validate_output_frame(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    error_name: str,
) -> None:
    observed = tuple(frame.columns)
    if any(type(column) is not str or not column for column in observed):
        raise ValueError(f"invalid_{error_name}_columns")
    if len(observed) != len(set(observed)):
        raise ValueError(f"duplicate_{error_name}_columns")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_{error_name}_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_{error_name}_columns: {', '.join(extra)}")
    if observed != required:
        raise ValueError(f"misordered_{error_name}_columns")
